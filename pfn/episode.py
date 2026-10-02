"""DiffTSP-style support/query episode splits on top of an existing KG.

Sampling is done on the *original* (non-inverse) triples only. Inverse edges
are a training-time post-process: every kept ``(h, r, t)`` is mirrored as
``(t, r^{-1}, h)`` when the support graph is materialized.

An episode is fully determined by a boolean ``keep`` mask over the base
graph's edges. Training samples a fresh mask online each time a new episode
starts (``sample_support_query_episode``). After the relation-balanced draw,
the mask is restricted to the largest weakly-connected component of the
sampled support; smaller components are discarded (neither support nor query).
The query pool is the set of original train edges that were not kept and
whose both endpoints lie in that largest component.
"""

import copy
from typing import Any

import torch


def _parse_rho_range(cfg) -> tuple[float, float]:
    rho = cfg.get("rho", [0.7, 0.9])
    if isinstance(rho, (list, tuple)):
        if len(rho) != 2:
            raise ValueError(f"episode.rho must be a scalar or [lo, hi], got {rho}")
        lo, hi = float(rho[0]), float(rho[1])
    else:
        lo = hi = float(rho)
    if not (0.0 < lo <= hi < 1.0):
        raise ValueError(f"episode.rho range must satisfy 0 < lo <= hi < 1, got {[lo, hi]}")
    return lo, hi


def _num_direct_relations(graph: Any) -> int:
    num_relations = int(graph.num_relations)
    if num_relations < 2 or num_relations % 2:
        raise ValueError(
            f"episode split expects even num_relations (original + inverse), got {num_relations}"
        )
    return num_relations // 2


def _sample_relation_rhos(
    edge_type: torch.Tensor,
    num_relations: int,
    rho_lo: float,
    rho_hi: float,
    jitter: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """One support ratio per original relation: global rho plus per-relation jitter."""
    device = edge_type.device
    base = rho_lo + (rho_hi - rho_lo) * torch.rand((), device=device, generator=generator)
    counts = torch.bincount(edge_type, minlength=num_relations).float()
    # rare relations get damped jitter so they are not wiped out by noise
    scale = (counts / (counts + 10.0)).clamp(min=0.1)
    noise = torch.randn(num_relations, device=device, generator=generator) * jitter * scale
    return (base + noise).clamp(min=0.05, max=0.98)


def sample_keep_mask(
    graph: Any,
    episode_cfg: dict,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a relation-balanced keep-mask on original (non-inverse) edges.

    Inverse-edge slots in the returned mask stay False; inverses are added
    later by ``materialize_support``.
    """
    if generator is None:
        generator = torch.Generator(device=graph.edge_index.device)
        generator.manual_seed(torch.seed() % 9223372036854775808)

    edge_index = graph.edge_index
    edge_type = graph.edge_type
    num_edges = edge_index.size(1)
    num_direct = _num_direct_relations(graph)
    device = edge_index.device

    rho_lo, rho_hi = _parse_rho_range(episode_cfg)
    jitter = float(episode_cfg.get("rho_jitter", 0.08))

    is_orig = edge_type < num_direct
    orig_type = edge_type[is_orig]
    rel_rho = _sample_relation_rhos(orig_type, num_direct, rho_lo, rho_hi, jitter, generator)

    keep = torch.zeros(num_edges, dtype=torch.bool, device=device)
    for r in range(num_direct):
        r_eids = (edge_type == r).nonzero(as_tuple=False).view(-1)
        n_r = r_eids.numel()
        if n_r == 0:
            continue
        n_support = int(round(float(rel_rho[r]) * n_r))
        n_support = max(0, min(n_r, n_support))
        if n_support == 0:
            continue
        perm = r_eids[torch.randperm(n_r, device=device, generator=generator)]
        keep[perm[:n_support]] = True
    return _restrict_keep_to_largest_cc(graph, keep)


def materialize_support(graph: Any, keep: torch.Tensor) -> dict:
    """Build the support graph and query pool determined by a keep-mask.

    Support edges = kept original triples in the largest CC, plus a synthesized
    inverse for each. Query pool = original train edges that were not kept and
    whose both endpoints lie in that largest component.
    """
    edge_index = graph.edge_index
    edge_type = graph.edge_type
    device = edge_index.device
    num_direct = _num_direct_relations(graph)
    keep = _restrict_keep_to_largest_cc(graph, keep.to(device))
    is_orig = edge_type < num_direct
    keep_orig = keep & is_orig

    support_h = edge_index[0, keep_orig]
    support_t = edge_index[1, keep_orig]
    support_r = edge_type[keep_orig]
    support_fwd = torch.stack([support_h, support_t])
    support_inv = torch.stack([support_t, support_h])
    support_edge_index = torch.cat([support_fwd, support_inv], dim=1)
    support_edge_type = torch.cat([support_r, support_r + num_direct])

    node_component = _connected_components(
        support_edge_index, int(graph.num_nodes), device
    )
    in_support = torch.zeros(int(graph.num_nodes), dtype=torch.bool, device=device)
    if support_edge_index.size(1) > 0:
        in_support[support_edge_index[0]] = True
        in_support[support_edge_index[1]] = True

    remain = is_orig & ~keep
    qh = edge_index[0, remain]
    qt = edge_index[1, remain]
    qr = edge_type[remain]
    if qh.numel() == 0:
        valid = qh.new_zeros((), dtype=torch.bool)
        query_target_index = edge_index.new_empty((2, 0))
        query_target_type = edge_type.new_empty((0,))
    else:
        valid = (
            in_support[qh]
            & in_support[qt]
            & (node_component[qh] == node_component[qt])
        )
        query_target_index = torch.stack([qh[valid], qt[valid]])
        query_target_type = qr[valid]

    support = copy.copy(graph)
    support.edge_index = support_edge_index
    support.edge_type = support_edge_type
    support.target_edge_index = query_target_index
    support.target_edge_type = query_target_type
    support.node_component = node_component

    from .tasks import build_relation_graph
    support = build_relation_graph(support)
    return {
        "support": support,
        "query_pool": int(query_target_index.size(1)),
        "keep": keep,
    }


def _restrict_keep_to_largest_cc(graph: Any, keep: torch.Tensor) -> torch.Tensor:
    """Zero out keep bits that are not in the largest support connected component."""
    edge_index = graph.edge_index
    edge_type = graph.edge_type
    device = edge_index.device
    keep = keep.to(device)
    num_direct = _num_direct_relations(graph)
    is_orig = edge_type < num_direct
    keep_orig = keep & is_orig
    if not keep_orig.any():
        return torch.zeros_like(keep)

    support_index = edge_index[:, keep_orig]
    comp = _connected_components(support_index, int(graph.num_nodes), device)
    counts = torch.bincount(comp)
    largest = int(counts.argmax())
    h, t = edge_index[0], edge_index[1]
    in_largest = (comp[h] == largest) & (comp[t] == largest)
    return keep_orig & in_largest


def sample_support_query_episode(
    graph: Any,
    episode_cfg: dict,
    generator: torch.Generator | None = None,
    max_retries: int = 8,
) -> dict:
    """Sample a fresh keep-mask and materialize the episode in one call.

    Empty query pools are retried up to ``max_retries`` times (a new mask
    each attempt). The last attempt is returned even if still empty.
    """
    last = None
    for _ in range(max(1, int(max_retries))):
        keep = sample_keep_mask(graph, episode_cfg, generator=generator)
        last = materialize_support(graph, keep)
        if last["query_pool"] > 0:
            return last
    return last


def _connected_components(
    edge_index: torch.Tensor, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """Weakly-connected component id per node via BFS over undirected edges."""
    if edge_index.numel() == 0:
        return torch.arange(num_nodes, dtype=torch.long, device=device)
    src = torch.cat([edge_index[0], edge_index[1]]).cpu().tolist()
    dst = torch.cat([edge_index[1], edge_index[0]]).cpu().tolist()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)

    comp = torch.full((num_nodes,), -1, dtype=torch.long)
    comp_id = 0
    for start in range(num_nodes):
        if comp[start] >= 0:
            continue
        stack = [start]
        comp[start] = comp_id
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = comp_id
                    stack.append(v)
        comp_id += 1
    return comp.to(device)
