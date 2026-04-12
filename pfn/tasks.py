from functools import reduce
from typing import Dict
import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph


def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)

def negative_sampling_tail(data, batch, num_negative, strict=True):
    """
    Tail-only negative sampling.
    输入 batch: [B, 3] (h, t, r)
    输出: [B, num_negative + 1, 3]
      - 第 0 列为原始正样本
      - 第 1..N 列仅替换 tail 为负样本，head / relation 保持不变
    """
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    if strict:
        # 仅使用 tail 侧的严格负采样掩码
        t_mask, _ = strict_negative_mask(data, batch)
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)

        # draw samples for negative tails
        rand = torch.rand(batch_size, num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]
    else:
        neg_t_index = torch.randint(
            data.num_nodes, (batch_size, num_negative), device=batch.device
        )

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:, 1:] = neg_t_index

    return torch.stack([h_index, t_index, r_index], dim=-1)

def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph

def _sample_relation_path_context(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_id: int,
    num_samples: int,
) -> torch.Tensor:
    """
    抽取 relation-aware 的 2-hop / 3-hop 路径模板，并映射成 (x, z, relation_id) 三元组。
    该函数单独封装，方便后续替换为更复杂的 path mining 策略。
    """
    device = edge_index.device
    if num_samples <= 0:
        return torch.empty(0, 3, dtype=torch.long, device=device)

    rel_eids = (edge_type == relation_id).nonzero(as_tuple=False).view(-1)
    if rel_eids.numel() == 0:
        return torch.empty(0, 3, dtype=torch.long, device=device)

    path_triples: list[tuple[int, int, int]] = []
    unique_pairs: set[tuple[int, int]] = set()
    max_tries = max(50, num_samples * 30)
    tries = 0

    while len(path_triples) < num_samples and tries < max_tries:
        tries += 1
        e1 = rel_eids[torch.randint(0, rel_eids.numel(), (1,), device=device).item()]
        x = int(edge_index[0, e1])
        y = int(edge_index[1, e1])

        out2 = (edge_index[0] == y).nonzero(as_tuple=False).view(-1)
        if out2.numel() == 0:
            continue
        e2 = out2[torch.randint(0, out2.numel(), (1,), device=device).item()]
        z2 = int(edge_index[1, e2])

        use_three_hop = bool(torch.rand(1, device=device).item() < 0.5)
        if use_three_hop:
            out3 = (edge_index[0] == z2).nonzero(as_tuple=False).view(-1)
            if out3.numel() == 0:
                z = z2
            else:
                e3 = out3[torch.randint(0, out3.numel(), (1,), device=device).item()]
                z = int(edge_index[1, e3])
        else:
            z = z2

        if x == z:
            continue
        pair = (x, z)
        if pair in unique_pairs:
            continue
        unique_pairs.add(pair)
        path_triples.append((x, z, relation_id))

    if not path_triples:
        return torch.empty(0, 3, dtype=torch.long, device=device)
    return torch.tensor(path_triples, dtype=torch.long, device=device)


def build_context_for_batch(
    data: Data,
    batch: torch.Tensor,
    entity_num_hops: int,
    relation_num_hops: int,
    num_pos: int,
    num_neg: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    针对 [B, N, 3] 输入，按每行固定 (h, r) 构建并复用上下文：
      1) node_list: 以 h 为中心的 k-hop 子图节点；
      2) relation_list: relation_graph 中以 r 为中心的 m-hop(默认1)关系节点；
      3) 正样本：从 node_list × relation_list 中抽取真实存在三元组；
      4) 负样本：
         - 一部分为 (h, r, ?) 的 tail 替换负样本；
         - 一部分为 node_list × relation_list 组合出的不存在三元组；
      5) 不足时全局补样；上下文样本去重，且不能和 batch 中已有三元组重复。

    返回长度为 B 的 list，每个 batch 行对应一份上下文。
    """
    assert batch.dim() == 3 and batch.size(-1) == 3, (
        f"`batch` 期望形状为 [B, N, 3]，但得到 {tuple(batch.shape)}"
    )

    edge_index = data.edge_index
    edge_type = data.edge_type
    num_nodes = getattr(data, "num_nodes", None)
    assert num_nodes is not None, "`data.num_nodes` 不能为空"

    batch_size, num_queries_per_row, _ = batch.shape
    triples_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    device = edge_index.device
    num_edges = edge_index.size(1)
    all_eids = torch.arange(num_edges, device=device)

    # 不再预构建全量 (h, t, r) set（大图会占用大量内存）。
    # 改为按需缓存 (h, r) -> tails，用原始 edge 查询三元组是否存在。
    hr_index = torch.stack([edge_index[0], edge_type])  # [2, E], 维度分别是 (h, r)
    hr_tail_cache: dict[tuple[int, int], set[int]] = {}

    def triple_exists(h: int, t: int, r: int) -> bool:
        key = (h, r)
        tails = hr_tail_cache.get(key, None)
        if tails is None:
            query_index = torch.tensor([[h], [r]], device=device, dtype=torch.long)
            edge_id, _ = edge_match(hr_index, query_index)
            if edge_id.numel() > 0:
                tails = set(edge_index[1, edge_id].detach().cpu().tolist())
            else:
                tails = set()
            hr_tail_cache[key] = tails
        return t in tails

    def get_khop_nodes(center: int, num_hops: int) -> torch.Tensor:
        nodes, _, _, _ = pyg_k_hop_subgraph(
            center,
            num_hops,
            edge_index,
            relabel_nodes=False,
            num_nodes=num_nodes,
        )
        return nodes

    relation_graph = data.relation_graph

    def get_relation_hop_nodes(center_rel: int, hops: int = 1) -> torch.Tensor:
        rel_nodes, _, _, _ = pyg_k_hop_subgraph(
            center_rel,
            hops,
            relation_graph.edge_index,
            relabel_nodes=False,
            num_nodes=relation_graph.num_nodes,
        )
        return rel_nodes

    for i in range(batch_size):
        # 仅当前行 [N, 3] 的三元组作为禁止集合（行内防泄漏，不跨行）
        forbidden_row_hrt = set()
        for j in range(num_queries_per_row):
            h_ij, t_ij, r_ij = batch[i, j]
            forbidden_row_hrt.add((int(h_ij), int(t_ij), int(r_ij)))

        # 假设当前行的 query 形如 (h, ?, r)，取该行第一个作为锚点
        h_anchor = int(batch[i, 0, 0])
        r_anchor = int(batch[i, 0, 2])

        # ===== 1) 节点集合 node_list（h 的 k-hop）=====
        nodes_h = get_khop_nodes(h_anchor, entity_num_hops)
        node_mask_h = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask_h[nodes_h] = True
        node_list = nodes_h if nodes_h.numel() > 0 else torch.arange(num_nodes, device=device)

        # ===== 2) 关系集合 relation_list（relation_graph 上 r 的 1-hop）=====
        rel_nodes = get_relation_hop_nodes(r_anchor, relation_num_hops)
        rel_list = rel_nodes if rel_nodes.numel() > 0 else torch.tensor([r_anchor], dtype=torch.long, device=device)
        rel_mask = torch.zeros(relation_graph.num_nodes, dtype=torch.bool, device=device)
        rel_mask[rel_list] = True

        # ===== 3) 正样本 =====
        # 3.1 先从 h 的 k-hop 子图中采样 num_pos/3（关系不限）
        local_edge_mask = node_mask_h[edge_index[0]] & node_mask_h[edge_index[1]]
        local_eids = local_edge_mask.nonzero(as_tuple=False).view(-1)
        local_pos_target = num_pos // 3

        pos_candidates: list[tuple[int, int, int]] = []
        if local_eids.numel() > 0 and local_pos_target > 0:
            perm = torch.randperm(local_eids.numel(), device=device)
            for idx in perm.tolist():
                eid = int(local_eids[idx])
                tri = (int(edge_index[0, eid]), int(edge_index[1, eid]), int(edge_type[eid]))
                if tri in forbidden_row_hrt or tri in pos_candidates:
                    continue
                pos_candidates.append(tri)
                if len(pos_candidates) >= local_pos_target:
                    break

        # 3.2 再按原有策略（node_list × relation_list）补剩余正样本
        local_rel_edge_mask = node_mask_h[edge_index[0]] & node_mask_h[edge_index[1]] & rel_mask[edge_type]
        rel_local_eids = local_rel_edge_mask.nonzero(as_tuple=False).view(-1)
        if rel_local_eids.numel() > 0 and len(pos_candidates) < num_pos:
            perm = torch.randperm(rel_local_eids.numel(), device=device)
            for idx in perm.tolist():
                eid = int(rel_local_eids[idx])
                tri = (int(edge_index[0, eid]), int(edge_index[1, eid]), int(edge_type[eid]))
                if tri in forbidden_row_hrt or tri in pos_candidates:
                    continue
                pos_candidates.append(tri)
                if len(pos_candidates) >= num_pos:
                    break

        # 正样本不足，按 relation_list 优先从全图补齐
        if len(pos_candidates) < num_pos and num_edges > 0:
            rel_global_eids = (rel_mask[edge_type]).nonzero(as_tuple=False).view(-1)
            if rel_global_eids.numel() > 0:
                perm = torch.randperm(rel_global_eids.numel(), device=device)
                for idx in perm.tolist():
                    eid = int(rel_global_eids[idx])
                    tri = (int(edge_index[0, eid]), int(edge_index[1, eid]), int(edge_type[eid]))
                    if tri in forbidden_row_hrt or tri in pos_candidates:
                        continue
                    pos_candidates.append(tri)
                    if len(pos_candidates) >= num_pos:
                        break

        # 仍不足再全图补齐
        if len(pos_candidates) < num_pos and num_edges > 0:
            perm = torch.randperm(all_eids.numel(), device=device)
            for idx in perm.tolist():
                eid = int(all_eids[idx])
                tri = (int(edge_index[0, eid]), int(edge_index[1, eid]), int(edge_type[eid]))
                if tri in forbidden_row_hrt or tri in pos_candidates:
                    continue
                pos_candidates.append(tri)
                if len(pos_candidates) >= num_pos:
                    break

        if len(pos_candidates) > 0:
            pos_triples = torch.tensor(pos_candidates[:num_pos], dtype=torch.long, device=device)
            pos_labels = torch.ones(pos_triples.size(0), dtype=torch.long, device=device)
        else:
            pos_triples = torch.empty(0, 3, dtype=torch.long, device=device)
            pos_labels = torch.empty(0, dtype=torch.long, device=device)

        # ===== 4) 负样本 =====
        # 4.1 (h, r, ?) tail 替换负样本，固定取 num_neg/2
        h_neg_target = num_neg // 2 if num_neg > 0 else 0
        relation_neg_target = num_neg - h_neg_target if num_neg > 0 else 0
        neg_candidates: list[tuple[int, int, int]] = []
        neg_set: set[tuple[int, int, int]] = set()

        tail_pool = node_list if node_list.numel() > 0 else torch.arange(num_nodes, device=device)
        tail_perm = torch.randperm(tail_pool.numel(), device=device)
        for k in tail_perm.tolist():
            t_neg = int(tail_pool[k])
            tri = (h_anchor, t_neg, r_anchor)
            if triple_exists(tri[0], tri[1], tri[2]) or tri in forbidden_row_hrt or tri in neg_set:
                continue
            neg_candidates.append(tri)
            neg_set.add(tri)
            if len(neg_candidates) >= h_neg_target:
                break

        # h-tail 不足时全局补齐
        tries = 0
        max_tries = max(50, (h_neg_target - len(neg_candidates)) * 30)
        while len(neg_candidates) < h_neg_target and tries < max_tries:
            tries += 1
            t_neg = int(torch.randint(0, num_nodes, (1,), device=device).item())
            tri = (h_anchor, t_neg, r_anchor)
            if triple_exists(tri[0], tri[1], tri[2]) or tri in forbidden_row_hrt or tri in neg_set:
                continue
            neg_candidates.append(tri)
            neg_set.add(tri)

        # 4.2 node_list × relation_list 组合负样本，补齐剩余部分
        tries = 0
        max_tries = max(100, relation_neg_target * 80)
        while len(neg_candidates) < num_neg and tries < max_tries:
            tries += 1
            h_neg = int(node_list[torch.randint(0, node_list.numel(), (1,), device=device).item()])
            t_neg = int(node_list[torch.randint(0, node_list.numel(), (1,), device=device).item()])
            r_neg = int(rel_list[torch.randint(0, rel_list.numel(), (1,), device=device).item()])
            tri = (h_neg, t_neg, r_neg)
            if triple_exists(tri[0], tri[1], tri[2]) or tri in forbidden_row_hrt or tri in neg_set:
                continue
            neg_candidates.append(tri)
            neg_set.add(tri)

        # 若随机采样未补满，则在 node_list × relation_list 上做一次确定性扫描补齐
        if len(neg_candidates) < num_neg:
            for h_neg_t in node_list.tolist():
                for t_neg_t in node_list.tolist():
                    for r_neg_t in rel_list.tolist():
                        tri = (int(h_neg_t), int(t_neg_t), int(r_neg_t))
                        if triple_exists(tri[0], tri[1], tri[2]) or tri in forbidden_row_hrt or tri in neg_set:
                            continue
                        neg_candidates.append(tri)
                        neg_set.add(tri)
                        if len(neg_candidates) >= num_neg:
                            break
                    if len(neg_candidates) >= num_neg:
                        break
                if len(neg_candidates) >= num_neg:
                    break

        if num_neg > 0 and len(neg_candidates) < num_neg:
            raise RuntimeError(
                f"负样本数量不足: 期望 {num_neg}, 实际 {len(neg_candidates)}. "
                "请增大 node_list/relation_list 覆盖范围。"
            )

        if len(neg_candidates) > 0:
            neg_triples = torch.tensor(neg_candidates[:num_neg], dtype=torch.long, device=device)
            neg_labels = torch.zeros(neg_triples.size(0), dtype=torch.long, device=device)
        else:
            neg_triples = torch.empty(0, 3, dtype=torch.long, device=device)
            neg_labels = torch.empty(0, dtype=torch.long, device=device)

        row_triples = torch.cat([pos_triples, neg_triples], dim=0)
        row_labels = torch.cat([pos_labels, neg_labels], dim=0)

        # 当前行只返回 1 份上下文（长度按 B 对齐）
        triples_list.append(row_triples)
        labels_list.append(row_labels)

    return triples_list, labels_list

def _sample_meta_context(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    h_anchor: int,
    r_anchor: int,
    num_meta: int,
    num_nodes: int,
    forbidden_hrt: set[tuple[int, int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 h_anchor 的 k-hop (k=1,2,3) 出射子图中采集 meta-context 三元组。
    - 任意 relation 均可
    - 采到的三元组标签为 1
    - 不足 num_meta 时，用 (h_anchor, random_t, r_anchor) 负样本 (标签 0) 补齐

    Returns:
        (meta_triples [num_meta, 3], meta_labels [num_meta])
    """
    if num_meta <= 0:
        return (
            torch.empty((0, 3), dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    # Build outgoing adjacency: src -> list of (dst, rel) using CPU tensors
    src_cpu = edge_index[0].cpu()
    dst_cpu = edge_index[1].cpu()
    rel_cpu = edge_type.cpu()

    collected: list[tuple[int, int, int]] = []  # (h, t, r)
    visited_edges: set[tuple[int, int, int]] = set()

    frontier = {h_anchor}

    for _hop in range(3):
        new_frontier: set[int] = set()
        for node in frontier:
            # Find outgoing edges from this node
            mask = (src_cpu == node)
            if not mask.any():
                continue
            dsts = dst_cpu[mask].tolist()
            rels = rel_cpu[mask].tolist()
            for t, r in zip(dsts, rels):
                tri = (node, t, r)
                if tri in visited_edges or tri in forbidden_hrt:
                    continue
                visited_edges.add(tri)
                collected.append(tri)
                new_frontier.add(t)
        frontier = new_frontier - {h_anchor}
        if len(collected) >= num_meta:
            break

    # Sample or keep all collected triples
    if len(collected) >= num_meta:
        indices = np.random.choice(len(collected), size=num_meta, replace=False)
        selected = [collected[idx] for idx in indices]
        meta_triples = torch.tensor(selected, dtype=torch.long, device=device)
        meta_labels = torch.ones(num_meta, dtype=torch.long, device=device)
    else:
        # All collected are positive
        num_real = len(collected)
        num_pad = num_meta - num_real

        if num_real > 0:
            real_triples = torch.tensor(collected, dtype=torch.long, device=device)
            real_labels = torch.ones(num_real, dtype=torch.long, device=device)
        else:
            real_triples = torch.empty((0, 3), dtype=torch.long, device=device)
            real_labels = torch.empty(0, dtype=torch.long, device=device)

        # Pad with same-relation negatives (h_anchor, random_t, r_anchor), label=0
        # Build true pairs for r_anchor to avoid false negatives
        rel_mask = (edge_type == r_anchor)
        if rel_mask.any():
            true_pairs = set(
                zip(edge_index[0, rel_mask].cpu().tolist(), edge_index[1, rel_mask].cpu().tolist())
            )
        else:
            true_pairs = set()

        pad_triples: list[tuple[int, int, int]] = []
        pad_set: set[int] = set()
        tries = 0
        max_tries = max(1000, num_pad * 50)
        while len(pad_triples) < num_pad and tries < max_tries:
            tries += 1
            t_rand = int(torch.randint(0, num_nodes, (1,)).item())
            if t_rand in pad_set:
                continue
            if (h_anchor, t_rand) in true_pairs:
                continue
            tri = (h_anchor, t_rand, r_anchor)
            if tri in forbidden_hrt or tri in visited_edges:
                continue
            pad_triples.append(tri)
            pad_set.add(t_rand)

        # Deterministic fallback if random wasn't enough
        if len(pad_triples) < num_pad:
            for t_cand in torch.randperm(num_nodes).tolist():
                if len(pad_triples) >= num_pad:
                    break
                if t_cand in pad_set or (h_anchor, t_cand) in true_pairs:
                    continue
                tri = (h_anchor, t_cand, r_anchor)
                if tri in forbidden_hrt or tri in visited_edges:
                    continue
                pad_triples.append(tri)
                pad_set.add(t_cand)

        if len(pad_triples) > 0:
            pad_t = torch.tensor(pad_triples[:num_pad], dtype=torch.long, device=device)
            pad_l = torch.zeros(pad_t.size(0), dtype=torch.long, device=device)
        else:
            pad_t = torch.empty((0, 3), dtype=torch.long, device=device)
            pad_l = torch.empty(0, dtype=torch.long, device=device)

        meta_triples = torch.cat([real_triples, pad_t], dim=0)
        meta_labels = torch.cat([real_labels, pad_l], dim=0)

    return meta_triples, meta_labels


def build_context_relation_aware(
    data: Data,
    batch: torch.Tensor,
    num_pos: int,
    num_neg: int,
    num_meta_context: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    针对 [B, N, 3] 输入，按每行固定 relation=r 构建并复用上下文：
      1) meta-context（可选）：从 head entity 的 k-hop 出射子图采样，标签为 1；
      2) 正样本：从全图中采样 relation=r 的真实三元组，最多 num_pos 个；
      3) 若正样本不足 num_pos，缺口自动并入负样本数量；
      4) 负样本：采样 (x, y, r) 且在图中不存在；
      5) 去重，且不能包含当前行 [N, 3] 中已有三元组。

    输出每行上下文形状: [num_meta_context + num_pos + num_neg, 3]
    返回长度为 B 的 list，每个 batch 行对应一份上下文。
    """
    assert batch.dim() == 3 and batch.size(-1) == 3, (
        f"`batch` 期望形状为 [B, N, 3]，但得到 {tuple(batch.shape)}"
    )

    edge_index = data.edge_index
    edge_type = data.edge_type
    num_nodes = getattr(data, "num_nodes", None)
    assert num_nodes is not None, "`data.num_nodes` 不能为空"

    device = edge_index.device
    batch_size, num_queries_per_row, _ = batch.shape

    triples_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for i in range(batch_size):
        # 当前行禁止集合：上下文中不能出现该行原始三元组
        forbidden_row_hrt: set[tuple[int, int, int]] = set()
        row_relations = []
        for j in range(num_queries_per_row):
            h_ij, t_ij, r_ij = batch[i, j]
            tri = (int(h_ij), int(t_ij), int(r_ij))
            forbidden_row_hrt.add(tri)
            row_relations.append(int(r_ij))

        # 约定当前行共享同一个 relation，取第一个作为锚点
        r_anchor = row_relations[0]

        rel_eids = (edge_type == r_anchor).nonzero(as_tuple=False).view(-1)
        if rel_eids.numel() == 0:
            true_pairs_r: set[tuple[int, int]] = set()
        else:
            hs = edge_index[0, rel_eids].detach().cpu().tolist()
            ts = edge_index[1, rel_eids].detach().cpu().tolist()
            true_pairs_r = set(zip(hs, ts))

        # ===== 1) 正样本：全图 relation=r 的真实三元组 =====
        pos_candidates: list[tuple[int, int, int]] = []
        pos_set: set[tuple[int, int, int]] = set()
        if rel_eids.numel() > 0 and num_pos > 0:
            perm = torch.randperm(rel_eids.numel(), device=device)
            for idx in perm.tolist():
                eid = int(rel_eids[idx])
                tri = (
                    int(edge_index[0, eid]),
                    int(edge_index[1, eid]),
                    r_anchor,
                )
                if tri in forbidden_row_hrt or tri in pos_set:
                    continue
                pos_candidates.append(tri)
                pos_set.add(tri)
                if len(pos_candidates) >= num_pos:
                    break

        actual_pos = len(pos_candidates)
        # 正样本不足的缺口并入负样本
        need_neg = max(0, num_neg + (num_pos - actual_pos))

        # ===== 2) 负样本：优先从 head 的 2-hop 子图采样 + 全局随机采样 =====
        neg_candidates: list[tuple[int, int, int]] = []
        neg_set: set[tuple[int, int, int]] = set()

        # 获取该行的 anchor head（使用 [B, 0, 3] 即每行第一个三元组的 head）
        h_anchor = int(batch[i, 0, 0])

        def invalid_neg(h: int, t: int, r: int) -> bool:
            tri = (h, t, r)
            if tri in forbidden_row_hrt or tri in pos_set or tri in neg_set:
                return True
            if (h, t) in true_pairs_r:
                return True
            return False

        # 计算 head 的 2-hop 邻居子图
        def get_2hop_neighbors(head: int, edge_idx: torch.Tensor) -> set[int]:
            """获取 head 的 2-hop 邻居（包括 head 自身）"""
            # 1-hop 邻居
            mask1 = edge_idx[0] == head
            neighbors_1hop = set(edge_idx[1, mask1].cpu().tolist())
            neighbors_1hop.add(head)  # 包含自身
            
            # 2-hop 邻居
            neighbors_2hop = set(neighbors_1hop)
            for n1 in neighbors_1hop:
                mask2 = edge_idx[0] == n1
                n2_set = set(edge_idx[1, mask2].cpu().tolist())
                neighbors_2hop.update(n2_set)
            return neighbors_2hop

        # 计算 2-hop 邻居
        neighbors_2hop = get_2hop_neighbors(h_anchor, edge_index)
        neighbors_list = list(neighbors_2hop)

        # 计算需要采样的数量
        target_2hop = int(need_neg * 0.8)  # 80% 来自 2-hop
        target_random = need_neg - target_2hop  # 20% 随机

        # 2.1 优先从 2-hop 子图采样（80%）
        # 先对 2-hop 邻居随机打乱
        np.random.shuffle(neighbors_list)
        
        for t_candidate in neighbors_list:
            if len(neg_candidates) >= target_2hop:
                break
            if invalid_neg(h_anchor, t_candidate, r_anchor):
                continue
            tri = (h_anchor, t_candidate, r_anchor)
            neg_candidates.append(tri)
            neg_set.add(tri)

        # 记录实际采到的 2-hop 数量
        actual_2hop = len(neg_candidates)

        # 2.2 从全局随机采样（目标 20%，如果 2-hop 不够则补足）
        remaining = need_neg - actual_2hop  # 还需要采的数量
        
        # 如果 2-hop 采样不足 80%，剩余的全部用随机采样补足
        if remaining > 0:
            tries = 0
            max_tries = max(1000, remaining * 50)
            while len(neg_candidates) < need_neg and tries < max_tries:
                tries += 1
                t_rand = int(torch.randint(0, num_nodes, (1,), device=device).item())
                if invalid_neg(h_anchor, t_rand, r_anchor):
                    continue
                tri = (h_anchor, t_rand, r_anchor)
                neg_candidates.append(tri)
                neg_set.add(tri)

        # 2.3 如果还不够，确定性扫描补齐
        if len(neg_candidates) < need_neg:
            all_nodes = torch.randperm(num_nodes, device=device).tolist()
            for t_candidate in all_nodes:
                if len(neg_candidates) >= need_neg:
                    break
                if invalid_neg(h_anchor, t_candidate, r_anchor):
                    continue
                tri = (h_anchor, t_candidate, r_anchor)
                neg_candidates.append(tri)
                neg_set.add(tri)

        if len(neg_candidates) < need_neg:
            raise RuntimeError(
                f"relation={r_anchor} 的负样本数量不足: 期望 {need_neg}, 实际 {len(neg_candidates)}. "
                "该关系可能接近全连接，无法继续构造不存在的 (x,y,r)。"
            )

        # ===== 3) meta-context：从 head 的 k-hop 出射子图采样 =====
        if num_meta_context > 0:
            meta_triples, meta_labels = _sample_meta_context(
                edge_index=edge_index,
                edge_type=edge_type,
                h_anchor=h_anchor,
                r_anchor=r_anchor,
                num_meta=num_meta_context,
                num_nodes=num_nodes,
                forbidden_hrt=forbidden_row_hrt,
                device=device,
            )
        else:
            meta_triples = torch.empty((0, 3), dtype=torch.long, device=device)
            meta_labels = torch.empty(0, dtype=torch.long, device=device)

        # ===== 4) 打包输出 =====
        # 如果没有正样本，只使用负样本
        if actual_pos > 0:
            pos_raw = torch.tensor(pos_candidates, dtype=torch.long, device=device)
            pos_triples = pos_raw.view(pos_raw.numel() // 3, 3)
            pos_labels = torch.ones(pos_triples.size(0), dtype=torch.long, device=device)
        else:
            # 无正样本，创建空张量
            pos_triples = torch.empty((0, 3), dtype=torch.long, device=device)
            pos_labels = torch.empty(0, dtype=torch.long, device=device)

        neg_raw = torch.tensor(neg_candidates[:need_neg], dtype=torch.long, device=device)
        neg_triples = neg_raw.view(neg_raw.numel() // 3, 3)
        neg_labels = torch.zeros(neg_triples.size(0), dtype=torch.long, device=device)

        row_triples = torch.cat([meta_triples, pos_triples, neg_triples], dim=0)
        row_labels = torch.cat([meta_labels, pos_labels, neg_labels], dim=0)

        triples_list.append(row_triples)
        labels_list.append(row_labels)

    return triples_list, labels_list