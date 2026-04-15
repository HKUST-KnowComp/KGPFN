"""
Test-only script: replaces KGPFN scoring with TabPFN binary classification.
- Context/query sampling: identical to pretrain_pfn.py test()
- Scoring: raw structure encoder embeddings (pre-adapter) -> TabPFN
- Metrics: MR, MRR, Hits@1/3/10, Accuracy
- Supports JointDataset (multiple graphs)
"""
import os
import sys
import copy
import logging
import yaml
import torch
import numpy as np
from torch.utils import data as torch_data
from torch_geometric.data import Data
from torch import distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pfn import tasks, util
from model.ultra.encoder import StructureEncoderRelationAware
from model.kgpfnsem import KGPFN
from utils.loading import build_custom_model
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

separator = ">" * 30
logger = logging.getLogger(__file__)


def _get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_model(cfg, ckpt_path=None):
    project_root = _get_project_root()
    limix_config_path = cfg.train.get(
        "limix_config_path",
        os.path.join(project_root, "config", "limix", "limix_config.yaml"),
    )
    with open(limix_config_path, "r") as f:
        config_ft = yaml.safe_load(f)

    entity_model_cfg = copy.deepcopy(cfg.model.entity_model)
    entity_chunk_size = entity_model_cfg.pop("entity_chunk_size", None)
    structure_encoder = StructureEncoderRelationAware(
        rel_model_cfg=copy.deepcopy(cfg.model.relation_model),
        entity_model_cfg=entity_model_cfg,
        entity_chunk_size=entity_chunk_size,
    )
    model = KGPFN(
        structure_encoder=structure_encoder,
        feature_transformer=build_custom_model(config_ft),
        entity_dim=int(cfg.model.entity_model.get("input_dim", 64)),
        relation_dim=int(cfg.model.relation_model.get("input_dim", 64)),
        with_relation=bool(cfg.model.get("with_relation", True)),
    )
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd, strict=False)
        logger.warning("Loaded checkpoint: %s", ckpt_path)
    model.eval()
    return model


@torch.no_grad()
def encode_triples(model: KGPFN, data, triples: torch.Tensor) -> np.ndarray:
    """triples: [N, 3] -> numpy [N, 3*D], raw pre-adapter embeddings."""
    h_emb, r_emb, t_emb = model._triples_to_embeddings(data, triples.unsqueeze(0))
    return torch.cat([h_emb, r_emb, t_emb], dim=-1).squeeze(0).cpu().numpy()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data, device, split="test"):
    world_size = util.get_world_size()
    rank = util.get_rank()
    eval_chunk_size = int(cfg.train.get("eval_chunk_size", 64))

    all_mrrs = []
    for graph_idx, (test_graph, filters) in enumerate(zip(test_data, filtered_data)):
        graph_name = getattr(test_graph, "dataset", f"graph_{graph_idx}")
        logger.warning(separator)
        logger.warning("[%s/%s] Evaluating: %s (%d triples, %d nodes)",
                       split, graph_idx, graph_name,
                       test_graph.target_edge_index.shape[1], test_graph.num_nodes)

        test_triplets = torch.cat([
            test_graph.target_edge_index,
            test_graph.target_edge_type.unsqueeze(0),
        ]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        rankings, num_negatives = [], []
        pos_correct = 0
        pos_total = 0

        # Build one TabPFN per graph: fit on context sampled from the full graph
        # We use a single shared context (sampled once) for the whole graph eval,
        # matching the pretrain_pfn.py pattern of per-batch context.
        # Here we do per-batch context + per-batch TabPFN fit/predict.

        for batch in loader:
            batch = batch.to(device)
            t_batch, _ = tasks.all_negative(test_graph, batch)  # [B, num_nodes, 3]
            B, num_nodes, _ = t_batch.shape
            t_mask, _ = tasks.strict_negative_mask(filters, batch)
            _, pos_t_index, _ = batch.t()

            # Sample context (same as pretrain_pfn.py test)
            context_triples, context_labels = tasks.build_context_relation_aware(
                test_graph,
                batch.unsqueeze(1),  # [B, 1, 3]
                num_pos=cfg.task.num_pos,
                num_neg=cfg.task.num_neg,
            )

            # Score all tail candidates via TabPFN, chunk by chunk
            t_pred = torch.zeros(B, num_nodes, device=device)
            for b in range(B):
                ctx = context_triples[b].to(device)   # [M, 3]
                ctx_y = context_labels[b].cpu().numpy()  # [M]
                X_train = encode_triples(model, test_graph, ctx)  # [M, 3*D]

                # Encode all candidate tails in chunks
                X_test_chunks = []
                for start in range(0, num_nodes, max(1, eval_chunk_size)):
                    end = min(start + eval_chunk_size, num_nodes)
                    chunk = t_batch[b, start:end, :]  # [chunk, 3]
                    X_test_chunks.append(encode_triples(model, test_graph, chunk))
                X_test = np.concatenate(X_test_chunks, axis=0)  # [num_nodes, 3*D]

                clf = TabPFNClassifier.create_default_for_version(
                    ModelVersion.V2, device=str(device)
                )
                clf.fit(X_train, ctx_y)
                # predict_proba returns [N, 2]; use prob of class 1 as score
                proba = clf.predict_proba(X_test)
                scores = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                t_pred[b] = torch.tensor(scores, device=device)

                # pos-only accuracy: is the positive triple predicted as class 1?
                pos_feat = encode_triples(model, test_graph, t_batch[b, pos_t_index[b].item():pos_t_index[b].item()+1])
                pos_correct += int(clf.predict(pos_feat)[0] == 1)
                pos_total += 1

            # Ranking metrics
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            rankings.append(t_ranking)
            num_negatives.append(t_mask.sum(dim=-1))

        ranking = torch.cat(rankings)
        # DDP gather
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)

        if rank == 0:
            mr  = all_ranking.float().mean().item()
            mrr = (1.0 / all_ranking.float()).mean().item()
            h1  = (all_ranking <= 1).float().mean().item()
            h3  = (all_ranking <= 3).float().mean().item()
            h10 = (all_ranking <= 10).float().mean().item()
            acc = pos_correct / pos_total if pos_total > 0 else 0.0
            logger.warning("[%s] MR=%.4f MRR=%.4f H@1=%.4f H@3=%.4f H@10=%.4f PosAcc=%.4f",
                           graph_name, mr, mrr, h1, h3, h10, acc)

        all_mrrs.append((1.0 / all_ranking.float()).mean())

    avg_mrr = sum(all_mrrs) / len(all_mrrs)
    if rank == 0:
        logger.warning("Average MRR across graphs: %.4f", avg_mrr.item())
    return avg_mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    device = util.get_device(cfg)

    logger = util.get_root_logger(file=False)

    dataset = util.build_dataset(cfg)
    test_data  = dataset._data[2]
    train_data = dataset._data[0]

    test_data  = [td.to(device) for td in test_data]
    train_data = [td.to(device) for td in train_data]

    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        logger.warning("Fast evaluation on %d samples per graph", num_val_edges)
        for g in test_data:
            mask = torch.randperm(g.target_edge_index.shape[1])[:num_val_edges]
            g.target_edge_index = g.target_edge_index[:, mask]
            g.target_edge_type  = g.target_edge_type[mask]

    def _make_filtered(graphs):
        return [
            Data(
                edge_index=torch.cat([g.edge_index, g.target_edge_index], dim=1),
                edge_type=torch.cat([g.edge_type, g.target_edge_type]),
                num_nodes=g.num_nodes,
            ).to(device)
            for g in graphs
        ]

    test_filtered = _make_filtered(test_data)

    ckpt_path = cfg.train.get("kgpfn_checkpoint", None)
    model = build_model(cfg, ckpt_path=ckpt_path).to(device)

    test(cfg, model, test_data, test_filtered, device, split="test")
