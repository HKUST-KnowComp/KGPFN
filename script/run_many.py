import os
import sys
import csv
import math
import time
import pprint
import argparse
import logging
from collections import defaultdict

import torch
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pfn import util, tasks
from model.ultra.models import Ultra
from script.pretrain_pfn import create_model as create_pfn_model

separator = ">" * 30
line = "-" * 30


# Define which datasets are transductive vs inductive
TRANSDUCTIVE_DATASETS = {
    'FB15k237', 'FB15k237_10', 'FB15k237_20', 'FB15k237_50',
    'WN18RR', 'CoDExSmall', 'CoDExMedium', 'CoDExLarge',
    'NELL995', 'ConceptNet100k', 'DBpedia100k', 'YAGO310', 'AristoV4',
    'Hetionet', 'WDsinger', 'NELL23k'
}


def is_transductive(dataset_name):
    """Check if a dataset is transductive based on its name."""
    # Extract base name without version suffix
    base_name = dataset_name.split(':')[0] if ':' in dataset_name else dataset_name
    return base_name in TRANSDUCTIVE_DATASETS


@torch.no_grad()
def test_single_graph(cfg, model, test_graph, filtered_data, logger, dataset_name="unknown"):
    """Test on a single graph and return metrics."""
    world_size = util.get_world_size()
    rank = util.get_rank()
    device = util.get_device(cfg)

    eval_chunk_size = int(cfg.train.get("eval_chunk_size", 64))

    if rank == 0:
        logger.warning(separator)
        logger.warning(f"Evaluating: {dataset_name} ({test_graph.target_edge_index.shape[1]} triples, {test_graph.num_nodes} nodes)")

    test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []

    for batch in test_loader:
        # Strict negative sampling: all tail candidates
        t_batch, _ = tasks.all_negative(test_graph, batch)  # (B, num_nodes, 3)
        B, num_nodes, _ = t_batch.shape

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        # Check if model is PFN (has context-based prediction)
        is_pfn = hasattr(model, 'get_context_embeddings_cache')

        if is_pfn:
            # PFN model: use context-based prediction
            row_anchor_batch = batch.unsqueeze(1)
            shared_context_x, shared_context_y = tasks.build_context_relation_aware(
                test_graph,
                row_anchor_batch,
                num_pos=cfg.task.num_pos,
                num_neg=cfg.task.num_neg,
                num_meta_context=int(cfg.task.get("num_meta_context", 0)),
            )
            shared_context_x = [t.to(device) for t in shared_context_x]
            shared_context_y = [y.to(device) for y in shared_context_y]

            context_cache, shared_context_y = model.get_context_embeddings_cache(
                test_graph, shared_context_x, shared_context_y,
                num_meta_context=int(cfg.task.get("num_meta_context", 0)),
            )

            # Chunk evaluation
            t_score_chunks = []
            for start in range(0, num_nodes, max(1, eval_chunk_size)):
                end = min(start + max(1, eval_chunk_size), num_nodes)
                t_batch_chunk = t_batch[:, start:end, :]
                t_scores_chunk = model.get_scores(
                    test_graph,
                    query_x=t_batch_chunk.to(device),
                    context_cache=context_cache,
                    context_y=shared_context_y,
                    task_type="reg",
                )
                t_score_chunks.append(t_scores_chunk.view(B, -1))
            t_pred = torch.cat(t_score_chunks, dim=1)
        else:
            # ULTRA model: direct prediction
            t_pred = model(test_graph, t_batch.to(device))  # (B, num_nodes)

        # Compute ranking
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        rankings.append(t_ranking)

    ranking = torch.cat(rankings)

    # Gather rankings from all processes
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)

    # Compute metrics
    metrics = {}
    if rank == 0:
        mr = all_ranking.float().mean()
        mrr = (1 / all_ranking.float()).mean()
        hits1 = (all_ranking <= 1).float().mean()
        hits10 = (all_ranking <= 10).float().mean()
        hits50 = (all_ranking <= 50).float().mean()

        metrics = {
            'mr': float(mr.item()),
            'mrr': float(mrr.item()),
            'hits@1': float(hits1.item()),
            'hits@10': float(hits10.item()),
            'hits@50': float(hits50.item()),
        }

        logger.warning(f"[{dataset_name}] MR: {metrics['mr']:.2f}, MRR: {metrics['mrr']:.4f}, "
                      f"Hits@1: {metrics['hits@1']:.4f}, Hits@10: {metrics['hits@10']:.4f}, "
                      f"Hits@50: {metrics['hits@50']:.4f}")

    return metrics


def create_working_directory(cfg, model_type, model_path):
    """Create working directory for logging."""
    output_dir = os.path.expanduser(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Extract model name from path
    model_name = os.path.splitext(os.path.basename(model_path))[0] if model_path else "unknown"
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    working_dir = os.path.join(output_dir, f"test_{model_type}_{model_name}_{timestamp}")
    os.makedirs(working_dir, exist_ok=True)

    return working_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("--model_type", help="Model type: ultra or pfn", default="ultra", choices=["ultra", "pfn"])
    parser.add_argument("--ckpt", help="Path to checkpoint file", required=True)
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    args, unparsed = parser.parse_known_args()

    # Parse dynamic variables from config
    dyn_vars = util.detect_variables(args.config)
    dyn_parser = argparse.ArgumentParser()
    for var in dyn_vars:
        dyn_parser.add_argument(f"--{var}")
    dyn_vals = dyn_parser.parse_known_args(unparsed)[0]
    vars_dict = {k: util.literal_eval(v) for k, v in dyn_vals._get_kwargs()}

    # Override config for testing
    vars_dict['epochs'] = 0
    vars_dict['bpe'] = 'null'

    cfg = util.load_config(args.config, context=vars_dict)

    # Create working directory and setup logging
    working_dir = create_working_directory(cfg, args.model_type, args.ckpt)
    logger = util.get_root_logger(file=False)
    if util.get_rank() == 0:
        log_file = os.path.join(working_dir, "test_results.log")
        handler = logging.FileHandler(log_file, mode="w")
        fmt = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        logger.warning("=" * 80)
        logger.warning(f"Testing {args.model_type.upper()} model on all datasets")
        logger.warning(f"Checkpoint: {args.ckpt}")
        logger.warning(f"Config: {args.config}")
        logger.warning(f"Working directory: {working_dir}")
        logger.warning("=" * 80)

    torch.manual_seed(args.seed + util.get_rank())
    device = util.get_device(cfg)

    # Build dataset with all graphs
    if util.get_rank() == 0:
        logger.warning("Loading all datasets...")
    dataset = util.build_dataset(cfg)
    train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]

    # Load checkpoint
    if util.get_rank() == 0:
        logger.warning(f"Loading checkpoint from {args.ckpt}...")
    ckpt_state = torch.load(args.ckpt, map_location="cpu")

    # Create model based on type
    if args.model_type == "ultra":
        model = Ultra(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
        )
        model.load_state_dict(ckpt_state["model"] if "model" in ckpt_state else ckpt_state)
    else:  # pfn
        model = create_pfn_model(cfg, init=False, ckpt_path=None, map_location="cpu")
        model.load_state_dict(ckpt_state["model"] if "model" in ckpt_state else ckpt_state, strict=False)

    model = model.to(device)
    model.eval()

    if util.get_rank() == 0:
        logger.warning(f"Model loaded successfully. Testing on {len(test_data)} datasets...")

    # Move data to device
    test_data = [td.to(device) for td in test_data]

    # Build filtered data for each test graph
    filtered_data = []
    for test_graph in test_data:
        filtered = Data(
            edge_index=torch.cat([test_graph.edge_index, test_graph.target_edge_index], dim=1),
            edge_type=torch.cat([test_graph.edge_type, test_graph.target_edge_type]),
            num_nodes=test_graph.num_nodes,
        ).to(device)
        filtered_data.append(filtered)

    # Test on each dataset
    all_results = []
    transductive_results = []
    inductive_results = []

    for idx, (test_graph, filtered) in enumerate(zip(test_data, filtered_data)):
        dataset_name = getattr(test_graph, "dataset", f"graph_{idx}")

        metrics = test_single_graph(cfg, model, test_graph, filtered, logger, dataset_name)

        if util.get_rank() == 0 and metrics:
            metrics['dataset'] = dataset_name
            all_results.append(metrics)

            if is_transductive(dataset_name):
                transductive_results.append(metrics)
            else:
                inductive_results.append(metrics)

    # Compute and log averages
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("=" * 80)
        logger.warning("SUMMARY")
        logger.warning("=" * 80)

        # Overall average
        if all_results:
            avg_metrics = defaultdict(float)
            for result in all_results:
                for key in ['mr', 'mrr', 'hits@1', 'hits@10', 'hits@50']:
                    avg_metrics[key] += result[key]
            for key in avg_metrics:
                avg_metrics[key] /= len(all_results)

            logger.warning(f"\nOVERALL AVERAGE ({len(all_results)} datasets):")
            logger.warning(f"  MR: {avg_metrics['mr']:.2f}")
            logger.warning(f"  MRR: {avg_metrics['mrr']:.4f}")
            logger.warning(f"  Hits@1: {avg_metrics['hits@1']:.4f}")
            logger.warning(f"  Hits@10: {avg_metrics['hits@10']:.4f}")
            logger.warning(f"  Hits@50: {avg_metrics['hits@50']:.4f}")

        # Transductive average
        if transductive_results:
            trans_avg = defaultdict(float)
            for result in transductive_results:
                for key in ['mr', 'mrr', 'hits@1', 'hits@10', 'hits@50']:
                    trans_avg[key] += result[key]
            for key in trans_avg:
                trans_avg[key] /= len(transductive_results)

            logger.warning(f"\nTRANSDUCTIVE AVERAGE ({len(transductive_results)} datasets):")
            logger.warning(f"  MR: {trans_avg['mr']:.2f}")
            logger.warning(f"  MRR: {trans_avg['mrr']:.4f}")
            logger.warning(f"  Hits@1: {trans_avg['hits@1']:.4f}")
            logger.warning(f"  Hits@10: {trans_avg['hits@10']:.4f}")
            logger.warning(f"  Hits@50: {trans_avg['hits@50']:.4f}")

        # Inductive average
        if inductive_results:
            ind_avg = defaultdict(float)
            for result in inductive_results:
                for key in ['mr', 'mrr', 'hits@1', 'hits@10', 'hits@50']:
                    ind_avg[key] += result[key]
            for key in ind_avg:
                ind_avg[key] /= len(inductive_results)

            logger.warning(f"\nINDUCTIVE AVERAGE ({len(inductive_results)} datasets):")
            logger.warning(f"  MR: {ind_avg['mr']:.2f}")
            logger.warning(f"  MRR: {ind_avg['mrr']:.4f}")
            logger.warning(f"  Hits@1: {ind_avg['hits@1']:.4f}")
            logger.warning(f"  Hits@10: {ind_avg['hits@10']:.4f}")
            logger.warning(f"  Hits@50: {ind_avg['hits@50']:.4f}")

        # Save results to CSV
        csv_file = os.path.join(working_dir, "test_results.csv")
        with open(csv_file, "w", newline='') as f:
            fieldnames = ['dataset', 'mr', 'mrr', 'hits@1', 'hits@10', 'hits@50']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)

            # Add average rows
            if all_results:
                writer.writerow({
                    'dataset': 'OVERALL_AVERAGE',
                    'mr': avg_metrics['mr'],
                    'mrr': avg_metrics['mrr'],
                    'hits@1': avg_metrics['hits@1'],
                    'hits@10': avg_metrics['hits@10'],
                    'hits@50': avg_metrics['hits@50'],
                })
            if transductive_results:
                writer.writerow({
                    'dataset': 'TRANSDUCTIVE_AVERAGE',
                    'mr': trans_avg['mr'],
                    'mrr': trans_avg['mrr'],
                    'hits@1': trans_avg['hits@1'],
                    'hits@10': trans_avg['hits@10'],
                    'hits@50': trans_avg['hits@50'],
                })
            if inductive_results:
                writer.writerow({
                    'dataset': 'INDUCTIVE_AVERAGE',
                    'mr': ind_avg['mr'],
                    'mrr': ind_avg['mrr'],
                    'hits@1': ind_avg['hits@1'],
                    'hits@10': ind_avg['hits@10'],
                    'hits@50': ind_avg['hits@50'],
                })

        logger.warning(f"\nResults saved to: {csv_file}")
        logger.warning("=" * 80)
