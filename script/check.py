import copy
import os
import sys
import logging
from typing import Any

import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pfn import util, tasks
from script.run import create_model, test

separator = ">" * 30
line = "-" * 30


def normalize_graph_splits(dataset):
    # Check for JointDataset: has .data tuple and .graph_specs
    if hasattr(dataset, "data") and isinstance(dataset.data, tuple) and len(dataset.data) == 3:
        train_graphs, valid_graphs, test_graphs = dataset.data
        names = list(getattr(dataset, "graph_specs", []))
        if not names:
            names = [f"graph_{i}" for i in range(len(train_graphs))]
        return train_graphs, valid_graphs, test_graphs, names

    # Legacy check for _data attribute
    if hasattr(dataset, "_data") and isinstance(dataset._data, list) and len(dataset._data) == 3:
        train_graphs, valid_graphs, test_graphs = dataset._data
        names = list(getattr(dataset, "graph_specs", []))
        if not names:
            names = [f"graph_{i}" for i in range(len(train_graphs))]
        return train_graphs, valid_graphs, test_graphs, names

    # Single dataset case
    train_graph, valid_graph, test_graph = dataset[0], dataset[1], dataset[2]
    name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
    version = getattr(dataset, "dataset_version", None)
    if version is not None:
        name = f"{name}:{version}"
    return [train_graph], [valid_graph], [test_graph], [name]


def build_filtered_graph(graph: Data) -> Data:
    filtered = Data(
        edge_index=torch.cat([graph.edge_index, graph.target_edge_index], dim=1),
        edge_type=torch.cat([graph.edge_type, graph.target_edge_type], dim=0),
        num_nodes=graph.num_nodes,
    )
    return filtered


def subsample_graph(graph: Data, num_edges: int | None, seed: int) -> Data:
    if num_edges is None or num_edges <= 0:
        return graph

    total_edges = int(graph.target_edge_type.numel())
    if num_edges >= total_edges:
        return graph

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(total_edges, generator=generator)[:num_edges]

    sampled = copy.copy(graph)
    sampled.target_edge_index = graph.target_edge_index[:, perm].clone()
    sampled.target_edge_type = graph.target_edge_type[perm].clone()
    return sampled


def _context_stats_dict(expected_total: int) -> dict[str, Any]:
    return {
        "rows_checked": 0,
        "failed_rows": 0,
        "zero_pos_rows": 0,
        "pos_shortfall_rows": 0,
        "neg_shortfall_rows": 0,
        "total_mismatch_rows": 0,
        "pos_counts": [],
        "neg_counts": [],
        "expected_total": expected_total,
    }


def _update_context_stats(stats: dict[str, Any], labels: torch.Tensor, cfg):
    pos_count = int((labels == 1).sum().item())
    neg_count = int((labels == 0).sum().item())
    total_count = int(labels.numel())

    stats["rows_checked"] += 1
    stats["pos_counts"].append(pos_count)
    stats["neg_counts"].append(neg_count)

    if pos_count == 0:
        stats["zero_pos_rows"] += 1
    if pos_count < int(cfg.task.num_pos):
        stats["pos_shortfall_rows"] += 1
    if neg_count < int(cfg.task.num_neg):
        stats["neg_shortfall_rows"] += 1
    if total_count != stats["expected_total"]:
        stats["total_mismatch_rows"] += 1


def finalize_context_stats(stats: dict[str, Any]) -> dict[str, Any]:
    rows_checked = int(stats["rows_checked"])
    if rows_checked == 0:
        return {
            **stats,
            "zero_pos_ratio": 0.0,
            "pos_mean": 0.0,
            "pos_min": 0,
            "pos_max": 0,
            "neg_mean": 0.0,
            "neg_min": 0,
            "neg_max": 0,
        }

    pos_counts = stats["pos_counts"]
    neg_counts = stats["neg_counts"]
    return {
        **stats,
        "zero_pos_ratio": stats["zero_pos_rows"] / rows_checked,
        "pos_mean": sum(pos_counts) / rows_checked,
        "pos_min": min(pos_counts),
        "pos_max": max(pos_counts),
        "neg_mean": sum(neg_counts) / rows_checked,
        "neg_min": min(neg_counts),
        "neg_max": max(neg_counts),
    }


@torch.no_grad()
def collect_context_sampling_stats(cfg, graph: Data, split_name: str, seed: int) -> dict[str, Any]:
    num_rows = int(cfg.check.context.get("num_rows", 0))
    batch_size = int(cfg.check.context.get("batch_size", cfg.train.batch_size))
    strict_negative = bool(cfg.check.context.get("strict_negative", cfg.task.get("strict_negative", True)))
    expected_total = int(cfg.task.num_pos) + int(cfg.task.num_neg)
    stats = _context_stats_dict(expected_total)

    sampled_graph = subsample_graph(graph, num_rows, seed)
    triplets = torch.cat([sampled_graph.target_edge_index, sampled_graph.target_edge_type.unsqueeze(0)]).t()

    if triplets.numel() == 0:
        return finalize_context_stats(stats)

    for start in range(0, triplets.size(0), max(1, batch_size)):
        batch = triplets[start:start + max(1, batch_size)]
        batch_with_neg = tasks.negative_sampling_tail(
            graph,
            batch,
            num_negative=int(cfg.task.num_negative),
            strict=strict_negative,
        )
        try:
            _, context_labels = tasks.build_context_relation_aware(
                graph,
                batch_with_neg,
                num_pos=int(cfg.task.num_pos),
                num_neg=int(cfg.task.num_neg),
            )
            for labels in context_labels:
                _update_context_stats(stats, labels, cfg)
        except RuntimeError:
            for row in batch_with_neg:
                try:
                    _, context_labels = tasks.build_context_relation_aware(
                        graph,
                        row.unsqueeze(0),
                        num_pos=int(cfg.task.num_pos),
                        num_neg=int(cfg.task.num_neg),
                    )
                    _update_context_stats(stats, context_labels[0], cfg)
                except RuntimeError:
                    stats["failed_rows"] += 1

    return finalize_context_stats(stats)


def log_context_stats(logger, dataset_name: str, split_name: str, stats: dict[str, Any], cfg):
    logger.warning(
        "[%s][%s][context] rows=%d failed=%d pos(mean/min/max)=%.2f/%d/%d neg(mean/min/max)=%.2f/%d/%d zero_pos=%d(%.2f%%) pos_shortfall=%d neg_shortfall=%d total_mismatch=%d",
        dataset_name,
        split_name,
        stats["rows_checked"],
        stats["failed_rows"],
        stats["pos_mean"],
        stats["pos_min"],
        stats["pos_max"],
        stats["neg_mean"],
        stats["neg_min"],
        stats["neg_max"],
        stats["zero_pos_rows"],
        stats["zero_pos_ratio"] * 100.0,
        stats["pos_shortfall_rows"],
        stats["neg_shortfall_rows"],
        stats["total_mismatch_rows"],
    )

    warn_zero_pos_ratio_ge = float(cfg.check.context.get("warn_zero_pos_ratio_ge", 1.0))
    warn_mean_pos_below = float(cfg.check.context.get("warn_mean_pos_below", -1.0))

    if stats["zero_pos_ratio"] >= warn_zero_pos_ratio_ge:
        logger.warning(
            "[%s][%s][context] warning: zero-positive ratio %.2f%% >= %.2f%%",
            dataset_name,
            split_name,
            stats["zero_pos_ratio"] * 100.0,
            warn_zero_pos_ratio_ge * 100.0,
        )
    if stats["pos_mean"] < warn_mean_pos_below:
        logger.warning(
            "[%s][%s][context] warning: mean positive count %.2f < %.2f",
            dataset_name,
            split_name,
            stats["pos_mean"],
            warn_mean_pos_below,
        )
    if stats["failed_rows"] > 0:
        logger.warning(
            "[%s][%s][context] warning: %d rows failed to construct enough negatives",
            dataset_name,
            split_name,
            stats["failed_rows"],
        )
    if stats["total_mismatch_rows"] > 0:
        logger.warning(
            "[%s][%s][context] warning: %d rows did not preserve expected context size %d",
            dataset_name,
            split_name,
            stats["total_mismatch_rows"],
            stats["expected_total"],
        )


@torch.no_grad()
def fast_evaluate_split(cfg, model, graph: Data, filtered_graph: Data, split_name: str, logger):
    metrics = test(
        cfg,
        model,
        graph,
        device=util.get_device(cfg),
        logger=logger,
        filtered_data=filtered_graph,
        return_metrics=True,
        split=split_name,
    )

    normalized = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            normalized[key] = float(value.item())
        else:
            normalized[key] = float(value)
    return normalized


def run_checks_for_graph(cfg, model, dataset_name: str, train_graph: Data, test_graph: Data, seed: int, logger):
    result = {"dataset": dataset_name}
    logger.warning(separator)
    logger.warning("Dataset: %s", dataset_name)
    logger.warning(
        "train_triples=%d test_triples=%d num_nodes=%d",
        train_graph.target_edge_index.shape[1],
        test_graph.target_edge_index.shape[1],
        train_graph.num_nodes,
    )

    enabled_splits = set(cfg.check.get("enabled_splits", ["train", "test"]))

    if "train" in enabled_splits:
        train_context_stats = collect_context_sampling_stats(cfg, train_graph, "train", seed)
        log_context_stats(logger, dataset_name, "train", train_context_stats, cfg)
        result["train_context"] = train_context_stats

        fast_train_graph = subsample_graph(train_graph, int(cfg.check.fast_eval.get("num_edges", 0)), seed)
        train_filtered_graph = build_filtered_graph(train_graph).to(util.get_device(cfg))
        train_metrics = fast_evaluate_split(cfg, model, fast_train_graph, train_filtered_graph, "train", logger)
        logger.warning("[%s][train][fast_eval] %s", dataset_name, train_metrics)
        result["train_eval"] = train_metrics

    if "test" in enabled_splits:
        test_context_stats = collect_context_sampling_stats(cfg, test_graph, "test", seed + 1)
        log_context_stats(logger, dataset_name, "test", test_context_stats, cfg)
        result["test_context"] = test_context_stats

        fast_test_graph = subsample_graph(test_graph, int(cfg.check.fast_eval.get("num_edges", 0)), seed + 1)
        test_filtered_graph = build_filtered_graph(test_graph).to(util.get_device(cfg))
        test_metrics = fast_evaluate_split(cfg, model, fast_test_graph, test_filtered_graph, "test", logger)
        logger.warning("[%s][test][fast_eval] %s", dataset_name, test_metrics)
        result["test_eval"] = test_metrics

    return result


def summarize_results(results: list[dict[str, Any]], logger, cfg):
    if util.get_rank() != 0:
        return

    logger.warning(separator)
    logger.warning("Final summary")
    logger.warning(line)

    warn_zero_pos_ratio_ge = float(cfg.check.context.get("warn_zero_pos_ratio_ge", 1.0))
    weak_context = []
    train_mrrs = []
    test_mrrs = []

    for result in results:
        dataset_name = result["dataset"]
        train_zero = result.get("train_context", {}).get("zero_pos_ratio", 0.0)
        test_zero = result.get("test_context", {}).get("zero_pos_ratio", 0.0)
        train_mrr = result.get("train_eval", {}).get("mrr")
        test_mrr = result.get("test_eval", {}).get("mrr")

        if train_mrr is not None:
            train_mrrs.append(train_mrr)
        if test_mrr is not None:
            test_mrrs.append(test_mrr)
        if train_zero >= warn_zero_pos_ratio_ge or test_zero >= warn_zero_pos_ratio_ge:
            weak_context.append(dataset_name)

        logger.warning(
            "%s | train_zero_pos=%.2f%% | train_mrr=%s | test_zero_pos=%.2f%% | test_mrr=%s",
            dataset_name,
            train_zero * 100.0,
            f"{train_mrr:.4f}" if train_mrr is not None else "n/a",
            test_zero * 100.0,
            f"{test_mrr:.4f}" if test_mrr is not None else "n/a",
        )

    if train_mrrs:
        logger.warning("average train mrr: %.4f", sum(train_mrrs) / len(train_mrrs))
    if test_mrrs:
        logger.warning("average test mrr: %.4f", sum(test_mrrs) / len(test_mrrs))
    if weak_context:
        logger.warning("datasets with high zero-positive ratio: %s", ", ".join(weak_context))

    logger.warning(separator)


if __name__ == "__main__":
    args, vars_dict = util.parse_args()
    cfg = util.load_config(args.config, context=vars_dict)

    logger = util.get_root_logger(file=False)
    torch.manual_seed(args.seed + util.get_rank())

    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    train_graphs, valid_graphs, test_graphs, dataset_names = normalize_graph_splits(dataset)
    del valid_graphs

    model = create_model(
        cfg,
        init=bool(cfg.train.get("init_model", True)),
        ckpt_path=cfg.train.get("kgpfn_checkpoint", None),
        map_location="cpu",
    )
    model = model.to(device)
    model.eval()

    train_graphs = [graph.to(device) for graph in train_graphs]
    test_graphs = [graph.to(device) for graph in test_graphs]

    results = []
    for idx, (dataset_name, train_graph, test_graph) in enumerate(zip(dataset_names, train_graphs, test_graphs)):
        result = run_checks_for_graph(
            cfg,
            model,
            dataset_name=dataset_name,
            train_graph=train_graph,
            test_graph=test_graph,
            seed=args.seed + idx,
            logger=logger,
        )
        results.append(result)

    summarize_results(results, logger, cfg)
