import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pfn import util, tasks
from model.ultra.encoder import StructureEncoderRelationAware
from script.pretrain_pfn import _build_structure_encoder, _build_semantic_encoder

separator = ">" * 30
line = "-" * 30


def compute_embeddings(model, data, triples, enhance_structure=False, structure_score_enhance=False):
    """
    计算三元组的 embeddings。

    Args:
        model: StructureEncoderRelationAware
        data: 图数据
        triples: [N, 3] 三元组
        enhance_structure: 是否计算 enhance 特征
        structure_score_enhance: 是否计算 score 特征

    Returns:
        embeddings: [N, *, D] 特征向量
    """
    with torch.no_grad():
        # 获取基础 embeddings
        h_emb, t_emb, r_emb = model(data, triples.unsqueeze(0))  # [1, N, D]
        h_emb = h_emb.squeeze(0)  # [N, D_entity]
        t_emb = t_emb.squeeze(0)  # [N, D_entity]

        # 基础特征: [N, 2, D] - 只使用 h 和 t，因为 relation-aware context 中 relation 相同
        base_feat = torch.stack([h_emb, t_emb], dim=1)  # [N, 2, D]

        features = [base_feat]

        # Enhance 特征
        if enhance_structure:
            h_s = h_emb
            r_s = r_emb
            t_s = t_emb

            # TransE, DistMult, Cosine
            transe_feat = h_s + r_s - t_s  # [N, D]
            distmult_feat = h_s * r_s * t_s  # [N, D]
            cos_feat = F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8).unsqueeze(-1).expand(-1, h_s.size(-1))  # [N, D]

            enhance_feat = torch.stack([transe_feat, distmult_feat, cos_feat], dim=1)  # [N, 3, D]
            features.append(enhance_feat)

        # Structure score 特征
        if structure_score_enhance:
            scores = model.get_mlp_scores(t_emb.unsqueeze(0)).squeeze(0)  # [N]
            score_feat = scores.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h_emb.size(-1))  # [N, 1, D]
            features.append(score_feat)

        # 拼接所有特征
        all_features = torch.cat(features, dim=1)  # [N, *, D]

        return all_features


def flatten_features(features):
    """将 [N, *, D] 展平为 [N, *×D]"""
    N = features.size(0)
    return features.reshape(N, -1)


def compute_distances(query_feat, context_feats, metric='cosine'):
    """
    计算 query 与 context 之间的距离。

    Args:
        query_feat: [D] 或 [1, D]
        context_feats: [M, D]
        metric: 'cosine' 或 'euclidean'

    Returns:
        distances: [M]
    """
    if query_feat.dim() == 1:
        query_feat = query_feat.unsqueeze(0)

    if metric == 'cosine':
        # Cosine similarity (higher is closer)
        sim = F.cosine_similarity(query_feat, context_feats, dim=-1)
        return 1 - sim  # Convert to distance (lower is closer)
    elif metric == 'euclidean':
        return torch.norm(context_feats - query_feat, dim=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def analyze_knn(query_feat, context_feats, context_labels, k=5, metric='cosine'):
    """
    分析 KNN 结果。

    Args:
        query_feat: [D]
        context_feats: [M, D]
        context_labels: [M] (0 or 1)
        k: top-k neighbors
        metric: distance metric

    Returns:
        dict with analysis results
    """
    distances = compute_distances(query_feat, context_feats, metric)

    # 找到 top-k 最近邻
    topk_dists, topk_indices = torch.topk(distances, k=min(k, len(distances)), largest=False)
    topk_labels = context_labels[topk_indices]

    # 分别计算正负样本的平均距离
    pos_mask = context_labels == 1
    neg_mask = context_labels == 0

    pos_dists = distances[pos_mask]
    neg_dists = distances[neg_mask]

    # 新增指标：交叉比较
    # 对于正样本：有多少负样本比最远的正样本还近
    max_pos_dist = pos_dists.max().item() if len(pos_dists) > 0 else 0.0
    num_neg_closer_than_max_pos = (neg_dists < max_pos_dist).sum().item() if len(neg_dists) > 0 else 0

    # 对于负样本：有多少正样本比最近的负样本还近
    min_neg_dist = neg_dists.min().item() if len(neg_dists) > 0 else float('inf')
    num_pos_closer_than_min_neg = (pos_dists < min_neg_dist).sum().item() if len(pos_dists) > 0 else 0

    results = {
        'topk_indices': topk_indices.cpu().numpy(),
        'topk_distances': topk_dists.cpu().numpy(),
        'topk_labels': topk_labels.cpu().numpy(),
        'avg_pos_dist': pos_dists.mean().item() if len(pos_dists) > 0 else float('inf'),
        'avg_neg_dist': neg_dists.mean().item() if len(neg_dists) > 0 else float('inf'),
        'min_pos_dist': pos_dists.min().item() if len(pos_dists) > 0 else float('inf'),
        'min_neg_dist': min_neg_dist,
        'max_pos_dist': max_pos_dist,
        'max_neg_dist': neg_dists.max().item() if len(neg_dists) > 0 else 0.0,
        'num_pos_in_topk': topk_labels.sum().item(),
        'num_neg_in_topk': (topk_labels == 0).sum().item(),
        'num_pos_context': len(pos_dists),
        'num_neg_context': len(neg_dists),
        # 新增交叉指标
        'num_neg_closer_than_max_pos': num_neg_closer_than_max_pos,
        'num_pos_closer_than_min_neg': num_pos_closer_than_min_neg,
    }

    return results


def test_knn_consistency(cfg, model, data, num_test_samples=100, k=10):
    """
    测试 KNN 一致性：正样本应该与正样本更近，负样本应该与负样本更近。
    同时测试欧氏距离和余弦距离。

    Args:
        cfg: 配置
        model: structure encoder
        data: 图数据
        num_test_samples: 测试样本数量
        k: KNN 的 k
    """
    device = data.edge_index.device
    enhance_structure = bool(cfg.model.get("enhance_structure", False))
    structure_score_enhance = bool(cfg.model.get("structure_score_enhance", False))

    logger = util.get_root_logger()
    logger.warning(separator)
    logger.warning("KNN Consistency Test")
    logger.warning(f"enhance_structure: {enhance_structure}")
    logger.warning(f"structure_score_enhance: {structure_score_enhance}")
    logger.warning(f"Testing {num_test_samples} samples with k={k}")
    logger.warning(separator)

    # 采样测试三元组
    num_available = min(num_test_samples, data.target_edge_index.shape[1])
    test_edges = data.target_edge_index[:, :num_available]
    test_types = data.target_edge_type[:num_available]
    test_triples = torch.cat([test_edges.t(), test_types.unsqueeze(-1)], dim=-1).to(device)

    logger.warning(f"Actually testing {num_available} samples")

    # 为每个测试样本构建上下文
    results_pos_cosine = []  # 正样本的结果 (cosine)
    results_neg_cosine = []  # 负样本的结果 (cosine)
    results_pos_euclidean = []  # 正样本的结果 (euclidean)
    results_neg_euclidean = []  # 负样本的结果 (euclidean)

    for i in range(len(test_triples)):
        if i % 20 == 0:
            logger.warning(f"Processing sample {i+1}/{len(test_triples)}...")

        test_triple = test_triples[i:i+1]  # [1, 3]

        # 构建上下文：使用 relation-aware context
        batch_with_neg = tasks.negative_sampling_tail(
            data,
            test_triple,
            num_negative=cfg.task.num_negative,
            strict=cfg.task.strict_negative,
        )  # [1, N+1, 3]

        context_triples, context_labels = tasks.build_context_relation_aware(
            data,
            batch_with_neg,
            num_pos=cfg.task.num_pos,
            num_neg=cfg.task.num_neg,
        )

        context_triples = context_triples[0].to(device)  # [M, 3]
        context_labels = context_labels[0].to(device)  # [M]

        # 计算 embeddings
        test_feat = compute_embeddings(model, data, test_triple, enhance_structure, structure_score_enhance)
        context_feat = compute_embeddings(model, data, context_triples, enhance_structure, structure_score_enhance)

        # 展平特征
        test_feat_flat = flatten_features(test_feat).squeeze(0)  # [D]
        context_feat_flat = flatten_features(context_feat)  # [M, D]

        # 分析 KNN - Cosine
        knn_results_cos = analyze_knn(test_feat_flat, context_feat_flat, context_labels, k=k, metric='cosine')
        results_pos_cosine.append(knn_results_cos)

        # 分析 KNN - Euclidean
        knn_results_euc = analyze_knn(test_feat_flat, context_feat_flat, context_labels, k=k, metric='euclidean')
        results_pos_euclidean.append(knn_results_euc)

        # 也测试一个负样本
        if len(batch_with_neg[0]) > 1:
            neg_triple = batch_with_neg[0, 1:2, :]  # [1, 3] 第一个负样本
            neg_feat = compute_embeddings(model, data, neg_triple, enhance_structure, structure_score_enhance)
            neg_feat_flat = flatten_features(neg_feat).squeeze(0)

            # Cosine
            knn_results_neg_cos = analyze_knn(neg_feat_flat, context_feat_flat, context_labels, k=k, metric='cosine')
            results_neg_cosine.append(knn_results_neg_cos)

            # Euclidean
            knn_results_neg_euc = analyze_knn(neg_feat_flat, context_feat_flat, context_labels, k=k, metric='euclidean')
            results_neg_euclidean.append(knn_results_neg_euc)

    # 报告结果 - 两种距离度量
    for metric_name, results_pos, results_neg in [
        ('COSINE', results_pos_cosine, results_neg_cosine),
        ('EUCLIDEAN', results_pos_euclidean, results_neg_euclidean)
    ]:
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning(f"METRIC: {metric_name} DISTANCE")
        logger.warning("=" * 80)

        # 正样本统计
        logger.warning(line)
        logger.warning(f"POSITIVE TEST SAMPLES (ground truth triples) - {len(results_pos)} samples:")
        logger.warning(line)

        avg_pos_dist_to_pos = np.mean([r['avg_pos_dist'] for r in results_pos])
        avg_pos_dist_to_neg = np.mean([r['avg_neg_dist'] for r in results_pos])
        avg_pos_in_topk = np.mean([r['num_pos_in_topk'] for r in results_pos])
        avg_num_pos_context = np.mean([r['num_pos_context'] for r in results_pos])
        avg_num_neg_context = np.mean([r['num_neg_context'] for r in results_pos])

        # 新增指标
        avg_neg_closer_than_max_pos = np.mean([r['num_neg_closer_than_max_pos'] for r in results_pos])
        total_neg_context = np.mean([r['num_neg_context'] for r in results_pos])
        ratio_neg_closer = avg_neg_closer_than_max_pos / total_neg_context if total_neg_context > 0 else 0

        logger.warning(f"Average #positive context: {avg_num_pos_context:.1f}")
        logger.warning(f"Average #negative context: {avg_num_neg_context:.1f}")
        logger.warning(f"Average distance to positive context: {avg_pos_dist_to_pos:.4f}")
        logger.warning(f"Average distance to negative context: {avg_pos_dist_to_neg:.4f}")
        logger.warning(f"Distance ratio (pos/neg): {avg_pos_dist_to_pos / avg_pos_dist_to_neg:.4f}")
        logger.warning(f"Average #positive in top-{k}: {avg_pos_in_topk:.2f} / {k}")
        logger.warning(f"Average #negative closer than farthest positive: {avg_neg_closer_than_max_pos:.2f} / {total_neg_context:.1f} ({ratio_neg_closer*100:.1f}%)")

        if avg_pos_dist_to_pos < avg_pos_dist_to_neg:
            logger.warning("✓ GOOD: Positive samples are closer to positive context")
        else:
            logger.warning("✗ BAD: Positive samples are NOT closer to positive context")

        # 负样本统计
        logger.warning(line)
        logger.warning(f"NEGATIVE TEST SAMPLES (corrupted triples) - {len(results_neg)} samples:")
        logger.warning(line)

        avg_neg_dist_to_pos = np.mean([r['avg_pos_dist'] for r in results_neg])
        avg_neg_dist_to_neg = np.mean([r['avg_neg_dist'] for r in results_neg])
        avg_neg_in_topk = np.mean([r['num_neg_in_topk'] for r in results_neg])

        # 新增指标
        avg_pos_closer_than_min_neg = np.mean([r['num_pos_closer_than_min_neg'] for r in results_neg])
        total_pos_context = np.mean([r['num_pos_context'] for r in results_neg])
        ratio_pos_closer = avg_pos_closer_than_min_neg / total_pos_context if total_pos_context > 0 else 0

        logger.warning(f"Average distance to positive context: {avg_neg_dist_to_pos:.4f}")
        logger.warning(f"Average distance to negative context: {avg_neg_dist_to_neg:.4f}")
        logger.warning(f"Distance ratio (neg/pos): {avg_neg_dist_to_neg / avg_neg_dist_to_pos:.4f}")
        logger.warning(f"Average #negative in top-{k}: {avg_neg_in_topk:.2f} / {k}")
        logger.warning(f"Average #positive closer than nearest negative: {avg_pos_closer_than_min_neg:.2f} / {total_pos_context:.1f} ({ratio_pos_closer*100:.1f}%)")

        if avg_neg_dist_to_neg < avg_neg_dist_to_pos:
            logger.warning("✓ GOOD: Negative samples are closer to negative context")
        else:
            logger.warning("✗ BAD: Negative samples are NOT closer to negative context")

        # 总结
        logger.warning(line)
        logger.warning(f"SUMMARY ({metric_name}):")
        logger.warning(line)

        pos_consistency = avg_pos_dist_to_pos < avg_pos_dist_to_neg
        neg_consistency = avg_neg_dist_to_neg < avg_neg_dist_to_pos

        if pos_consistency and neg_consistency:
            logger.warning("✓✓ EXCELLENT: Both positive and negative samples show expected distance patterns")
        elif pos_consistency:
            logger.warning("✓✗ PARTIAL: Only positive samples show expected pattern")
        elif neg_consistency:
            logger.warning("✗✓ PARTIAL: Only negative samples show expected pattern")
        else:
            logger.warning("✗✗ POOR: Neither positive nor negative samples show expected patterns")

    logger.warning("")
    logger.warning(separator)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    logger = util.get_root_logger()
    logger.warning("Random seed: %d" % args.seed)
    logger.warning("Config file: %s" % args.config)

    torch.manual_seed(args.seed)

    # 加载数据集
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    # 获取训练数据
    if hasattr(dataset, '_data') and isinstance(dataset._data, list):
        # JointDataset
        train_data = dataset._data[0][0]  # 第一个图的训练数据
    else:
        # 单个数据集
        train_data = dataset[0]

    train_data = train_data.to(device)

    logger.warning(f"Dataset: {train_data.dataset if hasattr(train_data, 'dataset') else 'unknown'}")
    logger.warning(f"Nodes: {train_data.num_nodes}, Edges: {train_data.edge_index.shape[1]}")
    logger.warning(f"Target edges: {train_data.target_edge_index.shape[1]}")

    # 加载模型
    structure_encoder = _build_structure_encoder(cfg)

    if structure_encoder is None:
        logger.error("structure_encoder is None! Please set structure_encoder_name in config.")
        sys.exit(1)

    # 加载 checkpoint
    ckpt_path = cfg.train.get("structure_encoder_path", None)
    if ckpt_path and os.path.exists(ckpt_path):
        logger.warning(f"Loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        structure_encoder.load_state_dict(sd, strict=False)
    else:
        logger.warning("No checkpoint loaded, using random initialization")

    structure_encoder = structure_encoder.to(device)
    structure_encoder.eval()

    # 运行 KNN 测试
    test_knn_consistency(
        cfg,
        structure_encoder,
        train_data,
        num_test_samples=100,
        k=10,
    )
