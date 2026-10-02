import glob
import os
import sys
import copy
import math
import csv
import pprint
import logging
import io
import contextlib
import yaml
from typing import Any

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pfn import tasks, util
from model.ultra.encoder import StructureEncoderRelationAware
from model.kgpfnsem import KGPFN
from huggingface_hub import hf_hub_download
from utils.loading import build_custom_model, load_state_dict_matching
from model.tabpfn.architectures.base.config import ModelConfig
from model.tabpfn.architectures.base.custom_transformer import CustomPerFeatureTransformer
import wandb

separator = ">" * 30
line = "-" * 30

TRANSDUCTIVE_DATASETS = {
    "FB15k237", "FB15k237_10", "FB15k237_20", "FB15k237_50",
    "WN18RR", "CoDExSmall", "CoDExMedium", "CoDExLarge", "NELL995",
    "ConceptNet100k", "DBpedia100k", "YAGO310", "AristoV4", "Hetionet",
    "WDsinger", "NELL23k",
}
INDUCTIVE_DATASETS = {
    "FB15k237Inductive", "WN18RRInductive", "NELLInductive",
    "ILPC2022", "HM",
}
FULL_INDUCTIVE_DATASETS = {
    "NLIngram", "FBIngram", "WKIngram", "WikiTopicsMT1", "WikiTopicsMT2",
    "WikiTopicsMT3", "WikiTopicsMT4", "Metafam", "FBNELL",
}


def _match_dataset_family(dataset_name: str) -> str:
    def _startswith_any(name: str, prefixes: set[str]) -> bool:
        return any(name == p or name.startswith(f"{p}-") or name.startswith(f"{p}_") for p in prefixes)

    if _startswith_any(dataset_name, TRANSDUCTIVE_DATASETS):
        return "transductive"
    if _startswith_any(dataset_name, INDUCTIVE_DATASETS):
        return "inductive"
    if _startswith_any(dataset_name, FULL_INDUCTIVE_DATASETS):
        return "full_inductive"
    return "unknown"


def _format_metric(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.6f}"


def _nanmean(vals: list[float]) -> float:
    valid_vals = [v for v in vals if not math.isnan(v)]
    if not valid_vals:
        return float("nan")
    return sum(valid_vals) / len(valid_vals)


EVAL_SIDES = ("head", "tail", "both")


def _empty_side_metrics() -> dict[str, float]:
    return {"mrr": float("nan"), "hits@10": float("nan")}


def _empty_dataset_record(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "family": _match_dataset_family(name),
        "head": _empty_side_metrics(),
        "tail": _empty_side_metrics(),
        "both": _empty_side_metrics(),
    }


def _side_record(side_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        side: {
            "mrr": side_metrics.get(side, {}).get("mrr", float("nan")),
            "hits@10": side_metrics.get(side, {}).get("hits@10", float("nan")),
        }
        for side in EVAL_SIDES
    }


def _avg_side_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        side: {
            "mrr": _nanmean([r[side]["mrr"] for r in records]),
            "hits@10": _nanmean([r[side]["hits@10"] for r in records]),
        }
        for side in EVAL_SIDES
    }


def _cat_or_empty(chunks: list[torch.Tensor], device) -> torch.Tensor:
    if chunks:
        return torch.cat(chunks)
    return torch.empty(0, dtype=torch.long, device=device)


def _gather_long_vector(local_vec: torch.Tensor, world_size: int, rank: int, device) -> torch.Tensor:
    local_vec = local_vec.to(device=device)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = local_vec.numel()
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    total = int(all_size.sum().item())
    gathered = torch.zeros(total, dtype=local_vec.dtype, device=device)
    if local_vec.numel() > 0:
        end = int(all_size.cumsum(0)[rank].item())
        start = end - int(all_size[rank].item())
        gathered[start:end] = local_vec
    if world_size > 1 and total > 0:
        dist.all_reduce(gathered, op=dist.ReduceOp.SUM)
    return gathered


def _ranking_metrics(
    all_ranking: torch.Tensor,
    all_num_negative: torch.Tensor,
    metric_names: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if all_ranking.numel() == 0:
        for metric in metric_names:
            if metric in ("precision", "recall", "f1"):
                continue
            out[metric] = float("nan")
        out.setdefault("mrr", float("nan"))
        out.setdefault("hits@10", float("nan"))
        return out

    ranking_f = all_ranking.float()
    for metric in metric_names:
        if metric == "mr":
            score = ranking_f.mean()
        elif metric == "mrr":
            score = (1 / ranking_f).mean()
        elif metric in ("precision", "recall", "f1"):
            continue
        elif metric.startswith("hits@"):
            values = metric[5:].split("_")
            threshold = int(values[0])
            if len(values) > 1:
                num_sample = int(values[1])
                fp_rate = (all_ranking - 1).float() / all_num_negative
                score = 0
                for i in range(threshold):
                    num_comb = math.factorial(num_sample - 1) / \
                            math.factorial(i) / math.factorial(num_sample - i - 1)
                    score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                score = score.mean()
            else:
                score = (ranking_f <= threshold).float().mean()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        out[metric] = float(score.item())
    out.setdefault("mrr", float((1 / ranking_f).mean().item()))
    out.setdefault("hits@10", float((ranking_f <= 10).float().mean().item()))
    return out


def _get_dataset_csv_path(cfg, split: str) -> str:
    csv_name = f"metrics.csv"
    checkpoint_dir = cfg.train.get("checkpoint_dir", ".")
    csv_dir = os.path.dirname(os.path.abspath(checkpoint_dir))
    os.makedirs(csv_dir, exist_ok=True)
    return os.path.join(csv_dir, csv_name)


def _write_side_block(rows: list[list[str]], name: str, rec: dict[str, Any]) -> None:
    first = True
    for side in EVAL_SIDES:
        label = name if first else ""
        first = False
        rows.append([label, side, "mrr", _format_metric(rec[side]["mrr"])])
        rows.append(["", "", "hit10", _format_metric(rec[side]["hits@10"])])


def _write_dataset_csv(
    csv_path: str,
    dataset_records: list[dict[str, Any]],
):

    rows: list[list[str]] = []
    for rec in dataset_records:
        _write_side_block(rows, rec["name"], rec)
        rows.append(["", "", "", ""])

    group_specs = [
        ("transductive_average", "transductive"),
        ("inductive_average", "inductive"),
        ("full_inductive_average", "full_inductive"),
    ]
    for idx, (label, group_name) in enumerate(group_specs):
        selected = [r for r in dataset_records if r["family"] == group_name]
        _write_side_block(rows, label, _avg_side_metrics(selected))
        if idx != len(group_specs) - 1:
            rows.append(["", "", "", ""])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return csv_path


def _build_semantic_encoder(cfg):
    sem_name = str(cfg.model.semantic_encoder.model_name).strip()
    if sem_name.lower() in ("", "none", "null"):
        return None
    from sentence_transformers import SentenceTransformer
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(sem_name)


def _build_structure_encoder(cfg):
    enabled = cfg.model.get("structure_encoder", True)
    if not enabled:
        return None
    entity_model_cfg = copy.deepcopy(cfg.model.entity_model)
    entity_chunk_size = entity_model_cfg.pop("entity_chunk_size", None)
    return StructureEncoderRelationAware(
        rel_model_cfg=copy.deepcopy(cfg.model.relation_model),
        entity_model_cfg=entity_model_cfg,
        entity_chunk_size=entity_chunk_size,
    )


def _set_module_trainable(module, trainable: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = trainable
    module.train(trainable)


def _resolve_inverse_relation_semantic_mode(cfg) -> str:
    mode = str(cfg.train.get("inverse_relation_semantic_mode", "text")).strip().lower()
    if mode not in ("text", "negate"):
        raise ValueError(f"unknown inverse_relation_semantic_mode={mode}, expected 'text' or 'negate'")
    return mode


def _triple_to_text(triple: torch.Tensor, id2e: dict[int, str], id2r: dict[int, str], num_relations: int):
    h = int(triple[0].item())
    t = int(triple[1].item())
    r = int(triple[2].item())
    h_text = id2e.get(h, str(h))
    t_text = id2e.get(t, str(t))
    if r in id2r:
        r_text = id2r[r]
    else:
        base_rel = num_relations // 2 if num_relations else 0
        if base_rel > 0 and r >= base_rel and (r - base_rel) in id2r:
            r_text = f"the reverse relation of {id2r[r - base_rel]}"
        else:
            # fallback：JointDataset 多图时 id2r/num_relations 可能不完整
            r_text = f"relation_{r}"
    return [h_text, r_text, t_text]


def _semantic_enabled(model_obj: Any) -> bool:
    inner = model_obj.module if hasattr(model_obj, "module") else model_obj
    return getattr(inner, "semantic_encoder", None) is not None


def _build_model_inputs(query_ids: torch.Tensor, context_ids: list[torch.Tensor], data_obj: Any, enable_text: bool):
    if not enable_text:
        return query_ids, context_ids
    id2e = getattr(data_obj, "train_id2entity", getattr(data_obj, "id2entity", {}))
    id2r = getattr(data_obj, "train_id2relation", getattr(data_obj, "id2relation", {}))
    num_relations = int(getattr(data_obj, "num_relations", 0))

    bsz, n_query, _ = query_ids.shape
    query_text = []
    context_text = []
    for i in range(bsz):
        q_rows = []
        for j in range(n_query):
            q_rows.append(_triple_to_text(query_ids[i, j], id2e, id2r, num_relations))
        query_text.append(q_rows)

        c_rows = []
        ctx_i = context_ids[i]
        for j in range(ctx_i.size(0)):
            c_rows.append(_triple_to_text(ctx_i[j], id2e, id2r, num_relations))
        context_text.append(c_rows)
    return {"id": query_ids, "text": query_text}, {"id": context_ids, "text": context_text}


def _score_all_candidate_tails(model, graph, eval_batch, context_ids, context_ys, eval_chunk_size: int) -> torch.Tensor:
    """[B, |V|] scores of every entity as the tail of each ``(h, r)`` row of ``eval_batch``.

    Structure-only TabICL models use the cached single-pass ``score_all_tails``;
    otherwise ``tasks.all_negative`` is scored in chunks through ``get_scores``,
    with text when the semantic encoder is on.
    """
    if model.supports_score_all_tails():
        return model.score_all_tails(
            graph, eval_batch, context_ids, context_ys, eval_chunk=eval_chunk_size, task_type="reg",
        )
    use_text = _semantic_enabled(model)
    t_batch, _ = tasks.all_negative(graph, eval_batch)
    batch_size, num_nodes, _ = t_batch.shape
    context_texts = None
    if use_text:
        _, context_pack = _build_model_inputs(eval_batch.unsqueeze(1), context_ids, graph, enable_text=True)
        context_texts = context_pack["text"]
    context_cache, context_ys = model.get_context_embeddings_cache(graph, context_ids, context_ys, context_texts)
    step = max(1, int(eval_chunk_size))
    chunks = []
    for start in range(0, num_nodes, step):
        query_x, _ = _build_model_inputs(t_batch[:, start:start + step, :], context_ids, graph, enable_text=use_text)
        scores = model.get_scores(
            graph,
            query_x=query_x,
            context_cache=context_cache,
            context_y=context_ys,
            task_type="reg",
        )
        chunks.append(scores.view(batch_size, -1))
    return torch.cat(chunks, dim=1)


def _get_context_sampling_args(cfg):
    # relation-aware context builder currently only needs pos/neg counts.
    return {
        "num_pos": int(cfg.task.num_pos),
        "num_neg": int(cfg.task.num_neg),
    }


def _subsample_targets(graphs, n):
    """Copy of each graph keeping at most ``n`` target triples. Observed edges stay intact."""
    n = int(n)
    out = []
    for graph in graphs:
        sampled = copy.deepcopy(graph)
        n_edges = sampled.target_edge_index.shape[1]
        if n_edges > n:
            mask = torch.randperm(n_edges)[:n]
            sampled.target_edge_index = sampled.target_edge_index[:, mask]
            sampled.target_edge_type = sampled.target_edge_type[mask]
        out.append(sampled)
    return out


def _get_project_root():
    """项目根目录（script 的上级），用于解析相对路径"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str | None, project_root: str) -> str | None:
    """相对路径转为相对于 project_root 的绝对路径；绝对路径或空则原样返回"""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def _get_limix_config_path(cfg):
    # 允许从配置覆盖 limix config 路径；默认沿用当前脚本内的本地配置文件
    default_dir = os.path.join(_get_project_root(), "config", "limix")
    default_path = os.path.join(default_dir, "limix_config.yaml")
    return cfg.train.get("limix_config_path", default_path)


def _get_limix_thinking_rows(cfg):
    limix_config_path = _get_limix_config_path(cfg)
    try:
        with open(limix_config_path, "r", encoding="utf-8") as f:
            limix_cfg = yaml.safe_load(f) or {}
        return int(limix_cfg.get("num_thinking_rows", 0))
    except Exception:
        return 0


def _save_configs(cfg, args_config: str, working_dir: str):
    """将本次运行的 input config 与 feature_transformer config 保存到 working_dir/config/"""
    import shutil
    config_save_dir = os.path.join(working_dir, "config")
    os.makedirs(config_save_dir, exist_ok=True)
    # 1) 保存 -c 指定的 input config
    src = os.path.abspath(args_config)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(config_save_dir, os.path.basename(src)))
    # 2) 根据 feature_transformer 类型保存对应 config
    ft_type = cfg.model.get("feature_transformer", "limix")
    if ft_type == "tabpfn":
        ft_path = cfg.train.get("tabpfn_config_path", "./config/tabpfn/tabpfn.yaml")
    elif ft_type == "tabicl":
        ft_path = cfg.train.get("tabicl_config_path", "./config/tabicl/tabicl.yaml")
    else:
        ft_path = _get_limix_config_path(cfg)
    if ft_path and os.path.isfile(ft_path):
        shutil.copy2(ft_path, os.path.join(config_save_dir, os.path.basename(ft_path)))


def wandb_init(cfg, logger):
    """
    初始化 wandb。支持通过 cfg.train.wandb_api_key 直接登录。
    返回 (enabled, run_or_none)。
    """
    use_wandb = bool(cfg.train.get("use_wandb", False))
    if not use_wandb:
        return False, None


    api_key = cfg.train.get("wandb_api_key", None)
    if api_key:
        # 允许通过配置里的 key 直接登录
        wandb.login(key=api_key, relogin=True)

    adv_temp = float(cfg.task.get("adversarial_temperature", 0.0))
    num_thinking_rows = _get_limix_thinking_rows(cfg)
    thinking_tag = "think" if num_thinking_rows > 0 else "nothink"
    temp_tag = f"advT{adv_temp:g}" if adv_temp > 0 else "advOff"
    auto_run_name = f"{temp_tag}_{thinking_tag}{num_thinking_rows}"
    run_name = cfg.train.get("wandb_run_name", None) or auto_run_name

    run = wandb.init(
        project=cfg.train.get("wandb_project", "ultrapfn-pretrain"),
        entity=cfg.train.get("wandb_entity", None),
        name=run_name,
        config={
            "num_pos": _get_context_sampling_args(cfg)["num_pos"],
            "num_neg": _get_context_sampling_args(cfg)["num_neg"],
            "num_negative": int(cfg.task.get("num_negative", 64)),
            "adversarial_temperature": adv_temp,
            "num_thinking_rows": num_thinking_rows,
            "train_structure_encoder": bool(cfg.train.get("train_structure_encoder", True)),
        },
    )
    logger.warning("wandb initialized: project=%s, run=%s", run.project, run.name)
    return True, run


def create_model(cfg, init: bool = False, ckpt_path: str | None = None, map_location: str = "cpu"):
    """
    根据配置创建 KGPFN。
    - init=True: 按 create_init_model 思路分别初始化 structure_encoder 与 feature_transformer
    - ckpt_path: 若提供，则在上述步骤后再加载整体 checkpoint 权重（优先级最高）
    """
    structure_encoder = _build_structure_encoder(cfg)

    # feature_transformer: limix (default) or tabpfn or tabicl
    ft_type = cfg.model.get("feature_transformer", "limix")
    if ft_type == "tabpfn":
        tabpfn_config_path = cfg.train.get("tabpfn_config_path", "./config/tabpfn/tabpfn.yaml")
        with open(tabpfn_config_path, "r", encoding="utf-8") as f:
            tabpfn_cfg = yaml.safe_load(f)
        # "rope" is not a valid ModelConfig Literal — extract it before validation
        use_rope = tabpfn_cfg.get("feature_positional_embedding") == "rope"
        cfg_for_model = dict(tabpfn_cfg)
        if use_rope:
            cfg_for_model["feature_positional_embedding"] = None
        model_config = ModelConfig(**ModelConfig.upgrade_config(cfg_for_model))
        # Restore rope so CustomPerFeatureTransformer can detect it
        if use_rope:
            model_config.feature_positional_embedding = "rope"
        structure_encoder_dim = tabpfn_cfg.get("structure_encoder_dim", 64)
        feature_transformer = CustomPerFeatureTransformer(
            config=model_config,
            structure_encoder_dim=structure_encoder_dim,
            n_out=model_config.max_num_classes or 10,
        )
    elif ft_type == "tabicl":
        from model.tabicl.model.custom_tabicl import build_custom_tabicl
        tabicl_config_path = cfg.train.get("tabicl_config_path", "./config/tabicl/tabicl.yaml")
        with open(tabicl_config_path, "r", encoding="utf-8") as f:
            tabicl_cfg = yaml.safe_load(f)
        structure_encoder_dim = tabicl_cfg.get("structure_encoder_dim", 64)
        feature_transformer = build_custom_tabicl(tabicl_cfg, structure_encoder_dim=structure_encoder_dim)
    else:
        yaml_path = _get_limix_config_path(cfg)
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_feature_transformer = yaml.safe_load(f)
        feature_transformer = build_custom_model(config_feature_transformer)

    semantic_encoder = _build_semantic_encoder(cfg)
    semantic_dim = int(cfg.model.semantic_encoder.dim)
    inverse_relation_semantic_mode = _resolve_inverse_relation_semantic_mode(cfg)
    train_structure_encoder = bool(cfg.train.get("train_structure_encoder", True))
    enhance_structure = bool(cfg.model.get("enhance_structure", False))
    structure_score_enhance = bool(cfg.model.get("structure_score_enhance", False))

    entity_dim = int(cfg.model.entity_model.get("input_dim", 64))
    relation_dim = int(cfg.model.relation_model.get("input_dim", 64))
    seq_chunk_size = cfg.model.get("seq_chunk_size", None)
    context_label_correction = bool(cfg.task.get("context_label_correction", False))
    with_relation = bool(cfg.model.get("with_relation", True))
    context_graph = int(cfg.model.get("context_graph", 0))
    context_tail = bool(cfg.model.get("context_tail", False))
    model = KGPFN(
        structure_encoder=structure_encoder,
        semantic_encoder=semantic_encoder,
        feature_transformer=feature_transformer,
        entity_dim=entity_dim,
        relation_dim=relation_dim,
        semantic_dim=semantic_dim,
        inverse_relation_semantic_mode=inverse_relation_semantic_mode,
        enhance_structure=enhance_structure,
        structure_score_enhance=structure_score_enhance,
        seq_chunk_size=seq_chunk_size,
        context_label_correction=context_label_correction,
        with_relation=with_relation,
        context_graph=context_graph,
        context_tail=context_tail,
    )

    if init:
        print("Initializing model...")
        # 1) 初始化 structure encoder
        structure_encoder_path = cfg.train.get("structure_encoder_path", None)
        if model.structure_encoder is not None and structure_encoder_path and os.path.exists(structure_encoder_path):
            state = torch.load(structure_encoder_path, map_location=map_location)
            sd = state["model"] if isinstance(state, dict) and "model" in state else state
            model.structure_encoder.load_state_dict(sd, strict=False)

        # 2) 初始化 feature transformer
        if ft_type == "tabpfn":
            tabpfn_ckpt = cfg.train.get(
                "tabpfn_ckpt_path",
                "/home/gaoyisen/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
            )
            tabpfn_state = torch.load(tabpfn_ckpt, map_location=map_location, weights_only=False)
            tabpfn_sd = tabpfn_state.get("state_dict", tabpfn_state)
            tabpfn_sd = {k: v for k, v in tabpfn_sd.items() if "criterion." not in k}
            model.feature_transformer.load_state_dict(tabpfn_sd, strict=False)
        elif ft_type == "tabicl":
            tabicl_ckpt = cfg.train.get("tabicl_ckpt_path", None)
            if tabicl_ckpt and os.path.exists(tabicl_ckpt):
                tabicl_state = torch.load(tabicl_ckpt, map_location=map_location, weights_only=False)
                tabicl_sd = tabicl_state.get("state_dict", tabicl_state)
                # filter decoder: custom head has different shape from pretrained
                tabicl_sd = {k: v for k, v in tabicl_sd.items() if not k.startswith("icl_predictor.decoder")}
                model.feature_transformer.load_state_dict(tabicl_sd, strict=False)
                print(f"Loaded tabicl ckpt from {tabicl_ckpt}")
        else:
            limix_repo_id = cfg.train.get("limix_repo_id", "stableai-org/LimiX-16M")
            limix_filename = cfg.train.get("limix_filename", "LimiX-16M.ckpt")
            limix_cache_dir = cfg.train.get("limix_cache_dir", "/data/gaoyisen/LimiX/cache")
            model_file = hf_hub_download(
                repo_id=limix_repo_id,
                filename=limix_filename,
                local_dir=limix_cache_dir,
            )
            limix_state = torch.load(model_file, map_location=map_location, weights_only=False)
            limix_sd = limix_state.get("state_dict", limix_state)
            load_state_dict_matching(model.feature_transformer, limix_sd, strict_shape=True)
        # print("Initialized complete")
    # 3) 可选：加载整体 KGPFN checkpoint（最高优先级）
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=map_location)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd, strict=False)
        print("Loaded model from checkpoint")

    _set_module_trainable(model.semantic_encoder, False)
    _set_module_trainable(model.structure_encoder, False)

    print("Model created complete")
    return model


# here we assume that train_data and valid_data are tuples of datasets
def train_and_validate(cfg, model, train_data, valid_data, filtered_data=None, batch_per_epoch=None, accelerator=None):
    """
    accelerator: optional Accelerator instance.
      - If provided: uses accelerator.prepare() for DDP + mixed precision,
        accelerator.backward() for loss, and accelerator.unwrap_model() for saves.
      - If None: falls back to manual DDP (original behavior).
      In both cases, graph selection is synchronized via dist.broadcast so all
      ranks always process the same graph per step (fixes DDP timeout from load imbalance).
    """

    if cfg.train.num_epoch == 0:
        return

    if accelerator is not None:
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        is_main = accelerator.is_main_process
        _device = accelerator.device
    else:
        world_size = util.get_world_size()
        rank = util.get_rank()
        is_main = (rank == 0)
        _device = device

    # Fallback: roughly one pass through all training triplets
    if batch_per_epoch is None:
        total = sum(g.target_edge_index.shape[1] for g in train_data)
        batch_per_epoch = max(1, total // (cfg.train.batch_size * world_size))

    # Graph selection probabilities proportional to graph size (same as original collator)
    graph_probs = torch.tensor(
        [g.edge_index.shape[1] for g in train_data], dtype=torch.float, device=_device
    )
    graph_probs /= graph_probs.sum()
    # Shared buffer for broadcasting the selected graph index across ranks
    graph_id_buf = torch.zeros(1, dtype=torch.long, device=_device)

    cls = cfg.optimizer.pop("class")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(optim, cls)(trainable_params, **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if accelerator is not None:
        # accelerate handles DDP wrapping and device placement
        parallel_model, optimizer = accelerator.prepare(model, optimizer)
    elif world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)

    # checkpoint：每个 epoch 结束保存 model_epoch_*.pth；valid 上 eval 仅在有新高时保存 model_best.pth
    checkpoint_dir = getattr(cfg.train, "checkpoint_dir", ".")  # 相对 working_dir 或绝对路径
    max_checkpoints = int(cfg.train.get("max_checkpoints", 20))  # 最多保留的 epoch checkpoint 数量（不含 model_best）
    adversarial_temperature = float(cfg.task.get("adversarial_temperature", 0.0))
    loss_weights = list(cfg.task.get("loss_weights", [1.0, 0.0]))  # [lambda_bce, lambda_softmax]
    label_smoothing = float(cfg.task.get("label_smoothing", 0.0))
    # 每条三元组额外训练一条 (t, r^-1, ?) 反向 query，实际 batch 变为 2 倍
    train_inverse = bool(cfg.task.get("train_inverse", True))
    use_wandb = bool(cfg.train.get("use_wandb", False))
    valid_eval_step_interval = int(cfg.train.get("valid_eval_step_interval", 0))
    best_mrr = -1.0  # 用于保存 MRR 最优的 checkpoint

    if is_main and train_inverse:
        logger.warning(
            "Bidirectional training enabled: effective batch size %d (%d forward + %d inverse queries)",
            cfg.train.batch_size * 2, cfg.train.batch_size, cfg.train.batch_size,
        )

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if is_main:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            bce_losses = []
            softmax_losses = []
            for _ in range(batch_per_epoch):
                # Rank 0 samples the graph; broadcast ensures all ranks use the same graph,
                # eliminating the compute-time divergence that causes DDP timeout.
                # This is necessary even with accelerate — no_sync/accumulate does NOT
                # fix load imbalance between GPUs processing different-sized graphs.
                if rank == 0:
                    graph_id_buf[0] = torch.multinomial(graph_probs, 1).item()
                if world_size > 1:
                    dist.broadcast(graph_id_buf, src=0)
                graph_id = int(graph_id_buf.item())
                train_graph = train_data[graph_id]

                bs = cfg.train.batch_size
                perm = torch.randperm(train_graph.target_edge_index.shape[1], device=_device)[:bs]
                batch = torch.cat([
                    train_graph.target_edge_index[:, perm],
                    train_graph.target_edge_type[perm].unsqueeze(0),
                ]).t()

                # 0) 每条三元组配一条反向 query，让 head / tail 两侧共享同一次更新
                if train_inverse:
                    batch = tasks.augment_with_inverse_queries(train_graph, batch)

                # 1) 负采样得到 (B, N, 3) 的 query 三元组
                batch_with_neg = tasks.negative_sampling_tail(
                    train_graph,
                    batch,
                    cfg.task.num_negative,
                    strict=cfg.task.strict_negative,
                )  # (B, N, 3)
               
                # 2) 为每条 query 构造上下文三元组
                context_triples, context_labels = tasks.build_context_relation_aware(
                    train_graph,
                    batch_with_neg,
                    num_pos=cfg.task.num_pos,
                    num_neg=cfg.task.num_neg,
                )

                # 3) 组织成 KGPFN / PFN 所需的输入格式（可选 text）
                query_ids = batch_with_neg.to(device)  # (B, S_q, 3)
                context_ids = [t.to(device) for t in context_triples]
                context_y = [t.to(device) for t in context_labels]
                use_text_input = _semantic_enabled(parallel_model)
                query_x, context_x = _build_model_inputs(query_ids, context_ids, train_graph, enable_text=use_text_input)

                # 4) 前向：这里用回归头（或按需改成 "cls"）
                pred = parallel_model(
                    train_graph,
                    query_x=query_x,
                    context_x=context_x,
                    context_y=context_y,
                    task_type="reg",
                )

                # 5) 目标：正样本在 col 0，负样本在 col 1:；loss 加权方式与 pretrain.py 一致
                loss = torch.tensor(0.0, device=pred.device)
                bce_loss = torch.tensor(0.0, device=pred.device)
                softmax_loss = torch.tensor(0.0, device=pred.device)
                if loss_weights[0] > 0:
                    target = torch.zeros_like(pred)
                    if target.numel() > 0:
                        target[:, 0] = 1.0
                    loss_raw = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                    neg_weight = torch.ones_like(pred)
                    if adversarial_temperature > 0:
                        with torch.no_grad():
                            neg_weight[:, 1:] = F.softmax(pred[:, 1:] / adversarial_temperature, dim=-1)
                    else:
                        neg_weight[:, 1:] = 1 / cfg.task.num_negative
                    bce_loss = ((loss_raw * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)).mean()
                    loss = loss + loss_weights[0] * bce_loss
                if loss_weights[1] > 0:
                    sm_target = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
                    softmax_loss = F.cross_entropy(pred.float(), sm_target, label_smoothing=label_smoothing)
                    loss = loss + loss_weights[1] * softmax_loss

                # NaN/Inf check before backward
                if not pred.isfinite().all():
                    raise RuntimeError(
                        f"Non-finite pred detected at epoch {epoch}, batch {batch_id}: "
                        f"nan={pred.isnan().sum().item()}, inf={pred.isinf().sum().item()}, "
                        f"pred stats: min={pred.min().item()}, max={pred.max().item()}"
                    )
                if not loss.isfinite():
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch {epoch}, batch {batch_id}: "
                        f"loss={loss.item()}, bce_loss={bce_loss.item()}, softmax_loss={softmax_loss.item()}"
                    )
                if not loss.requires_grad:
                    raise RuntimeError(
                        f"loss has no grad_fn at epoch {epoch}, batch {batch_id}: "
                        f"loss_weights={loss_weights}, loss={loss.item()}"
                    )

                # accelerate handles gradient scaling for mixed precision automatically
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("total loss: %g, bce_loss: %g, softmax_loss: %g" % (
                        loss.item(), bce_loss.item(), softmax_loss.item()))
                    if use_wandb and is_main and wandb is not None:
                        wandb.log(
                            {
                                "train/loss_step": loss.item(),
                                "train/epoch": epoch,
                                "train/step": batch_id,
                            },
                            step=batch_id,
                        )
                losses.append(loss.item())
                bce_losses.append(bce_loss.item())
                softmax_losses.append(softmax_loss.item())
                batch_id += 1

                # 可选：每隔若干个 step 在 valid 上评估一次；仅当 valid MRR 创新高时保存 model_best.pth
                if valid_eval_step_interval > 0 and (batch_id % valid_eval_step_interval == 0):
                    if is_main:
                        logger.warning(separator)
                        logger.warning("Evaluate on valid at step %d", batch_id)
                    _eval_model = accelerator.unwrap_model(parallel_model) if accelerator is not None else model
                    valid_mrr = test(cfg, _eval_model, valid_data, filtered_data=filtered_data, split="valid")
                    parallel_model.train()
                    if is_main:
                        logger.warning("valid mrr: %g", valid_mrr)
                        if use_wandb and wandb is not None:
                            wandb.log(
                                {
                                    "valid/mrr": float(valid_mrr),
                                    "train/epoch": epoch,
                                },
                                step=batch_id,
                            )
                        valid_mrr_f = float(valid_mrr)
                        if valid_mrr_f > best_mrr:
                            best_mrr = valid_mrr_f
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            best_path = os.path.join(checkpoint_dir, "model_best.pth")
                            _save_model = accelerator.unwrap_model(parallel_model) if accelerator is not None else model
                            state = {
                                "model": _save_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "step": batch_id,
                                "epoch": epoch,
                                "valid_mrr": valid_mrr_f,
                            }
                            best_state = {**state, "best_mrr": valid_mrr_f}
                            torch.save(best_state, best_path)
                            logger.warning(f"New best MRR {best_mrr:.4f}, save to {best_path}")


            avg_loss = sum(losses) / len(losses)
            avg_bce_loss = sum(bce_losses) / len(bce_losses)
            avg_softmax_loss = sum(softmax_losses) / len(softmax_losses)
            logger.warning(separator)
            logger.warning("Epoch %d end" % epoch)
            logger.warning(line)
            logger.warning("average loss: %g, avg_bce_loss: %g, avg_softmax_loss: %g" % (
                avg_loss, avg_bce_loss, avg_softmax_loss))
            if use_wandb and is_main and wandb is not None:
                wandb.log(
                    {
                        "train/loss_epoch": avg_loss,
                        "train/bce_loss_epoch": avg_bce_loss,
                        "train/softmax_loss_epoch": avg_softmax_loss,
                        "train/metric": avg_loss,
                        "train/epoch": epoch,
                    },
                    step=batch_id,
                )

            if is_main:
                os.makedirs(checkpoint_dir, exist_ok=True)
                epoch_ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                _save_model = accelerator.unwrap_model(parallel_model) if accelerator is not None else model
                epoch_state = {
                    "model": _save_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": batch_id,
                    "epoch": epoch,
                    "best_mrr_so_far": best_mrr,
                }
                torch.save(epoch_state, epoch_ckpt_path)
                logger.warning(f"Save epoch checkpoint to {epoch_ckpt_path}")
                epoch_ckpts = sorted(
                    glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth")),
                    key=lambda p: int(os.path.basename(p).split("model_epoch_")[-1].replace(".pth", "")),
                )
                for old_ckpt in epoch_ckpts[:-max_checkpoints]:
                    try:
                        os.remove(old_ckpt)
                        logger.warning(f"Remove old epoch checkpoint: {old_ckpt}")
                    except OSError:
                        pass

        util.synchronize()


@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None, split: str = "valid"):
    world_size = util.get_world_size()
    rank = util.get_rank()
    eval_chunk_size = int(cfg.train.get("eval_chunk_size", 64))
    eval_log_interval = int(cfg.train.get("eval_log_interval", 50))
    score_threshold = float(cfg.train.get("eval_score_threshold", 0.5))
    loss_weights = list(cfg.task.get("loss_weights", [1.0, 0.0]))
    adversarial_temperature = float(cfg.task.get("adversarial_temperature", 0.0))
    label_smoothing = float(cfg.task.get("label_smoothing", 0.0))

    def _mrr(ranking):
        return float((1.0 / ranking.float()).mean().item()) if ranking.numel() else float("nan")

    def _hits(ranking, k):
        return float((ranking <= k).float().mean().item()) if ranking.numel() else float("nan")

    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics: list[float] = []
    collected_metric_values: dict[str, list[float]] = {}
    dataset_records = [
        _empty_dataset_record(getattr(g, "dataset", f"graph_{i}")) for i, g in enumerate(test_data)
    ]
    csv_path = _get_dataset_csv_path(cfg, split)
    if rank == 0:
        # 先按当前评测数据集创建 CSV，后续每个数据集完成后实时覆盖更新
        _write_dataset_csv(csv_path, dataset_records)
        logger.warning("Per-dataset csv initialized at %s", csv_path)
    if filtered_data is None:
        filtered_data = [None] * len(test_data)
    default_num_neg = int(cfg.task.num_neg)
    for graph_idx, (test_graph, filters) in enumerate(zip(test_data, filtered_data)):
        graph_name = getattr(test_graph, "dataset", f"graph_{graph_idx}")
        is_nell_inductive_v1 = graph_name == "NELLInductive-v1"
        cfg.task.num_neg = 40 if is_nell_inductive_v1 else default_num_neg
        if rank == 0:
            logger.warning(separator)
            logger.warning("[%s/%s] Evaluating: %s (%d triples, %d nodes)",
                           split, graph_idx, graph_name,
                           test_graph.target_edge_index.shape[1],
                           test_graph.num_nodes)
            if is_nell_inductive_v1:
                logger.warning("[%s] Override context num_neg to %d for this dataset.", graph_name, int(cfg.task.num_neg))

        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        model.eval()
        tail_rankings, head_rankings = [], []
        tail_num_negatives, head_num_negatives = [], []
        # classification-style metrics（基于 logit 阈值，both 汇总）
        tp_total = torch.zeros(1, dtype=torch.float32, device=device)
        fp_total = torch.zeros(1, dtype=torch.float32, device=device)
        fn_total = torch.zeros(1, dtype=torch.float32, device=device)
        eval_losses, eval_bce_losses, eval_softmax_losses = [], [], []
        for batch in test_loader:
            # 原始 query (h, r, ?) 预测 tail；反向 query (t, r^-1, ?) 预测 head。
            # 两组都枚举全部候选 tail；tail-only 数据集没有反向 query。
            reverse_batch = tasks.inverse_relation_queries(test_graph, batch)
            eval_batch = torch.cat([batch, reverse_batch], dim=0)
            n_tail = int(batch.size(0))

            t_mask, _ = tasks.strict_negative_mask(test_graph if filters is None else filters, eval_batch)
            pos_t_index = eval_batch[:, 1]

            # 1) 对每个 batch 行只构建一次上下文，在全部候选 tail 间复用
            # build_context_relation_aware 期望 [B, N, 3]，这里用 N=1 的锚点 query
            try:
                shared_context_x, shared_context_y = tasks.build_context_relation_aware(
                    test_graph,
                    eval_batch.unsqueeze(1),
                    num_pos=cfg.task.num_pos,
                    num_neg=cfg.task.num_neg,
                )  # 长度为 B
            except RuntimeError as e:
                logger.warning(f"Skipping batch due to insufficient negatives: {e}")
                continue
            shared_context_x = [t.to(device) for t in shared_context_x]
            shared_context_y = [y.to(device) for y in shared_context_y]

            # 2) 所有候选 tail 的分数 (B, num_nodes)
            t_pred = _score_all_candidate_tails(
                model, test_graph, eval_batch, shared_context_x, shared_context_y, eval_chunk_size,
            )

            # compute loss on full candidate set
            eval_loss = torch.tensor(0.0, device=t_pred.device)
            eval_bce_loss = torch.tensor(0.0, device=t_pred.device)
            eval_softmax_loss = torch.tensor(0.0, device=t_pred.device)
            if loss_weights[0] > 0:
                target = torch.zeros_like(t_pred)
                target.scatter_(1, pos_t_index.unsqueeze(-1), 1.0)
                loss_raw = F.binary_cross_entropy_with_logits(t_pred, target, reduction="none")
                neg_weight = t_mask.float()
                if adversarial_temperature > 0:
                    with torch.no_grad():
                        adv_w = F.softmax(t_pred.masked_fill(~t_mask, float('-inf')) / adversarial_temperature, dim=-1)
                    neg_weight = adv_w * t_mask.float()
                else:
                    num_neg = t_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                    neg_weight = t_mask.float() / num_neg
                # positive weight = 1
                weight = neg_weight.clone()
                weight.scatter_(1, pos_t_index.unsqueeze(-1), 1.0)
                eval_bce_loss = ((loss_raw * weight).sum(dim=-1) / weight.sum(dim=-1)).mean()
                eval_loss = eval_loss + loss_weights[0] * eval_bce_loss
            if loss_weights[1] > 0:
                eval_softmax_loss = F.cross_entropy(t_pred.float(), pos_t_index.to(t_pred.device), label_smoothing=label_smoothing)
                eval_loss = eval_loss + loss_weights[1] * eval_softmax_loss
            eval_losses.append(eval_loss.item())
            eval_bce_losses.append(eval_bce_loss.item())
            eval_softmax_losses.append(eval_softmax_loss.item())

            # 3) ranking：前 n_tail 行是 tail，后面是反向 relation 的 head
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            num_t_negative = t_mask.sum(dim=-1)
            tail_rankings.append(t_ranking[:n_tail])
            tail_num_negatives.append(num_t_negative[:n_tail])
            head_rankings.append(t_ranking[n_tail:])
            head_num_negatives.append(num_t_negative[n_tail:])

            # 4) 计算 precision/recall/f1（只在 filtered 可比较集合内）
            valid_mask = t_mask.clone()
            valid_mask.scatter_(1, pos_t_index.unsqueeze(-1), True)  # 把正例位加入评估
            pred_pos = t_pred > score_threshold
            gt_pos = torch.zeros_like(pred_pos, dtype=torch.bool)
            gt_pos.scatter_(1, pos_t_index.unsqueeze(-1), True)
            pred_pos = pred_pos & valid_mask
            gt_pos = gt_pos & valid_mask

            tp = (pred_pos & gt_pos).sum().float()
            fp = (pred_pos & ~gt_pos).sum().float()
            fn = ((~pred_pos) & gt_pos).sum().float()
            tp_total += tp
            fp_total += fp
            fn_total += fn

            # 5) 周期性进度日志（防止验证过慢无反馈）
            if rank == 0 and (len(tail_rankings) % max(1, eval_log_interval) == 0):
                cur_tail = torch.cat(tail_rankings)
                cur_head = torch.cat(head_rankings)
                cur_both = torch.cat([cur_tail, cur_head])
                cur_precision = tp_total / (tp_total + fp_total + 1e-12)
                cur_recall = tp_total / (tp_total + fn_total + 1e-12)
                cur_f1 = 2 * cur_precision * cur_recall / (cur_precision + cur_recall + 1e-12)
                logger.warning(
                    "[eval-progress] steps=%d both_mrr=%.6f tail_mrr=%.6f head_mrr=%.6f "
                    "hits@1=%.6f hits@3=%.6f hits@10=%.6f precision=%.6f recall=%.6f f1=%.6f",
                    len(tail_rankings),
                    _mrr(cur_both), _mrr(cur_tail), _mrr(cur_head),
                    _hits(cur_both, 1), _hits(cur_both, 3), _hits(cur_both, 10),
                    cur_precision.item(),
                    cur_recall.item(),
                    cur_f1.item(),
                )

        all_tail_ranking = _gather_long_vector(_cat_or_empty(tail_rankings, device), world_size, rank, device)
        all_head_ranking = _gather_long_vector(_cat_or_empty(head_rankings, device), world_size, rank, device)
        all_tail_num_negative = _gather_long_vector(_cat_or_empty(tail_num_negatives, device), world_size, rank, device)
        all_head_num_negative = _gather_long_vector(_cat_or_empty(head_num_negatives, device), world_size, rank, device)
        all_ranking = torch.cat([all_tail_ranking, all_head_ranking])
        all_num_negative = torch.cat([all_tail_num_negative, all_head_num_negative])
        if all_ranking.numel() == 0:
            if rank == 0:
                logger.warning("[%s] No valid ranking generated; skip this dataset.", graph_name)
            continue
        if world_size > 1:
            dist.all_reduce(tp_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn_total, op=dist.ReduceOp.SUM)

        side_rankings = {
            "head": (all_head_ranking, all_head_num_negative),
            "tail": (all_tail_ranking, all_tail_num_negative),
            "both": (all_ranking, all_num_negative),
        }
        side_metrics = {
            side: _ranking_metrics(ranking, num_neg, list(cfg.task.metric))
            for side, (ranking, num_neg) in side_rankings.items()
        }

        if rank == 0:
            precision = tp_total / (tp_total + fp_total + 1e-12)
            recall = tp_total / (tp_total + fn_total + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            side_metrics["both"].update({
                "precision": float(precision.item()),
                "recall": float(recall.item()),
                "f1": float(f1.item()),
            })
            for side in EVAL_SIDES:
                ranking, _ = side_rankings[side]
                if ranking.numel() == 0:
                    logger.warning("[%s][%s] skipped (no queries on this side)", graph_name, side)
                    continue
                metrics = side_metrics[side]
                for metric in cfg.task.metric:
                    if metric in metrics:
                        logger.warning("[%s][%s] %s: %g", graph_name, side, metric, metrics[metric])
            logger.warning("[%s] loss: %g, bce_loss: %g, softmax_loss: %g",
                           graph_name, _nanmean(eval_losses), _nanmean(eval_bce_losses), _nanmean(eval_softmax_losses))

        # 模型选择只看 both 侧的 mrr
        mrr_value = side_metrics["both"]["mrr"]
        if not math.isnan(mrr_value):
            all_metrics.append(mrr_value)
        if rank == 0:
            logger.warning("[%s] mrr(both): %g | mrr(tail): %g | mrr(head): %g",
                           graph_name, mrr_value,
                           side_metrics["tail"]["mrr"], side_metrics["head"]["mrr"])
            dataset_records[graph_idx] = {
                "name": graph_name,
                "family": _match_dataset_family(graph_name),
                **_side_record(side_metrics),
            }
            _write_dataset_csv(csv_path, dataset_records)
            for side, metrics in side_metrics.items():
                for k, v in metrics.items():
                    collected_metric_values.setdefault(f"{k}_{side}", []).append(v)

    if not all_metrics:
        if rank == 0:
            logger.warning("[%s] No dataset produced valid rankings; return NaN metric.", split)
        avg_metric = torch.tensor(float("nan"), device=device)
    else:
        avg_metric = torch.tensor(_nanmean(all_metrics), device=device)
    if rank == 0:
        # 跨数据集平均：三侧分开汇报，返回值取 both
        for side in EVAL_SIDES:
            logger.warning(
                "[%s][%s] average mrr: %g, average hits@10: %g",
                split, side,
                _nanmean([r[side]["mrr"] for r in dataset_records]),
                _nanmean([r[side]["hits@10"] for r in dataset_records]),
            )
        _write_dataset_csv(csv_path, dataset_records)
        logger.warning("Per-dataset csv saved to %s", csv_path)
    if rank == 0 and wandb is not None and wandb.run is not None and collected_metric_values:
        # 记录每次完整评测后的平均指标到 summary
        for metric_name, values in collected_metric_values.items():
            wandb.run.summary[f"{split}/{metric_name}_avg"] = float(_nanmean(values))
        wandb.run.summary[f"{split}/mrr_return"] = float(avg_metric.item())
    return avg_metric

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    util.apply_model_config(cfg)
    # 在 chdir 前将相对路径解析为相对于项目根目录的绝对路径
    project_root = _get_project_root()
    for key in ("structure_encoder_path", "limix_cache_dir", "limix_config_path", "kgpfn_checkpoint"):
        val = cfg.train.get(key)
        if val:
            cfg.train[key] = _resolve_path(val, project_root)

    # ── Accelerate setup ────────────────────────────────────────────────────────
    # Enable via config: train.use_accelerate: true  (+ optional mixed_precision: "bf16")
    # Launch with: accelerate launch --num_processes N script/pretrain_pfn.py -c ...
    # or the existing: torchrun --nproc_per_node=N  (both work; accelerate detects either)
    #
    # NOTE: Accelerator() must be created BEFORE create_working_directory() so that
    # torch.distributed is initialized by accelerate first, and create_working_directory's
    # `if not dist.is_initialized()` guard correctly skips redundant init.
    _use_accelerate = bool(cfg.train.get("use_accelerate", False))
    if _use_accelerate:
        assert _ACCELERATE_AVAILABLE, "use_accelerate=true but `accelerate` is not installed. Run: pip install accelerate"
        _mixed_precision = cfg.train.get("mixed_precision", "no")
        # find_unused_parameters=True is required because structure_encoder and
        # semantic_encoder are frozen (requires_grad=False) and produce no gradients.
        _ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(mixed_precision=_mixed_precision, kwargs_handlers=[_ddp_kwargs])
    else:
        accelerator = None
    # ────────────────────────────────────────────────────────────────────────────

    working_dir = util.create_working_directory(cfg, chdir=False)

    if util.get_rank() == 0:
        _save_configs(cfg, args.config, working_dir)

    torch.manual_seed(args.seed + util.get_rank())

    # 日志：输出到控制台 + .log 文件（默认 pretrain_pfn.log，写入 working_dir）
    # 必须用 abspath，否则 chdir(working_dir) 后相对路径会重复解析
    logger = util.get_root_logger(file=False)
    if util.get_rank() == 0:
        log_file = getattr(cfg.train, "log_file", "pretrain_pfn.log")
        if not os.path.isabs(log_file):
            log_file = os.path.join(working_dir, log_file)
        # 使用 mode="w" 每次运行覆盖旧日志
        handler = logging.FileHandler(log_file, mode="w")
        fmt = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Context sampling args: %s", _get_context_sampling_args(cfg))

    use_wandb = False
    if util.get_rank() == 0:
        use_wandb, _ = wandb_init(cfg, logger)
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = accelerator.device if accelerator is not None else util.get_device(cfg)
    
    # Swap official valid and test. `valid_data` is the official test split and is what the
    # final eval scores. `test_data` is official valid and is what checkpointing uses.
    # fast_test only subsamples that official test eval.
    train_data, test_data, valid_data = dataset._data[0], dataset._data[1], dataset._data[2]
    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]
    if "fast_test" in cfg.train:
        eval_test = _subsample_targets(valid_data, cfg.train.fast_test)
        if util.get_rank() == 0:
            logger.warning("Fast test: %d target triples per graph of the official test split", int(cfg.train.fast_test))
    else:
        eval_test = valid_data

    model = create_model(
        cfg,
        init=bool(cfg.train.get("init_model", True)),
        ckpt_path=cfg.train.get("kgpfn_checkpoint", None),
        map_location="cpu",
    )

    model = model.to(device)
  
    assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"

    # Build per-split filtered data using each target graph's own edge space and node count.
    #
    # Using each graph's edge_index (the context graph) + target_edge_index (prediction targets)
    # correctly handles both transductive and inductive settings:
    #   - Transductive: edge_index already contains all edges; num_nodes is shared.
    #   - Inductive: test entities differ from train entities; using test_graph.num_nodes
    #     fixes the shape mismatch (test_graph.num_nodes != train_graph.num_nodes) that
    #     caused compute_ranking to crash with mismatched tensor dimensions.
    # Filtered ranking excludes every known fact: the eval graph's observed edges plus the
    # train / valid / test targets (and their inverses).
    def _make_filtered_data(eval_graphs):
        return [
            tasks.make_filtered_ranking_graph(
                g, train_data[i], valid_data[i], test_data[i]
            ).to(device)
            for i, g in enumerate(eval_graphs)
        ]

    valid_filtered_data = _make_filtered_data(valid_data)
    test_filtered_data = _make_filtered_data(test_data)

    # checkpoint_dir 相对 working_dir 解析（因未 chdir，需显式拼接）
    ckpt_dir = getattr(cfg.train, "checkpoint_dir", ".")
    if not os.path.isabs(ckpt_dir):
        cfg.train.checkpoint_dir = os.path.join(working_dir, ckpt_dir)

    train_and_validate(cfg, model, train_data, test_data, filtered_data=test_filtered_data, batch_per_epoch=cfg.train.batch_per_epoch, accelerator=accelerator)
    

    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on valid")
    # test(cfg, model, valid_data, filtered_data=filtered_data)
    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on test")

    # test(cfg, model, test_data, filtered_data=test_filtered_data, split="test")
    test(cfg, model, eval_test, filtered_data=valid_filtered_data, split="test")
    if util.get_rank() == 0 and use_wandb and wandb is not None:
        wandb.finish()
