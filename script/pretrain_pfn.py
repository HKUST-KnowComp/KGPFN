import glob
import os
import sys
import copy
import math
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


def _build_semantic_encoder(cfg):
    sem_name = str(cfg.model.semantic_encoder.model_name).strip()
    if sem_name.lower() in ("", "none", "null"):
        return None
    from sentence_transformers import SentenceTransformer
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(sem_name)


def _build_structure_encoder(cfg):
    name = str(cfg.model.get("structure_encoder_name", "ultra")).strip()
    if name.lower() in ("", "none", "null"):
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


def _get_context_sampling_args(cfg):
    # relation-aware context builder currently only needs pos/neg counts.
    return {
        "num_pos": int(cfg.task.num_pos),
        "num_neg": int(cfg.task.num_neg),
    }


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
    use_wandb = bool(cfg.train.get("use_wandb", False))
    valid_eval_step_interval = int(cfg.train.get("valid_eval_step_interval", 0))
    best_mrr = -1.0  # 用于保存 MRR 最优的 checkpoint

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
                    softmax_loss = F.cross_entropy(pred, sm_target, label_smoothing=label_smoothing)
                    loss = loss + loss_weights[1] * softmax_loss

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
    
    # test_data is a tuple of validation/test datasets
    # process sequentially
    all_metrics = []
    collected_metric_values: dict[str, list[float]] = {}
    for graph_idx, (test_graph, filters) in enumerate(zip(test_data, filtered_data)):
        graph_name = getattr(test_graph, "dataset", f"graph_{graph_idx}")
        if rank == 0:
            logger.warning(separator)
            logger.warning("[%s/%s] Evaluating: %s (%d triples, %d nodes)",
                           split, graph_idx, graph_name,
                           test_graph.target_edge_index.shape[1],
                           test_graph.num_nodes)

        test_triplets = torch.cat([test_graph.target_edge_index, test_graph.target_edge_type.unsqueeze(0)]).t()
        sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
        test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

        model.eval()
        rankings = []
        num_negatives = []
        # classification-style metrics（基于 logit 阈值）
        tp_total = torch.zeros(1, dtype=torch.float32, device=device)
        fp_total = torch.zeros(1, dtype=torch.float32, device=device)
        fn_total = torch.zeros(1, dtype=torch.float32, device=device)
        eval_losses, eval_bce_losses, eval_softmax_losses = [], [], []
        for batch in test_loader:
            # 1) 严格负采样评测：tail 全候选
            t_batch, _ = tasks.all_negative(test_graph, batch)  # (B, num_nodes, 3)
            B, num_nodes, _ = t_batch.shape

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_mask(test_graph, batch)
            else:
                t_mask, h_mask = tasks.strict_negative_mask(filters, batch)
            pos_h_index, pos_t_index, pos_r_index = batch.t()

            # 2) 对每个 batch 行只构建一次上下文，在多个 chunk 间复用
            # build_context_for_batch 期望 [B, N, 3]，这里用 N=1 的锚点 query
            row_anchor_batch = batch.unsqueeze(1)
            shared_context_x, shared_context_y = tasks.build_context_relation_aware(
                test_graph,
                row_anchor_batch,
                num_pos=cfg.task.num_pos,
                num_neg=cfg.task.num_neg,
            )  # 长度为 B
            shared_context_x = [t.to(device) for t in shared_context_x]
            shared_context_y = [y.to(device) for y in shared_context_y]

            # 3) 预计算上下文 embedding cache，同时获取（可能修正的）标签
            context_cache, shared_context_y = model.get_context_embeddings_cache(
                test_graph, shared_context_x, shared_context_y,
            )

            # 4) 分 chunk 评测所有 tail 候选
            t_score_chunks = []
            for start in range(0, num_nodes, max(1, eval_chunk_size)):
                end = min(start + max(1, eval_chunk_size), num_nodes)
                t_batch_chunk = t_batch[:, start:end, :]  # (B, chunk, 3)
                t_scores_chunk = model.get_scores(
                    test_graph,
                    query_x=t_batch_chunk.to(device),
                    context_cache=context_cache,
                    context_y=shared_context_y,
                    task_type="reg",
                )
                t_score_chunks.append(t_scores_chunk.view(B, -1))
            t_pred = torch.cat(t_score_chunks, dim=1)  # (B, num_nodes)

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
                eval_softmax_loss = F.cross_entropy(t_pred, pos_t_index.to(t_pred.device), label_smoothing=label_smoothing)
                eval_loss = eval_loss + loss_weights[1] * eval_softmax_loss
            eval_losses.append(eval_loss.item())
            eval_bce_losses.append(eval_bce_loss.item())
            eval_softmax_losses.append(eval_softmax_loss.item())

            # 4) ranking（这里按 tail-only 汇总，契合当前 (h,r,?) 设定）
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            num_t_negative = t_mask.sum(dim=-1)

            rankings += [t_ranking]
            num_negatives += [num_t_negative]

            # 5) 计算 precision/recall/f1（只在 filtered 可比较集合内）
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

            # 6) 周期性进度日志（防止验证过慢无反馈）
            if rank == 0 and (len(rankings) % max(1, eval_log_interval) == 0):
                cur_ranking = torch.cat(rankings)
                cur_mrr = (1.0 / cur_ranking.float()).mean()
                cur_hits1 = (cur_ranking <= 1).float().mean()
                cur_hits3 = (cur_ranking <= 3).float().mean()
                cur_hits10 = (cur_ranking <= 10).float().mean()
                cur_hits30 = (cur_ranking <= 30).float().mean()
                cur_precision = tp_total / (tp_total + fp_total + 1e-12)
                cur_recall = tp_total / (tp_total + fn_total + 1e-12)
                cur_f1 = 2 * cur_precision * cur_recall / (cur_precision + cur_recall + 1e-12)
                logger.warning(
                    "[eval-progress] steps=%d mrr=%.6f hits@1=%.6f hits@3=%.6f hits@10=%.6f hits@30=%.6f precision=%.6f recall=%.6f f1=%.6f",
                    len(rankings),
                    cur_mrr.item(),
                    cur_hits1.item(),
                    cur_hits3.item(),
                    cur_hits10.item(),
                    cur_hits30.item(),
                    cur_precision.item(),
                    cur_recall.item(),
                    cur_f1.item(),
                )

        ranking = torch.cat(rankings)
        num_negative = torch.cat(num_negatives)
        all_size = torch.zeros(world_size, dtype=torch.long, device=device)
        all_size[rank] = len(ranking)
        if world_size > 1:
            dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        cum_size = all_size.cumsum(0)
        all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
        all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
        all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
        if world_size > 1:
            dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
            dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
            dist.all_reduce(tp_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn_total, op=dist.ReduceOp.SUM)

        graph_metrics: dict[str, float] = {}
        if rank == 0:
            precision = tp_total / (tp_total + fp_total + 1e-12)
            recall = tp_total / (tp_total + fn_total + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            for metric in cfg.task.metric:
                if metric == "mr":
                    score = all_ranking.float().mean()
                elif metric == "mrr":
                    score = (1 / all_ranking.float()).mean()
                elif metric == "precision":
                    score = precision
                elif metric == "recall":
                    score = recall
                elif metric == "f1":
                    score = f1
                elif metric.startswith("hits@"):
                    values = metric[5:].split("_")
                    threshold = int(values[0])
                    if len(values) > 1:
                        num_sample = int(values[1])
                        # unbiased estimation
                        fp_rate = (all_ranking - 1).float() / all_num_negative
                        score = 0
                        for i in range(threshold):
                            # choose i false positive from num_sample - 1 negatives
                            num_comb = math.factorial(num_sample - 1) / \
                                    math.factorial(i) / math.factorial(num_sample - i - 1)
                            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                        score = score.mean()
                    else:
                        score = (all_ranking <= threshold).float().mean()
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                logger.warning("[%s] %s: %g", graph_name, metric, score)
                graph_metrics[metric] = float(score.item())
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            avg_eval_bce = sum(eval_bce_losses) / len(eval_bce_losses)
            avg_eval_softmax = sum(eval_softmax_losses) / len(eval_softmax_losses)
            logger.warning("[%s] loss: %g, bce_loss: %g, softmax_loss: %g",
                           graph_name, avg_eval_loss, avg_eval_bce, avg_eval_softmax)
        mrr = (1 / all_ranking.float()).mean()

        all_metrics.append(mrr)
        if rank == 0:
            graph_metrics["mrr"] = float(mrr.item())
            logger.warning("[%s] mrr: %g", graph_name, graph_metrics["mrr"])
            for k, v in graph_metrics.items():
                collected_metric_values.setdefault(k, []).append(v)

    avg_metric = sum(all_metrics) / len(all_metrics)
    if rank == 0 and wandb is not None and wandb.run is not None and collected_metric_values:
        # 记录每次完整评测后的平均指标到 summary
        for metric_name, values in collected_metric_values.items():
            wandb.run.summary[f"{split}/{metric_name}_avg"] = float(sum(values) / len(values))
        wandb.run.summary[f"{split}/mrr_return"] = float(avg_metric.item())
    return avg_metric


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
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
    
    train_data, valid_data, test_data = dataset._data[0], dataset._data[1], dataset._data[2]
    
    if "fast_test" in cfg.train:
        num_val_edges = cfg.train.fast_test
        if util.get_rank() == 0:
            logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
        short_valid = [copy.deepcopy(vd) for vd in test_data]
        for graph in short_valid:
            mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
            graph.target_edge_index = graph.target_edge_index[:, mask]
            graph.target_edge_type = graph.target_edge_type[mask]
        
        short_valid = [sv.to(device) for sv in short_valid]

    train_data = [td.to(device) for td in train_data]
    valid_data = [vd.to(device) for vd in valid_data]
    test_data = [tst.to(device) for tst in test_data]

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
    def _make_filtered_data(graphs):
        return [
            Data(
                edge_index=torch.cat([g.edge_index, g.target_edge_index], dim=1),
                edge_type=torch.cat([g.edge_type, g.target_edge_type]),
                num_nodes=g.num_nodes,
            ).to(device)
            for g in graphs
        ]

    valid_filtered_data = _make_filtered_data(valid_data)
    test_filtered_data = _make_filtered_data(test_data)

    # checkpoint_dir 相对 working_dir 解析（因未 chdir，需显式拼接）
    ckpt_dir = getattr(cfg.train, "checkpoint_dir", ".")
    if not os.path.isabs(ckpt_dir):
        cfg.train.checkpoint_dir = os.path.join(working_dir, ckpt_dir)

    train_and_validate(cfg, model, train_data, valid_data if "fast_test" not in cfg.train else short_valid, filtered_data=test_filtered_data, batch_per_epoch=cfg.train.batch_per_epoch, accelerator=accelerator)
    

    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on valid")
    # test(cfg, model, valid_data, filtered_data=filtered_data)
    # if util.get_rank() == 0:
    #     logger.warning(separator)
    #     logger.warning("Evaluate on test")

    # test(cfg, model, test_data, filtered_data=test_filtered_data, split="test")
    test(cfg, model, short_valid, filtered_data=test_filtered_data, split="test")
    if util.get_rank() == 0 and use_wandb and wandb is not None:
        wandb.finish()