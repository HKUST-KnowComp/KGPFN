import torch
import random
import numpy as np

from model.limix.model.custom_transformer import CustomFeaturesTransformer

def build_custom_model(config:dict):
    model = CustomFeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        feature_positional_embedding_type=config.get('feature_positional_embedding_type', "subortho"),
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=config.get('mask_prediction', False),
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        pre_norm=config.get('pre_norm', True),
        activation=config.get('activation', 'gelu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=config.get('device', None),
        dtype=config.get('dtype', None),
        recompute_attn=config['recompute_attn'],
        layer_arch=config.get('layer_arch', 'fmfmsm'),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
        structure_encoder_dim=config.get('structure_encoder_dim', 64),
        num_thinking_rows=config.get('num_thinking_rows', 20),
    )
    return model



def load_state_dict_matching(model, state_dict, strict_shape: bool = True):
    """
    只加载 checkpoint 里与当前 model 的 key 一致且（可选）shape 一致的部分，其余不加载。
    适用于你改了一部分结构、希望只恢复未改动的权重。

    Args:
        model: 当前模型（可来自 build_model 或自定义结构）
        state_dict: 从 checkpoint 里取出的 state_dict（例如 state_dict['state_dict']）
        strict_shape: True 时只加载 key 存在且 shape 完全一致的参数；False 时只要求 key 存在

    Returns:
        loaded_keys: 成功加载的 key 列表
        skipped_keys: 被跳过的 key 列表（不在 model 中或 shape 不一致）
    """
    model_sd = model.state_dict()
    to_load = {}
    skipped = []
    for k, v in state_dict.items():
        if k not in model_sd:
            skipped.append((k, "not in model"))
            continue
        if strict_shape and model_sd[k].shape != v.shape:
            skipped.append((k, f"shape mismatch: ckpt {v.shape} vs model {model_sd[k].shape}"))
            continue
        to_load[k] = v
    model.load_state_dict(to_load, strict=False)
    return list(to_load.keys()), skipped


def load_limix_transformer(model_path, mask_prediction: bool = False, strict_shape: bool = True):
    """
    用 checkpoint 的 config 建模型，但只加载与当前结构匹配的权重（你改了部分结构时用）。
    """
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = state_dict["config"]
    config["mask_prediction"] = mask_prediction
    # save_dir = "/data/gaoyisen/ultrapfn/config/limix"  # 自己改
    # import os, json
    # os.makedirs(save_dir, exist_ok=True)
    # config_path = os.path.join(save_dir, "limix_config.json")
    # with open(config_path, "w") as f:
    #     json.dump(config, f)
    model = build_custom_model(config)
    ckpt_sd = state_dict["state_dict"]
    loaded, skipped = load_state_dict_matching(model, ckpt_sd, strict_shape=strict_shape)
    print(f"Loaded {len(loaded)} keys from checkpoint.")
    if skipped:
        print(f"Skipped {len(skipped)} keys (e.g. {skipped[:3]}...).")
    model.train()
    return model

