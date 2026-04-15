# import os
# import sys
# import yaml
# import torch

# _PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# _MODEL_DIR = os.path.join(_PROJECT_ROOT, "model")
# # model/ first so `from tabpfn.xxx` resolves to local model/tabpfn/
# sys.path.insert(0, _MODEL_DIR)
# sys.path.insert(1, _PROJECT_ROOT)

# from model.tabpfn.architectures.base.config import ModelConfig
# from model.tabpfn.architectures.base.custom_transformer import CustomPerFeatureTransformer

# _CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "tabpfn", "tabpfn.yaml")
# _CKPT_PATH = os.path.expanduser(
#     "~/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
# )


# def load_tabpfn_model(ckpt_path: str, config_path: str, device: str = "cuda"):
#     """Build custom model from yaml config, load matching weights from ckpt."""
#     with open(config_path) as f:
#         cfg_dict = yaml.safe_load(f)

#     # Build ModelConfig from yaml
#     model_config, _ = ModelConfig.upgrade_config(cfg_dict), None
#     model_config = ModelConfig(**ModelConfig.upgrade_config(cfg_dict))

#     structure_encoder_dim = cfg_dict.get("structure_encoder_dim", 64)

#     model = CustomPerFeatureTransformer(
#         config=model_config,
#         structure_encoder_dim=structure_encoder_dim,
#         n_out=model_config.max_num_classes or 10,
#     )

#     # Load only state_dict from checkpoint
#     checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
#     full_state = checkpoint["state_dict"]
#     # Filter out criterion keys
#     model_state = {k: v for k, v in full_state.items() if "criterion." not in k}
#     matched, total = 0, 0
#     for k, v in model_state.items():
#         total += 1
#         if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
#             matched += 1
#     model.load_state_dict(model_state, strict=False)
#     print(f"Loaded weights: {matched}/{total} keys matched")

#     return model.to(device).eval()


# if __name__ == "__main__":
#     with open(_CONFIG_PATH) as f:
#         cfg = yaml.safe_load(f)
#     device = cfg.get("device", "cuda")

#     model = load_tabpfn_model(_CKPT_PATH, config_path=_CONFIG_PATH, device=device)
#     print(model)

#     # Simulate input:
#     # x: [B, S, F, D] — B=2, S=15 (10 train + 5 test), F=3 feature groups, D=64
#     # y: [B, S]       — train labels + NaN placeholders for test
#     B, M, N, F, D = 2, 10, 5, 3, 64
#     S = M + N
#     x = torch.randn(B, S, F, D, device=device)
#     y = torch.cat([
#         torch.randint(0, 2, (B, M), device=device).float(),
#         torch.full((B, N), float("nan"), device=device),
#     ], dim=1)  # [B, S]

#     with torch.no_grad():
#         out = model(x, y, eval_pos=M)  # [B, N, n_out]

#     print(f"x shape:   {x.shape}")
#     print(f"y shape:   {y.shape}")
#     print(f"out shape: {out.shape}")
#     print(f"out[0,0]:  {out[0, 0]}")
from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

model = TabPFNClassifier.create_default_for_version(ModelVersion.V2,device='cuda')
model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cuda')
