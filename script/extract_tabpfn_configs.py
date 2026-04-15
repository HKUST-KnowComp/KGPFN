"""Extract model configs from TabPFN classifier and regressor checkpoints and save as YAML."""
import os
import yaml
import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "tabpfn")
os.makedirs(OUT_DIR, exist_ok=True)

CKPTS = {
    "classifier": "/home/gaoyisen/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
    "regressor":  "/home/gaoyisen/.cache/tabpfn/tabpfn-v2-regressor.ckpt",
}

for name, ckpt_path in CKPTS.items():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    out_path = os.path.join(OUT_DIR, f"tabpfn_{name}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Saved {name} config -> {out_path}")

