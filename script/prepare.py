import os
import torch
import yaml
from huggingface_hub import hf_hub_download

os.makedirs("./cache", exist_ok=True)

files = {
    "classifier": "tabicl-classifier-v2-20260212.ckpt",
    "regressor": "tabicl-regressor-v2-20260212.ckpt",
}

for name, filename in files.items():
    path = hf_hub_download(
        repo_id="jingang/TabICL",
        filename=filename,
        local_dir="./cache",
    )
    print(f"\n=== {name}: {path} ===")
    ckpt = torch.load(path, map_location="cpu")
    print("Keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))

    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        # convert to plain dict if needed
        if hasattr(config, "__dict__"):
            config = vars(config)
        elif not isinstance(config, dict):
            config = dict(config)
        out_path = f"./cache/{name}_config.yaml"
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Config saved to {out_path}")
    else:
        print("No 'config' key found.")
