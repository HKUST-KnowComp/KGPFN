"""
Download pretrained checkpoints for KGPFN.

Usage:
  python script/download.py                          # ultra_50g + tabicl (default)
  python script/download.py --ultra 3g               # ultra_3g
  python script/download.py --ft limix               # limix instead of tabicl
  python script/download.py --ckpt_dir ./ckpts --cache_dir ./cache
  python script/download.py --kgpfn                  # kgpfn_icl_all.pth (default)
  python script/download.py --kgpfn icl_3g           # kgpfn_icl_3g.pth
  python script/download.py --kgpfn limix            # kgpfn_limix.pth
  python script/download.py --kgpfn iclsemantic      # kgpfn_iclsemantic.pth
"""
import argparse
import os
import urllib.request

import torch
import yaml
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_structure_encoder(variant: str, cache_dir: str) -> str:
    STRUCTURE_ENCODER_URLS = {
        "3g":  "https://github.com/DeepGraphLearning/ULTRA/raw/main/ckpts/ultra_3g.pth",
        "4g":  "https://github.com/DeepGraphLearning/ULTRA/raw/main/ckpts/ultra_4g.pth",
        "50g": "https://github.com/DeepGraphLearning/ULTRA/raw/main/ckpts/ultra_50g.pth",
    }
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"structure_encoder.pth")
    if os.path.exists(dest):
        print(f"[structure_encoder] Already exists: {dest}")
        return dest
    print(f"[structure_encoder] Downloading structure_encoder.pth ...")
    urllib.request.urlretrieve(STRUCTURE_ENCODER_URLS[variant], dest)
    print(f"[structure_encoder] Saved to {dest}")
    return dest


def download_tabicl(cache_dir: str) -> str:
    repo_id = "jingang/TabICL"
    filename = "tabicl-regressor-v2-20260212.ckpt"
    print(f"[tabicl] Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_dir)
    print(f"[tabicl] Saved to {path}")

    # extract and save config
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        if hasattr(config, "__dict__"):
            config = vars(config)
        elif not isinstance(config, dict):
            config = dict(config)
        config_path = os.path.join(PROJECT_ROOT, "config", "tabicl", "tabicl.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"[tabicl] Config saved to {config_path}")
    return path


def download_limix(cache_dir: str) -> str:
    repo_id = "stableai-org/LimiX-16M"
    filename = "LimiX-16M.ckpt"
    print(f"[limix] Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_dir)
    print(f"[limix] Saved to {path}")
    return path


KGPFN_VARIANTS = {
    "icl_all":     "kgpfn_icl_all.pth",
    "icl_3g":      "kgpfn_icl_3g.pth",
    "limix":       "kgpfn_limix.pth",
    "iclsemantic": "kgpfn_iclsemantic.pth",
}


def download_kgpfn(variant: str, cache_dir: str) -> str:
    repo_id = "Eason-nuo/KGPFN"
    filename = KGPFN_VARIANTS[variant]
    print(f"[kgpfn] Downloading {filename} from {repo_id} ...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_dir)
    print(f"[kgpfn] Saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure_encoder", choices=["3g", "4g", "50g"], default="50g")
    parser.add_argument("--ft", choices=["tabicl", "limix"], default="tabicl",
                        help="Feature transformer to download")
    parser.add_argument("--kgpfn", choices=list(KGPFN_VARIANTS.keys()), nargs="?",
                        const="icl_all", default=None,
                        help="Download a trained KGPFN model (default: icl_all)")
    parser.add_argument("--cache_dir", default="./cache")
    args = parser.parse_args()

    if args.kgpfn is not None:
        path = download_kgpfn(args.kgpfn, args.cache_dir)
        print("\n=== Done ===")
        print(f"  kgpfn_ckpt_path : {path}")
        return

    se_path = download_structure_encoder(args.structure_encoder, args.cache_dir)
    ft_path = download_tabicl(args.cache_dir) if args.ft == "tabicl" else download_limix(args.cache_dir)

    print("\n=== Done ===")
    print(f"  structure_encoder_path : {se_path}")
    print(f"  {args.ft}_ckpt_path    : {ft_path}")


if __name__ == "__main__":
    main()
