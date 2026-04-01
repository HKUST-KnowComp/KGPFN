## UltraPFN3

UltraPFN3 is a research codebase for knowledge graph link prediction with:

- structure encoder (ULTRA-style relational message passing),
- optional semantic encoder (e.g. SentenceTransformer),
- PFN-style feature transformer (LimiX-based component),
- training/inference pipelines for both single-graph and multi-graph settings.

The main runnable scripts are in `script/`, and the core workflow examples are provided in `demo.sh`.

## Repository Structure

- `script/`: train / pretrain / test entry points
- `config/transductive/`: task configs (inference, train, pretrain)
- `config/limix/`: LimiX transformer config
- `model/`: model implementations
- `pfn/`: task/util/dataset helpers
- `utils/`: loading and utility helpers
- `ckpts/`: optional pretrained checkpoints
- `cache/`: optional downloaded model cache (should be ignored by git)

## Environment Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub sentence-transformers wandb
```

> Note: `torch-scatter` / `torch-geometric` may require CUDA/PyTorch-version-matched wheels.  
> Install those according to your local CUDA and PyTorch versions if needed.

## Data and Checkpoint Preparation

Before running, check and update paths in config files:

- dataset root (e.g. `dataset.root` in `config/transductive/*.yaml`)
- output dir (`output_dir`)
- checkpoint paths (`train.structure_encoder_path`, `train.kgpfn_checkpoint`)
- cache dir (`train.limix_cache_dir`)

For open-source usage, avoid committing large files such as:

- dataset files,
- `*.ckpt` / `*.pth`,
- cache files in `cache/`,
- large `.tsv` mapping files.

## Quick Start

All examples below are adapted from `demo.sh`.

### 1) Inference / quick validation

```bash
CUDA_VISIBLE_DEVICES=1 python script/run.py \
  -c config/transductive/inference.yaml \
  --dataset CoDExSmall \
  --epochs 1 \
  --bpe null \
  --gpus [0]
```

### 2) Single-dataset training

```bash
CUDA_VISIBLE_DEVICES=5 python script/run.py \
  -c config/transductive/train.yaml \
  --dataset CoDExSmall \
  --bpe null \
  --gpus [0]
```

### 3) Multi-graph pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python script/pretrain_pfn.py \
  -c config/transductive/train_3g.yaml \
  --gpus [0]
```

Alternative config:

```bash
CUDA_VISIBLE_DEVICES=7 python script/pretrain_pfn.py \
  -c config/transductive/train_all.yaml \
  --gpus [0]
```

### 4) Batch run across multiple datasets

```bash
python script/run_many.py \
  -c config/transductive/inference.yaml \
  --gpus [0] \
  --ckpt ./ckpts/ultra_4g.pth \
  -d FB15k237Inductive:v1,FB15k237Inductive:v2,FB15k237Inductive:v3,FB15k237Inductive:v4
```

### 5) Test script example

```bash
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python script/test.py \
  -c config/transductive/inference.yaml \
  --dataset FB15k237 \
  --epochs 0 \
  --bpe null \
  --gpus [0] \
  --ckpt ./ckpts/ultra_50g.pth
```

## Command Arguments

Most scripts support:

- `-c, --config`: YAML config path (required)
- `--dataset`: dataset name for templated configs
- `--gpus`: GPU list, e.g. `[0]`
- `--epochs`: number of epochs (used in templated configs)
- `--bpe`: batches per epoch (`null` to disable limit)
- `--ckpt`: checkpoint path (for evaluation/inference flows)

Configs use Jinja-style template variables (e.g. `{{ dataset }}`, `{{ gpus }}`, `{{ epochs }}`), so these variables must be provided via CLI.

## Logging and Outputs

Outputs are written under `output_dir` in config, typically by timestamped subfolders, containing:

- logs,
- copied runtime config,
- checkpoints / metrics.

If `use_wandb: true`, metrics are also logged to Weights & Biases.

## Notes for Public Release

- Remove or rotate any exposed API keys or private tokens.
- Keep dataset and model artifacts outside git history.
- Store large checkpoints in external storage (Git LFS / Hugging Face / release assets) and reference them in README.
