# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UltraPFN3 is a research codebase for **knowledge graph link prediction** combining:
- **Structure encoder**: ULTRA-style relational message passing (NBFNet/Bellman-Ford GNN)
- **Semantic encoder**: Optional SentenceTransformer text embeddings
- **Feature transformer**: LimiX-based transformer for final scoring

## Environment Setup

Python 3.10+ required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub sentence-transformers wandb
```

`torch-scatter` and `torch-geometric` require version-matched wheels for your CUDA/PyTorch build — install accordingly if the above fails.

## Common Commands

**Single-dataset training:**
```bash
CUDA_VISIBLE_DEVICES=0 python script/run.py -c config/transductive/train.yaml --dataset CoDExSmall --bpe null --gpus [0]
```

**Single-dataset inference/validation:**
```bash
CUDA_VISIBLE_DEVICES=0 python script/run.py -c config/transductive/inference.yaml --dataset CoDExSmall --epochs 1 --bpe null --gpus [0]
```

**Multi-graph pretraining:**
```bash
CUDA_VISIBLE_DEVICES=0 python script/pretrain_pfn.py -c config/transductive/train_3g.yaml --gpus [0]
CUDA_VISIBLE_DEVICES=0 python script/pretrain_pfn.py -c config/transductive/train_all.yaml --gpus [0]
```

**Multi-GPU distributed training (torchrun):**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 script/run.py -c config/transductive/train.yaml --dataset CoDExSmall --bpe null --gpus [0,1]
```

**Batch evaluation across datasets:**
```bash
python script/run_many.py -c config/transductive/inference.yaml --gpus [0] --ckpt ./ckpts/ultra_4g.pth -d FB15k237Inductive:v1,FB15k237Inductive:v2
```

**Test script (inspect entity/relation name mappings):**
```bash
CUDA_VISIBLE_DEVICES=0 python script/test.py -c config/transductive/inference.yaml --dataset FB15k237 --epochs 0 --bpe null --gpus [0] --ckpt ./ckpts/ultra_50g.pth
```

## Config System

Configs are YAML with Jinja2 template variables. CLI args map to template slots:

| CLI arg | Template variable | Purpose |
|---------|------------------|---------|
| `--gpus [0]` | `{{ gpus }}` | GPU list |
| `--dataset NAME` | `{{ dataset }}` | Dataset name |
| `--epochs N` | `{{ epochs }}` | Epoch count |
| `--bpe null` | disables `batch_per_epoch` limit | `null` = no limit |
| `--ckpt PATH` | passed directly | Checkpoint path |

**Before running, update these paths in the YAML configs:**
- `dataset.root` — path to KG datasets
- `output_dir` — where logs and checkpoints go
- `train.structure_encoder_path` — pretrained ULTRA checkpoint
- `train.limix_cache_dir` — LimiX model cache location
- `train.kgpfn_checkpoint` — resume from checkpoint (null = fresh start)

## Architecture

### Model Pipeline (`model/kgpfnsem.py`)

1. **Structure path** (if `structure_encoder_name != "none"`): ULTRA encoder → `(h_emb, r_emb, t_emb)` → linear adapters → LayerNorm
2. **Semantic path** (if `semantic_encoder.model_name != "none"`): SentenceTransformer → 384-dim text embeddings → linear adapter → LayerNorm
3. **Fusion**: element-wise combination of both paths
4. **LimiX transformer**: final tabular-style scoring over candidate entities

### Structure Encoder (`model/ultra/`)

- `StructureEncoderRelationAware` in `encoder.py` — main encoder class
- `RelNBFNet` — Bellman-Ford message passing on the relation graph
- `EntityNBFNet` — entity embedding via graph convolution
- Memory controls: `seq_chunk_size`, `entity_chunk_size`, `eval_chunk_size` in config

### Entry Points

| Script | Purpose |
|--------|---------|
| `script/run.py` | Single-dataset train or inference |
| `script/pretrain_pfn.py` | Multi-graph joint pretraining |
| `script/run_many.py` | Batch inference across multiple datasets |
| `script/test.py` | Inspect entity/relation ID→name mappings |

### Dataset Classes (`pfn/datasets.py`)

The `Atlas` dataset class (used in `train.yaml`) loads from a versioned directory structure. Standard datasets (FB15k237, WN18RR, CoDEx variants, YAGO310) are loaded via PyKEEN/PyG. `JointDataset` combines multiple KGs for pretraining.

## Key Config Options

| Option | Values | Effect |
|--------|--------|--------|
| `model.structure_encoder_name` | `"ultra"` / `"none"` | Enable/disable structure encoder |
| `model.semantic_encoder.model_name` | model name / `"none"` | Enable/disable semantic encoder |
| `task.loss_type` | `"softmax"` / `"bce"` | Loss function |
| `task.context_label_correction` | `true` / `false` | Soft-label correction for false negatives |
| `train.inverse_relation_semantic_mode` | `"text"` / `"negate"` | How to embed inverse relations |
| `train.train_structure_encoder` | `true` / `false` | Whether to fine-tune the ULTRA encoder |

## Checkpoints

Pretrained ULTRA checkpoints in `ckpts/`:
- `ultra_3g.pth` — pretraining on 3 graphs (FB15k237, WN18RR, CoDExMedium)
- `ultra_4g.pth` — pretraining on 4 graphs
- `ultra_50g.pth` — pretraining on 50 graphs (best for zero-shot transfer)
