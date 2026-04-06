# Testing ULTRA and PFN Models on All KG Datasets

This guide explains how to use the updated `run_many.py` script to test models on all available KG datasets.

## Overview

The script now supports:
- Testing both ULTRA and PFN models
- Loading all datasets using `graphs: all` in config
- Computing MR, MRR, Hits@1, Hits@10, Hits@50 for each dataset
- Calculating separate averages for transductive, inductive, and overall datasets
- Logging results to both console and CSV file

## Usage

### Test ULTRA model (ultra_50g.pth)

```bash
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type ultra \
  --ckpt ./ckpts/ultra_50g.pth \
  --gpus [0]
```

### Test PFN model

```bash
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type pfn \
  --ckpt ./checkpoints/model_best.pth \
  --gpus [0]
```

### Multi-GPU testing

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type ultra \
  --ckpt ./ckpts/ultra_50g.pth \
  --gpus [0,1]
```

## Configuration

The `config/transductive/test_ultra.yaml` file is configured to:
- Load all available datasets via `graphs: all`
- Work with both ULTRA and PFN models
- Use appropriate evaluation parameters

Key settings:
- `dataset.graphs: all` - loads all transductive and inductive datasets
- `train.num_epoch: 0` - no training, only evaluation
- `train.eval_chunk_size: 256` - controls memory usage during evaluation

## Output

The script creates a timestamped directory in `output_dir` containing:

1. **test_results.log** - Detailed log with per-dataset metrics
2. **test_results.csv** - CSV file with all results including:
   - Individual dataset metrics
   - OVERALL_AVERAGE (all datasets)
   - TRANSDUCTIVE_AVERAGE (transductive datasets only)
   - INDUCTIVE_AVERAGE (inductive datasets only)

## Metrics Reported

For each dataset:
- **MR** (Mean Rank) - Lower is better
- **MRR** (Mean Reciprocal Rank) - Higher is better
- **Hits@1** - Percentage of correct predictions in top 1
- **Hits@10** - Percentage of correct predictions in top 10
- **Hits@50** - Percentage of correct predictions in top 50

## Dataset Classification

**Transductive datasets** (13 total):
- FB15k237, WN18RR, CoDExSmall, CoDExMedium, CoDExLarge
- NELL995, ConceptNet100k, DBpedia100k, YAGO310, AristoV4
- Hetionet, WDsinger, NELL23k

**Inductive datasets**:
- All other datasets (FB15k237Inductive variants, ILPC, Ingram, WikiTopics, etc.)

## Notes

- The script automatically detects whether a model is ULTRA or PFN based on model architecture
- For PFN models, context-based prediction is used with the configured num_pos/num_neg
- For ULTRA models, direct prediction is used
- Results are synchronized across all GPUs in distributed mode
- The working directory is NOT used as the root directory (follows run.py and pretrain_pfn.py conventions)
