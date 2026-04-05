#!/bin/bash

# 激活你的 conda 环境
source /aifs4su/hansirui_3rd/gaoyisen/miniconda3/bin/activate limix

# 设置分布式训练端口，避免和别的任务冲突
export MASTER_PORT=43001
export MASTER_ADDR=localhost

# 单机 4 卡训练
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" script/pretrain_pfn.py \
  -c config/transductive/train_all.yaml --gpus [0,1]