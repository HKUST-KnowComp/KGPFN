#!/bin/bash

# 激活你的 conda 环境
source /aifs4su/hansirui_3rd/gaoyisen/miniconda3/bin/activate limix

# 设置分布式训练端口，避免和别的任务冲突
export MASTER_PORT=43001
export MASTER_ADDR=localhost

accelerate launch --num_processes 4 script/pretrain_pfn.py -c config/transductive/train_all.yaml --gpus [0,1,2,3]
