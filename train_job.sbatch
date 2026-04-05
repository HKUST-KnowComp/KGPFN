#!/bin/bash
#SBATCH -J pfn-4g
#SBATCH -p llm
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=3-00:00:00
#SBATCH -o /aifs4su/hansirui_3rd/gaoyisen/pfn/logs/slurm-%j.out
#SBATCH -e /aifs4su/hansirui_3rd/gaoyisen/pfn/logs/slurm-%j.err

# 日志目录（与 -o/-e 一致；若目录不存在请先: mkdir -p pfn/logs）
mkdir -p /aifs4su/hansirui_3rd/gaoyisen/pfn/logs

cd /aifs4su/hansirui_3rd/gaoyisen/pfn || exit 1
bash train.sh
