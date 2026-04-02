CUDA_LAUNCH_BLOCKING=4 CUDA_VISIBLE_DEVICES=0 python script/test.py -c config/transductive/inference.yaml --dataset FB15k237 --epochs 0 --bpe null --gpus [0] --ckpt /data/gaoyisen/ULTRA/ckpts/ultra_50g.pth
 
CUDA_VISIBLE_DEVICES=1 python script/pretrain_pfn.py -c config/transductive/pretrain_3g.yaml --gpus [0]

CUDA_VISIBLE_DEVICES=1 python script/run.py -c config/transductive/inference.yaml --dataset CoDExSmall --epochs 1 --bpe null --gpus [0]

# for one dataset
CUDA_VISIBLE_DEVICES=7 python script/run.py -c config/transductive/train.yaml --dataset CoDExSmall  --bpe null --gpus [0]
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 script/run.py \
  -c config/transductive/train.yaml --dataset CoDExSmall --bpe null --gpus [0,1]

# for multi-graph pretraining
CUDA_VISIBLE_DEVICES=0 python script/pretrain_pfn.py -c config/transductive/train_3g.yaml --gpus [0]
CUDA_VISIBLE_DEVICES=7 python script/pretrain_pfn.py -c config/transductive/train_all.yaml --gpus [0]

#prepare data
python script/run_many.py -c /data/gaoyisen/ultrapfn2/config/transductive/inference.yaml --gpus [0] --ckpt /data/gaoyisen/ULTRA/ckpts/ultra_4g.pth -d FB15k237Inductive:v1,FB15k237Inductive:v2,FB15k237Inductive:v3,FB15k237Inductive:v4