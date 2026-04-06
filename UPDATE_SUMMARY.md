# 更新总结

## 完成的修改

### 1. `/data/gaoyisen/pfn/pfn/datasets.py`
- ✅ 补充完整了 `JointDataset.datasets_map`
- ✅ 取消了所有 inductive 数据集的注释
- ✅ 添加了 FB15k237_10, FB15k237_20, FB15k237_50 三个稀疏子集
- ✅ 添加了详细的分类注释

### 2. `/data/gaoyisen/pfn/script/run_many.py`
- ✅ 完全重写，支持测试 ULTRA 和 PFN 两种模型
- ✅ 支持通过 `--model_type` 参数选择模型类型
- ✅ 自动加载所有数据集（使用 `graphs: all`）
- ✅ 输出 MR, MRR, Hits@1, Hits@10, Hits@50 五个指标
- ✅ 计算三种平均值：overall, transductive, inductive
- ✅ 更新了 TRANSDUCTIVE_DATASETS 集合（包含 16 个数据集）
- ✅ 遵循 run.py 和 pretrain_pfn.py 的日志创建方法
- ✅ 结果保存到 CSV 文件

### 3. `/data/gaoyisen/pfn/config/transductive/test_ultra.yaml`
- ✅ 创建了新的测试配置文件
- ✅ 使用 `graphs: all` 加载所有数据集
- ✅ 同时支持 ULTRA 和 PFN 模型的参数配置
- ✅ 设置 `num_epoch: 0` 仅进行测试

### 4. 文档文件
- ✅ `/data/gaoyisen/pfn/TESTING_GUIDE.md` - 使用指南
- ✅ `/data/gaoyisen/pfn/DATASETS_STATISTICS.md` - 数据集统计

## 数据集总数：57个

### 分类统计
- **Transductive**: 16个
  - FB15k237 系列: 4个 (FB15k237, FB15k237_10, FB15k237_20, FB15k237_50)
  - 其他: 12个
- **Inductive (无新关系)**: 18个
  - GraIL: 12个
  - ILPC: 2个
  - HM: 4个
- **Inductive (有新关系)**: 23个
  - Ingram: 13个
  - WikiTopics: 8个 (注意：没有 mt/mt2/mt3/mt4 版本)
  - Other: 2个

## 使用示例

### 测试 ULTRA 模型 (ultra_50g.pth)
```bash
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type ultra \
  --ckpt ./ckpts/ultra_50g.pth \
  --gpus [0]
```

### 测试 PFN 模型
```bash
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type pfn \
  --ckpt ./checkpoints/model_best.pth \
  --gpus [0]
```

### 多GPU测试
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type ultra \
  --ckpt ./ckpts/ultra_50g.pth \
  --gpus [0,1]
```

## 输出结果

脚本会在 `output_dir` 下创建时间戳目录，包含：

1. **test_results.log** - 详细日志
   - 每个数据集的详细指标
   - 进度信息
   - 最终汇总

2. **test_results.csv** - CSV格式结果
   - 每个数据集一行
   - 包含所有5个指标
   - 最后三行是平均值：
     - OVERALL_AVERAGE
     - TRANSDUCTIVE_AVERAGE
     - INDUCTIVE_AVERAGE

## 关键特性

1. ✅ 自动检测模型类型（ULTRA vs PFN）
2. ✅ 自动分类数据集（transductive vs inductive）
3. ✅ 支持分布式训练（多GPU）
4. ✅ 完整的指标报告（MR, MRR, Hits@1, Hits@10, Hits@50）
5. ✅ 分类平均值计算
6. ✅ 遵循项目日志规范
7. ✅ 结果自动保存到CSV

## 验证

所有修改已完成，可以直接运行测试命令。
