# Structure Score Enhancement 功能实现总结

## ✅ 已完成的修改

### 1. 模型代码 (`model/kgpfnsem.py`)

**添加的参数**：
- `structure_score_enhance: bool = False` - 控制是否启用该功能

**添加的模块**：
```python
if structure_score_enhance:
    self.structure_score_adapter = nn.Sequential(
        nn.Linear(1, self.hidden_dim),
        nn.GELU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
    )
    self.structure_score_norm = nn.LayerNorm(self.hidden_dim)
```

**修改的方法**：
- `_build_structure_aligned()`: 计算 ULTRA MLP scores 并通过 adapter 转换为特征

### 2. 配置文件

**已更新**：
- ✅ `config/transductive/train_all.yaml.example`
- ✅ `config/transductive/test_ultra.yaml`

**新增配置项**：
```yaml
model:
  structure_score_enhance: False  # 设为 True 启用
```

### 3. 训练/测试脚本

**已更新**：
- ✅ `script/pretrain_pfn.py`
- ✅ `script/run.py`

**修改内容**：
```python
structure_score_enhance = bool(cfg.model.get("structure_score_enhance", False))
model = KGPFN(
    ...
    structure_score_enhance=structure_score_enhance,
)
```

### 4. 文档

**已创建**：
- ✅ `STRUCTURE_SCORE_ENHANCE.md` - 详细的功能说明文档

## 功能说明

### 工作原理

1. **计算 Score**: 对每个三元组使用 ULTRA 的 EntityNBFNet MLP 计算结构分数
2. **维度对齐**: 通过 2 层 MLP adapter 将 score (1维) 映射到 hidden_dim
3. **特征融合**: 将 score 特征拼接到 structure features 中，输入 feature_transformer

### 特征维度

| 配置 | 输出维度 |
|------|---------|
| 基础 | [B, S, 3, D] |
| + enhance_structure | [B, S, 6, D] |
| + structure_score_enhance | [B, S, 4, D] 或 [B, S, 7, D] |

### 向后兼容

- ✅ 默认 `structure_score_enhance=False`，行为与之前完全一致
- ✅ 旧的 checkpoint 可以正常加载
- ✅ 不影响现有功能

## 使用方法

### 启用功能

**方法 1**: 修改配置文件
```yaml
# 在 train_all.yaml.example 或 test_ultra.yaml 中
model:
  structure_score_enhance: True
```

**方法 2**: 命令行（如果支持）
```bash
# 训练
CUDA_VISIBLE_DEVICES=0 python script/pretrain_pfn.py \
  -c config/transductive/train_all.yaml.example \
  --gpus [0]

# 测试
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type pfn \
  --ckpt ./checkpoints/model_best.pth \
  --gpus [0]
```

### 禁用功能（默认）

保持配置文件中 `structure_score_enhance: False` 或不设置该项。

## 技术细节

### Score 计算
- 使用 `structure_encoder.get_mlp_scores(t_emb)` 获取分数
- 在 `torch.no_grad()` 下计算，不回传梯度到 structure encoder
- 对所有三元组（context + query）统一计算

### Adapter 架构
```python
nn.Sequential(
    nn.Linear(1, hidden_dim),      # 1 → D
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),  # D → D
)
+ LayerNorm(hidden_dim)
```

### 特征拼接
- 在 `dim=2` (特征维度) 上拼接
- Score 特征形状: [B, S, 1, D]
- 最终形状: [B, S, 3+enhance+score, D]

## 实验建议

### 对比实验设置

| 实验 | enhance_structure | structure_score_enhance | 说明 |
|------|------------------|------------------------|------|
| Baseline | False | False | 仅基础特征 |
| Enhance | True | False | 添加 TransE/DistMult/Cos |
| Score | False | True | 仅添加 ULTRA score |
| Both | True | True | 所有特征 |

### 评估维度
1. **性能**: MR, MRR, Hits@k
2. **数据集类型**: Transductive vs Inductive
3. **训练效率**: 时间、内存占用
4. **收敛速度**: Loss 曲线

## 注意事项

### 要求
- ✅ 必须有 `structure_encoder` (不能是 "none")
- ✅ Structure encoder 必须有 `get_mlp_scores()` 方法

### 潜在问题
1. **信息冗余**: Score 可能与 embeddings 重叠
2. **过拟合**: 增加了模型参数
3. **计算开销**: 额外的 MLP forward（但很小）

### 调试建议
- 检查 score 的数值范围（应该是 logits）
- 观察 adapter 输出的分布
- 对比有无 score 特征的性能差异

## 文件清单

### 修改的文件
1. `model/kgpfnsem.py` - 核心实现
2. `script/pretrain_pfn.py` - 训练脚本
3. `script/run.py` - 单数据集脚本
4. `config/transductive/train_all.yaml.example` - 训练配置
5. `config/transductive/test_ultra.yaml` - 测试配置

### 新增的文件
1. `STRUCTURE_SCORE_ENHANCE.md` - 功能文档

### 未修改的文件
- `script/run_many.py` - 使用 `create_pfn_model()` 会自动支持
- `model/ultra/encoder.py` - 已有 `get_mlp_scores()` 方法
- 其他配置文件 - 可按需添加该选项

## 验证清单

- ✅ 模型可以正常初始化（`structure_score_enhance=False`）
- ✅ 模型可以正常初始化（`structure_score_enhance=True`）
- ✅ 配置文件语法正确
- ✅ 训练脚本可以读取配置
- ✅ 向后兼容（旧配置仍然工作）
- ⏳ 功能测试（需要实际运行验证）

## 下一步

1. **测试运行**: 用小数据集验证功能正常
2. **性能对比**: 运行对比实验
3. **参数调优**: 调整 adapter 架构
4. **文档完善**: 根据实验结果更新文档
