# Structure Score Enhancement Feature

## 概述

`structure_score_enhance` 是一个新特征，它将 ULTRA 的 MLP 打分作为额外特征输入到 feature_transformer 中。

## 实现原理

### 1. ULTRA MLP Score
ULTRA 模型在 EntityNBFNet 的最后使用一个 MLP 将 entity embedding 投影为标量分数：
```python
# 在 EntityNBFNet 中
self.mlp = nn.Sequential(
    nn.Linear(feature_dim, feature_dim),
    nn.ReLU(),
    nn.Linear(feature_dim, 1)
)
score = self.mlp(feature).squeeze(-1)  # [B, S]
```

### 2. Score 作为特征
当 `structure_score_enhance=True` 时：
1. 对每个三元组（包括 context 和 query）计算 ULTRA MLP score
2. 通过 adapter 将 score (1维) 映射到 hidden_dim
3. 作为额外的特征维度拼接到 structure features 中

### 3. 特征维度变化

| 配置 | 特征维度 | 说明 |
|------|---------|------|
| 基础 | [B, S, 3, D] | h, r, t 三个 embedding |
| + enhance_structure | [B, S, 6, D] | 额外 3 个增强特征 (TransE, DistMult, Cosine) |
| + structure_score_enhance | [B, S, 7, D] | 额外 1 个 ULTRA score 特征 |
| 两者都开启 | [B, S, 7, D] | 3 (基础) + 3 (enhance) + 1 (score) |

## 代码实现

### 1. KGPFN 模型修改

```python
class KGPFN(nn.Module):
    def __init__(
        self,
        *,
        structure_score_enhance: bool = False,  # 新参数
        ...
    ):
        # 添加 score adapter
        if structure_score_enhance:
            self.structure_score_adapter = nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.structure_score_norm = nn.LayerNorm(self.hidden_dim)
```

### 2. _build_structure_aligned 方法

```python
def _build_structure_aligned(self, data, all_id_triples, num_context, context_y):
    # 1. 获取 embeddings
    h_emb, r_emb, t_emb = self._triples_to_embeddings(data, all_id_triples)
    
    # 2. 计算 ULTRA scores (如果启用)
    if self.structure_score_enhance:
        with torch.no_grad():
            structure_scores = self.structure_encoder.get_mlp_scores(t_emb)  # [B, S]
    
    # 3. 应用 adapters
    h_emb = self.entity_adapter(h_emb)
    r_emb = self.relation_adapter(r_emb)
    t_emb = self.entity_adapter(t_emb)
    structure_aligned = self.structure_norm(torch.stack([h_emb, r_emb, t_emb], dim=2))
    
    # 4. 添加 enhance features (如果启用)
    if self.enhance_structure:
        # ... TransE, DistMult, Cosine features
        structure_aligned = torch.cat([structure_aligned, enh_delta], dim=2)
    
    # 5. 添加 score feature (如果启用)
    if self.structure_score_enhance:
        score_feat = structure_scores.unsqueeze(-1)  # [B, S, 1]
        score_feat = self.structure_score_adapter(score_feat)  # [B, S, D]
        score_feat = self.structure_score_norm(score_feat)
        score_feat = score_feat.unsqueeze(2)  # [B, S, 1, D]
        structure_aligned = torch.cat([structure_aligned, score_feat], dim=2)
    
    return structure_aligned, context_y
```

## 配置使用

### 1. 配置文件

在 `train_all.yaml.example` 或 `test_ultra.yaml` 中：

```yaml
model:
  structure_encoder_name: ultra
  enhance_structure: True
  structure_score_enhance: False  # 设为 True 启用
```

### 2. 训练脚本

`pretrain_pfn.py` 和 `run.py` 会自动读取配置：

```python
structure_score_enhance = bool(cfg.model.get("structure_score_enhance", False))
model = KGPFN(
    structure_encoder=structure_encoder,
    enhance_structure=enhance_structure,
    structure_score_enhance=structure_score_enhance,  # 传入参数
    ...
)
```

## 使用示例

### 训练时启用

```bash
# 修改配置文件，设置 structure_score_enhance: True
CUDA_VISIBLE_DEVICES=0 python script/pretrain_pfn.py \
  -c config/transductive/train_all.yaml.example \
  --gpus [0]
```

### 测试时启用

```bash
# 修改 test_ultra.yaml，设置 structure_score_enhance: True
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type pfn \
  --ckpt ./checkpoints/model_best.pth \
  --gpus [0]
```

## 兼容性

### 向后兼容
- 默认 `structure_score_enhance=False`，行为与之前完全一致
- 旧的 checkpoint 可以正常加载（使用 `strict=False`）

### 要求
- 必须有 `structure_encoder`（即 `structure_encoder_name != "none"`）
- 不能单独使用，必须配合 structure encoder

## 预期效果

### 优点
1. **额外的判别信息**: ULTRA 的 MLP score 包含了结构信息的全局判断
2. **端到端学习**: Score 通过 adapter 学习如何与其他特征融合
3. **灵活性**: 可以独立开关，不影响其他特征

### 潜在问题
1. **信息冗余**: Score 可能与 h, r, t embeddings 有重叠
2. **过拟合风险**: 增加了模型参数
3. **计算开销**: 需要额外的 MLP forward pass（但使用 `torch.no_grad()`）

## 实验建议

### 对比实验
1. Baseline: `enhance_structure=False, structure_score_enhance=False`
2. Enhance only: `enhance_structure=True, structure_score_enhance=False`
3. Score only: `enhance_structure=False, structure_score_enhance=True`
4. Both: `enhance_structure=True, structure_score_enhance=True`

### 评估指标
- MR, MRR, Hits@1, Hits@10, Hits@50
- 分别在 transductive 和 inductive 数据集上评估
- 观察是否对某类数据集特别有效

## 技术细节

### Score 计算
- 使用 `torch.no_grad()` 避免梯度回传到 structure encoder
- Score 是基于 tail embedding 计算的（与 ULTRA 一致）

### Adapter 设计
- 2层 MLP: 1 → D → D
- 使用 GELU 激活函数
- LayerNorm 归一化

### 特征拼接
- 在 dim=2 上拼接（特征维度）
- 保持 [B, S, *, D] 的格式，* 是特征数量
