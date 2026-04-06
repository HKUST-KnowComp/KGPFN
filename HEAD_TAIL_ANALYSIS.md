# 训练和评测的 Head/Tail 使用情况分析

## 当前实现总结

### 1. run_many.py (你的新测试脚本)

**评测方式**: **仅 Tail 预测**

```python
# 第 63 行
t_batch, _ = tasks.all_negative(test_graph, batch)  # 只使用 t_batch，忽略 h_batch

# 第 110 行
t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
rankings.append(t_ranking)  # 只计算 tail ranking
```

**结论**: 
- ✅ 只评测 tail 预测 (h, r, ?)
- ❌ 不评测 head 预测 (?, r, t)
- 返回的指标（MR, MRR, Hits@k）都是基于 tail 预测

---

### 2. pretrain_pfn.py (训练脚本)

#### 训练阶段

**负采样方式**: **仅 Tail 负采样**

```python
# 第 382 行
batch_with_neg = tasks.negative_sampling_tail(
    train_graph,
    batch,
    cfg.task.num_negative,
    strict=cfg.task.strict_negative,
)  # 输出: [B, num_negative + 1, 3]
# 第 0 列为正样本，第 1..N 列仅替换 tail
```

**结论**:
- ✅ 训练时只做 tail 预测任务 (h, r, ?)
- ❌ 不训练 head 预测 (?, r, t)

#### 测试阶段

**评测方式**: **仅 Tail 预测**

```python
# 第 563 行
t_batch, _ = tasks.all_negative(test_graph, batch)  # 只使用 t_batch

# 第 605 行
t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
rankings += [t_ranking]  # 只计算 tail ranking
```

**结论**:
- ✅ 测试时只评测 tail 预测 (h, r, ?)
- ❌ 不评测 head 预测 (?, r, t)

---

## 对比：ULTRA 原始实现

让我检查 ULTRA 的原始实现是否同时评测 head 和 tail：

### script/pretrain.py (ULTRA 的训练脚本)

```python
# 训练时使用 negative_sampling (同时采样 head 和 tail)
batch = tasks.negative_sampling(train_graph, batch, cfg.task.num_negative, ...)

# 测试时同时评测 head 和 tail
t_batch, h_batch = tasks.all_negative(test_graph, batch)
# ... 计算 t_ranking 和 h_ranking
# ranking = torch.cat([t_ranking, h_ranking])  # 合并两者
```

---

## 总结对比表

| 脚本 | 训练负采样 | 测试评测 | 指标含义 |
|------|-----------|---------|---------|
| **run_many.py** | N/A (仅测试) | **仅 Tail** | Tail 预测性能 |
| **pretrain_pfn.py** | **仅 Tail** | **仅 Tail** | Tail 预测性能 |
| **ULTRA pretrain.py** | Head + Tail | Head + Tail | 平均性能 |

---

## 影响分析

### 优点
1. **更快**: 只评测 tail，速度是 head+tail 的约 2 倍
2. **一致性**: 训练和测试都只关注 tail 预测，任务一致
3. **标准做法**: 很多 KG 论文只报告 tail 预测结果

### 缺点
1. **不完整**: 没有评测 head 预测能力
2. **可能偏差**: 某些关系可能 head 预测更难/更容易
3. **与 ULTRA 不一致**: ULTRA 原始实现同时评测两者

---

## 建议

### 选项 1: 保持当前实现（仅 Tail）
**适用场景**: 
- 快速评测
- 与其他只报告 tail 的论文对比
- 训练和测试一致性

### 选项 2: 改为同时评测 Head + Tail
**需要修改**:
1. run_many.py 中使用 `h_batch` 并计算 `h_ranking`
2. 合并 `t_ranking` 和 `h_ranking`
3. pretrain_pfn.py 的测试部分同样修改

**优点**:
- 更全面的评测
- 与 ULTRA 原始实现一致
- 更公平的性能比较

---

## 如果要改为 Head + Tail 评测

需要修改的关键代码位置：

### run_many.py
```python
# 第 63 行，改为：
t_batch, h_batch = tasks.all_negative(test_graph, batch)  # 使用两者

# 第 110 行后，添加 head 评测：
h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
ranking = torch.cat([t_ranking, h_ranking])  # 合并
rankings.append(ranking)
```

### pretrain_pfn.py
```python
# 第 563 行，改为：
t_batch, h_batch = tasks.all_negative(test_graph, batch)

# 第 605 行后，添加 head 评测并合并
```

---

## 当前状态确认

✅ **run_many.py**: 仅评测 Tail  
✅ **pretrain_pfn.py 训练**: 仅 Tail 负采样  
✅ **pretrain_pfn.py 测试**: 仅评测 Tail  

**一致性**: ✅ 训练和测试都只关注 Tail 预测
