# 版本自动扩展修复说明

## 问题

当使用 `graphs: all` 时，JointDataset 会尝试加载所有数据集。对于有多个版本的数据集（如 FB15k237Inductive），如果数据集类没有 `versions` 属性，JointDataset 会尝试用 `dataset_version=None` 来初始化，导致 AssertionError。

## 错误信息

```
AssertionError at line 157 in GrailInductiveDataset.__init__
assert version in ["v1", "v2", "v3", "v4"]
```

## 解决方案

为所有需要版本扩展的数据集类添加 `versions` 属性，这样 JointDataset 会自动扩展为所有版本。

## 已修复的数据集类

### 1. GraIL Inductive Datasets
```python
class FB15k237Inductive(GrailInductiveDataset):
    versions = ["v1", "v2", "v3", "v4"]  # ✅ 添加

class WN18RRInductive(GrailInductiveDataset):
    versions = ["v1", "v2", "v3", "v4"]  # ✅ 添加

class NELLInductive(GrailInductiveDataset):
    versions = ["v1", "v2", "v3", "v4"]  # ✅ 添加
```

### 2. Ingram Datasets
```python
class FBIngram(IngramInductive):
    versions = ["25", "50", "75", "100"]  # ✅ 添加

class WKIngram(IngramInductive):
    versions = ["25", "50", "75", "100"]  # ✅ 添加

class NLIngram(IngramInductive):
    versions = ["0", "25", "50", "75", "100"]  # ✅ 添加
```

### 3. 其他已有 versions 的数据集（无需修改）
```python
class ILPC2022(InductiveDataset):
    versions = ["small", "large"]  # ✅ 已存在

class HM(InductiveDataset):
    versions = {'1k': "...", '3k': "...", '5k': "...", 'indigo': "..."}  # ✅ 已存在

class WikiTopicsMT1(WikiTopics):
    versions = ['health', 'tax']  # ✅ 已修正

class WikiTopicsMT2(WikiTopics):
    versions = ['org', 'sci']  # ✅ 已修正

class WikiTopicsMT3(WikiTopics):
    versions = ['art', 'infra']  # ✅ 已修正

class WikiTopicsMT4(WikiTopics):
    versions = ['sci', 'health']  # ✅ 已修正

class Metafam(MTDEAInductive):
    versions = ["Metafam"]  # ✅ 已存在

class FBNELL(MTDEAInductive):
    versions = ["FBNELL_v1"]  # ✅ 已存在
```

## JointDataset 的版本扩展逻辑

在 `JointDataset.__init__` 中（第 3747-3753 行）：

```python
versions = getattr(ds_cls, "versions", None)
if ds_version is None and isinstance(versions, dict) and len(versions) > 0:
    expanded_specs.extend([f"{ds_name}:{v}" for v in versions.keys()])
elif ds_version is None and isinstance(versions, (list, tuple)) and len(versions) > 0:
    expanded_specs.extend([f"{ds_name}:{v}" for v in versions])
else:
    expanded_specs.append(f"{ds_name}:{ds_version}" if ds_version is not None else ds_name)
```

**工作原理**：
1. 如果数据集类有 `versions` 属性且 `dataset_version=None`
2. JointDataset 会自动扩展为所有版本
3. 例如：`FB15k237Inductive` → `FB15k237Inductive:v1`, `FB15k237Inductive:v2`, `FB15k237Inductive:v3`, `FB15k237Inductive:v4`

## 验证

现在运行以下命令应该不会报错：

```bash
CUDA_VISIBLE_DEVICES=0 python script/run_many.py \
  -c config/transductive/test_ultra.yaml \
  --model_type ultra \
  --ckpt ./ckpts/ultra_50g.pth \
  --gpus [0]
```

## 数据集总数确认

使用 `graphs: all` 会加载：
- 16个 transductive 数据集
- 18个 inductive (无新关系) 数据集
  - 12个 GraIL (3类 × 4版本)
  - 2个 ILPC
  - 4个 HM
- 23个 inductive (有新关系) 数据集
  - 13个 Ingram (NL:5 + FB:4 + WK:4)
  - 8个 WikiTopics (4类 × 2版本)
  - 2个 Other

**总计：57个数据集** ✅
