# Knowledge Graph Datasets Statistics

## 总览

使用 `graphs: all` 时，JointDataset 会加载所有非 Atlas 数据集，共计 **57个数据集**。

## 详细分类

### 1. Transductive Datasets (16个)

这些数据集在训练和测试时使用相同的实体集合，只是边的划分不同。

| 序号 | 数据集名称 | 说明 |
|------|-----------|------|
| 1 | FB15k237 | Freebase子集 |
| 2 | FB15k237_10 | FB15k237稀疏子集 (10%训练数据) |
| 3 | FB15k237_20 | FB15k237稀疏子集 (20%训练数据) |
| 4 | FB15k237_50 | FB15k237稀疏子集 (50%训练数据) |
| 5 | WN18RR | WordNet关系推理 |
| 6 | CoDExSmall | CoDEx小规模 |
| 7 | CoDExMedium | CoDEx中等规模 |
| 8 | CoDExLarge | CoDEx大规模 |
| 9 | NELL995 | NELL知识库 |
| 10 | ConceptNet100k | 常识知识图谱 |
| 11 | DBpedia100k | DBpedia子集 |
| 12 | YAGO310 | YAGO知识库 |
| 13 | AristoV4 | 科学知识图谱 |
| 14 | Hetionet | 生物医学知识图谱 |
| 15 | WDsinger | Wikidata歌手子集 |
| 16 | NELL23k | NELL稀疏版本 |

### 2. Inductive Datasets - New Nodes, No New Relations (18个)

测试时出现新实体，但关系类型与训练时相同。

#### 2.1 GraIL Datasets (12个)

| 基础数据集 | 版本 | 数量 |
|-----------|------|------|
| FB15k237Inductive | v1, v2, v3, v4 | 4个 |
| WN18RRInductive | v1, v2, v3, v4 | 4个 |
| NELLInductive | v1, v2, v3, v4 | 4个 |

#### 2.2 ILPC Datasets (2个)

| 数据集 | 版本 |
|--------|------|
| ILPC2022 | small, large |

#### 2.3 Hamaguchi Datasets (4个)

| 数据集 | 版本 |
|--------|------|
| HM | 1k, 3k, 5k, indigo |

### 3. Inductive Datasets - New Nodes, New Relations (27个)

测试时同时出现新实体和新关系类型。

#### 3.1 Ingram Datasets (13个)

| 数据集 | 版本 | 数量 |
|--------|------|------|
| NLIngram | 0, 25, 50, 75, 100 | 5个 |
| FBIngram | 25, 50, 75, 100 | 4个 |
| WKIngram | 25, 50, 75, 100 | 4个 |

版本号表示新关系的百分比。

#### 3.2 MTDEA WikiTopics Datasets (8个)

| 数据集 | 版本 | 数量 |
|--------|------|------|
| WikiTopicsMT1 | health, tax | 2个 |
| WikiTopicsMT2 | org, sci | 2个 |
| WikiTopicsMT3 | art, infra | 2个 |
| WikiTopicsMT4 | sci, health | 2个 |

**注意**: WikiTopics 没有 mt/mt2/mt3/mt4 这些版本，只有具体的主题版本。

#### 3.3 MTDEA Other Datasets (2个)

| 数据集 | 版本 | 数量 |
|--------|------|------|
| Metafam | Metafam | 1个 |
| FBNELL | FBNELL_v1 | 1个 |

## 统计汇总

```
Transductive:                    16个
  - FB15k237系列:                 4个
  - 其他:                        12个
Inductive (no new relations):    18个
  - GraIL:                       12个
  - ILPC:                         2个
  - HM:                           4个
Inductive (new relations):       23个
  - Ingram:                      13个
  - WikiTopics:                   8个
  - Other:                        2个
─────────────────────────────────────
总计:                            57个
```

## 使用方法

### 加载所有数据集

```yaml
dataset:
  class: JointDataset
  graphs: all
  root: /path/to/kg-datasets
```

### 加载特定类型

```yaml
# 只加载 transductive 数据集（需要手动列出）
dataset:
  class: JointDataset
  graphs: [FB15k237, WN18RR, CoDExSmall, CoDExMedium, CoDExLarge, NELL995, ConceptNet100k, DBpedia100k, YAGO310, AristoV4, Hetionet, WDsinger, NELL23k]
  root: /path/to/kg-datasets
```

### 加载特定数据集的特定版本

```yaml
dataset:
  class: JointDataset
  graphs: [FB15k237Inductive:v1, ILPC2022:small, HM:1k]
  root: /path/to/kg-datasets
```

## 注意事项

1. **Atlas 数据集被排除**: 使用 `graphs: all` 时不会加载 Atlas 数据集，需要显式指定如 `Atlas:small` 或 `Atlas:one_hop`
2. **自动版本扩展**: 如果只指定数据集名称（如 `FB15k237Inductive`），会自动扩展为所有版本（v1, v2, v3, v4）
3. **数据集分类**: run_many.py 会自动将数据集分类为 transductive 或 inductive，并分别计算平均指标
