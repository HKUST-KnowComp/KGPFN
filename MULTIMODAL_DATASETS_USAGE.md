# Multimodal KG Datasets Usage Guide

This document shows how to use the three multimodal knowledge graph datasets: DB15K-mm, FB15K-mm, and YAGO15K-mm.

## Dataset Overview

| Dataset | Entities | Relations | Triples | Images | Numerical Attrs | Special Features |
|---------|----------|-----------|---------|--------|-----------------|------------------|
| DB15K-mm | 12,842 | 469 | 99,028 | 12,837 | 39,743 | DBpedia URIs |
| FB15K-mm | 14,951 | 1,461 | 592,213 | 13,444 | 29,395 | **Freebase MID → Name mapping** |
| YAGO15K-mm | 15,404 | 39 | 122,886 | 11,194 | 22,178 | YAGO URIs |

## Quick Start

### 1. Load a Dataset

```python
from pfn.datasets import DB15KMM, FB15KMM, YAGO15KMM

# Load DB15K-mm
dataset = DB15KMM(root="./datasets")

# Or load FB15K-mm (with automatic MID to name mapping)
dataset = FB15KMM(root="./datasets")
# Loaded 3,306,418 FB MID to name mappings

# Or load YAGO15K-mm
dataset = YAGO15KMM(root="./datasets")
```

### 2. Access Train/Valid/Test Splits

```python
# Get splits
train_data = dataset[0][0]
valid_data = dataset[1][0]
test_data = dataset[2][0]

# Check statistics
print(f"Entities: {train_data.num_nodes}")
print(f"Relations: {train_data.num_relations}")
print(f"Train triples: {train_data.target_edge_index.shape[1]}")
print(f"Valid triples: {valid_data.target_edge_index.shape[1]}")
print(f"Test triples: {test_data.target_edge_index.shape[1]}")
```

### 3. Load Image Embeddings

```python
# Get metadata
metadata = dataset.get_metadata()
entity_to_image = metadata['entity_to_image']

# Load embeddings for specific entities
entity_ids = [0, 1, 2]  # Entity IDs
embeddings = dataset.load_image_embeddings(entity_ids)

for ent_id, emb in embeddings.items():
    print(f"Entity {ent_id}: shape={emb.shape}, dtype={emb.dtype}")
    # emb is a torch.Tensor of shape (1, 4096)
```

### 4. Access Numerical Attributes

```python
# Get all numerical attributes
numerical_data = dataset.get_numerical_attributes()

for attr in numerical_data[:5]:
    entity_id = attr['entity']
    relation_id = attr['relation']
    value = attr['value']
    print(f"Entity {entity_id}, Relation {relation_id}: {value}")
```

### 5. Access Entity and Relation Vocabularies

```python
metadata = dataset.get_metadata()

# Entity vocabulary (URI -> ID)
inv_entity_vocab = metadata['inv_entity_vocab']
entity_vocab = {v: k for k, v in inv_entity_vocab.items()}

# Relation vocabulary (URI -> ID)
inv_rel_vocab = metadata['inv_rel_vocab']
relation_vocab = {v: k for k, v in inv_rel_vocab.items()}

# Example: get entity URI by ID
entity_uri = entity_vocab[0]
print(f"Entity 0: {entity_uri}")
```

### 6. FB15K-mm: Get Human-Readable Names

**FB15K-mm has a special feature**: it automatically loads a mapping from Freebase MIDs to human-readable names.

```python
from pfn.datasets import FB15KMM

# Load FB15K-mm
dataset = FB15KMM(root="./datasets")
# Output: Loaded 3,306,418 FB MID to name mappings

# Get human-readable name for an entity
entity_name = dataset.get_entity_name('/m/06rf7')
print(entity_name)  # Output: "Schleswig–Holstein"

# Or by entity ID
entity_name = dataset.get_entity_name(0)
print(entity_name)  # Output: "U.S.-attempted annexation of the Dominican Republic"
```

**Example output with names:**
```
Sample Entities:
  [0] U.S.-attempted annexation of the Dominican Republic (/m/027rn)
  [1] Schleswig–Holstein (/m/06rf7)
  [2] Sam fuller (/m/04258w)

Sample Numerical Attributes:
  Entity: Schleswig–Holstein (/m/06rf7)
  Relation: location.geocode.longitude
  Value: 9.70404945
```

## Data Format

### Graph Structure
- `edge_index`: Edge indices (2 x num_edges)
- `edge_type`: Edge types/relations (num_edges)
- `target_edge_index`: Target edges for prediction (2 x num_targets)
- `target_edge_type`: Target edge types (num_targets)
- `num_nodes`: Number of entities
- `num_relations`: Number of relations (x2 for inverse relations)

### Image Embeddings
- Stored in HDF5 format at `/data/gaoyisen/mmkb/{DATASET}/{DATASET}_ImageData.h5`
- Each embedding: shape (1, 4096), dtype float32
- Pre-extracted using ResNet or similar CNN

### Numerical Attributes
- Dictionary format: `{'entity': int, 'relation': int, 'value': float}`
- Includes attributes like population, area, coordinates, dates, etc.
- **One entity can have multiple numerical attributes** (1-to-many relationship)

### Entity Naming Conventions

**DB15K-mm (DBpedia):**
- Format: `<http://dbpedia.org/resource/Entity_Name>`
- Cleaned: `Anarchism`, `Alabama`, `The_Last_of_the_Mohicans_(1992_film)`

**FB15K-mm (Freebase):**
- Format: `/m/xxxxx` (Freebase MID)
- With mapping: `Schleswig–Holstein (/m/06rf7)`
- Mapping file: `/data/gaoyisen/pfn/fb_mid2name.tsv` (3.3M+ mappings)

**YAGO15K-mm (YAGO):**
- Format: `<http://yago-knowledge.org/resource/Entity_Name>`
- Cleaned: `Margrethe_II_of_Denmark`, `Greenland`, `Denmark`

## Testing

Run the test script to verify all datasets:

```bash
conda activate limix
python test_multimodal_datasets.py
```

Expected output:
```
================================================================================
  Test Summary
================================================================================
  DB15K-mm: ✓ PASSED
  FB15K-mm: ✓ PASSED
  YAGO15K-mm: ✓ PASSED

🎉 All tests passed!
```

## Data Source

Original data location: `/data/gaoyisen/mmkb/`

Each dataset contains:
- `{DATASET}_EntityTriples.txt` - Knowledge graph triples
- `{DATASET}_ImageIndex.txt` - Entity to image ID mapping
- `{DATASET}_ImageData.h5` - Pre-extracted image embeddings (4096-dim)
- `{DATASET}_NumericalTriples.txt` - Numerical attribute triples

**FB15K additional file:**
- `/data/gaoyisen/pfn/fb_mid2name.tsv` - Freebase MID to name mapping (7.6M lines)

## Implementation Details

The datasets are implemented in `pfn/datasets.py`:
- `MultiModalKGDataset` - Base class for multimodal KG datasets
- `DB15KMM` - DB15K multimodal dataset
- `FB15KMM` - FB15K multimodal dataset (with MID to name mapping)
- `YAGO15KMM` - YAGO15K multimodal dataset

All datasets inherit from `torch_geometric.data.InMemoryDataset` and support:
- Automatic data processing and caching
- Train/valid/test split (80/10/10)
- Lazy loading of image embeddings
- Metadata access for vocabularies and mappings
- **FB15K**: Automatic loading of 3.3M+ MID to name mappings
