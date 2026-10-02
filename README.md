# KGPFN: Unlocking the Potential of Knowledge Graph Foundation Model via In-Context Learning

Knowledge graph link prediction framework combining:
- **Structure encoder**:  relational message passing
- **Feature transformer**: LimiX or TabICL

## Quick Start

### 1. Environment Setup

```bash
conda create -n kgpfn python=3.12
conda activate kgpfn
pip install -r requirements.txt
```

**Flash Attention** (required for TabICL/LimiX):

Download the prebuilt wheel matching your CUDA/PyTorch version from the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases), then install:

```bash
# Example for CUDA 12.6 + PyTorch 2.7
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### 2. Download Pretrained Models

Download structure encoder and feature transformer checkpoints if you want to retrain KGPFN:

```bash
# Download with TabICL as feature transformer (default)
python script/download.py --ft tabicl

# Download with LimiX instead
python script/download.py --ft limix
```

Or directly download a fully pretrained KGPFN model checkpoint:

```bash
# tabicl trained on the ULTRA 50g structure encoder (default)
python script/download.py --kgpfn

# tabicl trained with the ULTRA 3g structure encoder on 3 kg datasets
python script/download.py --kgpfn icl_3g

# limix as the pfn architecture
python script/download.py --kgpfn limix

# tabicl as the pfn architecture with semantic encoder of all-MiniLM-L12-v2
python script/download.py --kgpfn iclsemantic
```

All files will be saved to `./cache/` directory.

### 3. Configure Dataset Path

Edit `config/script/train_all.yaml` and set your dataset root:

```yaml
dataset:
  root: /path/to/your/kg-datasets
```

### 4. Training


**Multi-GPU training:**

```bash
accelerate launch --num_processes 8 script/pretrain_pfn.py -c config/script/train_all.yaml --gpus [0,1,2,3,4,5,6,7]
```

### 5. Testing

```bash
CUDA_VISIBLE_DEVICES=0 python script/test_kgpfn.py \
  -c config/script/test.yaml \
  --gpus [0]
```
