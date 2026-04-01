import os
import sys
import math
import pprint
from itertools import islice

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.kgpfn import KGPFN
from model.ultra.encoder import StructureEncoder
from pfn import tasks, util
from huggingface_hub import hf_hub_download

separator = ">" * 30
line = "-" * 30


def build_id_name_maps_from_ds(ds):
    """
    从具体数据集对象重建实体/关系词表映射。
    返回:
      entity_id2name: dict[int, str]
      rel_id2name: dict[int, str]
    """
    if not hasattr(ds, "load_file") or not hasattr(ds, "raw_paths"):
        return {}, {}
    train_file = ds.raw_paths[0]
    res = ds.load_file(train_file, inv_entity_vocab={}, inv_rel_vocab={})
    inv_entity_vocab = res.get("inv_entity_vocab", {})
    inv_rel_vocab = res.get("inv_rel_vocab", {})
    entity_id2name = {v: k for k, v in inv_entity_vocab.items()}
    rel_id2name = {v: k for k, v in inv_rel_vocab.items()}
    return entity_id2name, rel_id2name


def print_mapping_examples(dataset, train_data):
    """
    打印 id->name 映射样例，验证实体/关系名是否可恢复。
    支持单图数据集和 JointDataset。
    """
    print(separator)
    print("Mapping examples")
    print(separator)

    # JointDataset: dataset.graphs 存放多个子数据集
    if hasattr(dataset, "graphs"):
        for ds in dataset.graphs[:3]:
            entity_id2name, rel_id2name = build_id_name_maps_from_ds(ds)
            print(f"[{ds.__class__.__name__}] #entity={len(entity_id2name)} #relation={len(rel_id2name)}")
            for eid in sorted(entity_id2name)[:3]:
                print(f"  entity[{eid}] -> {entity_id2name[eid]}")
            for rid in sorted(rel_id2name)[:3]:
                print(f"  relation[{rid}] -> {rel_id2name[rid]}")
        return

    # 单图数据集
    entity_id2name, rel_id2name = build_id_name_maps_from_ds(dataset)
    print(f"[{dataset.__class__.__name__}] #entity={len(entity_id2name)} #relation={len(rel_id2name)}")
    for eid in sorted(entity_id2name)[:5]:
        print(f"  entity[{eid}] -> {entity_id2name[eid]}")
    for rid in sorted(rel_id2name)[:5]:
        print(f"  relation[{rid}] -> {rel_id2name[rid]}")

    # 再打印一个 train 三元组映射示例
    h = int(train_data.target_edge_index[0, 0].item())
    t = int(train_data.target_edge_index[1, 0].item())
    r = int(train_data.target_edge_type[0].item())
    h_name = entity_id2name.get(h, f"<UNK_ENTITY_{h}>")
    t_name = entity_id2name.get(t, f"<UNK_ENTITY_{t}>")
    r_name = rel_id2name.get(r, f"<UNK_REL_{r}>")
    print(f"  sample triple ids: (h={h}, t={t}, r={r})")
    print(f"  sample triple names: ({h_name}, {r_name}, {t_name})")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    print_mapping_examples(dataset, train_data)
   