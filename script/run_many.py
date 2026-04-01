import os
import sys
import csv
import math
import time
import pprint
import argparse
import random

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pfn import util, datasets as pfn_datasets
from script.run import train_and_validate, test


default_finetuning_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (1, 4000),
    "CoDExMedium": (1, 4000),
    "CoDExLarge": (1, 2000),
    "FB15k237": (1, 'null'),
    "WN18RR": (1, 'null'),
    "YAGO310": (1, 2000),
    "DBpedia100k": (1, 1000),
    "AristoV4": (1, 2000),
    "ConceptNet100k": (1, 2000),
    # tail-only datasets (2)
    "NELL995": (1, 'null'),  # not implemented yet
    "Hetionet": (1, 4000),
    # sparse datasets (5)
    "WDsinger": (3, 'null'),
    "FB15k237_10": (1, 'null'),
    "FB15k237_20": (1, 'null'),
    "FB15k237_50": (1, 1000),
    "NELL23k": (3, 'null'),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (1, 'null'),    # for all 4 datasets
    "WN18RRInductive": (1, 'null'),      # for all 4 datasets
    "NELLInductive": (3, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (3, 'null'),
    "ILPC2022LargeInductive": (1, 1000),
    # Ingram datasets (13)
    "NLIngram": (3, 'null'),  # for all 5 datasets
    "FBIngram": (3, 'null'),  # for all 4 datasets
    "WKIngram": (3, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (3, 'null'),  # for all 2 test datasets
    "Metafam": (3, 'null'),
    "FBNELL": (3, 'null'),
    # Hamaguchi datasets (4)
    "HM": (1, 100)  # for all 4 datasets
}

default_train_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (10, 1000),
    "CoDExMedium": (10, 1000),
    "CoDExLarge": (10, 1000),
    "FB15k237": (10, 1000),
    "WN18RR": (10, 1000),
    "YAGO310": (10, 2000),
    "DBpedia100k": (10, 1000),
    "AristoV4": (10, 1000),
    "ConceptNet100k": (10, 1000),
    "ATOMIC": (10, 1000),
    # tail-only datasets (2)
    "NELL995": (10, 1000),  # not implemented yet
    "Hetionet": (10, 1000),
    # sparse datasets (5)
    "WDsinger": (10, 1000),
    "FB15k237_10": (10, 1000),
    "FB15k237_20": (10, 1000),
    "FB15k237_50": (10, 1000),
    "NELL23k": (10, 1000),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (10, 'null'),    # for all 4 datasets
    "WN18RRInductive": (10, 'null'),      # for all 4 datasets
    "NELLInductive": (10, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (10, 'null'),
    "ILPC2022LargeInductive": (10, 1000),
    # Ingram datasets (13)
    "NLIngram": (10, 'null'),  # for all 5 datasets
    "FBIngram": (10, 'null'),  # for all 4 datasets
    "WKIngram": (10, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (10, 'null'),  # for all 2 test datasets
    "Metafam": (10, 'null'),
    "FBNELL": (10, 'null'),
    # Hamaguchi datasets (4)
    "HM": (10, 1000)  # for all 4 datasets
}


separator = ">" * 30
line = "-" * 30


def _resolve_name_maps(data_obj):
    id2e = getattr(data_obj, "train_id2entity", getattr(data_obj, "id2entity", {}))
    id2r = getattr(data_obj, "train_id2relation", getattr(data_obj, "id2relation", {}))
    return id2e, id2r


def _debug_name_maps(train_data, valid_data, test_data, id2e, id2r, dataset_name=""):
    """遍历图中所有实体/关系 ID，检查 id2e/id2r 是否覆盖，打印缺失项及图统计信息。"""
    num_nodes = int(train_data.num_nodes)
    num_relations = int(getattr(train_data, "num_relations", 0)) or int(train_data.target_edge_type.max()) + 1
    base_rel = num_relations // 2

    # 收集图中出现的所有 entity / relation ID
    def _collect_ids(data):
        eids, rids = set(), set()
        if hasattr(data, "edge_index") and data.edge_index is not None:
            eids.update(data.edge_index[0].tolist() + data.edge_index[1].tolist())
        if hasattr(data, "target_edge_index"):
            eids.update(data.target_edge_index[0].tolist() + data.target_edge_index[1].tolist())
        if hasattr(data, "edge_type") and data.edge_type is not None:
            rids.update(data.edge_type.tolist())
        if hasattr(data, "target_edge_type"):
            rids.update(data.target_edge_type.tolist())
        return eids, rids

    train_e, train_r = _collect_ids(train_data)
    valid_e, valid_r = _collect_ids(valid_data)
    test_e, test_r = _collect_ids(test_data)
    all_entity_ids = train_e | valid_e | test_e
    all_rel_ids = train_r | valid_r | test_r

    # 图统计
    print(f"\n========== {dataset_name} 图统计 ==========")
    print(f"  num_nodes: {num_nodes}")
    print(f"  num_relations (from edge_type): {num_relations}, base_rel: {base_rel}")
    print(f"  train edges: {train_data.target_edge_index.shape[1]}, valid: {valid_data.target_edge_index.shape[1]}, test: {test_data.target_edge_index.shape[1]}")
    e_range = f"[{min(all_entity_ids)}, {max(all_entity_ids)}]" if all_entity_ids else "[]"
    r_range = f"[{min(all_rel_ids)}, {max(all_rel_ids)}]" if all_rel_ids else "[]"
    print(f"  图中出现的 entity ID 数量: {len(all_entity_ids)}, 范围: {e_range}")
    print(f"  图中出现的 relation ID 数量: {len(all_rel_ids)}, 范围: {r_range}")
    print(f"  id2entity 条目数: {len(id2e)}, id2relation 条目数: {len(id2r)}")

    # 检查 entity 覆盖
    missing_entities = [e for e in sorted(all_entity_ids) if e not in id2e]
    if missing_entities:
        print(f"\n  [MISSING] 以下 entity ID 在 id2entity 中找不到:")
        for e in missing_entities[:20]:
            print(f"    entity id={e}")
        if len(missing_entities) > 20:
            print(f"    ... 共 {len(missing_entities)} 个")
    else:
        print(f"\n  [OK] 所有 entity ID 均在 id2entity 中有映射")

    # 检查 relation 覆盖（含逆关系）
    def _can_resolve_rel(r):
        if r in id2r:
            return True
        if base_rel > 0 and r >= base_rel and (r - base_rel) in id2r:
            return True
        return False

    missing_relations = [r for r in sorted(all_rel_ids) if not _can_resolve_rel(r)]
    if missing_relations:
        print(f"\n  [MISSING] 以下 relation ID 在 id2relation 中找不到（含逆关系 base_rel={base_rel}）:")
        for r in missing_relations[:30]:
            base_id = r - base_rel if base_rel > 0 and r >= base_rel else r
            print(f"    relation id={r} (base_id={base_id}, base_id in id2r: {base_id in id2r})")
        if len(missing_relations) > 30:
            print(f"    ... 共 {len(missing_relations)} 个")
    else:
        print(f"\n  [OK] 所有 relation ID 均可解析（直接或逆关系）")

    print("=" * 50)


def _expand_dataset_variants(dataset_keys):
    """
    Expand base dataset keys to concrete dataset specs.
    Example: FB15k237Inductive -> FB15k237Inductive:v1..v4
    """
    version_map = {
        "FB15k237Inductive": ["v1", "v2", "v3", "v4"],
        "WN18RRInductive": ["v1", "v2", "v3", "v4"],
        "NELLInductive": ["v1", "v2", "v3", "v4"],
        "NLIngram": ["0", "25", "50", "75", "100"],
        "FBIngram": ["25", "50", "75", "100"],
        "WKIngram": ["25", "50", "75", "100"],
        "WikiTopicsMT1": ["tax", "health"],
        "WikiTopicsMT2": ["org", "sci"],
        "WikiTopicsMT3": ["art", "infra"],
        "WikiTopicsMT4": ["sci", "health"],
        "HM": ["1k", "3k", "5k", "indigo"],
    }
    expanded = []
    for ds in dataset_keys:
        versions = version_map.get(ds)
        if versions:
            expanded.extend([f"{ds}:{v}" for v in versions])
        else:
            expanded.append(ds)
    return expanded


def set_seed(seed):
    random.seed(seed + util.get_rank())
    # np.random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets", default='FB15k237Inductive:v1,NELLInductive:v4', type=str, required=True)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("-tr", "--train", help="train the model from scratch", action='store_true')
    parser.add_argument("--start_from", type=str, default='FB15k237Inductive:v1', help="start loading from this dataset key, e.g. NELL23k")
    args, unparsed = parser.parse_known_args()

    # 解析一次 config 中的动态变量（例如 ckpt、gpus 等），供所有数据集复用
    dyn_vars = util.detect_variables(args.config)
    dyn_parser = argparse.ArgumentParser()
    for var in dyn_vars:
        dyn_parser.add_argument(f"--{var}")
    dyn_vals = dyn_parser.parse_known_args(unparsed)[0]
    base_vars = {k: util.literal_eval(v) for k, v in dyn_vals._get_kwargs()}

    # 如存在 checkpoint，只从磁盘加载一次，后续数据集直接复用内存中的 state
    ckpt_path = base_vars.get("ckpt", None)
    base_state = torch.load(ckpt_path, map_location="cpu") if ckpt_path is not None else None

    dataset_args = args.datasets.split(",")
    path = os.path.dirname(os.path.expanduser(__file__))
    results_file = os.path.join(path, f"ultra_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
    datasets_list = list(default_finetuning_config.keys())
    graphs_to_run = _expand_dataset_variants(datasets_list)
    if args.start_from:
        start_from = args.start_from.strip()
        start_idx = None
        for idx, graph in enumerate(graphs_to_run):
            if graph == start_from or graph.split(":")[0] == start_from:
                start_idx = idx
                break
        if start_idx is None:
            raise RuntimeError(f"start_from `{start_from}` not found in expanded dataset list")
        graphs_to_run = graphs_to_run[start_idx:]
        print(f"Start from {graphs_to_run[0]}, remaining {len(graphs_to_run)} datasets")
    graphs_to_run = ['FB15k237', 'WN18RR', 'CoDExMedium']
    for graph in graphs_to_run:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        if not hasattr(pfn_datasets, ds):
            print(f"Skipping {graph}: pfn.datasets has no class `{ds}`")
            continue
        for i in range(args.repeats):
            seed = seeds[i] if i < len(seeds) else random.randint(0, 10000)
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            # 针对当前数据集构造 vars，在全局 base_vars 基础上覆盖 dataset / epochs / bpe / version
            vars = dict(base_vars)
            if args.finetune:
                epochs, batch_per_epoch = default_finetuning_config[ds] 
            elif args.train:
                epochs, batch_per_epoch = default_train_config[ds] 
            else:
                epochs, batch_per_epoch = 0, 'null'
            vars['epochs'] = epochs
            vars['bpe'] = batch_per_epoch
            vars['dataset'] = ds
            if version is not None:
                vars['version'] = version
            cfg = util.load_config(args.config, context=vars)
            # Some config templates do not expose {{ version }}.
            # Force-inject dataset version parsed from `DatasetName:version`.
            if version is not None:
                cfg.dataset["version"] = version

            root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
            os.makedirs(root_dir, exist_ok=True)
            # download-only mode: do not create timestamped working directories
            # (avoids FileExistsError when multiple runs start in same second)

            # logger = util.get_root_logger()
            # if util.get_rank() == 0:
            #     logger.warning("Random seed: %d" % seed)
            #     logger.warning("Config file: %s" % args.config)
            #     logger.warning(pprint.pformat(cfg))
            
            task_name = cfg.task["name"]
            dataset = util.build_dataset(cfg)
            device = util.get_device(cfg)
            
            train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
            id2e, id2r = _resolve_name_maps(train_data)
            _debug_name_maps(train_data, valid_data, test_data, id2e, id2r, dataset_name=graph)
            
            break
            # # 针对当前实验实例化模型，并用预加载的 checkpoint 初始化（如有）
            # model = Ultra(
            #     rel_model_cfg=cfg.model.relation_model,
            #     entity_model_cfg=cfg.model.entity_model,
            # )
            # if base_state is not None:
            #     model.load_state_dict(base_state["model"])
            # model = model.to(device)
            
            # if task_name == "InductiveInference":
            #     # filtering for inductive datasets
            #     # Grail, MTDEA, HM datasets have validation sets based off the training graph
            #     # ILPC, Ingram have validation sets from the inference graph
            #     # filtering dataset should contain all true edges (base graph + (valid) + test) 
            #     if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            #         # add inference, valid, test as the validation and test filtering graphs
            #         full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            #         full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            #         test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            #         val_filtered_data = test_filtered_data
            #     else:
            #         # test filtering graph: inference edges + test edges
            #         full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            #         full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            #         test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            #         # validation filtering graph: train edges + validation edges
            #         val_filtered_data = Data(
            #             edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
            #             edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            #         )
            #     #test_filtered_data = val_filtered_data = None
            # else:
            #     # for transductive setting, use the whole graph for filtered ranking
            #     filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
            #     val_filtered_data = test_filtered_data = filtered_data
            
            # val_filtered_data = val_filtered_data.to(device)
            # test_filtered_data = test_filtered_data.to(device)
            
            # train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            # if util.get_rank() == 0:
            #     logger.warning(separator)
            #     logger.warning("Evaluate on valid")
            # test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            # if util.get_rank() == 0:
            #     logger.warning(separator)
            #     logger.warning("Evaluate on test")
            # metrics = test(cfg, model, test_data, filtered_data=test_filtered_data, return_metrics=True, device=device, logger=logger)

            # metrics = {k:v.item() for k,v in metrics.items()}
            # metrics['dataset'] = graph
            # # write to the log file
            # with open(results_file, "a", newline='') as csv_file:
            #     fieldnames = ['dataset']+list(metrics.keys())[:-1]
            #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
            #     if csv_file.tell() == 0:
            #         writer.writeheader()
            #     writer.writerow(metrics)