import os
import sys
import ast
import copy
import time
import logging
import argparse
import shutil

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.data import Data
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR

from . import datasets


logger = logging.getLogger(__file__)


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def apply_model_config(cfg, project_root=None):
    """Optionally overlay ``cfg.model`` from a separate YAML file.

    The override file may contain a top-level ``model`` mapping (the preferred
    format) or the model mapping itself. Existing model values are retained
    when the override only specifies a subset of fields.
    """
    config_path = cfg.get("model_config_path", None)
    if not config_path:
        return cfg

    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(project_root, config_path))

    with open(config_path, "r", encoding="utf-8") as fin:
        override = yaml.safe_load(fin) or {}
    if not isinstance(override, dict):
        raise TypeError(f"model_config_path must contain a YAML mapping: {config_path}")
    model_override = override.get("model", override)
    if not isinstance(model_override, dict):
        raise TypeError(f"model_config_path.model must be a YAML mapping: {config_path}")

    def merge_dict(base, extra):
        merged = dict(base or {})
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    cfg.model = easydict.EasyDict(merge_dict(dict(cfg.get("model", {})), model_override))
    logger.info("Loaded model configuration override from %s", config_path)
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=0)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg, chdir=True):
    """
    创建并返回 working_dir。chdir=True 时切换当前目录到 working_dir（兼容旧行为）；
    chdir=False 时保持 cwd 不变，便于 dataset.root 等相对路径始终基于项目根目录解析。
    """
    file_name = "working_dir.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.abspath(os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S")))

    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    if chdir:
        os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    data_config = copy.deepcopy(cfg.dataset)
    cls = data_config.pop("class")
    version = data_config.pop("version", None)

    # 解析 dataset.root 相对项目根目录（chdir 后相对路径会错误解析到 working_dir）
    if "root" in data_config and data_config["root"] and not os.path.isabs(data_config["root"]):
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_config["root"] = os.path.abspath(os.path.join(_project_root, data_config["root"]))

    if(cls == "JointDataset"):
        ds_cls = getattr(datasets, cls)
        dataset = ds_cls(**data_config)
    else:
        ds_cls = getattr(datasets, cls)
        dataset = ds_cls(**data_config, dataset_name=cls, dataset_version=version)

    # Rebuild stale processed cache once if name maps are missing.
    # This typically happens when old processed files were generated before
    # train_id2entity / train_id2relation were added.
    if cls != "JointDataset":
        train_data = dataset[0]
        has_e = bool(getattr(train_data, "train_id2entity", {}))
        has_r = bool(getattr(train_data, "train_id2relation", {}))
        if not (has_e and has_r):
            processed_dir = getattr(dataset, "processed_dir", None)
            if processed_dir and os.path.isdir(processed_dir):
                logger.warning("Name maps missing in processed cache, rebuilding: %s", processed_dir)
                shutil.rmtree(processed_dir, ignore_errors=True)
                dataset = ds_cls(**data_config, dataset_name=cls, dataset_version=version)

    if get_rank() == 0:
        logger.warning("%s dataset" % (cls if "version" not in cfg.dataset else f'{cls}({cfg.dataset.version})'))
        if cls != "JointDataset":
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                        (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                            dataset[2].target_edge_index.shape[1]))
        else:
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                           (sum(d.target_edge_index.shape[1] for d in dataset._data[0]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[1]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[2]),
                            ))

    return dataset
