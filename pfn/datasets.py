import os
import csv
import shutil
import json
import time
import hashlib
import torch
import yaml
import re
import requests
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import index_sort
from itertools import chain
import os.path as osp
from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor

from .tasks import build_relation_graph

mydir = os.getcwd()


edge2id_WN18RR = {
    '_also_see': 0,
    '_derivationally_related_form': 1,
    '_has_part': 2,
    '_hypernym': 3,
    '_instance_hypernym': 4,
    '_member_meronym': 5,
    '_member_of_domain_region': 6,
    '_member_of_domain_usage': 7,
    '_similar_to': 8,
    '_synset_domain_topic_of': 9,
    '_verb_group': 10,
}
fb_id2entity = {}
fb_id2relation = {}
wn18rr_id2relation = {}
wn18rr_id2entity = {}

def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    headers = {
        # Wikidata 要求设置明确的 User-Agent
        "User-Agent": "semma-kg-loader/1.0 (research use; contact: gaoyisen5@gmail.com)"
    }
    max_retries = 4
    timeout_sec = 10
    base_sleep = 1.0
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout_sec)
            # 429 / 5xx 视为可重试错误
            if response.status_code == 200:
                return response
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            sleep_s = max(float(retry_after), base_sleep)
                        except ValueError:
                            sleep_s = base_sleep * (2 ** attempt)
                    else:
                        sleep_s = base_sleep * (2 ** attempt)
                    time.sleep(sleep_s)
                    continue
            return None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"An error occurred: {e}")
                return None
            time.sleep(base_sleep * (2 ** attempt))
    return None

def get_entities(entity_ids):
    params = {
        'action': 'wbgetentities',
        'ids': '|'.join(entity_ids),
        'format': 'json',
        'languages': 'en'
    }
    response = fetch_wikidata(params)
    if response:
        data = response.json()
        entities_dict = {}
        for entity_id in entity_ids:
            if entity_id in data['entities']:
                entity = data['entities'][entity_id]
                if 'labels' in entity and 'en' in entity['labels']:
                    entities_dict[entity_id] = entity['labels']['en']['value']
                else:
                    entities_dict[entity_id] = "No English label found"
            else:
                entities_dict[entity_id] = "Entity not found"
        return entities_dict
    else:
        return {entity_id: "Failed to retrieve data" for entity_id in entity_ids}

def get_properties(property_ids):
    params = {
        'action': 'wbgetentities',
        'ids': '|'.join(property_ids),
        'format': 'json',
        'languages': 'en'
    }
    response = fetch_wikidata(params)
    if response:
        data = response.json()
        properties_dict = {}
        for property_id in property_ids:
            if property_id in data['entities']:
                property_data = data['entities'][property_id]
                if 'labels' in property_data and 'en' in property_data['labels']:
                    properties_dict[property_id] = property_data['labels']['en']['value']
                else:
                    properties_dict[property_id] = "No English label found"
            else:
                properties_dict[property_id] = "Property not found"
        return properties_dict
    else:
        return {property_id: "Failed to retrieve data" for property_id in property_ids}

def fetch_in_parallel(ids_list, fetch_func):
    batch_size = 10  # 降低单次请求量，减少 429/超时概率
    batches = [ids_list[i:i + batch_size] for i in range(0, len(ids_list), batch_size)]
    results = {}
    
    # 低并发，减少被限流概率
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_func, batch): batch for batch in batches}
        for future in futures:
            batch = futures[future]
            # 失败时不回退原始 qid，统一标记为失败
            try:
                batch_results = future.result()
                results.update(batch_results)
            
            except Exception as e:
                for qid in batch:
                    results[qid] = "Failed to retrieve data"
    
    for key, value in results.items():
        if value == "parent organization/unit":
            results[key] = "parent organization"
        elif value == "child organization/unit":
            results[key] = "has subsidiary"
        elif value == "participated in conflict":
            results[key] = "conflict"

    return results

class GrailInductiveDataset(InMemoryDataset):

    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, merge_valid_test=True, **kwargs):
        self.version = version
        assert version in ["v1", "v2", "v3", "v4"]

       
        self.dataset_name = kwargs['dataset_name']
        self.dataset_version = kwargs['dataset_version']

        # by default, most models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        # you'll need to delete the processed datasets then and re-run to cache a new dataset
        self.merge_valid_test = merge_valid_test
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "grail", self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "valid_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:3]
        train_files = self.raw_paths[3:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        # creating fact graphs - those are graphs sent to a model, based on which we'll predict missing facts
        # also, those fact graphs will be used for filtered evaluation
        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]

        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        # by default, SOTA models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        test_slice = slice(sum(num_samples[:3]), sum(num_samples)) if self.merge_valid_test else slice(sum(num_samples[:4]), sum(num_samples))
        
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice], num_relations=num_relations*2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice], num_relations=num_relations*2)
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice], num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name
        
        train_data.dataset=current_dataset
        valid_data.dataset=current_dataset
        test_data.dataset=current_dataset
        train_data.id2entity = {value: key for key, value in inv_train_entity_vocab.items()}
        train_data.id2relation = {value: key for key, value in inv_relation_vocab.items()}
        valid_data.id2entity = {value: key for key, value in inv_train_entity_vocab.items()}
        valid_data.id2relation = {value: key for key, value in inv_relation_vocab.items()}
        test_data.id2entity = {value: key for key, value in inv_test_entity_vocab.items()}
        test_data.id2relation = {value: key for key, value in inv_relation_vocab.items()}
        train_data.edge2id = {key: value for key, value in inv_relation_vocab.items()}
        valid_data.edge2id = {key: value for key, value in inv_relation_vocab.items()}
        test_data.edge2id = {key: value for key, value in inv_relation_vocab.items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)

class FB15k237Inductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
    ]

    name = "IndFB15k237"
    versions = ["v1", "v2", "v3", "v4"]

    def __init__(self, root, **kwargs):
        super().__init__(root, version = kwargs['dataset_version'], **kwargs)

    def process(self):
        test_files = self.raw_paths[:3]
        train_files = self.raw_paths[3:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        # creating fact graphs - those are graphs sent to a model, based on which we'll predict missing facts
        # also, those fact graphs will be used for filtered evaluation
        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]

        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        # by default, SOTA models on Grail datasets merge inductive valid and test splits as the final test split
        # with this choice, the validation set is that of the transductive train (on the seen graph)
        # by default it's turned on but you can experiment with turning this option off
        test_slice = slice(sum(num_samples[:3]), sum(num_samples)) if self.merge_valid_test else slice(sum(num_samples[:4]), sum(num_samples))
        
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice], num_relations=num_relations*2)
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice], num_relations=num_relations*2)
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice], num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name
        
        fb_id2entity_train = {value: key for key, value in inv_train_entity_vocab.items()}
        fb_id2entity_test = {value: key for key, value in inv_test_entity_vocab.items()}
        # load fb_mid2name.tsv in a dictionary
        entities_dict = {}
        file_path_here = os.path.join(mydir, "fb_mid2name.tsv")
        with open(file_path_here) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: value for key, value in lines}

        for key, value in fb_id2entity_train.items():
            if value in entities_dict:
                fb_id2entity_train[key] = entities_dict[value]

        for key, value in fb_id2entity_test.items():
            if value in entities_dict:
                fb_id2entity_test[key] = entities_dict[value]

        entities_dict.clear()

        train_data.dataset=current_dataset
        valid_data.dataset=current_dataset
        test_data.dataset=current_dataset
        train_data.id2entity = fb_id2entity_train
        train_data.id2relation = {value: key for key, value in inv_relation_vocab.items()}
        valid_data.id2entity = fb_id2entity_train
        valid_data.id2relation = {value: key for key, value in inv_relation_vocab.items()}
        test_data.id2entity =fb_id2entity_test
        test_data.id2relation = {value: key for key, value in inv_relation_vocab.items()}
        train_data.edge2id = {key: value for key, value in inv_relation_vocab.items()}
        valid_data.edge2id = {key: value for key, value in inv_relation_vocab.items()}
        test_data.edge2id = {key: value for key, value in inv_relation_vocab.items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class WN18RRInductive(GrailInductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
    ]

    name = "IndWN18RR"
    versions = ["v1", "v2", "v3", "v4"]

    def __init__(self, root, **kwargs):
        super().__init__(root, version = kwargs['dataset_version'], **kwargs)

class NELLInductive(GrailInductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/valid.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt"
    ]
    name = "IndNELL"
    versions = ["v1", "v2", "v3", "v4"]

    def __init__(self, root, **kwargs):
        super().__init__(root, version = kwargs['dataset_version'], **kwargs)

class RelLinkPredDataset(InMemoryDataset):
    r"""The relational link prediction datasets from the
    `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by sets of triplets.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"FB15k-237"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 14,541
          - 544,230
          - 0
          - 0
    """

    urls = {
        'FB15k-237': ('https://raw.githubusercontent.com/MichSchli/'
                      'RelationPrediction/master/data/FB-Toutanova')
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name
        assert name in ['FB15k-237']
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def num_relations(self) -> int:
        return int(self._data.edge_type.max()) + 1  # type: ignore

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'entities.dict', 'relations.dict', 'test.txt', 'train.txt',
            'valid.txt'
        ]

    def download(self) -> None:
        for file_name in self.raw_file_names:
            download_url(f'{self.urls[self.name]}/{file_name}', self.raw_dir)

    def process(self) -> None:
        with open(osp.join(self.raw_dir, 'entities.dict')) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(osp.join(self.raw_dir, 'relations.dict')) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        global fb_id2entity
        global fb_id2relation

        fb_id2entity = {value: key for key, value in entities_dict.items()}
        fb_id2relation = {value: key for key, value in relations_dict.items()}

        kwargs = {}
        for split in ['train', 'valid', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.txt')) as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                kwargs[f'{split}_edge_index'] = torch.tensor([src, dst])
                kwargs[f'{split}_edge_type'] = torch.tensor(rel)

        # For message passing, we add reverse edges and types to the graph:
        row, col = kwargs['train_edge_index']
        edge_type = kwargs['train_edge_type']
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_type = torch.cat([edge_type, edge_type + len(relations_dict)])

        data = Data(num_nodes=len(entities_dict), edge_index=edge_index,
                    edge_type=edge_type, **kwargs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'

class WordNet18RR(InMemoryDataset):
    r"""The WordNet18RR dataset from the `"Convolutional 2D Knowledge Graph
    Embeddings" <https://arxiv.org/abs/1707.01476>`_ paper, containing 40,943
    entities, 11 relations and 93,003 fact triplets.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/WN18RR/original')

    edge2id = {
        '_also_see': 0,
        '_derivationally_related_form': 1,
        '_has_part': 2,
        '_hypernym': 3,
        '_instance_hypernym': 4,
        '_member_meronym': 5,
        '_member_of_domain_region': 6,
        '_member_of_domain_usage': 7,
        '_similar_to': 8,
        '_synset_domain_topic_of': 9,
        '_verb_group': 10,
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self) -> None:
        node2id, idx = {}, 0

        global wn18rr_id2entity
        global wn18rr_id2relation

        srcs, dsts, edge_types = [], [], []
        for path in self.raw_paths:
            with open(path, 'r') as f:
                edges = f.read().split()

                _src = edges[::3]
                _dst = edges[2::3]
                _edge_type = edges[1::3]

                for i in chain(_src, _dst):
                    if i not in node2id:
                        node2id[i] = idx
                        idx += 1

                srcs.append(torch.tensor([node2id[i] for i in _src]))
                dsts.append(torch.tensor([node2id[i] for i in _dst]))
                edge_types.append(
                    torch.tensor([self.edge2id[i] for i in _edge_type]))

        wn18rr_id2entity = {value: key for key, value in node2id.items()}
        wn18rr_id2relation = {value: key for key, value in self.edge2id.items()}

        src = torch.cat(srcs, dim=0)
        dst = torch.cat(dsts, dim=0)
        edge_type = torch.cat(edge_types, dim=0)

        train_mask = torch.zeros(src.size(0), dtype=torch.bool)
        train_mask[:srcs[0].size(0)] = True
        val_mask = torch.zeros(src.size(0), dtype=torch.bool)
        val_mask[srcs[0].size(0):srcs[0].size(0) + srcs[1].size(0)] = True
        test_mask = torch.zeros(src.size(0), dtype=torch.bool)
        test_mask[srcs[0].size(0) + srcs[1].size(0):] = True

        num_nodes = max(int(src.max()), int(dst.max())) + 1
        _, perm = index_sort(num_nodes * src + dst)

        edge_index = torch.stack([src[perm], dst[perm]], dim=0)
        edge_type = edge_type[perm]
        train_mask = train_mask[perm]
        val_mask = val_mask[perm]
        test_mask = test_mask[perm]

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

def FB15k237(root, **kwargs):
    dataset = RelLinkPredDataset(name="FB15k-237", root=root+"/fb15k237/")
    data = dataset.data
    num_relations = int(data.edge_type.max()) + 1
    train_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.train_edge_index, target_edge_type=data.train_edge_type,
                        num_relations=num_relations)
    valid_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.valid_edge_index, target_edge_type=data.valid_edge_type,
                        num_relations=num_relations)
    test_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes,
                        target_edge_index=data.test_edge_index, target_edge_type=data.test_edge_type,
                        num_relations=num_relations)
    
    train_data.dataset="FB15k237"
    valid_data.dataset="FB15k237"
    test_data.dataset="FB15k237"

    # 直接从 raw 文件读取 entity/relation 映射（避免依赖全局变量）
    raw_dir = os.path.join(root, "fb15k237/FB15k-237/raw")
    
    # 读取 entities.dict
    entities_dict = {}
    with open(os.path.join(raw_dir, "entities.dict"), "r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            entities_dict[value] = int(key)
    
    # 读取 relations.dict
    relations_dict = {}
    with open(os.path.join(raw_dir, "relations.dict"), "r") as f:
        for line in f:
            key, value = line.strip().split("\t")
            relations_dict[value] = int(key)
    
    # 构建 id2entity 和 id2relation（反转字典）
    id2entity = {v: k for k, v in entities_dict.items()}
    id2relation = {v: k for k, v in relations_dict.items()}
    
    # 尝试用 fb_mid2name.tsv 增强 entity 名称
    file_path_here = os.path.join(mydir, "fb_mid2name.tsv")
    if os.path.exists(file_path_here):
        with open(file_path_here) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1] if row]
            mid2name = {key: value for key, value in lines}
        for eid, mid in id2entity.items():
            if mid in mid2name:
                id2entity[eid] = mid2name[mid]

    train_data.id2entity = id2entity
    train_data.id2relation = id2relation
    valid_data.id2entity = id2entity
    valid_data.id2relation = id2relation
    test_data.id2entity = id2entity
    test_data.id2relation = id2relation
    
    train_data.edge2id = relations_dict
    valid_data.edge2id = relations_dict
    test_data.edge2id = relations_dict

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)
   
    train_data.train_id2entity = train_data.id2entity
    valid_data.train_id2entity = train_data.id2entity
    test_data.train_id2entity = train_data.id2entity
    valid_data.valid_id2entity = valid_data.id2entity
    train_data.valid_id2entity = valid_data.id2entity
    test_data.valid_id2entity = valid_data.id2entity
    test_data.test_id2entity = test_data.id2entity
    train_data.test_id2entity = test_data.id2entity
    valid_data.test_id2entity = test_data.id2entity

    train_data.train_id2relation = train_data.id2relation
    valid_data.train_id2relation = train_data.id2relation
    test_data.train_id2relation = train_data.id2relation
    valid_data.valid_id2relation = valid_data.id2relation
    train_data.valid_id2relation = valid_data.id2relation
    test_data.valid_id2relation = valid_data.id2relation
    test_data.test_id2relation = test_data.id2relation
    train_data.test_id2relation = test_data.id2relation
    valid_data.test_id2relation = test_data.id2relation

    train_data.train_edge2id = train_data.edge2id
    valid_data.train_edge2id = train_data.edge2id
    test_data.train_edge2id = train_data.edge2id
    valid_data.valid_edge2id = valid_data.edge2id
    train_data.valid_edge2id = valid_data.edge2id
    test_data.valid_edge2id = valid_data.edge2id
    test_data.test_edge2id = test_data.edge2id
    train_data.test_edge2id = test_data.edge2id
    valid_data.test_edge2id = test_data.edge2id

    attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
    for data in [train_data, valid_data, test_data]:
        for attr in attrs_to_remove:
            if hasattr(data, attr):
                delattr(data, attr)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    return dataset

def WN18RR(root, **kwargs):
    dataset = WordNet18RR(root=root+"/wn18rr/")
    # convert wn18rr into the same format as fb15k-237
    data = dataset.data
    num_nodes = int(data.edge_index.max()) + 1
    num_relations = int(data.edge_type.max()) + 1
    edge_index = data.edge_index[:, data.train_mask]
    edge_type = data.edge_type[data.train_mask]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
    edge_type = torch.cat([edge_type, edge_type + num_relations])
    train_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.train_mask],
                        target_edge_type=data.edge_type[data.train_mask],
                        num_relations=num_relations*2)
    valid_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.val_mask],
                        target_edge_type=data.edge_type[data.val_mask],
                        num_relations=num_relations*2)
    test_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                        target_edge_index=data.edge_index[:, data.test_mask],
                        target_edge_type=data.edge_type[data.test_mask],
                        num_relations=num_relations*2)
    
    # build relation graphs 

    train_data.dataset="WN18RR"
    valid_data.dataset="WN18RR"
    test_data.dataset="WN18RR"
    
    # 直接从 raw 文件读取 entity/relation 映射（WN18RR 格式不同，entity 和 relation 直接在三元组中）
    raw_dir = os.path.join(root, "wn18rr/raw")
    
    # 从 train.txt 提取所有唯一的 entity 和 relation
    entity_set = set()
    relation_set = set()
    with open(os.path.join(raw_dir, "train.txt"), "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                h, r, t = parts
                entity_set.add(h)
                entity_set.add(t)
                relation_set.add(r)
    
    # 构建 id2entity 和 id2relation（按字母序分配 ID，与原始处理一致）
    sorted_entities = sorted(entity_set)
    sorted_relations = sorted(relation_set)
    id2entity = {idx: e for idx, e in enumerate(sorted_entities)}
    id2relation = {idx: r for idx, r in enumerate(sorted_relations)}
    edge2id = {r: idx for idx, r in enumerate(sorted_relations)}
    
    train_data.id2entity = id2entity
    train_data.id2relation = id2relation
    valid_data.id2entity = id2entity
    valid_data.id2relation = id2relation
    test_data.id2entity = id2entity
    test_data.id2relation = id2relation
    train_data.edge2id = edge2id
    valid_data.edge2id = edge2id
    test_data.edge2id = edge2id

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    train_data.train_id2entity = train_data.id2entity
    valid_data.train_id2entity = train_data.id2entity
    test_data.train_id2entity = train_data.id2entity
    valid_data.valid_id2entity = valid_data.id2entity
    train_data.valid_id2entity = valid_data.id2entity
    test_data.valid_id2entity = valid_data.id2entity
    test_data.test_id2entity = test_data.id2entity
    train_data.test_id2entity = test_data.id2entity
    valid_data.test_id2entity = test_data.id2entity

    train_data.train_id2relation = train_data.id2relation
    valid_data.train_id2relation = train_data.id2relation
    test_data.train_id2relation = train_data.id2relation
    valid_data.valid_id2relation = valid_data.id2relation
    train_data.valid_id2relation = valid_data.id2relation
    test_data.valid_id2relation = valid_data.id2relation
    test_data.test_id2relation = test_data.id2relation
    train_data.test_id2relation = test_data.id2relation
    valid_data.test_id2relation = test_data.id2relation

    train_data.train_edge2id = train_data.edge2id
    valid_data.train_edge2id = train_data.edge2id
    test_data.train_edge2id = train_data.edge2id
    valid_data.valid_edge2id = valid_data.edge2id
    train_data.valid_edge2id = valid_data.edge2id
    test_data.valid_edge2id = valid_data.edge2id
    test_data.test_edge2id = test_data.edge2id
    train_data.test_edge2id = test_data.edge2id
    valid_data.test_edge2id = test_data.edge2id

    attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
    for data in [train_data, valid_data, test_data]:
        for attr in attrs_to_remove:
            if hasattr(data, attr):
                delattr(data, attr)

    dataset.data, dataset.slices = dataset.collate([train_data, valid_data, test_data])
    dataset.num_relations = num_relations * 2
    
    return dataset

class TransductiveDataset(InMemoryDataset):

    delimiter = None
    
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        self.dataset_name = kwargs['dataset_name']
        self.dataset_version = kwargs['dataset_version']

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
                triplets.append((u, v, r))

        # print("====================================")
        # print(transductive_edge2id)
        # print(triplets)
        # print("number of nodes: ", len(inv_entity_vocab))
        # print("number of relations: ", rel_cnt)
        # print(new)
        # exit()

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset
        train_data.id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        train_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        valid_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        test_data.id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        test_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        train_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        test_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}

        if(current_dataset == "Hetionet"):
            mapping = {
                "Gr>G": "Gene→regulates→Gene",
                "GpMF": "Gene–participates–Molecular Function",
                "AeG": "Anatomy–expresses–Gene",
                "GpBP": "Gene–participates–Biological Process",
                "GiG": "Gene–interacts–Gene",
                "AuG": "Anatomy–upregulates–Gene",
                "DaG": "Disease–associates–Gene",
                "GpPW": "Gene–participates–Pathway",
                "GcG": "Gene–covaries–Gene",
                "AdG": "Anatomy–downregulates–Gene",
                "CcSE": "Compound–causes–Side Effect",
                "GpCC": "Gene–participates–Cellular Component",
                "CdG": "Compound–downregulates–Gene",
                "DlA": "Disease–localizes–Anatomy",
                "DuG": "Disease–upregulates–Gene",
                "CuG": "Compound–upregulates–Gene",
                "DpS": "Disease–presents–Symptom",
                "CbG": "Compound–binds–Gene",
                "CrC": "Compound–resembles–Compound",
                "DdG": "Disease–downregulates–Gene",
                "PCiC": "Pharmacologic Class–includes–Compound",
                "CtD": "Compound–treats–Disease",
                "DrD": "Disease–resembles–Disease",
                "CpD": "Compound–palliates–Disease"
            }

            print(train_data.edge2id)


        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % (self.name)
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"
    
class CoDEx(TransductiveDataset):

    name = "codex"
    urls = [
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/train.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/valid.txt",
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/%s/test.txt"
    ]

    # parent organization/unit -> parent organization
    
    def download(self):
        # print("====================================")
        # print(self.urls)

        raw_paths_here = self.raw_paths
        raw_paths_here.append(os.path.join(self.raw_dir, "relations.json"))
        # print(raw_paths_here)

        for url, path in zip(self.urls, raw_paths_here):
            if '%' in url:
                download_path = download_url(url % self.name, self.raw_dir)
            else:
                download_path = download_url(url, self.raw_dir)

            os.rename(download_path, path)

    # default loading procedure: process train/valid/test files, create graphs from them

    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        # print("====================================")
        # print the max in train target etypes
        # print(train_target_etypes.max())

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        # print("====================================")
        # print(codex_edge2id)
        # exit()

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset

        codex_entity2descrip = fetch_in_parallel(list(test_results["inv_entity_vocab"].keys()), get_entities)
        codex_id2entity = {}
        for key, value in codex_entity2descrip.items():
            codex_id2entity[test_results["inv_entity_vocab"][key]] = value

        train_data.id2entity = codex_id2entity
        valid_data.id2entity = codex_id2entity
        test_data.id2entity = codex_id2entity
        
        codex_edge2id = {}
        relations =  {key: value for key, value in test_results["inv_rel_vocab"].items()}
        codex_id2edge = fetch_in_parallel(list(relations.keys()), get_properties)
        # print(codex_id2edge)

        for rel, id in relations.items():
            codex_edge2id[codex_id2edge[rel]] = id

        train_data.id2relation = {value: key for key, value in codex_edge2id.items()}
        valid_data.id2relation = {value: key for key, value in codex_edge2id.items()}
        test_data.id2relation = {value: key for key, value in codex_edge2id.items()}
        # print(test_data.id2relation)
        # exit()
        train_data.edge2id = codex_edge2id
        valid_data.edge2id = codex_edge2id
        test_data.edge2id = codex_edge2id
        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class CoDExSmall(CoDEx):
    """
    #node: 2034
    #edge: 36543
    #relation: 42
    """
    url = "https://zenodo.org/record/4281094/files/codex-s.tar.gz"
    md5 = "63cd8186fc2aeddc154e20cf4a10087e"
    name = "codex-s"

    def __init__(self, root, **kwargs):
        super(CoDExSmall, self).__init__(root=root, size='s', **kwargs)

class CoDExMedium(CoDEx):
    """
    #node: 17050
    #edge: 206205
    #relation: 51
    """
    url = "https://zenodo.org/record/4281094/files/codex-m.tar.gz"
    md5 = "43e561cfdca1c6ad9cc2f5b1ca4add76"
    name = "codex-m"
    def __init__(self, root, **kwargs):
        super(CoDExMedium, self).__init__(root=root, size='m', **kwargs)

class CoDExLarge(CoDEx):
    """
    #node: 77951
    #edge: 612437
    #relation: 69
    """
    url = "https://zenodo.org/record/4281094/files/codex-l.tar.gz"
    md5 = "9a10f4458c4bd2b16ef9b92b677e0d71"
    name = "codex-l"
    def __init__(self, root, **kwargs):
        super(CoDExLarge, self).__init__(root=root, size='l', **kwargs)

class NELL995(TransductiveDataset):

    # from the RED-GNN paper https://github.com/LARS-research/RED-GNN/tree/main/transductive/data/nell
    # the OG dumps were found to have test set leakages
    # training set is made out of facts+train files, so we sum up their samples to build one training graph

    urls = [
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/refs/heads/main/Static/transductive/data/nell/facts.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/refs/heads/main/Static/transductive/data/nell/train.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/refs/heads/main/Static/transductive/data/nell/valid.txt",
        "https://raw.githubusercontent.com/LARS-research/RED-GNN/refs/heads/main/Static/transductive/data/nell/test.txt",
    ]
    name = "nell995"

    @property
    def raw_file_names(self):
        return ["facts.txt", "train.txt", "valid.txt", "test.txt"]

    def process(self):
        train_files = self.raw_paths[:4]

        facts_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        train_results = self.load_file(train_files[1], facts_results["inv_entity_vocab"], facts_results["inv_rel_vocab"])
        valid_results = self.load_file(train_files[2], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[3], train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        num_node = valid_results["num_node"]
        num_relations = train_results["num_relation"]
        # print("====================================")
        # print(num_relations)

        train_triplets = facts_results["triplets"] + train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # print("====================================")
        # print(train_data.edge_index)
        # print("====================================")
        # print(train_data.num_relations)
        # print("======================================")

        # print("I was obv here")

        train_data.dataset="NELL995"
        valid_data.dataset="NELL995"
        test_data.dataset="NELL995"
        train_data.id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        train_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        valid_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        test_data.id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        test_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}

        nell_edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        train_data.edge2id = nell_edge2id
        valid_data.edge2id = nell_edge2id
        test_data.edge2id = nell_edge2id
            
        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class ConceptNet100k(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/train",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/valid",
        "https://raw.githubusercontent.com/guojiapub/BiQUE/master/src_data/conceptnet-100k/test",
    ]
    name = "cnet100k"
    delimiter = "\t"

class DBpedia100k(TransductiveDataset):
    urls = [
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_train.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_valid.txt",
        "https://raw.githubusercontent.com/iieir-km/ComplEx-NNE_AER/master/datasets/DB100K/_test.txt",
        ]
    name = "dbp100k"

    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset

        db_id2entity = fetch_in_parallel(list(test_results["inv_entity_vocab"].keys()), get_entities)
        for key, value in list(db_id2entity.items()):
            db_id2entity[test_results["inv_entity_vocab"][key]] = value
        
        train_data.id2entity = db_id2entity
        train_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.id2entity = db_id2entity
        valid_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        test_data.id2entity = db_id2entity
        test_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        train_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        test_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class YAGO310(TransductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
        ]
    name = "yago310"

class Hetionet(TransductiveDataset): 

    urls = [
        "https://www.dropbox.com/s/y47bt9oq57h6l5k/train.txt?dl=1",
        "https://www.dropbox.com/s/a0pbrx9tz3dgsff/valid.txt?dl=1",
        "https://www.dropbox.com/s/4dhrvg3fyq5tnu4/test.txt?dl=1",
        ]
    name = "hetionet"

class AristoV4(TransductiveDataset):

    url = "https://zenodo.org/record/5942560/files/aristo-v4.zip"

    name = "aristov4"
    delimiter = "\t"

    def download(self):
        download_path = download_url(self.url, self.raw_dir)
        extract_zip(download_path, self.raw_dir)
        os.unlink(download_path)
        for oldname, newname in zip(['train', 'valid', 'test'], self.raw_paths):
            os.rename(os.path.join(self.raw_dir, oldname), newname)

class SparserKG(TransductiveDataset):

    # 5 datasets based on FB/NELL/WD, introduced in https://github.com/THU-KEG/DacKGR
    # re-writing the loading function because dumps are in the format (h, t, r) while the standard is (h, r, t)

    url = "https://raw.githubusercontent.com/THU-KEG/DacKGR/master/data.zip"
    delimiter = "\t"
    base_name = "SparseKG"

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, "processed")

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        for dsname in ['NELL23K', 'WD-singer', 'FB15K-237-10', 'FB15K-237-20', 'FB15K-237-50']:
            for oldname, newname in zip(['train.triples', 'dev.triples', 'test.triples'], self.raw_file_names):
                os.renames(os.path.join(base_path, "data", dsname, oldname), os.path.join(base_path, dsname, "raw", newname))
        shutil.rmtree(os.path.join(base_path, "data"))

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, v, r = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
class WDsinger(SparserKG):   
    name = "WD-singer"

    # parent organization/unit -> parent organization
    # participated in conflict -> conflict
    # child organization/unit -> has subsidiary

    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        # print("====================================")
        # print the max in train target etypes
        # print(train_target_etypes.max())

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        # print("====================================")
        # print(codex_edge2id)
        # exit()

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset

        wiki_entity2descrip = fetch_in_parallel(list(test_results["inv_entity_vocab"].keys()), get_entities)
        wiki_id2entity = {}
        for key, value in wiki_entity2descrip.items():
            wiki_id2entity[test_results["inv_entity_vocab"][key]] = value

        train_data.id2entity = wiki_id2entity
        valid_data.id2entity = wiki_id2entity
        test_data.id2entity = wiki_id2entity
        
        wiki_edge2id = {}
        relations =  {key: value for key, value in test_results["inv_rel_vocab"].items()}
        # print("====================================")
        # print(relations)
        wiki_id2edge = fetch_in_parallel(list(relations.keys()), get_properties)
        # print(codex_id2edge)

        for rel, id in relations.items():
            wiki_edge2id[wiki_id2edge[rel]] = id

        # print("====================================")
        # print(wiki_edge2id)
        # exit()

        train_data.id2relation = {value: key for key, value in wiki_edge2id.items()}
        valid_data.id2relation = {value: key for key, value in wiki_edge2id.items()}
        test_data.id2relation = {value: key for key, value in wiki_edge2id.items()}
        # print(test_data.id2relation)
        # exit()
        train_data.edge2id = wiki_edge2id
        valid_data.edge2id = wiki_edge2id
        test_data.edge2id = wiki_edge2id

        # print("====================================")
        # print(train_data.edge2id)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class NELL23k(SparserKG):   
    name = "NELL23K"

class SparseFb15k237(SparserKG):
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        id2entity = {value: key for key, value in test_results["inv_entity_vocab"].items()}
        # load fb_mid2name.tsv in a dictionary
        entities_dict = {}
        file_path_here = os.path.join(mydir, "fb_mid2name.tsv")
        with open(file_path_here) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: value for key, value in lines}

        for key, value in id2entity.items():
            if value in entities_dict:
                id2entity[key] = entities_dict[value]

        entities_dict.clear()

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset
        train_data.id2entity = id2entity
        train_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.id2entity = id2entity
        valid_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        test_data.id2entity = id2entity
        test_data.id2relation = {value: key for key, value in test_results["inv_rel_vocab"].items()}
        train_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        valid_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}
        test_data.edge2id = {key: value for key, value in test_results["inv_rel_vocab"].items()}

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class FB15k237_10(SparseFb15k237):   
    name = "FB15K-237-10"

class FB15k237_20(SparseFb15k237):   
    name = "FB15K-237-20"

class FB15k237_50(SparseFb15k237):   
    name = "FB15K-237-50"

class InductiveDataset(InMemoryDataset):

    delimiter = None
    # some datasets (4 from Hamaguchi et al and Indigo) have validation set based off the train graph, not inference
    valid_on_inf = True  # 
    
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        
        if(kwargs['dataset_name'] != 'HM'):
            self.version = str(kwargs['dataset_version'])

        self.dataset_name = kwargs['dataset_name']
        self.dataset_version = kwargs['dataset_version']

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def process(self):
        
        train_files = self.raw_paths[:4]

        print(train_files)

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=valid_res["num_relation"]*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset
        train_data.id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
        train_data.id2relation = {v: k for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
        valid_data.id2relation = {v: k for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}
        test_data.id2relation = {v: k for k, v in test_res["inv_rel_vocab"].items()}
        train_data.edge2id = {k: v for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.edge2id = {k: v for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.edge2id = {k: v for k, v in test_res["inv_rel_vocab"].items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "inf_valid.txt", "inf_test.txt"
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)

class IngramInductive(InductiveDataset):

    @property
    def raw_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "ingram", self.name, self.version, "processed")

class FBIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/FB-%s/test.txt",
    ]
    name = "fb"
    versions = ["25", "50", "75", "100"]

    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2],
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name
        

        train_id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
        valid_id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
        test_id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}

        entities_dict = {}
        file_path_here = os.path.join(mydir, "fb_mid2name.tsv")
        with open(file_path_here) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: value for key, value in lines}

        for key, value in train_id2entity.items():
            if value in entities_dict:
                train_id2entity[key] = entities_dict[value]

        for key, value in valid_id2entity.items():
            if value in entities_dict:
                valid_id2entity[key] = entities_dict[value]

        for key, value in test_id2entity.items():
            if value in entities_dict:
                test_id2entity[key] = entities_dict[value]

        entities_dict.clear()

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset
        train_data.id2entity = train_id2entity
        train_data.id2relation = {v: k for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.id2entity = valid_id2entity
        valid_data.id2relation = {v: k for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.id2entity = test_id2entity
        test_data.id2relation = {v: k for k, v in test_res["inv_rel_vocab"].items()}
        train_data.edge2id = {k: v for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.edge2id = {k: v for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.edge2id = {k: v for k, v in test_res["inv_rel_vocab"].items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class WKIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/WK-%s/test.txt",
    ]
    name = "wk"
    versions = ["25", "50", "75", "100"]

    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        wiki_entity2descrip = fetch_in_parallel(list(train_res["inv_entity_vocab"].keys()), get_entities)
        wiki_id2entity = {}
        for key, value in wiki_entity2descrip.items():
            wiki_id2entity[train_res["inv_entity_vocab"][key]] = value

        wiki_relation2descrip = fetch_in_parallel(list(train_res["inv_rel_vocab"].keys()), get_properties)
        wiki_id2relation = {}
        for key, value in wiki_relation2descrip.items():
            wiki_id2relation[train_res["inv_rel_vocab"][key]] = value
        
        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset

        train_data.id2entity = wiki_id2entity
        train_data.id2relation = wiki_id2relation

        # print("====================================")
        # print(train_data.id2entity)
        # exit()

        wiki_id2entity_inf = {}
        wiki_id2relation_inf = {}

        wiki_inventity2descrip = fetch_in_parallel(list(test_res["inv_entity_vocab"].keys()), get_entities)
        for key, value in wiki_inventity2descrip.items():
            wiki_id2entity_inf[test_res["inv_entity_vocab"][key]] = value

        wiki_invrelation2descrip = fetch_in_parallel(list(test_res["inv_rel_vocab"].keys()), get_properties)
        for key, value in wiki_invrelation2descrip.items():
            wiki_id2relation_inf[test_res["inv_rel_vocab"][key]] = value

        test_data.id2entity = wiki_id2entity_inf
        test_data.id2relation = wiki_id2relation_inf

        if self.valid_on_inf:
            valid_data.id2entity = test_data.id2entity
            valid_data.id2relation = test_data.id2relation
        else:
            valid_data.id2entity = train_data.id2entity
            valid_data.id2relation = train_data.id2relation

        train_data.edge2id = {v: k for k, v in train_data.id2relation.items()}
        valid_data.edge2id = {v: k for k, v in valid_data.id2relation.items()}
        test_data.edge2id = {v: k for k, v in test_data.id2relation.items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class NLIngram(IngramInductive):

    urls = [
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/train.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/msg.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/valid.txt",
        "https://raw.githubusercontent.com/bdi-lab/InGram/master/data/NL-%s/test.txt",
    ]
    name = "nl"
    versions = ["0", "25", "50", "75", "100"]

class ILPC2022(InductiveDataset):

    urls = [
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/train.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_validation.txt",
        "https://raw.githubusercontent.com/pykeen/ilpc2022/master/data/%s/inference_test.txt",
    ]

    # parent organization/unit -> parent organization

    name = "ilpc2022"
    versions = ["small", "large"]

    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        wiki_entity2descrip = fetch_in_parallel(list(train_res["inv_entity_vocab"].keys()), get_entities)
        wiki_id2entity = {}
        for key, value in wiki_entity2descrip.items():
            wiki_id2entity[train_res["inv_entity_vocab"][key]] = value

        wiki_relation2descrip = fetch_in_parallel(list(train_res["inv_rel_vocab"].keys()), get_properties)
        wiki_id2relation = {}
        for key, value in wiki_relation2descrip.items():
            wiki_id2relation[train_res["inv_rel_vocab"][key]] = value
        
        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset

        train_data.id2entity = wiki_id2entity
        train_data.id2relation = wiki_id2relation

        wiki_id2entity_inf = {}
        wiki_id2relation_inf = {}

        wiki_inventity2descrip = fetch_in_parallel(list(test_res["inv_entity_vocab"].keys()), get_entities)
        for key, value in wiki_inventity2descrip.items():
            wiki_id2entity_inf[test_res["inv_entity_vocab"][key]] = value

        wiki_invrelation2descrip = fetch_in_parallel(list(test_res["inv_rel_vocab"].keys()), get_properties)
        for key, value in wiki_invrelation2descrip.items():
            wiki_id2relation_inf[test_res["inv_rel_vocab"][key]] = value

        test_data.id2entity = wiki_id2entity_inf
        test_data.id2relation = wiki_id2relation_inf

        if self.valid_on_inf:
            valid_data.id2entity = test_data.id2entity
            valid_data.id2relation = test_data.id2relation
        else:
            valid_data.id2entity = train_data.id2entity
            valid_data.id2relation = train_data.id2relation

        # print("====================================")
        # print(train_data.id2relation)
        # print("====================================")
        # print(test_data.id2relation)

        train_data.edge2id = {v: k for k, v in train_data.id2relation.items()}
        valid_data.edge2id = {v: k for k, v in valid_data.id2relation.items()}
        test_data.edge2id = {v: k for k, v in test_data.id2relation.items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class HM(InductiveDataset): 
    # benchmarks from Hamaguchi et al and Indigo BM

    urls = [
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/train.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-graph.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/train/valid.txt",
        "https://raw.githubusercontent.com/shuwen-liu-ox/INDIGO/master/data/%s/test/test-fact.txt",
    ]

    name = "hm"
    versions = {
        '1k': "Hamaguchi-BM_both-1000",
        '3k': "Hamaguchi-BM_both-3000",
        '5k': "Hamaguchi-BM_both-5000",
        'indigo': "INDIGO-BM" 
    }
    # in 4 HM graphs, the validation set is based off the training graph, so we'll adjust the dataset creation accordingly
    valid_on_inf = False 

    def __init__(self, root, **kwargs):
        self.version = self.versions[kwargs['dataset_version']]
        super().__init__(root, **kwargs)

    # HM datasets are a bit weird: validation set (based off the train graph) has a few hundred new nodes, so we need a custom processing
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])

        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                          edge_type=train_fact_type, 
                          num_nodes=valid_res["num_node"],  # the only fix in this function
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name
        
        train_data.dataset=current_dataset
        valid_data.dataset=current_dataset
        test_data.dataset=current_dataset
        train_data.id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
        train_data.id2relation = {v: k for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
        valid_data.id2relation = {v: k for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}
        test_data.id2relation = {v: k for k, v in test_res["inv_rel_vocab"].items()}
        train_data.edge2id = {k: v for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.edge2id = {k: v for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.edge2id = {k: v for k, v in test_res["inv_rel_vocab"].items()}

        if(train_data.dataset == "HM-indigo"):
            train_id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
            valid_id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
            test_id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}

            entities_dict = {}
            file_path_here = os.path.join(mydir, "fb_mid2name.tsv")
            with open(file_path_here) as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                entities_dict = {key: value for key, value in lines}

            for key, value in train_id2entity.items():
                if value in entities_dict:
                    train_id2entity[key] = entities_dict[value]

            for key, value in valid_id2entity.items():
                if value in entities_dict:
                    valid_id2entity[key] = entities_dict[value]

            for key, value in test_id2entity.items():
                if value in entities_dict:
                    test_id2entity[key] = entities_dict[value]

            entities_dict.clear()
            
            train_data.id2entity = train_id2entity
            valid_data.id2entity = valid_id2entity
            test_data.id2entity = test_id2entity

        else:
            entity_url = "https://raw.githubusercontent.com/takuo-h/GNN-for-OOKB/master/2-OOKB-setting/datasets/OOKB/vocab/entity"
            relation_url = "https://raw.githubusercontent.com/takuo-h/GNN-for-OOKB/master/2-OOKB-setting/datasets/OOKB/vocab/relation"

            def load_vocab(url):
                response = requests.get(url)
                response.raise_for_status()
                lines = response.text.strip().split('\n')
                vocab = {}
                for line in lines:
                    token, idx = line.strip().split('\t')
                    vocab[int(idx)] = token
                return vocab

            # Load into dictionaries
            entities_dict = load_vocab(entity_url)
            relation_dict = load_vocab(relation_url)

            train_id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
            valid_id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
            test_id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}
            train_id2relation = {v: k for k, v in train_res["inv_rel_vocab"].items()}
            valid_id2relation = {v: k for k, v in valid_res["inv_rel_vocab"].items()}
            test_id2relation = {v: k for k, v in test_res["inv_rel_vocab"].items()}

            for key, value in train_id2entity.items():
                if key in entities_dict:
                    train_id2entity[key] = entities_dict[key]

            for key, value in train_id2relation.items():
                if key in relation_dict:
                    train_id2relation[key] = relation_dict[key]

            for key, value in valid_id2entity.items():
                if key in entities_dict:
                    valid_id2entity[key] = entities_dict[key]
            
            for key, value in valid_id2relation.items():
                if key in relation_dict:
                    valid_id2relation[key] = relation_dict[key]

            for key, value in test_id2entity.items():
                if key in entities_dict:
                    test_id2entity[key] = entities_dict[key]

            for key, value in test_id2relation.items():
                if key in relation_dict:
                    test_id2relation[key] = relation_dict[key]

            entities_dict.clear()
            relation_dict.clear()

            train_data.id2entity = train_id2entity
            valid_data.id2entity = valid_id2entity
            test_data.id2entity = test_id2entity
            train_data.id2relation = train_id2relation
            valid_data.id2relation = valid_id2relation
            test_data.id2relation = test_id2relation
            train_data.edge2id = {v: k for k, v in train_data.id2relation.items()}
            valid_data.edge2id = {v: k for k, v in valid_data.id2relation.items()}
            test_data.edge2id = {v: k for k, v in test_data.id2relation.items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class MTDEAInductive(InductiveDataset):
    
    valid_on_inf = False
    url = "https://reltrans.s3.us-east-2.amazonaws.com/MTDEA_data.zip"
    base_name = "mtdea"

    def __init__(self, root, **kwargs):
        version = kwargs['dataset_version']
        assert version in self.versions, f"unknown version {version} for {self.name}, available: {self.versions}"
        super().__init__(root, **kwargs)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.base_name, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "transductive_valid.txt", "inf_test.txt"
        ]

    def download(self):
        base_path = os.path.join(self.root, self.base_name)
        download_path = download_url(self.url, base_path)
        extract_zip(download_path, base_path)
        # unzip all datasets at once
        for dsname in ['FBNELL', 'Metafam', 'WikiTopics-MT1', 'WikiTopics-MT2', 'WikiTopics-MT3', 'WikiTopics-MT4']:
            cl = globals()[dsname.replace("-","")]
            versions = cl.versions
            for version in versions:
                for oldname, newname in zip(['train.txt', 'observe.txt', 'valid.txt', 'test.txt'], self.raw_file_names):
                    foldername = cl.prefix % version + "-trans" if "transductive" in newname else cl.prefix % version + "-ind"
                    os.renames(
                        os.path.join(base_path, "MTDEA_datasets", dsname, foldername, oldname), 
                        os.path.join(base_path, dsname, version, "raw", newname)
                    )
        shutil.rmtree(os.path.join(base_path, "MTDEA_datasets"))

    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}, limit_vocab=False):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # limit_vocab is for dropping triples with unseen head/tail not seen in the main entity_vocab
        # can be used for FBNELL and MT3:art, other datasets seem to be ok and share num_nodes/num_relations in the train/inference graph  
        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    if limit_vocab:
                        continue
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    if limit_vocab:
                        continue
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))
        
        return {
            "triplets": triplets,
            "num_node": entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }

    # special processes for MTDEA datasets for one particular fix in the validation set loading
    def process(self):
    
        train_files = self.raw_paths[:4]

        
        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])
        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

     

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)
                
        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name

        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset
        train_data.id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
        train_data.id2relation = {v: k for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
        valid_data.id2relation = {v: k for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}
        test_data.id2relation = {v: k for k, v in test_res["inv_rel_vocab"].items()}
        train_data.edge2id = {k: v for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.edge2id = {k: v for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.edge2id = {k: v for k, v in test_res["inv_rel_vocab"].items()}

  

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        if hasattr(test_data, "harder_head_rg1"):
        #     print("Yes it has harder head rg")
            train_data.harder_head_rg1 = test_data.harder_head_rg1
            valid_data.harder_head_rg1 = test_data.harder_head_rg1
            train_data.harder_tail_rg1 = test_data.harder_tail_rg1
            valid_data.harder_tail_rg1 = test_data.harder_tail_rg1
        
        if hasattr(test_data, "harder_head_rg2"):
            train_data.harder_head_rg2 = test_data.harder_head_rg2
            valid_data.harder_head_rg2 = test_data.harder_head_rg2
            train_data.harder_tail_rg2 = test_data.harder_tail_rg2
            valid_data.harder_tail_rg2 = test_data.harder_tail_rg2

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class FBNELL(MTDEAInductive):

    name = "FBNELL"
    prefix = "%s"
    versions = ["FBNELL_v1"]

    def __init__(self, **kwargs):
        kwargs.pop('dataset_version')
        kwargs['dataset_version'] = self.versions[0]
        super(FBNELL, self).__init__(**kwargs)

    def process(self):
    
        train_files = self.raw_paths[:4]

        
        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])
        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

    

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name
        
        train_id2entity = {v: k for k, v in train_res["inv_entity_vocab"].items()}
        valid_id2entity = {v: k for k, v in valid_res["inv_entity_vocab"].items()}
        test_id2entity = {v: k for k, v in test_res["inv_entity_vocab"].items()}
        

        entities_dict = {}
        file_path_here = os.path.join(mydir, "fb_mid2name.tsv")
        with open(file_path_here) as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: value for key, value in lines}

        for key, value in train_id2entity.items():
            if value in entities_dict:
                train_id2entity[key] = entities_dict[value]

        for key, value in valid_id2entity.items():
            if value in entities_dict:
                valid_id2entity[key] = entities_dict[value]

        for key, value in test_id2entity.items():
            if value in entities_dict:
                test_id2entity[key] = entities_dict[value]

        entities_dict.clear()

        train_data.dataset=current_dataset
        valid_data.dataset=current_dataset
        test_data.dataset=current_dataset
        train_data.id2entity = train_id2entity
        train_data.id2relation = {v: k for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.id2entity = valid_id2entity
        valid_data.id2relation = {v: k for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.id2entity = test_id2entity
        test_data.id2relation = {v: k for k, v in test_res["inv_rel_vocab"].items()}
        train_data.edge2id = {k: v for k, v in train_res["inv_rel_vocab"].items()}
        valid_data.edge2id = {k: v for k, v in valid_res["inv_rel_vocab"].items()}
        test_data.edge2id = {k: v for k, v in test_res["inv_rel_vocab"].items()}

    

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        if hasattr(test_data, "harder_head_rg1"):
        #     print("Yes it has harder head rg")
            train_data.harder_head_rg1 = test_data.harder_head_rg1
            valid_data.harder_head_rg1 = test_data.harder_head_rg1
            train_data.harder_tail_rg1 = test_data.harder_tail_rg1
            valid_data.harder_tail_rg1 = test_data.harder_tail_rg1
        
        if hasattr(test_data, "harder_head_rg2"):
            train_data.harder_head_rg2 = test_data.harder_head_rg2
            valid_data.harder_head_rg2 = test_data.harder_head_rg2
            train_data.harder_tail_rg2 = test_data.harder_tail_rg2
            valid_data.harder_tail_rg2 = test_data.harder_tail_rg2
            
        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class Metafam(MTDEAInductive):

    name = "Metafam"
    prefix = "%s"
    versions = ["Metafam"]

    def __init__(self, **kwargs):
        kwargs.pop('dataset_version')
        kwargs['dataset_version'] = self.versions[0]
        super(Metafam, self).__init__(**kwargs)

class WikiTopics(MTDEAInductive):

    def __init__(self, **kwargs):
        assert kwargs['dataset_version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopics, self).__init__(**kwargs)

    def process(self):
    
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"],
            limit_vocab=True,  # the 1st fix in this function compared to the superclass processor
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])
        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=train_fact_index, 
                        edge_type=train_fact_type, 
                        num_nodes=valid_res["num_node"],  # the 2nd fix in this function
                        target_edge_index=inf_valid_edges[:, :2].T, 
                        target_edge_type=inf_valid_edges[:, 2], 
                        num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                        target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.dataset_version is not None:
            current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        else:
            current_dataset = self.dataset_name
        
        train_data.dataset=current_dataset
        valid_data.dataset=current_dataset
        test_data.dataset=current_dataset
        
        wiki_entity2descrip = fetch_in_parallel(list(train_res["inv_entity_vocab"].keys()), get_entities)
        wiki_id2entity = {}
        for key, value in wiki_entity2descrip.items():
            wiki_id2entity[train_res["inv_entity_vocab"][key]] = value

        wiki_relation2descrip = fetch_in_parallel(list(train_res["inv_rel_vocab"].keys()), get_properties)
        wiki_id2relation = {}
        for key, value in wiki_relation2descrip.items():
            wiki_id2relation[train_res["inv_rel_vocab"][key]] = value
        
        train_data.id2entity = wiki_id2entity
        train_data.id2relation = wiki_id2relation

        wiki_id2entity_inf = {}
        wiki_id2relation_inf = {}

        wiki_inventity2descrip = fetch_in_parallel(list(test_res["inv_entity_vocab"].keys()), get_entities)
        for key, value in wiki_inventity2descrip.items():
            wiki_id2entity_inf[test_res["inv_entity_vocab"][key]] = value

        wiki_invrelation2descrip = fetch_in_parallel(list(test_res["inv_rel_vocab"].keys()), get_properties)
        for key, value in wiki_invrelation2descrip.items():
            wiki_id2relation_inf[test_res["inv_rel_vocab"][key]] = value

        test_data.id2entity = wiki_id2entity_inf
        test_data.id2relation = wiki_id2relation_inf

        if self.valid_on_inf:
            valid_data.id2entity = test_data.id2entity
            valid_data.id2relation = test_data.id2relation
        else:
            valid_data.id2entity = train_data.id2entity
            valid_data.id2relation = train_data.id2relation
        
        train_data.edge2id = {v: k for k, v in train_data.id2relation.items()}
        valid_data.edge2id = {v: k for k, v in valid_data.id2relation.items()}
        test_data.edge2id = {v: k for k, v in test_data.id2relation.items()}

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        if hasattr(test_data, "harder_head_rg1"):
            train_data.harder_head_rg1 = test_data.harder_head_rg1
            valid_data.harder_head_rg1 = test_data.harder_head_rg1
            train_data.harder_tail_rg1 = test_data.harder_tail_rg1
            valid_data.harder_tail_rg1 = test_data.harder_tail_rg1

        if hasattr(test_data, "harder_head_rg2"):
            train_data.harder_head_rg2 = test_data.harder_head_rg2
            valid_data.harder_head_rg2 = test_data.harder_head_rg2
            train_data.harder_tail_rg2 = test_data.harder_tail_rg2
            valid_data.harder_tail_rg2 = test_data.harder_tail_rg2

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ['id2entity', 'id2relation', 'edge2id']
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

class WikiTopicsMT1(WikiTopics):

    name = "WikiTopics-MT1"
    prefix = "wikidata_%sv1"
    versions = ['health', 'tax']  # 注意：没有 'mt' 版本

    def __init__(self, **kwargs):
        assert kwargs['dataset_version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopicsMT1, self).__init__(**kwargs)

class WikiTopicsMT2(WikiTopics):

    name = "WikiTopics-MT2"
    prefix = "wikidata_%sv1"
    versions = ['org', 'sci']  # 注意：没有 'mt2' 版本

    def __init__(self, **kwargs):
        super(WikiTopicsMT2, self).__init__(**kwargs)

class WikiTopicsMT3(WikiTopics):

    name = "WikiTopics-MT3"
    prefix = "wikidata_%sv2"
    versions = ['art', 'infra']  # 注意：没有 'mt3' 版本

    def __init__(self, **kwargs):
        super(WikiTopicsMT3, self).__init__(**kwargs)

class WikiTopicsMT4(WikiTopics):

    name = "WikiTopics-MT4"
    prefix = "wikidata_%sv2"
    versions = ['sci', 'health']  # 注意：没有 'mt4' 版本

    def __init__(self, **kwargs):
        super(WikiTopicsMT4, self).__init__(**kwargs)
class Atlas(InMemoryDataset):
    name = "Atlas"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        dataset_name=None,
        dataset_version=None,
        split_seed=42,
        **kwargs,
    ):
        # Expected dataset_version format: "small:0" / "small:1" / "one_hop:12"
        version = str(dataset_version or "small:0").strip()
        if ":" not in version:
            raise ValueError(
                f"Atlas dataset_version must be '<graph>:<index>', got: {version}. "
                "Example: small:0"
            )
        graph_name, idx_str = version.split(":", 1)
        graph_name = graph_name.strip()
        idx_str = idx_str.strip()
        if graph_name not in {"small", "one_hop"}:
            raise ValueError(f"Unknown Atlas graph type: {graph_name}. Use 'small' or 'one_hop'.")
        try:
            graph_idx = int(idx_str)
        except ValueError as e:
            raise ValueError(f"Invalid Atlas graph index in dataset_version: {version}") from e

        self.dataset_name = dataset_name or "Atlas"
        self.dataset_version = version
        self.graph_name = graph_name
        self.graph_idx = graph_idx
        self.split_seed = int(split_seed)

        # 将 Atlas 数据固定放在 root/atlas/ 子目录下，和其他数据集保持一致风格
        # （例如 FB15k237 在 root/fb15k237/...）
        self._atlas_root = os.path.join(root, "atlas")

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self):
        # e.g. <root>/atlas/splits_small 或 <root>/atlas/splits_one_hop
        return os.path.join(self._atlas_root, f"splits_{self.graph_name}")

    @property
    def processed_dir(self):
        return os.path.join(
            self._atlas_root,
            "processed_atlas",
            f"{self.graph_name}-{self.graph_idx:04d}",
            "processed",
        )

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        nodes_csv = os.path.join(
            self._atlas_root,
            f"splits_{self.graph_name}",
            f"{self.graph_name}_{self.graph_idx:04d}_nodes.csv",
        )
        edges_csv = os.path.join(
            self._atlas_root,
            f"splits_{self.graph_name}",
            f"{self.graph_name}_{self.graph_idx:04d}_edges.csv",
        )
        if not os.path.exists(nodes_csv):
            raise FileNotFoundError(f"Atlas nodes file not found: {nodes_csv}")
        if not os.path.exists(edges_csv):
            raise FileNotFoundError(f"Atlas edges file not found: {edges_csv}")

        # 1) Load nodes and remap original node id -> [0..N-1]
        old_id_to_name = {}
        with open(nodes_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = str((row.get("id") or "")).strip()
                if not nid:
                    continue
                old_id_to_name[nid] = str((row.get("name") or "")).strip()
        old_ids = sorted(old_id_to_name.keys(), key=lambda x: int(x))
        old_to_new = {old: new for new, old in enumerate(old_ids)}
        id2entity = {new: old_id_to_name[old] for old, new in old_to_new.items()}

        # 2) Load edges; remap src/dst ids and relation ids
        rel2id = {}
        triplets = []
        with open(edges_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_old = str((row.get("src_id") or "")).strip()
                t_old = str((row.get("dst_id") or "")).strip()
                if not s_old or not t_old:
                    continue
                if s_old not in old_to_new or t_old not in old_to_new:
                    continue

                rel = str((row.get("relation") or "Relation")).strip() or "Relation"
                if rel not in rel2id:
                    rel2id[rel] = len(rel2id)
                r = rel2id[rel]
                triplets.append((old_to_new[s_old], old_to_new[t_old], r))

        if len(triplets) == 0:
            raise RuntimeError(f"No valid triplets loaded from: {edges_csv}")

        # 3) Split target edges into train/valid/test = 8:1:1
        triplets_tensor = torch.tensor(triplets, dtype=torch.long)
        num_edges = triplets_tensor.shape[0]
        g = torch.Generator()
        g.manual_seed(self.split_seed + self.graph_idx)
        perm = torch.randperm(num_edges, generator=g)
        triplets_tensor = triplets_tensor[perm]

        n_train = int(num_edges * 0.8)
        n_valid = int(num_edges * 0.1)
        n_test = num_edges - n_train - n_valid
        if n_train <= 0 or n_valid <= 0 or n_test <= 0:
            raise RuntimeError(
                f"Atlas split failed due to too few edges: total={num_edges}, "
                f"train={n_train}, valid={n_valid}, test={n_test}"
            )

        train_t = triplets_tensor[:n_train]
        valid_t = triplets_tensor[n_train:n_train + n_valid]
        test_t = triplets_tensor[n_train + n_valid:]

        num_nodes = len(old_to_new)
        num_relations = len(rel2id)

        train_target_edges = train_t[:, :2].T.contiguous()
        train_target_types = train_t[:, 2].contiguous()
        valid_target_edges = valid_t[:, :2].T.contiguous()
        valid_target_types = valid_t[:, 2].contiguous()
        test_target_edges = test_t[:, :2].T.contiguous()
        test_target_types = test_t[:, 2].contiguous()

        # Following existing pipelines: use train facts (+ inverse edges) as message graph.
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_types = torch.cat([train_target_types, train_target_types + num_relations])

        train_data = Data(
            edge_index=train_edges,
            edge_type=train_types,
            num_nodes=num_nodes,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_types,
            num_relations=num_relations * 2,
        )
        valid_data = Data(
            edge_index=train_edges,
            edge_type=train_types,
            num_nodes=num_nodes,
            target_edge_index=valid_target_edges,
            target_edge_type=valid_target_types,
            num_relations=num_relations * 2,
        )
        test_data = Data(
            edge_index=train_edges,
            edge_type=train_types,
            num_nodes=num_nodes,
            target_edge_index=test_target_edges,
            target_edge_type=test_target_types,
            num_relations=num_relations * 2,
        )

        current_dataset = f"{self.dataset_name}-{self.dataset_version}"
        train_data.dataset = current_dataset
        valid_data.dataset = current_dataset
        test_data.dataset = current_dataset

        id2relation = {v: k for k, v in rel2id.items()}
        train_data.id2entity = id2entity
        valid_data.id2entity = id2entity
        test_data.id2entity = id2entity
        train_data.id2relation = id2relation
        valid_data.id2relation = id2relation
        test_data.id2relation = id2relation
        train_data.edge2id = rel2id
        valid_data.edge2id = rel2id
        test_data.edge2id = rel2id

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        train_data.train_id2entity = train_data.id2entity
        valid_data.train_id2entity = train_data.id2entity
        test_data.train_id2entity = train_data.id2entity
        valid_data.valid_id2entity = valid_data.id2entity
        train_data.valid_id2entity = valid_data.id2entity
        test_data.valid_id2entity = valid_data.id2entity
        test_data.test_id2entity = test_data.id2entity
        train_data.test_id2entity = test_data.id2entity
        valid_data.test_id2entity = test_data.id2entity

        train_data.train_id2relation = train_data.id2relation
        valid_data.train_id2relation = train_data.id2relation
        test_data.train_id2relation = train_data.id2relation
        valid_data.valid_id2relation = valid_data.id2relation
        train_data.valid_id2relation = valid_data.id2relation
        test_data.valid_id2relation = valid_data.id2relation
        test_data.test_id2relation = test_data.id2relation
        train_data.test_id2relation = test_data.id2relation
        valid_data.test_id2relation = test_data.id2relation

        train_data.train_edge2id = train_data.edge2id
        valid_data.train_edge2id = train_data.edge2id
        test_data.train_edge2id = train_data.edge2id
        valid_data.valid_edge2id = valid_data.edge2id
        train_data.valid_edge2id = valid_data.edge2id
        test_data.valid_edge2id = valid_data.edge2id
        test_data.test_edge2id = test_data.edge2id
        train_data.test_edge2id = test_data.edge2id
        valid_data.test_edge2id = test_data.edge2id

        attrs_to_remove = ["id2entity", "id2relation", "edge2id"]
        for data in [train_data, valid_data, test_data]:
            for attr in attrs_to_remove:
                if hasattr(data, attr):
                    delattr(data, attr)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

# a joint dataset for pre-training ULTRA on several graphs
class JointDataset(InMemoryDataset): 


    datasets_map = {
        # ========== Transductive datasets (16个) ==========
        'FB15k237': FB15k237,
        'FB15k237_10': FB15k237_10,  # FB15k237 稀疏子集 (10%)
        'FB15k237_20': FB15k237_20,  # FB15k237 稀疏子集 (20%)
        'FB15k237_50': FB15k237_50,  # FB15k237 稀疏子集 (50%)
        'WN18RR': WN18RR,
        'CoDExSmall': CoDExSmall,
        'CoDExMedium': CoDExMedium,
        'CoDExLarge': CoDExLarge,
        'NELL995': NELL995,
        'ConceptNet100k': ConceptNet100k,
        'DBpedia100k': DBpedia100k,
        'YAGO310': YAGO310,
        'AristoV4': AristoV4,
        'Hetionet': Hetionet,
        'WDsinger': WDsinger,
        'NELL23k': NELL23k,

        # ========== Inductive datasets - new nodes, no new relations (18个) ==========
        # GraIL datasets (12个: 3个基础类 × 4个版本)
        'FB15k237Inductive': FB15k237Inductive,  # v1, v2, v3, v4 (4个)
        'WN18RRInductive': WN18RRInductive,      # v1, v2, v3, v4 (4个)
        'NELLInductive': NELLInductive,          # v1, v2, v3, v4 (4个)
        # ILPC datasets (2个)
        'ILPC2022': ILPC2022,                    # small, large (2个)
        # Hamaguchi datasets (4个)
        'HM': HM,                                # 1k, 3k, 5k, indigo (4个)

        # ========== Inductive datasets - new nodes, new relations (23个) ==========
        # Ingram datasets (13个)
        'NLIngram': NLIngram,                    # 0, 25, 50, 75, 100 (5个)
        'FBIngram': FBIngram,                    # 25, 50, 75, 100 (4个)
        'WKIngram': WKIngram,                    # 25, 50, 75, 100 (4个)
        # MTDEA WikiTopics datasets (8个)
        'WikiTopicsMT1': WikiTopicsMT1,          # health, tax (2个)
        'WikiTopicsMT2': WikiTopicsMT2,          # org, sci (2个)
        'WikiTopicsMT3': WikiTopicsMT3,          # art, infra (2个)
        'WikiTopicsMT4': WikiTopicsMT4,          # sci, health (2个)
        # MTDEA other datasets (2个)
        'Metafam': Metafam,                      # Metafam (1个)
        'FBNELL': FBNELL,                        # FBNELL_v1 (1个)

        # ========== Atlas datasets (排除在 all 之外) ==========
        # Atlas: individual graphs specified as Atlas:small:0, Atlas:one_hop:0 etc.
        # Use Atlas:small or Atlas:one_hop to auto-expand all available graphs.
        'Atlas': Atlas,
    }

    # 总计: 16 (transductive) + 18 (inductive-no-new-rel) + 23 (inductive-new-rel) = 57个数据集

    def __init__(self, root, graphs, transform=None, pre_transform=None):
        # Support:
        # - graphs: "all" / ["all"] -> load all keys in datasets_map
        # - graphs item: "Name" -> if class has `versions`, expand to all versions
        # - graphs item: "Name:version" -> load a specific version
        if isinstance(graphs, str):
            # Support comma-separated string: "all,Atlas:small,Atlas:one_hop"
            graphs = [g.strip() for g in graphs.split(",") if g.strip()]
        graphs = list(graphs)

        # Pre-expand "all" keyword before the main loop so each resulting key
        # goes through the per-dataset version expansion logic below.
        pre_expanded = []
        for spec in graphs:
            spec_s = str(spec).strip()
            if not spec_s:
                continue
            if spec_s.lower() == "all":
                pre_expanded.extend(k for k in self.datasets_map.keys() if k != "Atlas")
            else:
                pre_expanded.append(spec_s)
        graphs = pre_expanded

        expanded_specs: list[str] = []
        for spec in graphs:
            spec_s = str(spec).strip()
            if not spec_s:
                continue

            if ":" in spec_s:
                ds_name, ds_version = spec_s.split(":", 1)
                ds_name = ds_name.strip()
                ds_version = ds_version.strip()
            else:
                ds_name, ds_version = spec_s, None

            # Case-insensitive lookup for dataset name
            if ds_name not in self.datasets_map:
                lower_map = {k.lower(): k for k in self.datasets_map}
                canonical = lower_map.get(ds_name.lower())
                if canonical is None:
                    raise KeyError(
                        f"Unknown dataset key '{ds_name}' for JointDataset. "
                        f"Available keys: {sorted(self.datasets_map.keys())}"
                    )
                ds_name = canonical

            ds_cls = self.datasets_map[ds_name]

            # Atlas shortcut: "Atlas:small" or "Atlas:one_hop" -> scan disk for all
            # available graphs of that type and expand to Atlas:small:0, Atlas:small:1, ...
            if ds_name == "Atlas" and ds_version in ("small", "one_hop"):
                atlas_dir = os.path.join(root, "atlas", f"splits_{ds_version}")
                pat = re.compile(rf"^{re.escape(ds_version)}_(\d+)_nodes\.csv$")
                indices = []
                if os.path.isdir(atlas_dir):
                    for fname in sorted(os.listdir(atlas_dir)):
                        m = pat.match(fname)
                        if m:
                            indices.append(int(m.group(1)))
                if not indices:
                    raise FileNotFoundError(
                        f"No Atlas graphs found for type '{ds_version}' in {atlas_dir}"
                    )
                for idx in sorted(indices):
                    expanded_specs.append(f"Atlas:{ds_version}:{idx}")
                continue

            # Auto-expand versions when user didn't specify one.
            versions = getattr(ds_cls, "versions", None)
            if ds_version is None and isinstance(versions, dict) and len(versions) > 0:
                expanded_specs.extend([f"{ds_name}:{v}" for v in versions.keys()])
            elif ds_version is None and isinstance(versions, (list, tuple)) and len(versions) > 0:
                expanded_specs.extend([f"{ds_name}:{v}" for v in versions])
            else:
                expanded_specs.append(f"{ds_name}:{ds_version}" if ds_version is not None else ds_name)

        # De-duplicate while preserving order
        seen = set()
        self.graph_specs: list[str] = []
        for s in expanded_specs:
            if s not in seen:
                seen.add(s)
                self.graph_specs.append(s)

        self.graphs = []
        for spec_s in self.graph_specs:
            if ":" in spec_s:
                ds_name, ds_version = spec_s.split(":", 1)
                ds_name = ds_name.strip()
                ds_version = ds_version.strip()
            else:
                ds_name, ds_version = spec_s.strip(), None
            ds_cls = self.datasets_map[ds_name]
            self.graphs.append(ds_cls(root=root, dataset_name=ds_name, dataset_version=ds_version))

        self.num_graphs = len(self.graph_specs)
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0],weights_only=False)
        # print(self.data)
        # exit()

    @property
    def raw_dir(self):
        spec_key = ",".join(self.graph_specs)
        spec_hash = hashlib.md5(spec_key.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.root, "joint", f"{self.num_graphs}g-{spec_hash}", "raw")

    @property
    def processed_dir(self):
        spec_key = ",".join(self.graph_specs)
        spec_hash = hashlib.md5(spec_key.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.root, "joint", f"{self.num_graphs}g-{spec_hash}", "processed")

    @property
    def processed_file_names(self):
        return "data.pt"
    
    def process(self):
        
        train_data = [g[0] for g in self.graphs]
        valid_data = [g[1] for g in self.graphs]
        test_data = [g[2] for g in self.graphs]
        # filter_data = [
        #     Data(edge_index=g.data.target_edge_index, edge_type=g.data.target_edge_type, num_nodes=g[0].num_nodes) for g in self.graphs
        # ]

        torch.save((train_data, valid_data, test_data), self.processed_paths[0])


class MultiModalKGDataset(InMemoryDataset):
    """Base class for multimodal knowledge graph datasets with images and numerical attributes."""

    def __init__(self, root, dataset_name, transform=None, pre_transform=build_relation_graph, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_version = kwargs.get('dataset_version', None)
        self.mmkb_root = "/data/gaoyisen/mmkb"
        super().__init__(root, transform, pre_transform)
        # Load the full data structure
        self.full_data = torch.load(self.processed_paths[0], weights_only=False)
        # For compatibility with InMemoryDataset, we need data and slices
        # But we store the actual data differently
        self.data = None
        self.slices = None

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset_name.lower(), "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset_name.lower(), "processed")

    @property
    def raw_file_names(self):
        return ["entity_triples.txt", "image_index.txt", "numerical_triples.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # Data already exists in mmkb_root, just create symlinks
        os.makedirs(self.raw_dir, exist_ok=True)

    def load_entity_triples(self, file_path):
        """Load entity triples from file."""
        triplets = []
        inv_entity_vocab = {}
        inv_rel_vocab = {}
        entity_cnt, rel_cnt = 0, 0

        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                # Handle both RDF format and tab-separated format
                if line.endswith(" ."):
                    parts = line[:-2].split()
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                    else:
                        continue
                else:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                    else:
                        continue

                if h not in inv_entity_vocab:
                    inv_entity_vocab[h] = entity_cnt
                    entity_cnt += 1
                if t not in inv_entity_vocab:
                    inv_entity_vocab[t] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1

                h_id = inv_entity_vocab[h]
                t_id = inv_entity_vocab[t]
                r_id = inv_rel_vocab[r]
                triplets.append((h_id, t_id, r_id))

        return triplets, inv_entity_vocab, inv_rel_vocab

    def load_image_index(self, file_path, inv_entity_vocab):
        """Load image index mapping entities to image IDs."""
        entity_to_image = {}

        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    entity, img_id = parts[0], parts[1]
                    if entity in inv_entity_vocab:
                        entity_to_image[inv_entity_vocab[entity]] = img_id

        return entity_to_image

    def load_numerical_triples(self, file_path, inv_entity_vocab, inv_rel_vocab):
        """Load numerical attribute triples."""
        numerical_data = []

        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                # Parse RDF format with typed literals
                if line.endswith(" ."):
                    parts = line[:-2].split()
                    if len(parts) >= 3:
                        entity, relation = parts[0], parts[1]
                        # Extract value from typed literal
                        value_str = parts[2]
                        if "^^" in value_str:
                            value_str = value_str.split("^^")[0].strip('"')
                        else:
                            value_str = value_str.strip('"')
                        try:
                            value = float(value_str)
                            # Check if entity and relation exist in vocab
                            if entity in inv_entity_vocab:
                                # For numerical relations, we may not have them in relation vocab yet
                                if relation not in inv_rel_vocab:
                                    inv_rel_vocab[relation] = len(inv_rel_vocab)
                                numerical_data.append({
                                    'entity': inv_entity_vocab[entity],
                                    'relation': inv_rel_vocab[relation],
                                    'value': value
                                })
                        except ValueError:
                            continue
                else:
                    # Try tab-separated format
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        entity, relation, value_str = parts[0], parts[1], parts[2]
                        try:
                            value = float(value_str)
                            if entity in inv_entity_vocab:
                                if relation not in inv_rel_vocab:
                                    inv_rel_vocab[relation] = len(inv_rel_vocab)
                                numerical_data.append({
                                    'entity': inv_entity_vocab[entity],
                                    'relation': inv_rel_vocab[relation],
                                    'value': value
                                })
                        except ValueError:
                            continue

        return numerical_data

    def process(self):
        # Load entity triples
        entity_file = os.path.join(self.mmkb_root, self.dataset_name,
                                   f"{self.dataset_name}_EntityTriples.txt")
        triplets, inv_entity_vocab, inv_rel_vocab = self.load_entity_triples(entity_file)

        # Load image index
        image_file = os.path.join(self.mmkb_root, self.dataset_name,
                                 f"{self.dataset_name}_ImageIndex.txt")
        entity_to_image = self.load_image_index(image_file, inv_entity_vocab)

        # Load numerical triples
        numerical_file = os.path.join(self.mmkb_root, self.dataset_name,
                                     f"{self.dataset_name}_NumericalTriples.txt")
        numerical_data = self.load_numerical_triples(numerical_file, inv_entity_vocab, inv_rel_vocab)

        # Split into train/valid/test (80/10/10)
        num_triplets = len(triplets)
        train_size = int(0.8 * num_triplets)
        valid_size = int(0.1 * num_triplets)

        train_triplets = triplets[:train_size]
        valid_triplets = triplets[train_size:train_size + valid_size]
        test_triplets = triplets[train_size + valid_size:]

        num_node = len(inv_entity_vocab)
        num_relations = len(inv_rel_vocab)

        # Create train graph
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])
        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])

        # Create valid/test graphs
        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_data = Data(
            edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
            target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
            num_relations=num_relations * 2
        )
        valid_data = Data(
            edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
            target_edge_index=valid_edges, target_edge_type=valid_etypes,
            num_relations=num_relations * 2
        )
        test_data = Data(
            edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
            target_edge_index=test_edges, target_edge_type=test_etypes,
            num_relations=num_relations * 2
        )

        # Store metadata
        metadata = {
            'inv_entity_vocab': inv_entity_vocab,
            'inv_rel_vocab': inv_rel_vocab,
            'entity_to_image': entity_to_image,
            'numerical_data': numerical_data,
            'image_h5_path': os.path.join(self.mmkb_root, self.dataset_name,
                                         f"{self.dataset_name}_ImageData.h5")
        }

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(([train_data], [valid_data], [test_data], metadata), self.processed_paths[0])

    def get(self, idx):
        """Get train/valid/test split by index."""
        return self.full_data[idx]

    def __getitem__(self, idx):
        """Get train/valid/test split by index."""
        return self.full_data[idx]

    def load_image_embeddings(self, entity_ids=None):
        """Load image embeddings for specified entities."""
        import h5py

        metadata = self.full_data[3]
        entity_to_image = metadata['entity_to_image']
        h5_path = metadata['image_h5_path']

        embeddings = {}
        with h5py.File(h5_path, 'r') as f:
            if entity_ids is None:
                entity_ids = list(entity_to_image.keys())

            for ent_id in entity_ids:
                if ent_id in entity_to_image:
                    img_id = entity_to_image[ent_id]
                    if img_id in f:
                        embeddings[ent_id] = torch.tensor(f[img_id][:])

        return embeddings

    def get_numerical_attributes(self):
        """Get all numerical attributes."""
        return self.full_data[3]['numerical_data']

    def get_metadata(self):
        """Get dataset metadata."""
        return self.full_data[3]


class DB15KMM(MultiModalKGDataset):
    """DB15K multimodal dataset."""
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        super().__init__(root, 'DB15K', transform, pre_transform, **kwargs)


class FB15KMM(MultiModalKGDataset):
    """FB15K multimodal dataset."""
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        super().__init__(root, 'FB15K', transform, pre_transform, **kwargs)
        # Load FB mid to name mapping
        self.mid2name = self._load_fb_mid2name()

    def _load_fb_mid2name(self):
        """Load Freebase MID to name mapping."""
        mid2name = {}
        mapping_file = "/data/gaoyisen/pfn/fb_mid2name.tsv"

        if not os.path.exists(mapping_file):
            print(f"Warning: FB mid2name file not found at {mapping_file}")
            return mid2name

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        mid, name = parts[0], parts[1]
                        mid2name[mid] = name
            print(f"Loaded {len(mid2name)} FB MID to name mappings")
        except Exception as e:
            print(f"Warning: Failed to load FB mid2name: {e}")

        return mid2name

    def get_entity_name(self, entity_id_or_mid):
        """Get human-readable name for an entity ID or MID."""
        # If it's an integer, get the MID from vocab first
        if isinstance(entity_id_or_mid, int):
            metadata = self.get_metadata()
            entity_vocab = {v: k for k, v in metadata['inv_entity_vocab'].items()}
            mid = entity_vocab.get(entity_id_or_mid, None)
            if mid is None:
                return f"Entity_{entity_id_or_mid}"
        else:
            mid = entity_id_or_mid

        # Look up the name
        return self.mid2name.get(mid, mid)


class YAGO15KMM(MultiModalKGDataset):
    """YAGO15K multimodal dataset."""
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):
        super().__init__(root, 'YAGO15K', transform, pre_transform, **kwargs)
