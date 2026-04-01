import os
import csv
import shutil
import json
import time
import hashlib
import torch
import yaml
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
    versions = ['mt', 'health', 'tax']

    def __init__(self, **kwargs):
        assert kwargs['dataset_version'] in self.versions, f"unknown version {kwargs['version']}, available: {self.versions}"
        super(WikiTopicsMT1, self).__init__(**kwargs)

class WikiTopicsMT2(WikiTopics):

    name = "WikiTopics-MT2"
    prefix = "wikidata_%sv1"
    versions = ['mt2', 'org', 'sci']

    def __init__(self, **kwargs):
        super(WikiTopicsMT2, self).__init__(**kwargs)

class WikiTopicsMT3(WikiTopics):

    name = "WikiTopics-MT3"
    prefix = "wikidata_%sv2"
    versions = ['mt3', 'art', 'infra']

    def __init__(self, **kwargs):
        super(WikiTopicsMT3, self).__init__(**kwargs)

class WikiTopicsMT4(WikiTopics):

    name = "WikiTopics-MT4"
    prefix = "wikidata_%sv2"
    versions = ['mt4', 'sci', 'health']

    def __init__(self, **kwargs):
        super(WikiTopicsMT4, self).__init__(**kwargs)

# a joint dataset for pre-training ULTRA on several graphs
class JointDataset(InMemoryDataset): 


    datasets_map = {
    #     #共16个transductive,使用11个训练(去掉子集)
    #     'FB15k237': FB15k237, #包含了fb15k237的子图 FB15k237_10, FB15k237_20, FB15k237_50 共4个
    #     'WN18RR': WN18RR,
    #     'CoDExSmall': CoDExSmall,
    #     'CoDExMedium': CoDExMedium,
    #     'CoDExLarge': CoDExLarge, #使用codexlarge即可
    #     'NELL995': NELL995,
    #     'ConceptNet100k': ConceptNet100k,
    #     'DBpedia100k': DBpedia100k,
    #     'YAGO310': YAGO310,
    #     'AristoV4': AristoV4,
    #     'Hetionet': Hetionet,
    #     'WDsinger': WDsinger,
    #     'NELL23k': NELL23k, 
    #    # inductive  new nodes no new relations 18个
    #    # FB15k237Inductiv(4个),wn18rrinductive(4个),nellinductive(4个)
    #    'ILPC2022': ILPC2022, #small ,large 2个
       'HM': HM, #1k,3k,5k,indigo 4个
       # inductive  new nodes new relations 共23个  (13个 fb,wn,nl )
       'WikiTopicsMT1': WikiTopicsMT1, # tax, health 2个
       'WikiTopicsMT2': WikiTopicsMT2, #org, sci 2个
       'WikiTopicsMT3': WikiTopicsMT3, #art, infra 2个
       'WikiTopicsMT4': WikiTopicsMT4, #sci, health 2个
       'Metafam': Metafam,
       'FBNELL': FBNELL,
    }

    def __init__(self, root, graphs, transform=None, pre_transform=None):
        # Support:
        # - graphs: "all" / ["all"] -> load all keys in datasets_map
        # - graphs item: "Name" -> if class has `versions`, expand to all versions
        # - graphs item: "Name:version" -> load a specific version
        if isinstance(graphs, str):
            graphs = [graphs]
        graphs = list(graphs)

        if len(graphs) == 1 and str(graphs[0]).strip().lower() == "all":
            graphs = list(self.datasets_map.keys())

        expanded_specs: list[str] = []
        for spec in graphs:
            spec_s = str(spec).strip()
            if not spec_s:
                continue
            if spec_s.lower() == "all":
                expanded_specs.extend(list(self.datasets_map.keys()))
                continue

            if ":" in spec_s:
                ds_name, ds_version = spec_s.split(":", 1)
                ds_name = ds_name.strip()
                ds_version = ds_version.strip()
            else:
                ds_name, ds_version = spec_s, None

            ds_cls = self.datasets_map.get(ds_name)
            if ds_cls is None:
                raise KeyError(
                    f"Unknown dataset key '{ds_name}' for JointDataset. "
                    f"Available keys: {sorted(self.datasets_map.keys())}"
                )

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