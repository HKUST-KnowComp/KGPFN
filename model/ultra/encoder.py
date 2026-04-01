import torch
from torch import nn

from . import layers
from .base_nbfnet import BaseNBFNet
import numpy as np
from torch.nn import functional as F

def static_positional_encoding(max_arity, input_dim):
    """
    Generate a static positional encoding.

    Args:
    - max_arity (int): Maximum arity for which to create positional encodings.
    - input_dim (int): Dimension of the input feature vector.

    Returns:
    - torch.Tensor: A tensor containing positional encodings for each position.
    """
    # Initialize the positional encoding matrix
    position = torch.zeros(max_arity + 1, input_dim)

    # Compute the positional encodings
    for pos in range(max_arity + 1):
        # position[pos, pos] = 1
        for i in range(0, input_dim, 2):
            position[pos, i] = np.sin(pos / (10000 ** ((2 * i) / input_dim)))
            if i + 1 < input_dim:
                position[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))


    return position

class StructureEncoder(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(StructureEncoder, self).__init__()

        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)

        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]  #[bs]
        
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
  
        entity_representations = self.entity_model(data, relation_representations, batch)
        
        return relation_representations, entity_representations

class StructureEncoderRelationAware(nn.Module):
    def __init__(self, rel_model_cfg, entity_model_cfg, entity_chunk_size=None):
        """
        entity_chunk_size: 若设置，将 unique (head, relation) 分块送入 bellmanford，避免 OOM。
                          None 表示不分块（一次性计算）。
        """
        super(StructureEncoderRelationAware, self).__init__()
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)
        self.entity_chunk_size = entity_chunk_size

    def get_representations(self, data, batch):
        """
        relation-aware 结构编码：
          - 输入 batch 形状 [B, S, 3]（每行共享同一 relation）；
          - relation 表示按每行计算一次；
          - entity 表示按每个三元组的 unique head 分块做 bellmanford，每块计算完立即
            gather 出对应三元组的 h/t embedding 后释放 node_feature，避免预分配
            [U, num_nodes, D] 大张量。

        返回：
          relation_representations: [B, R, D]
          h_embs:                   [B, S, D]
          t_embs:                   [B, S, D]
        """
        assert batch.dim() == 3 and batch.size(-1) == 3, (
            f"`batch` 期望形状 [B, S, 3]，实际 {tuple(batch.shape)}"
        )
        bsz, seq_len, _ = batch.shape
        device = batch.device

        # 每行共享 relation，取第一个作为查询 relation
        query_rels = batch[:, 0, 2]
        if not (batch[:, :, 2] == query_rels.unsqueeze(1)).all():
            raise ValueError("StructureEncoderRelationAware 要求每行 relation 一致")

        # 1) relation 表示: [B, num_relations, D]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)

        # 2) 为每行收集 unique head 及其全局偏移
        unique_heads_all = []
        unique_rels_all = []
        row_unique_inverse = []  # 每行 [S]，映射到该行 unique head 的局部索引
        row_offsets = []
        offset = 0

        for i in range(bsz):
            h_row = batch[i, :, 0]  # [S]
            h_unique, h_inv = torch.unique(h_row, sorted=False, return_inverse=True)
            row_unique_inverse.append(h_inv)
            row_offsets.append(offset)
            offset += h_unique.numel()
            unique_heads_all.append(h_unique)
            unique_rels_all.append(torch.full_like(h_unique, query_rels[i]))

        flat_h = torch.cat(unique_heads_all, dim=0)  # [U]
        flat_r = torch.cat(unique_rels_all, dim=0)   # [U]
        row_ids = torch.cat(
            [
                torch.full((uh.numel(),), i, dtype=torch.long, device=device)
                for i, uh in enumerate(unique_heads_all)
            ],
            dim=0,
        )  # [U]
        flat_relation_representations = relation_representations[row_ids]  # [U, R, D]

        # 预计算每个 (b, s) 位置对应的全局 unique-head 索引 [B, S]
        global_idx_2d = torch.stack(
            [row_unique_inverse[i] + row_offsets[i] for i in range(bsz)],
            dim=0,
        )  # [B, S]

        h_index = batch[:, :, 0]  # [B, S]
        t_index = batch[:, :, 1]  # [B, S]

        # 3) 分块 bellmanford：每块算完立即 gather h/t emb，释放 node_feature
        U = flat_h.size(0)
        chunk_size = self.entity_chunk_size if self.entity_chunk_size is not None else U
        h_embs: torch.Tensor | None = None
        t_embs: torch.Tensor | None = None

        for start in range(0, U, chunk_size):
            end = min(start + chunk_size, U)
            h_chunk = flat_h[start:end]
            r_chunk = flat_r[start:end]
            rel_chunk = flat_relation_representations[start:end]
            self.entity_model.query = rel_chunk
            for layer in self.entity_model.layers:
                layer.relation = rel_chunk
            out = self.entity_model.bellmanford(data, h_chunk, r_chunk)
            feat = out["node_feature"]  # [chunk, num_nodes, D]
            del out

            if h_embs is None:
                feat_dim = feat.size(-1)
                h_embs = feat.new_zeros(bsz, seq_len, feat_dim)
                t_embs = feat.new_zeros(bsz, seq_len, feat_dim)

            # 找出属于本 chunk [start, end) 的所有 (b, s) 位置，立即 gather
            mask = (global_idx_2d >= start) & (global_idx_2d < end)  # [B, S]
            if mask.any():
                b_idx, s_idx = mask.nonzero(as_tuple=True)
                lu = global_idx_2d[b_idx, s_idx] - start  # 在 feat 中的局部下标
                h_embs[b_idx, s_idx] = feat[lu, h_index[b_idx, s_idx]]
                t_embs[b_idx, s_idx] = feat[lu, t_index[b_idx, s_idx]]

            del feat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return relation_representations, h_embs, t_embs


    def forward(self, data, batch):
        relation_representations, h_embed, t_embed = self.get_representations(data, batch)
        # h_embed, t_embed: [B, S, D]
        # relation_representations: [B, R, D]

        # 提取 r 的 embedding
        _, _, r_index = batch.unbind(-1)  # [B, S]
        index_r = r_index.unsqueeze(-1).expand(-1, -1, relation_representations.size(-1))
        r_embed = relation_representations.gather(1, index_r)  # [B, S, D]

        return h_embed, t_embed, r_embed
        
    def get_score(self, data, batch):
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch)

        return score

    def get_mlp_scores(self, t_emb: torch.Tensor) -> torch.Tensor:
        """
        直接用已 gather 好的 tail embedding 计算 MLP 得分。

        Args:
            t_emb: [B, S, feature_dim] — forward() 返回的 t_embed（adapter 之前）
        Returns:
            scores: [B, S] MLP logit
        """
        return self.entity_model.mlp(t_emb).squeeze(-1)
# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    
    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output
    

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        # if self.concat_hidden:
        #     output = hiddens
        # else:
        #     output = hiddens[-1]
        # return output
        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch, return_score=False):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # 约定输入已是 tail-prediction 形式：(h, r) 固定，仅 t 变化
        # 因此跳过 negative_sample_to_tail，减少一次不必要的张量变换。
        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  
        feature = output["node_feature"] # (batch_size, num_nodes, feature_dim）
        if not return_score:
            return feature, None
        else:
            index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
            feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
            score = self.mlp(feature).squeeze(-1)
            return score.view(shape)


class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines
    Almost the same as EntityNBFNet except that 
    (1) we already get the initial node features at the forward pass time 
    and don't have to read the triples batch
    (2) we get `query` from the outer loop
    (3) we return a distribution over all nodes (assuming t_index = all nodes)
    """
    
    def bellmanford(self, data, node_features, query, separate_grad=False):
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, node_features, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(data, node_features, query)  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1) # (bs, num_nodes)
        return score  

    


