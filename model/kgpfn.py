import torch
import torch.nn as nn

from typing import Any, Literal, List


class KGPFN(nn.Module):
    def __init__(self,
                 *,
                 structure_encoder,
                 feature_transformer,
                 entity_dim: int = 64,
                 relation_dim: int = 64,
                 ):
        super().__init__()
        self.structure_encoder = structure_encoder
        self.feature_transformer = feature_transformer
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

    def _triples_to_embeddings(self, data, all_triples: torch.Tensor):
        """
        relation 按每个 batch 行计算；
        entity 可按每个三元组独立计算（当 structure_encoder 返回 [B,S,num_nodes,D] 时）。
        返回: h_emb (B, S, entity_dim), r_emb (B, S, relation_dim), t_emb (B, S, entity_dim)
        """
        B, S, _ = all_triples.shape
        device = all_triples.device
        relation_representations, entity_representations = self.structure_encoder(data, all_triples)
        num_rel_nodes = relation_representations.size(1)
        r_idx = all_triples[:, :, 2].clamp(max=num_rel_nodes - 1)  # (B, S)
        r_emb = relation_representations[
            torch.arange(B, device=device).unsqueeze(1), r_idx, :
        ]  # (B, S, relation_dim)
        h_idx = all_triples[:, :, 0]  # (B, S)
        t_idx = all_triples[:, :, 1]

        if entity_representations.dim() == 4:
            # [B, S, num_nodes, D]：每个三元组一份实体表示
            b_index = torch.arange(B, device=device).view(B, 1).expand(B, S)
            s_index = torch.arange(S, device=device).view(1, S).expand(B, S)
            h_emb = entity_representations[b_index, s_index, h_idx, :]
            t_emb = entity_representations[b_index, s_index, t_idx, :]
        else:
            # [B, num_nodes, D]：每个 batch 行共享一份实体表示（旧行为）
            h_emb = entity_representations[
                torch.arange(B, device=device).unsqueeze(1), h_idx, :
            ]  # (B, S, entity_dim)
            t_emb = entity_representations[
                torch.arange(B, device=device).unsqueeze(1), t_idx, :
            ]  # (B, S, entity_dim)
        return h_emb, r_emb, t_emb

    def forward(
        self,
        data: Any,
        query_x: torch.Tensor,
        context_x: List[torch.Tensor],
        context_y: List[torch.Tensor],
        task_type: Literal["reg", "cls"] = "cls",
    ) -> torch.Tensor | dict:
        """
        query_x: (B, N, 3)，每个 batch 行有 N 个 query 三元组 (h, t, r)
        context_x: 长度为 B 的 list，context_x[i] 形状 (M, 3)
        context_y: 长度为 B 的 list，context_y[i] 形状 (M,)
        每个 batch 行最终拼成 [context(M), query(N)]，序列长度 S=M+N。
        """
        B, N, _ = query_x.shape
        assert len(context_x) == B and len(context_y) == B, (
            f"context_x/context_y 长度应为 B={B}，实际为 {len(context_x)} / {len(context_y)}"
        )
        device = query_x.device
        M = context_x[0].size(0)
        for i in range(B):
            assert context_x[i].size(0) == M and context_y[i].size(0) == M, (
                "当前实现要求每个 batch 行的 context 长度一致"
            )

        # 拼接成 (B, M+N, 3)：第 i 行为 [context_x[i](M,3), query_x[i](N,3)]
        all_triples = torch.stack(
            [
                torch.cat(
                    [context_x[i].to(device), query_x[i].to(device)],
                    dim=0,
                )
                for i in range(B)
            ],
            dim=0,
        )  # (B, M+N, 3)

        # 结构 encoder：根据 all_triples 得到三元组的 (h, r, t) 表征
        h_emb, r_emb, t_emb = self._triples_to_embeddings(
            data, all_triples
        )  # (B, M+N, dim) 各
        # 使用当前格式作为 x：三元组的每一项 (h, r, t) 看作一个 feature
        triple_feat = torch.stack(
            [h_emb, r_emb, t_emb], dim=2
        )  # (B, M+N, 3, dimension)
        # 直接符合 CustomFeaturesTransformer 期望的 [batch, seq, feature, dimension]
        x = triple_feat  # (B, seq=M+N, feature=3, dim)

        # y: 每条序列前 M 个位置是 context 标签，后 N 个位置是 query 的 NaN 占位
        y = torch.stack(
            [
                torch.cat(
                    [
                        context_y[i].to(device).to(torch.float32),
                        torch.full((N,), float("nan"), device=device, dtype=torch.float32),
                    ],
                    dim=0,
                )
                for i in range(B)
            ],
            dim=0,
        )  # (B, M+N)
        eval_pos = M  # 每条序列前 M 为 context，后 N 为 query

        out = self.feature_transformer(x, y, eval_pos=eval_pos, task_type=task_type)
        return out
    
    def get_scores(self, data: Any, query_x: torch.Tensor, context_x: List[torch.Tensor], context_y: List[torch.Tensor], task_type: Literal["reg", "cls"] = "reg") -> torch.Tensor:
        out = self.forward(data, query_x, context_x, context_y, task_type)
        scores = torch.sigmoid(out)
        return scores