import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Literal, List, Optional, Tuple


class KGPFN(nn.Module):
    def __init__(
        self,
        *,
        structure_encoder=None,
        semantic_encoder=None,
        feature_transformer,
        entity_dim: int = 64,
        relation_dim: int = 64,
        semantic_dim: int = 384,
        inverse_relation_semantic_mode: str = "text",
        enhance_structure: bool = False,
        seq_chunk_size: Optional[int] = None,
        context_label_correction: bool = False,
    ):
        super().__init__()
        assert structure_encoder is not None or semantic_encoder is not None, \
            "至少需要 structure_encoder 或 semantic_encoder 之一"
        assert not (enhance_structure and structure_encoder is None), \
            "enhance_structure=True 需要 structure_encoder"
        self.structure_encoder = structure_encoder
        self.semantic_encoder: Optional[Any] = semantic_encoder
        self.feature_transformer = feature_transformer
        assert entity_dim == relation_dim, "entity_dim and relation_dim must be the same"
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.inverse_relation_semantic_mode = inverse_relation_semantic_mode
        self.semantic_dim = semantic_dim
        self.hidden_dim = entity_dim
        self.enhance_structure = enhance_structure
        self.seq_chunk_size = seq_chunk_size
        # 是否启用 encoder 软标签修正（缓解 transductive 假负样本）
        self.context_label_correction = context_label_correction

        dropout_rate = getattr(self, 'dropout', 0.0)

        # 结构路模块：仅在有 structure_encoder 时创建
        if structure_encoder is not None:
            self.entity_adapter = nn.Sequential(
                nn.Linear(2 * self.entity_dim, self.hidden_dim),
                nn.Dropout(dropout_rate),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.relation_adapter = nn.Sequential(
                nn.Linear(self.relation_dim, self.hidden_dim),
                nn.Dropout(dropout_rate),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.structure_norm = nn.LayerNorm(self.hidden_dim)
            # Triple-level structure enhancement adapter.
            # Input: [TransE(h+r-t), DistMult(h*r*t), cos(h+r,t)] -> 3*D
            if enhance_structure:
                self.structure_enhance_adapter = nn.Sequential(
                    nn.Linear(3 * self.hidden_dim, 3 * self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(3 * self.hidden_dim, 3 * self.hidden_dim),
                )
                self.structure_enhance_norm = nn.LayerNorm(self.hidden_dim)

        # 文本路模块：仅在有 semantic_encoder 时创建
        if semantic_encoder is not None:
            self.text_adapter = nn.Sequential(
                nn.Linear(self.semantic_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.text_norm = nn.LayerNorm(self.hidden_dim)

    def _triples_to_embeddings(self, data, all_triples: torch.Tensor):
        """
        调用 structure_encoder.forward，返回 (h_emb, r_emb, t_emb)。
        返回的 t_emb [B, S, feature_dim] 在 adapter 之前，可直接送入
        structure_encoder.get_mlp_scores 计算结构得分。
        """
        h_emb, t_emb, r_emb = self.structure_encoder(data, all_triples)
        return h_emb, r_emb, t_emb

    def _parse_dual_input(self, query_x, context_x):
        # 新格式：{"id": ..., "text": ...}
        # 旧格式（兼容）：query_x 为 Tensor、context_x 为 List[Tensor]，此时 text=None
        if isinstance(query_x, dict):
            query_id = query_x["id"]
            query_text = query_x.get("text", None)
        else:
            query_id = query_x
            query_text = None

        if isinstance(context_x, dict):
            context_id = context_x["id"]
            context_text = context_x.get("text", None)
        else:
            context_id = context_x
            context_text = None

        return query_id, query_text, context_id, context_text

    def _encode_semantic(self, all_text_rows, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        all_text_rows: List[List[List[str]]], 形状 [B, S, 3]
          - B: batch
          - S: 序列长度 (= M + N)
          - 3: 每行三元组对应 [h_text, r_text, t_text]
        使用 SentenceTransformer 风格 model.encode 后 reshape 为 [B, S, 3, D]。
        """

        prefix = "the reverse relation of "
        inverse_mask_rows = []
        processed_rows = []
        for sample in all_text_rows:
            sample_mask = []
            sample_processed = []
            for row in sample:
                h_text, r_text, t_text = row
                is_inverse = False
                if (
                    self.inverse_relation_semantic_mode == "negate"
                    and isinstance(r_text, str)
                    and r_text.startswith(prefix)
                ):
                    r_text = r_text[len(prefix):]
                    is_inverse = True
                sample_processed.append([h_text, r_text, t_text])
                sample_mask.append(is_inverse)
            processed_rows.append(sample_processed)
            inverse_mask_rows.append(sample_mask)

        flat_text = []
        for sample in processed_rows:
            for row in sample:
                flat_text.extend(row)

        if hasattr(self.semantic_encoder, "encode"):
            sem = self.semantic_encoder.encode(
                flat_text,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        else:
            sem = self.semantic_encoder(flat_text)

        # SentenceTransformer.encode may return inference tensors that cannot be
        # saved by autograd in downstream trainable layers (e.g., Linear).
        sem = sem.to(device).clone()
        sem = sem.reshape(bsz, seq_len, 3, -1)  # [B, S, 3, D]
        if self.inverse_relation_semantic_mode == "negate":
            inv_mask = torch.tensor(inverse_mask_rows, dtype=torch.bool, device=device)  # [B, S]
            if inv_mask.any():
                rel_sem = sem[:, :, 1, :]
                rel_sem = torch.where(inv_mask.unsqueeze(-1), -rel_sem, rel_sem)
                sem[:, :, 1, :] = rel_sem
        return sem

    @staticmethod
    def _apply_label_correction(
        scores: torch.Tensor,
        context_y: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        根据已计算好的 encoder MLP 得分对上下文标签进行软修正：
          - 正样本（y≥1）中的最小得分作为阈值；
          - 负样本（y=0）中得分高于该阈值的，将标签改为 0.5。

        Args:
            scores:    [B, M] 预计算好的 MLP logit（由 get_mlp_scores 得到）
            context_y: B 个 [M] 上下文标签
        Returns:
            修正后的新标签列表（不修改原始张量）
        """
        new_y = []
        for i in range(scores.size(0)):
            y_i = context_y[i].float().clone().to(scores.device)
            pos_mask = y_i >= 1.0
            neg_mask = y_i == 0.0
            if pos_mask.any() and neg_mask.any():
                min_pos_score = scores[i][pos_mask].min()
                y_i[neg_mask & (scores[i] > min_pos_score)] = 0.5
            new_y.append(y_i)
        return new_y

    def _build_structure_aligned(
        self,
        data: Any,
        all_id_triples: torch.Tensor,
        num_context: int,
        context_y: List[torch.Tensor],
    ):
        """
        执行结构路编码，返回 (structure_aligned, context_y)。
        structure_aligned: [B, S, 3, D] 或 enhance 后的 [B, S, 6, D]
        """
        bsz, seq_len, _ = all_id_triples.shape
        h_emb, r_emb, t_emb = self._triples_to_embeddings(data, all_id_triples)

        if self.context_label_correction:
            with torch.no_grad():
                ctx_scores = self.structure_encoder.get_mlp_scores(
                    t_emb[:, :num_context]
                )
            context_y = self._apply_label_correction(ctx_scores, context_y)

        h_emb = self.entity_adapter(h_emb)
        r_emb = self.relation_adapter(r_emb)
        t_emb = self.entity_adapter(t_emb)
        id_feat = torch.stack([h_emb, r_emb, t_emb], dim=2)
        structure_aligned = self.structure_norm(id_feat)

        if self.enhance_structure:
            h_s = structure_aligned[:, :, 0, :]
            r_s = structure_aligned[:, :, 1, :]
            t_s = structure_aligned[:, :, 2, :]
            transe_feat = h_s + r_s - t_s
            distmult_feat = h_s * r_s * t_s
            cos_feat = (
                F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8)
                .unsqueeze(-1)
                .expand(-1, -1, self.hidden_dim)
            )
            enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)
            enh_delta = self.structure_enhance_adapter(enh_in).reshape(
                bsz, seq_len, 3, self.hidden_dim
            )
            enh_delta = self.structure_enhance_norm(enh_delta)
            structure_aligned = torch.cat([structure_aligned, enh_delta], dim=2)

        return structure_aligned, context_y

    def forward(
        self,
        data: Any,
        query_x,
        context_x,
        context_y: List[torch.Tensor],
        task_type: Literal["reg", "cls"] = "cls",
    ) -> torch.Tensor | dict:
        """
        支持双路输入：
          query_x = {
              "id": Tensor(B, N, 3),
              "text": List[List[List[str]]],  # [B, N, 3]
          }
          context_x = {
              "id": List[Tensor(M, 3)],       # 长度 B
              "text": List[List[List[str]]],  # [B, M, 3]
          }
          context_y: List[Tensor(M)]           # 长度 B

        其中 text 的最内层长度固定为 3，顺序为 [h_text, r_text, t_text]。
        模型内部会把 context/query 的 text 按序拼接成 [B, S, 3]（S=M+N），
        再用 semantic_encoder.encode(flatten_text) 得到 [B, S, 3, D]。

        兼容模式：
          - structure_encoder=None：纯语义路，text 必须提供
          - semantic_encoder=None ：纯结构路，id 必须提供
          - 两者均有：双路融合
        """

        query_id, query_text, context_id, context_text = self._parse_dual_input(query_x, context_x)

        bsz, num_query, _ = query_id.shape
        assert len(context_id) == bsz and len(context_y) == bsz, (
            f"context_x/context_y 长度应为 B={bsz}，实际为 {len(context_id)} / {len(context_y)}"
        )
        device = query_id.device
        num_context = context_id[0].size(0)
        for i in range(bsz):
            assert context_id[i].size(0) == num_context and context_y[i].size(0) == num_context, (
                "当前实现要求每个 batch 行的 context 长度一致"
            )

        # id 路拼接为 (B, M+N, 3)，无论是否用结构路都需要用于形状推断
        all_id_triples = torch.stack(
            [torch.cat([context_id[i].to(device), query_id[i].to(device)], dim=0) for i in range(bsz)],
            dim=0,
        )
        seq_len = all_id_triples.size(1)

        # 1) 结构路
        structure_aligned = None
        if self.structure_encoder is not None:
            structure_aligned, context_y = self._build_structure_aligned(
                data, all_id_triples, num_context, context_y
            )

        # 2) 文本路
        text_aligned = None
        if self.semantic_encoder is not None and query_text is not None and context_text is not None:
            all_text = [context_text[i] + query_text[i] for i in range(bsz)]  # [B, S, 3]
            text_feat = self._encode_semantic(all_text, bsz=bsz, seq_len=seq_len, device=device)
            text_aligned = self.text_norm(self.text_adapter(text_feat))

        # 3) 融合
        if structure_aligned is not None and text_aligned is not None:
            fused_x = torch.cat([structure_aligned, text_aligned], dim=2)  # [B, S, 6+, D]
        elif structure_aligned is not None:
            fused_x = structure_aligned
        else:
            assert text_aligned is not None, \
                "structure_encoder=None 时必须提供文本输入（query_text / context_text）"
            fused_x = text_aligned

        # y: 前 M 为 context 标签，后 N 为 query 占位 NaN
        y = torch.stack(
            [
                torch.cat(
                    [
                        context_y[i].to(device).to(torch.float32),
                        torch.full((num_query,), float("nan"), device=device, dtype=torch.float32),
                    ],
                    dim=0,
                )
                for i in range(bsz)
            ],
            dim=0,
        )
        eval_pos = num_context

        out = self.feature_transformer(fused_x, y, eval_pos=eval_pos, task_type=task_type)

        # reshape 为 [bsz, num_query]
        if out.dim() == 3:
            out = out.squeeze(-1)  # [B, N, 1] -> [B, N]
        if out.dim() == 1:
            out = out.view(bsz, num_query)  # [B*N] -> [B, N]
        elif out.dim() == 2 and out.size(0) == bsz * num_query:
            out = out.view(bsz, num_query)  # [B*N, 1] -> [B, N]

        return out  # [bsz, num_query]

    @torch.no_grad()
    def get_context_embeddings_cache(
        self,
        data: Any,
        context_ids: List[torch.Tensor],
        context_ys: List[torch.Tensor],
        context_texts: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        批量预计算 B 个上下文的 embedding。

        Args:
            data: 图数据
            context_ids: B 个上下文三元组，每个 [M, 3]
            context_ys: B 个上下文标签，每个 [M]
            context_texts: 可选的 B 个文本描述，每个 [[M, 3]]

        Returns:
            (fused_x, context_ys): [B, M, *, D] 的 embedding 和（可能修正后的）标签列表
        """
        B = len(context_ids)
        device = data.edge_index.device
        M = context_ids[0].size(0)

        for i in range(B):
            assert context_ids[i].size(0) == M, "所有 context 长度必须相同"

        all_ctx_ids = torch.stack([cid.to(device) for cid in context_ids], dim=0)  # [B, M, 3]

        # 结构路
        structure_aligned = None
        if self.structure_encoder is not None:
            structure_aligned, context_ys = self._build_structure_aligned(
                data, all_ctx_ids, M, context_ys
            )

        # 文本路
        text_aligned = None
        if self.semantic_encoder is not None and context_texts is not None:
            all_texts = [[context_texts[i][j] for j in range(M)] for i in range(B)]
            text_feat = self._encode_semantic(all_texts, bsz=B, seq_len=M, device=device)
            text_aligned = self.text_norm(self.text_adapter(text_feat))

        # 融合
        if structure_aligned is not None and text_aligned is not None:
            fused_x = torch.cat([structure_aligned, text_aligned], dim=2)
        elif structure_aligned is not None:
            fused_x = structure_aligned
        else:
            assert text_aligned is not None, \
                "structure_encoder=None 时 context_texts 不能为 None"
            fused_x = text_aligned

        return fused_x, context_ys  # [B, M, *, D] and (possibly relabeled) labels

    @torch.no_grad()
    def get_scores(
        self,
        data: Any,
        query_x,
        context_cache: torch.Tensor,
        context_y: List[torch.Tensor],
        task_type: Literal["reg", "cls"] = "reg",
    ) -> torch.Tensor:
        """
        根据 query 和缓存的 context embedding 计算分数。

        Args:
            data: 图数据
            query_x: [B, N, 3] Tensor 或 {"id": ..., "text": ...} dict
            context_cache: [B, M, *, D] 预计算的 context embedding
            context_y: B 个 [M] context 标签
            task_type: "reg" 或 "cls"

        Returns:
            scores: [B, N] 每个 query 的分数
        """
        query_id, query_text, _, _ = self._parse_dual_input(query_x, [])
        device = query_id.device
        B, N, _ = query_id.shape
        M = context_cache.size(1)

        # 结构路
        query_structure = None
        if self.structure_encoder is not None:
            h_emb, r_emb, t_emb = self._triples_to_embeddings(data, query_id)
            h_emb = self.entity_adapter(h_emb)
            r_emb = self.relation_adapter(r_emb)
            t_emb = self.entity_adapter(t_emb)
            query_id_feat = torch.stack([h_emb, r_emb, t_emb], dim=2)
            query_structure = self.structure_norm(query_id_feat)

            if self.enhance_structure:
                h_s = query_structure[:, :, 0, :]
                r_s = query_structure[:, :, 1, :]
                t_s = query_structure[:, :, 2, :]
                transe_feat = h_s + r_s - t_s
                distmult_feat = h_s * r_s * t_s
                cos_feat = (
                    F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8)
                    .unsqueeze(-1)
                    .expand(-1, -1, self.hidden_dim)
                )
                enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)
                enh_delta = self.structure_enhance_adapter(enh_in).reshape(B, N, 3, self.hidden_dim)
                enh_delta = self.structure_enhance_norm(enh_delta)
                query_structure = torch.cat([query_structure, enh_delta], dim=2)

        # 文本路
        query_text_aligned = None
        if self.semantic_encoder is not None and query_text is not None:
            text_feat = self._encode_semantic(query_text, bsz=B, seq_len=N, device=device)
            query_text_aligned = self.text_norm(self.text_adapter(text_feat))

        # 融合 query embedding
        if query_structure is not None and query_text_aligned is not None:
            query_fused = torch.cat([query_structure, query_text_aligned], dim=2)
        elif query_structure is not None:
            query_fused = query_structure
        else:
            assert query_text_aligned is not None, \
                "structure_encoder=None 时必须在 query_x 中提供 text"
            query_fused = query_text_aligned

        # 拼接 context 和 query
        full_fused = torch.cat([context_cache, query_fused], dim=1)  # [B, M+N, *, D]

        # 构建 y：context 标签 + query 占位 NaN
        y = torch.stack(
            [
                torch.cat([
                    context_y[i].to(device).to(torch.float32),
                    torch.full((N,), float("nan"), device=device, dtype=torch.float32),
                ], dim=0)
                for i in range(B)
            ],
            dim=0,
        )

        eval_pos = M
        out = self.feature_transformer(full_fused, y, eval_pos=eval_pos, task_type=task_type)

        if out.numel() == B * N:
            if out.dim() == 1:
                out = out.view(B, N)
            elif out.dim() == 2 and out.size(0) == B * N:
                out = out.view(B, N, -1).squeeze(-1)

        return out  # [B, N]


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
