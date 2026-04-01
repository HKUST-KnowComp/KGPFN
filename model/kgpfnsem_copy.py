import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Literal, List, Optional, Tuple


class KGPFN(nn.Module):
    def __init__(
        self,
        *,
        structure_encoder,
        semantic_encoder=None,
        feature_transformer,
        entity_dim: int = 64,
        relation_dim: int = 64,
        semantic_dim: int = 384,
        inverse_relation_semantic_mode: str = "text",
        enhance_structure: bool = False,
        seq_chunk_size: Optional[int] = None,
    ):
        super().__init__()
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

        # id / text 两路各自做 MLP 对齐，再做归一化
        self.structure_adapter = nn.Sequential(
            nn.Linear(self.entity_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.text_adapter = nn.Sequential(
            nn.Linear(self.semantic_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.structure_norm = nn.LayerNorm(self.hidden_dim)
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        # Triple-level structure enhancement adapter.
        # Input: [TransE(h+r-t), DistMult(h*r*t), cos(h+r,t)] -> 3*D
        # Output: per-token delta with shape [3, D].
       
        self.structure_enhance_adapter = nn.Sequential(
            nn.Linear(3*self.hidden_dim , 3*self.hidden_dim),
            nn.GELU(),
            nn.Linear(3*self.hidden_dim, 3*self.hidden_dim ),
            )
        self.structure_enhance_norm = nn.LayerNorm(self.hidden_dim)

    def _triples_to_embeddings(self, data, all_triples: torch.Tensor):
        """
        relation 仍按每个 batch 行计算一次；
        entity 按每个三元组 (x, y, r) 的 head=x 单独计算。
        seq_chunk_size 设置时，按序列维度分块计算，每块算完即提取并释放，避免 OOM。
        """
        bsz, seq_len, _ = all_triples.shape
        device = all_triples.device
        
        # 处理空输入的情况
        if seq_len == 0:
            empty_emb = torch.empty(bsz, 0, self.entity_dim, device=device)
            return empty_emb, empty_emb, empty_emb
        
        chunk_size = self.seq_chunk_size if self.seq_chunk_size is not None else seq_len

        h_emb = None
        r_emb = None
        t_emb = None

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            triples_chunk = all_triples[:, start:end, :]  # [B, chunk, 3]
            # print('triples_chunk:',triples_chunk.shape)
            relation_representations, entity_representations = self.structure_encoder(data, triples_chunk)
            # print('yes')
            chunk_len = triples_chunk.size(1)
            num_rel_nodes = relation_representations.size(1)
            r_idx_chunk = triples_chunk[:, :, 2].clamp(max=num_rel_nodes - 1)
            h_idx_chunk = triples_chunk[:, :, 0]
            t_idx_chunk = triples_chunk[:, :, 1]

            if entity_representations.dim() == 4:
                b_idx = torch.arange(bsz, device=device).view(bsz, 1).expand(bsz, chunk_len)
                s_idx = torch.arange(chunk_len, device=device).view(1, chunk_len).expand(bsz, chunk_len)
                h_chunk = entity_representations[b_idx, s_idx, h_idx_chunk, :]
                t_chunk = entity_representations[b_idx, s_idx, t_idx_chunk, :]
            else:
                h_chunk = entity_representations[torch.arange(bsz, device=device).unsqueeze(1), h_idx_chunk, :]
                t_chunk = entity_representations[torch.arange(bsz, device=device).unsqueeze(1), t_idx_chunk, :]

            r_chunk = relation_representations[
                torch.arange(bsz, device=device).unsqueeze(1), r_idx_chunk, :
            ]

            del entity_representations, relation_representations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if h_emb is None:
                h_emb = h_chunk.new_empty(bsz, seq_len, h_chunk.size(-1))
                r_emb = r_chunk.new_empty(bsz, seq_len, r_chunk.size(-1))
                t_emb = t_chunk.new_empty(bsz, seq_len, t_chunk.size(-1))
            h_emb[:, start:end], r_emb[:, start:end], t_emb[:, start:end] = h_chunk, r_chunk, t_chunk
            del h_chunk, r_chunk, t_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        # id 路输入拼接为 (B, M+N, 3)
        all_id_triples = torch.stack(
            [torch.cat([context_id[i].to(device), query_id[i].to(device)], dim=0) for i in range(bsz)],
            dim=0,
        )
        seq_len = all_id_triples.size(1)

        # 1) id 路: structure encoder -> 三元组特征 [B,S,3,D]
        h_emb, r_emb, t_emb = self._triples_to_embeddings(data, all_id_triples)
        id_feat = torch.stack([h_emb, r_emb, t_emb], dim=2)
        # 2) text 路: semantic encoder -> [B,S,3,D_sem]
        # 若 semantic_encoder=None，则自动退化为纯结构分支（忽略 text 输入）
        text_feat = None
        if self.semantic_encoder is not None and query_text is not None and context_text is not None:
            all_text = [context_text[i] + query_text[i] for i in range(bsz)]  # [B, S, 3]
            text_feat = self._encode_semantic(all_text, bsz=bsz, seq_len=seq_len, device=device)
           
        # 3) 融合
        # 仅在启用 semantic_encoder 时做 adapter + norm 融合；
        # 否则走纯结构分支，直接使用 id_feat。
        structure_aligned = self.structure_norm(self.structure_adapter(id_feat))
        if self.enhance_structure:
            h_s = structure_aligned[:, :, 0, :]
            r_s = structure_aligned[:, :, 1, :]
            t_s = structure_aligned[:, :, 2, :]

            # 1) TransE-style
            transe_feat = h_s + r_s - t_s # [B,S,D]
            # 2) DistMult-style (element-wise)
            distmult_feat = h_s * r_s * t_s # [B,S,D]
            # 3) RotatE-like cosine similarity between (h+r) and t
            cos_feat = F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8).unsqueeze(-1)
            cos_feat = cos_feat.expand(-1, -1, self.hidden_dim)

            enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)  # [B,S,3D]
            enh_delta = self.structure_enhance_adapter(enh_in).reshape(
                structure_aligned.size(0), structure_aligned.size(1), 3, self.hidden_dim
            )
            enh_delta = self.structure_enhance_norm(enh_delta)
            # Keep token layout [h, r, t] unchanged; enhance by residual update.
            structure_aligned = torch.cat([structure_aligned, enh_delta], dim=2)

        if text_feat is not None:
            text_aligned = self.text_norm(self.text_adapter(text_feat))
            # Concatenate along token axis -> [B, S, 6, D]
            fused_x = torch.cat([structure_aligned, text_aligned], dim=2)
        else:
            fused_x = structure_aligned

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
        return out

    def get_scores(
        self,
        data: Any,
        query_x,
        context_x,
        context_y: List[torch.Tensor],
        task_type: Literal["reg", "cls"] = "reg",
    ) -> torch.Tensor:
        out = self.forward(data, query_x, context_x, context_y, task_type)
        return torch.sigmoid(out)
    
    @torch.no_grad()
    def get_context_embeddings_batch(
        self,
        data: Any,
        context_ids: List[torch.Tensor],
        context_ys: List[torch.Tensor],
        context_texts: Optional[List] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        批量预计算 B 个上下文的 embedding。
        
        Args:
            data: 图数据
            context_ids: B 个上下文三元组，每个 [M, 3]
            context_ys: B 个上下文标签，每个 [M]
            context_texts: 可选的 B 个文本描述，每个 [[M, 3]]
            
        Returns:
            list of (context_fused, context_y_tensor)，每个为 [1, M, *, D] 和 [1, M]
        """
        B = len(context_ids)
        results = []
        
        for i in range(B):
            ctx_id = context_ids[i].unsqueeze(0)  # [1, M, 3]
            ctx_y = context_ys[i]
            M = ctx_id.size(1)
            device = ctx_id.device
            
            # 处理空上下文的情况 (M=0)
            if M == 0:
                # 创建空的特征张量
                token_dim = 6 if self.enhance_structure else 3
                if self.semantic_encoder is not None and context_texts is not None:
                    token_dim *= 2  # 结构和文本拼接
                empty_fused = torch.empty(1, 0, token_dim, self.hidden_dim, device=device)
                empty_y = torch.empty(1, 0, device=device, dtype=torch.long)
                results.append((empty_fused, empty_y))
                continue
            
            # 计算结构特征
            h_emb, r_emb, t_emb = self._triples_to_embeddings(data, ctx_id)
            id_feat = torch.stack([h_emb, r_emb, t_emb], dim=2)  # [1, M, 3, D]
            
            # 结构特征对齐
            structure_aligned = self.structure_norm(self.structure_adapter(id_feat))
            
            # 结构增强（与 forward 保持一致）
            if self.enhance_structure:
                h_s = structure_aligned[:, :, 0, :]
                r_s = structure_aligned[:, :, 1, :]
                t_s = structure_aligned[:, :, 2, :]
                
                transe_feat = h_s + r_s - t_s
                distmult_feat = h_s * r_s * t_s
                cos_feat = F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8).unsqueeze(-1)
                cos_feat = cos_feat.expand(-1, -1, self.hidden_dim)
                
                enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)
                enh_delta = self.structure_enhance_adapter(enh_in).reshape(1, M, 3, self.hidden_dim)
                enh_delta = self.structure_enhance_norm(enh_delta)
                structure_aligned = torch.cat([structure_aligned, enh_delta], dim=2)  # [1, M, 6, D]
            
            # 可选：添加文本特征
            if self.semantic_encoder is not None and context_texts is not None:
                text_feat = self._encode_semantic([context_texts[i]], bsz=1, seq_len=M, device=device)
                text_aligned = self.text_norm(self.text_adapter(text_feat))  # [1, M, 3, D]
                if self.enhance_structure:
                    # 文本也复制一份，与 query 保持一致
                    text_aligned = torch.cat([text_aligned, text_aligned], dim=2)  # [1, M, 6, D]
                fused_x = torch.cat([structure_aligned, text_aligned], dim=2)
            else:
                fused_x = structure_aligned
            
            y_tensor = ctx_y.unsqueeze(0) if ctx_y.dim() == 1 else ctx_y
            results.append((fused_x, y_tensor))
        
        return results
    
    @torch.no_grad()
    def get_query_hr_embeddings_batch(
        self,
        data: Any,
        h_ids: List[int],
        r_ids: List[int],
        query_texts: Optional[List[List[str]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        批量预计算 B 个 (h, r) 对的查询嵌入，用于 tail-only 评测。
        
        Args:
            data: 图数据
            h_ids: B 个 head entity id
            r_ids: B 个 relation id
            query_texts: 可选的 B 个文本，每个为 [h_text, r_text, t_text]
            
        Returns:
            hr_structure_list: B 个 [1, 1, *, D] 结构特征
            hr_text_list: B 个 [1, 1, *, D] 或 None 文本特征
        """
        device = data.edge_index.device
        B = len(h_ids)
        
        # 构建 batch 化的三元组 [B, 1, 3]
        triples = torch.tensor([[h_ids[i], 0, r_ids[i]] for i in range(B)], device=device).unsqueeze(1)
        
        # 批量计算结构特征
        h_emb, r_emb, t_emb = self._triples_to_embeddings(data, triples)
        id_feat = torch.stack([h_emb, r_emb, t_emb], dim=2)  # [B, 1, 3, D]
        
        # 结构对齐
        structure_aligned = self.structure_norm(self.structure_adapter(id_feat))  # [B, 1, 3, D]
        
        # 结构增强
        if self.enhance_structure:
            h_s = structure_aligned[:, :, 0, :]
            r_s = structure_aligned[:, :, 1, :]
            t_s = structure_aligned[:, :, 2, :]
            
            transe_feat = h_s + r_s - t_s
            distmult_feat = h_s * r_s * t_s
            cos_feat = F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8).unsqueeze(-1)
            cos_feat = cos_feat.expand(-1, -1, self.hidden_dim)
            
            enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)
            enh_delta = self.structure_enhance_adapter(enh_in).reshape(B, 1, 3, self.hidden_dim)
            enh_delta = self.structure_enhance_norm(enh_delta)
            structure_aligned = torch.cat([structure_aligned, enh_delta], dim=2)  # [B, 1, 6, D]
        
        # 拆分回列表
        hr_structure_list = [structure_aligned[i:i+1, :, :, :] for i in range(B)]
        
        # 文本特征（批量编码）
        hr_text_list = [None] * B
        if self.semantic_encoder is not None and query_texts is not None:
            # 批量编码所有文本 [B, 1, 3]
            all_texts = [[query_texts[i]] for i in range(B)]  # [[text1], [text2], ...]
            text_feat = self._encode_semantic(all_texts, bsz=B, seq_len=1, device=device)  # [B, 1, 3, D]
            text_aligned = self.text_norm(self.text_adapter(text_feat))
            if self.enhance_structure:
                text_aligned = torch.cat([text_aligned, text_aligned], dim=2)  # [B, 1, 6, D]
            # 拆分
            hr_text_list = [text_aligned[i:i+1, :, :, :] for i in range(B)]
        
        return hr_structure_list, hr_text_list
    
    @torch.no_grad()
    def fast_score_with_cached(
        self,
        data: Any,
        h_id: int,
        r_id: int,
        candidate_t_ids: torch.Tensor,
        context_fused: torch.Tensor,
        context_y: torch.Tensor,
        hr_structure_aligned: torch.Tensor,
        hr_text_aligned: Optional[torch.Tensor] = None,
        query_text_template: Optional[List[str]] = None,
        task_type: Literal["reg", "cls"] = "reg",
    ) -> torch.Tensor:
        """
        利用缓存的上下文和 (h,r) 嵌入，快速为多个 tail 候选打分。
        
        Args:
            data: 图数据
            h_id: head entity id
            r_id: relation id
            candidate_t_ids: [num_candidates] tail 候选 id
            context_fused: [1, M, *, D] 预计算的上下文特征
            context_y: [1, M] 上下文标签
            hr_structure_aligned: [1, 1, *, D] 预计算的 (h,r) 结构特征
            hr_text_aligned: [1, 1, *, D] 或 None 预计算的 (h,r) 文本特征
            query_text_template: 查询文本模板 [h_text, r_text, t_text]
            task_type: "reg" 或 "cls"
            
        Returns:
            scores: [num_candidates] 每个 tail 候选的分数
        """
        device = data.edge_index.device
        num_candidates = candidate_t_ids.size(0)
        M = context_fused.size(1)
        
        # 构建查询三元组 [h, t, r] 对所有候选 t
        h_ids = torch.full_like(candidate_t_ids, h_id)
        r_ids = torch.full_like(candidate_t_ids, r_id)
        query_triples = torch.stack([h_ids, candidate_t_ids, r_ids], dim=1).unsqueeze(0)  # [1, N, 3]
        
        # 计算每个候选 t 的嵌入（这里可以优化为批量计算）
        h_emb, r_emb, t_emb = self._triples_to_embeddings(data, query_triples)
        query_id_feat = torch.stack([h_emb, r_emb, t_emb], dim=2)  # [1, N, 3, D]
        
        # 对齐查询特征
        query_structure = self.structure_norm(self.structure_adapter(query_id_feat))
        
        # 查询结构增强
        if self.enhance_structure:
            h_s = query_structure[:, :, 0, :]
            r_s = query_structure[:, :, 1, :]
            t_s = query_structure[:, :, 2, :]
            
            transe_feat = h_s + r_s - t_s
            distmult_feat = h_s * r_s * t_s
            cos_feat = F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8).unsqueeze(-1)
            cos_feat = cos_feat.expand(-1, -1, self.hidden_dim)
            
            enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)
            enh_delta = self.structure_enhance_adapter(enh_in).reshape(1, num_candidates, 3, self.hidden_dim)
            enh_delta = self.structure_enhance_norm(enh_delta)
            query_structure = torch.cat([query_structure, enh_delta], dim=2)  # [1, N, 6, D]
        
        # 查询文本特征（如果使用）
        query_text_feat = None
        if self.semantic_encoder is not None and query_text_template is not None:
            # 为每个候选构建文本
            h_text, r_text, _ = query_text_template
            query_texts = [[h_text, r_text, f"entity_{t_id}"] for t_id in candidate_t_ids.tolist()]
            text_feat = self._encode_semantic([query_texts], bsz=1, seq_len=num_candidates, device=device)
            query_text_feat = self.text_norm(self.text_adapter(text_feat))
            if self.enhance_structure:
                query_text_feat = torch.cat([query_text_feat, query_text_feat], dim=2)
        
        # 融合查询特征
        if query_text_feat is not None:
            query_fused = torch.cat([query_structure, query_text_feat], dim=2)
        else:
            query_fused = query_structure
        
        # 拼接上下文和查询
        # context_fused: [1, M, *, D]
        # query_fused: [1, N, *, D] 
        # 需要确保 dim 一致
        full_fused = torch.cat([context_fused, query_fused], dim=1)  # [1, M+N, *, D]
        
        # 构建 y：上下文标签 + query 占位 NaN
        num_query = num_candidates
        y = torch.cat([
            context_y[0].to(device).to(torch.float32),
            torch.full((num_query,), float("nan"), device=device, dtype=torch.float32),
        ], dim=0).unsqueeze(0)  # [1, M+N]
        
        eval_pos = M
        
        # 通过 transformer
        out = self.feature_transformer(full_fused, y, eval_pos=eval_pos, task_type=task_type)
        return torch.sigmoid(out)
    
    @torch.no_grad()
    def fast_score_batch(
        self,
        data: Any,
        t_batch_chunk: torch.Tensor,
        context_cache,
        hr_structure_list,
        hr_text_list,
        task_type="reg",
    ):
        """
        批量为 B 个查询的 chunk 个 tail 候选打分。
        
        Args:
            t_batch_chunk: [B, N, 3] tail 候选三元组 (h, t, r)，每行共享 h,r
            context_cache: {i: (context_fused, context_y)} 每行的上下文缓存
            hr_structure_list: [B] 每行的预计算 (h,r) 结构特征 [1, 1, *, D]
            hr_text_list: [B] 每行的预计算 (h,r) 文本特征 [1, 1, *, D] 或 None
            
        Returns:
            scores: [B, N] 每个查询对每个 tail 候选的分数
        """
        device = t_batch_chunk.device
        B, N, _ = t_batch_chunk.shape
        
        # 1) 只计算 t 的嵌入（h 和 r 从预缓存获取）
        # 构造 dummy 三元组用于获取结构表示 [B, N, 3]
        h_emb, r_emb, t_emb = self._triples_to_embeddings(data, t_batch_chunk)
        # t_emb: [B, N, D]
        
        # 2) 为每个 batch 行构建查询特征，使用预计算的 h, r + 新计算的 t
        scores_list = []
        for i in range(B):
            # 获取该行的上下文
            context_fused, context_y = context_cache[i]  # [1, M, *, D], [1, M]
            M = context_fused.size(1)
            
            # 获取预计算的 (h, r) 结构特征
            hr_structure = hr_structure_list[i]  # [1, 1, *, D]
            D = hr_structure.size(3)
            
            # 提取预计算的 h 和 r
            h_cached = hr_structure[:, :, 0:1, :]  # [1, 1, 1, D]
            r_cached = hr_structure[:, :, 1:2, :]  # [1, 1, 1, D]
            
            # 计算当前行所有 tail 候选的 t 特征
            t_features = t_emb[i:i+1, :, :].unsqueeze(2)  # [1, N, 1, D]
            t_aligned = self.structure_norm(self.structure_adapter(t_features))  # [1, N, 1, D]
            
            # 构建完整的三元组特征 [h_cached, r_cached, t_aligned]
            h_expanded = h_cached.expand(1, N, -1, -1)  # [1, N, 1, D]
            r_expanded = r_cached.expand(1, N, -1, -1)  # [1, N, 1, D]
            
            query_structure = torch.cat([h_expanded, r_expanded, t_aligned], dim=2)  # [1, N, 3, D]
            
            # 结构增强
            if self.enhance_structure:
                h_s = query_structure[:, :, 0, :]
                r_s = query_structure[:, :, 1, :]
                t_s = query_structure[:, :, 2, :]
                
                transe_feat = h_s + r_s - t_s
                distmult_feat = h_s * r_s * t_s
                cos_feat = F.cosine_similarity(h_s + r_s, t_s, dim=-1, eps=1e-8).unsqueeze(-1)
                cos_feat = cos_feat.expand(-1, -1, self.hidden_dim)
                
                enh_in = torch.cat([transe_feat, distmult_feat, cos_feat], dim=-1)
                enh_delta = self.structure_enhance_adapter(enh_in).reshape(1, N, 3, self.hidden_dim)
                enh_delta = self.structure_enhance_norm(enh_delta)
                query_structure = torch.cat([query_structure, enh_delta], dim=2)  # [1, N, 6, D]
            
            # 添加文本特征（使用预计算的 hr_text）
            query_fused = query_structure
            hr_text = hr_text_list[i]
            if hr_text is not None and self.semantic_encoder is not None:
                hr_text_expanded = hr_text.expand(1, N, -1, -1)  # [1, N, *, D]
                query_fused = torch.cat([query_fused, hr_text_expanded], dim=2)
            
            # 拼接上下文和查询
            full_fused = torch.cat([context_fused, query_fused], dim=1)  # [1, M+N, *, D]
            
            # 构建 y（处理 M=0 的情况）
            if M > 0:
                y = torch.cat([
                    context_y[0].to(device).to(torch.float32),
                    torch.full((N,), float("nan"), device=device, dtype=torch.float32),
                ], dim=0).unsqueeze(0)  # [1, M+N]
            else:
                # M=0，只有查询部分
                y = torch.full((1, N), float("nan"), device=device, dtype=torch.float32)
            
            # 通过 transformer
            out = self.feature_transformer(full_fused, y, eval_pos=M, task_type=task_type)
            # out 可能是 [1, N, 1] 或 [1, N]，确保 squeeze 到 [N]
            scores = out.squeeze()
            if scores.dim() == 0:  # 标量情况 (N=1)
                scores = scores.unsqueeze(0)
            elif scores.dim() == 2:  # [1, N, 1] squeeze 后变成 [N, 1]
                scores = scores.squeeze(-1)
            scores_list.append(scores)  # [N]
        
        return torch.stack(scores_list, dim=0)  # [B, N]