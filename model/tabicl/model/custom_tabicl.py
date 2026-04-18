"""
CustomTabICL: stripped-down TabICL for KG embeddings.

Changes from original TabICL:
  - Skips ColEmbedding (input is already [B, S, F, D] from structure encoder)
  - Adds a linear adapter: D -> embed_dim
  - Keeps RowInteraction + ICLearning unchanged
  - Replaces the original decoder with a 2-layer MLP outputting score dim=1
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from typing import Literal

# Make tabicl importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../tabicl/src"))

from tabicl.model.interaction import RowInteraction
from tabicl.model.learning import ICLearning
from .thinking_tokens import AddThinkingRows


class CustomTabICL(nn.Module):
    def __init__(
        self,
        *,
        structure_encoder_dim: int = 64,
        # tabicl arch params (loaded from yaml)
        embed_dim: int = 128,
        ff_factor: int = 2,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_rope_interleaved: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        icl_ssmax: str = "qassmax-mlp-elementwise",
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
        num_thinking_rows: int = 0,
        **kwargs,  # absorb unused yaml keys (col_*, num_quantiles, max_classes, etc.)
    ):
        super().__init__()
        icl_dim = embed_dim * row_num_cls

        # Adapter: structure_encoder_dim -> embed_dim (no feature_group needed)
        self.x_adapter = nn.Linear(structure_encoder_dim, embed_dim)

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            rope_interleaved=row_rope_interleaved,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            recompute=recompute,
        )

        # ICLearning with max_classes=0 (regression mode) so y_encoder is nn.Linear(1, icl_dim)
        # We override the decoder below with our own 2-layer MLP -> 1
        self.icl_predictor = ICLearning(
            out_dim=1,
            max_classes=0,
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=icl_ssmax,
            recompute=recompute,
        )
        # Replace ICLearning's decoder with our 2-layer MLP -> score (dim=1)
        self.icl_predictor.decoder = nn.Sequential(
            nn.Linear(icl_dim, icl_dim),
            nn.GELU(),
            nn.Linear(icl_dim, 1),
        )

        self.num_thinking_rows = num_thinking_rows
        if num_thinking_rows > 0:
            self.thinking_rows = AddThinkingRows(num_thinking_rows, icl_dim)
        else:
            self.thinking_rows = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        task_type: Literal["reg", "cls"] = "reg",
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, F, D]  pre-embedded features from structure encoder
            y: [B, S]        labels; y[:, eval_pos:] are NaN (query)
            eval_pos: int    split point
        Returns:
            out: [B, N, 1]   scores for query rows (N = S - eval_pos)
        """
        B, S, F, D = x.shape

        # Adapt D -> embed_dim, then flatten F into the feature dim expected by RowInteraction
        x = self.x_adapter(x)          # [B, S, F, embed_dim]

        # RowInteraction expects [B, T, H+C, E] where C = num_cls (prepended by the module itself)
        # We pass [B, S, F, embed_dim] directly — RowInteraction prepends CLS tokens internally
        R = self.row_interactor(x)      # [B, S, icl_dim]

        # ICLearning: y_train = labels for context rows only
        y_train = y[:, :eval_pos]       # [B, eval_pos]

        if self.thinking_rows is not None:
            R, new_train_size = self.thinking_rows(R, eval_pos)
            y_train_icl = torch.zeros(
                B, new_train_size, dtype=y_train.dtype, device=y_train.device
            )
            y_train_icl[:, self.num_thinking_rows:] = y_train
        else:
            y_train_icl = y_train

        out = self.icl_predictor(R, y_train_icl)  # [B, N, 1]

        return out


def build_custom_tabicl(config: dict, structure_encoder_dim: int = 64) -> CustomTabICL:
    return CustomTabICL(structure_encoder_dim=structure_encoder_dim, **config)


def load_custom_tabicl(
    config_yaml: str,
    ckpt_path: str,
    structure_encoder_dim: int = 64,
    map_location: str = "cpu",
) -> CustomTabICL:
    with open(config_yaml) as f:
        config = yaml.safe_load(f)
    model = build_custom_tabicl(config, structure_encoder_dim=structure_encoder_dim)
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)
    return model
