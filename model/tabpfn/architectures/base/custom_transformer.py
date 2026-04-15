"""
Custom TabPFN PerFeatureTransformer: stripped-down version for KG embeddings.

Changes from original PerFeatureTransformer:
  - Removed x preprocessing pipeline (NanHandling, FeatureTransform, per-group grouping)
  - Replaced x encoder with a simple MLP: Linear(input_dim, emsize) -> GELU -> Linear
  - Input x is already embedded: [B, S, F, D] (from structure encoder)
  - Kept y encoder (NanHandling + Linear), thinking tokens, transformer layers, decoder
"""
import torch
import torch.nn as nn
from typing import Any, Literal
from torch.amp import autocast

from tabpfn.architectures.base.config import ModelConfig
from model.tabpfn.architectures.base.layer import PerFeatureEncoderLayer
from tabpfn.architectures.base.thinking_tokens import AddThinkingTokens
from tabpfn.architectures.base.transformer import LayerStack
from tabpfn.architectures.encoders import (
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    TorchPreprocessingPipeline,
)
from tabpfn.architectures.shared.column_embeddings import load_column_embeddings


class CustomPerFeatureTransformer(nn.Module):
    def __init__(
        self,
        *,
        config: ModelConfig,
        structure_encoder_dim: int = 64,
        n_out: int = 10,
        activation: Literal["gelu", "relu"] = "gelu",
        zero_init: bool = True,
        **layer_kwargs: Any,
    ):
        super().__init__()

        self.ninp = config.emsize
        self.nhid_factor = config.nhid_factor
        nhid = self.ninp * self.nhid_factor
        self.features_per_group = config.features_per_group
        self.n_out = n_out
        self.structure_encoder_dim = structure_encoder_dim

        # x encoder: simple MLP replacing the full preprocessing pipeline
        self.x_encoder = nn.Sequential(
            nn.Linear(self.structure_encoder_dim, self.ninp),
            nn.GELU(),
            nn.Linear(self.ninp, self.ninp),
        )
        self.x_encoder_norm = nn.LayerNorm(self.ninp, elementwise_affine=False)

        # y encoder: kept from original (NanHandling + Linear)
        self.y_encoder = TorchPreprocessingPipeline(
            steps=[
                NanHandlingEncoderStep(
                    in_keys=("main",),
                    out_keys=("main", "nan_indicators"),
                ),
                LinearInputEncoderStep(
                    num_features=2,
                    emsize=config.emsize,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            ],
            output_key="output",
        )

        # Thinking tokens
        if config.num_thinking_rows > 0:
            self.add_thinking_tokens = AddThinkingTokens(
                num_thinking_rows=config.num_thinking_rows,
                emsize=config.emsize,
            )
        else:
            self.add_thinking_tokens = None

        # Transformer layers
        use_rope = (config.feature_positional_embedding == "rope")
        layer_creator = lambda: PerFeatureEncoderLayer(
            config=config,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            use_rope=use_rope,
            **layer_kwargs,
        )
        self.recompute_layer = config.recompute_layer
        self.transformer_encoder = LayerStack.of_repeated_layer(
            layer_creator=layer_creator,
            num_layers=config.nlayers,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_out),
        )

        # Feature positional embedding (also accepts "rope")
        self.feature_positional_embedding = config.feature_positional_embedding
        if self.feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(
                self.ninp // 4, self.ninp
            )
        elif self.feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(
                1_000, self.ninp
            )

        self.random_embedding_seed = config.seed
        self.pre_generated_column_embeddings = load_column_embeddings()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        *,
        task_type: str | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, F, D] pre-embedded features (from structure encoder)
               F = num feature groups, D = structure_encoder_dim
            y: [B, S] labels. y[:, eval_pos:] will be masked to NaN.
            eval_pos: split point — rows [0, eval_pos) are train, [eval_pos, S) are test.
        Returns:
            out: [B, N, n_out] predictions for test rows (N = S - eval_pos)
        """
        batch_size, seq_len, num_features, feat_dim = x.shape

        # x encoder: MLP per token
        embedded_x = self.x_encoder(x)              # [B, S, F, emsize]
        embedded_x = self.x_encoder_norm(embedded_x)

        # Feature positional embedding
        embedded_x = self._add_feature_pos_emb(embedded_x)

        # y processing: [B, S] -> [S, B, 1] for y_encoder
        y = y.clone()
        y[:, eval_pos:] = float("nan")
        if y.ndim == 2:
            y = y.unsqueeze(-1)          # [B, S, 1]
        y = y.transpose(0, 1)            # [S, B, 1]
        y_dict = {"main": y}

        single_eval_pos = eval_pos

        embedded_y = self.y_encoder(
            y_dict,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
        ).transpose(0, 1)  # [S, B, emsize] -> [B, S, emsize]

        # Concat: [B, S, F+1, emsize]
        embedded_input = torch.cat([embedded_x, embedded_y.unsqueeze(2)], dim=2)

        # Thinking tokens
        if self.add_thinking_tokens is not None:
            embedded_input, single_eval_pos = self.add_thinking_tokens(
                embedded_input, single_eval_pos,
            )

        # Transformer
        encoder_out = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
            recompute_layer=self.recompute_layer,
            save_peak_mem_factor=None,
        )

        # Decode: take last feature column (y column) of test rows
        test_encoder_out = encoder_out[:, single_eval_pos:, -1]  # [B, N, emsize]
        output = self.decoder(test_encoder_out)                   # [B, N, n_out]

        return output

    def _add_feature_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Add feature positional embeddings to x: [B, S, F, E]."""
        if self.feature_positional_embedding == "subspace":
            rng = torch.Generator(device=x.device).manual_seed(self.random_embedding_seed)
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4),
                device=x.device, dtype=x.dtype, generator=rng,
            )
            if embs.shape[1] == 48 and self.random_embedding_seed == 42:
                embs[:2000] = self.pre_generated_column_embeddings[:embs.shape[0]].to(
                    device=embs.device, dtype=embs.dtype
                )
            embs = self.feature_positional_embedding_embeddings(embs)
            x = x + embs[None, None]
        elif self.feature_positional_embedding == "learned":
            w = self.feature_positional_embedding_embeddings.weight
            rng = torch.Generator(device=x.device).manual_seed(self.random_embedding_seed)
            embs = w[torch.randint(0, w.shape[0], (x.shape[2],), generator=rng)]
            x = x + embs[None, None]
        return x
