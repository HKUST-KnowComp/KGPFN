from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class AddThinkingRows(Module):
    """Prepends learnable thinking rows to the row representations before ICL.

    Args:
        num_thinking_rows: Number of thinking rows to prepend.
        d_model: Dimension of row representations.
    """

    def __init__(self, num_thinking_rows: int, d_model: int) -> None:
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        self.row_token_values = Parameter(torch.empty(num_thinking_rows, d_model))
        torch.nn.init.normal_(self.row_token_values)

    def forward(self, R: Tensor, train_size: int) -> tuple[Tensor, int]:
        """Prepend thinking rows to R and adjust train_size.

        Args:
            R: (B, T, D)
            train_size: number of training rows

        Returns:
            (R with thinking rows prepended, updated train_size)
        """
        B = R.shape[0]
        thinking = self.row_token_values.unsqueeze(0).expand(B, -1, -1)
        R = torch.cat([thinking, R], dim=1)
        train_size += self.num_thinking_rows
        return R, train_size
