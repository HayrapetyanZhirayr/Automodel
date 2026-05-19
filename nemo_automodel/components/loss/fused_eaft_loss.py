# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunked fused linear + EAFT loss via custom autograd (Liger-style).

Same ``forward`` contract as :class:`~nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy`
(``hidden_states``, ``labels``, ``lm_weight``, ``num_label_tokens``), but uses chunked
``grad_and_value`` over the LM projection instead of ``cut_cross_entropy``.
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.loss.eaft_loss import _topk_entropy_adaptive_weights


def _chunk_eaft_loss(
    input_chunk: torch.Tensor,
    weight: torch.Tensor,
    target_chunk: torch.Tensor,
    *,
    ignore_index: int = -100,
    alpha: float = 1.0,
    topk: int = 20,
    fp32_upcast: bool = True,
) -> torch.Tensor:
    """Scalar EAFT loss for one token chunk (differentiable w.r.t. input and weight)."""
    logits = F.linear(input_chunk, weight)
    if fp32_upcast:
        logits = logits.float()
    per_token_loss = F.cross_entropy(
        logits,
        target_chunk.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    with torch.no_grad():
        adaptive_weight = _topk_entropy_adaptive_weights(
            logits.detach(),
            alpha=alpha,
            topk=topk,
        )
    return (per_token_loss * adaptive_weight).sum()


class FusedLinearEAFTFunction(torch.autograd.Function):
    """Chunked EAFT with precomputed per-chunk grads (see ``fused_kd_loss``)."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.LongTensor,
        ignore_index: int = -100,
        alpha: float = 1.0,
        topk: int = 20,
        fp32_upcast: bool = True,
        chunk_size: int = 1024,
        compiled: bool = True,
    ) -> torch.Tensor:
        grad_weight = torch.zeros_like(weight)
        grad_inputs: list[torch.Tensor] = []
        loss_acc = torch.zeros((), device=input.device, dtype=torch.float32)

        loss_fn = partial(
            _chunk_eaft_loss,
            ignore_index=ignore_index,
            alpha=alpha,
            topk=topk,
            fp32_upcast=fp32_upcast,
        )

        def accumulate_chunk(input_chunk: torch.Tensor, target_chunk: torch.Tensor) -> torch.Tensor:
            (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(
                loss_fn,
                argnums=(0, 1),
            )(input_chunk, weight, target_chunk)
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        num_chunks = max(1, math.ceil(input.shape[0] / chunk_size))
        for input_chunk, target_chunk in zip(
            torch.chunk(input, num_chunks, dim=0),
            torch.chunk(target, num_chunks, dim=0),
        ):
            grad_inputs.append(accumulate_chunk(input_chunk, target_chunk))

        ctx.save_for_backward(torch.cat(grad_inputs, dim=0), grad_weight)
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input, grad_weight = ctx.saved_tensors
        if not torch.equal(grad_output, torch.ones_like(grad_output)):
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None, None, None, None, None, None


class FusedLinearEAFTLoss(nn.Module):
    """EAFT loss on ``hidden_states @ lm_weight.T`` without full-sequence logits materialization.
    """

    def __init__(
        self,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
        alpha: float = 1.0,
        topk: int = 20,
        chunk_size: Optional[int] = None,
        compiled: bool = True,
    ):
        """
        Args:
            fp32_upcast (bool): Cast chunk logits to float32 before the loss. Default: True.
            ignore_index (int): Label to ignore. Defaults to -100.
            reduction (str): Must be ``"sum"`` (only supported value).
            alpha (float): Entropy gating exponent. Defaults to 1.0.
            topk (int): Top-*k* logits for entropy approximation. Defaults to 20.
            chunk_size (int, optional): Tokens per chunk. ``None`` → vocab-aware default.
            compiled (bool): ``torch.compile`` the per-chunk accumulator. Default: True.
        """
        super().__init__()
        if reduction != "sum":
            raise ValueError(f"FusedLinearEAFTLoss only supports reduction='sum', got {reduction!r}")
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha
        self.topk = topk
        self.chunk_size = chunk_size
        self.compiled = compiled

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_weight: torch.Tensor,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Hidden states before the LM head.
            labels: Target token ids.
            lm_weight: LM head weight ``[vocab_size, hidden_size]``.
            num_label_tokens: If set, divide the summed loss by this value.

        Returns:
            Scalar loss.
        """
        hidden_size = hidden_states.shape[-1]
        input_flat = hidden_states.reshape(-1, hidden_size)
        target_flat = labels.reshape(-1)

        vocab_size = lm_weight.shape[0]
        num_tokens = input_flat.shape[0]
        chunk_size = self.chunk_size or max(1, min(num_tokens, 8 * 1024 * 1024 // max(vocab_size, 1)))

        loss = FusedLinearEAFTFunction.apply(
            input_flat,
            lm_weight,
            target_flat,
            self.ignore_index,
            self.alpha,
            self.topk,
            self.fp32_upcast,
            chunk_size,
            self.compiled,
        )

        if num_label_tokens is not None:
            if num_label_tokens == 0:
                return loss * 0.0
            loss = loss / num_label_tokens
        return loss
