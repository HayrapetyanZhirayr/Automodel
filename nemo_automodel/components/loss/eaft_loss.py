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

"""Entropy-Adaptive Fine-Tuning (EAFT) loss.

Reference: Diao et al., "Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts
to Mitigate Forgetting", arXiv:2601.02151 (https://arxiv.org/abs/2601.02151).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def _compute_eaft_sum_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    alpha: float,
    topk: int,
    fp32_upcast: bool,
) -> torch.Tensor:
    """Sum of entropy-weighted per-token CE for aligned ``[N, V]`` / ``[N]`` tensors."""
    if fp32_upcast:
        logits = logits.float()
    per_token_loss = F.cross_entropy(
        logits,
        labels,
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


def _topk_entropy_adaptive_weights(
    logits: torch.Tensor,
    *,
    alpha: float,
    topk: int,
) -> torch.Tensor:
    """Per-token EAFT weights for flattened logits ``[num_tokens, vocab_size]``."""
    k = min(topk, logits.shape[-1])
    topk_val, _ = torch.topk(logits, k=k, dim=-1)
    logsumexp_topk = torch.logsumexp(topk_val, dim=-1, keepdim=True)
    log_probs_topk = topk_val - logsumexp_topk
    probs_topk = torch.exp(log_probs_topk)
    entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
    entropy_term = entropy_approx / math.log(k)
    return torch.pow(entropy_term, alpha)


class EAFTLoss(nn.Module):
    """Entropy-adaptive cross-entropy loss for supervised fine-tuning.

    Same layout and masking contract as :class:`MaskedCrossEntropy`, but each token's
    CE is scaled by normalized top-*k* entropy: ``(H_tilde_t) ** alpha`` with
    ``H_tilde_t = H_approx / ln(k)``.

    See https://arxiv.org/pdf/2601.02151
    """

    def __init__(
        self,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
        alpha: float = 1.0,
        topk: int = 20,
    ):
        """
        Args:
            fp32_upcast (bool): if True it will cast logits to float32 before computing
                cross entropy. Default: True.
            ignore_index (int): label to ignore in CE calculation. Defaults to -100.
            reduction (str): Must be ``"sum"`` (only supported value).
            alpha (float): exponent on normalized entropy weights. Defaults to 1.0.
            topk (int): number of logits for entropy approximation. Defaults to 20.
        """
        super().__init__()
        if reduction != "sum":
            raise ValueError(f"EAFTLoss only supports reduction='sum', got {reduction!r}")
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha
        self.topk = topk

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the entropy-adaptive cross-entropy loss between logits and targets.

        Args:
            logits (torch.Tensor): The predicted logits with shape
                [batch_size, seq_len, vocab_size].
            labels (torch.Tensor): The ground truth class indices with shape
                [batch_size, seq_len].
            mask (torch.Tensor, optional): A tensor that masks the loss computation.
                Items marked with 1 will be used to calculate loss, otherwise ignored.
                Must be broadcastable to the shape of the loss. Defaults to None.
            num_label_tokens (int, optional): If provided, the summed loss is divided
                by this value.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # this may happen with CPUOffloadPolicy
        if labels.device != logits.device:
            labels = labels.to(logits.device)  # pragma: no cover
        # reshape to (N, C) and (N,) respectively
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)  # pragma: no cover
                labels.masked_fill_(mask.view(-1) == 0, self.ignore_index)
                del mask
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        loss = _compute_eaft_sum_from_logits(
            logits,
            labels,
            ignore_index=self.ignore_index,
            alpha=self.alpha,
            topk=self.topk,
            fp32_upcast=self.fp32_upcast,
        )
        if num_label_tokens is not None:
            if num_label_tokens == 0:
                return loss * 0.0
            loss = loss / num_label_tokens
        return loss
