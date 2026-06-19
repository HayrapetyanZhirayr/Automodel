# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Helpers for combining CE and KD losses for homogeneous KD packs."""

from __future__ import annotations

from typing import Any, Optional

import torch


def parse_batch_use_kd(use_kd: Any) -> Optional[bool]:
    """Return whether this microbatch should run KD, or ``None`` for legacy global KD.

    With source-homogeneous neat packing each pack carries a single ``use_kd`` flag.
    When ``local_batch_size > 1`` mixes KD and CE packs, teacher runs if any pack
    needs KD.
    """
    if use_kd is None:
        return None
    if isinstance(use_kd, torch.Tensor):
        if use_kd.numel() == 0:
            return False
        return bool(use_kd.bool().any().item())
    if isinstance(use_kd, (list, tuple)):
        return any(bool(v) for v in use_kd)
    return bool(use_kd)


def combine_kd_ce_loss(
    ce_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    kd_ratio: float,
    use_kd: Optional[bool] = None,
) -> torch.Tensor:
    """Combine scalar CE and KD losses for a (homogeneous) pack."""
    if use_kd is False:
        return ce_loss
    if use_kd is None:
        return (1.0 - kd_ratio) * ce_loss + kd_ratio * kd_loss
    return (1.0 - kd_ratio) * ce_loss + kd_ratio * kd_loss
