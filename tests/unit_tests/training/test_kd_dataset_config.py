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

import pytest
import torch

from nemo_automodel.components.datasets.llm import chat_dataset as chat_dataset_module
from nemo_automodel.components.datasets.llm.neat_packing import _build_packed_sample, neat_pack_dataset
from nemo_automodel.components.training.kd_dataset_config import (
    parse_dataset_sources,
    raw_row_use_kd,
    resolve_kd_dataset_mask,
)
from nemo_automodel.components.training.kd_loss_utils import combine_kd_ce_loss, parse_batch_use_kd
from datasets import Dataset


class _Cfg(dict):
    def get(self, key, default=None):
        if "." in key:
            parts = key.split(".")
            cur = self
            for part in parts:
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur
        return super().get(key, default)


class _RawDs:
    def __init__(self, rows, kd_dataset_mask):
        self.dataset = rows
        self.kd_dataset_mask = kd_dataset_mask


def test_parse_dataset_sources_named_map():
    sources = parse_dataset_sources({"kimi": "/data/kimi.jsonl", "big_qwen": "/data/qwen.jsonl"})
    assert sources.paths == ["/data/kimi.jsonl", "/data/qwen.jsonl"]
    assert sources.names == ["kimi", "big_qwen"]


def test_resolve_kd_dataset_mask_from_ce_only_datasets():
    cfg = _Cfg(
        {
            "dataset": {
                "path_or_dataset_id": {
                    "kimi": "/data/kimi.jsonl",
                    "big_qwen": "/data/qwen.jsonl",
                },
            },
            "kd_ce_only_datasets": ["big_qwen"],
        }
    )
    assert resolve_kd_dataset_mask(cfg) == [True, False]


def test_raw_row_use_kd():
    rows = [{"_source_dataset_id": 0}, {"_source_dataset_id": 1}]
    ds = _RawDs(rows, [True, False])
    assert raw_row_use_kd(ds, 0) is True
    assert raw_row_use_kd(ds, 1) is False


def test_combine_kd_ce_loss_ce_only_pack():
    ce = torch.tensor(2.0)
    kd = torch.tensor(4.0)
    assert combine_kd_ce_loss(ce, kd, kd_ratio=0.5, use_kd=False).item() == pytest.approx(2.0)


def test_parse_batch_use_kd_scalar_and_tensor():
    assert parse_batch_use_kd(False) is False
    assert parse_batch_use_kd(torch.tensor([False])) is False
    assert parse_batch_use_kd(torch.tensor([False, True])) is True


def test_homogeneous_neat_packing_splits_kd_and_ce_bins():
    ds = Dataset.from_list(
        [
            {"input_ids": [1, 2], "labels": [1, 2], "use_kd": True},
            {"input_ids": [3, 4], "labels": [3, 4], "use_kd": False},
        ]
    )
    packed = neat_pack_dataset(ds, split="train", pack_size=4, padding_idx=0)
    use_kd_flags = sorted(bool(x) for x in packed["use_kd"])
    assert use_kd_flags == [False, True]


def test_build_packed_sample_sets_use_kd_flag():
    packed = _build_packed_sample(
        [{"input_ids": [1, 2], "labels": [1, 2]}, {"input_ids": [3], "labels": [3]}],
        pack_size=4,
        padding_idx=0,
        use_kd=True,
    )
    assert packed["use_kd"] is True


def test_load_openai_messages_tags_source_dataset_id_and_name(tmp_path):
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    file_a.write_text('{"messages": [{"role": "user", "content": "a"}]}\n', encoding="utf-8")
    file_b.write_text('{"messages": [{"role": "user", "content": "b"}]}\n', encoding="utf-8")

    rows = chat_dataset_module._load_openai_messages(
        {"kimi": str(file_a), "big_qwen": str(file_b)},
        split="train",
    )
    assert rows[0]["_source_dataset_name"] == "kimi"
    assert rows[1]["_source_dataset_name"] == "big_qwen"
