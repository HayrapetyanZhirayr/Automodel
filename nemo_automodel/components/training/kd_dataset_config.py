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

"""Resolve per-dataset knowledge-distillation settings from recipe YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSources:
    """Ordered train dataset sources parsed from ``path_or_dataset_id``."""

    paths: list[str]
    names: list[Optional[str]]

    @property
    def name_to_id(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.names) if name is not None}

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def has_named_sources(self) -> bool:
        return any(name is not None for name in self.names)


def _coerce_alias_map(raw: Any) -> Optional[Mapping[str, str]]:
    """Return an alias->path mapping when *raw* uses named dataset sources."""
    if isinstance(raw, Mapping):
        if raw and all(isinstance(v, str) for v in raw.values()):
            return {str(k): str(v) for k, v in raw.items()}
        return None

    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        as_dict = to_dict()
        if isinstance(as_dict, dict) and as_dict and all(isinstance(v, str) for v in as_dict.values()):
            return {str(k): str(v) for k, v in as_dict.items()}

    if hasattr(raw, "__dict__"):
        reserved = {"raise_on_missing_attr", "_raw_config", "_original_strings", "_target_"}
        as_dict = {k: v for k, v in raw.__dict__.items() if k not in reserved and not str(k).startswith("_")}
        if as_dict and all(isinstance(v, str) for v in as_dict.values()):
            return {str(k): str(v) for k, v in as_dict.items()}
    return None


def parse_dataset_sources(path_or_dataset_id: Any) -> DatasetSources:
    """Parse ``path_or_dataset_id`` into ordered paths and optional aliases.

    Supported forms:

    * ``"/path/to/data.jsonl"`` — single unnamed source.
    * ``["/a.jsonl", "/b.jsonl"]`` — multiple unnamed sources (legacy list).
    * ``{"kimi": "/a.jsonl", "big_qwen": "/b.jsonl"}`` — named aliases (preferred
      for multi-source KD configs).
    """
    if path_or_dataset_id is None:
        return DatasetSources(paths=[], names=[])

    alias_map = _coerce_alias_map(path_or_dataset_id)
    if alias_map is not None:
        names = [str(name) for name in alias_map.keys()]
        paths = [alias_map[name] for name in names]
        return DatasetSources(paths=paths, names=names)

    if isinstance(path_or_dataset_id, str):
        return DatasetSources(paths=[path_or_dataset_id], names=[None])

    paths: list[str] = []
    for entry in path_or_dataset_id:
        if not isinstance(entry, str):
            raise ValueError("path_or_dataset_id list entries must be strings")
        paths.append(entry)
    return DatasetSources(paths=paths, names=[None] * len(paths))


def _normalize_path(path: str) -> str:
    return str(Path(path).resolve())


def _read_cfg_field(node: Any, key: str, default: Any = None) -> Any:
    """Read a field from a ConfigNode or plain mapping."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, None)
        if value is not None:
            return value
    return getattr(node, key, default)


def _resolve_source_indices(
    sources: DatasetSources,
    requested: Sequence[Union[str, int]],
    *,
    field_name: str,
) -> list[int]:
    """Map user-provided aliases, paths, or legacy indices to source indices."""
    resolved: list[int] = []
    normalized_paths = {_normalize_path(path): idx for idx, path in enumerate(sources.paths)}
    name_to_id = sources.name_to_id

    for item in requested:
        if isinstance(item, int):
            if item < 0 or item >= len(sources):
                raise ValueError(f"{field_name}: dataset index {item} is out of range for {len(sources)} source(s)")
            resolved.append(item)
            continue

        token = str(item)
        if token in name_to_id:
            resolved.append(name_to_id[token])
            continue

        if token in sources.paths:
            resolved.append(sources.paths.index(token))
            continue

        norm = _normalize_path(token)
        if norm in normalized_paths:
            resolved.append(normalized_paths[norm])
            continue

        if sources.has_named_sources:
            known = ", ".join(sorted(name_to_id))
            raise ValueError(f"{field_name}: unknown dataset alias {token!r}; known aliases: {known}")
        raise ValueError(f"{field_name}: dataset path {token!r} not found in dataset.path_or_dataset_id")

    return resolved


def _mask_from_named_mapping(sources: DatasetSources, mask_by_name: Mapping[str, Any]) -> list[bool]:
    if not sources.has_named_sources:
        raise ValueError(
            "kd_dataset_mask as a mapping requires named dataset aliases, e.g.\n"
            "  dataset:\n"
            "    path_or_dataset_id:\n"
            "      kimi: /path/to/kimi.jsonl\n"
            "      big_qwen: /path/to/big_qwen.jsonl\n"
            "  kd_dataset_mask:\n"
            "    kimi: true\n"
            "    big_qwen: false"
        )
    unknown = sorted(set(str(k) for k in mask_by_name) - set(sources.name_to_id))
    if unknown:
        known = ", ".join(sorted(sources.name_to_id))
        raise ValueError(f"kd_dataset_mask references unknown aliases {unknown}; known aliases: {known}")
    return [bool(mask_by_name.get(name, False)) if name is not None else False for name in sources.names]


def _get_dataset_sources(cfg: Any) -> DatasetSources:
    dataset_cfg = cfg.get("dataset", None)
    raw = _read_cfg_field(dataset_cfg, "path_or_dataset_id")
    if raw is None:
        raw = _read_cfg_field(dataset_cfg, "path_or_dataset")
    return parse_dataset_sources(raw)


def resolve_kd_dataset_mask(cfg: Any) -> Optional[list[bool]]:
    """Return per-source KD flags aligned with ``dataset.path_or_dataset_id`` order.

    When ``None`` is returned, every dataset uses the global ``kd_ratio`` (backward
    compatible default). When a list is returned, ``True`` means KL distillation is
    enabled for that source and ``False`` means pure cross-entropy.

    Preferred config (named aliases):

    .. code-block:: yaml

        dataset:
          path_or_dataset_id:
            kimi: /path/to/kimi.jsonl
            big_qwen: /path/to/big_qwen.jsonl
        kd_ce_only_datasets: [big_qwen]

    Also supported:

    * ``kd_datasets`` — aliases that **use** KD; others are CE-only.
    * ``kd_ce_only_datasets`` — aliases that are CE-only; others use KD.
    * ``kd_dataset_mask`` — explicit mapping ``{alias: bool}`` or list ``[bool, ...]``.

    Legacy index/path keys remain supported for unnamed list sources:

    * ``kd_dataset_indices`` / ``kd_ce_only_dataset_indices``
    * ``kd_dataset_paths`` / ``kd_ce_only_dataset_paths``

    The same keys may also be placed under the ``dataset`` section.
    """
    sources = _get_dataset_sources(cfg)
    if len(sources) <= 1:
        return None

    dataset_cfg = cfg.get("dataset", None)

    def _cfg_get(key: str):
        if dataset_cfg is not None and hasattr(dataset_cfg, "get") and dataset_cfg.get(key, None) is not None:
            return dataset_cfg.get(key)
        return cfg.get(key, None)

    explicit_mask = _cfg_get("kd_dataset_mask")
    if explicit_mask is not None:
        if isinstance(explicit_mask, Mapping):
            mask = _mask_from_named_mapping(sources, explicit_mask)
        else:
            mask = [bool(v) for v in explicit_mask]
            if len(mask) != len(sources):
                raise ValueError(
                    f"kd_dataset_mask length ({len(mask)}) must match number of dataset sources ({len(sources)})"
                )
        logger.info("Per-dataset KD mask (explicit): %s", list(zip(sources.names, mask)))
        return mask

    kd_names = _cfg_get("kd_datasets")
    kd_indices = _cfg_get("kd_dataset_indices")
    kd_paths = _cfg_get("kd_dataset_paths")
    ce_only_names = _cfg_get("kd_ce_only_datasets")
    ce_only_indices = _cfg_get("kd_ce_only_dataset_indices")
    ce_only_paths = _cfg_get("kd_ce_only_dataset_paths")

    named_kd_keys = sum(x is not None for x in (kd_names, kd_indices, kd_paths))
    named_ce_keys = sum(x is not None for x in (ce_only_names, ce_only_indices, ce_only_paths))
    if named_kd_keys > 1:
        raise ValueError("Specify only one of kd_datasets, kd_dataset_indices, and kd_dataset_paths")
    if named_ce_keys > 1:
        raise ValueError("Specify only one of kd_ce_only_datasets, kd_ce_only_dataset_indices, and kd_ce_only_paths")
    if named_kd_keys and named_ce_keys:
        raise ValueError("Specify either KD-enabled datasets or CE-only datasets, not both")

    if kd_names is not None:
        enabled = set(_resolve_source_indices(sources, kd_names, field_name="kd_datasets"))
        mask = [idx in enabled for idx in range(len(sources))]
        logger.info("Per-dataset KD mask (kd_datasets): %s", list(zip(sources.names, mask)))
        return mask

    if kd_indices is not None or kd_paths is not None:
        enabled = set(_resolve_source_indices(sources, kd_indices or kd_paths or [], field_name="kd_dataset_indices"))
        mask = [idx in enabled for idx in range(len(sources))]
        logger.info("Per-dataset KD mask (kd_dataset_indices/paths): %s", list(zip(sources.names, mask)))
        return mask

    if ce_only_names is not None:
        disabled = set(_resolve_source_indices(sources, ce_only_names, field_name="kd_ce_only_datasets"))
        mask = [idx not in disabled for idx in range(len(sources))]
        logger.info("Per-dataset KD mask (kd_ce_only_datasets): %s", list(zip(sources.names, mask)))
        return mask

    if ce_only_indices is not None or ce_only_paths is not None:
        disabled = set(
            _resolve_source_indices(
                sources, ce_only_indices or ce_only_paths or [], field_name="kd_ce_only_dataset_indices"
            )
        )
        mask = [idx not in disabled for idx in range(len(sources))]
        logger.info("Per-dataset KD mask (kd_ce_only_dataset_indices/paths): %s", list(zip(sources.names, mask)))
        return mask

    return None


def apply_kd_dataset_mask_to_cfg(cfg: Any) -> Optional[list[bool]]:
    """Resolve and inject ``kd_dataset_mask`` into the dataset config when configured."""
    mask = resolve_kd_dataset_mask(cfg)
    if mask is None:
        return None
    dataset_cfg = cfg.get("dataset", None)
    if dataset_cfg is None:
        raise ValueError("Per-dataset KD settings require a dataset section in the config")
    if isinstance(dataset_cfg, dict):
        dataset_cfg["kd_dataset_mask"] = mask
    else:
        setattr(dataset_cfg, "kd_dataset_mask", mask)
    return mask


def expand_use_kd_to_token_mask(labels: Sequence[int], use_kd: bool, *, ignore_index: int = -100) -> list[float]:
    """Expand a per-sample KD flag into a per-token mask aligned with ``labels``."""
    kd_value = 1.0 if use_kd else 0.0
    return [kd_value if lab != ignore_index else 0.0 for lab in labels]


def raw_row_use_kd(ds_raw: Any, idx: int) -> bool:
    """Return whether sample *idx* in a raw ChatDataset should use KD."""
    kd_dataset_mask = getattr(ds_raw, "kd_dataset_mask", None)
    if kd_dataset_mask is None:
        return True
    dataset = getattr(ds_raw, "dataset", None)
    if dataset is None:
        return True
    row = dataset[idx]
    source_id = row.get("_source_dataset_id", 0)
    if source_id < 0 or source_id >= len(kd_dataset_mask):
        raise ValueError(
            f"Sample source dataset id {source_id} is out of range for kd_dataset_mask "
            f"(len={len(kd_dataset_mask)})"
        )
    return bool(kd_dataset_mask[source_id])
