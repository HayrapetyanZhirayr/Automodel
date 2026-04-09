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

# This script can be used to consolidate sharded HF safetensors checkpoints
# to the consolidated format.

# Example model directory structure:
# model/
# ├── shard-00001-model-00001-of-00001.safetensors
# └── shard-00002-model-00001-of-00001.safetensors
#  ...

# This script works on both single and multiple workers:
# Example usage on 2 GPUs:
# torchrun --nproc-per-node=2 tools/offline_hf_consolidation.py --model-name meta-llama/Llama-3.2-1B --input-dir checkpoints/epoch_0_step_19/model/ --output-dir checkpoints/epoch_0_step_19/model/consolidated/
#
# Example usage on 1 GPU:
# python tools/offline_hf_consolidation.py --model-name meta-llama/Llama-3.2-1B --input-dir checkpoints/epoch_0_step_19/model/ --output-dir checkpoints/epoch_0_step_19/model/consolidated/

import argparse
import json
import os
import shutil
import re
import glob
import torch
import torch.distributed as dist

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from nemo_automodel.components.distributed.init_utils import (
    get_rank_safe,
    get_world_size_safe,
    initialize_distributed,
)

_metadata_fn: str = "model.safetensors.index.json"
SUFFIX = ".safetensors"
_HF_WEIGHT_NUM_FILES_RE = re.compile(r"-of-(\d+)\.safetensors$", re.IGNORECASE)


def copy_metadata_files(input_dir, output_dir):
    """
    Copy the metadata files over from the input directory to the output directory.
    """
    for item_name in os.listdir(input_dir):
        if item_name == "fqn_to_file_index_mapping.json":
            continue  # consolidation step may emit an updated mapping for output
        src_path = os.path.join(input_dir, item_name)
        dst_path = os.path.join(output_dir, item_name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def infer_hf_safetensors_num_shards(model_dir: str):
    """
    Infer the Hugging Face weight shard count ``N`` (the ``of-NNNNN`` part) from a **local**
    model directory.

    Uses ``model.safetensors.index.json`` when present (parses ``weight_map`` basenames), else
    globs ``model-*-of-*.safetensors``. Returns ``1`` if only ``model.safetensors`` exists.

    Hub repo ids (non-paths) are not resolved offline — returns ``None``.

    Args:
        model_dir: Absolute or user path to a HF snapshot directory on disk.

    Returns:
        Shard count ``N``, or ``None`` if the path is not a directory or layout is unknown.
    """
    if not model_dir or not isinstance(model_dir, str):
        return None
    model_dir = os.path.expanduser(model_dir)
    if not os.path.isdir(model_dir):
        return None

    index_path = os.path.join(model_dir, _metadata_fn)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index_obj = json.load(f)
        weight_map = index_obj.get("weight_map") or {}
        nums: set[int] = set()
        for filename in weight_map.values():
            base = os.path.basename(str(filename))
            m = _HF_WEIGHT_NUM_FILES_RE.search(base)
            if m:
                nums.add(int(m.group(1)))
        if nums:
            return max(nums)

    shard_files = glob.glob(os.path.join(model_dir, "model-*-of-*" + SUFFIX))
    nums_glob: set[int] = set()
    for path in shard_files:
        m = _HF_WEIGHT_NUM_FILES_RE.search(os.path.basename(path))
        if m:
            nums_glob.add(int(m.group(1)))
    if nums_glob:
        return max(nums_glob)

    if os.path.isfile(os.path.join(model_dir, "model" + SUFFIX)):
        return 1

    return None


def spread_fqns_to_hf_file_indices(
    fqn_to_index_mapping: dict[str, int],
    num_files: int,
) -> dict[str, int]:
    """
    Reassign each tensor FQN to a Hugging Face-style output file index in ``1 .. num_files``.

    Values in the saved JSON from training often encode ``model-00001-of-00001``
    per rank, so every value is ``1`` and consolidation produces a single huge file. This helper
    spreads keys (stable sort by FQN, round-robin) so ``_gen_file_name`` yields
    ``model-XXXXX-of-{num_files:05d}.safetensors`` with multiple shards.

    Args:
        fqn_to_index_mapping: Existing mapping (only keys are required; values are replaced).
        num_files: Number of output safetensors files (must be >= 1).

    Returns:
        New mapping from FQN to 1-based file index.
    """
    if num_files < 1:
        raise ValueError(f"num_files must be >= 1, got {num_files}")
    keys = sorted(fqn_to_index_mapping.keys())
    if num_files == 1:
        return {fqn: 1 for fqn in keys}
    return {fqn: (i % num_files) + 1 for i, fqn in enumerate(keys)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate sharded HF safetensors checkpoints into consolidated files, "
            "preserving original sharding layout where possible."
        )
    )

    parser.add_argument(
        "--model-name",
        "-m",
        required=True,
        help=(
            "Hugging Face repo id (e.g. meta-llama/Llama-3.2-1B) or absolute path to a HF snapshot directory. "
            "Used as reference to copy metadata and derive FQN->file index mapping."
        ),
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing sharded safetensors files to consolidate.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where consolidated safetensors and metadata will be written.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=5,
        help="Number of threads for writing consolidated data (default: 5).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend to initialize (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.device_count() > 0 else "gloo"
    initialize_distributed(backend, timeout_minutes=10)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError("Could not locate the input directory. Pass an absolute path to the input directory.")

    hf_metadata_dir = os.path.join(args.input_dir, ".hf_metadata")

    if not os.path.exists(hf_metadata_dir) or not os.path.isdir(hf_metadata_dir):
        raise FileNotFoundError("Expected to find the .hf_metadata directory in the input directory.")

    with open(os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json"), "r") as f:
        fqn_to_index_mapping = json.load(f)

    num_output_shards = infer_hf_safetensors_num_shards(args.model_name)
    if num_output_shards and num_output_shards > 1:
        fqn_to_index_mapping = spread_fqns_to_hf_file_indices(fqn_to_index_mapping, num_output_shards)

    consolidate_safetensors_files_on_every_rank(
        args.input_dir,
        args.output_dir,
        fqn_to_index_mapping,
        num_threads=args.num_threads,
    )

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        copy_metadata_files(hf_metadata_dir, args.output_dir)

    if get_world_size_safe() > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
