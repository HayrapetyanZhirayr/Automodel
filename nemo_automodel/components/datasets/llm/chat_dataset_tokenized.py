from __future__ import annotations

import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from filelock import FileLock
from datasets import VerificationMode, load_dataset
from torch.utils.data import Dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _has_chat_template,
    _resolve_chat_template,
    format_chat_template_from_tokenized
)

def _load_tokenized_dataset(path_or_dataset_id):
    output = []
    keys = (
        "input_ids",
        "attention_mask",
        "assistant_masks",
        "labels",
        "___PAD_TOKEN_IDS___"
    )
    with open(path_or_dataset_id, "r") as f:
        for line in f:
            d = json.loads(line)
            output.append(
                {k:d[k] for k in keys if k in d}
            )
    return output

class ChatDatasetFromTokenized(Dataset):
    """Dataset for OpenAI-format tool-calling chat transcripts.

    This class expects each row to contain a `messages` list in OpenAI chat format,
    potentially including tool calls and tool responses. The datasetformats the
    conversation via the tokenizer's chat template to produce `input_ids`, `labels`,
    and `attention_mask` suitable for SFT.
    """

    def __init__(
        self,
        path_or_dataset_id: Union[str, Sequence[str]],
        tokenizer,
        *,
        split: Optional[str] = None,
        name: Optional[str] = None,
        seq_length: Optional[int] = None,
        padding: Union[str, bool] = "do_not_pad",
        truncation: Union[str, bool] = "do_not_truncate",
        start_of_turn_token: Optional[str] = None,
        chat_template: Optional[str] = None,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        # Enforce chat-template availability for tool-calling data
        if chat_template is not None:
            tokenizer.chat_template = _resolve_chat_template(chat_template)

        if not _has_chat_template(tokenizer):
            raise ValueError("ChatDataset requires a tokenizer with chat template support.")

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.padding = padding
        self.truncation = truncation
        self.start_of_turn_token = start_of_turn_token

        self.dataset = _load_tokenized_dataset(path_or_dataset_id)

        # Ensure pad token presence for downstream padding
        eos_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        self.pad_token_id = _add_pad_token(self.tokenizer) or eos_token_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        tokenized_chat = self.dataset[idx]
        if "labels" in tokenized_chat:
            return tokenized_chat

        if "assistant_masks" in tokenized_chat:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", 0)
            sample = format_chat_template_from_tokenized(
                self.tokenizer,
                tokenized_chat,
                eos_token_id,
                self.pad_token_id,
                seq_length=self.seq_length,
                padding=self.padding,
                truncation=self.truncation,
            )
            return sample
        raise ValueError(
            f"Unsupported sample format at idx={idx}. "
            f"Expected either 'labels' or 'assistant_masks'. "
            f"Got keys={sorted(tokenized_chat.keys())}"
        )
