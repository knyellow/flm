import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer as LlamaTokenizer


@dataclass(frozen=True)
class ByteTokenizer:
    eos_token_id: int = 256
    pad_token_id: int = 257
    bos_token_id: int = 258
    vocab_size: int = 259

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        token_ids = list(text.encode("utf-8", errors="ignore"))
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        buf = bytearray()
        for token_id in token_ids:
            if token_id in {self.eos_token_id, self.pad_token_id, self.bos_token_id}:
                continue
            if 0 <= token_id < 256:
                buf.append(token_id)
        return buf.decode("utf-8", errors="ignore")


class LlamaTokenizerWrapper:
    def __init__(self, model_path: str | Path, vocab_size: int = 10004) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4 to reserve base, bos, eos, and unk tokens")
        self.model_path = str(model_path)
        self.tokenizer = LlamaTokenizer(self.model_path)
        self.vocab_size = vocab_size
        self.base_vocab_size = vocab_size - 3
        if self.base_vocab_size > self.tokenizer.bos_id:
            raise ValueError(
                f"requested vocab_size={vocab_size} exceeds supported reduced-vocab range for {model_path}"
            )

        self.bos_token_id = self.base_vocab_size
        self.eos_token_id = self.base_vocab_size + 1
        self.unk_token_id = self.base_vocab_size + 2
        self.pad_token_id = self.eos_token_id

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        token_ids = self.tokenizer.encode(text, bos=False, eos=False)
        reduced_ids = [token_id if token_id < self.base_vocab_size else self.unk_token_id for token_id in token_ids]
        if add_bos:
            reduced_ids.insert(0, self.bos_token_id)
        if add_eos:
            reduced_ids.append(self.eos_token_id)
        return reduced_ids

    def decode(self, token_ids: list[int]) -> str:
        cleaned = [token_id for token_id in token_ids if 0 <= token_id < self.base_vocab_size]
        return self.tokenizer.decode(cleaned)


class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: ByteTokenizer,
        seq_len: int,
        split: str,
        text_key: str = "text",
        random_crop: bool = True,
        max_examples: Optional[int] = None,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.text_key = text_key
        self.random_crop = random_crop and split == "train"
        self.seed = seed

        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} does not exist")

        self.offsets_path = self.data_path.with_suffix(self.data_path.suffix + ".offsets.npy")
        self.offsets = self._load_or_build_offsets()
        if max_examples is not None:
            self.offsets = self.offsets[: max_examples]

    def _load_or_build_offsets(self) -> np.ndarray:
        if self.offsets_path.exists():
            return np.load(self.offsets_path)

        offsets = []
        with self.data_path.open("rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                offsets.append(offset)
                offset = f.tell()
                line = f.readline()
        offsets_arr = np.asarray(offsets, dtype=np.int64)
        np.save(self.offsets_path, offsets_arr)
        return offsets_arr

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_record(self, idx: int) -> dict:
        with self.data_path.open("rb") as f:
            f.seek(int(self.offsets[idx]))
            line = f.readline()
        return json.loads(line.decode("utf-8"))

    def _fit_to_context(self, token_ids: list[int], idx: int) -> list[int]:
        target_len = self.seq_len + 1
        if len(token_ids) <= target_len:
            return token_ids + [self.tokenizer.pad_token_id] * (target_len - len(token_ids))
        if not self.random_crop:
            return token_ids[:target_len]

        rng = random.Random(self.seed + idx)
        start = rng.randint(0, len(token_ids) - target_len)
        return token_ids[start : start + target_len]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self._read_record(idx)
        token_ids = self.tokenizer.encode(record[self.text_key], add_bos=True, add_eos=True)
        token_ids = self._fit_to_context(token_ids, idx)

        tokens = torch.tensor(token_ids, dtype=torch.long)
        input_ids = tokens[:-1]
        labels = tokens[1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_text": record.get("prompt", ""),
            "reference_text": record.get("continuation", ""),
        }
