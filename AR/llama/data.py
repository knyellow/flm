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
    """Wraps the raw LLaMA tiktoken tokenizer with a reduced vocabulary.

    Two modes are supported:

    1. **Index-cutoff mode** (legacy, ``vocab_map_path=None``):
       Keeps tokens whose original LLaMA id < ``base_vocab_size`` and maps
       everything else to UNK.  This is fast but the top-K tokens by index are
       very short BPE sub-pieces, so common English words often fall outside
       the window.

    2. **Frequency-map mode** (recommended, ``vocab_map_path`` provided):
       Loads a JSON mapping produced by ``AR/llama/build_vocab.py`` that
       selects the *K most frequent* tokens in the training corpus and assigns
       them new compact ids 0 … K-1.  Tokens outside the map are mapped to
       UNK.  This dramatically lowers the UNK rate for TinyStories.
    """

    def __init__(
        self,
        model_path: str | Path,
        vocab_size: int = 10004,
        vocab_map_path: str | Path | None = None,
    ) -> None:
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4 to reserve BOS, EOS, and UNK tokens")
        self.model_path = str(model_path)
        self.tokenizer = LlamaTokenizer(self.model_path)
        self.vocab_size = vocab_size
        self.base_vocab_size = vocab_size - 3  # slots for real tokens

        # Special token ids (placed right after the real-token range)
        self.bos_token_id = self.base_vocab_size
        self.eos_token_id = self.base_vocab_size + 1
        self.unk_token_id = self.base_vocab_size + 2
        self.pad_token_id = self.eos_token_id

        # --- Frequency-map mode ---
        if vocab_map_path is not None:
            import json as _json
            vocab_map_path = Path(vocab_map_path)
            if not vocab_map_path.exists():
                raise FileNotFoundError(
                    f"vocab_map_path={vocab_map_path} not found. "
                    "Run AR/llama/build_vocab.py first to generate it."
                )
            with vocab_map_path.open("r", encoding="utf-8") as f:
                raw = _json.load(f)
            # JSON keys are strings → convert to int
            self._old_to_new: dict[int, int] = {int(k): v for k, v in raw.items()}
            # Reverse map for decode: new_id → original LLaMA id
            self._new_to_old: dict[int, int] = {v: k for k, v in self._old_to_new.items()}
            if len(self._old_to_new) > self.base_vocab_size:
                raise ValueError(
                    f"vocab_map has {len(self._old_to_new)} entries but base_vocab_size={self.base_vocab_size}"
                )
            self._use_freq_map = True
        else:
            # Legacy: only valid when base_vocab_size ≤ bos_id (index cutoff)
            if self.base_vocab_size > self.tokenizer.bos_id:
                raise ValueError(
                    f"requested vocab_size={vocab_size} exceeds supported reduced-vocab range for {model_path}"
                )
            self._old_to_new = {}
            self._new_to_old = {}
            self._use_freq_map = False

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        token_ids = self.tokenizer.encode(text, bos=False, eos=False)
        if self._use_freq_map:
            # Map via frequency-based remapping; out-of-vocab → UNK
            reduced_ids = [
                self._old_to_new.get(tid, self.unk_token_id)
                for tid in token_ids
            ]
        else:
            # Legacy: top-K by index
            reduced_ids = [
                tid if tid < self.base_vocab_size else self.unk_token_id
                for tid in token_ids
            ]
        if add_bos:
            reduced_ids.insert(0, self.bos_token_id)
        if add_eos:
            reduced_ids.append(self.eos_token_id)
        return reduced_ids

    def decode(self, token_ids: list[int]) -> str:
        if self._use_freq_map:
            # Convert new compact ids back to original LLaMA ids for decoding
            original_ids = [
                self._new_to_old[tid]
                for tid in token_ids
                if tid in self._new_to_old  # skip BOS/EOS/UNK
            ]
        else:
            # Legacy: discard anything ≥ base_vocab_size
            original_ids = [tid for tid in token_ids if 0 <= tid < self.base_vocab_size]
        return self.tokenizer.decode(original_ids)


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
