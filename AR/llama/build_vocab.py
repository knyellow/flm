"""
Build a frequency-based vocabulary mapping for the LLaMA tokenizer.

Instead of keeping the top-K tokens by index (which are short BPE substrings),
this script finds the top-K tokens by *actual frequency* in the TinyStories data.
The resulting mapping (original LLaMA id → new compact id) is saved as a JSON file
and loaded by LlamaTokenizerWrapper at training/inference time.

Usage
-----
# Build vocab from training data (run from the repo root, inside the container)
python3 -m AR.llama.build_vocab \
    --train_path  processed/tinystories/train.jsonl \
    --tokenizer_path AR/tokenizer.model \
    --vocab_size 10004 \
    --output     AR/llama/vocab_map.json

Notes
-----
- vocab_size should match the value used in train.py / sample.py.
- 3 slots are reserved for BOS, EOS, UNK  →  base_vocab = vocab_size - 3.
- The mapping JSON has the format {"old_id": new_id, ...}.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from AR.llama.tokenizer import Tokenizer as LlamaTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="frequency-based llama tokenizer")
    parser.add_argument("--train_path", default="processed/tinystories/train.jsonl",
                        help="Path to training JSONL file.")
    parser.add_argument("--tokenizer_path", default="AR/tokenizer.model",
                        help="Path to LLaMA tokenizer.model file.")
    parser.add_argument("--vocab_size", type=int, default=10004,
                        help="Total vocab size including BOS/EOS/UNK (3 special tokens).")
    parser.add_argument("--text_key", default="text",
                        help="Key in JSONL containing the story text.")
    parser.add_argument("--output", default="AR/llama/vocab_map.json",
                        help="Output path for the vocab mapping JSON.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of examples processed (for quick tests).")
    return parser.parse_args()


def build_freq_vocab(
    train_path: str,
    tokenizer_path: str,
    vocab_size: int,
    text_key: str = "text",
    max_examples: int | None = None,
) -> dict[int, int]:
    """
    Returns a dict mapping original LLaMA token id → new compact id (0-based).
    Special tokens are placed at the end:
        new_id  base_vocab   → BOS
        new_id  base_vocab+1 → EOS
        new_id  base_vocab+2 → UNK
    where base_vocab = vocab_size - 3.
    """
    base_vocab = vocab_size - 3  # number of "real" tokens

    tokenizer = LlamaTokenizer(tokenizer_path)
    freq: Counter = Counter()

    train_path_obj = Path(train_path)
    if not train_path_obj.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    print(f"Counting token frequencies in {train_path} ...")
    with train_path_obj.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get(text_key, "")
            if not text:
                continue
            # Encode without special tokens so we only count real content tokens
            token_ids = tokenizer.encode(text, bos=False, eos=False)
            freq.update(token_ids)
            if (i + 1) % 50000 == 0:
                print(f"  processed {i + 1:,} examples, unique tokens so far: {len(freq):,}")

    print(f"Total unique tokens seen: {len(freq):,}")
    print(f"Total token occurrences:  {sum(freq.values()):,}")

    # Select top-base_vocab tokens by frequency
    top_tokens = [tok for tok, _ in freq.most_common(base_vocab)]
    top_tokens.sort()  # sort by original id for determinism

    # Build old_id → new_id mapping
    id_map: dict[int, int] = {old: new for new, old in enumerate(top_tokens)}

    # Coverage stats
    total = sum(freq.values())
    covered = sum(freq[t] for t in top_tokens)
    print(f"\nVocab coverage: {covered / total * 100:.2f}% of all token occurrences")
    print(f"UNK rate (approx): {(total - covered) / total * 100:.2f}%")

    return id_map


def main() -> None:
    args = parse_args()
    id_map = build_freq_vocab(
        train_path=args.train_path,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        text_key=args.text_key,
        max_examples=args.max_examples,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON keys must be strings
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)

    print(f"\nVocab map saved to: {output_path}")
    print(f"  {len(id_map):,} tokens mapped  (vocab_size={args.vocab_size})")


if __name__ == "__main__":
    main()
