import argparse
from pathlib import Path

import torch

from AR.llama.data import ByteTokenizer, LlamaTokenizerWrapper
from AR.llama.model import LlamaConfig, LlamaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained TinyStories Llama model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", choices=["llama", "byte"], default="llama")
    parser.add_argument("--tokenizer_path", default="AR/tokenizer.model")
    parser.add_argument("--vocab_size", type=int, default=10004)
    parser.add_argument("--vocab_map", default=None,
                        help="Path to vocab_map.json from build_vocab.py (frequency-based mode).")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = (
        LlamaTokenizerWrapper(
            args.tokenizer_path,
            vocab_size=args.vocab_size,
            vocab_map_path=args.vocab_map,
        )
        if args.tokenizer == "llama"
        else ByteTokenizer()
    )

    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    config = LlamaConfig(**checkpoint["config"])
    model = LlamaForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
