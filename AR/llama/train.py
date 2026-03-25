import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from AR.llama.data import ByteTokenizer, LlamaTokenizerWrapper, TinyStoriesDataset
from AR.llama.model import LlamaConfig, LlamaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Llama-style model on TinyStories.")
    parser.add_argument("--train_path", default="processed/tinystories/train.jsonl")
    parser.add_argument("--val_path", default="processed/tinystories/validation.jsonl")
    parser.add_argument("--output_dir", default="AR/llama/outputs/tinystories-small")
    parser.add_argument("--tokenizer", choices=["llama", "byte"], default="llama")
    parser.add_argument("--tokenizer_path", default="AR/tokenizer.model")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--sample_max_new_tokens", type=int, default=160)
    parser.add_argument("--sample_temperature", type=float, default=0)
    parser.add_argument("--sample_top_k", type=int, default=50)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vocab_size", type=int, default=10004)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=6144)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def build_dataloader(
    path: str,
    tokenizer,
    seq_len: int,
    split: str,
    batch_size: int,
    num_workers: int,
    max_examples: int | None,
    seed: int,
) -> DataLoader:
    dataset = TinyStoriesDataset(
        data_path=path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        split=split,
        max_examples=max_examples,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def cosine_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    sample_count: int,
    sample_max_new_tokens: int,
    sample_temperature: float,
    sample_top_k: int,
    max_batches: int = 100,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    text_samples = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        loss = model(input_ids, labels=labels)["loss"]
        total_loss += loss.item()
        total_batches += 1

        if len(text_samples) < sample_count:
            prompts = batch["prompt_text"]
            references = batch["reference_text"]
            for prompt, reference in zip(prompts, references):
                if len(text_samples) >= sample_count:
                    break
                if not prompt:
                    continue
                prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                prompt_tensor = torch.tensor([prompt_ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)
                pred_ids = model.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=sample_max_new_tokens,
                    min_new_tokens=min(64, sample_max_new_tokens),
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=sample_temperature,
                    top_k=sample_top_k,
                )[0].tolist()
                prediction = tokenizer.decode(pred_ids[len(prompt_tensor[0]) :])
                text_samples.append(
                    {
                        "prompt": prompt,
                        "prediction": prediction,
                        "reference": reference,
                    }
                )

    mean_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(min(mean_loss, 20.0))
    if text_samples:
        print("\n[validation] text samples")
        for sample in text_samples:
            print("-" * 80)
            print(f"Prompt: {sample['prompt']}")
            print(f"Prediction: {sample['prediction']}")
            print(f"Reference: {sample['reference']}")
    model.train()
    return mean_loss, perplexity


def save_checkpoint(
    output_dir: Path,
    model: LlamaForCausalLM,
    optimizer: AdamW,
    step: int,
    config: LlamaConfig,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": vars(config),
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / f"checkpoint_step_{step}.pt")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    tokenizer = (
        LlamaTokenizerWrapper(args.tokenizer_path, vocab_size=args.vocab_size)
        if args.tokenizer == "llama"
        else ByteTokenizer()
    )
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.seq_len,
        window_size=args.window_size,
        dropout=args.dropout,
    )

    model = LlamaForCausalLM(config).to(device)
    print("args.hidden_dim =", args.hidden_dim)
    print("config.hidden_dim =", config.hidden_dim)

    print(f"model_params={model.num_parameters():,}")
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    
    train_loader = build_dataloader(
        path=args.train_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_examples=args.max_train_examples,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        path=args.val_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        split="validation",
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_examples=args.max_val_examples,
        seed=args.seed,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "model_config": vars(config)}, f, indent=2)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    start_time = time.time()

    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = cosine_lr(step, args.learning_rate, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=device.type == "cuda"):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / args.log_every
            ppl = math.exp(min(avg_loss, 20.0))
            tokens_per_sec = (args.batch_size * args.seq_len * args.log_every) / max(elapsed, 1e-6)
            print(
                f"step={step + 1} lr={lr:.6f} train_loss={avg_loss:.4f} "
                f"train_ppl={ppl:.2f} tok/s={tokens_per_sec:.0f}"
            )
            running_loss = 0.0
            start_time = time.time()

        if (step + 1) % args.eval_every == 0:
            val_loss, val_ppl = evaluate(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                sample_count=args.sample_count,
                sample_max_new_tokens=args.sample_max_new_tokens,
                sample_temperature=args.sample_temperature,
                sample_top_k=args.sample_top_k,
            )
            print(f"step={step + 1} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if (step + 1) % args.save_every == 0:
            save_checkpoint(output_dir, model, optimizer, step + 1, config, args)

    save_checkpoint(output_dir, model, optimizer, args.max_steps, config, args)


if __name__ == "__main__":
    main()
