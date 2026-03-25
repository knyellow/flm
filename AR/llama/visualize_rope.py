import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from AR.llama.model import apply_rotary_emb, precompute_freqs_cis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RoPE frequencies and positional effect.")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--anchor_pos", type=int, default=255)
    parser.add_argument("--output", default="AR/llama/outputs/rope_frequencies.png")
    return parser.parse_args()


def build_constant_qk(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.linspace(-1.0, 1.0, steps=head_dim, dtype=torch.float32)
    base = base / base.norm()
    q = base.view(1, 1, 1, head_dim).repeat(1, seq_len, 1, 1)
    k = base.view(1, 1, 1, head_dim).repeat(1, seq_len, 1, 1)
    return q, k


def main() -> None:
    args = parse_args()
    head_dim = args.dim // args.n_heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    freqs_cis = precompute_freqs_cis(head_dim=head_dim, end=args.seq_len, theta=args.rope_theta)
    inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pair_ids = torch.arange(inv_freq.numel())
    positions = torch.arange(args.seq_len)

    phase = torch.outer(positions.float(), inv_freq)
    selected_pairs = torch.unique(torch.linspace(0, inv_freq.numel() - 1, steps=min(6, inv_freq.numel())).long())

    q_plain, k_plain = build_constant_qk(args.seq_len, head_dim)
    q_rope, k_rope = apply_rotary_emb(q_plain, k_plain, freqs_cis)
    anchor = max(0, min(args.anchor_pos, args.seq_len - 1))
    rope_scores = torch.matmul(q_rope[0, anchor, 0], k_rope[0, :, 0].T)
    plain_scores = torch.matmul(q_plain[0, anchor, 0], k_plain[0, :, 0].T)

    visible_start = max(0, anchor - args.window_size + 1)
    visible_mask = (positions >= visible_start) & (positions <= anchor)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(pair_ids.numpy(), inv_freq.numpy(), linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_title("RoPE inverse frequencies by rotary pair")
    axes[0].set_xlabel("Rotary pair index")
    axes[0].set_ylabel("Inverse frequency")
    axes[0].grid(alpha=0.3)

    for pair_idx in selected_pairs.tolist():
        axes[1].plot(positions.numpy(), phase[:, pair_idx].numpy(), label=f"pair {pair_idx}")
    axes[1].set_title("RoPE phase growth over positions")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Phase (radians)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(positions.numpy(), plain_scores.numpy(), label="No positional rotation", linewidth=2)
    axes[2].plot(positions.numpy(), rope_scores.numpy(), label="With RoPE", linewidth=2)
    axes[2].fill_between(
        positions.numpy(),
        rope_scores.min().item(),
        rope_scores.max().item(),
        where=visible_mask.numpy(),
        alpha=0.12,
        label=f"Visible window [{visible_start}, {anchor}]",
    )
    axes[2].set_title(f"Similarity from anchor position {anchor}")
    axes[2].set_xlabel("Key position")
    axes[2].set_ylabel("Q·K similarity")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
