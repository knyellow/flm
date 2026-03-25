import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LlamaConfig:
    vocab_size: int = 10004
    dim: int = 1024
    n_layers: int = 2
    n_heads: int = 16
    n_kv_heads: int = 16
    hidden_dim: int = 6144
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 512
    window_size: int = 256
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def precompute_freqs_cis(head_dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(end, dtype=torch.float32)
    phase = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(phase), phase)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return freqs_cis.view(1, x.shape[1], 1, x.shape[-1])


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_out = torch.view_as_real(xq_complex * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq, n_kv_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(batch, seq, n_kv_heads, n_rep, head_dim).reshape(
        batch, seq, n_kv_heads * n_rep, head_dim
    )


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.window_size = config.window_size

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xq = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seq_len, device=x.device)
        causal_mask = positions[None, :] > positions[:, None]
        if self.window_size > 0:
            causal_mask |= (positions[:, None] - positions[None, :]) >= self.window_size
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn = self.dropout(attn)
        output = torch.matmul(attn, xv)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        hidden_dim = int(2 * config.hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        self.output.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        for layer in self.layers:
            h = layer(h, freqs_cis)
        logits = self.output(self.norm(h)).float()

        output = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        generated = input_ids
        for step in range(max_new_tokens):
            idx = generated[:, -self.config.max_seq_len :]
            logits = self(idx)["logits"][:, -1, :]
            if eos_token_id is not None and step < min_new_tokens:
                logits[:, eos_token_id] = float("-inf")
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, float("-inf")), logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break
        return generated
