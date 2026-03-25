## Train

```bash
python3 -m AR.llama.train \
  --tokenizer llama \
  --tokenizer_path "AR/tokenizer.model" \
  --vocab_size 10001 \
  --seq_len 512 \
  --window_size 256 \
  --batch_size 32 \
  --dim 1024 \
  --n_layers 2 \
  --n_heads 16 \
  --n_kv_heads 16 \
  --hidden_dim 3328 \
  --max_steps 100000
```

```bash
python3 -m AR.llama.train \
  --tokenizer byte \
  --seq_len 512 \
  --window_size 256 \
  --batch_size 32 \
  --dim 1024 \
  --n_layers 2 \
  --n_heads 16 \
  --n_kv_heads 16 \
  --hidden_dim 6144 \
  --max_steps 100000
```
## Sample

```bash
python3 -m AR.llama.sample \
  --checkpoint "AR/llama/outputs/tinystories-small/checkpoint_step_20000.pt" \
  --tokenizer llama \
  --vocab_size 10001 \
  --prompt "Once upon a time"
```

## Visualize RoPE

```bash
python3 -m AR.llama.visualize_rope \
  --dim 1024 \
  --n_heads 16 \
  --seq_len 512 \
  --window_size 256 \
  --output "AR/llama/outputs/rope_frequencies.png"
```
