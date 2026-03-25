# 빠른 디버깅 스크립트
import torch
ckpt = torch.load("outputs/tinystories-small/checkpoint_step_100000.pt", map_location="cpu", weights_only=False)
print("Config:", ckpt["config"])
print("Embedding weight shape:", ckpt["model"]["tok_embeddings.weight"].shape)
