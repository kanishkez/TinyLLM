import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import yaml

from models.transformer import Transformer
from inference.generate import generate


def load_cfg():
    root = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(root, "configs", "model.yaml")

    with open(cfg_path) as f:
        return yaml.safe_load(f)


def test_forward():

    cfg = load_cfg()
    model = Transformer(cfg)

    B = 2
    T = cfg["seq_len"]

    x = torch.randint(0, cfg["vocab_size"], (B, T))
    y = torch.randint(0, cfg["vocab_size"], (B, T))

    logits, loss, _ = model(x, labels=y)

    print("Forward pass OK")
    print("logits shape:", logits.shape)
    print("loss:", loss.item())


def test_backward():

    cfg = load_cfg()
    model = Transformer(cfg)

    B = 2
    T = cfg["seq_len"]

    x = torch.randint(0, cfg["vocab_size"], (B, T))
    y = torch.randint(0, cfg["vocab_size"], (B, T))

    logits, loss, _ = model(x, labels=y)

    loss.backward()

    print("Backward pass OK")


def test_generation():

    cfg = load_cfg()
    model = Transformer(cfg)

    idx = torch.randint(0, cfg["vocab_size"], (1, 10))

    out = generate(
        model,
        idx,
        max_new_tokens=20,
        temp=0.8,
        top_k=40,
        top_p=0.9
    )

    print("Generation OK")
    print("output shape:", out.shape)


def test_kv_cache():

    cfg = load_cfg()
    model = Transformer(cfg)

    idx = torch.randint(0, cfg["vocab_size"], (1, 5))

    logits, _, past = model(idx, use_cache=True)

    next_token = torch.randint(0, cfg["vocab_size"], (1, 1))

    logits2, _, past2 = model(next_token, past_kvs=past, use_cache=True)

    print("KV cache OK")


def run_all():

    print("\nTesting Forward Pass:")
    test_forward()

    print("\nTesting Backward Pass:")
    test_backward()

    print("\nTesting KV Caching:")
    test_kv_cache()

    print("\nTesting Generation:")
    test_generation()

    print("\nTesting Done")


if __name__ == "__main__":
    run_all()
