import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model, cfg):

    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.95))


def cosine_schedule(optimizer, warmup, total, min_ratio=0.1):

    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * p)))

    return LambdaLR(optimizer, f)
