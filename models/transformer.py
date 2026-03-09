import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .attention import Attention
from .moe import MoE
from .deltanet import DeltaNet


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()

        self.eps = float(eps)
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)

        return self.w * x


class FFN(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim,  bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):

    def __init__(self, cfg, layer_idx):
        super().__init__()

        dim    = cfg["dim"]
        hidden = dim * cfg["ff_mult"]

        use_delta     = cfg.get("use_deltanet", False) and layer_idx % 2 == 1
        self.attn     = DeltaNet(dim, cfg["n_heads"]) if use_delta else Attention(cfg)
        self.is_delta = use_delta
        self.ffn      = MoE(dim, hidden, cfg["num_experts"], cfg.get("num_experts_per_tok", 2)) if cfg.get("use_moe") else FFN(dim, hidden)
        self.n1       = RMSNorm(dim, cfg.get("norm_eps", 1e-5))
        self.n2       = RMSNorm(dim, cfg.get("norm_eps", 1e-5))

    def forward(self, x, past_kv=None, use_cache=False):

        if self.is_delta:
            h, _   = self.attn(self.n1(x))
            new_kv = None
        else:
            h, new_kv = self.attn(self.n1(x), past_kv=past_kv, use_cache=use_cache)

        x = x + h
        x = x + self.ffn(self.n2(x))

        return x, new_kv


class Transformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg["n_layers"])])
        self.norm   = RMSNorm(cfg["dim"], cfg.get("norm_eps", 1e-5))
        self.head   = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx, past_kvs=None, labels=None, use_cache=False):
        x        = self.embed(idx)
        past_kvs = past_kvs or [None] * len(self.blocks)
        new_kvs  = []

        for block, past in zip(self.blocks, past_kvs):
            x, kv = block(x, past_kv=past, use_cache=use_cache)
            new_kvs.append(kv)

        x      = self.norm(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.cfg["vocab_size"]),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_kvs if use_cache else None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def load_cfg(path="configs/model.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
