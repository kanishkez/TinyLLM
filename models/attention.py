import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPE

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.n_heads    = cfg["n_heads"]
        self.n_kv_heads = cfg.get("n_kv_heads", cfg["n_heads"])
        self.head_dim   = cfg["dim"] // self.n_heads
        self.n_rep      = self.n_heads // self.n_kv_heads
        dim = cfg["dim"]

        self.q = nn.Linear(dim, self.n_heads    * self.head_dim, bias=False)
        self.k = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        self.rope = RoPE(self.head_dim, cfg["seq_len"], cfg.get("rope_base", 10000))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        h, kh, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q(x).view(B, T, h,  hd).transpose(1, 2)
        k = self.k(x).view(B, T, kh, hd).transpose(1, 2)
        v = self.v(x).view(B, T, kh, hd).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if HAS_FLASH:
            out = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
            ).transpose(1, 2)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o(out), new_kv
