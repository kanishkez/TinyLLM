import torch
import torch.nn as nn


class RoPE(nn.Module):

    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x):

        T = x.shape[-2]

        if T > self.max_len:
            self._extend(T, x.device)

        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        rotated = torch.cat([-x2, x1], dim=-1)

        return (x * cos) + (rotated * sin)

    def _extend(self, seq_len, device):

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )

        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)

        self.cos = emb.cos()
        self.sin = emb.sin()

        self.max_len = seq_len
