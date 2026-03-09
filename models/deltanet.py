import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaNet(nn.Module):

    def __init__(self, dim, n_heads=8, key_dim=64, val_dim=64):
        super().__init__()

        self.h  = n_heads
        self.dk = key_dim
        self.dv = val_dim

        self.q    = nn.Linear(dim, n_heads * key_dim, bias=False)
        self.k    = nn.Linear(dim, n_heads * key_dim, bias=False)
        self.v    = nn.Linear(dim, n_heads * val_dim,  bias=False)
        self.o    = nn.Linear(n_heads * val_dim, dim,  bias=False)
        self.beta = nn.Linear(dim, n_heads, bias=False)
        self.qn   = nn.RMSNorm(key_dim)
        self.kn   = nn.RMSNorm(key_dim)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        h, dk, dv = self.h, self.dk, self.dv

        q = self.qn(self.q(x).view(B, T, h, dk))
        k = self.kn(self.k(x).view(B, T, h, dk))
        v = self.v(x).view(B, T, h, dv)
        b = torch.sigmoid(self.beta(x)).unsqueeze(-1)

        if state is None:
            state = torch.zeros(B, h, dk, dv, device=x.device, dtype=x.dtype)

        outs = []
        for t in range(T):
            sk = torch.einsum("bhd,bhdv->bhv", k[:, t], state)
            state = state + b[:, t] * torch.einsum("bhd,bhv->bhdv", k[:, t], v[:, t] - sk)
            outs.append(torch.einsum("bhd,bhdv->bhv", q[:, t], state))

        out = torch.stack(outs, dim=1).reshape(B, T, h * dv)
        return self.o(out), state
