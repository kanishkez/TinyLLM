import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim,  bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):

    def __init__(self, dim, hidden, num_experts, top_k=2):
        super().__init__()

        self.top_k   = top_k
        self.gate    = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(num_experts)])

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(-1, D)

        weights, idx = torch.topk(F.softmax(self.gate(flat), dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            for i, expert in enumerate(self.experts):
                mask = idx[:, k] == i
                if mask.any():
                    out[mask] += weights[mask, k].unsqueeze(-1) * expert(flat[mask])

        return out.view(B, T, D)
