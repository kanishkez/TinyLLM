import torch
import torch.nn.functional as F


def top_k(logits, k):
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = float("-inf")
    return logits


def top_p(logits, p):
    sorted_l, idx = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
    sorted_l[cum - F.softmax(sorted_l, dim=-1) > p] = float("-inf")
    return logits.scatter(-1, idx, sorted_l)


def sample(logits, temp=1.0, k=50, p=0.95):

    logits = logits / temp

    if k > 0:
        logits = top_k(logits, k)

    if p < 1.0:
        logits = top_p(logits, p)

    return torch.multinomial(F.softmax(logits, dim=-1), 1)
