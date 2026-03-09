import torch
from inference.sampler import sample


def generate(model, idx, max_new_tokens=200, temp=0.8, top_k=50, top_p=0.95, eos_id=2):

    model.eval()
    past_kvs = None

    with torch.no_grad():
        for _ in range(max_new_tokens):

            inp = idx if past_kvs is None else idx[:, -1:]

            with torch.autocast(device_type=idx.device.type, dtype=torch.bfloat16):
                logits, _, past_kvs = model(inp, past_kvs=past_kvs, use_cache=True)

            next_tok = sample(logits[:, -1, :], temp=temp, k=top_k, p=top_p)
            idx = torch.cat([idx, next_tok], dim=1)

            if (next_tok == eos_id).all():
                break

    return idx


def stream(model, idx, tokenizer, max_new_tokens=200, temp=0.8, top_k=50, top_p=0.95, eos_id=2):

    model.eval()
    past_kvs = None

    with torch.no_grad():
        for _ in range(max_new_tokens):

            inp = idx if past_kvs is None else idx[:, -1:]

            with torch.autocast(device_type=idx.device.type, dtype=torch.bfloat16):
                logits, _, past_kvs = model(inp, past_kvs=past_kvs, use_cache=True)

            next_tok = sample(logits[:, -1, :], temp=temp, k=top_k, p=top_p)
            idx = torch.cat([idx, next_tok], dim=1)

            yield tokenizer.decode(next_tok[0].tolist())

            if (next_tok == eos_id).all():
                break
