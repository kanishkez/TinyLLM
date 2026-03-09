import math
import yaml
import torch
import torch.nn as nn

from models.transformer import Transformer, load_cfg
from training.dataset import get_loader
from training.optimizer import build_optimizer, cosine_schedule


def train():

    mcfg = load_cfg("configs/model.yaml")
    tcfg = yaml.safe_load(open("configs/train.yaml"))

    device = tcfg.get("device", "cuda")
    dtype  = torch.bfloat16 if tcfg.get("dtype") == "bfloat16" else torch.float32

    train_loader = get_loader(tcfg["dataset_path"], mcfg["seq_len"], tcfg["batch_size"])
    val_loader   = get_loader(tcfg["val_path"],     mcfg["seq_len"], tcfg["batch_size"])

    model = Transformer(mcfg).to(device)

    if tcfg.get("compile"):
        model = torch.compile(model)

    print(f"params: {model.num_params() / 1e6:.1f}M")

    accum  = tcfg.get("grad_accum", 1)
    total  = (len(train_loader) // accum) * tcfg["epochs"]
    opt    = build_optimizer(model, tcfg)
    sched  = cosine_schedule(opt, tcfg.get("warmup_steps", 100), total,
                              tcfg.get("min_lr", 3e-5) / tcfg["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    step   = 0

    for epoch in range(tcfg["epochs"]):

        model.train()

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device, dtype=dtype):
                _, loss, _ = model(x, labels=y)
                loss = loss / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                if step % tcfg.get("log_every", 10) == 0:
                    lr = sched.get_last_lr()[0]
                    l  = loss.item() * accum
                    print(f"step={step}  loss={l:.4f}  ppl={math.exp(min(l, 20)):.1f}  lr={lr:.2e}")

                if step % tcfg.get("save_every", 1000) == 0:
                    torch.save(model.state_dict(), f"{tcfg['save_path']}/step_{step}.pt")

        val_loss = evaluate(model, val_loader, device, dtype)
        print(f"epoch={epoch + 1}  val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), f"{tcfg['save_path']}/final.pt")


@torch.no_grad()
def evaluate(model, loader, device, dtype, max_batches=100):

    model.eval()
    total, n = 0.0, 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss, _ = model(x, labels=y)
        total += loss.item()
        n += 1

    model.train()
    return total / max(n, 1)


if __name__ == "__main__":
    train()
