import argparse
import glob
import random
import numpy as np
from pathlib import Path
from tokenizer.train_tokenizer import load


def prepare(files, tokenizer_path, out_dir, val_frac=0.005):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sp  = load(tokenizer_path)
    rng = random.Random(42)

    train_toks, val_toks = [], []

    for path in files:
        print(f"tokenizing {path}")
        text = open(path, encoding="utf-8", errors="ignore").read()
        toks = sp.encode(text) + [sp.eos_id()]

        if rng.random() < val_frac:
            val_toks.extend(toks)
        else:
            train_toks.extend(toks)

    np.array(train_toks, dtype=np.uint16).tofile(f"{out_dir}/train.bin")
    np.array(val_toks,   dtype=np.uint16).tofile(f"{out_dir}/val.bin")

    print(f"train: {len(train_toks):,}  val: {len(val_toks):,} tokens")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",     nargs="+", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output",    default="datasets/processed/")
    p.add_argument("--val-frac",  type=float, default=0.005)
    args = p.parse_args()

    files = []
    for pat in args.input:
        files.extend(glob.glob(pat))

    prepare(files, args.tokenizer, args.output, args.val_frac)
