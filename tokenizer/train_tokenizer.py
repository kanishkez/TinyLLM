import argparse
import glob
from pathlib import Path


def train(files, out_dir, vocab_size=32000):

    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("pip install sentencepiece")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=",".join(files),
        model_prefix=f"{out_dir}/tokenizer",
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        byte_fallback=True,
        num_threads=16,
        shuffle_input_sentence=True,
    )

    print(f"saved to {out_dir}/tokenizer.model")


def load(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      nargs="+", required=True)
    p.add_argument("--output",     default="tokenizer/")
    p.add_argument("--vocab-size", type=int, default=32000)
    args = p.parse_args()

    files = []
    for pat in args.input:
        files.extend(glob.glob(pat))

    train(files, args.output, args.vocab_size)
