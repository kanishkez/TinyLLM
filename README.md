# TinyLLM

A lightweight, from-scratch implementation of a large language model (LLM) in Python. TinyLLM covers the full pipeline: tokenization, dataset preparation, model architecture, training, and inference.

---

## Overview

TinyLLM is a self-contained framework for building and training transformer-based language models from the ground up. The project is organized into discrete components that mirror how production LLM systems are structured, making it a practical reference for understanding the mechanics of modern language models without the overhead of large-scale libraries.

---

## Repository Structure

```
TinyLLM/
├── configs/        # Hyperparameter and architecture configuration files
├── datasets/       # Data loading, preprocessing, and batching utilities
├── inference/      # Text generation and sampling logic
├── models/         # Transformer architecture implementation
├── tests/          # Unit tests for individual components
├── tokenizer/      # Tokenizer training and encoding/decoding utilities
└── training/       # Training loop, loss computation, and checkpointing
```

---

## Architecture

TinyLLM implements a decoder-only autoregressive transformer. The reference configuration targets approximately 200M parameters.

| Component | Choice |
|---|---|
| Architecture | Decoder-only Transformer (Pre-Norm) |
| Normalization | RMSNorm |
| Positional Encoding | RoPE |
| Attention | Multi-head with Grouped Query Attention (GQA) |
| Feedforward | SwiGLU |
| KV Caching | Supported |
| Mixture of Experts | Optional (8 experts, top-2 routing) |
| Linear Attention | Optional (DeltaNet, alternating layers) |
| Training Objective | Next-token prediction (cross-entropy) |
| Inference Sampling | Temperature, top-k, top-p |


---

### Component Overview

**`configs/`**
Stores configuration files (YAML or JSON) that define model hyperparameters such as embedding dimension, number of attention heads, number of layers, learning rate, and sequence length. Separating configuration from code makes it easy to reproduce and compare experiments.

**`datasets/`**
Handles raw text ingestion, dataset splitting, and the construction of input/target token sequences for language modeling. Includes utilities for batching and shuffling during training.

**`inference/`**
Implements autoregressive text generation. Supports greedy decoding and sampling strategies such as top-k and temperature scaling. Loads trained checkpoints for interactive use.

**`models/`**
Contains the core transformer architecture including multi-head self-attention, feed-forward layers, positional encoding, and layer normalization. Implemented in pure Python/PyTorch with no external model libraries.

**`tests/`**
Unit tests covering individual components such as attention mechanisms, tokenizer correctness, and data pipeline integrity.

**`tokenizer/`**
Implements or wraps a tokenizer (e.g., Byte-Pair Encoding) that can be trained on a custom corpus or used with a pre-built vocabulary. Includes encode and decode utilities.

**`training/`**
The main training loop including forward passes, loss computation (cross-entropy), backpropagation, optimizer stepping, gradient clipping, and checkpoint saving and resumption.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (CPU or CUDA)

### Installation

```bash
git clone https://github.com/kanishkez/TinyLLM.git
cd TinyLLM
pip install -r requirements.txt
```

### Training

1. Place your training corpus in the `datasets/` directory.
2. Configure your model and training parameters in `configs/`.
3. Train the tokenizer on your corpus:

```bash
python tokenizer/train_tokenizer.py --data datasets/your_corpus.txt
```

4. Run the training loop:

```bash
python training/train.py --config configs/default.yaml
```

Checkpoints will be saved at the interval specified in your config file.

### Inference

Once training is complete, generate text using a saved checkpoint:

```bash
python inference/generate.py --checkpoint path/to/checkpoint.pt --prompt "Once upon a time"
```

### Running Tests

```bash
python -m pytest tests/
```

---

## Design Goals

- **Readability** — Code is written to be read and understood, not just executed. Each module is self-contained with minimal external dependencies.
- **Modularity** — Components (tokenizer, model, trainer) are decoupled and can be swapped or extended independently.
- **Educational** — The project is structured to mirror professional LLM codebases, making it a practical learning resource.

---

## Contributing

Contributions are welcome. To propose a change, open an issue first to discuss the approach, then submit a pull request with a clear description of what was changed and why.

---

## License

This project is licensed under the [MIT License](LICENSE).




