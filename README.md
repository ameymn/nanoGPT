# NanoGPT

A minimal GPT (Generative Pre-trained Transformer) implementation from scratch using PyTorch.

## Overview

This project implements a character-level language model based on the transformer architecture. It includes:

- Token and positional embeddings
- Multi-head self-attention mechanism
- Transformer blocks with layer normalization
- Feed-forward neural network layers
- Text generation capabilities

## Architecture

- **Embedding dimension**: 384
- **Attention heads**: 8
- **Transformer layers**: 6
- **Context length**: 256 tokens
- **Dropout**: 0.2

## Requirements

- Python 3.x
- PyTorch

## Usage

1. Place your training text in `input.txt`
2. Run the training script:

```bash
python gpt.py
```

The model will train on your text data and generate sample output after training.

## Files

- `gpt.py` - Main model implementation and training loop
- `input.txt` - Training data (text file)

## Credits

This project was built following [Andrej Karpathy's](https://github.com/karpathy) excellent tutorial ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY).

