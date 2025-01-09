# Transformer-Based Language Modeling for Text Data

This project implements a Transformer-based language model from scratch using PyTorch, focusing on two main components:

1. A character-level counter using a custom Transformer encoder
2. A character-level language model using Transformer architecture

## Project Structure

- `transformer.py`: Implementation of the custom Transformer encoder
- `transformer_lm.py`: Implementation of the Transformer-based language model
- `letter_counting.py`: Driver for the character counting task
- `lm.py`: Driver for the language modeling task
- `utils.py`: Utility functions for data processing and model evaluation

## Tasks

### 1. Character Counter with Custom Transformer
- Implements a simplified Transformer from scratch
- Task: Count character occurrences in a sequence
- Features:
  - Self-attention mechanism
  - Positional encodings
  - Single-head attention
  - Residual connections

### 2. Language Modeling with Transformer
- Character-level language modeling using Transformer architecture
- Features:
  - Causal masking for autoregressive prediction
  - Chunk-based processing
  - Perplexity-based evaluation

## Requirements
- pytorch
- numpy

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `data` directory and add the required text files:
   - text8-100k.txt (training data)
   - text8-dev.txt (development set)
   - text8-test.txt (test set)
   - lettercounting-train.txt
   - lettercounting-dev.txt

## Model Architecture

The Transformer implementation includes:
- Embedding layer with positional encodings
- Self-attention mechanism
- Feed-forward neural networks
- Residual connections
- Output projection layer

## Performance

- Character Counter: Achieves >95% accuracy on the counting task
- Language Model: Achieves perplexity ≤7 on the development set
