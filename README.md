# Deep learning from scratch

Deep learning studied from first principles up to training small language models, one topic at a time, with code that runs. Each folder has a `notes.md` with what I learned, what I measured and what did not work.

## Contents

| Folder | Topic | Based on |
|---|---|---|
| `00-foundations/udl` | Supervised learning, shallow and deep networks, losses, optimization, backprop, initialization, bias and variance, regularization | *Understanding Deep Learning*, Simon J. D. Prince |
| `00-foundations/math-dl` | ANN architectures, approximation with ReLU networks, GD/SGD theory and optimizers | *Mathematical Introduction to Deep Learning*, Jentzen, Kuckuck, von Wurstemberger |
| `01-micrograd` | Scalar autograd engine and a small MLP, checked against PyTorch | karpathy/micrograd, nn-zero-to-hero |
| `02-makemore` | Character-level language models: bigram, MLP, batchnorm, manual backprop, WaveNet | karpathy/makemore, nn-zero-to-hero |
| `03-transformer` | GPT trained on Tiny Shakespeare | karpathy/nanoGPT |
| `04-tokenization` | Byte pair encoding: basic and regex tokenizers with special tokens | karpathy/minbpe |
| `05-cuda` | Thrust, kernels, grid-stride loops, reductions, tiled matmul, streams | NVIDIA accelerated-computing-hub (CUDA C++) |
| `06-llm-training` | Llama-style model (RMSNorm, RoPE, SwiGLU, GQA) and a full training loop | FareedKhan-dev/train-llm-from-scratch |
| `07-nanochat` | Chat format, loss masking, KV cache, supervised fine-tuning | karpathy/nanochat |
| `08-ai-engineering` | Retrieval, RAG, tool calling and evals without frameworks | rohitg00/ai-engineering-from-scratch |
| `09-autoresearch` | Fixed-budget experiments driven by a search loop | karpathy/autoresearch |

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

Datasets are downloaded by the scripts into `data/` (ignored by git). Tests live next to the code they test; run `pytest` inside a folder.

The CUDA examples need the CUDA toolkit (`nvcc`):

```bash
cd 05-cuda && make run
```

The AI engineering examples talk to any OpenAI-compatible endpoint, a local Ollama model by default (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`).
