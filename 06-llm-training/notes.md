# Training an LLM from scratch — notes (FareedKhan-dev/train-llm-from-scratch, nanoGPT)

The nanoGPT model of `03-transformer` is the 2019 recipe. This module rebuilds the pipeline with the pieces most current open models use, small enough to run on a laptop and the same code for a GPU.

## Model (`model.py`)
- **RMSNorm** instead of LayerNorm: no mean subtraction, no bias, one gain vector. Cheaper and just as stable.
- **RoPE**: queries and keys are rotated by an angle proportional to the position, so `q·k` depends only on the *relative* position. No learned position table, and it extrapolates a little beyond the training length.
- **SwiGLU** feed-forward: `W2(SiLU(W1 x) ⊙ W3 x)`, hidden size `8/3·d` rounded to a multiple of 64 so the parameter count matches a 4·d GELU MLP.
- **Grouped-query attention**: 6 query heads share 2 key/value heads. Fewer KV parameters and, more importantly, a 3× smaller KV cache at inference.
- `F.scaled_dot_product_attention(is_causal=True)` dispatches to a fused (Flash) kernel on GPU.
- Output projections initialised with `std = 0.02/√(2·n_layers)` so the residual stream does not grow with depth (GPT-2 trick).

## Data (`prepare.py`)
- Byte-level tokens (vocab 256) written once to `train.bin` / `val.bin` as `uint16`, read with `np.memmap`: the dataset never has to fit in RAM and batches are random slices. A BPE tokenizer from `04-tokenization` drops in by replacing the encoding step.

## Training loop (`train.py`)
- AdamW with `betas=(0.9, 0.95)`, weight decay only on matrices (not on norms/gains).
- Linear warm-up then cosine decay to 10% of the peak learning rate.
- Gradient accumulation to reach a large effective batch with little memory; loss divided by the number of micro-steps.
- Gradient clipping at norm 1.0; printing the norm is the quickest way to see instabilities.
- bf16 autocast on GPUs that support it (no loss scaling needed, unlike fp16).
- Validation loss reported also as bits per byte, comparable across tokenizers.
- Checkpoint on best validation loss with optimizer state, `--resume` to continue.

`generate.py` samples with temperature and nucleus (top-p) sampling.

On CPU I only ran a 0.1M-parameter configuration to check the pipeline end to end (loss from 5.54 to about 3.0 in 150 steps); the default configuration is meant for a GPU.
