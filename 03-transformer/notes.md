# Transformers — notes (UDL ch. 12, "Let's build GPT", nanoGPT)

## UDL ch. 12
- Self-attention computes, for each position, a weighted average of value vectors; the weights are a softmax over query·key dot products. Parameters are shared across positions, so the layer works for any sequence length.
- Scaled dot product: divide by `√d_k` so the logits keep unit variance and the softmax does not saturate.
- Multi-head attention runs several smaller attentions in parallel and concatenates them; each head can attend to different relations.
- A transformer layer = attention + MLP, each wrapped in a residual connection and a LayerNorm. Position must be injected explicitly (absolute embeddings, sinusoids, or relative/rotary encodings).
- Encoder (BERT, bidirectional), decoder (GPT, causal mask), encoder–decoder (translation).

## Building GPT
- `attention_math.py`: the "mathematical trick" — averaging over the past with a loop, with a lower-triangular matrix multiply, and with a masked softmax all give the same result; attention just makes the weights data-dependent. It also checks how the `1/√d` scaling shrinks the logits' variance.
- `gpt.py`: a nanoGPT-style decoder: token + position embeddings, pre-LayerNorm blocks with causal multi-head attention and a GELU MLP, weight tying between the embedding and the output head, `std=0.02` initialization.
- `train.py`: AdamW, periodic estimate of train/val loss. The default config (6 layers, 6 heads, 384 dims, context 256, ~10.8M parameters) is the one from the lecture; on a GPU it reaches a val loss around 1.48 on Tiny Shakespeare. On CPU I trained a tiny version just to check everything runs.
- `sample.py`: generation with temperature and top-k.

## What changed in my understanding
- Attention is communication between tokens, the MLP is computation per token.
- Residual connections + LayerNorm are what make depth trainable (UDL ch. 11 again).
- Dropout on the attention weights and on the residual branches is the main regularizer at this scale.
