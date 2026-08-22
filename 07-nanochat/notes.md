# From base model to chat model — notes (karpathy/nanochat)

nanochat is the full pipeline in one small repo: tokenizer training → pretraining → mid-training on conversations → supervised fine-tuning → (optional) RL → evaluation → a chat UI, with an inference engine that keeps a KV cache. I reproduced the parts that were new to me at toy scale.

## Chat format (`chat_format.py`)
- Conversations become one token stream with special tokens: `<|bos|>`, `<|user_start|>…<|user_end|>`, `<|assistant_start|>…<|assistant_end|>`. The special ids sit above the 256 byte tokens and can never be produced by encoding user text.
- The loss mask is 1 only on the assistant's reply and its end token: the model learns to answer, not to imitate the user. `<|assistant_end|>` must be trained, otherwise the model never learns to stop.
- At inference the prompt ends with `<|assistant_start|>` and generation stops at `<|assistant_end|>`.

## KV cache (`chat_model.py`, `engine.py`)
- Without a cache every new token recomputes attention over the whole prefix: O(T²) work per token. With it, each layer stores the keys and values already computed and the new token attends to them: O(T).
- The prefill (whole prompt at once) still needs the causal mask; single-token decode steps do not. Positions must be offset by the cache length.
- `test_chat.py` checks that greedy generation with and without the cache produces exactly the same tokens.
- The cache size is `2 · layers · heads · head_dim · T` per sequence, which is why grouped-query attention (see `06-llm-training`) matters for serving.

## SFT on a synthetic task (`tasks.py`, `sft.py`)
- A small model trained from scratch on 20k addition conversations (templates in English and Portuguese), loss masked to the answer. Test accuracy on held-out pairs: 0% → 97% after 1,500 steps on CPU.
- In nanochat the same mechanics are applied to a pretrained base model with real conversation data (SmolTalk), plus mixtures for multiple choice and tool use, and evaluated on ARC, MMLU, GSM8K and HumanEval.

## What I took from it
- "Chat" is a data format plus a loss mask; the architecture does not change.
- Evaluation needs the same rendering as training, or the numbers mean nothing.
