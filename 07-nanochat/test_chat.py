import torch

from chat_format import SPECIAL_IDS, VOCAB_SIZE, decode, render_conversation, render_prompt
from chat_model import ChatModel, Config
from engine import generate


def test_mask_covers_only_the_assistant_reply():
    conv = [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "olá"}]
    ids, mask = render_conversation(conv)
    trained = [i for i, m in zip(ids, mask) if m]
    assert decode(trained) == "olá"
    assert trained[-1] == SPECIAL_IDS["<|assistant_end|>"]
    assert render_prompt(conv[:1])[-1] == SPECIAL_IDS["<|assistant_start|>"]


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    model = ChatModel(Config(vocab_size=VOCAB_SIZE))
    prompt = render_prompt([{"role": "user", "content": "What is 2 plus 3?"}])
    with_cache = generate(model, prompt, max_new_tokens=20, use_cache=True, stop=-1)
    without = generate(model, prompt, max_new_tokens=20, use_cache=False, stop=-1)
    assert with_cache == without
