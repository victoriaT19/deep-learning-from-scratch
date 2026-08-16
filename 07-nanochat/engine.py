import torch

from chat_format import SPECIAL_IDS


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=32, temperature=0.0, use_cache=True, stop=SPECIAL_IDS["<|assistant_end|>"]):
    model.eval()
    ids = list(prompt_ids)
    if not use_cache:
        for _ in range(max_new_tokens):
            logits, _ = model(torch.tensor([ids[-model.cfg.max_len:]]))
            nxt = pick(logits[0, -1], temperature)
            ids.append(nxt)
            if nxt == stop:
                break
        return ids[len(prompt_ids):]
    caches = [{} for _ in model.blocks]
    logits, _ = model(torch.tensor([ids]), caches=caches)
    out = []
    for _ in range(max_new_tokens):
        nxt = pick(logits[0, -1], temperature)
        out.append(nxt)
        if nxt == stop or len(ids) + len(out) >= model.cfg.max_len:
            break
        logits, _ = model(torch.tensor([[nxt]]), caches=caches, start=len(ids) + len(out) - 1)
    return out


def pick(logits, temperature):
    if temperature == 0.0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
