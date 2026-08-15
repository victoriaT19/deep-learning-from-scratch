SPECIAL = ["<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"]
SPECIAL_IDS = {name: 256 + i for i, name in enumerate(SPECIAL)}
VOCAB_SIZE = 256 + len(SPECIAL)


def encode_text(text):
    return list(text.encode("utf-8"))


def decode(ids):
    out = bytearray()
    for i in ids:
        if i < 256:
            out.append(i)
    return out.decode("utf-8", errors="replace")


def render_conversation(messages):
    ids, mask = [SPECIAL_IDS["<|bos|>"]], [0]
    for m in messages:
        if m["role"] == "user":
            part = [SPECIAL_IDS["<|user_start|>"]] + encode_text(m["content"]) + [SPECIAL_IDS["<|user_end|>"]]
            ids += part
            mask += [0] * len(part)
        else:
            ids.append(SPECIAL_IDS["<|assistant_start|>"])
            mask.append(0)
            part = encode_text(m["content"]) + [SPECIAL_IDS["<|assistant_end|>"]]
            ids += part
            mask += [1] * len(part)
    return ids, mask


def render_prompt(messages):
    ids, _ = render_conversation(messages)
    return ids + [SPECIAL_IDS["<|assistant_start|>"]]
