import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06-llm-training"))
from model import ModelConfig, Transformer

DEFAULT = {"dim": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "seq_len": 128, "batch_size": 16,
           "lr": 1e-3, "warmup": 50, "weight_decay": 0.1}


def run(config, data_dir, budget_s, seed=0):
    c = {**DEFAULT, **config}
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = {s: np.memmap(Path(data_dir) / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}

    def batch(split):
        d = data[split]
        ix = np.random.randint(0, len(d) - c["seq_len"] - 1, c["batch_size"])
        x = torch.from_numpy(np.stack([d[i:i + c["seq_len"]] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + 1 + c["seq_len"]] for i in ix]).astype(np.int64))
        return x, y

    model = Transformer(ModelConfig(vocab_size=256, seq_len=c["seq_len"], dim=c["dim"], n_layers=c["n_layers"],
                                    n_heads=c["n_heads"], n_kv_heads=c["n_kv_heads"]))
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"], betas=(0.9, 0.95))
    t0, step = time.time(), 0
    while time.time() - t0 < budget_s:
        for g in opt.param_groups:
            g["lr"] = c["lr"] * min(1.0, (step + 1) / c["warmup"])
        _, loss = model(*batch("train"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    model.eval()
    with torch.no_grad():
        val = float(np.mean([model(*batch("val"))[1].item() for _ in range(20)]))
    return {"val_bpb": val / math.log(2), "steps": step, "params": model.num_params()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="{}")
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "06-llm-training" / "data"))
    p.add_argument("--budget", type=float, default=300)
    args = p.parse_args()
    print(json.dumps(run(json.loads(args.config), args.data_dir, args.budget)))
