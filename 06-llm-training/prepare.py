import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--out-dir", default="data")
p.add_argument("--val-fraction", type=float, default=0.05)
args = p.parse_args()

text = Path(args.input).read_text(encoding="utf-8")
ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)
n_val = int(len(ids) * args.val_fraction)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
ids[:-n_val].tofile(out / "train.bin")
ids[-n_val:].tofile(out / "val.bin")
print(f"{len(ids):,} byte tokens: train {len(ids) - n_val:,}, val {n_val:,}")
