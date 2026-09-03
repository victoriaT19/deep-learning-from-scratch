import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = {
    "dim": [64, 128, 192, 256],
    "n_layers": [2, 4, 6, 8],
    "lr": [3e-4, 6e-4, 1e-3, 2e-3, 3e-3],
    "seq_len": [64, 128, 256],
    "batch_size": [8, 16, 32],
    "warmup": [20, 50, 100],
    "weight_decay": [0.0, 0.05, 0.1],
}


def propose(best, rng):
    key = rng.choice(list(SPACE))
    options = [v for v in SPACE[key] if v != best.get(key)]
    return key, {**best, key: rng.choice(options)}


def evaluate(config, budget):
    out = subprocess.run([sys.executable, str(HERE / "experiment.py"), "--config", json.dumps(config), "--budget", str(budget)],
                         capture_output=True, text=True, timeout=budget * 3 + 120)
    if out.returncode != 0:
        return None, out.stderr.strip().splitlines()[-1] if out.stderr else "crashed"
    return json.loads(out.stdout.strip().splitlines()[-1]), ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--budget", type=float, default=300)
    p.add_argument("--log", default=str(HERE / "results.tsv"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    best = {"dim": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "lr": 1e-3, "seq_len": 128,
            "batch_size": 16, "warmup": 50, "weight_decay": 0.1}
    result, _ = evaluate(best, args.budget)
    best_bpb = result["val_bpb"]
    new_file = not Path(args.log).exists()
    with open(args.log, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new_file:
            w.writerow(["time", "change", "val_bpb", "steps", "params", "kept", "config", "error"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), "baseline", f"{best_bpb:.4f}", result["steps"], result["params"], True, json.dumps(best), ""])
        print(f"baseline {best_bpb:.4f} bits/byte")
        tried = {json.dumps(best, sort_keys=True)}
        for r in range(args.rounds):
            key, candidate = propose(best, rng)
            signature = json.dumps(candidate, sort_keys=True)
            if candidate["dim"] % candidate["n_heads"] or signature in tried:
                continue
            tried.add(signature)
            result, error = evaluate(candidate, args.budget)
            bpb = result["val_bpb"] if result else float("inf")
            kept = bpb < best_bpb
            if kept:
                best, best_bpb = candidate, bpb
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), f"{key}={candidate[key]}", f"{bpb:.4f}",
                        result["steps"] if result else "", result["params"] if result else "", kept, json.dumps(candidate), error])
            f.flush()
            print(f"round {r + 1}: {key}={candidate[key]} -> {bpb:.4f} {'kept' if kept else 'reverted'} (best {best_bpb:.4f})")
    print("best config:", json.dumps(best))


if __name__ == "__main__":
    main()
