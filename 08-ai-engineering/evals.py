import json
import re
import statistics
import time


def normalize(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)


def contains(pred, gold):
    return normalize(gold) in normalize(pred)


def evaluate(cases, system, metric=contains):
    rows = []
    for case in cases:
        t0 = time.time()
        try:
            pred, error = system(case["input"]), None
        except Exception as e:
            pred, error = "", f"{type(e).__name__}: {e}"
        rows.append({"id": case["id"], "pred": pred, "gold": case["gold"], "ok": error is None and metric(pred, case["gold"]),
                     "error": error, "seconds": round(time.time() - t0, 3)})
    summary = {"cases": len(rows), "accuracy": sum(r["ok"] for r in rows) / max(len(rows), 1),
               "errors": sum(r["error"] is not None for r in rows),
               "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else 0.0}
    return rows, summary


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
