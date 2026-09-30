"""Aggregate paired adaptive/full-search records without rerunning inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--input1", type=Path, required=True)
    parser.add_argument("--input2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names1 = {p.name for p in args.input1.iterdir() if p.is_file()}
    names2 = {p.name for p in args.input2.iterdir() if p.is_file()}
    names = sorted(names1 & names2)
    if not names:
        raise ValueError("No paired input files found")
    records = []
    missing = []
    for name in names:
        path = args.results / (name + ".json")
        if not path.exists():
            missing.append(name)
            continue
        record = json.loads(path.read_text())
        if record.get("name") != name or not {"adaptive", "full"} <= record.keys():
            missing.append(name)
            continue
        records.append(record)
    if missing:
        raise ValueError(f"Missing/incomplete paired results: {len(missing)}; first: {missing[:5]}")

    def vals(method, key):
        return np.asarray([record[method][key] for record in records], dtype=np.float64)

    def mean(method, key):
        return float(vals(method, key).mean())

    rng = np.random.default_rng(20260922)
    n = len(records)
    bootstrap_indices = rng.integers(0, n, size=(5000, n))

    def paired_delta(key):
        delta = vals("adaptive", key) - vals("full", key)
        means = delta[bootstrap_indices].mean(axis=1)
        return {
            "mean": float(delta.mean()),
            "paired_bootstrap_95pct": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "adaptive_higher": int((delta > 1e-9).sum()),
            "equal": int((np.abs(delta) <= 1e-9).sum()),
            "adaptive_lower": int((delta < -1e-9).sum()),
        }

    result = {
        "n": n,
        "protocol": "Paired image files; adaptive and full search on the same inputs; 512x512 resize",
        "adaptive": {key: mean("adaptive", key) for key in
                     ("calls", "fallback", "ssim", "psnr", "elapsed_seconds")},
        "full": {key: mean("full", key) for key in
                 ("calls", "ssim", "psnr", "elapsed_seconds")},
        "speedup_ratio_of_mean_times": mean("full", "elapsed_seconds") / mean("adaptive", "elapsed_seconds"),
        "delta_ssim": paired_delta("ssim"),
        "delta_psnr_db": paired_delta("psnr"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
