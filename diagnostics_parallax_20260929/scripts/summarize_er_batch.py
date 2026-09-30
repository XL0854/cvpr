"""Aggregate one-scene e-r diagnostics without changing locked pair results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair_dirs = sorted(path for path in args.root.glob("er_courtyard_*") if (path/"matches.npz").exists())
    pair_rows, pooled = [], {key: [] for key in ("ratio", "correct", "large10", "small10", "incorrect")}
    for path in pair_dirs:
        data = np.load(path/"matches.npz", allow_pickle=True)
        correctness, residual, ratio = data["correctness"], data["r_canvas"], data["ratios"]
        correct = correctness == "correct"
        incorrect = correctness == "incorrect"
        large = correct & (residual >= 10)
        small = correct & (residual < 10)
        source = data["candidates"][:, :2]
        cells = np.floor(source[large] / np.array([6048/8, 4032/8])).astype(int)
        occupied = len(np.unique(cells[:, 0]+8*cells[:, 1])) if len(cells) else 0
        pair_rows.append({
            "pair": path.name.removeprefix("er_courtyard_"),
            "candidates": len(correctness), "evaluable": int((correct|incorrect|(correctness=="uncertain")).sum()),
            "correct": int(correct.sum()), "incorrect": int(incorrect.sum()),
            "correct_r_ge_5": int((correct & (residual >= 5)).sum()),
            "correct_r_ge_10": int(large.sum()), "correct_r_ge_20": int((correct & (residual >= 20)).sum()),
            "large10_source_cells_8x8": occupied,
            "correct_r_median": float(np.median(residual[correct])) if correct.any() else None,
            "correct_r_p90": float(np.percentile(residual[correct], 90)) if correct.any() else None,
        })
        for key, value in (("ratio", ratio), ("correct", correct), ("large10", large),
                           ("small10", small), ("incorrect", incorrect)):
            pooled[key].append(value)
    pooled = {key: np.concatenate(value) for key, value in pooled.items()}
    comparisons = {}
    for target in (.01, .05, .10):
        wrong_ratios = np.sort(pooled["ratio"][pooled["incorrect"]])
        index = max(0, min(len(wrong_ratios)-1, int(np.floor(target*len(wrong_ratios)))-1))
        threshold = float(wrong_ratios[index]) if len(wrong_ratios) else 0.0
        accepted = pooled["ratio"] <= threshold
        comparisons[str(target)] = {
            "ratio_threshold": threshold,
            "actual_incorrect_acceptance": float((accepted & pooled["incorrect"]).sum()/max(pooled["incorrect"].sum(), 1)),
            "all_correct_retention": float((accepted & pooled["correct"]).sum()/max(pooled["correct"].sum(), 1)),
            "large_correct_retention_r_ge_10": float((accepted & pooled["large10"]).sum()/max(pooled["large10"].sum(), 1)),
            "small_correct_retention_r_lt_10": float((accepted & pooled["small10"]).sum()/max(pooled["small10"].sum(), 1)),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"per_pair.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    total_correct = int(pooled["correct"].sum())
    summary = {
        "status": "single_scene_small_batch_complete", "pairs": len(pair_rows),
        "per_pair": pair_rows,
        "pooled": {
            "correct": total_correct, "incorrect": int(pooled["incorrect"].sum()),
            "correct_r_ge_10": int(pooled["large10"].sum()),
            "correct_r_ge_10_fraction": float(pooled["large10"].sum()/max(total_correct, 1)),
        },
        "lowe_ratio_at_matched_incorrect_acceptance": comparisons,
        "interpretation_limit": "Exploratory five-pair result from one ETH3D scene; pairs intentionally span baseline difficulty and are not an i.i.d. benchmark sample.",
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
