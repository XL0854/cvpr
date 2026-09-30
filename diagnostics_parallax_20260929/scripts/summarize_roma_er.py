"""Audit fixed reliability filtering with explicit denominators and per-pair rows."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnose_roma_batch import retention_at_wrong_acceptance


def classified_stats(accepted, correct, incorrect, large, small, cell):
    good, bad = int((accepted & correct).sum()), int((accepted & incorrect).sum())
    classified = good + bad
    return {
        "kept_correct": good, "kept_incorrect": bad, "kept_classified": classified,
        "post_filter_precision": good / classified if classified else None,
        "incorrect_acceptance_numerator": bad,
        "incorrect_acceptance_denominator": int(incorrect.sum()),
        "incorrect_acceptance": bad / max(int(incorrect.sum()), 1),
        "correct_large_kept": int((accepted & large).sum()),
        "correct_large_total": int(large.sum()),
        "correct_large_retention": float((accepted & large).sum()/max(int(large.sum()), 1)),
        "correct_small_kept": int((accepted & small).sum()),
        "correct_small_total": int(small.sum()),
        "correct_small_retention": float((accepted & small).sum()/max(int(small.sum()), 1)),
        "large_correct_cells_before": int(len(np.unique(cell[large]))),
        "large_correct_cells_after": int(len(np.unique(cell[large & accepted]))),
        "all_kept_source_cells": int(len(np.unique(cell[accepted]))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    items = []
    for path in sorted(args.diagnosis.glob("DSC_*/diagnosis.npz")):
        candidate = np.load(args.candidates/path.parent.name/"candidates.npz")
        result = np.load(path, allow_pickle=True)
        items.append({"id": path.parent.name,
                      **{key: result[key] for key in result.files},
                      "source_512": candidate["source_512"]})
    join = lambda key: np.concatenate([item[key] for item in items])
    correctness, residual = join("correctness"), join("r_canvas")
    correct, incorrect = correctness == "correct", correctness == "incorrect"
    uncertain, unable = correctness == "uncertain", correctness == "unable"
    large, small = correct & (residual >= 10), correct & (residual < 10)
    confidence = np.minimum(join("confidence"), join("reverse_confidence"))
    cycle = join("cycle_512_px")
    standard = ((join("confidence") >= .5) & (join("reverse_confidence") >= .5) & (cycle <= 2))
    points = join("source_512")
    cells = np.floor(points/64).astype(int)
    cell_id = cells[:, 0] + 8*cells[:, 1]
    pair_rows, offset = [], 0
    for item in items:
        n = len(item["correctness"]); sl = slice(offset, offset+n); offset += n
        pair_rows.append({"pair": item["id"], **classified_stats(
            standard[sl], correct[sl], incorrect[sl], large[sl], small[sl], cell_id[sl]),
            "uncertain_total": int(uncertain[sl].sum()), "unable_total": int(unable[sl].sum())})
    with (args.diagnosis/"filter_audit_per_pair.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    report = {
        "status": "fixed_filter_audit_complete", "pairs": len(items),
        "far_denominator_definition": "truth-evaluable incorrect candidates (e_native>6 px); uncertain 3<e<=6 and unavailable excluded",
        "precision_denominator_definition": "kept classified candidates: correct plus incorrect; uncertain/unavailable excluded",
        "counts": {"correct": int(correct.sum()), "incorrect": int(incorrect.sum()),
                   "uncertain": int(uncertain.sum()), "unable": int(unable.sum()),
                   "correct_large_r_ge_10": int(large.sum()), "correct_small_r_lt_10": int(small.sum())},
        "filters_at_matched_incorrect_acceptance": {
            "min_bidirectional_confidence": retention_at_wrong_acceptance(confidence, True, correct, large, small, incorrect),
            "cycle_error": retention_at_wrong_acceptance(cycle, False, correct, large, small, incorrect),
            "current_shape_residual": retention_at_wrong_acceptance(residual, False, correct, large, small, incorrect)},
        "frozen_C_rule": {"rule": "forward confidence>=0.5, reverse confidence>=0.5, cycle<=2 px",
                          **classified_stats(standard, correct, incorrect, large, small, cell_id)},
        "per_pair_csv": "filter_audit_per_pair.csv",
        "limits": ["small preselected scene batches", "pooled counts can be dominated by pairs with more evaluable candidates"]}
    (args.diagnosis/"pooled_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
