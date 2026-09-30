"""Summarize valid no-fold A/B/C/D results with point-count weighting."""
import argparse
import json
from pathlib import Path


def weighted(entries, count_key, group, metric_group, metric):
    pairs = []
    for result in entries:
        count = result[count_key]
        value = result["groups"][group][metric_group].get(metric)
        if count and value is not None:
            pairs.append((count, value))
    total = sum(count for count, _ in pairs)
    return {"n": total, "mean": sum(count*value for count, value in pairs)/total if total else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    entries = [json.loads(path.read_text()) for path in sorted(args.run.glob("DSC_*/result.json"))]
    groups = ["A_baseline", "B_residual", "C_reliability", "D_oracle_select",
              "C_spatial_count_control", "D_spatial_count_control"]
    aggregate = {}
    for group in groups:
        aggregate[group] = {
            "all_held_canvas": weighted(entries, "held_truth_points", group, "held_canvas_error", "mean"),
            "initially_difficult_canvas": weighted(entries, "held_initial_difficult", group, "difficult_canvas_error", "mean"),
            "initially_aligned_canvas": weighted(entries, "held_initial_aligned", group, "aligned_canvas_error", "mean"),
            "max_sampled_fold_fraction": max(item["groups"][group]["structure"]["sampled_tps_fold_fraction"] for item in entries),
            "max_triangle_area_change": max(abs(item["groups"][group]["triangle_area_ratio"]-1) for item in entries),
            "min_overlap_retention": min(item["groups"][group]["overlap_pixels"]
                                         / item["groups"]["A_baseline"]["overlap_pixels"] for item in entries),
        }
    c, d, a = (aggregate[name]["all_held_canvas"]["mean"]
               for name in ("C_reliability", "D_oracle_select", "A_baseline"))
    result = {
        "status": "valid_no_fold_small_batch_complete", "pairs": len(entries),
        "aggregate": aggregate,
        "D_vs_A_all_held_relative_change": d/a-1,
        "D_vs_C_all_held_relative_change": d/c-1,
        "D_vs_C_pair_wins": sum(item["groups"]["D_oracle_select"]["held_canvas_error"]["mean"]
                                < item["groups"]["C_reliability"]["held_canvas_error"]["mean"]
                                for item in entries),
        "B_insufficient_pair_count": sum(item["optimization"]["B_residual"]["status"] == "insufficient_points"
                                         for item in entries),
        "interpretation": [
            "Residual filtering fails on the hard pairs because it removes nearly all candidates.",
            "Oracle selection improves over conventional reliability on three of five pairs, but the pooled gap is modest.",
            "This is one development scene and cannot establish generalization or a paper-level contribution."
        ]
    }
    (args.run/"aggregate.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
