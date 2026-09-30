"""Re-evaluate a saved plain residual model without training or test leakage."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from model import PlainResidualGridNet
from train_plain import aggregate, evaluate, load_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--splits", nargs="+", default=["validation"])
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PlainResidualGridNet(checkpoint["base_channels"], checkpoint["max_displacement"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    _, records = load_records(args.cache)
    rows = []
    for name in args.splits:
        if name not in records:
            raise ValueError(f"unknown split: {name}")
        rows.extend(evaluate(model, records[name], device))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_pair.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {name: aggregate([row for row in rows if row["split"] == name])
               for name in args.splits}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
