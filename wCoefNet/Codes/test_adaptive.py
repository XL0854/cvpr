"""Evaluate conservative few-evaluation alpha search without changing test.py."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from adaptive_alpha_search import adaptive_alpha_search
from network import CoefNetwork, Network
from test import ternary_search, test_once


def load_pair(path1, path2, device):
    images = []
    for path in (path1, path2):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Unable to read {}".format(path))
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 127.5 - 1.0
        images.append(torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", required=True)
    parser.add_argument("--input2", required=True)
    parser.add_argument("--woCoefNet_path", required=True)
    parser.add_argument("--coef_path", required=True)
    parser.add_argument("--output", default="adaptive_alpha_results.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    net, coef = Network().to(device), CoefNetwork().to(device)
    net.load_state_dict(torch.load(args.woCoefNet_path, map_location=device)["model"], strict=True)
    coef.load_state_dict(torch.load(args.coef_path, map_location=device)["model"], strict=True)
    net.eval(); coef.eval()
    files1 = {p.name: p for p in Path(args.input1).glob("*") if p.is_file()}
    files2 = {p.name: p for p in Path(args.input2).glob("*") if p.is_file()}
    names = sorted(files1.keys() & files2.keys())
    if args.limit:
        names = names[:args.limit]
    rows = []
    for index, name in enumerate(names, 1):
        input1, input2 = load_pair(files1[name], files2[name], device)
        alpha, ass, aps, _, astats = adaptive_alpha_search(net, coef, input1, input2)
        started = time.perf_counter()
        falpha, fss, fps, _ = ternary_search(net, coef, input1, input2, low=-1.0, high=2.0, max_iter=20)
        if device.type == "cuda": torch.cuda.synchronize(device)
        full_time = time.perf_counter() - started
        rows.append(dict(name=name, adaptive_alpha=float(alpha), full_alpha=float(falpha),
                         adaptive_ssim=float(ass), full_ssim=float(fss),
                         adaptive_psnr=float(aps), full_psnr=float(fps),
                         adaptive=astats, full_calls=40, full_seconds=float(full_time)))
        print("[{}/{}] {} calls={} fallback={} ΔSSIM={:+.6f}".format(
            index, len(names), name, astats["calls"], astats["fallback"], ass - fss), flush=True)
    if not rows:
        raise RuntimeError("No paired files found")
    result = dict(n=len(rows), mean_adaptive_calls=float(np.mean([r["adaptive"]["calls"] for r in rows])),
                  fallback_rate=float(np.mean([r["adaptive"]["fallback"] for r in rows])),
                  mean_adaptive_seconds=float(np.mean([r["adaptive"]["elapsed_seconds"] for r in rows])),
                  mean_full_seconds=float(np.mean([r["full_seconds"] for r in rows])),
                  mean_adaptive_ssim=float(np.mean([r["adaptive_ssim"] for r in rows])),
                  mean_full_ssim=float(np.mean([r["full_ssim"] for r in rows])),
                  mean_adaptive_psnr=float(np.mean([r["adaptive_psnr"] for r in rows])),
                  mean_full_psnr=float(np.mean([r["full_psnr"] for r in rows])), rows=rows)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
