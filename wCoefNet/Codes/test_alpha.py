"""Evaluate one-shot alpha prediction with an optional ternary-search fallback."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from network import AlphaPredictor, CoefNetwork, Network
from test import ternary_search, test_once


def load_pair(path1, path2, device):
    image1, image2 = cv2.imread(str(path1)), cv2.imread(str(path2))
    if image1 is None or image2 is None:
        raise ValueError("Unable to read {} or {}".format(path1, path2))
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    image1 = torch.from_numpy(np.transpose(image1.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))).unsqueeze(0)
    image2 = torch.from_numpy(np.transpose(image2.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))).unsqueeze(0)
    return image1.to(device), image2.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--wo_coef_path", required=True)
    parser.add_argument("--coef_path", required=True)
    parser.add_argument("--alpha_path", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--fallback_iter", type=int, default=20)
    parser.add_argument("--max_out_height", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_fallback", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    net, coef_net = Network().to(device), CoefNetwork().to(device)
    net.load_state_dict(torch.load(args.wo_coef_path, map_location=device)["model"])
    coef_net.load_state_dict(torch.load(args.coef_path, map_location=device)["model"])
    predictor = AlphaPredictor().to(device)
    predictor.load_state_dict(torch.load(args.alpha_path, map_location=device)["model"])
    net.eval(); coef_net.eval(); predictor.eval()

    root = Path(args.data_path)
    input1 = {p.name: p for p in (root / "input1").glob("*") if p.is_file()}
    input2 = {p.name: p for p in (root / "input2").glob("*") if p.is_file()}
    names = sorted(input1.keys() & input2.keys())
    if args.limit:
        names = names[:args.limit]
    if not names:
        raise RuntimeError("No same-name image pairs found.")

    scores, elapsed, fallback_count = [], 0.0, 0
    for index, name in enumerate(names, start=1):
        image1, image2 = load_pair(input1[name], input2[name], device)
        started = time.perf_counter()
        with torch.no_grad():
            frozen_corr, active_corr = net.extract_correlations(image1, image2)
            alpha, confidence = predictor(frozen_corr, active_corr)
        alpha_value, confidence_value = alpha.item(), confidence.item()
        if not args.no_fallback and confidence_value < args.confidence_threshold:
            fallback_count += 1
            alpha_value, score, _, _ = ternary_search(
                net, coef_net, image1, image2, low=-1.0, high=2.0,
                max_iter=args.fallback_iter, max_out_height=args.max_out_height,
            )
        else:
            score, _, _ = test_once(net, coef_net, image1, image2, alpha=alpha_value,
                                    max_out_height=args.max_out_height)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        elapsed += duration
        scores.append(score)
        print("[{}/{}] {} alpha={:.4f} confidence={:.4f} fallback={} mSSIM={:.6f} time={:.3f}s".format(
            index, len(names), name, alpha_value, confidence_value,
            confidence_value < args.confidence_threshold and not args.no_fallback, score, duration))
    print("mean_mSSIM={:.6f}, mean_time={:.3f}s, fallback_rate={:.2%}".format(
        float(np.mean(scores)), elapsed / len(scores), fallback_count / len(scores)))


if __name__ == "__main__":
    main()
