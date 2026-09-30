"""Generate alpha teacher labels with the original RopStitch ternary search.

Run this once after placing the two supplied checkpoints in the project.  The
result is a small torch file used by train_alpha.py; input images are never
copied.  Start with --limit 100 --max_iter 8 as a smoke test, then use 20
iterations for the final labels.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from network import CoefNetwork, Network
from test import ternary_search, test_once


def load_pair(path1, path2, device):
    image1 = cv2.imread(str(path1))
    image2 = cv2.imread(str(path2))
    if image1 is None or image2 is None:
        raise ValueError("Unable to read {} or {}".format(path1, path2))
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    tensor1 = torch.from_numpy(np.transpose(image1.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))).unsqueeze(0)
    tensor2 = torch.from_numpy(np.transpose(image2.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))).unsqueeze(0)
    return tensor1.to(device), tensor2.to(device)


def load_models(args, device):
    net, coef_net = Network().to(device), CoefNetwork().to(device)
    net.load_state_dict(torch.load(args.wo_coef_path, map_location=device)["model"])
    coef_net.load_state_dict(torch.load(args.coef_path, map_location=device)["model"])
    net.eval()
    coef_net.eval()
    return net, coef_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Directory containing input1/ and input2/.")
    parser.add_argument("--wo_coef_path", required=True)
    parser.add_argument("--coef_path", required=True)
    parser.add_argument("--output", default="alpha_labels.pt")
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--max_out_height", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every matched pair.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.data_path)
    input1 = {path.name: path for path in (root / "input1").glob("*") if path.is_file()}
    input2 = {path.name: path for path in (root / "input2").glob("*") if path.is_file()}
    names = sorted(input1.keys() & input2.keys())
    if not names:
        raise RuntimeError("No same-name image pairs found in input1/ and input2/.")
    if args.limit:
        names = names[:args.limit]

    device = torch.device(args.device)
    net, coef_net = load_models(args, device)
    labels = []
    for index, name in enumerate(names, start=1):
        tensor1, tensor2 = load_pair(input1[name], input2[name], device)
        fixed_ssim, _, _ = test_once(net, coef_net, tensor1, tensor2, alpha=0.5, max_out_height=args.max_out_height)
        alpha, best_ssim, best_psnr, _ = ternary_search(
            net, coef_net, tensor1, tensor2, low=-1.0, high=2.0,
            max_iter=args.max_iter, max_out_height=args.max_out_height,
        )
        labels.append({
            "name": name,
            "alpha": float(alpha),
            "best_ssim": float(best_ssim),
            "fixed_ssim": float(fixed_ssim),
            "best_psnr": float(best_psnr),
        })
        print("[{}/{}] {} alpha={:.4f}, gain={:.6f}".format(index, len(names), name, alpha, best_ssim - fixed_ssim))

    torch.save({"labels": labels, "max_iter": args.max_iter}, args.output)
    print("Saved {} labels to {}".format(len(labels), args.output))


if __name__ == "__main__":
    main()
