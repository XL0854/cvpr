"""Train the lightweight alpha predictor from ternary-search teacher labels."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loss import alpha_distillation_loss
from network import AlphaPredictor, CoefNetwork, Network


class AlphaLabelDataset(Dataset):
    def __init__(self, data_path, labels_path, safe_gain=0.003):
        records = torch.load(labels_path, map_location="cpu")["labels"]
        root = Path(data_path)
        self.samples = []
        for record in records:
            path1, path2 = root / "input1" / record["name"], root / "input2" / record["name"]
            if path1.exists() and path2.exists():
                safe = float(record["best_ssim"] - record["fixed_ssim"] <= safe_gain)
                self.samples.append((path1, path2, float(record["alpha"]), safe))
        if not self.samples:
            raise RuntimeError("No label records match the supplied image directory.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path1, path2, alpha, safe = self.samples[index]
        image1, image2 = cv2.imread(str(path1)), cv2.imread(str(path2))
        image1 = cv2.resize(image1, (512, 512)).astype(np.float32) / 127.5 - 1.0
        image2 = cv2.resize(image2, (512, 512)).astype(np.float32) / 127.5 - 1.0
        return (torch.from_numpy(np.transpose(image1, (2, 0, 1))),
                torch.from_numpy(np.transpose(image2, (2, 0, 1))),
                torch.tensor([alpha], dtype=torch.float32),
                torch.tensor([safe], dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--wo_coef_path", required=True)
    parser.add_argument("--coef_path", required=True)
    parser.add_argument("--output", default="model_alpha/best_alpha_predictor.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--safe_gain", type=float, default=0.003)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = AlphaLabelDataset(args.data_path, args.labels, args.safe_gain)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    net, coef_net = Network().to(device), CoefNetwork().to(device)
    net.load_state_dict(torch.load(args.wo_coef_path, map_location=device)["model"])
    coef_net.load_state_dict(torch.load(args.coef_path, map_location=device)["model"])
    net.eval()
    coef_net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in coef_net.parameters():
        parameter.requires_grad_(False)

    predictor = AlphaPredictor().to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        predictor.train()
        running_loss = 0.0
        for image1, image2, teacher_alpha, safe_target in loader:
            image1, image2 = image1.to(device), image2.to(device)
            frozen_corr, active_corr = net.extract_correlations(image1, image2)
            pred_alpha, pred_confidence = predictor(frozen_corr, active_corr)
            loss, _, _ = alpha_distillation_loss(
                pred_alpha, teacher_alpha.to(device), pred_confidence, safe_target.to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * image1.shape[0]
        mean_loss = running_loss / len(dataset)
        print("epoch {:03d}: loss={:.6f}".format(epoch + 1, mean_loss))
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({"model": predictor.state_dict(), "epoch": epoch + 1, "loss": mean_loss}, output)


if __name__ == "__main__":
    main()
