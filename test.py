#!/usr/bin/env python
from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from depth_model import ResNet50DepthModel
from metrics import abs_rel_metric, solve_scale_shift
from scannet_dataset import ScanNetDepthDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ScanNet depth evaluation (aligned AbsRel only).")
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True, help="Scenes to evaluate. scannetv2_val.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, required=True, help="Required for baseline mode.")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_json", type=str, default=None)
    return parser.parse_args()


def _read_scene_file(path: str) -> List[str]:
    scenes: List[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line)
    return scenes


def _build_eval_dataset(args: argparse.Namespace) -> ScanNetDepthDataset:
    scenes = _read_scene_file(args.split_file)
    print(f"Evaluation scenes: {len(scenes)}")
    return ScanNetDepthDataset(
        scannet_root=args.scannet_root,
        scenes=scenes,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
        max_samples=args.max_samples,
    )


@torch.no_grad()
def _evaluate_baseline(
    args: argparse.Namespace,
    dataset: ScanNetDepthDataset,
    device: torch.device,
) -> float:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when --mode baseline")

    model = ResNet50DepthModel(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    total_abs_rel = 0.0
    count = 0

    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        pred = model(image)
        pred = solve_scale_shift(pred, depth, valid_mask)

        abs_rel = abs_rel_metric(pred, depth, valid_mask)
        if math.isnan(abs_rel):
            continue
        total_abs_rel += abs_rel
        count += 1

        if i % 20 == 0:
            print(f"[baseline] step {i}/{len(loader)} abs_rel={abs_rel:.4f}")

    if count == 0:
        return float("inf")
    return total_abs_rel / count


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _build_eval_dataset(args)
    abs_rel = _evaluate_baseline(args, dataset, device)


    print("\n=== Evaluation Results ===")
    print(f"{'abs_rel':>8s}: {abs_rel:.6f}")

    if args.save_json is not None:
        metrics = {"abs_rel": float(abs_rel)}
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nSaved metrics to: {save_path}")


if __name__ == "__main__":
    main()

