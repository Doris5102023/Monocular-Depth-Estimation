#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Union, Tuple

import torch
from torch.utils.data import DataLoader

# Import your modules
from scannet_dataset import ScanNetDepthDataset
from metrics import abs_rel_metric, solve_scale_shift
from foundation_models import VGGTInferencer, DA3Inferencer
from depth_model import ResNet50DepthModel

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline / VGGT / DA3 on ScanNet Test Set (Task 4)"
    )
    parser.add_argument("--scannet_root", type=str, required=True,
                        help="Root directory of ScanNet dataset")
    parser.add_argument("--split_file", type=str, required=True,
                        help="Path to test split file (e.g., scannetv2_test.txt)")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "vggt", "da3"],
                        help="Model to evaluate")
    parser.add_argument("--baseline_checkpoint", type=str,
                        help="Path to baseline model checkpoint (required for 'baseline' model)")
    parser.add_argument("--da3_model_type", type=str, default="DA3NESTED-GIANT-LARGE",
                        help="DA3 pre-trained model name")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--image_height", type=int, default=240,
                        help="Input image height")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Input image width")
    parser.add_argument("--min_depth", type=float, default=0.1,
                        help="Minimum valid depth (meters)")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Maximum valid depth (meters)")
    parser.add_argument("--save_json", type=str, default="comparison_results.json",
                        help="Path to save evaluation results (JSON)")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Build Test Dataset
# -----------------------------------------------------------------------------
def build_test_dataset(args: argparse.Namespace) -> ScanNetDepthDataset:
    scenes = []
    with open(args.split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                scenes.append(line)
    print(f"Loaded {len(scenes)} test scenes from {args.split_file}.")
    
    return ScanNetDepthDataset(
        scannet_root=args.scannet_root,
        scenes=scenes,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=False,
        max_samples=None
    )

# -----------------------------------------------------------------------------
# Unified Evaluation Function (Fixed: Returns Both Metrics and Sample Count)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: Union[ResNet50DepthModel, VGGTInferencer, DA3Inferencer],
    model_type: str,
    dataset: ScanNetDepthDataset,
    args: argparse.Namespace
) -> Tuple[float, int]:  # Return (avg_abs_rel, sample_count)
    """
    Fixed: Returns sample_count to fix scoping error
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    total_abs_rel = 0.0
    sample_count = 0

    for batch_idx, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        # Predict depth (all models return [B, 1, H, W] now)
        if model_type == "baseline":
            depth_pred = model(image)
        else:
            depth_pred = model.predict_depth(image)

        # Scale-shift alignment (no shape errors)
        depth_pred_aligned = solve_scale_shift(depth_pred, depth_gt, valid_mask)

        # Calculate AbsRel
        batch_abs_rel = abs_rel_metric(depth_pred_aligned, depth_gt, valid_mask)
        if not math.isnan(batch_abs_rel):
            total_abs_rel += batch_abs_rel * image.shape[0]
            sample_count += image.shape[0]

        if batch_idx % 20 == 0:
            print(f"[{model_type}] Batch {batch_idx}/{len(loader)} | AbsRel: {batch_abs_rel:.4f}")

    avg_abs_rel = total_abs_rel / sample_count if sample_count > 0 else float("inf")
    return avg_abs_rel, sample_count  # Return sample_count to main function

# -----------------------------------------------------------------------------
# Main Execution (Fixed: Properly Receives sample_count)
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating {args.model} on ScanNet test set...")

    # Load dataset
    dataset = build_test_dataset(args)

    # Initialize model
    if args.model == "baseline":
        if not args.baseline_checkpoint:
            raise ValueError("Must provide --baseline_checkpoint for baseline model!")
        model = ResNet50DepthModel(min_depth=args.min_depth, max_depth=args.max_depth).to(device)
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        print("Baseline model initialized.")
    elif args.model == "vggt":
        model = VGGTInferencer(device=device)
    elif args.model == "da3":
        model = DA3Inferencer(model_type=args.da3_model_type, device=device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Evaluate (Fixed: Receive both avg_abs_rel and sample_count)
    avg_abs_rel, sample_count = evaluate_model(model, model_type=args.model, dataset=dataset, args=args)

    # Save results (sample_count is now defined in main)
    results = {
        "model": args.model,
        "avg_abs_rel": float(avg_abs_rel),
        "input_resolution": f"{args.image_height}x{args.image_width}",
        "depth_range": [args.min_depth, args.max_depth],
        "total_valid_samples": sample_count
    }
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation Complete ===")
    print(f"Model: {args.model.upper()}")
    print(f"Average AbsRel on Test Set: {avg_abs_rel:.6f}")
    print(f"Total Valid Samples: {sample_count}")
    print(f"Results saved to: {os.path.abspath(args.save_json)}")

if __name__ == "__main__":
    main()