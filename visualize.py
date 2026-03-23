#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Import your existing modules
from depth_model import ResNet50DepthModel
from scannet_dataset import ScanNetDepthDataset
from metrics import abs_rel_metric, solve_scale_shift

# Set plot configuration
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

def load_trained_model(
    checkpoint_path: str,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    device: torch.device = torch.device("cuda")
) -> ResNet50DepthModel:
    """Load pre-trained depth estimation model from checkpoint"""
    model = ResNet50DepthModel(
        min_depth=min_depth,
        max_depth=max_depth,
        pretrained_backbone=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    print(f"Successfully loaded model from {checkpoint_path}")
    return model

def get_single_sample_data(
    scannet_root: str,
    scene_name: str,
    image_size: Tuple[int, int] = (240, 320),
    min_depth: float = 0.1,
    max_depth: float = 10.0
) -> dict:
    """Get single sample data (image, depth GT, valid mask) from ScanNet dataset"""
    dataset = ScanNetDepthDataset(
        scannet_root=scannet_root,
        scenes=[scene_name],
        image_size=image_size,
        min_depth=min_depth,
        max_depth=max_depth,
        augment=False,
        max_samples=1  # Only get 1 sample
    )
    
    sample = dataset[0]
    return {
        "image": sample["image"],          # (3, H, W) normalized tensor
        "depth_gt": sample["depth"],       # (1, H, W) tensor
        "valid_mask": sample["valid_mask"],# (1, H, W) tensor
        "scene_name": sample["scene"],
        "frame_id": sample["frame_id"]
    }

def denormalize_image(image: torch.Tensor, 
                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert normalized tensor image to RGB numpy array for visualization"""
    # Convert (3, H, W) tensor → (H, W, 3) numpy array (matplotlib compatible)
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    # Denormalize
    mean_np = np.array(mean).reshape(1, 1, 3)
    std_np = np.array(std).reshape(1, 1, 3)
    img_np = (img_np * std_np + mean_np).clip(0, 1)
    return img_np

def visualize_depth_comparison(
    sample_data: dict,
    depth_pred_aligned: torch.Tensor,
    abs_rel_score: float,
    save_path: str
):
    """Create qualitative visualization of depth estimation results"""
    # --------------------------
    # KEY FIX 1: Remove singleton dimensions (convert to 2D arrays)
    # --------------------------
    # Input RGB: (3, H, W) → (H, W, 3) (RGB image for matplotlib)
    image = denormalize_image(sample_data["image"])
    
    # Depth GT: (1, H, W) → (H, W) (2D grayscale for matplotlib)
    depth_gt = sample_data["depth_gt"].squeeze(0).cpu().numpy()  # Remove channel dim
    
    # Predicted depth: (1, 1, H, W) → (H, W) (remove batch + channel dims)
    depth_pred = depth_pred_aligned.squeeze().cpu().numpy()      # Squeeze ALL singleton dims
    
    # Valid mask: (1, H, W) → (H, W) (2D boolean mask)
    valid_mask = sample_data["valid_mask"].squeeze(0).cpu().numpy()

    # --------------------------
    # KEY FIX 2: Safe masking of invalid regions (2D arrays only)
    # --------------------------
    # Replace invalid depth values with NaN (matplotlib ignores NaNs)
    depth_gt = np.where(valid_mask, depth_gt, np.nan)
    depth_pred = np.where(valid_mask, depth_pred, np.nan)
    
    # Calculate per-pixel AbsRel error (avoid division by zero)
    error = np.abs(depth_pred - depth_gt) / (depth_gt + 1e-6)  # Add epsilon to prevent division by zero
    error = np.where(valid_mask, error, np.nan)

    # --------------------------
    # Handle edge cases (all NaN values)
    # --------------------------
    # Set common depth range for consistent color mapping
    valid_gt = depth_gt[~np.isnan(depth_gt)]
    if len(valid_gt) == 0:
        print("Warning: No valid depth values in GT")
        vmin, vmax = 0.1, 10.0
    else:
        vmin = np.nanpercentile(depth_gt, 5)
        vmax = np.nanpercentile(depth_gt, 95)
    
    # Set error range (clip outliers to 95th percentile)
    valid_error = error[~np.isnan(error)]
    error_vmax = np.nanpercentile(valid_error, 95) if len(valid_error) > 0 else 0.5

    # --------------------------
    # Create visualization (all arrays now have valid shapes)
    # --------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Depth Estimation Results | Scene: {sample_data['scene_name']} | Frame: {sample_data['frame_id']} | AbsRel: {abs_rel_score:.4f}",
        fontsize=14, y=1.02
    )

    # 1. Input RGB Image (shape: (H, W, 3) → valid for imshow)
    axes[0].imshow(image)
    axes[0].set_title("Input RGB Image", fontsize=12)
    axes[0].axis("off")

    # 2. Ground Truth Depth (shape: (H, W) → valid for imshow)
    im1 = axes[1].imshow(depth_gt, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth Depth", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (m)")

    # 3. Predicted Depth (Aligned) (shape: (H, W) → valid for imshow)
    im2 = axes[2].imshow(depth_pred, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[2].set_title("Predicted Depth (Scale-Shift Aligned)", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (m)")

    # 4. AbsRel Error Map (shape: (H, W) → valid for imshow)
    im3 = axes[3].imshow(error, cmap="Reds", vmin=0, vmax=error_vmax)
    axes[3].set_title("Per-Pixel AbsRel Error", fontsize=12)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="AbsRel")

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Qualitative visualization saved to: {save_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Depth Map Qualitative Visualization")
    parser.add_argument("--scannet_root", type=str, required=True, 
                        help="Root directory of ScanNet dataset")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to trained model checkpoint")
    parser.add_argument("--scene_name", type=str, default="scene0001_00", 
                        help="ScanNet scene name to visualize")
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--save_path", type=str, default="depth_visualization.png")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load trained model
    model = load_trained_model(
        checkpoint_path=args.checkpoint,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        device=device
    )
    
    # Step 2: Get sample data (image + GT depth)
    sample_data = get_single_sample_data(
        scannet_root=args.scannet_root,
        scene_name=args.scene_name,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )
    
    # Step 3: Predict depth map (add batch dimension for model input)
    with torch.no_grad():
        image_input = sample_data["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
        depth_pred = model(image_input)  # Output shape: (1, 1, H, W)
    
    # Step 4: Scale-shift alignment (match dimensions for alignment)
    depth_gt = sample_data["depth_gt"].unsqueeze(0).to(device)  # (1, 1, H, W)
    valid_mask = sample_data["valid_mask"].unsqueeze(0).to(device)  # (1, 1, H, W)
    depth_pred_aligned = solve_scale_shift(depth_pred, depth_gt, valid_mask)
    
    # Step 5: Calculate AbsRel score
    abs_rel_score = abs_rel_metric(depth_pred_aligned, depth_gt, valid_mask)
    print(f"Calculated AbsRel score: {abs_rel_score:.6f}")
    
    # Step 6: Generate qualitative visualization
    visualize_depth_comparison(
        sample_data=sample_data,
        depth_pred_aligned=depth_pred_aligned,
        abs_rel_score=abs_rel_score,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()