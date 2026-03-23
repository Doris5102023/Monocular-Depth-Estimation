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
from typing import Tuple, Dict

# Import your existing modules
from depth_model import ResNet50DepthModel
from scannet_dataset import ScanNetDepthDataset
from metrics import abs_rel_metric, solve_scale_shift
# Import foundation model wrappers (from your previous code)
from foundation_models import VGGTInferencer, DA3Inferencer

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
    """Load pre-trained baseline depth estimation model from checkpoint"""
    model = ResNet50DepthModel(
        min_depth=min_depth,
        max_depth=max_depth,
        pretrained_backbone=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    print(f"Successfully loaded baseline model from {checkpoint_path}")
    return model

def init_foundation_models(
    device: torch.device,
    da3_model_type: str = "DA3NESTED-GIANT-LARGE"
) -> Dict[str, torch.nn.Module]:
    """Initialize DA3 and VGGT foundation models"""
    models = {}
    # Initialize VGGT
    models["vggt"] = VGGTInferencer(device=device)
    print("Successfully initialized VGGT model")
    # Initialize DA3
    models["da3"] = DA3Inferencer(model_type=da3_model_type, device=device)
    print("Successfully initialized DA3 model")
    return models

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
        "frame_id": sample["frame_id"],
        "image_original": sample.get("image_original", None)  # For foundation model input
    }

def denormalize_image(image: torch.Tensor, 
                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert normalized tensor image to RGB numpy array for visualization"""
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    mean_np = np.array(mean).reshape(1, 1, 3)
    std_np = np.array(std).reshape(1, 1, 3)
    img_np = (img_np * std_np + mean_np).clip(0, 1)
    return img_np

def predict_depth_for_all_models(
    sample_data: dict,
    baseline_model: ResNet50DepthModel,
    foundation_models: Dict[str, torch.nn.Module],
    device: torch.device,
    image_size: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    """Predict depth for Baseline/DA3/VGGT and apply scale-shift alignment"""
    depth_predictions = {}
    depth_gt = sample_data["depth_gt"].unsqueeze(0).to(device)  # (1, 1, H, W)
    valid_mask = sample_data["valid_mask"].unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # 1. Baseline prediction (using normalized input)
    with torch.no_grad():
        baseline_input = sample_data["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
        baseline_pred = baseline_model(baseline_input)  # (1, 1, H, W)
        baseline_pred_aligned = solve_scale_shift(baseline_pred, depth_gt, valid_mask)
    depth_predictions["baseline"] = baseline_pred_aligned
    print(f"Baseline AbsRel: {abs_rel_metric(baseline_pred_aligned, depth_gt, valid_mask):.6f}")
    
    # 2. Foundation models prediction (DA3/VGGT - using denormalized input)
    # Prepare input for foundation models (denormalized + resized)
    img_denorm = denormalize_image(sample_data["image"])
    img_denorm_tensor = torch.from_numpy(img_denorm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    for model_name, model in foundation_models.items():
        with torch.no_grad():
            # Predict depth with foundation model
            pred = model.predict_depth(img_denorm_tensor)  # (1, 1, H, W)
            # Apply scale-shift alignment (match baseline process)
            pred_aligned = solve_scale_shift(pred, depth_gt, valid_mask)
        depth_predictions[model_name] = pred_aligned
        # Calculate and print AbsRel
        abs_rel = abs_rel_metric(pred_aligned, depth_gt, valid_mask)
        print(f"{model_name.upper()} AbsRel: {abs_rel:.6f}")
    
    return depth_predictions

def visualize_depth_comparison(
    sample_data: dict,
    depth_predictions: Dict[str, torch.Tensor],
    save_path: str,
    min_depth: float = 0.1,
    max_depth: float = 10.0
):
    """Create side-by-side visualization for Baseline/DA3/VGGT depth estimation results"""
    # Prepare base data
    image = denormalize_image(sample_data["image"])
    depth_gt = sample_data["depth_gt"].squeeze(0).cpu().numpy()
    valid_mask = sample_data["valid_mask"].squeeze(0).cpu().numpy()
    
    # Mask invalid regions
    depth_gt = np.where(valid_mask, depth_gt, np.nan)
    
    # Set common depth range (5th-95th percentile of GT)
    valid_gt = depth_gt[~np.isnan(depth_gt)]
    if len(valid_gt) == 0:
        print("Warning: No valid depth values in GT")
        vmin, vmax = min_depth, max_depth
    else:
        vmin = np.nanpercentile(depth_gt, 5)
        vmax = np.nanpercentile(depth_gt, 95)
    
    # Create figure (1 row: RGB + GT + Baseline + DA3 + VGGT + Error Maps)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(
        f"Depth Estimation Comparison | Scene: {sample_data['scene_name']} | Frame: {sample_data['frame_id']}",
        fontsize=16, y=0.98
    )
    
    # --------------------------
    # Row 1: RGB + GT + Model Predictions
    # --------------------------
    # 1. Input RGB
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input RGB Image", fontsize=12)
    axes[0, 0].axis("off")
    
    # 2. Ground Truth Depth
    im_gt = axes[0, 1].imshow(depth_gt, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Ground Truth Depth", fontsize=12)
    axes[0, 1].axis("off")
    plt.colorbar(im_gt, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Depth (m)")
    
    # 3. Model Predictions (Baseline/DA3/VGGT)
    model_names = ["baseline", "da3", "vggt"]
    model_titles = ["Baseline", "DA3 (Aligned)", "VGGT (Aligned)"]
    
    for idx, (model_name, title) in enumerate(zip(model_names, model_titles)):
        pred = depth_predictions[model_name].squeeze().cpu().numpy()
        pred = np.where(valid_mask, pred, np.nan)
        
        # Plot predicted depth
        ax = axes[0, 2 + idx]
        im = ax.imshow(pred, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")
        
        # Calculate AbsRel for title
        abs_rel = abs_rel_metric(
            depth_predictions[model_name],
            sample_data["depth_gt"].unsqueeze(0).to(next(iter(depth_predictions.values())).device),
            sample_data["valid_mask"].unsqueeze(0).to(next(iter(depth_predictions.values())).device)
        )
        ax.text(0.02, 0.98, f"AbsRel: {abs_rel:.4f}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --------------------------
    # Row 2: Error Maps
    # --------------------------
    # Empty first two columns (align with row 1)
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    
    # Plot error maps for each model
    for idx, model_name in enumerate(model_names):
        pred = depth_predictions[model_name].squeeze().cpu().numpy()
        pred = np.where(valid_mask, pred, np.nan)
        
        # Calculate per-pixel AbsRel error
        error = np.abs(pred - depth_gt) / (depth_gt + 1e-6)
        error = np.where(valid_mask, error, np.nan)
        
        # Set error range (95th percentile)
        valid_error = error[~np.isnan(error)]
        error_vmax = np.nanpercentile(valid_error, 95) if len(valid_error) > 0 else 0.5
        
        # Plot error map
        ax = axes[1, 2 + idx]
        im_error = ax.imshow(error, cmap="Reds", vmin=0, vmax=error_vmax)
        ax.set_title(f"{model_name.capitalize()} AbsRel Error", fontsize=12)
        ax.axis("off")
        plt.colorbar(im_error, ax=ax, fraction=0.046, pad=0.04, label="AbsRel")
    
    # Empty last column (symmetry)
    axes[1, 4].axis("off")
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison visualization saved to: {save_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Depth Map Comparison Visualization (Baseline/DA3/VGGT)")
    parser.add_argument("--scannet_root", type=str, required=True, 
                        help="Root directory of ScanNet dataset")
    parser.add_argument("--baseline_checkpoint", type=str, required=True, 
                        help="Path to trained baseline model checkpoint")
    parser.add_argument("--scene_name", type=str, default="scene0001_01", 
                        help="ScanNet scene name to visualize")
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--save_path", type=str, default="depth_comparison.png")
    parser.add_argument("--da3_model_type", type=str, default="DA3NESTED-GIANT-LARGE",
                        help="DA3 model type (from foundation_models.py)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load models
    baseline_model = load_trained_model(
        checkpoint_path=args.baseline_checkpoint,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        device=device
    )
    foundation_models = init_foundation_models(
        device=device,
        da3_model_type=args.da3_model_type
    )
    
    # Step 2: Get sample data
    sample_data = get_single_sample_data(
        scannet_root=args.scannet_root,
        scene_name=args.scene_name,
        image_size=(args.image_height, args.image_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )
    
    # Step 3: Predict depth for all models
    depth_predictions = predict_depth_for_all_models(
        sample_data=sample_data,
        baseline_model=baseline_model,
        foundation_models=foundation_models,
        device=device,
        image_size=(args.image_height, args.image_width)
    )
    
    # Step 4: Generate comparison visualization
    visualize_depth_comparison(
        sample_data=sample_data,
        depth_predictions=depth_predictions,
        save_path=args.save_path,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )

if __name__ == "__main__":
    main()