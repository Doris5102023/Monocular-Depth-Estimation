#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List

# -----------------------------------------------------------------------------
# Helper Function: Standardize Depth Tensor Shape
# -----------------------------------------------------------------------------
def standardize_depth_shape(depth: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Convert any depth tensor shape to standard [B, 1, H, W] format.
    Handles:
    - 5D: [1, B, H, W, 1] → 4D [B, 1, H, W]
    - 4D: [B, H, W, 1] → 4D [B, 1, H, W]
    - 3D: [B, H, W] → 4D [B, 1, H, W]
    Args:
        depth: Input depth tensor (any shape)
        target_shape: (H, W) of original input
    Returns:
        Standardized depth tensor [B, 1, H, W]
    """
    # Step 1: Remove all singleton dimensions
    depth = depth.squeeze()
    
    # Step 2: Add missing dimensions to reach 3D [B, H, W]
    while depth.dim() < 3:
        depth = depth.unsqueeze(0)
    
    # Step 3: If 4D (e.g., [1, B, H, W]), squeeze batch singleton
    if depth.dim() == 4:
        depth = depth.squeeze(0)
    
    # Step 4: Ensure shape is [B, H, W], then add channel dim
    assert depth.dim() == 3, f"Expected 3D tensor after squeezing, got {depth.dim()}D"
    depth = depth.unsqueeze(1)  # [B, 1, H, W]
    
    # Step 5: Resize to target shape if needed
    if depth.shape[2:] != target_shape:
        depth = F.interpolate(
            depth, size=target_shape, mode="bilinear", align_corners=False
        )
    
    return depth

# -----------------------------------------------------------------------------
# VGGT Wrapper (Dimension-Agnostic)
# -----------------------------------------------------------------------------
class VGGTInferencer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.patch_size = 14
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        print("✅ VGGT model initialized (patch size = 14x14).")

    def adjust_resolution(self, h: int, w: int) -> tuple[int, int]:
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        return new_h, new_w

    @torch.no_grad()
    def predict_depth(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fixed: Handles 5D input from VGGT, returns [B, 1, H, W]
        """
        orig_h, orig_w = image_tensor.shape[2:]
        new_h, new_w = self.adjust_resolution(orig_h, orig_w)
        target_shape = (orig_h, orig_w)
        
        # Step 1: Preprocess image
        image = image_tensor * self.std + self.mean
        image_resized = F.interpolate(
            image, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        
        # Step 2: VGGT forward pass (may return 5D tensor [1, B, H, W, 1])
        outputs = self.model(image_resized)
        depth_resized = outputs["depth"]  # Could be 5D/4D/3D
        
        # Step 3: Standardize shape (critical fix for permute error)
        depth = standardize_depth_shape(depth_resized, target_shape)
        
        return depth  # Final shape: [B, 1, orig_h, orig_w]

# -----------------------------------------------------------------------------
# DA3 Wrapper (Official API + Shape Safety)
# -----------------------------------------------------------------------------
class DA3Inferencer:
    def __init__(self, model_type: str = "DA3NESTED-GIANT-LARGE", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(f"depth-anything/{model_type}")
        self.model = self.model.to(device=device)
        self.model.eval()
        print(f"✅ DA3 model initialized: {model_type} (official API)")

    @torch.no_grad()
    def predict_depth(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fixed: Safe shape handling for DA3
        """
        B, C, H, W = image_tensor.shape
        target_shape = (H, W)
        
        # Step 1: Convert tensor to PIL images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        image_np = (image_tensor * std + mean) * 255.0
        image_np = image_np.byte().cpu().numpy()
        
        pil_images = []
        for i in range(B):
            img_pil = Image.fromarray(image_np[i].transpose(1, 2, 0))
            pil_images.append(img_pil)
        
        # Step 2: DA3 inference (returns [B, H, W])
        prediction = self.model.inference(pil_images)
        depth_np = prediction.depth
        
        # Step 3: Convert to tensor and standardize shape
        depth_tensor = torch.from_numpy(depth_np).to(self.device)
        depth = standardize_depth_shape(depth_tensor, target_shape)
        
        return depth  # Final shape: [B, 1, H, W]