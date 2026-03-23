#!/usr/bin/env python
from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


from depth_model import ResNet50DepthModel
from scannet_dataset import ScanNetDepthDataset, build_train_val_scenes

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ScanNet monocular depth training.")
    parser.add_argument("--scannet_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_split_file", type=str, required=True, help="scannetv2_train.txt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10.0)

    parser.add_argument("--max_train_samples", type=int, default=160000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, ...]:
    image = batch["image"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
    return image, depth, valid_mask


def _silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    variance_focus: float = 0.85,
) -> torch.Tensor:
    losses = []
    for b in range(pred.shape[0]):
        mask = valid_mask[b, 0]
        if mask.sum().item() < 16:
            continue
        log_diff = torch.log(torch.clamp(pred[b, 0][mask], min=1e-3)) - torch.log(
            torch.clamp(target[b, 0][mask], min=1e-3)
        )
        mean = log_diff.mean()
        sq = (log_diff * log_diff).mean()
        silog = torch.sqrt(torch.clamp(sq - variance_focus * mean * mean, min=1e-6))
        losses.append(silog)

    if not losses:
        return pred.sum() * 0.0

    return torch.stack(losses).mean()



def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_abs_rel: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_abs_rel": best_abs_rel,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_height, args.image_width)

    train_scenes = build_train_val_scenes(
        scannet_root=args.scannet_root,
        split_file=args.train_split_file,
    )

    print(f"Train scenes: {len(train_scenes)}")

    train_gt = ScanNetDepthDataset(
        scannet_root=args.scannet_root,
        scenes=train_scenes,
        image_size=image_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        augment=True,
        max_samples=args.max_train_samples,
    )

    train_dataset = train_gt
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = ResNet50DepthModel(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)

    start_epoch = 0
    best_abs_rel = float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_abs_rel = float(ckpt.get("best_abs_rel", best_abs_rel))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        tic = time.time()

        for step, batch in enumerate(train_loader):
            image, depth, valid_mask = _to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(image)
            loss = _silog_loss(pred, depth, valid_mask)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            if step % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()
        mean_train_loss = epoch_loss / max(1, len(train_loader))
        elapsed = time.time() - tic
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={mean_train_loss:.4f} | "
            f"time={elapsed:.1f}s"
        )

        _save_checkpoint(
            output_dir / f"{epoch}.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_abs_rel=best_abs_rel,
            args=args,
        )



if __name__ == "__main__":
    main()
