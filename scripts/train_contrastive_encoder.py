"""
使用 NAS-Bench-201 的 CIFAR-10, CIFAR-100 訓練 Siamese Encoder

用法:
    # CIFAR-10
    python scripts/train_contrastive_encoder.py

    # 自訂參數
    python scripts/train_contrastive_encoder.py \
        --dataset cifar100 \
        --epochs 30 \
        --batch_size 64 \
        --margin 1.5 \
        --lr 1e-4 \
        --device cuda

    # 同時用 CIFAR-10 + CIFAR-100 (跨 dataset contrastive)
    python scripts/train_contrastive_encoder.py --dataset both
"""
from __future__ import annotations

import argparse
import os
import sys
import random
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from PIL import Image

from src.retrieval.contrastive_encoder import (
    ContrastiveLoss,
    ContrastivePairDataset,
    SiameseEncoder,
)


# ---------------------------------------------------------------------------
# Step 1: 資料準備 — 從 torchvision 下載 CIFAR
# ---------------------------------------------------------------------------

def download_and_organize_cifar(
    dataset_name: str = "cifar10",
    data_root: str = "data/NAS-Bench/cifar_images",
) -> dict[str, list[str]]:
    """
    下載 CIFAR 資料集，把圖片存為 .png 檔案，按 class 歸類。

    Returns:
        task_to_images: {"class_0": [path1, path2, ...], "class_1": [...], ...}
    """
    data_root = Path(data_root)

    print(f"\n下載 {dataset_name} ...")

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=str(data_root / "raw"), train=True, download=True
        )
        class_names = ds.classes  # ['airplane', 'automobile', ...]
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root=str(data_root / "raw"), train=True, download=True
        )
        class_names = ds.classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 組織為 task_to_images dict
    # 這裡 "task" = 類別名稱（同一類別 = positive pair, 不同類別 = negative pair）
    task_to_images: dict[str, list[str]] = {}

    img_dir = data_root / dataset_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 只需要每個 class 存一部分圖片（訓練用）
    max_per_class = 100  # 每個 class 最多取 100 張

    print(f"整理圖片到 {img_dir} ...")
    class_counts: dict[str, int] = {}

    for idx in range(len(ds)):
        img_pil, label = ds[idx]
        cls_name = class_names[label]

        if cls_name not in class_counts:
            class_counts[cls_name] = 0
        if class_counts[cls_name] >= max_per_class:
            continue
        class_counts[cls_name] += 1

        # Save image
        cls_dir = img_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        save_path = cls_dir / f"{class_counts[cls_name]:04d}.png"

        if not save_path.exists():
            img_pil.save(str(save_path))

        if cls_name not in task_to_images:
            task_to_images[cls_name] = []
        task_to_images[cls_name].append(str(save_path))

    print(f"共 {len(task_to_images)} 個類別, 每類最多 {max_per_class} 張")
    for cls, paths in list(task_to_images.items())[:5]:
        print(f"   {cls}: {len(paths)} 張")
    if len(task_to_images) > 5:
        print(f"   ... 還有 {len(task_to_images) - 5} 個類別")

    return task_to_images


# NOTE: Pair dataset logic is now provided by ContrastivePairDataset
# from src.retrieval.contrastive_encoder (imported above).


# ---------------------------------------------------------------------------
# Step 3: 訓練迴圈
# ---------------------------------------------------------------------------

def train(
    dataset_name: str = "cifar10",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    margin: float = 1.0,
    beta_pos: float = 1.0,
    beta_neg: float = 1.0,
    pairs_per_epoch: int = 4000,
    embed_dim: int = 128,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
):
    """完整訓練流程"""

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ─── Step 1: 下載 + 整理資料 ───
    if dataset_name == "both":
        t2i_10 = download_and_organize_cifar("cifar10")
        t2i_100 = download_and_organize_cifar("cifar100")
        # 合併，加 prefix 避免 class name 衝突
        task_to_images = {}
        for k, v in t2i_10.items():
            task_to_images[f"c10_{k}"] = v
        for k, v in t2i_100.items():
            task_to_images[f"c100_{k}"] = v
    else:
        task_to_images = download_and_organize_cifar(dataset_name)

    # ─── Step 2: 建 Dataset & DataLoader ───
    print(f"\n建立 Pair Dataset (每 epoch {pairs_per_epoch} pairs) ...")
    pair_dataset = ContrastivePairDataset(
        task_to_images=task_to_images,
        pairs_per_epoch=pairs_per_epoch,
    )
    loader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # ─── Step 3: 建立 Model & Loss ───
    print(f"\n建立 SiameseEncoder (ResNet-18 → {embed_dim}d) ...")
    encoder = SiameseEncoder(backbone="resnet18", embed_dim=embed_dim, pretrained=True)
    encoder = encoder.to(device)

    loss_fn = ContrastiveLoss(margin=margin, beta_pos=beta_pos, beta_neg=beta_neg)
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ─── Step 4: 訓練 ───
    print(f"\n開始訓練 ({epochs} epochs) ...\n")
    print(f"{'Epoch':>6}  {'Loss':>10}  {'LR':>12}  {'Time':>8}")
    print("-" * 45)

    history = []
    best_loss = float("inf")
    best_path = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # 每個 epoch 重新產生 pairs（增加多樣性）
        pair_dataset.reshuffle()

        for img1, img2, labels in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            # Forward: shared encoder
            v1 = encoder(img1)  # (B, 128)
            v2 = encoder(img2)  # (B, 128)

            # Contrastive loss
            loss = loss_fn(v1, v2, labels)

            # Backward + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        history.append(avg_loss)

        print(f"{epoch:>6d}  {avg_loss:>10.5f}  {current_lr:>12.6f}  {elapsed:>7.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_path = os.path.join(checkpoint_dir, "contrastive_encoder_best.pt")
            torch.save({
                "encoder_state": encoder.state_dict(),
                "backbone": encoder.backbone_name,
                "embed_dim": encoder.embed_dim,
                "epoch": epoch,
                "loss": avg_loss,
            }, best_path)

    # ─── Step 5: 儲存最終 checkpoint ───
    final_path = os.path.join(checkpoint_dir, "contrastive_encoder_final.pt")
    torch.save({
        "encoder_state": encoder.state_dict(),
        "backbone": encoder.backbone_name,
        "embed_dim": encoder.embed_dim,
        "epoch": epochs,
        "loss": history[-1],
        "history": history,
    }, final_path)

    print(f"\n訓練完成！")
    print(f"   Best loss: {best_loss:.5f}")
    print(f"   Checkpoint: {final_path}")
    if best_path:
        print(f"   Best model: {best_path}")
    else:
        print("   Best model: (no improvement recorded)")

    return encoder, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Contrastive Encoder on NB201 datasets")
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "both"],
                    help="Training dataset (default: cifar10)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="Contrastive loss margin (default: 1.0)")
    ap.add_argument("--beta_pos", type=float, default=1.0,
                    help="Weight for same-class loss")
    ap.add_argument("--beta_neg", type=float, default=1.0,
                    help="Weight for different-class loss")
    ap.add_argument("--pairs_per_epoch", type=int, default=4000)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    train(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        beta_pos=args.beta_pos,
        beta_neg=args.beta_neg,
        pairs_per_epoch=args.pairs_per_epoch,
        embed_dim=args.embed_dim,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
