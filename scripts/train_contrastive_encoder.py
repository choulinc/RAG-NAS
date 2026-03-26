"""
Production Contrastive Encoder Training for RAG-NAS.

Supports multiple datasets from torchvision, custom ImageFolder directories,
and Kaggle CSV-label format conversion.

Usage:
    # Single dataset
    python scripts/train_contrastive_encoder.py --dataset cifar10

    # All built-in datasets
    python scripts/train_contrastive_encoder.py --dataset all

    # With custom ImageFolder directories
    python scripts/train_contrastive_encoder.py --dataset all --extra_dirs /path/to/dir1 /path/to/dir2

    # With Kaggle CSV-label format
    python scripts/train_contrastive_encoder.py --dataset cifar10 \
        --csv_label /path/to/labels.csv --csv_image_dir /path/to/images
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
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
    CombinedLoss,
    ContrastiveLoss,
    ContrastivePairDataset,
    ImageDecoder,
    SiameseEncoder,
)

# All built-in dataset names (order used by --dataset all)
BUILTIN_DATASETS = [
    "cifar10", "cifar100", "stl10", "svhn",
    "fashionmnist", "flowers102", "food101",
]


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def _save_images_from_dataset(
    ds,
    class_names: list[str],
    img_dir: Path,
    dataset_label: str,
) -> dict[str, list[str]]:
    """Save images from a torchvision dataset into class-organised folders.

    Returns task_to_images dict with prefixed class keys.
    """
    task_to_images: dict[str, list[str]] = {}
    class_counts: dict[str, int] = {}

    print(f"  Organising {len(ds)} images into {img_dir} ...")

    for idx in range(len(ds)):
        img_pil, label = ds[idx]
        cls_name = class_names[label]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        cls_dir = img_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        save_path = cls_dir / f"{class_counts[cls_name]:06d}.png"

        if not save_path.exists():
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            img_pil.save(str(save_path))

        key = f"{dataset_label}_{cls_name}"
        task_to_images.setdefault(key, []).append(str(save_path))

    return task_to_images


def download_dataset(
    dataset_name: str,
    data_root: str = "data/NAS-Bench/cifar_images",
) -> dict[str, list[str]]:
    """Download and organise a single torchvision dataset.

    Returns task_to_images: {"<prefix>_<class>": [path, ...], ...}
    """
    data_root = Path(data_root)
    img_dir = data_root / dataset_name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = str(data_root / "raw")

    print(f"\n[{dataset_name}] Downloading ...")

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(root=raw_dir, train=True, download=True)
        return _save_images_from_dataset(ds, ds.classes, img_dir, "c10")

    if dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(root=raw_dir, train=True, download=True)
        return _save_images_from_dataset(ds, ds.classes, img_dir, "c100")

    if dataset_name == "stl10":
        ds = torchvision.datasets.STL10(root=raw_dir, split="train", download=True)
        class_names = [
            "airplane", "bird", "car", "cat", "deer",
            "dog", "horse", "monkey", "ship", "truck",
        ]
        return _save_images_from_dataset(ds, class_names, img_dir, "stl10")

    if dataset_name == "svhn":
        ds = torchvision.datasets.SVHN(root=raw_dir, split="train", download=True)
        class_names = [str(i) for i in range(10)]
        # SVHN __getitem__ returns (PIL, label) only with transform; wrap manually
        task_to_images: dict[str, list[str]] = {}
        class_counts: dict[str, int] = {}
        print(f"  Organising {len(ds)} images into {img_dir} ...")
        for idx in range(len(ds)):
            img_arr, label = ds[idx]
            cls_name = class_names[label]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            cls_dir = img_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            save_path = cls_dir / f"{class_counts[cls_name]:06d}.png"

            if not save_path.exists():
                if isinstance(img_arr, Image.Image):
                    img_pil = img_arr
                else:
                    img_pil = Image.fromarray(np.transpose(img_arr, (1, 2, 0))) if hasattr(img_arr, 'shape') else img_arr
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")
                img_pil.save(str(save_path))

            key = f"svhn_{cls_name}"
            task_to_images.setdefault(key, []).append(str(save_path))
        return task_to_images

    if dataset_name == "fashionmnist":
        ds = torchvision.datasets.FashionMNIST(root=raw_dir, train=True, download=True)
        return _save_images_from_dataset(ds, ds.classes, img_dir, "fmnist")

    if dataset_name == "flowers102":
        ds = torchvision.datasets.Flowers102(root=raw_dir, split="train", download=True)
        # Flowers102 labels are 0-101; use numeric class names
        class_names = [f"flower_{i:03d}" for i in range(102)]
        task_to_images = {}
        class_counts: dict[str, int] = {}
        print(f"  Organising {len(ds)} images into {img_dir} ...")
        for idx in range(len(ds)):
            img_pil, label = ds[idx]
            cls_name = class_names[label]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            cls_dir = img_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            save_path = cls_dir / f"{class_counts[cls_name]:06d}.png"

            if not save_path.exists():
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")
                img_pil.save(str(save_path))

            key = f"fl102_{cls_name}"
            task_to_images.setdefault(key, []).append(str(save_path))
        return task_to_images

    if dataset_name == "food101":
        ds = torchvision.datasets.Food101(root=raw_dir, split="train", download=True)
        # Food101 provides (PIL, label); class names via ds.classes
        class_names = ds.classes
        task_to_images = {}
        class_counts: dict[str, int] = {}
        print(f"  Organising {len(ds)} images into {img_dir} ...")
        for idx in range(len(ds)):
            img_pil, label = ds[idx]
            cls_name = class_names[label]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            cls_dir = img_dir / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            save_path = cls_dir / f"{class_counts[cls_name]:06d}.png"

            if not save_path.exists():
                if img_pil.mode != "RGB":
                    img_pil = img_pil.convert("RGB")
                img_pil.save(str(save_path))

            key = f"food_{cls_name}"
            task_to_images.setdefault(key, []).append(str(save_path))
        return task_to_images

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_imagefolder(
    folder: str,
    prefix: str | None = None,
) -> dict[str, list[str]]:
    """Load an ImageFolder-style directory (class_name/images...)."""
    folder = Path(folder)
    if prefix is None:
        prefix = folder.name
    task_to_images: dict[str, list[str]] = {}
    for cls_dir in sorted(folder.iterdir()):
        if not cls_dir.is_dir():
            continue
        key = f"{prefix}_{cls_dir.name}"
        imgs = sorted(
            str(p) for p in cls_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if imgs:
            task_to_images[key] = imgs
    return task_to_images


def convert_csv_label_to_imagefolder(
    csv_path: str,
    image_dir: str,
    output_dir: str | None = None,
) -> str:
    """Convert Kaggle-style CSV (filename, label) into ImageFolder layout.

    Returns the path to the output ImageFolder directory.
    """
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    if output_dir is None:
        output_dir = str(image_dir.parent / f"{image_dir.name}_organised")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[csv_label] Converting {csv_path} -> {output_dir}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        count = 0
        for row in reader:
            if len(row) < 2:
                continue
            filename, label = row[0].strip(), row[1].strip()
            src = image_dir / filename
            # Try common extensions if not found
            if not src.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = image_dir / (filename + ext)
                    if candidate.exists():
                        src = candidate
                        break
            if not src.exists():
                continue
            dst_dir = output_dir / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            count += 1

    print(f"  Organised {count} images into {output_dir}")
    return str(output_dir)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_name: str = "cifar10",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    margin: float = 1.0,
    beta_pos: float = 1.0,
    beta_neg: float = 1.0,
    pairs_per_epoch: int = 20000,
    embed_dim: int = 128,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    num_workers: int = 4,
    warmup_epochs: int = 5,
    extra_dirs: list[str] | None = None,
    csv_label: str | None = None,
    csv_image_dir: str | None = None,
    clean: bool = False,
    use_decoder: bool = False,
    recon_weight: float = 0.5,
):
    """Full training pipeline."""

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ─── Step 1: Collect all datasets ───
    data_root = Path("data/NAS-Bench/cifar_images")

    if clean and data_root.exists():
        print(f"\n[clean] Removing old images in {data_root} ...")
        for ds_dir in data_root.iterdir():
            img_dir = ds_dir / "images"
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
                print(f"  Removed {img_dir}")

    task_to_images: dict[str, list[str]] = {}
    datasets_info: dict[str, dict] = {}

    if dataset_name == "all":
        ds_list = BUILTIN_DATASETS
    elif dataset_name == "both":
        ds_list = ["cifar10", "cifar100"]
    else:
        ds_list = [dataset_name]

    for ds_name in ds_list:
        t2i = download_dataset(ds_name)
        n_classes = len(t2i)
        n_images = sum(len(v) for v in t2i.values())
        datasets_info[ds_name] = {"classes": n_classes, "images": n_images}
        print(f"  [{ds_name}] {n_classes} classes, {n_images} images")
        task_to_images.update(t2i)

    # Extra ImageFolder directories
    if extra_dirs:
        for folder in extra_dirs:
            t2i = load_imagefolder(folder)
            n_classes = len(t2i)
            n_images = sum(len(v) for v in t2i.values())
            name = Path(folder).name
            datasets_info[f"imagefolder_{name}"] = {"classes": n_classes, "images": n_images}
            print(f"  [imagefolder:{name}] {n_classes} classes, {n_images} images")
            task_to_images.update(t2i)

    # CSV label -> ImageFolder conversion
    if csv_label:
        if not csv_image_dir:
            raise ValueError("--csv_image_dir is required when using --csv_label")
        organised_dir = convert_csv_label_to_imagefolder(csv_label, csv_image_dir)
        t2i = load_imagefolder(organised_dir, prefix="csv")
        n_classes = len(t2i)
        n_images = sum(len(v) for v in t2i.values())
        datasets_info["csv_label"] = {"classes": n_classes, "images": n_images}
        print(f"  [csv_label] {n_classes} classes, {n_images} images")
        task_to_images.update(t2i)

    total_classes = len(task_to_images)
    total_images = sum(len(v) for v in task_to_images.values())
    print(f"\nTotal: {total_classes} classes, {total_images} images")

    # ─── Step 2: Build Dataset & DataLoader ───
    print(f"\nBuilding pair dataset ({pairs_per_epoch} pairs/epoch) ...")
    pair_dataset = ContrastivePairDataset(
        task_to_images=task_to_images,
        pairs_per_epoch=pairs_per_epoch,
    )
    loader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # ─── Step 3: Model & Loss ───
    print(f"\nSiameseEncoder (ResNet-18 -> {embed_dim}d)")
    encoder = SiameseEncoder(backbone="resnet18", embed_dim=embed_dim, pretrained=True)
    encoder = encoder.to(device)

    loss_fn = ContrastiveLoss(margin=margin, beta_pos=beta_pos, beta_neg=beta_neg)
    loss_fn = loss_fn.to(device)

    # Optional decoder for reconstruction regularization
    decoder = None
    actual_loss = loss_fn
    if use_decoder:
        decoder = ImageDecoder(embed_dim=embed_dim)
        decoder = decoder.to(device)
        actual_loss = CombinedLoss(
            contrastive_loss=loss_fn,
            contrastive_weight=1.0,
            recon_weight=recon_weight,
        )
        actual_loss = actual_loss.to(device)
        print(f"  Decoder enabled (recon_weight={recon_weight})")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + (list(decoder.parameters()) if decoder else []),
        lr=lr,
    )

    # Warmup + CosineAnnealing scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )

    # ─── Step 4: Train ───
    print(f"\nTraining ({epochs} epochs, warmup={warmup_epochs}) ...\n")
    print(f"{'Epoch':>6}  {'Loss':>10}  {'LR':>12}  {'Time':>8}")
    print("-" * 45)

    history = []
    best_loss = float("inf")
    best_epoch = -1
    best_path = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pair_dataset.reshuffle()

        for img1, img2, labels in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            v1 = encoder(img1)
            v2 = encoder(img2)

            if decoder is not None and isinstance(actual_loss, CombinedLoss):
                recon1 = decoder(v1)
                recon2 = decoder(v2)
                # Undo ImageNet normalization to [0,1] for MSE
                mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
                orig1_01 = (img1 * std + mean).clamp(0, 1)
                orig2_01 = (img2 * std + mean).clamp(0, 1)
                loss = actual_loss(v1, v2, labels, recon1, orig1_01, recon2, orig2_01)
            else:
                loss = actual_loss(v1, v2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Step the appropriate scheduler
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(avg_loss)

        print(f"{epoch:>6d}  {avg_loss:>10.5f}  {current_lr:>12.6f}  {elapsed:>7.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_path = os.path.join(checkpoint_dir, "contrastive_encoder_best.pt")
            ckpt_data = {
                "encoder_state": encoder.state_dict(),
                "backbone": encoder.backbone_name,
                "embed_dim": encoder.embed_dim,
                "epoch": epoch,
                "loss": avg_loss,
            }
            if decoder is not None:
                ckpt_data["decoder_state"] = decoder.state_dict()
            torch.save(ckpt_data, best_path)

    # ─── Step 5: Save final checkpoint ───
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "contrastive_encoder_final.pt")
    final_ckpt = {
        "encoder_state": encoder.state_dict(),
        "backbone": encoder.backbone_name,
        "embed_dim": encoder.embed_dim,
        "epoch": epochs,
        "loss": history[-1] if history else None,
        "history": history,
    }
    if decoder is not None:
        final_ckpt["decoder_state"] = decoder.state_dict()
    torch.save(final_ckpt, final_path)

    # ─── Step 6: Write training log ───
    log_dir = os.path.join(str(PROJECT_ROOT), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    log_name = f"contrastive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join(log_dir, log_name)

    log_data = {
        "timestamp": timestamp,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "margin": margin,
            "beta_pos": beta_pos,
            "beta_neg": beta_neg,
            "pairs_per_epoch": pairs_per_epoch,
            "embed_dim": embed_dim,
            "warmup_epochs": warmup_epochs,
            "num_workers": num_workers,
            "device": device,
            "use_decoder": use_decoder,
            "recon_weight": recon_weight,
        },
        "datasets": datasets_info,
        "total_images": total_images,
        "total_classes": total_classes,
        "history": history,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "checkpoint_best": best_path,
        "checkpoint_final": final_path,
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"\nTraining log: {log_path}")

    print(f"\nDone!")
    print(f"  Best loss: {best_loss:.5f} (epoch {best_epoch})")
    print(f"  Final:     {final_path}")
    if best_path:
        print(f"  Best:      {best_path}")

    return encoder, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Contrastive Encoder (production)")
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=BUILTIN_DATASETS + ["both", "all"],
                    help="Dataset to train on (default: cifar10). Use 'all' for all built-in datasets.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--beta_pos", type=float, default=1.0)
    ap.add_argument("--beta_neg", type=float, default=1.0)
    ap.add_argument("--pairs_per_epoch", type=int, default=20000)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--extra_dirs", nargs="+", default=None,
                    help="Additional ImageFolder directories to include")
    ap.add_argument("--csv_label", type=str, default=None,
                    help="Path to Kaggle-style CSV (filename, label)")
    ap.add_argument("--csv_image_dir", type=str, default=None,
                    help="Directory containing images referenced by --csv_label")
    ap.add_argument("--clean", action="store_true",
                    help="Remove old downloaded images before training (avoids duplicates from previous runs)")
    ap.add_argument("--use_decoder", action="store_true",
                    help="Enable reconstruction decoder for regularization")
    ap.add_argument("--recon_weight", type=float, default=0.5,
                    help="Weight for reconstruction loss (default: 0.5)")
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
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        extra_dirs=args.extra_dirs,
        csv_label=args.csv_label,
        csv_image_dir=args.csv_image_dir,
        clean=args.clean,
        use_decoder=args.use_decoder,
        recon_weight=args.recon_weight,
    )


if __name__ == "__main__":
    main()
