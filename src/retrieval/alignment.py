"""
SigLIP-style Contrastive Alignment for RAG-NAS.

Aligns a frozen SiameseEncoder (image, 128-d) and a frozen
SentenceTransformer (text, 384-d) into a shared embedding space
using learned ProjectionHeads and sigmoid pairwise contrastive loss.

Reference:
    Zhai et al., "Sigmoid Loss for Language Image Pre-Training",
    ICCV 2023.  arXiv:2303.15343

Usage:
    python -m src.retrieval.alignment \
        --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
        --text_model sentence-transformers/all-MiniLM-L6-v2 \
        --dataset_dir data/NAS-Bench/cifar_images \
        --epochs 20 --batch_size 256
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise RuntimeError("PyTorch is required. Install: pip install torch")

try:
    from PIL import Image
    import torchvision.transforms as T
except ImportError:
    raise RuntimeError("torchvision and Pillow required.")


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Two-layer MLP that maps encoder output to a shared embedding space.

    Architecture: in_dim → hidden → LayerNorm → GELU → shared_dim → L2-norm
    """

    def __init__(self, in_dim: int, shared_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize."""
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


# ---------------------------------------------------------------------------
# SigLIP Loss
# ---------------------------------------------------------------------------

class SigLIPLoss(nn.Module):
    """
    Sigmoid pairwise contrastive loss (SigLIP, ICCV 2023).

    L = -1/n Σᵢ Σⱼ log σ(yᵢⱼ · (zᵢᵀ·zⱼ · t + b))

    where yᵢⱼ = +1 for positive pairs, -1 for negative pairs,
    t = learnable temperature, b = learnable bias.

    Advantages over InfoNCE/CLIP:
      - No global softmax → works with small batch sizes
      - Pairwise → each pair contributes independently
    """

    def __init__(self, init_temperature: float = 10.0, init_bias: float = -10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(
        self, image_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_emb: (B, D) L2-normalized image projections
            text_emb:  (B, D) L2-normalized text projections
            (assumes diagonal = positive pairs, off-diagonal = negative)

        Returns:
            Scalar loss.
        """
        # Cosine similarity matrix: (B, B)
        logits = image_emb @ text_emb.T * self.temperature + self.bias

        # Labels: +1 on diagonal (positive), -1 off-diagonal (negative)
        B = image_emb.shape[0]
        labels = 2 * torch.eye(B, device=logits.device) - 1  # +1 diag, -1 off

        # Sigmoid loss: -log σ(y · logits)
        loss = -F.logsigmoid(labels * logits).mean()
        return loss


# ---------------------------------------------------------------------------
# Aligned Encoder
# ---------------------------------------------------------------------------

class AlignedEncoder(nn.Module):
    """
    Wraps frozen image/text encoders + learned projection heads.

    After training, encode_image() and encode_text() produce vectors
    in the same 256-d space, enabling cross-modal similarity search.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_dim: int = 128,
        text_dim: int = 384,
        shared_dim: int = 256,
    ):
        super().__init__()
        # Frozen backbone encoders
        self.image_encoder = image_encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

        self._text_encoder_name = text_encoder_name
        self._text_model = None  # lazy init

        # Learnable projection heads
        self.image_proj = ProjectionHead(image_dim, shared_dim)
        self.text_proj = ProjectionHead(text_dim, shared_dim)

        self.shared_dim = shared_dim
        self.image_dim = image_dim
        self.text_dim = text_dim

    def _get_text_model(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer(self._text_encoder_name)
            # Freeze text encoder
            for p in self._text_model.parameters():
                p.requires_grad = False
            self._text_model.eval()
        return self._text_model

    @torch.no_grad()
    def _encode_images_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Get raw image features from frozen encoder."""
        self.image_encoder.eval()
        return self.image_encoder(images)  # (B, image_dim)

    @torch.no_grad()
    def _encode_texts_raw(self, texts: List[str]) -> torch.Tensor:
        """Get raw text features from frozen SentenceTransformer."""
        model = self._get_text_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return torch.from_numpy(np.array(vecs))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to shared space. (B, C, H, W) → (B, shared_dim)"""
        with torch.no_grad():
            raw = self._encode_images_raw(images)
        return self.image_proj(raw)

    def encode_text(self, texts: List[str], device: Optional[str] = None) -> torch.Tensor:
        """Encode texts to shared space. List[str] → (N, shared_dim)"""
        raw = self._encode_texts_raw(texts)
        if device:
            raw = raw.to(device)
        return self.text_proj(raw)

    def forward(
        self, images: torch.Tensor, texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image_emb, text_emb) both in shared space."""
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(texts, device=images.device)
        return img_emb, txt_emb


# ---------------------------------------------------------------------------
# Image-Text Pair Dataset
# ---------------------------------------------------------------------------

class ImageTextPairDataset(Dataset):
    """
    Dataset producing (image, text_label) pairs for alignment training.

    Reads from an ImageFolder structure:
        root/class_name/image.png → text = class_name
    """

    def __init__(
        self,
        root_dirs: List[str],
        image_size: int = 32,
        max_per_class: int = 50,
    ):
        self.samples: List[Tuple[str, str]] = []  # (image_path, text_label)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        for root in root_dirs:
            root_path = Path(root)
            if not root_path.exists():
                continue
            # Walk ImageFolder structure
            for cls_dir in sorted(root_path.iterdir()):
                if not cls_dir.is_dir():
                    continue
                class_name = cls_dir.name
                imgs = sorted(
                    p for p in cls_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                )
                if len(imgs) > max_per_class:
                    imgs = random.sample(imgs, max_per_class)
                for img_path in imgs:
                    # Create descriptive text from class name
                    text = class_name.replace("_", " ")
                    self.samples.append((str(img_path), text))

        random.shuffle(self.samples)
        print(f"ImageTextPairDataset: {len(self.samples)} pairs from {len(root_dirs)} dirs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, text


def collate_fn(batch):
    """Custom collate to handle (tensor, string) pairs."""
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)


# ---------------------------------------------------------------------------
# Alignment Trainer
# ---------------------------------------------------------------------------

class AlignmentTrainer:
    """
    Train projection heads to align image/text embeddings using SigLIP loss.

    Only the ProjectionHead parameters are updated; both backbone encoders
    remain frozen throughout training.
    """

    def __init__(
        self,
        aligned_encoder: AlignedEncoder,
        dataset: ImageTextPairDataset,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: Optional[str] = None,
        num_workers: int = 4,
    ):
        self.encoder = aligned_encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

        self.loss_fn = SigLIPLoss().to(self.device)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            collate_fn=collate_fn,
            drop_last=True,
        )

        # Only train projection heads + loss params
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.image_proj.parameters())
            + list(self.encoder.text_proj.parameters())
            + list(self.loss_fn.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        self.history: List[float] = []

    def train(self, epochs: int = 20, log_every: int = 1) -> List[float]:
        """Train and return per-epoch losses."""
        print(f"\nTraining alignment ({epochs} epochs, device={self.device})")
        print(f"  Trainable params: image_proj + text_proj + loss = "
              f"{sum(p.numel() for p in self.encoder.image_proj.parameters()):,} + "
              f"{sum(p.numel() for p in self.encoder.text_proj.parameters()):,} + "
              f"{sum(p.numel() for p in self.loss_fn.parameters()):,}")
        print(f"{'Epoch':>6}  {'Loss':>10}  {'Temp':>8}  {'Bias':>8}  {'Time':>7}")
        print("-" * 50)

        for epoch in range(1, epochs + 1):
            self.encoder.image_proj.train()
            self.encoder.text_proj.train()
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for images, texts in self.loader:
                images = images.to(self.device)

                img_emb, txt_emb = self.encoder(images, texts)
                loss = self.loss_fn(img_emb, txt_emb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            self.history.append(avg)
            elapsed = time.time() - t0

            if epoch % log_every == 0 or epoch == 1:
                t = self.loss_fn.temperature.item()
                b = self.loss_fn.bias.item()
                print(f"{epoch:>6d}  {avg:>10.5f}  {t:>8.3f}  {b:>8.3f}  {elapsed:>6.1f}s")

        return self.history

    def save_checkpoint(self, path: str):
        """Save projection heads, loss params, and config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "image_proj_state": self.encoder.image_proj.state_dict(),
            "text_proj_state": self.encoder.text_proj.state_dict(),
            "loss_state": self.loss_fn.state_dict(),
            "image_dim": self.encoder.image_dim,
            "text_dim": self.encoder.text_dim,
            "shared_dim": self.encoder.shared_dim,
            "text_encoder_name": self.encoder._text_encoder_name,
        }, path)
        print(f"Saved alignment checkpoint → {path}")

    @staticmethod
    def load_aligned_encoder(
        alignment_path: str,
        encoder_checkpoint: str,
        device: Optional[str] = None,
    ) -> AlignedEncoder:
        """Load a fully trained AlignedEncoder from checkpoints."""
        from src.retrieval.contrastive_encoder import ContrastiveTrainer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load image encoder
        image_encoder = ContrastiveTrainer.load_checkpoint(encoder_checkpoint, device=device)

        # Load alignment config
        ckpt = torch.load(alignment_path, map_location=device, weights_only=False)

        aligned = AlignedEncoder(
            image_encoder=image_encoder,
            text_encoder_name=ckpt["text_encoder_name"],
            image_dim=ckpt["image_dim"],
            text_dim=ckpt["text_dim"],
            shared_dim=ckpt["shared_dim"],
        )
        aligned.image_proj.load_state_dict(ckpt["image_proj_state"])
        aligned.text_proj.load_state_dict(ckpt["text_proj_state"])
        aligned.to(device)
        aligned.image_proj.eval()
        aligned.text_proj.eval()
        return aligned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train SigLIP contrastive alignment")
    ap.add_argument("--encoder_checkpoint", required=True,
                    help="Path to SiameseEncoder checkpoint (.pt)")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model name")
    ap.add_argument("--dataset_dir", nargs="+",
                    default=["data/NAS-Bench/cifar_images/cifar10/images",
                             "data/NAS-Bench/cifar_images/cifar100/images",
                             "data/NAS-Bench/cifar_images/stl10/images",
                             "data/NAS-Bench/cifar_images/svhn/images",
                             "data/NAS-Bench/cifar_images/fashionmnist/images",
                             "data/NAS-Bench/cifar_images/flowers102/images",
                             "data/NAS-Bench/cifar_images/food101/images"],
                    help="ImageFolder directories for training pairs")
    ap.add_argument("--max_per_class", type=int, default=50,
                    help="Max images per class for training (default: 50)")
    ap.add_argument("--shared_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="checkpoints/aligned_encoder.pt",
                    help="Output checkpoint path")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load frozen image encoder
    from src.retrieval.contrastive_encoder import ContrastiveTrainer
    image_encoder = ContrastiveTrainer.load_checkpoint(
        args.encoder_checkpoint, device=args.device
    )
    embed_dim = image_encoder.embed_dim

    # Build AlignedEncoder
    aligned = AlignedEncoder(
        image_encoder=image_encoder,
        text_encoder_name=args.text_model,
        image_dim=embed_dim,
        shared_dim=args.shared_dim,
    )

    # Build dataset
    dataset = ImageTextPairDataset(
        root_dirs=args.dataset_dir,
        max_per_class=args.max_per_class,
    )

    if len(dataset) == 0:
        print("ERROR: No training data found. Run train_contrastive_encoder.py first.")
        sys.exit(1)

    # Train
    trainer = AlignmentTrainer(
        aligned_encoder=aligned,
        dataset=dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    trainer.train(epochs=args.epochs)
    trainer.save_checkpoint(args.output)

    # Write log
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"alignment_training_{ts}.json")
    log_data = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "config": {
            "encoder_checkpoint": args.encoder_checkpoint,
            "text_model": args.text_model,
            "shared_dim": args.shared_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "n_pairs": len(dataset),
        "history": trainer.history,
        "output": args.output,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
