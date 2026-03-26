"""
Contrastive Learning Encoder for RAG-NAS image pathway.

Provides:
  - ContrastiveLoss: conditional loss switching between same/different task types
  - SiameseEncoder: ResNet-18 backbone → 128-d embedding (backbone swappable)
  - ContrastiveTrainer: pair-based training on NAS-Bench-201 datasets
  - ImageRetriever: cosine similarity search against a FeatureStore
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise RuntimeError("PyTorch is required. Install: pip install torch torchvision")

try:
    import torchvision.models as tv_models
    import torchvision.transforms as T
except ImportError:
    raise RuntimeError("torchvision is required. Install: pip install torchvision")

try:
    from PIL import Image
except ImportError:
    raise RuntimeError("Pillow is required. Install: pip install Pillow")


# ---------------------------------------------------------------------------
# Contrastive Loss
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with conditional switching.

    L(V1, V2, y) = y · β₁ · ‖V1 - V2‖²
                 + (1-y) · β₂ · max(0, margin - ‖V1 - V2‖)²

    Args:
        margin: minimum desired distance between different-class pairs.
        beta_pos: weight coefficient for same-class (positive) pairs.
        beta_neg: weight coefficient for different-class (negative) pairs.
    """

    def __init__(self, margin: float = 1.0, beta_pos: float = 1.0, beta_neg: float = 1.0):
        super().__init__()
        self.margin = margin
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg

    def forward(
        self, v1: torch.Tensor, v2: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            v1: (B, D) embedding batch from encoder arm 1
            v2: (B, D) embedding batch from encoder arm 2
            y:  (B,) labels — 1.0 if same task type, 0.0 if different

        Returns:
            Scalar loss averaged over the batch.
        """
        # Euclidean distance
        dist = F.pairwise_distance(v1, v2)  # (B,)

        # Same-class loss: minimize distance
        loss_same = y * self.beta_pos * dist.pow(2)

        # Different-class loss: push apart beyond margin
        loss_diff = (1.0 - y) * self.beta_neg * F.relu(self.margin - dist).pow(2)

        loss = loss_same + loss_diff
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Joint loss: contrastive + reconstruction.

    L_total = λ_c · L_contrastive + λ_r · L_reconstruction

    Reconstruction loss uses MSE between the original image and the
    decoder output, encouraging the embedding to retain fine-grained
    visual details.
    """

    def __init__(
        self,
        contrastive_loss: ContrastiveLoss,
        contrastive_weight: float = 1.0,
        recon_weight: float = 0.5,
    ):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.contrastive_weight = contrastive_weight
        self.recon_weight = recon_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        y: torch.Tensor,
        recon1: Optional[torch.Tensor] = None,
        orig1: Optional[torch.Tensor] = None,
        recon2: Optional[torch.Tensor] = None,
        orig2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            v1, v2: (B, D) embeddings
            y: (B,) labels
            recon1, orig1: reconstructed and original images for arm 1
            recon2, orig2: reconstructed and original images for arm 2

        Returns:
            Scalar combined loss.
        """
        loss_c = self.contrastive_loss(v1, v2, y)
        loss = self.contrastive_weight * loss_c

        if recon1 is not None and orig1 is not None:
            loss_r1 = self.mse(recon1, orig1)
            loss += self.recon_weight * loss_r1
        if recon2 is not None and orig2 is not None:
            loss_r2 = self.mse(recon2, orig2)
            loss += self.recon_weight * loss_r2

        return loss


# ---------------------------------------------------------------------------
# Siamese Encoder
# ---------------------------------------------------------------------------

class SiameseEncoder(nn.Module):
    """
    Shared-weight encoder producing a compact embedding.

    Architecture: backbone → Global Average Pool → FC → L2-normalize → embed_dim
    Default backbone: ResNet-18 (pretrained on ImageNet).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embed_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim

        if backbone == "resnet18":
            base = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = base.fc.in_features  # 512
            # Remove the original FC layer
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Currently only 'resnet18' is supported."
            )

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            (B, embed_dim) L2-normalized embedding
        """
        feat = self.backbone(x)  # (B, feat_dim)
        emb = self.projector(feat)  # (B, embed_dim)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def encode_pair(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: encode two images through the shared encoder."""
        return self.forward(x1), self.forward(x2)


# ---------------------------------------------------------------------------
# Image Decoder
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    """
    Decoder that reconstructs images from compact embeddings.

    Architecture: embed_dim → FC → Reshape → ConvTranspose2d layers → (3, 32, 32)

    Used as a reconstruction regularizer during contrastive training
    to encourage the encoder to retain fine-grained visual details.
    """

    def __init__(self, embed_dim: int = 128, image_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size

        # FC: embed_dim → 512 * 2 * 2 = 2048
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512 * 2 * 2),
            nn.ReLU(inplace=True),
        )

        # Transposed convolutions: (512, 2, 2) → (3, 32, 32)
        self.deconv = nn.Sequential(
            # (512, 2, 2) → (256, 4, 4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # (256, 4, 4) → (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (128, 8, 8) → (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (64, 16, 16) → (3, 32, 32)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, embed_dim) — L2-normalized embedding from encoder

        Returns:
            (B, 3, image_size, image_size) reconstructed image in [0, 1]
        """
        x = self.fc(embedding)          # (B, 512*2*2)
        x = x.view(-1, 512, 2, 2)      # (B, 512, 2, 2)
        x = self.deconv(x)              # (B, 3, 32, 32)
        return x


# ---------------------------------------------------------------------------
# Pair Dataset
# ---------------------------------------------------------------------------

@dataclass
class PairSample:
    path1: str
    path2: str
    label: float  # 1.0 = same task type, 0.0 = different


class ContrastivePairDataset(Dataset):
    """
    Dataset yielding (img1, img2, label) pairs for contrastive training.

    Expects a dict mapping task_type → list of image paths.
    Generates balanced positive/negative pairs each epoch.
    """

    def __init__(
        self,
        task_to_images: Dict[str, List[str]],
        pairs_per_epoch: int = 2000,
        image_size: int = 32,
        augment: bool = True,
    ):
        self.task_to_images = task_to_images
        self.task_names = list(task_to_images.keys())
        # Filtered pools for safe sampling
        self.positive_pool = [k for k, v in task_to_images.items() if len(v) >= 2]
        self.negative_pool = [k for k, v in task_to_images.items() if len(v) >= 1]
        self.pairs_per_epoch = pairs_per_epoch
        self.pairs: List[PairSample] = []
        self._build_pairs()

        # Transforms
        base_transforms = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
        if augment:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomCrop(image_size, padding=4),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
        else:
            self.transform = T.Compose(base_transforms)

    def _build_pairs(self):
        """Generate balanced positive and negative pairs."""
        self.pairs = []
        n_pos = self.pairs_per_epoch // 2
        n_neg = self.pairs_per_epoch - n_pos

        # Positive pairs: same task type (requires ≥2 images)
        for _ in range(n_pos):
            if not self.positive_pool:
                break
            task = random.choice(self.positive_pool)
            imgs = self.task_to_images[task]
            i1, i2 = random.sample(range(len(imgs)), 2)
            self.pairs.append(PairSample(imgs[i1], imgs[i2], 1.0))

        # Negative pairs: different task types (requires ≥1 image each)
        if len(self.negative_pool) >= 2:
            for _ in range(n_neg):
                t1, t2 = random.sample(self.negative_pool, 2)
                i1 = random.choice(self.task_to_images[t1])
                i2 = random.choice(self.task_to_images[t2])
                self.pairs.append(PairSample(i1, i2, 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img1 = Image.open(pair.path1).convert("RGB")
        img2 = Image.open(pair.path2).convert("RGB")
        return self.transform(img1), self.transform(img2), torch.tensor(pair.label)

    def reshuffle(self):
        """Rebuild pairs for a new epoch."""
        self._build_pairs()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContrastiveTrainer:
    """
    Training pipeline for the Siamese contrastive encoder.

    Optionally trains with an ImageDecoder for reconstruction regularization.

    Usage:
        trainer = ContrastiveTrainer(encoder, loss_fn, task_to_images)
        trainer.train(epochs=50)
        trainer.save_checkpoint("checkpoints/encoder.pt")

        # With decoder:
        decoder = ImageDecoder(embed_dim=128)
        combined = CombinedLoss(loss_fn, recon_weight=0.5)
        trainer = ContrastiveTrainer(encoder, combined, task_to_images, decoder=decoder)
    """

    def __init__(
        self,
        encoder: SiameseEncoder,
        loss_fn: nn.Module,
        task_to_images: Dict[str, List[str]],
        lr: float = 1e-4,
        batch_size: int = 64,
        pairs_per_epoch: int = 2000,
        image_size: int = 32,
        device: Optional[str] = None,
        decoder: Optional["ImageDecoder"] = None,
    ):
        self.encoder = encoder
        self.loss_fn = loss_fn
        self.decoder = decoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.loss_fn.to(self.device)
        if self.decoder is not None:
            self.decoder.to(self.device)

        self.dataset = ContrastivePairDataset(
            task_to_images=task_to_images,
            pairs_per_epoch=pairs_per_epoch,
            image_size=image_size,
        )
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        # Optimise both encoder and decoder params
        params = list(encoder.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.history: List[float] = []

    def train(self, epochs: int = 50, log_every: int = 5) -> List[float]:
        """Train and return per-epoch average losses."""
        self.encoder.train()
        for epoch in range(1, epochs + 1):
            self.dataset.reshuffle()
            self.loader = DataLoader(
                self.dataset, batch_size=self.loader.batch_size,
                shuffle=True, num_workers=2, pin_memory=True,
            )
            epoch_loss = 0.0
            n_batches = 0
            for img1, img2, labels in self.loader:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)

                v1, v2 = self.encoder.encode_pair(img1, img2)

                if self.decoder is not None and isinstance(self.loss_fn, CombinedLoss):
                    recon1 = self.decoder(v1)
                    recon2 = self.decoder(v2)
                    # Normalize originals to [0,1] for MSE (undo ImageNet normalization)
                    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.2023, 0.1994, 0.2010], device=self.device).view(1, 3, 1, 1)
                    orig1_01 = (img1 * std + mean).clamp(0, 1)
                    orig2_01 = (img2 * std + mean).clamp(0, 1)
                    loss = self.loss_fn(v1, v2, labels, recon1, orig1_01, recon2, orig2_01)
                else:
                    loss = self.loss_fn(v1, v2, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            self.history.append(avg)
            if epoch % log_every == 0 or epoch == 1:
                print(f"[Epoch {epoch:3d}/{epochs}] Loss: {avg:.5f}")

        return self.history

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "encoder_state": self.encoder.state_dict(),
            "backbone": self.encoder.backbone_name,
            "embed_dim": self.encoder.embed_dim,
        }
        if self.decoder is not None:
            ckpt["decoder_state"] = self.decoder.state_dict()
        torch.save(ckpt, path)
        print(f"Saved checkpoint → {path}")

    @staticmethod
    def load_checkpoint(
        path: str,
        device: Optional[str] = None,
        load_decoder: bool = False,
    ) -> "SiameseEncoder | Tuple[SiameseEncoder, ImageDecoder]":
        """Load encoder (and optionally decoder) from checkpoint.

        Returns:
            SiameseEncoder if load_decoder is False.
            (SiameseEncoder, ImageDecoder) if load_decoder is True and
            decoder state is present in the checkpoint.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        encoder = SiameseEncoder(
            backbone=ckpt["backbone"],
            embed_dim=ckpt["embed_dim"],
            pretrained=False,
        )
        encoder.load_state_dict(ckpt["encoder_state"])
        encoder.to(device)
        encoder.eval()

        if load_decoder and "decoder_state" in ckpt:
            decoder = ImageDecoder(embed_dim=ckpt["embed_dim"])
            decoder.load_state_dict(ckpt["decoder_state"])
            decoder.to(device)
            decoder.eval()
            return encoder, decoder

        return encoder


# ---------------------------------------------------------------------------
# Image Retriever
# ---------------------------------------------------------------------------

class ImageRetriever:
    """
    Given a trained encoder and a FeatureStore, retrieve the most similar
    RAG DB entries for a batch of query images.
    """

    def __init__(self, encoder: SiameseEncoder, device: Optional[str] = None):
        self.encoder = encoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

        self.transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode a list of images into feature vectors. Returns (N, embed_dim)."""
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)
        embeddings = self.encoder(batch)
        return embeddings.cpu().numpy()

    def retrieve(
        self,
        query_image_paths: List[str],
        store_vectors: np.ndarray,
        store_doc_ids: List[str],
        topk: int = 5,
        aggregate: str = "mean",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k RAG DB entries most similar to query images.

        Args:
            query_image_paths: paths to images from the user's dataset
            store_vectors: (M, D) pre-computed feature vectors for RAG DB entries
            store_doc_ids: doc_id for each row in store_vectors
            topk: number of results to return
            aggregate: how to aggregate multiple query images ("mean" or "max")

        Returns:
            List of {"doc_id": str, "image_score": float}
        """
        q_vecs = self.encode_images(query_image_paths)  # (N, D)

        if aggregate == "mean":
            q_vec = q_vecs.mean(axis=0, keepdims=True)  # (1, D)
        else:
            q_vec = q_vecs  # (N, D)

        # Cosine similarity (vectors are L2-normalized)
        sims = q_vec @ store_vectors.T  # (1, M) or (N, M)

        if aggregate == "max" and sims.ndim == 2 and sims.shape[0] > 1:
            sims = sims.max(axis=0, keepdims=True)  # (1, M)

        sims = sims.squeeze(0)  # (M,)
        top_idx = np.argsort(-sims)[:topk]

        hits = []
        for idx in top_idx:
            hits.append({
                "doc_id": store_doc_ids[idx],
                "image_score": float(sims[idx]),
            })
        return hits
