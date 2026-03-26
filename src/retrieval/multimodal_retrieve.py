"""
Multi-Modal Retriever — fuses text and image pathways.

Orchestrates:
  1. DatasetAnalyzer → text query → existing retrieve()
  2. ContrastiveEncoder → image similarity vs FeatureStore
  3. Score fusion → ranked hits
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.retrieval.dataset_analyzer import DatasetAnalyzer, DatasetProfile
from src.retrieval.retrieve import retrieve as text_retrieve


# ---------------------------------------------------------------------------
# Multi-Modal Retriever
# ---------------------------------------------------------------------------

class MultiModalRetriever:
    """
    Fuses text-pathway and image-pathway retrieval scores.

    final_score = α · text_score + (1 - α) · image_score

    α is adjusted based on dataset characteristics:
      - Has rich text metadata → higher α (text-heavy)
      - Only images, no metadata → lower α (image-heavy)
    """

    def __init__(
        self,
        uir_path: str,
        encoder_checkpoint: Optional[str] = None,
        feature_store_dir: Optional[str] = None,
        alpha: float = 0.6,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.uir_path = uir_path
        self.alpha = alpha
        self.embed_model = embed_model

        self.analyzer = DatasetAnalyzer()

        # Image pathway (lazy init — only if checkpoint + store exist)
        self._encoder = None
        self._image_retriever = None
        self._feature_store = None
        self._encoder_checkpoint = encoder_checkpoint
        self._feature_store_dir = feature_store_dir

    def _init_image_pathway(self):
        """Lazy initialization of image encoder and feature store."""
        if self._encoder is not None:
            return

        if not self._encoder_checkpoint or not os.path.exists(self._encoder_checkpoint):
            return
        if not self._feature_store_dir or not os.path.exists(self._feature_store_dir):
            return

        from src.retrieval.contrastive_encoder import ContrastiveTrainer, ImageRetriever
        from src.retrieval.feature_store import FeatureStore

        self._encoder = ContrastiveTrainer.load_checkpoint(self._encoder_checkpoint)
        self._image_retriever = ImageRetriever(self._encoder)
        self._feature_store = FeatureStore.load(self._feature_store_dir)

    # ---- public API ----

    def retrieve(
        self,
        dataset_path: str,
        topk: int = 5,
        max_sample_images: int = 20,
    ) -> Dict[str, Any]:
        """
        Full multi-modal retrieval pipeline.

        Args:
            dataset_path: path to the user's dataset directory
            topk: number of results to return
            max_sample_images: max images to sample for image pathway

        Returns:
            {
                "profile": DatasetProfile,
                "hits": [...],
                "text_hits": [...],
                "image_hits": [...],
            }
        """
        # 1. Analyze dataset
        profile = self.analyzer.analyze(dataset_path)

        # 2. Text pathway
        query = profile.to_query()
        text_hits = text_retrieve(
            uir_path=self.uir_path,
            query=query,
            topk=topk * 2,  # over-retrieve for fusion
            embed_model=self.embed_model,
        )

        # Normalize text scores
        text_scores_by_id = {}
        if text_hits:
            max_s = max(h["score"] for h in text_hits) or 1.0
            for h in text_hits:
                text_scores_by_id[h["doc_id"]] = h["score"] / max_s

        # 3. Image pathway (if available)
        image_scores_by_id = {}
        image_hits_raw = []
        self._init_image_pathway()

        if self._image_retriever and self._feature_store:
            sample_images = self._sample_images(dataset_path, max_sample_images)
            if sample_images:
                store_vectors = self._feature_store.get_all_vectors()
                store_doc_ids = self._feature_store.get_all_doc_ids()

                image_hits_raw = self._image_retriever.retrieve(
                    query_image_paths=sample_images,
                    store_vectors=store_vectors,
                    store_doc_ids=store_doc_ids,
                    topk=topk * 2,
                )
                for ih in image_hits_raw:
                    image_scores_by_id[ih["doc_id"]] = ih["image_score"]

        # 4. Dynamic alpha adjustment
        alpha = self._compute_alpha(profile, bool(image_scores_by_id))

        # 5. Fusion
        all_doc_ids = set(text_scores_by_id.keys()) | set(image_scores_by_id.keys())
        fused = []
        for did in all_doc_ids:
            ts = text_scores_by_id.get(did, 0.0)
            im_s = image_scores_by_id.get(did, 0.0)
            final = alpha * ts + (1.0 - alpha) * im_s
            fused.append({"doc_id": did, "score": final, "text_score": ts, "image_score": im_s})

        fused.sort(key=lambda x: x["score"], reverse=True)
        fused = fused[:topk]

        # Enrich fused hits with full info from text_hits
        text_hit_map = {h["doc_id"]: h for h in text_hits}
        enriched_hits = []
        for f in fused:
            base = text_hit_map.get(f["doc_id"], {"doc_id": f["doc_id"]})
            base["score"] = f["score"]
            base["text_score"] = f["text_score"]
            base["image_score"] = f["image_score"]
            enriched_hits.append(base)

        return {
            "profile": profile,
            "hits": enriched_hits,
            "text_hits": text_hits[:topk],
            "image_hits": image_hits_raw[:topk],
            "alpha": alpha,
            "query": query,
        }

    def _compute_alpha(self, profile: DatasetProfile, has_image: bool) -> float:
        """Dynamically adjust α (text weight) based on dataset characteristics.

        .. warning:: **Heuristic α-schedule — requires ablation before publication.**

            The adjustment rules below (+0.1 for README, +0.05 for keywords,
            -0.2 for image-only) were hand-crafted.  They have NOT been
            validated on a held-out set of (dataset, best-α) pairs.
            Reviewers will ask: "How did you choose these deltas? Did you
            ablate text-only vs image-only vs fused?"  An ablation table
            covering at least three α values on the NAS retrieval validation
            set is needed for a credible submission.
        """
        alpha = self.alpha

        # Rich text metadata → boost text weight
        if profile.readme_text:
            alpha = min(alpha + 0.1, 0.9)
        if profile.keywords:
            alpha = min(alpha + 0.05, 0.9)

        # No image pathway → text only
        if not has_image:
            alpha = 1.0

        # No text info → image only
        if profile.task == "Image Classification" and not profile.keywords and not profile.readme_text:
            alpha = max(alpha - 0.2, 0.1)

        return alpha

    @staticmethod
    def _sample_images(dataset_path: str, max_n: int = 20) -> List[str]:
        """Sample up to max_n image paths from the dataset."""
        from src.retrieval.dataset_analyzer import IMAGE_EXTS
        from pathlib import Path

        root = Path(dataset_path)
        images: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if Path(fn).suffix.lower() in IMAGE_EXTS:
                    images.append(os.path.join(dirpath, fn))
                    if len(images) >= max_n * 3:  # over-collect then sample
                        break
            if len(images) >= max_n * 3:
                break

        if len(images) > max_n:
            import random
            images = random.sample(images, max_n)

        return images
