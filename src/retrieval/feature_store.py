"""
Feature Store for RAG-NAS image pathway.

Stores pre-computed image feature vectors for RAG DB entries,
enabling fast cosine similarity search.
Uses FAISS for indexing and pickle for metadata.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FeatureStore:
    """
    Persistent store for image feature vectors, keyed by doc_id.

    Storage layout:
        {store_dir}/
            index.faiss     — FAISS inner-product index
            metadata.pkl    — doc_ids + extra info per entry

    Usage:
        store = FeatureStore(dim=128)
        store.add("doc_001", vector, {"task": "Image Classification"})
        store.build_index()
        store.save("data/processed/feature_store")

        # later
        store = FeatureStore.load("data/processed/feature_store")
        hits = store.search(query_vector, topk=5)
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.doc_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index: Optional[Any] = None  # faiss.IndexFlatIP

    def add(self, doc_id: str, vector: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        """Add a single entry."""
        assert vector.shape == (self.dim,), f"Expected ({self.dim},), got {vector.shape}"
        # L2-normalize for cosine similarity via inner product
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self.vectors.append(vector.astype(np.float32))
        self.doc_ids.append(doc_id)
        self.metadata.append(meta or {})

    def add_batch(self, doc_ids: List[str], vectors: np.ndarray, metas: Optional[List[Dict]] = None):
        """Add multiple entries at once."""
        assert vectors.shape[1] == self.dim
        if metas is None:
            metas = [{}] * len(doc_ids)
        for did, vec, m in zip(doc_ids, vectors, metas):
            self.add(did, vec, m)

    def build_index(self):
        """Build FAISS index from accumulated vectors."""
        if not faiss:
            raise RuntimeError("FAISS is required. Install: pip install faiss-cpu")
        if not self.vectors:
            raise ValueError("No vectors to index.")

        mat = np.stack(self.vectors).astype(np.float32)
        self._index = faiss.IndexFlatIP(self.dim)  # inner product ≈ cosine (L2-normed)
        self._index.add(mat)

    def search(self, query: np.ndarray, topk: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the closest entries.

        Args:
            query: (D,) or (1, D) normalized query vector
            topk: number of results

        Returns:
            List of {"doc_id", "score", "metadata"} dicts, sorted by score desc.
        """
        if self._index is None:
            self.build_index()

        query = query.astype(np.float32).reshape(1, -1)
        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores, indices = self._index.search(query, min(topk, len(self.doc_ids)))

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append({
                "doc_id": self.doc_ids[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return hits

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def get_all_vectors(self) -> np.ndarray:
        """Return (N, D) matrix of all stored vectors."""
        if not self.vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self.vectors).astype(np.float32)

    def get_all_doc_ids(self) -> List[str]:
        return list(self.doc_ids)

    # ---- persistence ----

    def save(self, store_dir: str):
        """Save index + metadata to disk."""
        if not faiss:
            raise RuntimeError("FAISS is required for persistence.")
        os.makedirs(store_dir, exist_ok=True)

        if self._index is None:
            self.build_index()

        faiss.write_index(self._index, os.path.join(store_dir, "index.faiss"))
        with open(os.path.join(store_dir, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "dim": self.dim,
                "doc_ids": self.doc_ids,
                "metadata": self.metadata,
                "vectors": [v.tolist() for v in self.vectors],
            }, f)
        print(f"FeatureStore saved → {store_dir} ({self.size} entries)")

    @classmethod
    def load(cls, store_dir: str) -> "FeatureStore":
        """Load a previously saved store."""
        if not faiss:
            raise RuntimeError("FAISS is required for persistence.")

        with open(os.path.join(store_dir, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)

        store = cls(dim=data["dim"])
        store.doc_ids = data["doc_ids"]
        store.metadata = data["metadata"]
        store.vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
        store._index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
        return store
