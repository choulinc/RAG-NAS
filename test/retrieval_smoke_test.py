"""
Smoke test for the retrieval modules:
1. DatasetAnalyzer
2. ContrastiveEncoder & ImageRetriever
3. MultiModalRetriever

This test creates temporary directories, dummy datasets, and dummy UIR JSON
to verify the End-to-End retrieval fusion logic.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from src.retrieval.contrastive_encoder import ContrastiveLoss, SiameseEncoder, ImageRetriever
from src.retrieval.dataset_analyzer import DatasetAnalyzer
from src.retrieval.multimodal_retrieve import MultiModalRetriever
from src.retrieval.feature_store import FeatureStore


class MockTextRetrieve:
    """Mock for src.retrieval.retrieve.retrieve"""
    def __call__(self, uir_path, query, topk, embed_model=None):
        return [
            {"doc_id": "doc_1", "score": 0.9, "text": "Mock hit 1"},
            {"doc_id": "doc_2", "score": 0.7, "text": "Mock hit 2"},
            {"doc_id": "doc_3", "score": 0.4, "text": "Mock hit 3"},
        ]


class TestRetrievalModules(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for datasets and feature store
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "mock_dataset")
        self.feature_store_dir = os.path.join(self.temp_dir, "feature_store")
        self.uir_path = os.path.join(self.temp_dir, "mock_uir.json")

        # 1. Setup Mock Dataset
        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "images"))
        os.makedirs(os.path.join(self.dataset_dir, "annotations"))

        # Create dummy README
        with open(os.path.join(self.dataset_dir, "README.md"), "w") as f:
            f.write("# Autonomous Driving Dataset\nThis is a dummy object detection dataset for autonomous vehicles.\n")

        # Create dummy detection annotation
        with open(os.path.join(self.dataset_dir, "annotations", "labels.json"), "w") as f:
            json.dump({"annotations": [{"id": 1}], "categories": [{"name": "car"}]}, f)

        # Create dummy images
        self.img1_path = os.path.join(self.dataset_dir, "images", "img1.jpg")
        self.img2_path = os.path.join(self.dataset_dir, "images", "img2.jpg")
        Image.new("RGB", (64, 64), color="red").save(self.img1_path)
        Image.new("RGB", (64, 64), color="blue").save(self.img2_path)

        # 2. Setup Mock UIR
        dummy_uir = [
            {"paper_id": "doc_1", "title": "Paper 1", "task": "Object Detection", "components": []},
            {"paper_id": "doc_2", "title": "Paper 2", "task": "Image Classification", "components": []},
        ]
        with open(self.uir_path, "w") as f:
            json.dump(dummy_uir, f)

        # 3. Setup Mock Feature Store (so MultiModalRetriever can load it)
        os.makedirs(self.feature_store_dir)
        db_path = os.path.join(self.feature_store_dir, "features.db")
        # Initialize FeatureStore
        fs = FeatureStore(dim=32)
        
        # Insert some dummy vectors for doc_1, doc_2
        dummy_vec_1 = np.random.rand(32).astype(np.float32)
        dummy_vec_2 = np.random.rand(32).astype(np.float32)
        fs.add("doc_1", dummy_vec_1)
        fs.add("doc_2", dummy_vec_2)
        
        # Build index and save to directory
        fs.build_index()
        fs.save(self.feature_store_dir)

    def tearDown(self):
        # Cleanup temporary directory
        shutil.rmtree(self.temp_dir)

    def test_01_dataset_analyzer(self):
        print("\n--- Testing DatasetAnalyzer ---")
        analyzer = DatasetAnalyzer()
        profile = analyzer.analyze(self.dataset_dir)

        print(f"  -> Inferred Task: {profile.task}")
        print(f"  -> Parsed Keywords: {profile.keywords}")
        print(f"  -> Total Images: {profile.image_stats.total_count}")
        print(f"  -> Median Image Width: {profile.image_stats.median_width}")

        # Check inference from directory structure and keywords
        self.assertEqual(profile.task, "Object Detection")
        # In this dummy structure, it detects directory names as class names
        self.assertIn("annotations", profile.keywords)
        self.assertEqual(profile.image_stats.total_count, 2)
        self.assertEqual(profile.image_stats.median_width, 64)
        print("DatasetAnalyzer: OK")

    def test_02_contrastive_encoder(self):
        print("\n--- Testing ContrastiveEncoder ---")
        # Small encoder to avoid downloading big weights during test
        encoder = SiameseEncoder(backbone="resnet18", embed_dim=32, pretrained=False)
        loss_fn = ContrastiveLoss(margin=1.0)

        # Forward pass & loss & backward
        b, c, h, w = 2, 3, 32, 32
        img1 = torch.rand(b, c, h, w)
        img2 = torch.rand(b, c, h, w)
        labels = torch.tensor([1.0, 0.0])  # One same, one diff

        v1, v2 = encoder.encode_pair(img1, img2)
        print(f"  -> Encoded vector shape: {v1.shape}")
        self.assertEqual(v1.shape, (2, 32))

        loss = loss_fn(v1, v2, labels)
        print(f"  -> Computed Contrastive Loss: {loss.item():.4f}")
        self.assertTrue(loss.requires_grad)

        loss.backward()
        # Verify gradients are populated in the projector
        self.assertIsNotNone(encoder.projector[0].weight.grad)

        # Test ImageRetriever
        retriever = ImageRetriever(encoder, device="cpu")
        store_vecs = np.random.rand(5, 32).astype(np.float32)
        store_vecs /= np.linalg.norm(store_vecs, axis=1, keepdims=True)
        doc_ids = [f"doc_{i}" for i in range(5)]

        hits = retriever.retrieve([self.img1_path], store_vecs, doc_ids, topk=2)
        print(f"  -> Image Retrieval Top-K Hits:")
        for h in hits:
             print(f"     Doc: {h['doc_id']}, Score: {h['image_score']:.4f}")
        
        self.assertEqual(len(hits), 2)
        self.assertIn("doc_id", hits[0])
        self.assertIn("image_score", hits[0])
        print("ContrastiveEncoder: OK")

    @unittest.mock.patch("src.retrieval.multimodal_retrieve.text_retrieve")
    def test_03_multimodal_retriever(self, mock_text_retrieve):
        print("\n--- Testing MultiModalRetriever ---")
        # Replace text retrieval with mock
        mock_text_retrieve.side_effect = MockTextRetrieve()

        # Save a mock encoder checkpoint to be loaded by MultiModalRetriever
        encoder = SiameseEncoder(backbone="resnet18", embed_dim=32, pretrained=False)
        ckpt_path = os.path.join(self.temp_dir, "encoder.pt")
        torch.save(
            {
                "encoder_state": encoder.state_dict(),
                "backbone": "resnet18",
                "embed_dim": 32,
            },
            ckpt_path,
        )

        retriever = MultiModalRetriever(
            uir_path=self.uir_path,
            encoder_checkpoint=ckpt_path,
            feature_store_dir=self.feature_store_dir,
            alpha=0.6,
        )

        # Run full retrieval
        result = retriever.retrieve(dataset_path=self.dataset_dir, topk=3)

        # Check structure
        self.assertIn("profile", result)
        self.assertIn("hits", result)
        self.assertIn("text_hits", result)
        self.assertIn("image_hits", result)
        self.assertIn("alpha", result)

        hits = result["hits"]
        # Max topk = 3
        self.assertLessEqual(len(hits), 3)

        # Ensure fusion fields are present
        if len(hits) > 0:
            first_hit = hits[0]
            self.assertIn("score", first_hit)
            self.assertIn("text_score", first_hit)
            self.assertIn("image_score", first_hit)

        print(f"  -> Dynamic Alpha computed: {result['alpha']:.2f}")
        print(f"  -> Fused Hits Results:")
        for rank, h in enumerate(hits, 1):
            print(f"     #{rank} Doc: {h['doc_id']}")
            print(f"         Final Score : {h['score']:.4f}")
            print(f"         Text Score  : {h.get('text_score', 0):.4f}")
            print(f"         Image Score : {h.get('image_score', 0):.4f}")

        print("MultiModalRetriever: OK\n")


if __name__ == "__main__":
    unittest.main()
