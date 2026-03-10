"""
Unit tests for DatasetAnalyzer and ContrastiveEncoder.
Run: pytest test/test_retrieval_modules.py -v
"""
import os
import sys
import json
import tempfile
import shutil

# Ensure src in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.dataset_analyzer import DatasetAnalyzer, DatasetProfile, ImageStats


# ============================================================================
# DatasetAnalyzer Tests
# ============================================================================

class TestDatasetAnalyzer:
    """Test dataset analysis from directory structures and metadata."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.analyzer = DatasetAnalyzer()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_img(self, path: str, size=(32, 32)):
        """Create a tiny dummy image."""
        try:
            from PIL import Image
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img = Image.new("RGB", size, color=(128, 128, 128))
            img.save(path)
        except ImportError:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"\x89PNG" + b"\x00" * 100)  # dummy

    # ---- Classification layout ----

    def test_classification_from_subdirs(self):
        """train/class_a/*.jpg + train/class_b/*.jpg → Image Classification"""
        for cls in ["cat", "dog", "bird"]:
            for i in range(3):
                self._make_img(os.path.join(self.tmpdir, "train", cls, f"{i}.jpg"))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.task == "Image Classification"
        assert profile.num_classes == 3
        assert set(profile.class_names) == {"bird", "cat", "dog"}

    # ---- Detection layout ----

    def test_detection_from_annotations_xml(self):
        """annotations/ with .xml files → Object Detection"""
        os.makedirs(os.path.join(self.tmpdir, "images"), exist_ok=True)
        self._make_img(os.path.join(self.tmpdir, "images", "001.jpg"))
        ann_dir = os.path.join(self.tmpdir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        with open(os.path.join(ann_dir, "001.xml"), "w") as f:
            f.write("<annotation><object><name>car</name></object></annotation>")

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.task == "Object Detection"

    def test_detection_from_yolo_labels(self):
        """labels/ with YOLO-format .txt → Object Detection"""
        os.makedirs(os.path.join(self.tmpdir, "images"), exist_ok=True)
        self._make_img(os.path.join(self.tmpdir, "images", "001.jpg"))
        labels_dir = os.path.join(self.tmpdir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        with open(os.path.join(labels_dir, "001.txt"), "w") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.task == "Object Detection"

    # ---- Segmentation layout ----

    def test_segmentation_from_mask_dir(self):
        """masks/ directory → Image Segmentation"""
        os.makedirs(os.path.join(self.tmpdir, "images"), exist_ok=True)
        self._make_img(os.path.join(self.tmpdir, "images", "001.jpg"))
        os.makedirs(os.path.join(self.tmpdir, "masks"), exist_ok=True)
        self._make_img(os.path.join(self.tmpdir, "masks", "001.png"))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.task == "Image Segmentation"

    # ---- Metadata files ----

    def test_data_yaml(self):
        """data.yaml with nc and names → correct num_classes"""
        data = {"nc": 5, "names": ["a", "b", "c", "d", "e"], "task": "detect"}
        with open(os.path.join(self.tmpdir, "data.yaml"), "w") as f:
            import yaml
            yaml.dump(data, f)

        os.makedirs(os.path.join(self.tmpdir, "images"), exist_ok=True)
        self._make_img(os.path.join(self.tmpdir, "images", "001.jpg"))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.num_classes == 5
        assert profile.task == "Object Detection"

    def test_labels_txt(self):
        """labels.txt listing class names"""
        with open(os.path.join(self.tmpdir, "labels.txt"), "w") as f:
            f.write("airplane\nautomobile\nbird\ncat\ndeer\n")

        for cls in ["a", "b"]:
            self._make_img(os.path.join(self.tmpdir, "train", cls, "0.jpg"))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.num_classes in [5, 2]  # from labels.txt or subdirs
        assert "airplane" in profile.class_names or len(profile.class_names) > 0

    def test_classes_json(self):
        """classes.json listing class names"""
        with open(os.path.join(self.tmpdir, "classes.json"), "w") as f:
            json.dump(["cat", "dog", "horse"], f)

        self._make_img(os.path.join(self.tmpdir, "0.jpg"))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.num_classes == 3

    # ---- README keyword matching ----

    def test_readme_detection_keyword(self):
        """README mentions 'object detection' → task = Object Detection"""
        with open(os.path.join(self.tmpdir, "README.md"), "w") as f:
            f.write("# My Dataset\nThis is an object detection dataset for autonomous driving.\n")

        self._make_img(os.path.join(self.tmpdir, "0.jpg"))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.task == "Object Detection"

    # ---- Domain inference ----

    def test_cifar_domain(self):
        """Path containing 'cifar' → domain = cifar"""
        cifar_dir = os.path.join(self.tmpdir, "cifar100_subset")
        os.makedirs(cifar_dir, exist_ok=True)
        for cls in ["a", "b"]:
            self._make_img(os.path.join(cifar_dir, "train", cls, "0.jpg"))

        profile = self.analyzer.analyze(cifar_dir)
        assert profile.domain == "cifar"

    # ---- to_query ----

    def test_to_query(self):
        """DatasetProfile.to_query() returns a usable search string."""
        profile = DatasetProfile(
            task="Image Classification",
            domain="cifar",
            keywords=["cifar100", "airplane", "automobile"],
            num_classes=100,
        )
        q = profile.to_query()
        assert "Image Classification" in q
        assert "cifar" in q
        assert "100 classes" in q

    # ---- Image stats ----

    def test_image_stats(self):
        """Verify image stats collection."""
        for i in range(5):
            self._make_img(os.path.join(self.tmpdir, "train", "a", f"{i}.jpg"), size=(32, 32))

        profile = self.analyzer.analyze(self.tmpdir)
        assert profile.image_stats.total_count >= 5
        if profile.image_stats.sample_heights:  # PIL available
            assert profile.image_stats.median_height == 32
            assert profile.image_stats.median_width == 32
            assert profile.image_stats.channels == 3


# ============================================================================
# ContrastiveLoss Tests
# ============================================================================

class TestContrastiveLoss:
    """Test contrastive loss computations."""

    def test_same_class_zero_distance(self):
        """Same-class pair with identical vectors → loss ≈ 0"""
        import torch
        from src.retrieval.contrastive_encoder import ContrastiveLoss

        loss_fn = ContrastiveLoss(margin=1.0)
        v = torch.randn(4, 128)
        y = torch.ones(4)  # same class
        loss = loss_fn(v, v, y)
        assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"

    def test_same_class_loss_increases_with_distance(self):
        """Same-class loss should increase when vectors are farther apart."""
        import torch
        from src.retrieval.contrastive_encoder import ContrastiveLoss

        loss_fn = ContrastiveLoss(margin=1.0)
        v1 = torch.zeros(4, 128)
        y = torch.ones(4)

        v2_close = torch.ones(4, 128) * 0.1
        v2_far = torch.ones(4, 128) * 1.0

        loss_close = loss_fn(v1, v2_close, y).item()
        loss_far = loss_fn(v1, v2_far, y).item()
        assert loss_far > loss_close, f"Expected loss_far > loss_close, got {loss_far} <= {loss_close}"

    def test_diff_class_beyond_margin_zero_loss(self):
        """Different-class pair beyond margin → loss = 0"""
        import torch
        from src.retrieval.contrastive_encoder import ContrastiveLoss

        loss_fn = ContrastiveLoss(margin=1.0)
        v1 = torch.zeros(4, 128)
        v2 = torch.ones(4, 128) * 5.0  # very far apart
        y = torch.zeros(4)  # different class

        loss = loss_fn(v1, v2, y)
        assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"

    def test_diff_class_within_margin_positive_loss(self):
        """Different-class pair within margin → positive loss"""
        import torch
        from src.retrieval.contrastive_encoder import ContrastiveLoss

        loss_fn = ContrastiveLoss(margin=2.0)
        v1 = torch.zeros(4, 128)
        v2 = torch.zeros(4, 128)
        v2[:, 0] = 0.1  # very close → within margin
        y = torch.zeros(4)

        loss = loss_fn(v1, v2, y)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_beta_coefficients(self):
        """Beta coefficients scale the loss correctly."""
        import torch
        from src.retrieval.contrastive_encoder import ContrastiveLoss

        v1 = torch.zeros(4, 128)
        v2 = torch.ones(4, 128) * 0.5
        y = torch.ones(4)

        loss_b1 = ContrastiveLoss(beta_pos=1.0)(v1, v2, y).item()
        loss_b2 = ContrastiveLoss(beta_pos=2.0)(v1, v2, y).item()
        assert abs(loss_b2 - 2 * loss_b1) < 1e-4, f"Beta scaling failed: {loss_b2} != 2 * {loss_b1}"


# ============================================================================
# SiameseEncoder Tests
# ============================================================================

class TestSiameseEncoder:
    """Test encoder shape and properties."""

    def test_output_shape(self):
        """Input (B, 3, 32, 32) → output (B, 128)"""
        import torch
        from src.retrieval.contrastive_encoder import SiameseEncoder

        encoder = SiameseEncoder(embed_dim=128, pretrained=False)
        x = torch.randn(2, 3, 32, 32)
        out = encoder(x)
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_identical_inputs_identical_outputs(self):
        """Same image → same embedding"""
        import torch
        from src.retrieval.contrastive_encoder import SiameseEncoder

        encoder = SiameseEncoder(embed_dim=128, pretrained=False)
        encoder.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            v1 = encoder(x)
            v2 = encoder(x)
        assert torch.allclose(v1, v2, atol=1e-6), "Same input should produce same output"

    def test_l2_normalized(self):
        """Output vectors should be L2-normalized (unit norm)."""
        import torch
        from src.retrieval.contrastive_encoder import SiameseEncoder

        encoder = SiameseEncoder(embed_dim=128, pretrained=False)
        encoder.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = encoder(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Norms should be ~1.0, got {norms}"

    def test_encode_pair(self):
        """encode_pair returns two embeddings of correct shape."""
        import torch
        from src.retrieval.contrastive_encoder import SiameseEncoder

        encoder = SiameseEncoder(embed_dim=128, pretrained=False)
        x1 = torch.randn(2, 3, 32, 32)
        x2 = torch.randn(2, 3, 32, 32)
        v1, v2 = encoder.encode_pair(x1, x2)
        assert v1.shape == (2, 128) and v2.shape == (2, 128)


# ============================================================================
# FeatureStore Tests
# ============================================================================

class TestFeatureStore:
    """Test feature store add/search/save/load."""

    def test_add_and_search(self):
        """Basic add + search returns correct doc_id."""
        import numpy as np
        from src.retrieval.feature_store import FeatureStore

        store = FeatureStore(dim=4)
        store.add("doc_a", np.array([1.0, 0, 0, 0]))
        store.add("doc_b", np.array([0, 1.0, 0, 0]))
        store.add("doc_c", np.array([0, 0, 1.0, 0]))

        hits = store.search(np.array([1.0, 0, 0, 0]), topk=1)
        assert hits[0]["doc_id"] == "doc_a"
        assert hits[0]["score"] > 0.9

    def test_save_and_load(self):
        """Save + load preserves entries."""
        import numpy as np
        from src.retrieval.feature_store import FeatureStore

        tmpdir = tempfile.mkdtemp()
        try:
            store = FeatureStore(dim=4)
            store.add("doc_x", np.array([0.5, 0.5, 0, 0]))
            store.add("doc_y", np.array([0, 0, 0.5, 0.5]))
            store.save(tmpdir)

            loaded = FeatureStore.load(tmpdir)
            assert loaded.size == 2
            hits = loaded.search(np.array([0.5, 0.5, 0, 0]), topk=1)
            assert hits[0]["doc_id"] == "doc_x"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
