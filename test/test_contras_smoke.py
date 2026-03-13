import os
import json
import torch
from pathlib import Path
from PIL import Image
import pytest
from unittest.mock import patch, MagicMock

import sys

# Add project root to path for local execution
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.dataset_analyzer import DatasetAnalyzer
from src.retrieval.contrastive_encoder import SiameseEncoder, ContrastiveLoss, ContrastivePairDataset
from src.retrieval.multimodal_retrieve import MultiModalRetriever

def create_dummy_image(path: Path):
    """Creates a small 32x32 RGB dummy image."""
    img = Image.new('RGB', (32, 32), color=(73, 109, 137))
    img.save(path)

@pytest.fixture
def mock_datasets(tmp_path):
    """
    Creates multiple mock dataset structures for testing.
    Returns a dictionary mapping dataset type to its root path.
    """
    datasets = {}

    # 1. Image Classification: train/class_a/, train/class_b/
    cls_dir = tmp_path / "classification"
    cls_dir.mkdir()
    for split in ["train", "val"]:
        for cls_name in ["cats", "dogs"]:
            d = cls_dir / split / cls_name
            d.mkdir(parents=True)
            create_dummy_image(d / "img1.jpg")
            create_dummy_image(d / "img2.jpg")
    datasets["classification"] = cls_dir

    # 2. Object Detection: images/ and labels/ + data.yaml
    det_dir = tmp_path / "detection"
    det_dir.mkdir()
    (det_dir / "images").mkdir()
    (det_dir / "labels").mkdir()
    create_dummy_image(det_dir / "images" / "img1.jpg")
    create_dummy_image(det_dir / "images" / "img2.jpg")
    (det_dir / "labels" / "img1.txt").write_text("0 0.5 0.5 1.0 1.0")
    (det_dir / "data.yaml").write_text("names: ['car', 'pedestrian']")
    datasets["detection"] = det_dir

    # 3. Image Segmentation: images/ and masks/
    seg_dir = tmp_path / "segmentation"
    seg_dir.mkdir()
    (seg_dir / "images").mkdir()
    (seg_dir / "masks").mkdir()
    create_dummy_image(seg_dir / "images" / "img1.jpg")
    create_dummy_image(seg_dir / "masks" / "img1.png")
    datasets["segmentation"] = seg_dir

    # 4. Text-heavy: Only images but with a rich readme
    txt_dir = tmp_path / "textual"
    txt_dir.mkdir()
    create_dummy_image(txt_dir / "img1.jpg")
    (txt_dir / "readme.md").write_text("This dataset is about autonomous driving in winter conditions. keywords: snow, driving, autonomous.")
    datasets["textual"] = txt_dir

    return datasets

def test_dataset_analyzer(mock_datasets):
    analyzer = DatasetAnalyzer()

    # Classification
    prof = analyzer.analyze(str(mock_datasets["classification"]))
    print(f"\n[Classification Dataset]: {mock_datasets['classification']}")
    print(f" -> Inferred Task: {prof.task}, Domain: {prof.domain}, Classes: {prof.class_names}")
    assert prof.task == "Image Classification"
    assert "cats" in prof.class_names or "dogs" in prof.class_names

    # Detection
    prof = analyzer.analyze(str(mock_datasets["detection"]))
    print(f"\n[Detection Dataset]: {mock_datasets['detection']}")
    print(f" -> Inferred Task: {prof.task}, Domain: {prof.domain}")
    assert prof.task == "Object Detection"

    # Segmentation
    prof = analyzer.analyze(str(mock_datasets["segmentation"]))
    print(f"\n[Segmentation Dataset]: {mock_datasets['segmentation']}")
    print(f" -> Inferred Task: {prof.task}, Domain: {prof.domain}")
    assert prof.task == "Image Segmentation"

    # Text-heavy
    prof = analyzer.analyze(str(mock_datasets["textual"]))
    print(f"\n[Textual Dataset]: {mock_datasets['textual']}")
    print(f" -> Inferred Task: {prof.task}, Domain: {prof.domain}, Keywords: {prof.keywords}")
    # Based on the heuristic, missing specific annotations falls back to Classification if there are images.
    # But it should extract keywords.
    assert prof.task == "Image Classification" # Fallback task
    assert "autonomous" in prof.readme_text.lower()

def test_contrastive_encoder(mock_datasets):
    # Setup mock tasks map
    task_map = {
        "classification": [str(p) for p in mock_datasets["classification"].rglob("*.jpg")],
        "detection": [str(p) for p in mock_datasets["detection"].rglob("*.jpg")]
    }

    dataset = ContrastivePairDataset(task_to_images=task_map, pairs_per_epoch=10, image_size=32, augment=False)
    assert len(dataset) == 10

    # Ensure getitem works
    img1, img2, label = dataset[0]
    assert img1.shape == (3, 32, 32)
    assert img2.shape == (3, 32, 32)
    assert isinstance(label, torch.Tensor)

    # Test Encoder
    encoder = SiameseEncoder(pretrained=False, embed_dim=16) # use untrained, small dim for speed
    encoder.eval() # fix batchnorm error on batch size 1
    v1, v2 = encoder.encode_pair(img1.unsqueeze(0), img2.unsqueeze(0))
    assert v1.shape == (1, 16)
    assert v2.shape == (1, 16)

    # Test Loss
    loss_fn = ContrastiveLoss()
    loss = loss_fn(v1, v2, label.unsqueeze(0))
    assert loss.dim() == 0 # scalar
    assert not torch.isnan(loss)

@patch('src.retrieval.multimodal_retrieve.text_retrieve')
def test_multimodal_retriever(mock_text_retrieve, mock_datasets, tmp_path):
    # Mock the text retrieve to return a dummy hit
    mock_text_retrieve.return_value = [
        {"doc_id": "dummy_hit", "score": 8.5, "meta": "test"}
    ]

    # Initialize MM Retriever without image store first
    retriever = MultiModalRetriever(uir_path="dummy_uir.json")

    # Run retrieve on the textual dataset
    result = retriever.retrieve(dataset_path=str(mock_datasets["textual"]), topk=1)

    assert "profile" in result
    assert "hits" in result
    assert len(result["hits"]) == 1
    hit = result["hits"][0]
    assert hit["doc_id"] == "dummy_hit"
    assert mock_text_retrieve.called

    # Since there's no feature store initialized, image scores won't be calculated
    assert hit.get("image_score") == 0.0
    # Also verify alpha adjustment worked and hit holds multimodal scores

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
