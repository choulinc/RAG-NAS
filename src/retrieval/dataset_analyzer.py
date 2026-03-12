"""
Text analyzer for RAG-NAS retrieval

Scans a user-provided dataset directory to infer:
  - task type (classification, detection, segmentation, ...)
  - domain / keywords
  - number of classes
  - image statistics (size, channels, count)

This information is packaged into a DatasetProfile which drives
both the text retrieval query and the LLM template generation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# DatasetProfile
# ---------------------------------------------------------------------------

@dataclass
class ImageStats:
    """Aggregate statistics about images found in the dataset."""
    total_count: int = 0
    sample_heights: List[int] = field(default_factory=list)
    sample_widths: List[int] = field(default_factory=list)
    channels: Optional[int] = None  # 1=grayscale, 3=RGB, 4=RGBA

    @property
    def median_height(self) -> Optional[int]:
        if not self.sample_heights:
            return None
        s = sorted(self.sample_heights)
        return s[len(s) // 2]

    @property
    def median_width(self) -> Optional[int]:
        if not self.sample_widths:
            return None
        s = sorted(self.sample_widths)
        return s[len(s) // 2]


@dataclass
class DatasetProfile:
    """Complete profile of an analyzed dataset."""
    task: str  # e.g. "Image Classification", "Object Detection", ...
    domain: str  # e.g. "natural", "medical", "satellite", "cifar"
    keywords: List[str] = field(default_factory=list)
    num_classes: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    image_stats: ImageStats = field(default_factory=ImageStats)
    readme_text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_query(self) -> str:
        """Convert profile into a text query for the existing retriever."""
        parts = [self.task]
        if self.domain and self.domain != "unknown":
            parts.append(self.domain)
        if self.num_classes:
            parts.append(f"{self.num_classes} classes")
        if self.keywords:
            parts.extend(self.keywords[:5])
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ANNOTATION_EXTS_DETECTION = {".xml", ".json"}  # VOC xml or COCO json
MASK_EXTS = {".png", ".bmp", ".tif"}

KNOWN_DATASETS = {
    "cifar": "cifar",
    "cifar10": "cifar",
    "cifar100": "cifar",
    "imagenet": "imagenet",
    "imagenet16": "imagenet",
    "imagenet16-120": "imagenet",
    "coco": "coco",
    "voc": "voc",
    "pascal": "voc",
    "chest": "medical",
    "x-ray": "medical",
    "xray": "medical",
    "mvtec": "industrial",
    "cityscapes": "autonomous",
}

TASK_KEYWORDS = {
    "classification": "Image Classification",
    "classify": "Image Classification",
    "cls": "Image Classification",
    "detection": "Object Detection",
    "detect": "Object Detection",
    "det": "Object Detection",
    "segmentation": "Image Segmentation",
    "segment": "Image Segmentation",
    "seg": "Image Segmentation",
    "regression": "Regression",
}


# ---------------------------------------------------------------------------
# DatasetAnalyzer
# ---------------------------------------------------------------------------

class DatasetAnalyzer:
    """Analyze a dataset directory to produce a DatasetProfile."""

    def __init__(self, max_image_samples: int = 50):
        self.max_image_samples = max_image_samples

    # ---- public API ----

    def analyze(self, dataset_path: str) -> DatasetProfile:
        """
        Main entry point. Scans *dataset_path* and returns a DatasetProfile.

        Detection heuristics (in priority order):
          1. Explicit metadata files (data.yaml, classes.json, labels.txt)
          2. README / text files mentioning task keywords
          3. Directory structure patterns (train/class_a/... → classification)
          4. Presence of annotation files (.xml, COCO .json) → detection
          5. Presence of mask directories → segmentation
          6. Fallback to "Image Classification"
        """
        root = Path(dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Collect signals
        readme_text = self._read_readme(root)
        meta = self._read_metadata_files(root)
        dir_info = self._analyze_directory_structure(root)
        img_stats = self._sample_image_stats(root)

        # Resolve task
        task = self._infer_task(readme_text, meta, dir_info, root)

        # Resolve domain / keywords
        domain, keywords = self._infer_domain_keywords(root, readme_text, meta, dir_info)

        # Class info
        num_classes = meta.get("num_classes") or dir_info.get("num_classes")
        class_names = meta.get("class_names", []) or dir_info.get("class_names", [])

        return DatasetProfile(
            task=task,
            domain=domain,
            keywords=keywords,
            num_classes=num_classes,
            class_names=class_names,
            image_stats=img_stats,
            readme_text=readme_text[:500] if readme_text else "",
            extra={"dir_info": dir_info},
        )

    # ---- README ----

    @staticmethod
    def _read_readme(root: Path) -> str:
        for name in ["README.md", "README.txt", "README", "readme.md", "readme.txt"]:
            p = root / name
            if p.is_file():
                try:
                    return p.read_text(encoding="utf-8", errors="ignore")[:2000]
                except Exception:
                    pass
        return ""

    # ---- metadata files ----

    @staticmethod
    def _read_metadata_files(root: Path) -> Dict[str, Any]:
        """Read structured metadata files that explicitly describe the dataset."""
        meta: Dict[str, Any] = {}

        # data.yaml (YOLO style)
        for name in ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"]:
            p = root / name
            if p.is_file():
                try:
                    import yaml
                    obj = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
                    if isinstance(obj, dict):
                        if "nc" in obj:
                            meta["num_classes"] = int(obj["nc"])
                        if "names" in obj and isinstance(obj["names"], (list, dict)):
                            names = obj["names"]
                            if isinstance(names, dict):
                                names = list(names.values())
                            meta["class_names"] = [str(n) for n in names]
                            if "num_classes" not in meta:
                                meta["num_classes"] = len(meta["class_names"])
                        if "task" in obj:
                            meta["task_hint"] = str(obj["task"]).lower()
                except Exception:
                    pass

        # classes.json / labels.json
        for name in ["classes.json", "labels.json", "categories.json"]:
            p = root / name
            if p.is_file():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    if isinstance(obj, list):
                        meta["class_names"] = [str(c) if not isinstance(c, dict) else c.get("name", str(c)) for c in obj]
                        meta["num_classes"] = len(meta["class_names"])
                    elif isinstance(obj, dict):
                        meta["class_names"] = list(obj.values()) if all(isinstance(v, str) for v in obj.values()) else list(obj.keys())
                        meta["num_classes"] = len(meta["class_names"])
                except Exception:
                    pass

        # labels.txt / classes.txt
        for name in ["labels.txt", "classes.txt"]:
            p = root / name
            if p.is_file():
                try:
                    lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
                    if lines and "class_names" not in meta:
                        meta["class_names"] = lines
                        meta["num_classes"] = len(lines)
                except Exception:
                    pass

        return meta

    # ---- directory structure ----

    @staticmethod
    def _analyze_directory_structure(root: Path) -> Dict[str, Any]:
        """Infer task and class info from directory layout."""
        info: Dict[str, Any] = {
            "has_train_val": False,
            "has_class_subdirs": False,
            "has_annotations_dir": False,
            "has_masks_dir": False,
            "num_classes": None,
            "class_names": [],
        }

        children = {c.name.lower(): c for c in root.iterdir() if c.is_dir()}

        # Check for train/val/test splits
        splits_found = [s for s in ["train", "val", "valid", "validation", "test"] if s in children]
        info["has_train_val"] = len(splits_found) >= 1

        # Look for class subdirectories inside a split (classification pattern)
        split_dir = None
        for s in ["train", "val", "valid"]:
            if s in children:
                split_dir = children[s]
                break
        if split_dir is None:
            # Maybe classes are directly under root
            split_dir = root

        class_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if class_dirs:
            # Check if these dirs contain images (classification pattern)
            has_imgs = False
            for cd in class_dirs[:3]:
                for f in cd.iterdir():
                    if f.suffix.lower() in IMAGE_EXTS:
                        has_imgs = True
                        break
                if has_imgs:
                    break
            if has_imgs:
                info["has_class_subdirs"] = True
                info["num_classes"] = len(class_dirs)
                info["class_names"] = sorted([d.name for d in class_dirs])

        # Check for annotation directories (detection)
        for ann_name in ["annotations", "annotation", "labels", "annots", "Annotations"]:
            if ann_name.lower() in children:
                info["has_annotations_dir"] = True
                break

        # Check for mask directories (segmentation)
        for mask_name in ["masks", "mask", "segmentation", "seg", "SegmentationClass"]:
            if mask_name.lower() in children:
                info["has_masks_dir"] = True
                break

        return info

    # ---- image statistics ----

    def _sample_image_stats(self, root: Path) -> ImageStats:
        """Sample a few images to get size/channel stats."""
        stats = ImageStats()

        # Collect image paths (BFS, limited)
        img_paths: List[Path] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if Path(fn).suffix.lower() in IMAGE_EXTS:
                    img_paths.append(Path(dirpath) / fn)
                    if len(img_paths) >= self.max_image_samples:
                        break
            if len(img_paths) >= self.max_image_samples:
                break

        stats.total_count = self._count_images_fast(root)

        if not img_paths:
            return stats

        try:
            from PIL import Image
        except ImportError:
            return stats

        for p in img_paths[:self.max_image_samples]:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    stats.sample_widths.append(w)
                    stats.sample_heights.append(h)
                    if stats.channels is None:
                        mode_map = {"L": 1, "RGB": 3, "RGBA": 4}
                        stats.channels = mode_map.get(im.mode, 3)
            except Exception:
                continue

        return stats

    @staticmethod
    def _count_images_fast(root: Path, limit: int = 100_000) -> int:
        count = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if Path(fn).suffix.lower() in IMAGE_EXTS:
                    count += 1
                    if count >= limit:
                        return count
        return count

    # ---- task inference ----

    def _infer_task(
        self,
        readme_text: str,
        meta: Dict[str, Any],
        dir_info: Dict[str, Any],
        root: Path,
    ) -> str:
        # 1. Explicit metadata hint
        task_hint = meta.get("task_hint", "")
        if task_hint:
            for kw, task in TASK_KEYWORDS.items():
                if kw in task_hint:
                    return task

        # 2. README keyword matching
        if readme_text:
            rl = readme_text.lower()
            for kw, task in TASK_KEYWORDS.items():
                if kw in rl:
                    return task

        # 3. Mask directory → segmentation
        if dir_info.get("has_masks_dir"):
            return "Image Segmentation"

        # 4. Annotation directory → detection (check for COCO json or VOC xml)
        if dir_info.get("has_annotations_dir"):
            if self._has_detection_annotations(root):
                return "Object Detection"

        # 5. Class subdirectories → classification
        if dir_info.get("has_class_subdirs"):
            return "Image Classification"

        # 6. Folder name hints
        root_name = root.name.lower()
        for kw, task in TASK_KEYWORDS.items():
            if kw in root_name:
                return task

        # Default
        return "Image Classification"

    @staticmethod
    def _has_detection_annotations(root: Path) -> bool:
        """Check if annotation dir contains detection-format files."""
        for ann_name in ["annotations", "annotation", "labels", "annots", "Annotations"]:
            ann_dir = root / ann_name
            if ann_dir.is_dir():
                for f in ann_dir.iterdir():
                    ext = f.suffix.lower()
                    if ext == ".xml":
                        return True  # VOC format
                    if ext == ".json":
                        # Check if COCO-like
                        try:
                            obj = json.loads(f.read_text(encoding="utf-8", errors="ignore")[:5000])
                            if isinstance(obj, dict) and ("annotations" in obj or "images" in obj):
                                return True
                        except Exception:
                            pass
                    if ext == ".txt":
                        # YOLO format (class x y w h)
                        try:
                            line = f.read_text(encoding="utf-8", errors="ignore").split("\n")[0].strip()
                            parts = line.split()
                            if len(parts) == 5 and all(_is_number(p) for p in parts):
                                return True
                        except Exception:
                            pass
        return False

    # ---- domain / keywords ----

    @staticmethod
    def _infer_domain_keywords(
        root: Path,
        readme_text: str,
        meta: Dict[str, Any],
        dir_info: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        domain = "unknown"
        keywords: List[str] = []

        # From root path name
        path_parts = root.as_posix().lower()
        for dataset_kw, dom in KNOWN_DATASETS.items():
            if dataset_kw in path_parts:
                domain = dom
                keywords.append(dataset_kw)

        # From class names
        class_names = meta.get("class_names", []) or dir_info.get("class_names", [])
        if class_names:
            # Add up to 5 class names as keywords
            keywords.extend([cn.lower().replace("_", " ") for cn in class_names[:5]])

        # From README
        if readme_text:
            rl = readme_text.lower()
            for dataset_kw, dom in KNOWN_DATASETS.items():
                if dataset_kw in rl and domain == "unknown":
                    domain = dom
                    keywords.append(dataset_kw)

        # Deduplicate
        seen = set()
        unique_kw = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_kw.append(kw)

        return domain, unique_kw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
