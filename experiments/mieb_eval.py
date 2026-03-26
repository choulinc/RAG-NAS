"""
MIEB Benchmark Evaluation for RAG-NAS AlignedEncoder.

Evaluates the AlignedEncoder on MIEB tasks using the `mteb` library
and outputs results in Paper Table 2 format.

Paper: "MIEB: Massive Image Embedding Benchmark" (arXiv:2504.10471)
Columns: Model | Retrieval | Clustering | ZeroShot Cls | Linear Probe |
         Visual STS | Doc Und. | Compositionality | VCQA | MIEB | MIEB-lite

Usage:
    # Full MIEB evaluation
    python experiments/mieb_eval.py \
        --alignment_checkpoint checkpoints/aligned_encoder.pt \
        --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
        --output experiments/results.csv

    # Quick test with subset
    python experiments/mieb_eval.py \
        --alignment_checkpoint checkpoints/aligned_encoder.pt \
        --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
        --tasks quick \
        --output experiments/results_quick.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Paper Table 2 column definitions
# ---------------------------------------------------------------------------

# The 8 MIEB categories from Table 2 of the paper
MIEB_CATEGORIES = [
    "Retrieval",
    "Clustering",
    "ZeroShot Cls",
    "Linear Probe",
    "Visual STS",
    "Doc Und.",
    "Compositionality",
    "VCQA",
]

# Mapping from mteb task type → MIEB category
TASK_TYPE_TO_CATEGORY = {
    "Retrieval": "Retrieval",
    "Any2AnyRetrieval": "Retrieval",
    "Any2AnyMultilingualRetrieval": "Retrieval",
    "ImageClustering": "Clustering",
    "Clustering": "Clustering",
    "ImageClassification": "Linear Probe",
    "Classification": "Linear Probe",
    "ZeroShotClassification": "ZeroShot Cls",
    "ZeroshotClassification": "ZeroShot Cls",
    "VisualSTS(eng)": "Visual STS",
    "VisualSTS(multi)": "Visual STS",
    "VisualSTS": "Visual STS",
    "DocumentUnderstanding": "Doc Und.",
    "Compositionality": "Compositionality",
    "ImageTextPairClassification": "Compositionality",
    "VisionCentricQA": "VCQA",
    "Any2AnyMultiChoice": "VCQA",
}

# Category → primary metric (from paper)
CATEGORY_METRICS = {
    "Retrieval": "ndcg_at_10",
    "Clustering": "v_measure",
    "ZeroShot Cls": "accuracy",
    "Linear Probe": "accuracy",
    "Visual STS": "cosine_spearman",
    "Doc Und.": "ndcg_at_5",
    "Compositionality": "accuracy",
    "VCQA": "ndcg_at_10",
}

# Quick evaluation: small representative tasks per category (actual MIEB task names)
QUICK_TASKS = {
    "Clustering": [
        "ImageNetDog15Clustering",
    ],
    "Linear Probe": [
        "Country211",
    ],
    "Retrieval": [
        "CUB200I2IRetrieval",
    ],
    "ZeroShot Cls": [
        "CIFAR100ZeroShot",
    ],
    "Visual STS": [
        "STS13VisualSTS",
    ],
    "Compositionality": [
        "Winoground",
    ],
}


# ---------------------------------------------------------------------------
# MTEB Model Wrapper for AlignedEncoder
# ---------------------------------------------------------------------------

class AlignedEncoderMTEBWrapper:
    """
    Wraps AlignedEncoder to be compatible with the mteb evaluation API.

    mteb expects:
        model.encode(images_or_texts, ...) → np.ndarray
    """

    def __init__(self, aligned_encoder, device: str = "cpu"):
        self.encoder = aligned_encoder
        self.device = device
        self.encoder.to(device)

    @property
    def mteb_model_meta(self):
        """Return ModelMeta declaring support for both image and text."""
        from mteb.models import ModelMeta
        from datetime import date

        # Use a lambda loader function as required by mteb
        def custom_loader():
            return self

        return ModelMeta(
            loader=custom_loader,
            name="anthropic/RAG-NAS-AlignedEncoder",
            revision="v0.1.0",
            release_date=date.today(),
            n_parameters=sum(p.numel() for p in self.encoder.parameters()) // 1_000_000,
            memory_usage_mb=1024,  # Rough estimate
            max_tokens=512,
            embed_dim=self.encoder.shared_dim,
            license="apache-2.0",  # lowercase as per enum
            open_weights=True,
            public_training_code="https://github.com/anthropics/RAG-NAS",  # string URL required
            public_training_data=True,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=set(),  # set, not dict
            languages=["eng"],
            modalities=["image", "text"],  # Key: support both modalities
        )

    def encode(
        self,
        inputs,  # DataLoader[BatchedInput] in mteb 2.x
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode inputs using the new mteb 2.x API.

        Args:
            inputs: DataLoader yielding batches of images or texts
            task_metadata: Task metadata from mteb
            hf_split: HuggingFace dataset split
            hf_subset: HuggingFace dataset subset
            prompt_type: Query or passage (for retrieval tasks)
            **kwargs: Additional encoding arguments
        """
        import torch
        import torchvision.transforms as T
        from PIL import Image

        all_embeddings = []

        # inputs is a DataLoader in mteb 2.x - it yields batches
        for batch in inputs:
            # batch is a dict like {'image': [PIL.Image, ...], 'cls': tensor}
            # or {'text': [...], ...} depending on the task

            if isinstance(batch, dict):
                # Try to extract the data - check for common keys
                data = None
                for key in ['image', 'images', 'text', 'texts', 'input', 'inputs', 'sentence']:
                    if key in batch:
                        data = batch[key]
                        break

                if data is None:
                    # Fallback: use the first non-label value
                    for k, v in batch.items():
                        if k not in ['label', 'labels', 'cls', 'y']:
                            data = v
                            break
            elif isinstance(batch, (list, tuple)):
                data = batch
            else:
                data = [batch]

            if data is None or (isinstance(data, (list, tuple)) and len(data) == 0):
                continue

            # Ensure data is a list
            if not isinstance(data, (list, tuple)):
                data = [data]

            # Get first item to determine type
            first_item = data[0]

            # Detect if image or text
            is_image = isinstance(first_item, Image.Image)

            if is_image:
                # Encode image batch
                transform = T.Compose([
                    T.Resize((32, 32)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    ),
                ])
                tensors = torch.stack([transform(img.convert("RGB")) for img in data])
                tensors = tensors.to(self.device)
                with torch.no_grad():
                    emb = self.encoder.encode_image(tensors)
                all_embeddings.append(emb.cpu().numpy())
            else:
                # Encode text batch
                texts = [str(t) for t in data]
                with torch.no_grad():
                    emb = self.encoder.encode_text(texts, device=self.device)
                all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            return np.zeros((0, self.encoder.shared_dim), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings."""
        # Normalize embeddings
        embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity
        return embeddings1_norm @ embeddings2_norm.T

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity between corresponding embeddings."""
        # Normalize
        embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        # Element-wise dot product
        return np.sum(embeddings1_norm * embeddings2_norm, axis=1)


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def get_available_tasks(category: Optional[str] = None) -> list:
    """Get available MIEB tasks from mteb, optionally filtered by category."""
    try:
        import mteb
    except ImportError:
        raise RuntimeError("mteb is required. Install: pip install mteb")

    # Try to get MIEB benchmark tasks
    try:
        benchmark = mteb.get_benchmark("MIEB(lite)")
        tasks = benchmark.tasks if hasattr(benchmark, "tasks") else list(benchmark)
    except Exception:
        # Fallback: get individual tasks by type
        tasks = []
        for task_type in ["ImageClustering", "ImageClassification", "Any2AnyRetrieval", "Retrieval",
                          "ZeroShotClassification", "Compositionality", "ImageTextPairClassification",
                          "DocumentUnderstanding", "VisionCentricQA", "VisualSTS", "VisualSTS(eng)", "VisualSTS(multi)"]:
            try:
                found = mteb.get_tasks(task_types=[task_type])
                tasks.extend(found)
            except Exception:
                pass

    if category:
        filtered = []
        for t in tasks:
            task_type = getattr(t, "metadata", {})
            if isinstance(task_type, dict):
                tt = task_type.get("type", "")
            else:
                tt = getattr(task_type, "type", "")
            mapped = TASK_TYPE_TO_CATEGORY.get(str(tt), "")
            if mapped == category:
                filtered.append(t)
        return filtered

    return list(tasks)


def run_evaluation(
    aligned_encoder,
    device: str = "cpu",
    tasks: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    model_name: str = "RAG-NAS-AlignedEncoder",
    skip_gated: bool = True,
) -> Dict[str, Any]:
    """
    Run MIEB evaluation and return results in Paper Table 2 format.

    Args:
        skip_gated: If True, skip gated datasets that require authentication

    Returns:
        Dict with category scores and overall MIEB/MIEB-lite averages.
    """
    try:
        import mteb
    except ImportError:
        raise RuntimeError("mteb is required. Install: pip install mteb")

    wrapper = AlignedEncoderMTEBWrapper(aligned_encoder, device=device)

    # Collect results by category
    category_scores: Dict[str, List[float]] = {cat: [] for cat in MIEB_CATEGORIES}
    skipped_tasks = []
    failed_tasks = []

    if tasks and tasks[0] == "quick":
        # Quick evaluation mode
        for cat, task_names in QUICK_TASKS.items():
            try:
                mteb_tasks = mteb.get_tasks(tasks=task_names)
                results = mteb.evaluate(wrapper, tasks=mteb_tasks)
                for r in results:
                    # Extract primary metric
                    metric_key = CATEGORY_METRICS.get(cat, "main_score")
                    score = _extract_score(r, metric_key)
                    if score is not None:
                        category_scores[cat].append(score)
            except Exception as e:
                print(f"  Warning: Could not evaluate {cat}: {e}")
                failed_tasks.extend(task_names)
    else:
        # Full evaluation - evaluate tasks one by one to handle errors gracefully
        try:
            if tasks:
                mteb_tasks = mteb.get_tasks(tasks=tasks)
            else:
                mteb_tasks = get_available_tasks()

            print(f"\n📋 Total tasks to evaluate: {len(mteb_tasks)}")

            for idx, task in enumerate(mteb_tasks, 1):
                task_name = task.metadata.name if hasattr(task.metadata, 'name') else str(task)
                
                if task_name in ["Fashion200kI2TRetrieval", "NIGHTSI2IRetrieval", "VisualSTS17Multilingual", "VisualSTS-b-Multilingual"]:
                    print(f"\n[{idx}/{len(mteb_tasks)}] ⊗ Skipped (Known Missing/Broken HF Dataset): {task_name}")
                    continue

                print(f"\n[{idx}/{len(mteb_tasks)}] Evaluating: {task_name}")

                try:
                    # Evaluate single task
                    results = mteb.evaluate(wrapper, tasks=[task])

                    for r in results:
                        task_type = _get_task_type(r)
                        cat = TASK_TYPE_TO_CATEGORY.get(task_type, "")
                        if cat:
                            metric_key = CATEGORY_METRICS.get(cat, "main_score")
                            score = _extract_score(r, metric_key)
                            if score is not None:
                                category_scores[cat].append(score)
                                print(f"  ✓ Score: {score:.4f} ({metric_key})")

                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a gated dataset error
                    if "gated dataset" in error_msg.lower() or "must be authenticated" in error_msg.lower():
                        if skip_gated:
                            print(f"  ⊗ Skipped (gated dataset): {task_name}")
                            skipped_tasks.append(task_name)
                            continue

                    # Other errors
                    print(f"  ⊗ Skipped (Benchmark Load Error): {task_name}")
                    print(f"    Reason: {error_msg}")
                    failed_tasks.append(task_name)

        except Exception as e:
            print(f"  Error during evaluation setup: {e}")
            import traceback
            traceback.print_exc()

    # Print summary of skipped/failed tasks
    if skipped_tasks:
        print(f"\n⊗ Skipped {len(skipped_tasks)} gated dataset(s):")
        for task in skipped_tasks:
            print(f"  - {task}")

    if failed_tasks:
        print(f"\n✗ Failed {len(failed_tasks)} task(s):")
        for task in failed_tasks:
            print(f"  - {task}")

    # Compute category averages
    result_row = {"Model": model_name}
    all_scores = []

    for cat in MIEB_CATEGORIES:
        scores = category_scores[cat]
        if scores:
            avg = np.mean(scores) * 100  # Convert to percentage
            result_row[cat] = f"{avg:.1f}"
            all_scores.extend(scores)
        else:
            result_row[cat] = "—"

    # Overall MIEB / MIEB-lite score
    if all_scores:
        overall = np.mean(all_scores) * 100
        result_row["MIEB"] = f"{overall:.1f}"
        result_row["MIEB-lite"] = f"{overall:.1f}"
    else:
        result_row["MIEB"] = "—"
        result_row["MIEB-lite"] = "—"

    # Print table
    _print_table(result_row)

    # Save CSV
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fieldnames = ["Model"] + MIEB_CATEGORIES + ["MIEB", "MIEB-lite"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(result_row)
        print(f"\nResults saved → {output_path}")

    return result_row


def _extract_score(result, metric_key: str) -> Optional[float]:
    """Extract a score from an mteb result object."""
    try:
        if hasattr(result, "scores"):
            # mteb result format
            for split_name, split_scores in result.scores.items():
                for score_dict in split_scores:
                    if metric_key in score_dict:
                        return float(score_dict[metric_key])
                    if "main_score" in score_dict:
                        return float(score_dict["main_score"])
        return None
    except Exception:
        return None


def _get_task_type(result) -> str:
    """Get the task type string from an mteb result."""
    try:
        if hasattr(result, "task_name"):
            # Look up from task metadata
            import mteb
            tasks = mteb.get_tasks(tasks=[result.task_name])
            if tasks:
                t = tasks[0]
                md = getattr(t, "metadata", {})
                if isinstance(md, dict):
                    return md.get("type", "")
                return str(getattr(md, "type", ""))
        return ""
    except Exception:
        return ""


def _print_table(row: Dict[str, str]):
    """Pretty-print results in Paper Table 2 format."""
    print("\n" + "=" * 90)
    print("MIEB Benchmark Results (Paper Table 2 Format)")
    print("=" * 90)

    # Header
    cols = ["Model"] + MIEB_CATEGORIES + ["MIEB", "MIEB-lite"]
    widths = [max(len(c), len(str(row.get(c, "—")))) + 2 for c in cols]

    header = " | ".join(c.center(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    # Data row
    values = " | ".join(str(row.get(c, "—")).center(w) for c, w in zip(cols, widths))
    print(values)
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MIEB Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (skips gated datasets by default)
  python experiments/mieb_eval.py \\
      --alignment_checkpoint checkpoints/aligned_encoder.pt \\
      --encoder_checkpoint checkpoints/contrastive_encoder_best.pt

  # Include gated datasets (requires HuggingFace authentication)
  python experiments/mieb_eval.py \\
      --alignment_checkpoint checkpoints/aligned_encoder.pt \\
      --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \\
      --no-skip-gated

Note: Gated datasets like 'facebook/winoground' require:
  1. HuggingFace account with access granted to the dataset
  2. Authentication via: huggingface-cli login
        """
    )
    ap.add_argument("--alignment_checkpoint", required=True,
                    help="Path to alignment checkpoint (.pt)")
    ap.add_argument("--encoder_checkpoint", required=True,
                    help="Path to SiameseEncoder checkpoint (.pt)")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Task names to evaluate. Use 'quick' for fast subset.")
    ap.add_argument("--output", default="experiments/results.csv",
                    help="Output CSV path")
    ap.add_argument("--model_name", default="RAG-NAS-AlignedEncoder",
                    help="Model name for the results table")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-gated", dest="skip_gated", action="store_true", default=True,
                    help="Skip gated datasets that require authentication (default: True)")
    ap.add_argument("--no-skip-gated", dest="skip_gated", action="store_false",
                    help="Attempt to evaluate gated datasets (requires HF auth)")
    args = ap.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            print(f"Warning: Could not check CUDA availability ({e}). Defaulting to CPU.")
            args.device = "cpu"

    # Load aligned encoder
    from src.retrieval.alignment import AlignmentTrainer
    aligned = AlignmentTrainer.load_aligned_encoder(
        alignment_path=args.alignment_checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        device=args.device,
    )

    run_evaluation(
        aligned_encoder=aligned,
        device=args.device,
        tasks=args.tasks,
        output_path=args.output,
        model_name=args.model_name,
        skip_gated=args.skip_gated,
    )


if __name__ == "__main__":
    main()
