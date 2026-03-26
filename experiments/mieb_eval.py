"""
MIEB Benchmark Evaluation for RAG-NAS AlignedEncoder.

Evaluates the AlignedEncoder on MIEB tasks using the `mteb` library
and outputs results in Paper Table 2 format.

Paper: "MIEB: Massive Image Embedding Benchmark" (arXiv:2504.10471)
Columns: Model | Retrieval | Clustering | ZeroShot Cls | Linear Probe |
         Visual STS | Doc Und. | Compositionality | VCQA | MIEB-lite

Evaluation protocol notes
--------------------------
* Image preprocessing: 224×224 centre-crop + ImageNet normalisation
  (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
  Using 32×32 CIFAR preprocessing during benchmark evaluation would
  systematically degrade high-resolution tasks (doc understanding,
  compositionality, visual STS) and make the results incomparable to
  published MIEB numbers.

* Denominator: category averages are computed over the *full* set of
  expected tasks for the selected benchmark variant (MIEB-lite or MIEB).
  Skipped/failed tasks contribute a score of 0.0 and are flagged in the
  output so reviewers can distinguish "not evaluated" from "poor score".
  This mirrors the denominator convention used in the MIEB paper.

* Benchmark variant: pass ``--benchmark MIEB-lite`` (default) or
  ``--benchmark MIEB``.  The overall column is labelled accordingly;
  outputting a single "MIEB" and a separate "MIEB-lite" column with
  the same value (as prior code did) is misleading and is now removed.

Usage:
    # MIEB-lite evaluation (recommended for development)
    python experiments/mieb_eval.py \\
        --alignment_checkpoint checkpoints/aligned_encoder.pt \\
        --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \\
        --benchmark MIEB-lite \\
        --output experiments/results_mieb_lite.csv

    # Full MIEB evaluation
    python experiments/mieb_eval.py \\
        --alignment_checkpoint checkpoints/aligned_encoder.pt \\
        --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \\
        --benchmark MIEB \\
        --output experiments/results_mieb.csv

    # Quick sanity-check with one task per category
    python experiments/mieb_eval.py \\
        --alignment_checkpoint checkpoints/aligned_encoder.pt \\
        --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \\
        --tasks quick \\
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

    # Image-only keys tried in priority order (covers I2I retrieval, clustering, linear probe)
    _IMAGE_KEYS = ("image", "images", "img", "imgs")
    # Text-only keys (covers T2T retrieval, zero-shot, STS)
    _TEXT_KEYS  = ("text", "texts", "sentence", "sentences",
                   "query", "passage", "caption", "captions",
                   "sentence_1", "sentence_2", "anchor", "positive",
                   "input", "inputs")
    # Non-data keys to skip in the fallback scan
    _SKIP_KEYS  = frozenset({"label", "labels", "cls", "y", "score",
                              "idx", "index", "id"})

    # Standard ImageNet preprocessing (consistent with training pipeline)
    _IMG_TRANSFORM = None  # lazy-init to avoid importing T at class definition time

    def _get_transform(self):
        if AlignedEncoderMTEBWrapper._IMG_TRANSFORM is None:
            import torchvision.transforms as T
            AlignedEncoderMTEBWrapper._IMG_TRANSFORM = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return AlignedEncoderMTEBWrapper._IMG_TRANSFORM

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
        Encode inputs for ANY mteb task type.

        Routing logic
        -------------
        mteb passes ``prompt_type="query"`` when encoding retrieval queries
        and ``prompt_type="passage"`` (or None) when encoding the corpus.
        We use this signal together with batch key inspection to route each
        batch to the image or text encoder:

        1. If prompt_type == "query" and batch has only text keys → text encoder.
        2. If batch has image keys (PIL.Image values) → image encoder.
        3. If batch has text keys → text encoder.
        4. Fallback: try every non-skip key, pick image encoder if PIL found.

        ImageTextPairClassification (Compositionality)
        -----------------------------------------------
        Tasks like SugarCrepe, EqBen send a batch with BOTH an 'image' and a
        'caption' / 'text' field.  mteb calls encode() twice: once for images
        (corpus) and once for texts (captions).  We handle this by checking
        prompt_type first, then falling through to per-item type detection.
        """
        import torch
        from PIL import Image as PILImage

        transform = self._get_transform()
        all_embeddings = []

        for batch in inputs:
            imgs, txts = self._split_batch(batch, prompt_type, PILImage)

            if imgs:
                tensors = torch.stack(
                    [transform(im.convert("RGB")) for im in imgs]
                ).to(self.device)
                with torch.no_grad():
                    emb = self.encoder.encode_image(tensors)
                all_embeddings.append(emb.cpu().numpy())
            elif txts:
                with torch.no_grad():
                    emb = self.encoder.encode_text(txts, device=self.device)
                all_embeddings.append(emb.cpu().numpy())
            # else: empty batch — skip

        if not all_embeddings:
            return np.zeros((0, self.encoder.shared_dim), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)

    def _split_batch(self, batch, prompt_type, PILImage):
        """Return (image_list, text_list) for a single batch.

        Exactly one of the two lists will be non-empty (or both empty for
        an unrecognised batch format).
        """
        # ── Normalise batch into a dict ──────────────────────────────────
        if isinstance(batch, dict):
            bdict = batch
        elif isinstance(batch, (list, tuple)) and batch:
            # Bare list/tuple: determine type from first element
            first = batch[0]
            if isinstance(first, PILImage.Image):
                return list(batch), []
            return [], [str(x) for x in batch]
        else:
            return [], []

        # ── prompt_type-guided routing (Any2AnyRetrieval, cross-modal) ──
        # "query" usually means text query even in I2T retrieval tasks.
        if str(prompt_type).lower() == "query":
            for k in self._TEXT_KEYS:
                if k in bdict:
                    items = bdict[k]
                    if not isinstance(items, (list, tuple)):
                        items = [items]
                    if items and not isinstance(items[0], PILImage.Image):
                        return [], [str(x) for x in items]

        # ── Check explicit image keys first ─────────────────────────────
        for k in self._IMAGE_KEYS:
            if k in bdict:
                items = bdict[k]
                if not isinstance(items, (list, tuple)):
                    items = [items]
                if items and isinstance(items[0], PILImage.Image):
                    return list(items), []

        # ── Check explicit text keys ─────────────────────────────────────
        for k in self._TEXT_KEYS:
            if k in bdict:
                items = bdict[k]
                if not isinstance(items, (list, tuple)):
                    items = [items]
                if items and not isinstance(items[0], PILImage.Image):
                    return [], [str(x) for x in items]

        # ── Fallback: scan all non-skip keys ─────────────────────────────
        for k, v in bdict.items():
            if k in self._SKIP_KEYS:
                continue
            items = v if isinstance(v, (list, tuple)) else [v]
            if not items:
                continue
            if isinstance(items[0], PILImage.Image):
                return list(items), []
            if isinstance(items[0], str):
                return [], [str(x) for x in items]

        return [], []

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

def get_available_tasks(
    benchmark_variant: str = "MIEB-lite",
    category: Optional[str] = None,
) -> list:
    """Return the task list for the requested benchmark variant.

    Args:
        benchmark_variant: "MIEB-lite" or "MIEB".  Controls which canonical
            task set is loaded so the denominator matches the paper tables.
        category: Optional category filter (one of MIEB_CATEGORIES).
    """
    try:
        import mteb
    except ImportError:
        raise RuntimeError("mteb is required. Install: pip install mteb")

    # Map friendly name → mteb benchmark identifier
    _benchmark_id = {
        "MIEB-lite": "MIEB(lite)",
        "MIEB": "MIEB",
    }.get(benchmark_variant, "MIEB(lite)")

    try:
        benchmark = mteb.get_benchmark(_benchmark_id)
        tasks = benchmark.tasks if hasattr(benchmark, "tasks") else list(benchmark)
    except Exception:
        # Fallback when mteb does not know the benchmark name
        tasks = []
        for task_type in [
            "ImageClustering", "ImageClassification", "Any2AnyRetrieval", "Retrieval",
            "ZeroShotClassification", "Compositionality", "ImageTextPairClassification",
            "DocumentUnderstanding", "VisionCentricQA", "VisualSTS", "VisualSTS(eng)",
            "VisualSTS(multi)",
        ]:
            try:
                tasks.extend(mteb.get_tasks(task_types=[task_type]))
            except Exception:
                pass

    if category:
        filtered = []
        for t in tasks:
            md = getattr(t, "metadata", {})
            tt = md.get("type", "") if isinstance(md, dict) else str(getattr(md, "type", ""))
            if TASK_TYPE_TO_CATEGORY.get(str(tt), "") == category:
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
    benchmark_variant: str = "MIEB-lite",
) -> Dict[str, Any]:
    """
    Run MIEB evaluation and return results in Paper Table 2 format.

    Args:
        skip_gated: If True, skip gated datasets that require HF authentication.
        benchmark_variant: "MIEB-lite" or "MIEB". Controls the canonical task list
            and therefore the denominator used when computing category averages.

    Denominator convention (matches MIEB paper):
        Category averages are computed over ALL tasks in the benchmark variant,
        not just the ones that were successfully evaluated.  Skipped or failed
        tasks contribute 0.0 so the number is conservative but comparable.
        The output CSV includes an "N_run/N_total" column per category so
        reviewers can see exactly which tasks were missing.

    Returns:
        Dict with category scores and the overall benchmark average.
    """
    try:
        import mteb
    except ImportError:
        raise RuntimeError("mteb is required. Install: pip install mteb")

    wrapper = AlignedEncoderMTEBWrapper(aligned_encoder, device=device)

    # Per-category accumulators: list of (score_or_None) for every expected task
    # We separate evaluated scores from the total task count per category so we
    # can compute the correct denominator later.
    category_scores: Dict[str, List[float]] = {cat: [] for cat in MIEB_CATEGORIES}
    # Track how many tasks were expected vs run per category for the coverage report
    category_total: Dict[str, int] = {cat: 0 for cat in MIEB_CATEGORIES}
    skipped_tasks: List[str] = []
    failed_tasks: List[str] = []

    if tasks and tasks[0] == "quick":
        # Quick evaluation mode — uses a fixed representative subset per category.
        # Category averages here are over the quick subset, NOT the full benchmark.
        print("\n[Quick mode] Evaluating one representative task per category.")
        for cat, task_names in QUICK_TASKS.items():
            for tname in task_names:
                category_total[cat] += 1
                try:
                    mteb_tasks = mteb.get_tasks(tasks=[tname])
                    results = mteb.evaluate(wrapper, tasks=mteb_tasks)
                    score = None
                    for r in results:
                        metric_key = CATEGORY_METRICS.get(cat, "main_score")
                        score = _extract_score(r, metric_key)
                        if score is not None:
                            break
                    if score is not None:
                        category_scores[cat].append(score)
                        print(f"  ✓ [{cat}] {tname}: {score:.4f}")
                    else:
                        category_scores[cat].append(0.0)
                        failed_tasks.append(tname)
                        print(f"  ✗ [{cat}] {tname}: score extraction failed (counted as 0)")
                except Exception as e:
                    category_scores[cat].append(0.0)
                    failed_tasks.append(tname)
                    print(f"  ✗ [{cat}] {tname}: {e} (counted as 0)")
    else:
        # Full benchmark evaluation.
        try:
            if tasks:
                mteb_tasks = mteb.get_tasks(tasks=tasks)
            else:
                mteb_tasks = get_available_tasks(benchmark_variant=benchmark_variant)

            print(f"\nBenchmark variant : {benchmark_variant}")
            print(f"Total tasks       : {len(mteb_tasks)}")

            # Build per-category expected-task counts from the canonical task list
            # BEFORE evaluation so skips/failures are reflected in the denominator.
            for task in mteb_tasks:
                md = getattr(task, "metadata", {})
                tt = md.get("type", "") if isinstance(md, dict) else str(getattr(md, "type", ""))
                cat = TASK_TYPE_TO_CATEGORY.get(str(tt), "")
                if cat:
                    category_total[cat] += 1

            # Tasks that are permanently broken upstream (HF dataset removed or
            # never properly hosted).  Counted as 0.0 in the denominator.
            # ┌─────────────────────────────────────────┬──────────────────────────────┐
            # │ Task name                               │ Reason                       │
            # ├─────────────────────────────────────────┼──────────────────────────────┤
            # │ Fashion200kI2TRetrieval                 │ HF dataset not publicly avail│
            # │ NIGHTSI2IRetrieval                      │ HF dataset removed           │
            # │ VisualSTS17Multilingual                 │ Only multilingual; no eng sub│
            # │ VisualSTS-b-Multilingual                │ Only multilingual; no eng sub│
            # └─────────────────────────────────────────┴──────────────────────────────┘
            _known_broken = {
                "Fashion200kI2TRetrieval",   # dataset not publicly hosted
                "NIGHTSI2IRetrieval",         # dataset removed from HF Hub
                "VisualSTS17Multilingual",    # no English subset; multilingual only
                "VisualSTS-b-Multilingual",   # no English subset; multilingual only
            }

            for idx, task in enumerate(mteb_tasks, 1):
                task_name = (
                    task.metadata.name if hasattr(task.metadata, "name") else str(task)
                )

                md = getattr(task, "metadata", {})
                tt = md.get("type", "") if isinstance(md, dict) else str(getattr(md, "type", ""))
                cat = TASK_TYPE_TO_CATEGORY.get(str(tt), "")

                if task_name in _known_broken:
                    print(f"\n[{idx}/{len(mteb_tasks)}] ⊗ Known broken dataset: {task_name} (counted as 0)")
                    if cat:
                        category_scores[cat].append(0.0)
                    failed_tasks.append(task_name)
                    continue

                print(f"\n[{idx}/{len(mteb_tasks)}] Evaluating: {task_name}")
                try:
                    results = mteb.evaluate(wrapper, tasks=[task])
                    score = None
                    for r in results:
                        metric_key = CATEGORY_METRICS.get(cat, "main_score") if cat else "main_score"
                        score = _extract_score(r, metric_key)
                        if score is not None:
                            break

                    if score is not None and cat:
                        category_scores[cat].append(score)
                        print(f"  ✓ Score: {score:.4f} ({metric_key})")
                    elif cat:
                        category_scores[cat].append(0.0)
                        failed_tasks.append(task_name)
                        print(f"  ✗ Score extraction failed — counted as 0.0")

                except Exception as e:
                    error_msg = str(e)
                    is_gated = (
                        "gated dataset" in error_msg.lower()
                        or "must be authenticated" in error_msg.lower()
                        or "access to its metadata" in error_msg.lower()
                        or "repository" in error_msg.lower() and "private" in error_msg.lower()
                    )
                    if is_gated and skip_gated:
                        print(
                            f"  ⊗ Gated dataset — requires HF token (counted as 0): {task_name}\n"
                            f"    Fix: huggingface-cli login  (then re-run with --no-skip-gated)"
                        )
                        skipped_tasks.append(task_name)
                    else:
                        # Print full error to make it easy to diagnose batch-format issues
                        print(f"  ✗ Error (counted as 0): {task_name}")
                        print(f"    {error_msg[:300]}")
                        failed_tasks.append(task_name)
                    # Skipped/failed tasks count as 0 in the denominator
                    if cat:
                        category_scores[cat].append(0.0)

        except Exception as e:
            print(f"Error during evaluation setup: {e}")
            import traceback
            traceback.print_exc()

    # Print summary of skipped/failed tasks
    if skipped_tasks:
        print(f"\n⊗ Skipped {len(skipped_tasks)} gated dataset(s):")
        for t in skipped_tasks:
            print(f"  - {t}")
    if failed_tasks:
        print(f"\n✗ Failed/broken {len(failed_tasks)} task(s):")
        for t in failed_tasks:
            print(f"  - {t}")

    # Compute category averages over the FULL expected task count (correct denominator).
    result_row = {"Model": model_name}
    all_scores: List[float] = []

    for cat in MIEB_CATEGORIES:
        scores = category_scores[cat]
        n_total = category_total[cat]
        if n_total == 0:
            # Category not present in this benchmark variant
            result_row[cat] = "—"
            result_row[f"{cat} (N_run/N_total)"] = "—"
        else:
            # Pad with zeros for any expected tasks that were never attempted
            # (e.g., benchmark loading failed before the per-task loop)
            while len(scores) < n_total:
                scores.append(0.0)
            avg = float(np.mean(scores)) * 100
            n_run = sum(1 for s in scores if s > 0.0)
            result_row[cat] = f"{avg:.1f}"
            result_row[f"{cat} (N_run/N_total)"] = f"{n_run}/{n_total}"
            all_scores.extend(scores)

    # Overall score: labelled after the benchmark variant to avoid confusion
    # between MIEB and MIEB-lite (they have different task sets and cannot share
    # a single number).
    overall_key = benchmark_variant  # e.g., "MIEB-lite" or "MIEB"
    if all_scores:
        result_row[overall_key] = f"{float(np.mean(all_scores)) * 100:.1f}"
    else:
        result_row[overall_key] = "—"

    _print_table(result_row, benchmark_variant=benchmark_variant)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        coverage_cols = [f"{cat} (N_run/N_total)" for cat in MIEB_CATEGORIES]
        fieldnames = ["Model"] + MIEB_CATEGORIES + coverage_cols + [overall_key]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(result_row)
        print(f"\nResults saved → {output_path}")

    return result_row


def _extract_score(result, metric_key: str) -> Optional[float]:
    """Extract a score from an mteb result object.

    Search order:
      1. Preferred split ("test", then any other split).
      2. Within a split: primary metric_key → "main_score" → any numeric value.

    This handles the diversity of mteb result shapes across task types
    (retrieval ndcg_at_10, clustering v_measure, STS cosine_spearman, etc.).
    """
    try:
        if not hasattr(result, "scores"):
            return None

        scores_dict = result.scores
        if not scores_dict:
            return None

        # Prefer the test split; fall back to the first available split.
        preferred_splits = ["test"] + [s for s in scores_dict if s != "test"]

        for split_name in preferred_splits:
            split_scores = scores_dict.get(split_name)
            if not split_scores:
                continue
            for score_dict in split_scores:
                if not isinstance(score_dict, dict):
                    continue
                # 1. Exact metric key
                if metric_key in score_dict:
                    v = score_dict[metric_key]
                    if isinstance(v, (int, float)):
                        return float(v)
                # 2. main_score fallback
                if "main_score" in score_dict:
                    v = score_dict["main_score"]
                    if isinstance(v, (int, float)):
                        return float(v)
                # 3. Any numeric scalar key (last resort)
                for k, v in score_dict.items():
                    if isinstance(v, (int, float)) and k not in ("num_samples",):
                        return float(v)
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


def _print_table(row: Dict[str, str], benchmark_variant: str = "MIEB-lite"):
    """Pretty-print results in Paper Table 2 format."""
    print("\n" + "=" * 100)
    print(f"MIEB Benchmark Results — {benchmark_variant} (Paper Table 2 Format)")
    print("=" * 100)

    cols = ["Model"] + MIEB_CATEGORIES + [benchmark_variant]
    widths = [max(len(c), len(str(row.get(c, "—")))) + 2 for c in cols]

    header = " | ".join(c.center(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    values = " | ".join(str(row.get(c, "—")).center(w) for c, w in zip(cols, widths))
    print(values)
    print("=" * 100)

    # Coverage summary (N_run / N_total per category)
    print("\nTask coverage (N_run / N_total):")
    for cat in MIEB_CATEGORIES:
        cov = row.get(f"{cat} (N_run/N_total)", "—")
        print(f"  {cat:<20} {cov}")


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
    ap.add_argument("--benchmark", default="MIEB-lite", choices=["MIEB-lite", "MIEB"],
                    help="Benchmark variant to evaluate. 'MIEB-lite' (default) uses the "
                         "reduced canonical task set; 'MIEB' uses the full task set. "
                         "The overall score column is labelled accordingly.")
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
        benchmark_variant=args.benchmark,
    )


if __name__ == "__main__":
    main()
