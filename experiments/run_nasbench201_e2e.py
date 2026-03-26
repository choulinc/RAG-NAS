import argparse
import os
import sys
import json
import csv
import shutil
import tempfile
import random
from pathlib import Path
import numpy as np
import torch

# Ensure project root in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.dataset_analyzer import DatasetAnalyzer
from src.retrieval.retrieve import retrieve as text_retrieve
from src.retrieval.feature_store import FeatureStore
from src.retrieval.contrastive_encoder import SiameseEncoder, ImageRetriever, ContrastiveTrainer
from src.retrieval.llm_template_generator import get_template_generator
from src.nas.nasbench201_evaluator import NASBench201Evaluator
from src.nas.evolutionary_search import REA, gene_to_string


def create_mock_dataset(root: str) -> str:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# CIFAR-100 Subset\nA subset of CIFAR-100 for image classification.\n"
        "Contains 5 classes with 32x32 RGB images.\n"
    )
    class_names = ["airplane", "automobile", "bird", "cat", "deer"]
    (root / "labels.txt").write_text("\n".join(class_names) + "\n")
    try:
        from PIL import Image
        for cls in class_names:
            cls_dir = root / "train" / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                img = Image.new("RGB", (32, 32), color=(
                    random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                ))
                img.save(str(cls_dir / f"{i:03d}.png"))
    except ImportError:
        for cls in class_names:
            (root / "train" / cls).mkdir(parents=True, exist_ok=True)
    return str(root)

def create_mock_uir(uir_path: str) -> str:
    entries = [
        {"doc_id": "resnet50_cifar100", "name": "ResNet-50 on CIFAR-100", "results": [{"task": "Image Classification", "dataset": "CIFAR-100"}]},
        {"doc_id": "vit_base_imagenet", "name": "ViT-Base on ImageNet", "results": [{"task": "Image Classification", "dataset": "ImageNet-1k"}]},
        {"doc_id": "convnext_tiny_cifar10", "name": "ConvNeXt-Tiny on CIFAR-10", "results": [{"task": "Image Classification", "dataset": "CIFAR-10"}]},
    ]
    Path(uir_path).parent.mkdir(parents=True, exist_ok=True)
    with open(uir_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return uir_path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training; acceptable for NAS search)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_nasbench201_evaluation(seed: int = 42) -> dict:
    """Run one complete RAG-NAS trial and return the result metrics dict."""
    seed_everything(seed)
    tmpdir = tempfile.mkdtemp(prefix="ragnas_nb201_")

    try:
        print("========================================================================")
        print(f"  RAG-NAS: NAS-Bench-201 Evaluation  (seed={seed})")
        print("========================================================================")

        # 1. Dataset Analysis
        print("\n\n[1/5] Analyzing Input Dataset...")
        dataset_dir = create_mock_dataset(os.path.join(tmpdir, "my_dataset"))
        analyzer = DatasetAnalyzer()
        profile = analyzer.analyze(dataset_dir)
        query = profile.to_query()
        print(f"      Generated Query: '{query}'")

        # 2. Text and Image Retrieval
        print("\n\n[2/5] Running Multimodal Retrieval...")
        
        real_uir_path = os.path.join(PROJECT_ROOT, "data", "processed", "uir", "mmpretrain_uir.jsonl")
        if os.path.exists(real_uir_path):
            print(f"      Using real UIR text database from {real_uir_path}")
            uir_path = real_uir_path
        else:
            print(f"      Real UIR not found, generating mock UIR database...")
            uir_path = create_mock_uir(os.path.join(tmpdir, "uir", "mock_uir.jsonl"))

        text_hits = text_retrieve(uir_path=uir_path, query=query, topk=5)
        print(f"      Text Retrieval found {len(text_hits)} templates.")

        # Note: In a full rigorous pipeline we'd fuse with Image Retrieval.
        # However, for EA generation the LLM template generator primarily consumes `text_hits`
        # so we proceed directly to LLM Template Generation to save unnecessary compute, 
        # or we just pass the text hits.
        
        # 3. LLM Template Generation
        print("\n\n[3/5] Generating Evolutionary Search Templates (Qwen Local LLM)...")
        use_local_llm = torch.cuda.is_available()
        if use_local_llm:
            print("      CUDA available. Using Qwen Local LLM.")
        else:
            print("      CUDA NOT available! This might fail or use mock local if not properly configured.")
            
        generator = get_template_generator(use_local=True) # Forces our Qwen local model
        templates = generator.generate_templates(query, text_hits, profile=profile)
        if not isinstance(templates, list):
            templates = [templates]
        print(f"      LLM Generated {len(templates)} architectural templates.")

        # 4. Evolutionary Search
        print("\n\n[4/5] Running Evolutionary Search on NAS-Bench-201...")
        import yaml
        ea_config_path = os.path.join(PROJECT_ROOT, "src", "nas", "ea_config.yaml")
        if os.path.exists(ea_config_path):
            with open(ea_config_path, "r", encoding="utf-8") as f:
                ea_config = yaml.safe_load(f).get("ea", {})
        else:
            ea_config = {"pop_size": 20, "sample_size": 10, "cycles": 50}

        nb201_api_path = os.path.join(PROJECT_ROOT, "data", "NAS-Bench", "NAS-Bench-201-v1_1-096897.pth")
        if not os.path.exists(nb201_api_path):
            raise FileNotFoundError(f"NAS-Bench-201 Database not found at {nb201_api_path}")

        print("      Loading NAS-Bench-201 API...")
        evaluator = NASBench201Evaluator(nb201_api_path)

        # Search uses CIFAR-100 *validation* accuracy as fitness.
        # The test set is never queried during search; it is held out for
        # final reporting in step 5 below.
        ea = REA(
            templates,
            evaluator,
            ea_config=ea_config,
            search_dataset="cifar100",
            search_metric="x-valid",   # validation metric — NOT test set
            seed=seed,
        )
        ea.initialize_population()
        best_gene, best_fit = ea.run()
        best_arch_str = gene_to_string(best_gene)

        print(f"\n      Found Best Architecture (CIFAR-100 valid proxy): {best_arch_str}")
        print(f"      Proxy Validation Accuracy: {best_fit:.2f}%")

        # 5. Full NAS-Bench-201 Evaluation for target datasets
        print("\n\n[5/5] Extracting true NAS-Bench-201 Test/Valid Metrics...")
        # NAS-Bench-201 metric mapping -> (valid, test)
        # cifar10-valid -> (x-valid, ori-test)
        # cifar100 -> (x-valid, x-test)
        # ImageNet16-120 -> (x-valid, x-test)
        
        cifar10_valid = evaluator.evaluate(best_arch_str, dataset="cifar10-valid", metric="x-valid")
        cifar10_test = evaluator.evaluate(best_arch_str, dataset="cifar10-valid", metric="ori-test")
        cifar100_valid = evaluator.evaluate(best_arch_str, dataset="cifar100", metric="x-valid")
        cifar100_test = evaluator.evaluate(best_arch_str, dataset="cifar100", metric="x-test")
        in16_valid = evaluator.evaluate(best_arch_str, dataset="ImageNet16-120", metric="x-valid")
        in16_test = evaluator.evaluate(best_arch_str, dataset="ImageNet16-120", metric="x-test")

        # ── Selection-protocol metadata ─────────────────────────────────────
        # Search used CIFAR-100 *validation* accuracy as fitness signal.
        # The architecture was committed BEFORE querying any test numbers.
        # Multi-dataset test results are read from the benchmark table AFTER
        # commitment — they are not used for selection.
        SEARCH_DATASET = "cifar100"
        SEARCH_METRIC  = "x-valid"

        print("========================================================================")
        print("  FINAL NAS-BENCH-201 RESULTS")
        print(f"  Selection protocol: fitness = {SEARCH_DATASET} / {SEARCH_METRIC}")
        print("  Test numbers are lookup-only; they do NOT influence selection.")
        print("========================================================================")
        print(f" Method          | CIFAR-10      | CIFAR-100     | ImageNet-16-120")
        print(f"                 | valid | test  | valid | test  | valid | test")
        print("------------------------------------------------------------------------")
        row = f" RAG-NAS (Qwen)  | {cifar10_valid:5.2f} | {cifar10_test:5.2f} | {cifar100_valid:5.2f} | {cifar100_test:5.2f} | {in16_valid:5.2f} | {in16_test:5.2f}"
        print(row)
        print("========================================================================")
        print(f"  Best arch : {best_arch_str}")
        print(f"  Seed      : {seed}")

        trial_metrics = {
            "seed": seed,
            "search_dataset": SEARCH_DATASET,
            "search_metric":  SEARCH_METRIC,
            "best_arch": best_arch_str,
            "cifar10_valid": cifar10_valid,
            "cifar10_test": cifar10_test,
            "cifar100_valid": cifar100_valid,
            "cifar100_test": cifar100_test,
            "in16_valid": in16_valid,
            "in16_test": in16_test,
        }
        return trial_metrics

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _aggregate_trials(all_metrics: list) -> dict:
    """Compute mean ± std across trials for each numeric metric."""
    keys = ["cifar10_valid", "cifar10_test", "cifar100_valid",
            "cifar100_test", "in16_valid", "in16_test"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="RAG-NAS end-to-end evaluation on NAS-Bench-201"
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Base random seed (default: 42).")
    ap.add_argument("--trials", type=int, default=1,
                    help="Number of independent trials to run. Each trial uses "
                         "seed + trial_index so results are reproducible. "
                         "Use ≥3 to report mean ± std (required for top venues).")
    args = ap.parse_args()

    all_metrics = []
    for trial_idx in range(args.trials):
        trial_seed = args.seed + trial_idx
        print(f"\n{'='*72}")
        print(f"  Trial {trial_idx + 1}/{args.trials}  (seed={trial_seed})")
        print(f"{'='*72}")
        metrics = run_nasbench201_evaluation(seed=trial_seed)
        all_metrics.append(metrics)

    # ── Single-trial output ──────────────────────────────────────────────────
    if args.trials == 1:
        m = all_metrics[0]
        output_csv = os.path.join(PROJECT_ROOT, "experiments", "results_nasbench201.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        # Include search_dataset / search_metric so readers can verify the
        # selection protocol.  The test columns are lookup-only; they are NOT
        # used for selection and do NOT contaminate the search procedure.
        row = {
            "Methods":           f"RAG-NAS (seed={m['seed']})",
            "search_dataset":    m["search_dataset"],
            "search_metric":     m["search_metric"],
            "best_arch":         m["best_arch"],
            "CIFAR-10 valid":         f"{m['cifar10_valid']:.2f}",
            "CIFAR-10 test":          f"{m['cifar10_test']:.2f}",
            "CIFAR-100 valid":        f"{m['cifar100_valid']:.2f}",
            "CIFAR-100 test":         f"{m['cifar100_test']:.2f}",
            "ImageNet-16-120 valid":  f"{m['in16_valid']:.2f}",
            "ImageNet-16-120 test":   f"{m['in16_test']:.2f}",
        }
        fieldnames = list(row.keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"\n[Success] Results saved to {output_csv}")

    # ── Multi-trial output (mean ± std) ─────────────────────────────────────
    else:
        agg = _aggregate_trials(all_metrics)
        output_csv = os.path.join(
            PROJECT_ROOT, "experiments",
            f"results_nasbench201_{args.trials}trials_seed{args.seed}.csv"
        )
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # Write one row per trial + a summary row.
        # search_dataset and search_metric are identical across all trials
        # (validation fitness); included for traceability.
        fieldnames = [
            "Methods",
            "search_dataset", "search_metric", "best_arch",
            "CIFAR-10 valid", "CIFAR-10 test",
            "CIFAR-100 valid", "CIFAR-100 test",
            "ImageNet-16-120 valid", "ImageNet-16-120 test",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({
                    "Methods":               f"RAG-NAS (seed={m['seed']})",
                    "search_dataset":        m["search_dataset"],
                    "search_metric":         m["search_metric"],
                    "best_arch":             m["best_arch"],
                    "CIFAR-10 valid":        f"{m['cifar10_valid']:.2f}",
                    "CIFAR-10 test":         f"{m['cifar10_test']:.2f}",
                    "CIFAR-100 valid":       f"{m['cifar100_valid']:.2f}",
                    "CIFAR-100 test":        f"{m['cifar100_test']:.2f}",
                    "ImageNet-16-120 valid": f"{m['in16_valid']:.2f}",
                    "ImageNet-16-120 test":  f"{m['in16_test']:.2f}",
                })
            # Summary row
            writer.writerow({
                "Methods":        f"RAG-NAS mean±std ({args.trials} trials)",
                "search_dataset": all_metrics[0]["search_dataset"],
                "search_metric":  all_metrics[0]["search_metric"],
                "best_arch":      "—",
                "CIFAR-10 valid":
                    f"{agg['cifar10_valid']['mean']:.2f}±{agg['cifar10_valid']['std']:.2f}",
                "CIFAR-10 test":
                    f"{agg['cifar10_test']['mean']:.2f}±{agg['cifar10_test']['std']:.2f}",
                "CIFAR-100 valid":
                    f"{agg['cifar100_valid']['mean']:.2f}±{agg['cifar100_valid']['std']:.2f}",
                "CIFAR-100 test":
                    f"{agg['cifar100_test']['mean']:.2f}±{agg['cifar100_test']['std']:.2f}",
                "ImageNet-16-120 valid":
                    f"{agg['in16_valid']['mean']:.2f}±{agg['in16_valid']['std']:.2f}",
                "ImageNet-16-120 test":
                    f"{agg['in16_test']['mean']:.2f}±{agg['in16_test']['std']:.2f}",
            })

        print("\n" + "=" * 72)
        print(f"  Multi-Trial Summary ({args.trials} trials, base seed={args.seed})")
        print("=" * 72)
        for k, v in agg.items():
            print(f"  {k:<24}  {v['mean']:.2f} ± {v['std']:.2f}")
        print(f"\n[Success] Results saved to {output_csv}")
