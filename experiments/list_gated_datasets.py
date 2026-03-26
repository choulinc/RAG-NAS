"""
List gated datasets in the MIEB benchmark.

This script checks which MIEB benchmark tasks use gated datasets
that require HuggingFace authentication.

Usage:
    python experiments/list_gated_datasets.py
"""

import sys
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_gated_datasets():
    """Check which MIEB tasks use gated datasets."""
    try:
        import mteb
        from datasets import load_dataset_builder
        from datasets.exceptions import DatasetNotFoundError
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install mteb datasets")
        sys.exit(1)

    # Known gated datasets
    KNOWN_GATED = [
        "facebook/winoground",
        # Add more as discovered
    ]

    print("=" * 80)
    print("MIEB Gated Datasets Check")
    print("=" * 80)

    # Get MIEB tasks
    try:
        benchmark = mteb.get_benchmark("MIEB(lite)")
        tasks = benchmark.tasks if hasattr(benchmark, "tasks") else list(benchmark)
    except Exception:
        print("Could not load MIEB benchmark")
        tasks = []

    print(f"\nTotal MIEB tasks: {len(tasks)}")
    print("\nChecking for gated datasets...\n")

    gated_tasks = []
    for task in tasks:
        task_name = task.metadata.name if hasattr(task.metadata, 'name') else str(task)

        # Check if task uses known gated dataset
        if hasattr(task.metadata, 'dataset'):
            dataset_info = task.metadata.dataset
            if isinstance(dataset_info, dict):
                dataset_path = dataset_info.get('path', '')
                if dataset_path in KNOWN_GATED:
                    gated_tasks.append({
                        'task': task_name,
                        'dataset': dataset_path,
                        'reason': 'Known gated dataset'
                    })
                    print(f"⊗ {task_name}")
                    print(f"  Dataset: {dataset_path}")
                    print(f"  Status: Gated (requires authentication)")
                    print()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Gated tasks found: {len(gated_tasks)}")
    print(f"Public tasks: {len(tasks) - len(gated_tasks)}")

    if gated_tasks:
        print("\n" + "=" * 80)
        print("How to Access Gated Datasets")
        print("=" * 80)
        print("""
1. Create a HuggingFace account: https://huggingface.co/join

2. Request access to each gated dataset:
   - Visit the dataset page (e.g., https://huggingface.co/datasets/facebook/winoground)
   - Click "Request access" and accept the terms

3. Authenticate locally:
   pip install -U huggingface_hub
   huggingface-cli login

   Or set HF_TOKEN environment variable:
   export HF_TOKEN=your_token_here

4. Re-run the evaluation with --no-skip-gated flag:
   python experiments/mieb_eval.py \\
       --alignment_checkpoint checkpoints/aligned_encoder.pt \\
       --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \\
       --no-skip-gated
        """)

    return gated_tasks


if __name__ == "__main__":
    check_gated_datasets()
