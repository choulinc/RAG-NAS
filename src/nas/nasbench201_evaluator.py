import os
import threading
import functools
from typing import Dict, Any
import argparse
from tqdm import tqdm

# PyTorch >= 2.6 changed torch.load default to weights_only=True.
# NAS-Bench-201's .pth was saved with numpy globals, which requires weights_only=False.
import torch
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

try:
    from nas_201_api import NASBench201API as API
except ImportError:
    API = None

class NASBench201Evaluator:
    def __init__(self, api_path: str):
        if not API:
            raise RuntimeError(
                "API client not available"
            )
        if not os.path.exists(api_path):
            raise FileNotFoundError(f"NAS-Bench-201 API not found at {api_path}")
        
        print(f"Loading NAS-Bench-201 from {api_path} (this may take 1-2 minutes)...")
        self.api = API(api_path, verbose=False)
        print("Loaded NAS-Bench-201 successfully.")

    def evaluate(self, arch_str: str, dataset: str = "cifar100", metric: str = "x-valid") -> float:
        """
        Evaluate a cell structure string on NAS-Bench-201.

        Args:
            arch_str: The architecture string (e.g., "|nor_conv_3x3~0|+|...|")
            dataset: Target dataset ("cifar10-valid", "cifar100", "ImageNet16-120").
            metric: Target metric.
                - "x-valid"  → validation accuracy (USE THIS for architecture search /
                               fitness evaluation to avoid test-set leakage).
                - "x-test"   → test accuracy (reserved for final held-out reporting).
                - "ori-test" → alternative test split for cifar10-valid.
        Returns:
            The accuracy (float 0–100). Returns 0.0 if arch is invalid or not found.

        Protocol note:
            Architecture selection (EA fitness) MUST use "x-valid".
            "x-test" / "ori-test" should only be queried AFTER the best architecture
            has been committed, to report the final held-out performance.
        """
        try:
            index = self.api.query_index_by_arch(arch_str)
            if index < 0:
                print(f"[Warning] Architecture {arch_str} not found in NAS-Bench-201.")
                return 0.0
            
            results = self.api.query_meta_info_by_index(index, hp="200")
            
            if dataset not in results.get_dataset_names():
                print(f"[Warning] Dataset {dataset} not supported. Options: {results.get_dataset_names()}")
                return 0.0
                
            res_dict = results.get_metrics(dataset, metric)
            if 'accuracy' in res_dict:
                return res_dict['accuracy']
            return 0.0
            
        except Exception as e:
            print(f"[Error] Failed to evaluate {arch_str}: {e}")
            return 0.0

if __name__ == "__main__":
    # CLI 
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_path", type=str, required=True, help="Path to NAS-Bench-201-v1_1-096897.pth")
    ap.add_argument("--arch", type=str, default="|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|")
    ap.add_argument("--dataset", type=str, default="cifar100")
    args = ap.parse_args()

    evaluator = NASBench201Evaluator(args.api_path)
    acc = evaluator.evaluate(args.arch, dataset=args.dataset)
    print(f"Arch: {args.arch}\nDataset: {args.dataset}\nAccuracy: {acc:.2f}%")
