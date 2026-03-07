"""
Smoke test for the EA engine.
- Without --api_path: uses MockEvaluator (no NAS-Bench-201 needed)
- With --api_path:    uses the real NASBench201Evaluator
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nas.evolutionary_search import REA, sample_gene_from_template, gene_to_string


class MockEvaluator:
    """A fake NAS-Bench-201 Evaluator for testing the EA loop without the real API."""
    def evaluate(self, arch_str: str, **kwargs) -> float:
        score = 50.0
        if "skip_connect" in arch_str:
            score += 20 * arch_str.count("skip_connect")
        if "avg_pool_3x3" in arch_str:
            score -= 15 * arch_str.count("avg_pool_3x3")
        if "nor_conv_3x3" in arch_str:
            score += 10 * arch_str.count("nor_conv_3x3")
        return min(95.0, max(0.0, score))


MOCK_TEMPLATES = [
    {
        "paradigm": "Heavy Convolutional",
        "micro": {
            "nb201": {
                "allowed_ops": ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"],
                "op_prior": {
                    "nor_conv_3x3": 0.60,
                    "skip_connect": 0.20,
                    "nor_conv_1x1": 0.15,
                    "avg_pool_3x3": 0.03,
                    "none": 0.02
                },
                "constraints": [
                    {"type": "max_count", "op": "none", "value": 1},
                    {"type": "min_count", "op": "nor_conv_3x3", "value": 2}
                ]
            }
        }
    },
    {
        "paradigm": "Mobile Lightweight",
        "micro": {
            "nb201": {
                "allowed_ops": ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"],
                "op_prior": {
                    "nor_conv_3x3": 0.10,
                    "skip_connect": 0.40,
                    "nor_conv_1x1": 0.40,
                    "avg_pool_3x3": 0.05,
                    "none": 0.05
                },
                "edge_prior": {
                    "0->1": {"skip_connect": 1.0}
                },
                "constraints": [
                    {"type": "max_count", "op": "nor_conv_3x3", "value": 1}
                ]
            }
        }
    }
]


def run_smoke_test(evaluator, use_real: bool):
    label = "Real NB201" if use_real else "Mock"
    print(f"=== EA Engine Smoke Test ({label}) ===\n")

    # --- Part 1: Sampling ---
    print("[1] Testing Initialization & Roulette Sampling...")
    for i, t in enumerate(MOCK_TEMPLATES):
        print(f"\nSampling from Paradigm: '{t['paradigm']}'")
        if i == 0:
            print("  Expected: Heavy bias towards nor_conv_3x3. Min 2 nor_conv_3x3, max 1 none.")
        else:
            print("  Expected: Heavy bias towards 1x1 & skip. Edge 0->1 MUST be skip_connect. Max 1 nor_conv_3x3.")
        print("  Samples:")
        for _ in range(3):
            gene = sample_gene_from_template(t)
            arch = gene_to_string(gene)
            acc = evaluator.evaluate(arch) if use_real else evaluator.evaluate(arch)
            print(f"    {arch}  ->  {acc:.2f}%")

    # --- Part 2: EA ---
    print("\n[2] Testing REA Evolution Loop...")
    ea_config = {
        "pop_size": 10,
        "sample_size": 3,
        "cycles": 50,
        "max_sampling_retries": 100,
        "max_mutation_retries": 50
    }
    ea = REA(templates=MOCK_TEMPLATES, evaluator=evaluator, ea_config=ea_config)
    ea.initialize_population()

    print("\nInitial Population (top 5):")
    sorted_pop = sorted(ea.population, key=lambda x: x[2], reverse=True)
    for gene, tmpl, fit in sorted_pop[:5]:
        print(f"  [{tmpl['paradigm'][:15]:15s}] {gene_to_string(gene)}  ->  {fit:.2f}%")

    print("\nRunning Evolution...")
    best_gene, best_fit = ea.run()

    print(f"\n{'='*60}")
    print(f"Best Architecture : {gene_to_string(best_gene)}")
    print(f"Best Accuracy     : {best_fit:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_path", type=str, default=None,
                    help="Path to NAS-Bench-201 .pth file. If omitted, uses MockEvaluator.")
    args = ap.parse_args()

    if args.api_path:
        from src.nas.nasbench201_evaluator import NASBench201Evaluator
        evaluator = NASBench201Evaluator(args.api_path)
        run_smoke_test(evaluator, use_real=True)
    else:
        evaluator = MockEvaluator()
        run_smoke_test(evaluator, use_real=False)
