import os
import sys
from typing import Dict, Any

# Ensure src in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nas.evolutionary_search import REA, sample_gene_from_template, gene_to_string

class MockEvaluator:
    """A fake NAS-Bench-201 Evaluator for testing the EA loop without the real API."""
    def evaluate(self, arch_str: str) -> float:
        # Give higher scores to archs with skip_connects and no avg_pool_3x3 to simulate a structured space
        score = 50.0
        if "skip_connect" in arch_str:
            score += 20 * arch_str.count("skip_connect")
        if "avg_pool_3x3" in arch_str:
            score -= 15 * arch_str.count("avg_pool_3x3")
        if "nor_conv_3x3" in arch_str:
            score += 10 * arch_str.count("nor_conv_3x3")
        
        # cap at 95.0
        return min(95.0, max(0.0, score))

def run_smoke_test():
    print("=== EA Engine Smoke Test ===\n")

    # MOCK TEMPLATES (simulating what LLM would output)
    mock_templates = [
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
                        {"type": "min_count", "op": "nor_conv_3x3", "value": 2} # must have at least 2 conv3x3
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
                        "0->1": {"skip_connect": 1.0} # Force first edge to be skip_connect
                    },
                    "constraints": [
                        {"type": "max_count", "op": "nor_conv_3x3", "value": 1} # max 1 heavy conv
                    ]
                }
            }
        }
    ]

    print("[1] Testing Initialization & Roulette Sampling...")
    for i, t in enumerate(mock_templates):
        print(f"\nSampling from Paradigm: '{t['paradigm']}'")
        print("  Expected behavior:")
        if i == 0:
            print("   -> Heavy bias towards nor_conv_3x3. Should never see < 2 nor_conv_3x3 or > 1 none.")
        else:
            print("   -> Heavy bias towards 1x1 & skip. First edge MUST be skip_connect. Max 1 nor_conv_3x3.")
        
        print("  Samples generated:")
        for _ in range(3):
            gene = sample_gene_from_template(t)
            print(f"    - {gene_to_string(gene)}")


    print("\n[2] Testing REA Evolution Loop...")
    evaluator = MockEvaluator()
    
    ea_config = {
        "pop_size": 10,
        "sample_size": 3,
        "cycles": 50,
        "max_sampling_retries": 100,
        "max_mutation_retries": 50
    }
    
    # Run EA
    ea = REA(templates=mock_templates, evaluator=evaluator, ea_config=ea_config)
    ea.initialize_population()
    
    print("\nInitial Population Samples:")
    for i in range(3):
        gene, tmpl, fit = ea.population[i]
        print(f"  [{tmpl['paradigm'][:5]}...] {gene_to_string(gene)} -> {fit:.1f}%")

    print("\nRunning Evolution...")
    best_gene, best_fit = ea.run()
    
    print("\n=== Result ===")
    print(f"Best Arch : {gene_to_string(best_gene)}")
    print(f"Best Score: {best_fit:.2f}% (Mock)")

if __name__ == "__main__":
    run_smoke_test()
