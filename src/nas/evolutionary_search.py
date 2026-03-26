import json
import random
import argparse
import sys
import os
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# relative import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.nas.nasbench201_evaluator import NASBench201Evaluator

NB201_OPS = ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"]

def is_valid_cell(gene: List[str], constraints: List[Dict[str, Any]]) -> bool:
    # template constraints
    counts = Counter(gene)
    for c in constraints:
        ctype = c.get("type")
        op = c.get("op")
        val = c.get("value", 0)
        if ctype == "max_count" and op in counts and counts[op] > val:
            return False
        if ctype == "min_count" and counts.get(op, 0) < val:
            return False
    return True

def sample_op(prior_dict: Dict[str, float]) -> str:
    ops = list(prior_dict.keys())
    probs = list(prior_dict.values())
    s = sum(probs)
    if s == 0:
        return random.choice(ops)
    probs = [p/s for p in probs]
    return random.choices(ops, weights=probs, k=1)[0]

def sample_gene_from_template(template: Dict[str, Any], max_retries: int = 100) -> List[str]:
    """
    Sample 6 edges using Roulette Wheel Selection based on Prior dicts.
    Returns array of 6 strings.
    """
    micro = template.get("micro", {}).get("nb201", {})
    op_prior = micro.get("op_prior", {op: 1.0/len(NB201_OPS) for op in NB201_OPS})
    edge_prior = micro.get("edge_prior", {})
    constraints = micro.get("constraints", [])
    
    # Edges order: 0->1, 0->2, 1->2, 0->3, 1->3, 2->3
    edge_names = ["0->1", "0->2", "1->2", "0->3", "1->3", "2->3"]

    for _ in range(max_retries):
        gene = []
        for e in edge_names:
            if e in edge_prior:
                gene.append(sample_op(edge_prior[e]))
            else:
                gene.append(sample_op(op_prior))
        
        if is_valid_cell(gene, constraints):
            return gene
            
    # Fallback if too constrained
    return [random.choice(list(op_prior.keys())) for _ in range(6)]

def gene_to_string(gene: List[str]) -> str:
    """
    Format gene array to NB201 string: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    """
    return f"|{gene[0]}~0|+|{gene[1]}~0|{gene[2]}~1|+|{gene[3]}~0|{gene[4]}~1|{gene[5]}~2|"

class REA:
    """
    Regularized Evolution Algorithm for NAS-Bench-201.

    Search protocol:
        - Fitness during evolution uses ``search_dataset`` / ``search_metric``
          (defaults: cifar100 / x-valid) so the test set is never seen during search.
        - Final held-out test metrics must be queried separately by the caller
          AFTER committing to the best architecture.
    """
    def __init__(
        self,
        templates: List[Dict[str, Any]],
        evaluator: NASBench201Evaluator,
        ea_config: Dict[str, Any],
        search_dataset: str = "cifar100",
        search_metric: str = "x-valid",
        seed: Optional[int] = None,
    ):
        self.templates = templates
        self.evaluator = evaluator
        self.pop_size = ea_config.get("pop_size", 20)
        self.sample_size = ea_config.get("sample_size", 10)
        self.cycles = ea_config.get("cycles", 100)
        self.max_mutation_retries = ea_config.get("max_mutation_retries", 50)
        self.max_sampling_retries = ea_config.get("max_sampling_retries", 100)
        # Search uses validation set only — test set is held out for final reporting.
        self.search_dataset = search_dataset
        self.search_metric = search_metric
        self.population: List[Tuple[List[str], Dict[str, Any], float]] = []  # [(gene, template, fitness)]
        self.history = []
        if seed is not None:
            random.seed(seed)

    def evaluate_gene(self, gene: List[str]) -> float:
        """Evaluate fitness on the held-in validation set (never the test set)."""
        arch_str = gene_to_string(gene)
        return self.evaluator.evaluate(
            arch_str, dataset=self.search_dataset, metric=self.search_metric
        )

    def initialize_population(self):
        print("Initializing population...")
        for i in range(self.pop_size):
            tmpl = random.choice(self.templates)
            gene = sample_gene_from_template(tmpl, max_retries=self.max_sampling_retries)
            fit = self.evaluate_gene(gene)
            self.population.append((gene, tmpl, fit))
            self.history.append((gene, fit))

    def run(self):
        print(f"Running REA for {self.cycles} cycles...")
        for i in range(self.cycles):
            # Sample tournament
            tournament = random.sample(self.population, self.sample_size)
            best_parent = max(tournament, key=lambda x: x[2])
            parent_gene, parent_tmpl, _ = best_parent
            
            # Mutate: pick one edge and resample it using the template's per-edge prior
            # (falls back to op_prior if no edge-specific entry exists).
            child_gene = list(parent_gene)
            mut_idx = random.randint(0, 5)

            micro = parent_tmpl.get("micro", {}).get("nb201", {})
            constraints = micro.get("constraints", [])
            op_prior = micro.get("op_prior", {op: 1.0 / len(NB201_OPS) for op in NB201_OPS})
            edge_prior = micro.get("edge_prior", {})
            edge_name = ["0->1", "0->2", "1->2", "0->3", "1->3", "2->3"][mut_idx]
            # Use edge-specific prior when available; otherwise fall back to global op_prior.
            mutation_prior = edge_prior.get(edge_name, op_prior)

            valid = False
            for _ in range(self.max_mutation_retries):
                new_gene = list(child_gene)
                new_gene[mut_idx] = sample_op(mutation_prior)
                if is_valid_cell(new_gene, constraints):
                    child_gene = new_gene
                    valid = True
                    break
            
            if not valid:
                child_gene = list(parent_gene) # Fallback
                
            fit = self.evaluate_gene(child_gene)
            self.population.append((child_gene, parent_tmpl, fit))
            self.history.append((child_gene, fit))
            
            # Remove oldest (Regularized Evolution)
            self.population.pop(0)
            
            if (i+1) % 10 == 0:
                best = max(self.population, key=lambda x: x[2])
                print(f"Cycle {i+1}/{self.cycles} | Best Acc: {best[2]:.2f}% | Pop Size: {len(self.population)}")

        return max(self.history, key=lambda x: x[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=str, required=True, help="Path to llm_templates.json")
    ap.add_argument("--api_path", type=str, required=True, help="Path to NAS-Bench-201 .pth file")
    ap.add_argument("--config", type=str, default="src/nas/ea_config.yaml", help="Path to EA YAML config")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    ea_config = config_data.get("ea", {})

    with open(args.templates, "r", encoding="utf-8") as f:
        templates = json.load(f)

    if not templates:
        print("No templates found.")
        return

    evaluator = NASBench201Evaluator(args.api_path)
    ea = REA(templates, evaluator, ea_config=ea_config)
    
    ea.initialize_population()
    best_gene, best_fit = ea.run()
    
    print("\n--- Search Completed ---")
    print(f"Found Best Architecture: {gene_to_string(best_gene)}")
    print(f"Accuracy: {best_fit:.2f}%")

if __name__ == "__main__":
    main()
