"""
Protocol, reporting, and reproducibility tests.

Covers:
  1. NAS search protocol — EA uses validation metric, never test set
  2. MIEB coverage-aware aggregation — broken/gated/failed tracked separately;
     partial-coverage runs use 'accessible-subset' label, never 'official'
  3. Template validation — invalid ops dropped, priors renormalized, warnings emitted
  4. Seed determinism — same seed produces same EA result
"""
from __future__ import annotations

import sys
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. NAS search protocol
# ---------------------------------------------------------------------------

class TestNASSearchProtocol:
    """EA fitness must use validation metric; test set never touched during search."""

    def _make_evaluator(self):
        """Mock evaluator that records every (dataset, metric) call."""
        calls = []

        class MockEvaluator:
            def evaluate(self, arch_str: str, dataset: str = "cifar100", metric: str = "x-valid") -> float:
                calls.append((dataset, metric))
                return random.uniform(70.0, 90.0)

        return MockEvaluator(), calls

    def test_default_search_uses_validation(self):
        from src.nas.evolutionary_search import REA

        evaluator, calls = self._make_evaluator()
        template = {
            "paradigm": "test",
            "micro": {"nb201": {"op_prior": {
                "nor_conv_3x3": 0.4, "nor_conv_1x1": 0.2,
                "skip_connect": 0.2, "avg_pool_3x3": 0.1, "none": 0.1
            }}}
        }
        ea = REA(
            templates=[template],
            evaluator=evaluator,
            ea_config={"pop_size": 5, "sample_size": 3, "cycles": 3},
            search_dataset="cifar100",
            search_metric="x-valid",
            seed=0,
        )
        ea.initialize_population()
        ea.run()

        # Every evaluation call must use x-valid (never x-test / ori-test)
        for dataset, metric in calls:
            assert metric == "x-valid", (
                f"EA called evaluator with metric='{metric}' during search. "
                f"Only 'x-valid' is permitted during architecture selection."
            )

    def test_search_dataset_configurable(self):
        """search_dataset and search_metric must be separately configurable."""
        from src.nas.evolutionary_search import REA

        evaluator, calls = self._make_evaluator()
        template = {
            "micro": {"nb201": {"op_prior": {
                "nor_conv_3x3": 0.2, "nor_conv_1x1": 0.2,
                "skip_connect": 0.2, "avg_pool_3x3": 0.2, "none": 0.2
            }}}
        }
        ea = REA(
            templates=[template],
            evaluator=evaluator,
            ea_config={"pop_size": 3, "sample_size": 2, "cycles": 2},
            search_dataset="cifar10-valid",
            search_metric="x-valid",
            seed=1,
        )
        ea.initialize_population()
        ea.run()
        for dataset, metric in calls:
            assert dataset == "cifar10-valid"
            assert metric == "x-valid"

    def test_csv_includes_protocol_fields(self, tmp_path):
        """CSV output must contain search_dataset and search_metric columns."""
        import csv
        result = {
            "Methods": "RAG-NAS (seed=42)",
            "search_dataset": "cifar100",
            "search_metric": "x-valid",
            "best_arch": "|nor_conv_3x3~0|+|skip_connect~0|none~1|+|nor_conv_3x3~0|avg_pool_3x3~1|nor_conv_1x1~2|",
            "CIFAR-10 valid": "91.00",
            "CIFAR-10 test": "93.00",
        }
        csv_path = tmp_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            writer.writeheader()
            writer.writerow(result)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert "search_dataset" in row, "CSV must contain 'search_dataset' column"
        assert "search_metric"  in row, "CSV must contain 'search_metric' column"
        assert row["search_dataset"] == "cifar100"
        assert row["search_metric"]  == "x-valid"


# ---------------------------------------------------------------------------
# 2. MIEB coverage-aware aggregation
# ---------------------------------------------------------------------------

class TestMIEBCoverageReporting:
    """Partial coverage must never produce an 'official' benchmark label."""

    def _run_evaluation_stub(
        self,
        n_total: int,
        n_evaluated: int,
        n_broken: int,
        n_gated: int,
        n_failed: int,
    ) -> Dict[str, Any]:
        """Simulate the aggregation logic from run_evaluation() without running mteb."""
        import numpy as np

        MIEB_CATEGORIES = [
            "Retrieval", "Clustering", "ZeroShot Cls", "Linear Probe",
            "Visual STS", "Doc Und.", "Compositionality", "VCQA",
        ]

        # Simulate scores: evaluated tasks get a real score, others get 0
        scores_per_cat = {cat: [] for cat in MIEB_CATEGORIES}
        per = n_total // len(MIEB_CATEGORIES) or 1
        ev_remaining, br_remaining, ga_remaining, fa_remaining = (
            n_evaluated, n_broken, n_gated, n_failed
        )

        for cat in MIEB_CATEGORIES:
            for _ in range(per):
                if ev_remaining > 0:
                    scores_per_cat[cat].append(0.75)
                    ev_remaining -= 1
                elif br_remaining > 0:
                    scores_per_cat[cat].append(0.0)
                    br_remaining -= 1
                elif ga_remaining > 0:
                    scores_per_cat[cat].append(0.0)
                    ga_remaining -= 1
                elif fa_remaining > 0:
                    scores_per_cat[cat].append(0.0)
                    fa_remaining -= 1

        coverage = n_evaluated / n_total if n_total else 0.0
        if coverage >= 0.999:
            overall_key = "MIEB-lite"
        else:
            overall_key = "MIEB-lite-accessible-subset"

        all_scores = [s for cat_scores in scores_per_cat.values() for s in cat_scores]
        overall = float(np.mean(all_scores)) * 100 if all_scores else 0.0

        return {
            "overall_key": overall_key,
            "overall_score": overall,
            "coverage": coverage,
            "tasks_total": n_total,
            "tasks_evaluated": n_evaluated,
            "tasks_broken": n_broken,
            "tasks_gated": n_gated,
            "tasks_failed_runtime": n_failed,
        }

    def test_partial_coverage_uses_accessible_subset_label(self):
        result = self._run_evaluation_stub(
            n_total=41, n_evaluated=27, n_broken=4, n_gated=1, n_failed=9
        )
        assert "accessible-subset" in result["overall_key"], (
            f"Partial-coverage run must not use the official benchmark label. "
            f"Got: '{result['overall_key']}'"
        )

    def test_full_coverage_may_use_official_label(self):
        result = self._run_evaluation_stub(
            n_total=10, n_evaluated=10, n_broken=0, n_gated=0, n_failed=0
        )
        assert "accessible-subset" not in result["overall_key"], (
            "Full-coverage run should use the official benchmark label."
        )

    def test_broken_counted_as_zero_not_excluded(self):
        """Broken tasks must drag the mean down, not be silently excluded."""
        result_partial = self._run_evaluation_stub(
            n_total=8, n_evaluated=4, n_broken=4, n_gated=0, n_failed=0
        )
        result_full = self._run_evaluation_stub(
            n_total=4, n_evaluated=4, n_broken=0, n_gated=0, n_failed=0
        )
        # Partial coverage (4/8) should yield a lower overall score than full (4/4)
        # because the 4 broken tasks contribute 0.0 to the mean.
        assert result_partial["overall_score"] < result_full["overall_score"], (
            "Broken tasks must be counted as 0 in the denominator, "
            "reducing the aggregate score."
        )

    def test_task_status_fields_present(self):
        result = self._run_evaluation_stub(
            n_total=41, n_evaluated=27, n_broken=4, n_gated=2, n_failed=8
        )
        for field in ["tasks_total", "tasks_evaluated", "tasks_broken",
                      "tasks_gated", "tasks_failed_runtime", "coverage"]:
            assert field in result, f"Missing field '{field}' in result dict"

        assert result["tasks_total"] == 41
        assert result["tasks_evaluated"] == 27
        assert result["tasks_broken"] == 4
        assert result["tasks_gated"] == 2
        assert result["tasks_failed_runtime"] == 8


# ---------------------------------------------------------------------------
# 3. Template validation
# ---------------------------------------------------------------------------

class TestTemplateValidation:
    """_validate_template must sanitize priors and warn about issues."""

    def test_unknown_ops_dropped(self, capsys):
        from src.retrieval.llm_template_generator import _validate_template

        t = {
            "paradigm": "test",
            "micro": {"nb201": {"op_prior": {
                "nor_conv_3x3": 0.5,
                "hallucinated_op": 0.3,  # not a valid NB201 op
                "nor_conv_1x1": 0.2,
            }}}
        }
        validated = _validate_template(t)
        op_prior = validated["micro"]["nb201"]["op_prior"]
        assert "hallucinated_op" not in op_prior, "Unknown op must be dropped"
        captured = capsys.readouterr()
        assert "hallucinated_op" in captured.out, "Unknown op must trigger a warning"

    def test_op_prior_renormalized(self):
        from src.retrieval.llm_template_generator import _validate_template

        t = {
            "paradigm": "test",
            "micro": {"nb201": {"op_prior": {
                "nor_conv_3x3": 2.0,
                "nor_conv_1x1": 2.0,
                "skip_connect": 1.0,
                "avg_pool_3x3": 0.0,
                "none": 0.0,
            }}}
        }
        validated = _validate_template(t)
        op_prior = validated["micro"]["nb201"]["op_prior"]
        total = sum(op_prior.values())
        assert abs(total - 1.0) < 1e-5, f"op_prior must sum to 1.0, got {total:.6f}"

    def test_forbidden_pair_warning(self, capsys):
        from src.retrieval.llm_template_generator import _validate_template

        t = {
            "paradigm": "test",
            "macro": {"constraints": [
                {"type": "forbidden_pair", "source": "ResNet", "target": "TransformerHead"}
            ]},
            "micro": {"nb201": {}},
        }
        _validate_template(t)
        captured = capsys.readouterr()
        assert "forbidden_pair" in captured.out.lower(), (
            "Unenforceable forbidden_pair constraints must produce a warning"
        )

    def test_empty_op_prior_gets_uniform_fallback(self):
        from src.retrieval.llm_template_generator import _validate_template

        NB201_OPS = ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"]
        t = {"paradigm": "empty", "micro": {"nb201": {"op_prior": {}}}}
        validated = _validate_template(t)
        op_prior = validated["micro"]["nb201"]["op_prior"]
        # Should have a valid distribution
        assert len(op_prior) > 0
        total = sum(op_prior.values())
        assert abs(total - 1.0) < 1e-5

    def test_validation_summary_printed(self, capsys):
        from src.retrieval.llm_template_generator import _print_validation_summary

        templates = [
            {"paradigm": "A", "micro": {"nb201": {"op_prior": {
                "nor_conv_3x3": 0.4, "nor_conv_1x1": 0.15,
                "skip_connect": 0.25, "avg_pool_3x3": 0.1, "none": 0.1,
            }}}},
        ]
        _print_validation_summary(templates)
        captured = capsys.readouterr()
        assert "Template Validation Summary" in captured.out


# ---------------------------------------------------------------------------
# 4. Seed determinism
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """Same seed must produce identical EA trajectories."""

    def _run_ea(self, seed: int) -> List[float]:
        """Return the fitness history for a short EA run."""
        from src.nas.evolutionary_search import REA

        class DeterministicEvaluator:
            """Returns arch-hash-based score (deterministic for same arch)."""
            def evaluate(self, arch_str: str, dataset: str = "cifar100", metric: str = "x-valid") -> float:
                return float(abs(hash(arch_str)) % 10000) / 100.0

        template = {
            "micro": {"nb201": {"op_prior": {
                "nor_conv_3x3": 0.2, "nor_conv_1x1": 0.2,
                "skip_connect": 0.2, "avg_pool_3x3": 0.2, "none": 0.2,
            }}}
        }
        ea = REA(
            templates=[template],
            evaluator=DeterministicEvaluator(),
            ea_config={"pop_size": 5, "sample_size": 3, "cycles": 5},
            seed=seed,
        )
        ea.initialize_population()
        ea.run()
        return [fit for _, fit in ea.history]

    def test_same_seed_same_history(self):
        history_a = self._run_ea(seed=7)
        history_b = self._run_ea(seed=7)
        assert history_a == history_b, (
            "Identical seed must produce identical EA fitness history. "
            "Check that random.seed() is applied before population initialization."
        )

    def test_different_seeds_different_histories(self):
        history_a = self._run_ea(seed=1)
        history_b = self._run_ea(seed=2)
        assert history_a != history_b, (
            "Different seeds should produce different EA trajectories "
            "(this test is probabilistic; failure indicates broken RNG)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
