# RAG-NAS: Retrieval-Augmented Generation for Neural Architecture Search

A prototype system that combines multimodal retrieval with LLM-guided evolutionary search on NAS-Bench-201.

---

## ⚠ Scope and Claim Boundaries

This codebase is a **research prototype**. Before citing any numbers from this repository, read the limitations in each section below. Reviewers are encouraged to verify the evaluation protocols described here against the source code.

---

## 1. NAS-Bench-201 Evaluation Protocol

### Selection protocol (Issue 1 in review)

Architecture search fitness is evaluated on the **validation set only**:

- `search_dataset = "cifar100"`
- `search_metric  = "x-valid"` (NAS-Bench-201 validation accuracy)

The test set is **never queried during search**. Final test numbers (CIFAR-10 / CIFAR-100 / ImageNet-16-120) are read from the NAS-Bench-201 lookup table *after* the best architecture has been committed, and are not used for selection.

Every results CSV produced by `experiments/run_nasbench201_e2e.py` includes `search_dataset` and `search_metric` columns to make the protocol verifiable.

### Multi-dataset reporting

We report valid/test for all three NAS-Bench-201 datasets as a service to readers comparing against baselines. This is purely a **post-hoc lookup** — no cross-dataset signal leaks into selection. The rationale:

- CIFAR-100 valid → architecture selection fitness
- CIFAR-10, CIFAR-100, ImageNet16-120 test → final reporting (lookup-only, no feedback)

### Reproducibility

All randomness sources are seeded:

```bash
# Single trial (seed=42)
python experiments/run_nasbench201_e2e.py --seed 42

# Multiple trials (recommended for publication: ≥3)
python experiments/run_nasbench201_e2e.py --seed 42 --trials 5
```

The multi-trial run produces a CSV with one row per trial plus a mean±std summary row.

---

## 2. MIEB Evaluation

### ⚠ Coverage disclaimer — DO NOT report as official MIEB score

The score produced by `experiments/mieb_eval.py` is **not** directly comparable to the official MIEB or MIEB-lite scores in the paper (arXiv:2504.10471) unless full task coverage is achieved.

Current known coverage gaps:

| Task | Category | Status | Reason |
|---|---|---|---|
| `Fashion200kI2TRetrieval` | Retrieval | **Permanently unavailable** | Dataset not publicly hosted on HF |
| `NIGHTSI2IRetrieval` | Retrieval | **Permanently unavailable** | Dataset removed from HF Hub |
| `VisualSTS17Multilingual` | Visual STS | **Permanently unavailable** | Multilingual-only; no English subset |
| `VisualSTS-b-Multilingual` | Visual STS | **Permanently unavailable** | Multilingual-only; no English subset |
| `Winoground` | Compositionality | **Gated** (obtainable) | Requires HF auth: `huggingface-cli login` |
| Other gated tasks | Various | **Gated** (obtainable) | See `--no-skip-gated` flag |

The output CSV uses the column label `MIEB-lite-accessible-subset` (not `MIEB-lite`) when coverage < 100%, and includes these fields:

| Column | Meaning |
|---|---|
| `tasks_total` | Total tasks in the canonical benchmark |
| `tasks_evaluated` | Tasks that completed successfully |
| `tasks_broken` | Permanently unavailable upstream |
| `tasks_gated` | Skipped due to missing HF auth |
| `tasks_failed_runtime` | Unexpected errors |
| `coverage_ratio` | `tasks_evaluated / tasks_total` |
| `score_label_note` | Human-readable explanation of score validity |

**How to claim the score in a paper:**

- If `coverage_ratio < 1.0`: report as "accessible-subset score (N/M tasks)" and cite the specific broken/gated tasks
- If `coverage_ratio = 1.0`: may call it the official benchmark score

### Image preprocessing

MIEB evaluation uses **224×224 centre-crop + ImageNet normalisation** (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). This matches standard evaluation practice. The model backbone was also trained with 224×224 inputs; using 32×32/CIFAR preprocessing during evaluation (as earlier code versions did) would systematically degrade high-resolution tasks.

**Limitation**: The underlying SiameseEncoder is a ResNet-18 trained at 32×32 with dataset-level contrastive loss (see Section 4 below). While the evaluation preprocessing is now correct, the backbone capacity may be insufficient for high-resolution MIEB tasks. This is a model limitation, not a preprocessing issue.

### How to unlock gated datasets

```bash
# 1. Create a HuggingFace account and request access to gated datasets
#    (e.g., https://huggingface.co/datasets/facebook/winoground)

# 2. Authenticate
pip install -U huggingface_hub
huggingface-cli login

# 3. Run evaluation with gated datasets
python experiments/mieb_eval.py \
    --alignment_checkpoint checkpoints/aligned_encoder.pt \
    --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
    --no-skip-gated
```

---

## 3. Image Encoder: What Is Actually Learned

### Contrastive encoder (SiameseEncoder)

The encoder is trained with **dataset-level** positive/negative pairs:
- Positive: two images from the *same dataset folder* (e.g., both from CIFAR-10)
- Negative: two images from *different dataset folders* (e.g., CIFAR-10 vs STL-10)

This produces embeddings that cluster by **dataset/domain**, which is appropriate for the RAG-NAS retrieval step (finding models trained on similar datasets). It is **not** a task-type discriminator or a general-purpose visual encoder.

**Claim scope**: "dataset-domain-aware image retriever for architecture search" — not "task-aware architecture retrieval encoder" and not "general visual foundation model".

### Cross-modal alignment (AlignedEncoder)

The alignment projection heads are trained with SigLIP loss on (image, text) pairs where text is sampled from CLIP-style prompt templates over class names (e.g., "a photo of a cat").

**Known limitations** (disclosed in `src/retrieval/alignment.py`):
1. **Supervision scope**: Class-name-derived text does not cover the full diversity of MIEB tasks (retrieval captions, doc-understanding queries, VQA questions, compositionality). Scores on these categories are expected to be low.
2. **False-negative noise**: SigLIP treats all off-diagonal pairs as hard negatives. Semantically similar class names in the same batch (e.g., "cat" / "kitten") are incorrectly penalised.

**Claim scope**: "lightweight domain-aligned embedding baseline for cross-modal architecture retrieval" — not "general multimodal foundation embedding".

---

## 4. Retrieval & Fusion Pipeline

### Heuristic components

The following components are explicitly **heuristic-based**, not learned:

| Component | File | Nature |
|---|---|---|
| `DatasetAnalyzer` | `src/retrieval/dataset_analyzer.py` | Rule-based task/domain inference from directory structure and README keywords |
| Query parsing | `src/retrieval/rag.py` | Keyword extraction via string matching |
| Alpha fusion | `src/retrieval/retrieve.py` | Fixed weights (0.55 dense, 0.45 BM25) |
| Rerank boosts | `src/retrieval/retrieve.py` | Rule-based score adjustments |
| Task/dataset detection | `src/retrieval/multimodal_retrieve.py` | Keyword heuristics |

Each heuristic decision is **logged at runtime** so callers can verify the inference is reasonable. For `DatasetAnalyzer`, the triggering signal (metadata file / README keyword / directory structure / fallback) is printed alongside the inferred task.

**Claim scope**: These are heuristic baselines suitable for prototyping. Ablations or learned alternatives would be needed for a rigorous learned-retrieval claim.

---

## 5. LLM Template Generation

### Validation

Every LLM-generated template passes through `_validate_template()` and `_print_validation_summary()`:
- Unknown NAS-Bench-201 ops are dropped and warned
- `op_prior` and `edge_prior` are renormalized to sum to 1.0
- Unsatisfiable constraints (e.g., min_count > 6) are flagged
- Macro `forbidden_pair` constraints that the EA does **not** enforce are explicitly warned

### Unenforced constraints

The EA mutation step enforces `max_count` and `min_count` micro constraints. **Macro `forbidden_pair` constraints** (backbone ↔ head incompatibilities) are generated by the LLM but **not enforced** by the EA. This gap is logged at template generation time and should be mentioned in any paper that claims "constraint-aware search space".

---

## 6. Reproducibility Instructions

### Full pipeline

```bash
# Step 1: Train contrastive encoder (seed controls pair sampling)
python scripts/train_contrastive_encoder.py \
    --dataset all --epochs 100 --seed 42

# Step 2: Train alignment projection heads
python -m src.retrieval.alignment \
    --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
    --epochs 20 --seed 42

# Step 3: NAS-Bench-201 evaluation (3 trials recommended for publication)
python experiments/run_nasbench201_e2e.py \
    --seed 42 --trials 3

# Step 4: MIEB evaluation
python experiments/mieb_eval.py \
    --alignment_checkpoint checkpoints/aligned_encoder.pt \
    --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
    --output experiments/results_mieb.csv
```

### Seed coverage

| Component | Seed parameter | Controls |
|---|---|---|
| `train_contrastive_encoder.py` | `--seed` | Python random, NumPy, PyTorch, cuDNN |
| `src/retrieval/alignment.py` | `--seed` | Python random, NumPy, PyTorch, cuDNN |
| `experiments/run_nasbench201_e2e.py` | `--seed` | Python random, NumPy, PyTorch, EA sampling |
| `REA` (evolutionary search) | `seed` arg | `random.seed()` for EA population sampling |

All seeds default to `42`. For multi-trial reporting, trial `i` uses `base_seed + i`.

### Reported numbers

Single-trial runs report one row. Multi-trial runs (`--trials N`) report one row per trial plus a `mean±std` summary row. **For top-venue submission, use `--trials ≥ 3`.**

---

## 7. Structural Limitations (Not Fully Fixed)

These limitations cannot be resolved without significant architecture changes:

| Limitation | Impact | Mitigation in code |
|---|---|---|
| Backbone capacity (ResNet-18 at 32×32) | Low scores on high-res MIEB tasks | Honest score labelling; 224×224 eval |
| Class-name-only alignment supervision | Weak MIEB Compositionality / Doc Und. / VQA | Disclosed in `alignment.py` docstrings |
| Dataset-level (not task-level) contrastive objective | Encoder clusters domains, not task types | Disclosed in `contrastive_encoder.py` |
| Macro constraints not enforced by EA | LLM-generated backbone/head rules are ignored | Warned in validation summary at generation time |
| Single dataset used as fitness proxy | Selection generalises via lookup, not training | Selection protocol documented in CSV |
