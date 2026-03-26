# MIEB Benchmark Evaluation

Evaluate the RAG-NAS AlignedEncoder on the [MIEB benchmark](https://arxiv.org/abs/2504.10471).

## Prerequisites
Your environment is already configured to automatically use the correct PyTorch Nightly `cu128` packages for Blackwell (RTX 5080) compatibility. All CUDA and HuggingFace dependencies are managed seamlessly.

## Pipeline

### Step 1: Train Contrastive Encoder
```bash
python scripts/train_contrastive_encoder.py --dataset cifar10 --epochs 100
```

### Step 2: Train Alignment (SigLIP)
```bash
python -m src.retrieval.alignment \
    --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
    --epochs 20 --batch_size 256
```

### Step 3: Run Evaluation (GPU + No Skip Gated)
We have consolidated all the scripts into a single, robust runner located at the project root:

```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Execute the evaluation script (runs on CUDA with gated datasets enabled)
bash run_mieb.sh
```

## Output Format (Paper Table 2)

| Model | Retrieval | Clustering | ZeroShot Cls | Linear Probe | Visual STS | Doc Und. | Compositionality | VCQA | MIEB | MIEB-lite |
|-------|-----------|------------|--------------|--------------|------------|----------|-----------------|------|------|-----------|
| RAG-NAS | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Results are automatically saved to `experiments/results.csv`.
