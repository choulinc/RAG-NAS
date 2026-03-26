#!/bin/bash
# Simple script to run MIEB evaluation with GPU
# Uses current active environment (venv or conda)

# Ensure we're in project directory
cd /home/choulin/RAG-NAS

echo "=========================================="
echo "MIEB Evaluation (GPU + No Skip Gated)"
echo "=========================================="
echo ""

# Check PyTorch
echo "Checking PyTorch..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "❌ PyTorch CUDA not working in the current environment!"
    echo ""
    echo "Please install PyTorch with CUDA:"
    echo "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo "Or:"
    echo "  pip install torch torchvision"
    exit 1
fi

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import mteb" 2>/dev/null || { echo "Installing mteb..."; pip install mteb; }
python -c "import datasets" 2>/dev/null || { echo "Installing datasets..."; pip install datasets; }
python -c "import huggingface_hub" 2>/dev/null || { echo "Installing huggingface_hub..."; pip install huggingface_hub; }

# Check HF auth
echo ""
echo "Checking HuggingFace authentication..."
if python -c "from huggingface_hub import whoami; print(f'✓ Logged in as: {whoami()[\"name\"]}')" 2>/dev/null; then
    # Check gated dataset access
    if python -c "from datasets import load_dataset; load_dataset('facebook/winoground', split='test[:1]')" 2>/dev/null; then
        echo "✓ Can access gated datasets"
    else
        echo "⚠️  Cannot access facebook/winoground"
        echo "   1. Visit: https://huggingface.co/datasets/facebook/winoground"
        echo "   2. Click 'Request access'"
        echo ""
        read -p "Press Enter after requesting access, or Ctrl+C to cancel..."
    fi
else
    echo "⚠️  Not logged in to HuggingFace"
    echo ""
    echo "To access gated datasets:"
    echo "1. Visit: https://huggingface.co/datasets/facebook/winoground"
    echo "2. Click 'Request access'"
    echo "3. Get token: https://huggingface.co/settings/tokens"
    echo "4. Login:"
    echo ""
    read -p "Login now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        hf auth login
    else
        echo "Skipping authentication. Run 'huggingface-cli login' later."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Starting Evaluation..."
echo "=========================================="
echo ""
echo "Config:"
echo "  Device: cuda"
echo "  Skip gated: NO"
echo "  Time: ~2-3 hours"
echo ""

# Run evaluation
python experiments/mieb_eval.py \
    --alignment_checkpoint checkpoints/aligned_encoder.pt \
    --encoder_checkpoint checkpoints/contrastive_encoder_best.pt \
    --device cuda \
    --no-skip-gated

echo ""
echo "=========================================="
echo "✓ Done! Results saved to:"
echo "  experiments/results.csv"
echo "=========================================="
