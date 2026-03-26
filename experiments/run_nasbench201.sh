#!/bin/bash
# Simple script to run NAS-Bench-201 E2E evaluation with GPU

# Ensure we're in project directory
cd "$(dirname "$0")/.."

echo "=========================================="
echo "NAS-Bench-201 Evaluation (Full Pipeline)"
echo "=========================================="
echo ""

# Check PyTorch
echo "Checking PyTorch..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "❌ PyTorch CUDA not working in the current environment!"
    echo "Please activate the correct conda/python environment with PyTorch Nightly and CUDA 12.8 (e.g. cv_final or .venv)"
    exit 1
fi

# Check missing pip packages
echo ""
echo "Checking dependencies..."
python -c "import rank_bm25" 2>/dev/null || { echo "Installing rank_bm25..."; python -m pip install rank_bm25; }

echo "Checking nas-201-api..."
python -c "import nas_201_api" 2>/dev/null || { 
    echo "Installing nas-201-api..."; 
    python -m pip install git+https://github.com/D-X-Y/NAS-Bench-201.git; 
}

echo ""
echo "=========================================="
echo "Starting NAS-Bench-201 Evaluation..."
echo "=========================================="
echo "Any arguments passed to this script will be forwarded to the python script."
echo "For example: ./run_nasbench201.sh --trials 3 --seed 42"
echo ""

# Run evaluation with provided arguments (defaults to 1 trial if none provided)
python experiments/run_nasbench201_e2e.py "$@"

echo ""
echo "=========================================="
echo "✓ Done! Results saved to experiments/ directory."
echo "=========================================="
