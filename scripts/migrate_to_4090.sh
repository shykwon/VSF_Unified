#!/bin/bash
# =============================================================================
# RTX 4090 Server Migration Script
# =============================================================================
#
# Usage:
#   1. On old server: tar the project and copy
#   2. On new server: run this script
#
# Prerequisites on new server:
#   - CUDA 11.8+ installed
#   - Python 3.10+
#   - Git
# =============================================================================

set -e

echo "=============================================="
echo "  VSF Research Platform - RTX 4090 Setup"
echo "=============================================="

# Check GPU
echo ""
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -ge 20000 ]; then
        echo "✅ GPU has sufficient memory (${GPU_MEM}MB)"
    else
        echo "⚠️  Warning: GPU memory (${GPU_MEM}MB) may be insufficient for full paper settings"
    fi
else
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/6] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created venv"
else
    echo "✅ venv already exists"
fi

source venv/bin/activate

# Install PyTorch with CUDA
echo ""
echo "[3/6] Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo ""
echo "[4/6] Installing dependencies..."
pip install -r requirements.txt

# Install PyTorch Geometric (for GIMCC)
echo ""
echo "[5/6] Installing PyTorch Geometric..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Verify installation
echo ""
echo "[6/6] Verifying installation..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test model imports
try:
    from src.models.fdw.wrapper import FDWWrapper
    from src.models.csdi.wrapper import CSDIWrapper
    from src.models.srdi.wrapper import SRDIWrapper
    print("✅ All model wrappers imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
EOF

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Download datasets: python scripts/download_datasets.py"
echo "  3. Test run: python scripts/train.py --model fdw --dataset metr-la --debug"
echo "  4. Full experiment: python scripts/train.py --config configs/gpu_4090.yaml --model all"
echo ""
echo "Note: Models now default to full paper settings (gpu_profile=4090)"
echo "      To use reduced settings, add: --gpu-profile 1080ti"
