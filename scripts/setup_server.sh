#!/bin/bash
#
# VSF Research Platform - Server Setup Script
#
# Usage:
#   chmod +x scripts/setup_server.sh
#   ./scripts/setup_server.sh
#

set -e

echo "============================================================"
echo "    VSF Research Platform - Server Setup"
echo "============================================================"

# 1. Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version || { echo "Error: Python3 not found"; exit 1; }

# 2. Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  venv already exists, skipping..."
else
    python3 -m venv venv
    echo "  Created venv/"
fi

# 3. Activate and upgrade pip
echo ""
echo "[3/5] Activating venv and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

# 4. Install requirements
echo ""
echo "[4/5] Installing requirements..."
pip install -r requirements.txt

# 5. Verify GPU
echo ""
echo "[5/5] Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'         Memory: {mem:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

# 6. Download datasets (optional)
echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download datasets:"
echo "     python scripts/download_datasets.py"
echo ""
echo "  2. Test single model:"
echo "     python scripts/train.py --model fdw --dataset metr-la --debug"
echo ""
echo "  3. Run full experiments:"
echo "     python scripts/run_parallel.py --dry-run"
echo "     python scripts/run_parallel.py"
echo "============================================================"
