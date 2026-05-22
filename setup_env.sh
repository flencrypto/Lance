#!/bin/bash

# setup_env.sh - install Lance runtime dependencies.
# Usage: ./setup_env.sh [python_path]

set -euo pipefail

PYTHON=${1:-python}
TIMEOUT=300

# PyTorch CUDA wheels are hosted on the PyTorch index rather than PyPI.
PYTORCH_PACKAGES=(
    "torch==2.5.1+cu124"
    "torchvision==0.20.1+cu124"
    "torchaudio==2.5.1+cu124"
)

echo ">>> 开始卸载pynvml..."
$PYTHON -m pip uninstall -y pynvml || true

echo ">>> 开始安装PyTorch CUDA依赖..."
timeout $TIMEOUT $PYTHON -m pip install --upgrade --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    "${PYTORCH_PACKAGES[@]}"

echo ">>> 开始从requirements.txt安装软件包..."
timeout $TIMEOUT $PYTHON -m pip install --upgrade --no-cache-dir -r requirements.txt

echo "✓ 所有包均已成功安装或更新。"
