#!/bin/bash
# Run this ONCE on the login node to install GPU-capable PyTorch into the conda env.
# H200 GPUs on Borg require CUDA 12.x.

ENV=/scratch_tide/yc5224/envs/ccus_gym
CACHE=/scratch_tide/yc5224/.pip_cache

TMPDIR=/scratch_tide/yc5224/.tmp
export TMPDIR

mkdir -p "$CACHE"
mkdir -p "$TMPDIR"

echo "Uninstalling CPU torch..."
$ENV/bin/pip uninstall torch -y

echo "Installing torch with CUDA 12.1 support (cache -> $CACHE)..."
$ENV/bin/pip install torch \
    --index-url https://download.pytorch.org/whl/cu121 \
    --cache-dir "$CACHE"

echo "Verifying..."
$ENV/bin/python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
