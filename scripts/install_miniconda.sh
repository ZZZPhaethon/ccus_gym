#!/bin/bash
# Install Miniconda to scratch, create ccus_gym env, install all deps including GPU torch.
# Run once on the login node: bash /scratch_tide/yc5224/install_miniconda.sh

set -e

SCRATCH=/scratch_tide/yc5224
CONDA_PREFIX=$SCRATCH/miniconda3
ENV_NAME=ccus_gym
ENV_PREFIX=$SCRATCH/miniconda3/envs/$ENV_NAME
CODE=$SCRATCH/CYW_code
TMPDIR=$SCRATCH/.tmp
export TMPDIR
mkdir -p "$TMPDIR"

# ── 1. Download and install Miniconda ──────────────────────────────────────
if [ ! -f "$CONDA_PREFIX/bin/conda" ]; then
    echo "==> Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O "$SCRATCH/miniconda_installer.sh"
    echo "==> Installing Miniconda to $CONDA_PREFIX ..."
    bash "$SCRATCH/miniconda_installer.sh" -b -p "$CONDA_PREFIX"
    rm "$SCRATCH/miniconda_installer.sh"
else
    echo "==> Miniconda already installed at $CONDA_PREFIX"
fi

# ── 2. Init conda for this shell ───────────────────────────────────────────
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda config --set always_yes true

# ── 3. Create env (Python 3.11) ────────────────────────────────────────────
if conda env list | grep -q "^$ENV_NAME "; then
    echo "==> Env '$ENV_NAME' already exists, skipping creation"
else
    echo "==> Creating conda env '$ENV_NAME' (Python 3.11)..."
    conda create -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"
echo "==> Active env: $CONDA_PREFIX"

# ── 4. Install GPU PyTorch (CUDA 12.1) ─────────────────────────────────────
echo "==> Installing PyTorch with CUDA 12.1..."
pip install torch \
    --index-url https://download.pytorch.org/whl/cu121 \
    --cache-dir "$SCRATCH/.pip_cache"

# ── 5. Install remaining dependencies ─────────────────────────────────────
echo "==> Installing project dependencies..."
pip install \
    numpy gymnasium pettingzoo pyyaml scikit-learn matplotlib tensorboard \
    --cache-dir "$SCRATCH/.pip_cache"

# ── 6. Verify ─────────────────────────────────────────────────────────────
echo ""
echo "==> Verification:"
python -c "
import torch, numpy, gymnasium, pettingzoo, matplotlib, tensorboard
print('torch      :', torch.__version__)
print('CUDA avail :', torch.cuda.is_available())
print('numpy      :', numpy.__version__)
print('gymnasium  :', gymnasium.__version__)
print('pettingzoo :', pettingzoo.__version__)
"

echo ""
echo "==> Done! Activate later with:"
echo "    source $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "    export PYTHONPATH=$CODE:\$PYTHONPATH"
