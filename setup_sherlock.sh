#!/bin/bash
# setup_sherlock.sh
# -----------------
# Run this on Sherlock to set up the full environment and run the analysis.
# Usage: bash setup_sherlock.sh
#
# What it does:
#   1. Installs Miniconda in $SCRATCH if conda not available
#   2. Creates a conda env with Python 3.11 + all dependencies
#   3. Downloads data (Replogle K562_essential if not present)
#   4. Preprocesses, computes delta_z, generates figures
#   5. Prints rsync command to copy results back to your laptop

set -e  # exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="${SCRATCH:-$HOME/scratch}"
CONDA_DIR="$SCRATCH_DIR/miniconda3"
ENV_NAME="paralog"

echo "======================================================"
echo "  Paralog Upregulation — Sherlock Setup"
echo "  Project: $PROJECT_DIR"
echo "======================================================"

# ── Step 1: Get conda ─────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    echo "[1/5] conda already available: $(conda --version)"
else
    echo "[1/5] Installing Miniconda to $CONDA_DIR ..."
    if [ ! -d "$CONDA_DIR" ]; then
        INSTALLER="$SCRATCH_DIR/miniconda_installer.sh"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O "$INSTALLER"
        bash "$INSTALLER" -b -p "$CONDA_DIR"
        rm "$INSTALLER"
    fi
    source "$CONDA_DIR/bin/activate"
    echo "  Miniconda installed."
fi

# Make conda available in this shell
if [ -f "$CONDA_DIR/bin/activate" ]; then
    source "$CONDA_DIR/bin/activate"
fi

# ── Step 2: Create / activate environment ────────────────────────────────────
echo "[2/5] Setting up conda environment '$ENV_NAME' ..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment already exists, activating."
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

# Install dependencies via conda for compiled packages (avoids BLAS issues)
echo "  Installing dependencies ..."
conda install -y -c conda-forge \
    numpy scipy pandas matplotlib seaborn tqdm openpyxl anndata \
    2>/dev/null || true

# Remaining packages via pip
pip install -q -r "$PROJECT_DIR/requirements.txt" \
    --no-deps 2>/dev/null || \
pip install -q -r "$PROJECT_DIR/requirements.txt"

echo "  Dependencies installed."

# ── Step 3: Download data ─────────────────────────────────────────────────────
echo "[3/5] Downloading data ..."
cd "$PROJECT_DIR"
python scripts/00_download_data.py --replogle

# Download X-Atlas only if explicitly requested (large files)
if [ "${DOWNLOAD_XATLAS:-0}" = "1" ]; then
    echo "  Downloading X-Atlas (warning: ~560 GB) ..."
    python scripts/00_download_data.py --xatlas
    echo "  Preprocessing X-Atlas single-cell data ..."
    python scripts/05_preprocess.py --all
fi

# ── Step 4: Preprocess K562_essential ────────────────────────────────────────
echo "[4/5] Preprocessing K562_essential ..."
python scripts/05_preprocess.py --cell-line K562_essential

# ── Step 5: Run full analysis ─────────────────────────────────────────────────
echo "[5/5] Running analysis ..."
python scripts/02_compute_delta_z.py --all
python scripts/03_compare_visualize.py --all
python scripts/04_cross_cell_line.py

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Analysis complete!"
echo "  Results: $PROJECT_DIR/results/"
echo "  Figures: $PROJECT_DIR/figures/"
echo ""
echo "  To copy results back to your laptop, run this"
echo "  command ON YOUR LOCAL MACHINE:"
echo ""
echo "  rsync -avz ${USER}@sherlock.stanford.edu:'$PROJECT_DIR/results/' \\"
echo "    \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/results/\""
echo "  rsync -avz ${USER}@sherlock.stanford.edu:'$PROJECT_DIR/figures/' \\"
echo "    \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/figures/\""
echo "======================================================"
