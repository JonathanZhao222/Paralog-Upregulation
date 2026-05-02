#!/bin/bash
# setup_sherlock.sh
# -----------------
# Run this on Sherlock to set up the full environment and run the analysis.
# Usage: bash setup_sherlock.sh
#
# What it does:
#   1. Loads Sherlock's system Python module (no GLIBC issues)
#   2. Creates a venv with all dependencies installed via pip
#   3. Downloads Replogle data if not present
#   4. Preprocesses, computes delta_z, generates figures
#   5. Prints rsync command to copy results back to your laptop

set -e  # exit on error

# Guard: must be run on Sherlock, not locally
if ! command -v module &>/dev/null; then
    echo "ERROR: 'module' command not found."
    echo "This script must be run on Sherlock, not your local machine."
    echo ""
    echo "Steps:"
    echo "  1. rsync this script to Sherlock (run on your Mac):"
    echo "     rsync -avz \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/\" \\"
    echo "       jzhao222@sherlock.stanford.edu:~/paralog_upregulation/"
    echo "  2. SSH in and run:"
    echo "     ssh jzhao222@sherlock.stanford.edu"
    echo "     cd ~/paralog_upregulation && bash setup_sherlock.sh"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRATCH:-$HOME/scratch}/paralog_venv"

echo "======================================================"
echo "  Paralog Upregulation — Sherlock Setup"
echo "  Project: $PROJECT_DIR"
echo "======================================================"

# ── Step 1: Load Python and create venv ──────────────────────────────────────
echo "[1/4] Loading Python module ..."
module load python/3.12.1

echo "  Python: $(python3 --version)"
echo "  Creating venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "  Venv active: $(which python3)"

echo "  Installing dependencies via pip ..."
pip install --quiet --upgrade pip
# numpy must be installed first — pandas needs it at build time
pip install --quiet --prefer-binary numpy
pip install --quiet --prefer-binary \
    scipy pandas matplotlib seaborn tqdm openpyxl \
    anndata mygene natsort packaging requests
echo "  Dependencies installed."

# ── Step 2: Download data ─────────────────────────────────────────────────────
echo "[2/4] Downloading Replogle data ..."
cd "$PROJECT_DIR"
python3 scripts/00_download_data.py --replogle

# ── Step 3: Run full analysis (Replogle cell lines) ───────────────────────────
echo "[3/4] Running analysis on Replogle cell lines ..."
python3 scripts/02_compute_delta_z.py --all
python3 scripts/03_compare_visualize.py --all
python3 scripts/04_cross_cell_line.py

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Setup complete!"
echo "  Results: $PROJECT_DIR/results/"
echo "  Figures: $PROJECT_DIR/figures/"
echo ""
echo "  To copy results back to your laptop, run this"
echo "  command ON YOUR LOCAL MACHINE:"
echo ""
echo "  rsync -avz jzhao222@sherlock.stanford.edu:'$PROJECT_DIR/results/' \\"
echo "    \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/results/\""
echo "  rsync -avz jzhao222@sherlock.stanford.edu:'$PROJECT_DIR/figures/' \\"
echo "    \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/figures/\""
echo ""
echo "  To run the CD4+ T cell analysis (primary immune cells, 44.6 GB):"
echo "  sbatch run_cd4t_sherlock.sbatch"
echo "======================================================"
