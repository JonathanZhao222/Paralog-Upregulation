#!/bin/bash
#SBATCH --job-name=cd4t_paralog
#SBATCH --output=/home/users/jzhao222/paralog_upregulation/logs/cd4t_%j.log
#SBATCH --error=/home/users/jzhao222/paralog_upregulation/logs/cd4t_%j.log
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=bigmem
#
# CD4+ T Cell Paralog Upregulation Analysis
# -----------------------------------------
# Submits from project root on Sherlock:
#   sbatch run_cd4t_sherlock.sbatch
#
# If the job OOMs during h5ad loading, increase to --mem=256G.
# Check job progress: squeue -u $USER
# Check logs:         tail -f logs/cd4t_<JOBID>.log

set -euo pipefail

PROJECT_DIR="$HOME/paralog_upregulation"
PSEUDOBULK_URL="https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.pseudobulk_merged.h5ad"

# Store large data files in $SCRATCH (no quota) and symlink into project
SCRATCH_DATA="$SCRATCH/paralog_data/raw"
mkdir -p "$SCRATCH_DATA" "$PROJECT_DIR/logs" "$PROJECT_DIR/data"

# Replace data/raw with a symlink to scratch if it isn't one already
if [ -d "$PROJECT_DIR/data/raw" ] && [ ! -L "$PROJECT_DIR/data/raw" ]; then
    cp -r "$PROJECT_DIR/data/raw/." "$SCRATCH_DATA/"
    rm -rf "$PROJECT_DIR/data/raw"
fi
ln -sfn "$SCRATCH_DATA" "$PROJECT_DIR/data/raw"

DATA_DIR="$SCRATCH_DATA"
PSEUDOBULK_FILE="$DATA_DIR/GWCD4i.pseudobulk_merged.h5ad"

echo "======================================================"
echo "  CD4+ T Cell Paralog Analysis — Sherlock"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  Start: $(date)"
echo "======================================================"

# ── Activate environment ──────────────────────────────────────────────────────
module load python/3.12.1
VENV_DIR="${SCRATCH:-$HOME/scratch}/paralog_venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run setup_sherlock.sh first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

python --version
echo "Project: $PROJECT_DIR"

# ── Download pseudobulk h5ad (44.6 GB) ───────────────────────────────────────
if [ ! -f "$PSEUDOBULK_FILE" ]; then
    echo ""
    echo "[1/4] Downloading GWCD4i.pseudobulk_merged.h5ad (44.6 GB) ..."
    wget --progress=dot:giga --continue \
         --timeout=60 --tries=20 --waitretry=30 \
         -O "$PSEUDOBULK_FILE" \
         "$PSEUDOBULK_URL"
    echo "  Download complete: $(date)"
else
    echo "[1/4] Pseudobulk h5ad already present — skipping download."
fi

cd "$PROJECT_DIR"

# ── Preprocess: aggregate per gene, normalise, z-score ───────────────────────
echo ""
echo "[2/4] Preprocessing CD4+ T cell data (all 3 conditions) ..."
python scripts/07_preprocess_cd4t.py
echo "  Preprocessing complete: $(date)"

# ── Compute Δz ────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Computing Δz ..."
for CL in cd4t_rest cd4t_stim8hr cd4t_stim48hr; do
    echo "  → $CL"
    python scripts/02_compute_delta_z.py --cell-line "$CL"
done
echo "  Δz complete: $(date)"

# ── Visualise + cross-cell-line comparison ────────────────────────────────────
echo ""
echo "[4/4] Generating figures ..."
for CL in cd4t_rest cd4t_stim8hr cd4t_stim48hr; do
    echo "  → $CL"
    python scripts/03_compare_visualize.py --cell-line "$CL"
done
python scripts/04_cross_cell_line.py
echo "  Figures complete: $(date)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Done: $(date)"
echo "  Results: $PROJECT_DIR/results/"
echo "  Figures: $PROJECT_DIR/figures/"
echo ""
echo "  Sync results to your laptop (run ON YOUR LOCAL MACHINE):"
echo ""
echo "  rsync -avz jzhao222@sherlock.stanford.edu:'$PROJECT_DIR/results/' \\"
echo "    \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/results/\""
echo ""
echo "  rsync -avz jzhao222@sherlock.stanford.edu:'$PROJECT_DIR/figures/' \\"
echo "    \"/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Upregulation/figures/\""
echo "======================================================"
