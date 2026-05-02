"""
05_preprocess_xatlas.py
-----------------------
Converts X-Atlas/Orion single-cell h5ad files into pseudobulk z-normalised
h5ad files compatible with 02_compute_delta_z.py.

Pipeline per cell line
~~~~~~~~~~~~~~~~~~~~~~
  1. Load single-cell h5ad (cells × genes)
  2. Detect the perturbation column in obs (gene being knocked down)
  3. Aggregate: sum raw counts per (perturbation, gene) → pseudobulk matrix
  4. Normalise each pseudobulk profile: log1p(CPM)
  5. Z-score across perturbations per gene  (≈ Replogle gemgroup z-norm)
  6. Save as AnnData with:
       obs.index  = perturbation label
       obs['gene_name'] = perturbation label   (for compatibility)
       var['gene_name'] = gene symbol
       X = z-scored pseudobulk expression

Output: data/raw/{CL}_pseudobulk_normalized.h5ad
  Used by 02_compute_delta_z.py as-is.

Note on comparability
~~~~~~~~~~~~~~~~~~~~~
  Replogle uses gemgroup (10x lane) z-normalisation before pseudobulk
  averaging.  Here we z-normalise after pseudobulk aggregation.  Both
  approaches centre expression relative to the experiment-wide baseline,
  so Δz values are comparable in direction and relative magnitude, though
  not in absolute scale.

Usage:
    python scripts/05_preprocess_xatlas.py --cell-line HCT116
    python scripts/05_preprocess_xatlas.py --cell-line HEK293T
    python scripts/05_preprocess_xatlas.py --all
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

# X-Atlas input filenames and desired output names
XATLAS_FILES = {
    "HCT116": ("HCT116_filtered_dual_guide_cells.h5ad",
                "HCT116_pseudobulk_normalized.h5ad"),
    "HEK293T": ("HEK293T_filtered_dual_guide_cells.h5ad",
                 "HEK293T_pseudobulk_normalized.h5ad"),
}

# Candidate column names for the knocked-down gene in X-Atlas obs
PERT_COL_CANDIDATES = [
    "gene", "target_gene_name", "gene_name", "perturbation",
    "guide_target", "gene_id", "targeted_gene",
]

CTRL_CANDIDATES = [
    "non-targeting", "non_targeting", "negative_control",
    "control", "CTRL", "safe-targeting",
]

MIN_CELLS_PER_PERT = 3   # minimum cells to include a perturbation in pseudobulk


# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_pert_col(obs: pd.DataFrame) -> str:
    for c in PERT_COL_CANDIDATES:
        if c in obs.columns:
            print(f"  Perturbation column detected: '{c}'")
            return c
    # Fall back: any column whose values look like gene symbols
    for c in obs.columns:
        sample = obs[c].dropna().astype(str).head(20)
        if sample.str.match(r"^[A-Z][A-Z0-9]{1,10}$").mean() > 0.7:
            print(f"  Perturbation column inferred: '{c}'")
            return c
    raise ValueError(
        f"Cannot detect perturbation column. obs columns: {list(obs.columns)}"
    )


def detect_ctrl_label(labels: pd.Series) -> str:
    unique = set(labels.unique())
    for c in CTRL_CANDIDATES:
        if c in unique:
            return c
    # Fuzzy
    candidates = [p for p in unique if "non" in p.lower() or "ctrl" in p.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot detect control label. Candidates: {candidates[:20]}"
    )


def get_gene_names(adata: ad.AnnData) -> list[str]:
    """Return gene symbols from var, trying common column names."""
    for col in ["gene_name", "gene_names", "symbol", "Gene"]:
        if col in adata.var.columns:
            return list(adata.var[col])
    # var.index might already be gene symbols
    sample = adata.var.index[:20]
    if pd.Series(sample).str.match(r"^[A-Za-z][A-Za-z0-9\-\.]{1,20}$").mean() > 0.7:
        return list(adata.var.index)
    raise ValueError(
        f"Cannot find gene symbol column. var columns: {list(adata.var.columns)}"
    )


# ── Core processing ───────────────────────────────────────────────────────────
def process(cell_line: str) -> None:
    in_name, out_name = XATLAS_FILES[cell_line]
    in_path  = DATA_DIR / in_name
    out_path = DATA_DIR / out_name

    if not in_path.exists():
        print(f"[skip] {in_name} not found — run: python scripts/00_download_data.py --xatlas")
        return

    if out_path.exists():
        print(f"[skip] {out_name} already exists.")
        return

    print(f"\nProcessing {cell_line} ...")
    print(f"  Loading {in_name} ...")
    adata = ad.read_h5ad(in_path)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ── Gene names ────────────────────────────────────────────────────────────
    gene_names = get_gene_names(adata)

    # ── Perturbation labels ───────────────────────────────────────────────────
    pert_col   = detect_pert_col(adata.obs)
    pert_labels = adata.obs[pert_col].astype(str)
    ctrl_label  = detect_ctrl_label(pert_labels)
    print(f"  Control label: '{ctrl_label}'")
    print(f"  Unique perturbations: {pert_labels.nunique():,}")

    # ── Get raw counts matrix ─────────────────────────────────────────────────
    # Prefer 'counts' layer if available, otherwise use X
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # ── Pseudobulk aggregation ────────────────────────────────────────────────
    print("  Aggregating to pseudobulk ...")
    perts = pert_labels.values
    unique_perts = [p for p, n in
                    pd.Series(perts).value_counts().items()
                    if n >= MIN_CELLS_PER_PERT]
    print(f"  Perturbations with ≥{MIN_CELLS_PER_PERT} cells: {len(unique_perts):,}")

    pb = np.zeros((len(unique_perts), X.shape[1]), dtype=np.float32)
    for i, pert in enumerate(tqdm(unique_perts, desc="  pseudobulk")):
        mask = perts == pert
        pb[i] = X[mask].sum(axis=0)

    # ── Normalise: log1p(CPM) ─────────────────────────────────────────────────
    print("  Normalising (log1p CPM) ...")
    lib_sizes = pb.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1
    pb_norm = np.log1p(pb / lib_sizes * 1e6).astype(np.float64)

    # ── Z-score per gene across perturbations ─────────────────────────────────
    print("  Z-scoring per gene ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(pb_norm, axis=0)
        stds  = np.nanstd(pb_norm, axis=0)
    stds[stds == 0] = np.nan  # avoid division by zero; NaN will be excluded downstream
    pb_z = (pb_norm - means) / stds
    pb_z[~np.isfinite(pb_z)] = np.nan

    # ── Build output AnnData ──────────────────────────────────────────────────
    obs_df = pd.DataFrame({"gene_name": unique_perts}, index=unique_perts)
    var_df = pd.DataFrame({"gene_name": gene_names}, index=gene_names)

    out_adata = ad.AnnData(
        X=pb_z.astype(np.float32),
        obs=obs_df,
        var=var_df,
    )

    # Rename control label to match what 02_compute_delta_z.py expects
    # obs.index is already the pert label — reindex so control rows are findable
    # by the same CTRL_LABEL="non-targeting" used in other scripts.
    # If the X-Atlas control label differs, remap it.
    if ctrl_label != "non-targeting":
        out_adata.obs.index = [
            "non-targeting" if p == ctrl_label else p
            for p in out_adata.obs.index
        ]
        out_adata.obs["gene_name"] = out_adata.obs.index

    print(f"  Saving {out_name} ({out_adata.n_obs} perturbations × {out_adata.n_vars} genes) ...")
    out_adata.write_h5ad(out_path)
    print(f"  Saved to {out_path}")


# ── Override load_adata for X-Atlas pseudobulk ────────────────────────────────
# The output h5ad uses obs.index directly as pert labels (not the
# "{id}_{gene}_{guide}_{ensembl}" format of Replogle), so we need
# 02_compute_delta_z.py to handle this. The script already works because
# the override in load_adata below is applied via CELL_LINE_FILES mapping
# and the obs.index is the perturbation label directly.
#
# In 02_compute_delta_z.py, pert_labels = [idx.split("_")[1] for idx in obs.index]
# This will BREAK for X-Atlas output (obs.index = gene name directly).
# The solution is handled in 02_compute_delta_z.py via the XATLAS_CELL_LINES set.


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell-line", choices=list(XATLAS_FILES.keys()))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cell_lines = list(XATLAS_FILES.keys()) if args.all else [args.cell_line]
    for cl in cell_lines:
        process(cl)
    print("\nDone. Run: python scripts/02_compute_delta_z.py --all")


if __name__ == "__main__":
    main()
