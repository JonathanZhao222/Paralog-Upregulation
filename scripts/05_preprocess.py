"""
05_preprocess.py
----------------
Converts raw or single-cell h5ad files into pseudobulk z-normalised h5ad
files compatible with 02_compute_delta_z.py.

Handles two input types:
  1. Replogle raw pseudobulk  (K562_essential_raw_bulk_01.h5ad)
     — already pseudobulk, just needs log1p(CPM) + z-score
  2. X-Atlas single-cell h5ad (HCT116, HEK293T)
     — aggregate cells per perturbation → pseudobulk → log1p(CPM) + z-score

Pipeline
~~~~~~~~
  1. Load h5ad
  2. If single-cell: aggregate sum per perturbation → pseudobulk matrix
  3. Normalise: log1p(CPM) per row
  4. Z-score each gene across all perturbations
  5. Save as AnnData(obs.index=pert_label, var['gene_name']=symbol)

Output files (data/raw/)
~~~~~~~~~~~~~~~~~~~~~~~~
  K562_essential_pseudobulk_normalized.h5ad
  HCT116_pseudobulk_normalized.h5ad
  HEK293T_pseudobulk_normalized.h5ad

Usage:
    python scripts/05_preprocess.py --cell-line K562_essential
    python scripts/05_preprocess.py --cell-line HCT116
    python scripts/05_preprocess.py --all
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

CTRL_LABEL = "non-targeting"   # target label in output (must match 02_compute_delta_z.py)
MIN_CELLS_PER_PERT = 3

# input filename, output filename, is_singlecell, ctrl_label_in_source
CONFIGS = {
    "K562_essential": (
        "K562_essential_raw_bulk_01.h5ad",
        "K562_essential_pseudobulk_normalized.h5ad",
        False,   # already pseudobulk
        None,    # auto-detect
    ),
    "HCT116": (
        "HCT116_filtered_dual_guide_cells.h5ad",
        "HCT116_pseudobulk_normalized.h5ad",
        True,
        None,
    ),
    "HEK293T": (
        "HEK293T_filtered_dual_guide_cells.h5ad",
        "HEK293T_pseudobulk_normalized.h5ad",
        True,
        None,
    ),
}

PERT_COL_CANDIDATES = [
    "gene", "target_gene_name", "gene_name", "perturbation",
    "guide_target", "gene_id", "targeted_gene",
]
CTRL_CANDIDATES = [
    "non-targeting", "non_targeting", "negative_control",
    "control", "CTRL", "safe-targeting",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_pert_labels(adata: ad.AnnData, is_singlecell: bool) -> pd.Series:
    """Return per-row perturbation gene labels."""
    if not is_singlecell:
        # Replogle pseudobulk: obs.index = "{id}_{gene}_{guide}_{ensembl}"
        return pd.Series(
            [idx.split("_")[1] for idx in adata.obs.index],
            index=adata.obs.index,
        )
    # Single-cell: find the perturbation column
    for c in PERT_COL_CANDIDATES:
        if c in adata.obs.columns:
            print(f"  Perturbation column: '{c}'")
            return adata.obs[c].astype(str)
    # Heuristic: column whose values look like gene symbols
    for c in adata.obs.columns:
        sample = adata.obs[c].dropna().astype(str).head(30)
        if sample.str.match(r"^[A-Z][A-Z0-9\-]{1,15}$").mean() > 0.6:
            print(f"  Perturbation column (inferred): '{c}'")
            return adata.obs[c].astype(str)
    raise ValueError(f"Cannot detect perturbation column. obs: {list(adata.obs.columns)}")


def detect_ctrl(labels: pd.Series) -> str:
    unique = set(labels.unique())
    for c in CTRL_CANDIDATES:
        if c in unique:
            return c
    cands = [p for p in unique if "non" in p.lower() or "ctrl" in p.lower()]
    if cands:
        return cands[0]
    raise ValueError(f"Cannot detect control label. Candidates: {list(unique)[:20]}")


def get_gene_names(adata: ad.AnnData) -> list[str]:
    for col in ["gene_name", "gene_names", "symbol", "Gene"]:
        if col in adata.var.columns:
            return list(adata.var[col])
    # var.index might already be gene symbols
    if pd.Series(adata.var.index[:20]).str.match(r"^[A-Za-z][A-Za-z0-9\-\.]{1,20}$").mean() > 0.7:
        return list(adata.var.index)
    raise ValueError(f"Cannot find gene symbols. var columns: {list(adata.var.columns)}")


def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def pseudobulk_aggregate(X: np.ndarray, labels: np.ndarray,
                          unique_perts: list) -> np.ndarray:
    pb = np.zeros((len(unique_perts), X.shape[1]), dtype=np.float32)
    for i, pert in enumerate(tqdm(unique_perts, desc="  pseudobulk")):
        mask = labels == pert
        pb[i] = X[mask].sum(axis=0)
    return pb


def lognorm_zscore(pb: np.ndarray) -> np.ndarray:
    lib = pb.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    pb_norm = np.log1p(pb / lib * 1e6).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu  = np.nanmean(pb_norm, axis=0)
        std = np.nanstd(pb_norm, axis=0)
    std[std == 0] = np.nan
    pb_z = (pb_norm - mu) / std
    pb_z[~np.isfinite(pb_z)] = np.nan
    return pb_z


# ── Main processing ───────────────────────────────────────────────────────────
def process(cell_line: str) -> None:
    in_name, out_name, is_singlecell, _ = CONFIGS[cell_line]
    in_path  = DATA_DIR / in_name
    out_path = DATA_DIR / out_name

    if not in_path.exists():
        print(f"[skip] {in_name} not found — run: python scripts/00_download_data.py")
        return
    if out_path.exists():
        print(f"[skip] {out_name} already exists.")
        return

    print(f"\n{'='*55}\n  Preprocessing: {cell_line}\n{'='*55}")
    adata = ad.read_h5ad(in_path)
    print(f"  Loaded: {adata.n_obs:,} {'cells' if is_singlecell else 'rows'} × {adata.n_vars:,} genes")

    gene_names  = get_gene_names(adata)
    pert_labels = get_pert_labels(adata, is_singlecell)
    src_ctrl    = detect_ctrl(pert_labels)
    print(f"  Control label in source: '{src_ctrl}'")

    # Raw counts matrix
    raw = to_dense(adata.layers["counts"] if "counts" in adata.layers else adata.X)

    if is_singlecell:
        counts = pd.Series(pert_labels.values).value_counts()
        unique_perts = [p for p, n in counts.items() if n >= MIN_CELLS_PER_PERT]
        print(f"  Perturbations with ≥{MIN_CELLS_PER_PERT} cells: {len(unique_perts):,}")
        pb = pseudobulk_aggregate(raw, pert_labels.values, unique_perts)
    else:
        unique_perts = list(pert_labels.values)
        pb = raw   # already pseudobulk

    pb_z = lognorm_zscore(pb)

    # Remap control label to standard "non-targeting"
    out_index = [
        CTRL_LABEL if p == src_ctrl else p
        for p in unique_perts
    ]

    obs_df = pd.DataFrame({"gene_name": out_index}, index=out_index)
    var_df = pd.DataFrame({"gene_name": gene_names}, index=gene_names)
    out_adata = ad.AnnData(X=pb_z.astype(np.float32), obs=obs_df, var=var_df)

    print(f"  Saving → {out_name}  ({out_adata.n_obs} perturbations × {out_adata.n_vars} genes)")
    out_adata.write_h5ad(out_path)
    print(f"  Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell-line", choices=list(CONFIGS.keys()))
    group.add_argument("--all", action="store_true",
                       help="Preprocess all cell lines whose source files exist")
    args = parser.parse_args()

    cell_lines = list(CONFIGS.keys()) if args.all else [args.cell_line]
    for cl in cell_lines:
        process(cl)
    print("\nNext: python scripts/02_compute_delta_z.py --all")


if __name__ == "__main__":
    main()
