"""
17_preprocess_ipsc.py
---------------------
Converts the KOLF2.1J iPSC CRISPRi perturbation atlas (Nature Biotech 2026,
Figshare doi:10.25452/figshare.plus.27261219) into a pseudobulk z-normalised
h5ad compatible with 02_compute_delta_z.py.

Pipeline
~~~~~~~~
  1. Load the downloaded h5ad (single-cell raw counts or pseudobulk)
  2. If single-cell: aggregate sum per perturbation → pseudobulk matrix
  3. Normalise: log1p(CPM) per row
  4. Z-score each gene across all perturbations
  5. Save as AnnData(obs.index=pert_label, var['gene_name']=symbol)
     with control rows relabelled to "non-targeting"

Output
~~~~~~
  data/raw/iPSC_KOLF2_pseudobulk_normalized.h5ad

Usage
~~~~~
  # First download and rename the file — see instructions at the top of main()
  python scripts/17_preprocess_ipsc.py
  python scripts/17_preprocess_ipsc.py --input data/raw/my_ipsc_file.h5ad
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

CTRL_LABEL     = "non-targeting"        # standard label used by the rest of the pipeline
MIN_CELLS      = 3                      # minimum cells per perturbation to keep
DEFAULT_INPUT  = DATA_DIR / "iPSC_KOLF2_raw.h5ad"
OUTPUT_NAME    = "iPSC_KOLF2_pseudobulk_normalized.h5ad"

# Columns to search for perturbation gene labels (in priority order)
PERT_COL_CANDIDATES = [
    "gene", "target_gene_name", "gene_name", "perturbation",
    "guide_target", "gene_id", "targeted_gene", "target",
]
# Strings that indicate control wells
CTRL_CANDIDATES = [
    "non-targeting", "non_targeting", "negative_control",
    "control", "CTRL", "safe-targeting", "NegCtrl",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def strip_guide_suffix(labels: pd.Series) -> pd.Series:
    """
    Strip guide number suffix from labels like 'GENE_1', 'GENE_2' → 'GENE'.
    Only strips if the suffix is _<integer>. Leaves other labels unchanged.
    """
    stripped = labels.str.replace(r"_\d+$", "", regex=True)
    # Only apply if stripping actually reduces the number of unique labels
    if stripped.nunique() < labels.nunique():
        n_before = labels.nunique()
        n_after  = stripped.nunique()
        print(f"  Guide suffix stripped: {n_before} guides → {n_after} genes")
        return stripped
    return labels


def detect_pert_labels(adata: ad.AnnData) -> pd.Series:
    """Return per-row perturbation gene labels (gene symbol or 'non-targeting')."""
    for c in PERT_COL_CANDIDATES:
        if c in adata.obs.columns:
            print(f"  Perturbation column: '{c}'")
            labels = adata.obs[c].astype(str)
            return strip_guide_suffix(labels)
    for c in adata.obs.columns:
        sample = adata.obs[c].dropna().astype(str).head(50)
        if sample.str.match(r"^[A-Z][A-Z0-9\-]{1,15}$").mean() > 0.6:
            print(f"  Perturbation column (inferred): '{c}'")
            labels = adata.obs[c].astype(str)
            return strip_guide_suffix(labels)
    raise ValueError(
        f"Cannot detect perturbation column. obs columns: {list(adata.obs.columns)}"
    )


def detect_ctrl(labels: pd.Series) -> str:
    unique = set(labels.unique())
    for c in CTRL_CANDIDATES:
        if c in unique:
            return c
    # Use startswith to avoid matching genes that contain "non" (e.g. SELENON)
    cands = [p for p in unique
             if p.lower().startswith("non") or p.lower().startswith("ctrl")
             or p.lower().startswith("neg") or p.lower().startswith("control")]
    if cands:
        print(f"  Control candidates found: {cands} — using '{cands[0]}'")
        return cands[0]
    raise ValueError(
        f"Cannot detect control label. Sample labels: {list(unique)[:20]}"
    )


def get_gene_names(adata: ad.AnnData) -> list[str]:
    for col in ["gene_name", "gene_names", "symbol", "Gene", "gene_symbol"]:
        if col in adata.var.columns:
            return list(adata.var[col].astype(str))
    # var.index may already be gene symbols
    if pd.Series(adata.var.index[:30]).str.match(r"^[A-Za-z][A-Za-z0-9\-\.]{1,20}$").mean() > 0.7:
        return list(adata.var.index.astype(str))
    raise ValueError(
        f"Cannot find gene symbols. var columns: {list(adata.var.columns)}"
    )


def is_pseudobulk(adata: ad.AnnData) -> bool:
    """Heuristic: if n_obs < 50,000 and X contains floats, assume pseudobulk."""
    X_sample = to_dense(adata.X[:min(10, adata.n_obs)])
    has_floats = not np.all(X_sample == np.floor(X_sample))
    return adata.n_obs < 50_000 and has_floats


def pseudobulk_aggregate(X, labels: np.ndarray, unique_perts: list) -> np.ndarray:
    """
    Aggregate by perturbation using sparse matrix multiply — O(n_cells x n_genes),
    much faster than looping over perturbations.
    """
    pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
    cell_idx    = np.array([pert_to_idx.get(l, -1) for l in labels])
    valid       = cell_idx >= 0
    cell_idx    = cell_idx[valid]

    n_perts = len(unique_perts)
    n_valid = valid.sum()

    # Indicator matrix: (n_perts x n_cells), one 1 per cell in its perturbation row
    indicator = sp.csr_matrix(
        (np.ones(n_valid, dtype=np.float32), (cell_idx, np.arange(n_valid))),
        shape=(n_perts, n_valid),
    )

    X_valid = X[valid]
    if not sp.issparse(X_valid):
        X_valid = sp.csr_matrix(X_valid)

    print(f"  Matrix multiply: ({n_perts} x {n_valid}) @ ({n_valid} x {X.shape[1]}) ...")
    pb = (indicator @ X_valid).toarray().astype(np.float32)
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
    return np.where(std > 0, (pb_norm - mu) / std, np.nan).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Path to downloaded iPSC h5ad (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: same directory as input)")
    args = parser.parse_args()

    in_path  = args.input
    out_path = args.output if args.output else in_path.parent / OUTPUT_NAME

    if not in_path.exists():
        print(
            f"[ERROR] Input file not found: {in_path}\n\n"
            "Download instructions:\n"
            "  1. Visit https://plus.figshare.com/articles/dataset/27261219\n"
            "     (free Figshare account required)\n"
            "  2. Download the processed h5ad file\n"
            f"  3. Save it to: {in_path}\n"
            "  4. Re-run this script"
        )
        return

    if out_path.exists():
        print(f"[skip] {OUTPUT_NAME} already exists. Delete it to reprocess.")
        return

    print(f"\nLoading {in_path.name} ...")
    adata = ad.read_h5ad(in_path)
    print(f"  {adata.n_obs:,} obs × {adata.n_vars:,} vars")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  var columns: {list(adata.var.columns)}")

    gene_names  = get_gene_names(adata)
    pert_labels = detect_pert_labels(adata)
    src_ctrl    = detect_ctrl(pert_labels)
    print(f"  Control label detected: '{src_ctrl}'")

    already_pb = is_pseudobulk(adata)
    print(f"  Detected format: {'pseudobulk' if already_pb else 'single-cell'}")

    # Keep sparse until after aggregation to avoid densifying the full matrix
    raw = adata.layers.get("counts", adata.X)

    if already_pb:
        unique_perts = list(pert_labels.values)
        pb = to_dense(raw)
        print(f"  {len(unique_perts)} perturbation rows (already pseudobulk)")
    else:
        counts = pd.Series(pert_labels.values).value_counts()
        unique_perts = [p for p, n in counts.items() if n >= MIN_CELLS]
        print(f"  Perturbations with >={MIN_CELLS} cells: {len(unique_perts):,}")
        pb = pseudobulk_aggregate(raw, pert_labels.values, np.array(unique_perts))

    print("  log1p(CPM) normalising + z-scoring ...")
    pb_z = lognorm_zscore(pb)

    # Standardise control label
    out_index = [CTRL_LABEL if p == src_ctrl else p for p in unique_perts]
    n_ctrl = out_index.count(CTRL_LABEL)
    print(f"  Control rows: {n_ctrl}")

    obs_df = pd.DataFrame({"gene_name": out_index}, index=out_index)
    var_df = pd.DataFrame({"gene_name": gene_names}, index=gene_names)
    out_adata = ad.AnnData(X=pb_z, obs=obs_df, var=var_df)

    print(f"\nSaving → {out_path}")
    print(f"  {out_adata.n_obs} perturbations × {out_adata.n_vars} genes")
    out_adata.write_h5ad(out_path)
    print("  Done.")
    print(f"\nNext steps:")
    print(f"  python scripts/02_compute_delta_z.py --cell-line iPSC")
    print(f"  python scripts/10_rank_all_pairs.py --cell-line iPSC")
    print(f"  python scripts/15_per_pair_pvalues.py --cell-line iPSC --all-pairs")


if __name__ == "__main__":
    main()
