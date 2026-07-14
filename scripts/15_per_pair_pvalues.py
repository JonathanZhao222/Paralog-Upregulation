"""
15_per_pair_pvalues.py
----------------------
Computes a per-pair empirical p-value using a permutation approach.

Default mode — sig pairs only (updates sig_results.csv):
  For each sig pair (dep_gene → paralog_gene):
    1. Compute Δz for EVERY perturbation against paralog_gene — null distribution.
    2. Empirical p-value = fraction of perturbations with Δz ≥ observed Δz
       (one-sided; H₁: dep_gene KD upregulates paralog_gene more than chance).
    3. Apply Benjamini-Hochberg FDR correction across all sig pairs.

--all-pairs mode (updates all_pairs_ranked.csv):
  Same approach but applied to every row in all_pairs_ranked.csv.
  Null distributions are precomputed once per unique paralog gene for efficiency.
  BH FDR is applied across all pairs jointly.

Usage:
    python scripts/15_per_pair_pvalues.py --cell-line K562
    python scripts/15_per_pair_pvalues.py --cell-line K562 --all-pairs
    python scripts/15_per_pair_pvalues.py --cell-line K562 --all-pairs --all
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
CTRL_LABEL = "non-targeting"

CELL_LINE_FILES = {
    "K562":           "K562_gwps_normalized_bulk_01.h5ad",
    "K562_essential": "K562_essential_pseudobulk_normalized.h5ad",
    "rpe1":           "rpe1_normalized_bulk_01.h5ad",
    "HCT116":         "HCT116_pseudobulk_normalized.h5ad",
    "HEK293T":        "HEK293T_pseudobulk_normalized.h5ad",
    "melanoma":       "melanoma_pseudobulk_normalized.h5ad",
    "cd4t_rest":      "cd4t_rest_pseudobulk_normalized.h5ad",
    "cd4t_stim8hr":   "cd4t_stim8hr_pseudobulk_normalized.h5ad",
    "cd4t_stim48hr":  "cd4t_stim48hr_pseudobulk_normalized.h5ad",
    "neuron":         "neuron_pseudobulk_normalized.h5ad",
    "iPSC":           "iPSC_KOLF2_pseudobulk_normalized.h5ad",
}

DIRECT_INDEX_CELL_LINES = {
    "K562_essential", "HCT116", "HEK293T", "melanoma",
    "cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr", "neuron", "iPSC",
}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_adj
    return np.minimum(result, 1.0)


def load_h5ad(cell_line: str):
    """Load h5ad and return (pert_mean dict, gene_index dict, n_perts)."""
    h5ad = DATA_DIR / CELL_LINE_FILES[cell_line]
    if not h5ad.exists():
        return None, None, 0

    print(f"\nLoading {h5ad.name} ...")
    adata = ad.read_h5ad(h5ad)

    if cell_line in DIRECT_INDEX_CELL_LINES:
        pert_labels = list(adata.obs.index)
    else:
        pert_labels = [idx.split("_")[1] for idx in adata.obs.index]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan

    gene_index = {g: i for i, g in enumerate(adata.var["gene_name"])}

    rows_by_pert: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(pert_labels):
        rows_by_pert[label].append(i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pert_mean = {
            label: np.nanmean(X[rows, :], axis=0)
            for label, rows in rows_by_pert.items()
        }

    return pert_mean, gene_index, len([p for p in pert_mean if p != CTRL_LABEL])


def compute_pvals(df: pd.DataFrame, pert_mean: dict, gene_index: dict) -> pd.DataFrame:
    """
    Compute empirical p-values for every row in df (must have delta_z column).
    Precomputes null distributions once per unique paralog gene.
    Returns df with empirical_pval, empirical_fdr, n_perturbations added/updated.
    """
    ctrl_vec  = pert_mean[CTRL_LABEL]
    all_perts = [p for p in pert_mean if p != CTRL_LABEL]

    # Precompute null Δz vector for each unique paralog gene
    unique_paralogs = df["paralog_gene"].unique()
    null_cache: dict[str, np.ndarray] = {}
    print(f"  Precomputing null distributions for {len(unique_paralogs)} paralog genes ...")
    for paralog in tqdm(unique_paralogs, desc="paralogs"):
        j = gene_index.get(paralog)
        if j is None:
            null_cache[paralog] = np.array([])
            continue
        null_dz = np.array([pert_mean[p][j] - ctrl_vec[j] for p in all_perts])
        null_cache[paralog] = null_dz[np.isfinite(null_dz)]

    pval_arr = np.full(len(df), np.nan)
    n_arr    = np.zeros(len(df), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        null_dz = null_cache.get(row["paralog_gene"], np.array([]))
        if len(null_dz) == 0:
            continue
        dz_obs    = row["delta_z"]
        p_emp     = (null_dz >= dz_obs).sum() / len(null_dz)
        p_emp     = max(p_emp, 1.0 / len(null_dz))
        pval_arr[i] = p_emp
        n_arr[i]    = len(null_dz)

    fdr_arr = np.full(len(pval_arr), np.nan)
    valid   = np.isfinite(pval_arr)
    if valid.sum() > 0:
        fdr_arr[valid] = bh_fdr(pval_arr[valid])

    df = df.copy()
    df["n_perturbations"] = n_arr
    df["empirical_pval"]  = pval_arr
    df["empirical_fdr"]   = fdr_arr
    return df


def run_sig_only(cell_line: str) -> None:
    h5ad = DATA_DIR / CELL_LINE_FILES[cell_line]
    if not h5ad.exists():
        print(f"[skip] {h5ad.name} not found.")
        return

    sig_path = ROOT / "results" / cell_line / "sig_results.csv"
    if not sig_path.exists():
        print(f"[skip] sig_results.csv not found for {cell_line}.")
        return

    sig      = pd.read_csv(sig_path)
    testable = sig[(sig["testable"] == True) & sig["delta_z"].notna()].copy()
    if len(testable) == 0:
        print(f"[skip] No testable pairs for {cell_line}.")
        return

    pert_mean, gene_index, n_perts = load_h5ad(cell_line)
    if pert_mean is None:
        return
    print(f"  {n_perts} perturbations in null distribution")

    result = compute_pvals(testable, pert_mean, gene_index)

    sig.loc[result.index, "n_perturbations"] = result["n_perturbations"].values
    sig.loc[result.index, "empirical_pval"]  = result["empirical_pval"].values
    sig.loc[result.index, "empirical_fdr"]   = result["empirical_fdr"].values

    sig.to_csv(sig_path, index=False)
    print(f"\nUpdated {sig_path}")

    print(f"\n{'dep_gene':12s} {'paralog':12s} {'Δz':>7s} "
          f"{'p_emp':>10s} {'FDR':>10s}  {'sig?':5s}")
    print("─" * 62)
    for _, r in result.sort_values("delta_z", ascending=False).iterrows():
        p   = r["empirical_pval"]
        fdr = r["empirical_fdr"]
        if not np.isfinite(p):
            continue
        star = "**" if fdr < 0.01 else ("*" if fdr < 0.05 else "")
        print(f"{r['dep_gene']:12s} {r['paralog_gene']:12s} "
              f"{r['delta_z']:+7.3f} {p:10.4f} {fdr:10.4f}  {star}")


def run_all_pairs(cell_line: str) -> None:
    h5ad = DATA_DIR / CELL_LINE_FILES[cell_line]
    if not h5ad.exists():
        print(f"[skip] {h5ad.name} not found.")
        return

    ranked_path = ROOT / "results" / cell_line / "all_pairs_ranked.csv"
    if not ranked_path.exists():
        print(f"[skip] all_pairs_ranked.csv not found for {cell_line}. "
              f"Run: python scripts/10_rank_all_pairs.py --cell-line {cell_line}")
        return

    ranked = pd.read_csv(ranked_path)
    ranked = ranked.dropna(subset=["delta_z"]).copy()
    print(f"\n[{cell_line}] {len(ranked):,} pairs to score")

    pert_mean, gene_index, n_perts = load_h5ad(cell_line)
    if pert_mean is None:
        return
    print(f"  {n_perts} perturbations in null distribution")

    result = compute_pvals(ranked, pert_mean, gene_index)

    # Merge back into the full (possibly larger) ranked CSV preserving row order
    full = pd.read_csv(ranked_path)
    for col in ["n_perturbations", "empirical_pval", "empirical_fdr"]:
        full[col] = np.nan
    full.loc[result.index, "n_perturbations"] = result["n_perturbations"].values
    full.loc[result.index, "empirical_pval"]  = result["empirical_pval"].values
    full.loc[result.index, "empirical_fdr"]   = result["empirical_fdr"].values

    full.to_csv(ranked_path, index=False)
    print(f"\nUpdated {ranked_path}")
    sig_rows = full[full["group"] == "Aneuploidy vulnerability"].dropna(subset=["empirical_fdr"])
    n_sig = (sig_rows["empirical_fdr"] < 0.05).sum()
    print(f"  Aneuploidy vulnerability pairs with FDR < 0.05: {n_sig}/{len(sig_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-line", choices=list(CELL_LINE_FILES.keys()))
    parser.add_argument("--all-pairs", action="store_true",
                        help="Score every row in all_pairs_ranked.csv (not just sig pairs)")
    parser.add_argument("--all", action="store_true",
                        help="Run for all cell lines that have the required files")
    args = parser.parse_args()

    if not args.cell_line and not args.all:
        parser.error("Provide --cell-line or --all")

    cell_lines = list(CELL_LINE_FILES.keys()) if args.all else [args.cell_line]

    for cl in cell_lines:
        print(f"\n{'='*60}\n  {cl}\n{'='*60}")
        if args.all_pairs:
            run_all_pairs(cl)
        else:
            run_sig_only(cl)


if __name__ == "__main__":
    main()
