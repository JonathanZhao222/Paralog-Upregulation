"""
15_per_pair_pvalues.py
----------------------
Computes a per-pair empirical p-value for each significant aneuploid
vulnerability pair using a permutation approach.

For each sig pair (dep_gene → paralog_gene):
  1. Compute Δz for EVERY perturbation against paralog_gene using the full
     pert_mean dict — this is the null distribution (~11,000 values).
  2. Empirical p-value = fraction of perturbations with Δz ≥ observed Δz
     (one-sided; H₁: dep_gene KD upregulates paralog_gene more than chance).
  3. Apply Benjamini-Hochberg FDR correction across all sig pairs.

Adds columns to sig_results.csv:
  n_perturbations   — size of null distribution used
  empirical_pval    — raw one-sided empirical p-value
  empirical_fdr     — BH-corrected FDR

Usage:
    python scripts/15_per_pair_pvalues.py --cell-line K562
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
}

DIRECT_INDEX_CELL_LINES = {
    "K562_essential", "HCT116", "HEK293T", "melanoma",
    "cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr", "neuron",
}


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    # Enforce monotonicity (cumulative min from right)
    fdr_adj = np.minimum.accumulate(fdr[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_adj
    return np.minimum(result, 1.0)


def run(cell_line: str) -> None:
    h5ad = DATA_DIR / CELL_LINE_FILES[cell_line]
    if not h5ad.exists():
        print(f"[skip] {h5ad.name} not found.")
        return

    sig_path = ROOT / "results" / cell_line / "sig_results.csv"
    if not sig_path.exists():
        print(f"[skip] sig_results.csv not found for {cell_line}.")
        return

    sig = pd.read_csv(sig_path)
    testable = sig[(sig["testable"] == True) & sig["delta_z"].notna()].copy()
    if len(testable) == 0:
        print(f"[skip] No testable pairs for {cell_line}.")
        return

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

    # Build per-perturbation mean vectors
    from collections import defaultdict
    rows_by_pert: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(pert_labels):
        rows_by_pert[label].append(i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pert_mean = {
            label: np.nanmean(X[rows, :], axis=0)
            for label, rows in rows_by_pert.items()
        }

    ctrl_vec = pert_mean[CTRL_LABEL]
    all_perts = [p for p in pert_mean if p != CTRL_LABEL]
    n_perts = len(all_perts)
    print(f"  {n_perts} perturbations in null distribution")

    # For each testable sig pair, compute empirical p-value
    results = []
    for _, row in tqdm(testable.iterrows(), total=len(testable), desc="pairs"):
        dep_gene    = row["dep_gene"]
        paralog     = row["paralog_gene"]
        dz_obs      = row["delta_z"]

        j = gene_index.get(paralog)
        if j is None:
            results.append((row.name, np.nan, n_perts, np.nan))
            continue

        # Δz for every perturbation against this paralog
        null_dz = np.array([
            pert_mean[p][j] - ctrl_vec[j]
            for p in all_perts
        ])
        null_dz = null_dz[np.isfinite(null_dz)]

        # One-sided: fraction of null ≥ observed
        p_emp = (null_dz >= dz_obs).sum() / len(null_dz)
        # Minimum p-value is 1/n_perts
        p_emp = max(p_emp, 1.0 / len(null_dz))

        results.append((row.name, dz_obs, len(null_dz), p_emp))

    idx_arr   = [r[0] for r in results]
    dz_arr    = [r[1] for r in results]
    n_arr     = [r[2] for r in results]
    pval_arr  = np.array([r[3] for r in results], dtype=float)

    # BH FDR on non-nan values
    fdr_arr = np.full(len(pval_arr), np.nan)
    valid   = np.isfinite(pval_arr)
    if valid.sum() > 0:
        fdr_arr[valid] = bh_fdr(pval_arr[valid])

    # Write back into full sig dataframe
    sig.loc[idx_arr, "n_perturbations"] = n_arr
    sig.loc[idx_arr, "empirical_pval"]  = pval_arr
    sig.loc[idx_arr, "empirical_fdr"]   = fdr_arr

    sig.to_csv(sig_path, index=False)
    print(f"\nUpdated {sig_path}")

    print(f"\n{'dep_gene':12s} {'paralog':12s} {'Δz':>7s} "
          f"{'p_emp':>10s} {'FDR':>10s}  {'sig?':5s}")
    print("─" * 62)
    for _, r in testable.sort_values("delta_z", ascending=False).iterrows():
        idx = r.name
        p   = sig.loc[idx, "empirical_pval"]
        fdr = sig.loc[idx, "empirical_fdr"]
        if not np.isfinite(p):
            continue
        star = "**" if fdr < 0.01 else ("*" if fdr < 0.05 else "")
        print(f"{r['dep_gene']:12s} {r['paralog_gene']:12s} "
              f"{r['delta_z']:+7.3f} {p:10.4f} {fdr:10.4f}  {star}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-line", choices=list(CELL_LINE_FILES.keys()), required=True)
    args = parser.parse_args()
    run(args.cell_line)


if __name__ == "__main__":
    main()
