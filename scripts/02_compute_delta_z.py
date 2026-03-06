"""
02_compute_delta_z.py
---------------------
Computes the paralog upregulation metric (Δz) for every testable
significant and non-significant paralog pair.

Metric — Δz (delta z-score)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Replogle 2022 normalized_bulk AnnData contains gemgroup
z-normalised pseudobulk expression values.  For each gene, values are
centred on the gemgroup mean (≈ baseline expression) and expressed in
standard-deviation units.

For a pair (dep_gene → paralog_gene):

    Δz = mean z-score of paralog_gene across all dep_gene KD rows
       − mean z-score of paralog_gene across all non-targeting control rows

Interpretation:
  Δz > 0 → paralog is expressed ABOVE the control baseline when dep_gene is KD'd
  Δz < 0 → paralog is expressed BELOW baseline
  Δz ≈ 0 → no change

Outputs
~~~~~~~
  results/sig_results.csv     — one row per sig pair
  results/nonsig_results.csv  — one row per testable non-sig direction

Usage:
    python scripts/02_compute_delta_z.py

Replogle 2022 AnnData structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
obs.index format: "{id}_{gene}_{guide}_{ensembl}"
  e.g. "0_A1BG_P1_ENSG00000121410"  → gene = "A1BG"
       "10748_non-targeting_non-targeting_non-targeting"  → control row

Controls are flagged by obs["core_control"] == True and parse to the
gene label "non-targeting".  CTRL_LABEL below must match that value.
"""

import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CTRL_LABEL = "non-targeting"  # gene label parsed from control obs.index rows

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
H5AD       = ROOT / "data" / "raw" / "K562_gwps_normalized_bulk_01.h5ad"
SIG_XL     = ROOT / "data" / "raw" / "sig_37_paralog.xlsx"
NONSIG_XL  = ROOT / "data" / "raw" / "non_sig_paralog.xlsx"
RESULTS    = ROOT / "results"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_adata(path: Path):
    """
    Load the AnnData and return:
      X           – dense float32 array (n_perturbations × n_genes)
      pert_labels – list of gene-name strings aligned to rows of X,
                    parsed from obs.index as the 2nd underscore-delimited field
      gene_index  – dict {gene_symbol: column_index}
    """
    print(f"Loading {path.name} ...")
    adata = ad.read_h5ad(path)

    # Parse gene name from obs.index: "{id}_{gene}_{guide}_{ensembl}"
    # Controls parse to "non-targeting" which matches CTRL_LABEL.
    pert_labels = [idx.split("_")[1] for idx in adata.obs.index]

    # Dense matrix (pseudobulk is small enough to fit in RAM)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    # Replace ±inf (can arise from log-transform of zeros) with NaN so
    # they are excluded from mean calculations
    X[~np.isfinite(X)] = np.nan

    # var.index is Ensembl IDs; gene symbols are in var['gene_name']
    gene_index = {g: i for i, g in enumerate(adata.var["gene_name"])}

    n_ctrl = pert_labels.count(CTRL_LABEL)
    print(
        f"  {X.shape[0]} perturbation rows, {X.shape[1]} genes, "
        f"{n_ctrl} non-targeting control rows."
    )
    if n_ctrl == 0:
        raise ValueError(
            f"No control rows found (label='{CTRL_LABEL}'). "
            "Check CTRL_LABEL constant."
        )
    return X, pert_labels, gene_index


# ── Precompute per-perturbation mean expression ───────────────────────────────
def build_pert_mean(X: np.ndarray, pert_labels: list) -> dict[str, np.ndarray]:
    """
    Returns a dict {perturbation_label: mean_expression_vector}.
    Averages over multiple rows when a perturbation appears in >1 gemgroup.
    """
    from collections import defaultdict
    rows_by_pert: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(pert_labels):
        rows_by_pert[label].append(i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # suppress all-NaN slice warnings
        pert_mean = {
            label: np.nanmean(X[rows, :], axis=0)
            for label, rows in rows_by_pert.items()
        }
    return pert_mean


# ── Core metric ───────────────────────────────────────────────────────────────
def delta_z(
    pert_mean: dict[str, np.ndarray],
    gene_index: dict[str, int],
    kd_gene: str,
    paralog_gene: str,
    ctrl_vec: np.ndarray,
) -> float | None:
    """
    Returns Δz = mean_z(paralog | kd_gene KD) − mean_z(paralog | control).
    Returns None if kd_gene or paralog_gene is not in the dataset.
    """
    if kd_gene not in pert_mean:
        return None
    if paralog_gene not in gene_index:
        return None
    j = gene_index[paralog_gene]
    z_kd   = pert_mean[kd_gene][j]
    z_ctrl = ctrl_vec[j]
    return float(z_kd - z_ctrl)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    X, pert_labels, gene_index = load_adata(H5AD)

    print("Building per-perturbation mean expression vectors ...")
    pert_mean = build_pert_mean(X, pert_labels)

    # Control vector: mean expression across all non-targeting control rows
    ctrl_vec = pert_mean[CTRL_LABEL]

    # ── Significant pairs ─────────────────────────────────────────────────────
    sig = pd.read_excel(SIG_XL)
    print(f"\nComputing Δz for {len(sig)} significant pairs ...")

    sig_rows = []
    for _, row in tqdm(sig.iterrows(), total=len(sig), desc="sig pairs"):
        dz = delta_z(pert_mean, gene_index, row["dep_gene"], row["paralog_gene"], ctrl_vec)
        sig_rows.append({
            "dep_gene":             row["dep_gene"],
            "paralog_gene":         row["paralog_gene"],
            "para_pair":            row["para_pair"],
            "mean_identical_score": row["mean_identical_score"],
            "aneuploid_loss_chr":   row["aneuploid_loss_chr"],
            "paralog_chr":          row["paralog_chr"],
            "p_value_dep":          row["p_value"],   # original dep significance
            "delta_z":              dz,
            "testable":             dz is not None,
        })

    sig_df = pd.DataFrame(sig_rows)
    out_sig = RESULTS / "sig_results.csv"
    sig_df.to_csv(out_sig, index=False)

    n_testable = sig_df["testable"].sum()
    print(f"  Saved {len(sig_df)} rows ({n_testable} testable) → {out_sig.name}")
    sig_test = sig_df.dropna(subset=["delta_z"])
    print(
        f"  Δz  mean={sig_test['delta_z'].mean():.4f}, "
        f"median={sig_test['delta_z'].median():.4f}, "
        f"fraction > 0: {(sig_test['delta_z'] > 0).mean():.2f}"
    )

    # ── Non-significant pairs (bidirectional) ─────────────────────────────────
    nonsig = pd.read_excel(NONSIG_XL)
    print(f"\nComputing Δz for {len(nonsig):,} non-sig pairs (bidirectional) ...")

    nonsig_rows = []
    for _, row in tqdm(nonsig.iterrows(), total=len(nonsig), desc="non-sig pairs"):
        g1 = row["para_gene_1"]
        g2 = row["para_gene_2"]
        mid_score = row["mean_identical_score"]

        # Direction 1: KD g1 → measure g2
        dz1 = delta_z(pert_mean, gene_index, g1, g2, ctrl_vec)
        if dz1 is not None:
            nonsig_rows.append({
                "dep_gene":             g1,
                "paralog_gene":         g2,
                "mean_identical_score": mid_score,
                "delta_z":              dz1,
                "direction":            "g1_to_g2",
            })

        # Direction 2: KD g2 → measure g1
        dz2 = delta_z(pert_mean, gene_index, g2, g1, ctrl_vec)
        if dz2 is not None:
            nonsig_rows.append({
                "dep_gene":             g2,
                "paralog_gene":         g1,
                "mean_identical_score": mid_score,
                "delta_z":              dz2,
                "direction":            "g2_to_g1",
            })

    nonsig_df = pd.DataFrame(nonsig_rows)
    out_ns = RESULTS / "nonsig_results.csv"
    nonsig_df.to_csv(out_ns, index=False)
    print(f"  Saved {len(nonsig_df):,} rows → {out_ns.name}")
    if not nonsig_df.empty:
        print(
            f"  Δz  mean={nonsig_df['delta_z'].mean():.4f}, "
            f"median={nonsig_df['delta_z'].median():.4f}, "
            f"fraction > 0: {(nonsig_df['delta_z'] > 0).mean():.2f}"
        )
    else:
        print("  No testable non-sig pairs found. Check gene name matching.")

    print("\nDone. Run 03_compare_visualize.py to generate figures.")


if __name__ == "__main__":
    main()
