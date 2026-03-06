"""
01_explore_dataset.py
---------------------
Loads the downloaded AnnData and reports:
  - Shape, obs columns, obs index, var index
  - Perturbation column name and unique labels
  - Non-targeting control label and row count
  - How many sig / non-sig genes are present as perturbations or
    as measured genes in the expression matrix

Run BEFORE 02_compute_delta_z.py to verify the perturbation column
name and control label; update PERT_COL / CTRL_LABEL in that script
if the auto-detection here differs from the defaults.

Usage:
    python scripts/01_explore_dataset.py
"""

import anndata as ad
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
H5AD     = ROOT / "data" / "raw" / "K562_gwps_normalized_bulk_01.h5ad"
SIG_XL   = ROOT / "data" / "raw" / "sig_37_paralog.xlsx"
NONSIG_XL = ROOT / "data" / "raw" / "non_sig_paralog.xlsx"

# Candidate column names used by Replogle et al. for perturbation labels
PERT_COL_CANDIDATES = ["gene", "target_gene_name", "perturbation", "gene_name", "guide_id"]
# Candidate strings used for non-targeting controls
CTRL_CANDIDATES = ["non-targeting", "non_targeting", "control", "negative_control", "CTRL"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_pert_col(obs: pd.DataFrame) -> str | None:
    for c in PERT_COL_CANDIDATES:
        if c in obs.columns:
            return c
    return None


def detect_ctrl_label(unique_perts: set) -> str | None:
    for c in CTRL_CANDIDATES:
        if c in unique_perts:
            return c
    # Fuzzy fallback
    candidates = [p for p in unique_perts if "non" in p.lower() or "ctrl" in p.lower()]
    return candidates[0] if len(candidates) == 1 else None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not H5AD.exists():
        print(f"ERROR: {H5AD} not found.\nRun 00_download_data.py first.")
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {H5AD.name} (backed mode) ...")
    adata = ad.read_h5ad(H5AD, backed="r")

    print("\n=== AnnData overview ===")
    print(adata)
    print(f"\n  obs  : {adata.n_obs} rows  ×  {len(adata.obs.columns)} columns")
    print(f"  var  : {adata.n_vars} genes")

    # ── obs columns ───────────────────────────────────────────────────────────
    print(f"\nobs columns : {list(adata.obs.columns)}")
    print(f"obs index (first 10): {list(adata.obs.index[:10])}")
    print(f"var index (first 10): {list(adata.var.index[:10])}")

    # ── Perturbation column ────────────────────────────────────────────────────
    pert_col = detect_pert_col(adata.obs)
    if pert_col:
        pert_labels = adata.obs[pert_col].tolist()
        print(f"\nPerturbation column detected: '{pert_col}'")
    else:
        pert_labels = adata.obs.index.tolist()
        print("\nNo known perturbation column found in obs — using obs.index as labels.")
        print("  (Update PERT_COL_CANDIDATES in this script if needed.)")

    unique_perts = set(pert_labels)
    print(f"Unique perturbation labels: {len(unique_perts)}")
    print(f"Sample labels: {sorted(unique_perts)[:10]}")

    # ── Control label ──────────────────────────────────────────────────────────
    ctrl_label = detect_ctrl_label(unique_perts)
    if ctrl_label:
        n_ctrl = pert_labels.count(ctrl_label)
        print(f"\nControl label: '{ctrl_label}'  ({n_ctrl} rows)")
    else:
        maybe = [p for p in unique_perts if "non" in p.lower() or "ctrl" in p.lower()]
        print(f"\nCould not auto-detect control label.")
        print(f"  Possible matches: {maybe[:20]}")
        print("  Set CTRL_LABEL manually in 02_compute_delta_z.py.")

    # ── Gene sets ─────────────────────────────────────────────────────────────
    perturbed_genes = unique_perts - ({ctrl_label} if ctrl_label else set())
    measured_genes  = set(adata.var.index)

    print(f"\nGenes with a KD perturbation : {len(perturbed_genes)}")
    print(f"Genes in expression matrix   : {len(measured_genes)}")

    # ── Significant pairs ──────────────────────────────────────────────────────
    sig = pd.read_excel(SIG_XL)
    dep_genes  = set(sig["dep_gene"])
    para_genes = set(sig["paralog_gene"])

    dep_in_lib   = dep_genes  & perturbed_genes
    para_in_mat  = para_genes & measured_genes
    dep_missing  = dep_genes  - perturbed_genes
    para_missing = para_genes - measured_genes

    # A pair is testable only if dep_gene is in the KD library AND paralog is measured
    sig["testable"] = (
        sig["dep_gene"].isin(perturbed_genes) &
        sig["paralog_gene"].isin(measured_genes)
    )
    n_testable_sig = sig["testable"].sum()

    print(f"\n--- Significant paralog pairs (n=37) ---")
    print(f"  dep_genes in KD library          : {len(dep_in_lib)} / {len(dep_genes)}")
    print(f"  paralog_genes in expression mat  : {len(para_in_mat)} / {len(para_genes)}")
    print(f"  Fully testable pairs             : {n_testable_sig} / 37")
    if dep_missing:
        print(f"  dep_genes missing from library   : {sorted(dep_missing)}")
    if para_missing:
        print(f"  paralog_genes missing from mat   : {sorted(para_missing)}")

    # ── Non-significant pairs (bidirectional) ─────────────────────────────────
    nonsig = pd.read_excel(NONSIG_XL)
    g1 = nonsig["para_gene_1"]
    g2 = nonsig["para_gene_2"]

    dir1_testable = (g1.isin(perturbed_genes)) & (g2.isin(measured_genes))
    dir2_testable = (g2.isin(perturbed_genes)) & (g1.isin(measured_genes))

    print(f"\n--- Non-significant paralog pairs (n={len(nonsig):,}) ---")
    print(f"  Testable dir 1 (KD gene_1 → measure gene_2) : {dir1_testable.sum():,}")
    print(f"  Testable dir 2 (KD gene_2 → measure gene_1) : {dir2_testable.sum():,}")
    print(f"  Total testable directions                    : {dir1_testable.sum() + dir2_testable.sum():,}")

    # ── Summary for user ──────────────────────────────────────────────────────
    print("\n=== Action items before running 02_compute_delta_z.py ===")
    print(f"  PERT_COL  = '{pert_col or 'UPDATE_ME'}'")
    print(f"  CTRL_LABEL = '{ctrl_label or 'UPDATE_ME'}'")
    print("  Update these constants at the top of 02_compute_delta_z.py if they differ.")

    adata.file.close()


if __name__ == "__main__":
    main()
