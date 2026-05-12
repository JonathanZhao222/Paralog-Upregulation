"""
08_preprocess_neuron.py
-----------------------
Converts the PerturBase iPSC-derived neuron CRISPRi Perturb-seq dataset
(Tian et al. 2021, GSE152988) into a pseudobulk z-normalised h5ad
compatible with 02_compute_delta_z.py.

Source
~~~~~~
  PerturBase data index: CRISPRi
  Paper: "Genome-wide CRISPRi/a screens in human neurons link lysosomal
          failure to ferroptosis" (Tian et al. 2021, Nature Neuroscience)
  185 perturbations, 32 300 cells, 33 538 genes
  Download: python scripts/00_download_data.py --neuron

Format of raw input (data/raw/neuron_raw.h5ad)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  obs['gene']  : perturbation label (gene symbol or 'CTRL')
  obs['batch'] : batch identifier (1–4, pooled across batches here)
  var.index    : gene symbols
  X            : raw UMI counts (float32, sparse)

Output: data/raw/neuron_pseudobulk_normalized.h5ad
  obs.index     = perturbation label
  obs['gene_name'] = perturbation label
  var['gene_name'] = gene symbol
  X             = z-scored log1p(CPM) pseudobulk expression

Usage:
    python scripts/08_preprocess_neuron.py
"""

import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

INPUT_NAME  = "neuron_raw.h5ad"
OUTPUT_NAME = "neuron_pseudobulk_normalized.h5ad"
CTRL_LABEL  = "non-targeting"   # must match CTRL_LABEL in 02_compute_delta_z.py
SOURCE_CTRL = "CTRL"            # label used in the raw file
MIN_CELLS   = 3


def main() -> None:
    in_path  = DATA_DIR / INPUT_NAME
    out_path = DATA_DIR / OUTPUT_NAME

    if not in_path.exists():
        print(f"[error] {INPUT_NAME} not found.")
        print("Run: python scripts/00_download_data.py --neuron")
        return

    if out_path.exists():
        print(f"[skip] {OUTPUT_NAME} already exists.")
        return

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"Loading {INPUT_NAME} ...")
    adata = ad.read_h5ad(in_path)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # ── 2. Extract raw counts ─────────────────────────────────────────────────
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # ── 3. Perturbation labels ────────────────────────────────────────────────
    pert_labels = adata.obs["gene"].astype(str).values
    # Normalise control label
    pert_labels = np.where(pert_labels == SOURCE_CTRL, CTRL_LABEL, pert_labels)

    # Drop combinatorial perturbations (comma-separated) — not useful for Δz
    is_single = np.array(["," not in p for p in pert_labels])
    X          = X[is_single]
    pert_labels = pert_labels[is_single]
    print(f"  After dropping combinatorial: {X.shape[0]:,} cells")

    # ── 4. Gene symbols from var.index ────────────────────────────────────────
    gene_names = list(adata.var.index)

    # ── 5. Pseudobulk aggregation ─────────────────────────────────────────────
    print("Aggregating to pseudobulk ...")
    counts       = pd.Series(pert_labels).value_counts()
    unique_perts = [p for p, n in counts.items() if n >= MIN_CELLS]
    n_ctrl       = counts.get(CTRL_LABEL, 0)
    print(f"  Perturbations ≥{MIN_CELLS} cells: {len(unique_perts):,}")
    print(f"  Non-targeting control cells:  {n_ctrl:,}")

    pb = np.zeros((len(unique_perts), X.shape[1]), dtype=np.float32)
    for i, pert in enumerate(tqdm(unique_perts, desc="  pseudobulk", ncols=80)):
        mask   = pert_labels == pert
        pb[i]  = X[mask].sum(axis=0)

    # ── 6. log1p(CPM) + z-score ───────────────────────────────────────────────
    print("Normalising (log1p CPM) ...")
    lib = pb.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    pb_norm = np.log1p(pb / lib * 1e6).astype(np.float64)

    print("Z-scoring per gene ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu  = np.nanmean(pb_norm, axis=0)
        std = np.nanstd(pb_norm,  axis=0)
    std[std == 0] = np.nan
    pb_z = (pb_norm - mu) / std
    pb_z[~np.isfinite(pb_z)] = np.nan

    # ── 7. Save ───────────────────────────────────────────────────────────────
    obs_df    = pd.DataFrame({"gene_name": unique_perts}, index=unique_perts)
    var_df    = pd.DataFrame({"gene_name": gene_names},   index=gene_names)
    out_adata = ad.AnnData(X=pb_z.astype(np.float32), obs=obs_df, var=var_df)

    print(f"\nSaving → {OUTPUT_NAME}")
    print(f"  {out_adata.n_obs} perturbations × {out_adata.n_vars} genes")
    out_adata.write_h5ad(out_path)
    print("  Done.")
    print("\nNext: python scripts/02_compute_delta_z.py --cell-line neuron")


if __name__ == "__main__":
    main()
