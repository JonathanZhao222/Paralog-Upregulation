"""
07_preprocess_cd4t.py
---------------------
Converts GWCD4i.pseudobulk_merged.h5ad (Zhu R., Dann E. et al. 2025,
genome-scale CRISPRi Perturb-seq in primary human CD4+ T cells) into
per-condition pseudobulk z-normalised h5ad files compatible with
02_compute_delta_z.py.

Dataset overview
~~~~~~~~~~~~~~~~
  ~10,000 genes knocked down by CRISPRi in primary human CD4+ T cells
  from 4 donors under 3 stimulation conditions. 22 million cells total.
  Source: s3://genome-scale-tcell-perturb-seq/marson2025_data/

  File used (data/raw/):
    GWCD4i.pseudobulk_merged.h5ad
      Aggregated UMI counts per (guide × donor × condition).
      obs columns include: perturbed_gene_name, donor_id,
        culture_condition, n_cells, keep_for_DE (quality flag)
      var columns include: gene_name
      X: sum of raw UMI counts (not normalised)

Prerequisites
~~~~~~~~~~~~~
  Download the pseudobulk h5ad (44.6 GB):
      python scripts/00_download_data.py --cd4t
  Or directly on Sherlock:
      wget --progress=dot:giga \\
        -O data/raw/GWCD4i.pseudobulk_merged.h5ad \\
        "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.pseudobulk_merged.h5ad"

Output (data/raw/)
~~~~~~~~~~~~~~~~~~
  cd4t_rest_pseudobulk_normalized.h5ad
  cd4t_stim8hr_pseudobulk_normalized.h5ad
  cd4t_stim48hr_pseudobulk_normalized.h5ad
    obs.index = perturbation label (gene symbol or "non-targeting")
    var["gene_name"] = gene symbol
    X = z-scored pseudobulk log1p(CPM) expression

Memory note
~~~~~~~~~~~
  Loading the 44.6 GB h5ad requires ~100 GB RAM.
  On Sherlock use: #SBATCH --mem=128G (or 256G if it OOMs).
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

INPUT_FILE     = "GWCD4i.pseudobulk_merged.h5ad"
CTRL_LABEL_OUT = "non-targeting"   # must match CTRL_LABEL in 02_compute_delta_z.py
NTC_NAMES      = {"CTRL", "non-targeting", "NTC", "non_targeting",
                  "negative_control", "safe-targeting"}

CONDITIONS = ["Rest", "Stim8hr", "Stim48hr"]
COND_OUTNAME = {
    "Rest":     "cd4t_rest_pseudobulk_normalized.h5ad",
    "Stim8hr":  "cd4t_stim8hr_pseudobulk_normalized.h5ad",
    "Stim48hr": "cd4t_stim48hr_pseudobulk_normalized.h5ad",
}

MIN_CELLS = 10   # minimum total cells across guides+donors to keep a perturbation


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


def process(conditions: list[str]) -> None:
    in_path = DATA_DIR / INPUT_FILE
    if not in_path.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found in {DATA_DIR}.\n"
            "Download with:\n"
            "  python scripts/00_download_data.py --cd4t\n"
            "Or on Sherlock:\n"
            f"  wget --progress=dot:giga -O '{in_path}' \\\n"
            "    'https://genome-scale-tcell-perturb-seq.s3.amazonaws.com"
            "/marson2025_data/GWCD4i.pseudobulk_merged.h5ad'"
        )

    todo = [c for c in conditions if not (DATA_DIR / COND_OUTNAME[c]).exists()]
    if not todo:
        print("All requested conditions already processed — nothing to do.")
        return

    print(f"\nLoading {INPUT_FILE} (this may take several minutes) ...")
    adata = ad.read_h5ad(in_path)
    print(f"  {adata.n_obs:,} pseudobulk rows × {adata.n_vars:,} genes")

    obs = adata.obs.copy()
    X_raw = adata.X
    if sp.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float32)

    gene_col = "perturbed_gene_name"
    if gene_col not in obs.columns:
        raise KeyError(
            f"Expected column '{gene_col}' in obs.\n"
            f"Available columns: {list(obs.columns)}"
        )

    # Normalise NTC label variants to the standard "non-targeting"
    obs[gene_col] = obs[gene_col].astype(str).where(
        ~obs[gene_col].astype(str).isin(NTC_NAMES), CTRL_LABEL_OUT
    )

    # Quality filter
    qual_col = "keep_for_DE"
    if qual_col in obs.columns:
        keep_mask  = obs[qual_col].astype(bool).values
        obs_filt   = obs[keep_mask].reset_index(drop=True)
        X_filt     = X_raw[keep_mask]
        print(f"  After {qual_col} filter: {len(obs_filt):,} rows")
    else:
        print(f"  Warning: '{qual_col}' not found — using all rows")
        obs_filt = obs.reset_index(drop=True)
        X_filt   = X_raw

    gene_names = (
        list(adata.var["gene_name"])
        if "gene_name" in adata.var.columns
        else list(adata.var.index)
    )
    del adata   # free memory before per-condition processing

    for cond in todo:
        out_path = DATA_DIR / COND_OUTNAME[cond]
        print(f"\n── Processing condition: {cond} ──")

        cond_mask = obs_filt["culture_condition"].values == cond
        obs_c = obs_filt[cond_mask]
        X_c   = X_filt[cond_mask]
        print(f"  Rows in this condition: {len(obs_c):,}")

        # Sum UMI counts across all (guide × donor) rows for the same gene
        gene_labels = obs_c[gene_col].values
        n_cells_sum = (
            obs_c.groupby(gene_col)["n_cells"].sum()
            if "n_cells" in obs_c.columns
            else obs_c.groupby(gene_col).size()
        )
        valid = n_cells_sum[n_cells_sum >= MIN_CELLS].index.tolist()
        valid_set = set(valid)
        unique_genes = [g for g in n_cells_sum.index if g in valid_set]
        print(f"  Perturbations ≥{MIN_CELLS} total cells: {len(unique_genes):,}")

        pb = np.zeros((len(unique_genes), X_c.shape[1]), dtype=np.float32)
        for i, gene in enumerate(tqdm(unique_genes, desc="  pseudobulk", ncols=80)):
            row_mask = gene_labels == gene
            pb[i] = X_c[row_mask].sum(axis=0)

        if CTRL_LABEL_OUT not in valid_set:
            print(f"  WARNING: no '{CTRL_LABEL_OUT}' control rows in {cond}!")

        print("  Normalising (log1p CPM) and z-scoring ...")
        pb_z = lognorm_zscore(pb)

        obs_df = pd.DataFrame({"gene_name": unique_genes}, index=unique_genes)
        var_df = pd.DataFrame({"gene_name": gene_names}, index=gene_names)
        out_adata = ad.AnnData(X=pb_z.astype(np.float32), obs=obs_df, var=var_df)

        print(f"  Saving → {COND_OUTNAME[cond]}")
        print(f"    {out_adata.n_obs:,} perturbations × {out_adata.n_vars:,} genes")
        out_adata.write_h5ad(out_path)
        print("  Done.")

    print(f"\nAll conditions processed.")
    print("Next: python scripts/02_compute_delta_z.py --cell-line cd4t_rest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess GWCD4i pseudobulk h5ad into per-condition z-scored h5ad files."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        metavar="CONDITION",
        help=f"Conditions to process: {CONDITIONS} (default: all)",
    )
    args = parser.parse_args()
    process(args.conditions)


if __name__ == "__main__":
    main()
