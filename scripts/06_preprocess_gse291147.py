"""
06_preprocess_gse291147.py
--------------------------
Converts GSE291147 (CRISPRi Perturb-seq in melanoma cells, Nature 2026) raw
count data into a pseudobulk z-normalised h5ad compatible with
02_compute_delta_z.py.

Dataset overview
~~~~~~~~~~~~~~~~
  143 vemurafenib-resistance genes + NO-TARGET controls, CRISPRi knockdown
  in melanoma cells, profiled by multi-modal single-cell sequencing with
  DMSO or vemurafenib treatment.

  Files used (data/raw/):
    GSE291147_RNA_matrix.mtx / _rownames.csv / _colnames.csv
      Gene expression matrix (60,706 genes × 100,590 cells). Converted from
      GSE291147_Dual_omics_RNA_gene_count_matrix.RDS by 06_convert_rds.R.

    GSE291147_Tri_sgRNA_matrix.mtx / _rownames.csv / _colnames.csv
      Guide RNA capture matrix (155 guides × 158,432 cells). Converted from
      GSE291147_Tri_omics_sgRNA_count_matrix.RDS by 06_convert_rds.R.
      Used to assign each cell to its knocked-down gene.

    GSE291147_Dual_omics_RNA_cell_metadata.csv.gz
      Per-cell metadata including drug condition (DMSO / vemurafenib).

Cell-to-perturbation assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The 20-character sequence after the barcode dot is shared between the RNA
  matrix (dual-omics experiment) and the sgRNA matrix (tri-omics experiment)
  for the same physical cell.  For each unique 20-char sequence, the guide
  with the highest cumulative UMI count is taken as the perturbation.
  99.8 % of RNA cells receive an unambiguous assignment this way.

Prerequisites
~~~~~~~~~~~~~
1. Download raw files:
       python scripts/00_download_data.py --geo
   Plus the Tri-omics sgRNA matrix (small, 2.8 MB):
       BASE=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE291nnn/GSE291147/suppl
       curl -L -o data/raw/GSE291147_Tri_omics_sgRNA_count_matrix.RDS \\
         ${BASE}/GSE291147_Tri_omics_sgRNA_count_matrix.RDS

2. Convert RDS → Matrix Market:
       Rscript scripts/06_convert_rds.R
   Then convert the Tri-omics sgRNA RDS (already done if you ran the helper
   Rscript line above, otherwise run):
       Rscript -e "
         suppressPackageStartupMessages(library(Matrix))
         out <- 'data/raw/GSE291147_Tri_sgRNA_matrix'
         mat <- as(readRDS('data/raw/GSE291147_Tri_omics_sgRNA_count_matrix.RDS'),
                   'CsparseMatrix')
         writeMM(mat, paste0(out,'.mtx'))
         write.csv(data.frame(name=rownames(mat)),paste0(out,'_rownames.csv'),row.names=F,quote=F)
         write.csv(data.frame(name=colnames(mat)),paste0(out,'_colnames.csv'),row.names=F,quote=F)
       "

3. Run this script:
       python scripts/06_preprocess_gse291147.py [--condition dmso]

Output (data/raw/)
~~~~~~~~~~~~~~~~~~
  melanoma_pseudobulk_normalized.h5ad
     obs.index = perturbation label (gene name or "non-targeting")
     var["gene_name"] = gene symbol (ENSG ID from source)
     X = z-scored pseudobulk log1p(CPM) expression
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scipy.io
import scipy.sparse as sp
import mygene
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

OUTPUT_NAME    = "melanoma_pseudobulk_normalized.h5ad"
CTRL_LABEL_OUT = "non-targeting"   # must match CTRL_LABEL in 02_compute_delta_z.py
MIN_CELLS      = 3                 # minimum cells per perturbation to include
NTC_LABELS     = {"NO-TARGET", "non-targeting", "NTC", "non_targeting",
                  "negative_control", "CTRL", "control"}


# ── Load helpers ──────────────────────────────────────────────────────────────
def load_mtx(prefix: str) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Load a Matrix Market file + companion rownames/colnames CSVs."""
    mtx_path  = DATA_DIR / f"{prefix}.mtx"
    rows_path = DATA_DIR / f"{prefix}_rownames.csv"
    cols_path = DATA_DIR / f"{prefix}_colnames.csv"

    missing = [p for p in (mtx_path, rows_path, cols_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Matrix Market files not found:\n"
            + "\n".join(f"  {p.name}" for p in missing)
            + "\nRun: Rscript scripts/06_convert_rds.R"
        )

    mat       = scipy.io.mmread(mtx_path).tocsr()
    row_names = pd.read_csv(rows_path)["name"].tolist()
    col_names = pd.read_csv(cols_path)["name"].tolist()
    print(f"    {mat.shape[0]:,} rows × {mat.shape[1]:,} cols")
    return mat, row_names, col_names


# ── Guide assignment ──────────────────────────────────────────────────────────
def build_seq_to_guide(
    sgrna_mat: sp.csr_matrix,
    guide_names: list[str],
    sgrna_barcodes: list[str],
) -> dict[str, str]:
    """Map 20-char cell barcode sequence → dominant guide target gene.

    For each unique barcode sequence (shared between RNA and sgRNA modalities),
    accumulate total UMIs per guide across all sgRNA-matrix cells with that
    sequence, then take the guide with the highest total.
    Returns a dict: sequence (str) → guide label (str, e.g. 'BRAF_1').
    """
    sgrna_mat_csc = sgrna_mat.tocsc()

    seq_counts: dict[str, dict[str, float]] = {}
    for i, bc in enumerate(tqdm(sgrna_barcodes, desc="  guide assignment", ncols=80)):
        seq = bc.split(".")[1]
        col = sgrna_mat_csc.getcol(i).toarray().ravel()
        total = col.sum()
        if total == 0:
            continue
        best = guide_names[int(col.argmax())]
        if seq not in seq_counts:
            seq_counts[seq] = {}
        seq_counts[seq][best] = seq_counts[seq].get(best, 0.0) + total

    return {seq: max(gd, key=gd.get) for seq, gd in seq_counts.items()}


def guide_to_gene(guide_label: str) -> str:
    """Strip the guide number suffix: 'BRAF_1' → 'BRAF'."""
    return guide_label.rsplit("_", 1)[0]


# ── Pseudobulk + normalisation ────────────────────────────────────────────────
def pseudobulk_aggregate(
    X: np.ndarray,
    labels: np.ndarray,
    unique_perts: list[str],
) -> np.ndarray:
    pb = np.zeros((len(unique_perts), X.shape[1]), dtype=np.float32)
    for i, pert in enumerate(tqdm(unique_perts, desc="  pseudobulk", ncols=80)):
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


# ── ENSG ID → gene symbol mapping ────────────────────────────────────────────
def ensg_to_symbols(ensg_ids: list[str]) -> list[str]:
    """Convert ENSG IDs (possibly versioned, e.g. 'ENSG00000223972.5') to
    gene symbols using the mygene.info service.  IDs that cannot be mapped
    are kept as their original ENSG string.
    """
    mg = mygene.MyGeneInfo()
    # Strip version suffix before querying
    stripped = [eid.split(".")[0] for eid in ensg_ids]
    results  = mg.querymany(
        stripped,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        verbose=False,
        as_dataframe=False,
    )
    # Build id → symbol dict (take the first hit per query id)
    id_to_sym: dict[str, str] = {}
    for hit in results:
        qid = hit.get("query", "")
        if qid not in id_to_sym and "symbol" in hit:
            id_to_sym[qid] = hit["symbol"]

    mapped = [id_to_sym.get(eid.split(".")[0], eid) for eid in ensg_ids]
    n_mapped = sum(1 for m, e in zip(mapped, ensg_ids) if m != e)
    print(f"  {n_mapped:,}/{len(ensg_ids):,} ENSG IDs mapped to gene symbols")
    return mapped


# ── Main ──────────────────────────────────────────────────────────────────────
def process(condition: str) -> None:
    out_path = DATA_DIR / OUTPUT_NAME
    if out_path.exists():
        print(f"[skip] {OUTPUT_NAME} already exists.")
        return

    # ── 1. Load RNA count matrix (60,706 genes × 100,590 cells) ──────────────
    print("\nLoading RNA count matrix ...")
    rna_mat, gene_names, rna_barcodes = load_mtx("GSE291147_RNA_matrix")
    # rna_mat is genes × cells (R convention); we'll transpose per-cell later

    # ── 2. Load cell metadata (condition filter) ──────────────────────────────
    meta_path = DATA_DIR / "GSE291147_Dual_omics_RNA_cell_metadata.csv.gz"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path.name} not found.\n"
            "Run: python scripts/00_download_data.py --geo"
        )
    print("Loading cell metadata ...")
    meta = pd.read_csv(meta_path, index_col=0)
    print(f"  {len(meta):,} cells, conditions: {meta['conditions'].value_counts().to_dict()}")

    # ── 3. Build barcode-seq → guide mapping (from Tri-omics sgRNA matrix) ────
    tri_sgrna_prefix = "GSE291147_Tri_sgRNA_matrix"
    print("\nLoading Tri-omics sgRNA matrix for guide assignment ...")
    sgrna_mat, guide_names, sgrna_bcs = load_mtx(tri_sgrna_prefix)
    print("Building guide assignment map ...")
    seq_to_guide = build_seq_to_guide(sgrna_mat, guide_names, sgrna_bcs)
    print(f"  {len(seq_to_guide):,} unique barcode sequences mapped to guides")

    # ── 4. Assign perturbation to each RNA cell ───────────────────────────────
    print("\nAssigning perturbation labels to RNA cells ...")
    assignments: dict[str, str] = {}
    for bc in rna_barcodes:
        seq  = bc.split(".")[1]
        if seq in seq_to_guide:
            gene = guide_to_gene(seq_to_guide[seq])
            # Normalise NTC labels → standard "non-targeting"
            assignments[bc] = CTRL_LABEL_OUT if gene in NTC_LABELS else gene

    n_assigned = len(assignments)
    print(f"  Assigned: {n_assigned:,} / {len(rna_barcodes):,} cells ({n_assigned/len(rna_barcodes):.1%})")

    # ── 5. Condition filtering ─────────────────────────────────────────────────
    if condition != "all":
        keep_bcs = [
            bc for bc in rna_barcodes
            if bc in assignments
            and bc in meta.index
            and condition.lower() in meta.loc[bc, "conditions"].lower()
        ]
        print(f"  After '{condition}' filter: {len(keep_bcs):,} cells")
    else:
        keep_bcs = [bc for bc in rna_barcodes if bc in assignments]

    if len(keep_bcs) == 0:
        raise RuntimeError(
            f"No cells remaining after condition filter '{condition}'. "
            "Try --condition all."
        )

    # ── 6. Slice RNA matrix → cells × genes ───────────────────────────────────
    print("\nBuilding cells × genes matrix ...")
    bc_to_col = {bc: i for i, bc in enumerate(rna_barcodes)}
    keep_cols  = np.array([bc_to_col[bc] for bc in keep_bcs])
    X_sub      = rna_mat[:, keep_cols].T.toarray().astype(np.float32)
    pert_labels = np.array([assignments[bc] for bc in keep_bcs])
    print(f"  {X_sub.shape[0]:,} cells × {X_sub.shape[1]:,} genes")

    # ── 7. Pseudobulk aggregation ─────────────────────────────────────────────
    print("Aggregating to pseudobulk ...")
    counts       = pd.Series(pert_labels).value_counts()
    unique_perts = [p for p, n in counts.items() if n >= MIN_CELLS]
    n_ctrl = counts.get(CTRL_LABEL_OUT, 0)
    print(f"  Perturbations ≥{MIN_CELLS} cells: {len(unique_perts):,}")
    print(f"  Non-targeting control cells:  {n_ctrl:,}")

    pb = pseudobulk_aggregate(X_sub, pert_labels, unique_perts)

    # ── 8. log1p(CPM) + z-score ───────────────────────────────────────────────
    print("Normalising and z-scoring ...")
    pb_z = lognorm_zscore(pb)

    # ── 9. Map ENSG IDs → gene symbols ───────────────────────────────────────
    print("Mapping ENSG IDs to gene symbols ...")
    gene_symbols = ensg_to_symbols(gene_names)

    # ── 10. Save h5ad ─────────────────────────────────────────────────────────
    obs_df    = pd.DataFrame({"gene_name": unique_perts}, index=unique_perts)
    var_df    = pd.DataFrame({"gene_name": gene_symbols}, index=gene_symbols)
    out_adata = ad.AnnData(X=pb_z.astype(np.float32), obs=obs_df, var=var_df)

    print(f"\nSaving → {OUTPUT_NAME}")
    print(f"  {out_adata.n_obs} perturbations × {out_adata.n_vars} genes")
    out_adata.write_h5ad(out_path)
    print("  Done.\n")
    print("Next: python scripts/02_compute_delta_z.py --cell-line melanoma")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess GSE291147 melanoma Perturb-seq data to pseudobulk h5ad."
    )
    parser.add_argument(
        "--condition",
        choices=["dmso", "vemurafenib", "all"],
        default="dmso",
        help=(
            "Drug condition to include. 'dmso' (default) gives a drug-free "
            "baseline comparable to Replogle/X-Atlas. 'all' uses every cell."
        ),
    )
    args = parser.parse_args()
    process(args.condition)


if __name__ == "__main__":
    main()
