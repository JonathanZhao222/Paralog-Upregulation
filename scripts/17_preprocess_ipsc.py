"""
17_preprocess_ipsc.py
---------------------
Converts the KOLF2.1J iPSC CRISPRi perturbation atlas (Nature Biotech 2026,
Figshare doi:10.25452/figshare.plus.27261219) into a pseudobulk z-normalised
h5ad compatible with 02_compute_delta_z.py.

Pipeline (standard)
~~~~~~~~~~~~~~~~~~~~
  1. Load the downloaded h5ad (single-cell raw counts or pseudobulk)
  2. If single-cell: aggregate sum per perturbation → pseudobulk matrix
  3. Normalise: log1p(CPM) per pseudobulk row
  4. Z-score each gene across all perturbations
  5. Save as AnnData(obs.index=pert_label, var['gene_name']=symbol)
     with control rows relabelled to "non-targeting"

Pipeline (with --regress-cell-cycle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Load h5ad
  2. Sparse log1p(CPM) normalisation at single-cell level
  3. Score cells: S_score, G2M_score (mean expression of Tirosh 2015 gene sets)
  4. Fit global regression: X ~ 1 + S_score + G2M_score  (3 x n_genes matrix)
  5. Pseudobulk mean of log-normalised expression
  6. Subtract cell-cycle component at pseudobulk level:
       pb_corrected[p] = mean_p(X) - β_S·mean_p(S) - β_G2M·mean_p(G2M)
     (algebraically equivalent to per-cell regression then averaging)
  7. Z-score across perturbations

Output
~~~~~~
  data/raw/iPSC_KOLF2_pseudobulk_normalized.h5ad          (default)
  data/raw/iPSC_KOLF2_pseudobulk_cc_corrected.h5ad        (with --regress-cell-cycle)

Usage
~~~~~
  python scripts/17_preprocess_ipsc.py
  python scripts/17_preprocess_ipsc.py --regress-cell-cycle
  python scripts/17_preprocess_ipsc.py --input data/raw/my_ipsc_file.h5ad --output data/raw/out.h5ad
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

CTRL_LABEL     = "non-targeting"
MIN_CELLS      = 3
DEFAULT_INPUT  = DATA_DIR / "iPSC_KOLF2_raw.h5ad"
OUTPUT_NAME    = "iPSC_KOLF2_pseudobulk_normalized.h5ad"
OUTPUT_NAME_CC = "iPSC_KOLF2_pseudobulk_cc_corrected.h5ad"

PERT_COL_CANDIDATES = [
    "gene", "target_gene_name", "gene_name", "perturbation",
    "guide_target", "gene_id", "targeted_gene", "target",
]
CTRL_CANDIDATES = [
    "non-targeting", "non_targeting", "negative_control",
    "control", "CTRL", "safe-targeting", "NegCtrl",
]

# ── Cell-cycle gene lists (Tirosh et al. 2015 / Seurat / Scanpy standard) ─────
S_GENES = [
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1",
    "GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51",
    "RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1",
    "CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
]
G2M_GENES = [
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","SMC4","CCNB2","CKAP2L",
    "CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B",
    "HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2",
    "DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1","ANLN",
    "LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def strip_guide_suffix(labels: pd.Series) -> pd.Series:
    stripped = labels.str.replace(r"_\d+$", "", regex=True)
    if stripped.nunique() < labels.nunique():
        print(f"  Guide suffix stripped: {labels.nunique()} guides → {stripped.nunique()} genes")
        return stripped
    return labels


def detect_pert_labels(adata: ad.AnnData) -> pd.Series:
    for c in PERT_COL_CANDIDATES:
        if c in adata.obs.columns:
            print(f"  Perturbation column: '{c}'")
            return strip_guide_suffix(adata.obs[c].astype(str))
    for c in adata.obs.columns:
        sample = adata.obs[c].dropna().astype(str).head(50)
        if sample.str.match(r"^[A-Z][A-Z0-9\-]{1,15}$").mean() > 0.6:
            print(f"  Perturbation column (inferred): '{c}'")
            return strip_guide_suffix(adata.obs[c].astype(str))
    raise ValueError(
        f"Cannot detect perturbation column. obs columns: {list(adata.obs.columns)}"
    )


def detect_ctrl(labels: pd.Series) -> str:
    unique = set(labels.unique())
    for c in CTRL_CANDIDATES:
        if c in unique:
            return c
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
    if pd.Series(adata.var.index[:30]).str.match(r"^[A-Za-z][A-Za-z0-9\-\.]{1,20}$").mean() > 0.7:
        return list(adata.var.index.astype(str))
    raise ValueError(
        f"Cannot find gene symbols. var columns: {list(adata.var.columns)}"
    )


def is_pseudobulk(adata: ad.AnnData) -> bool:
    X_sample = to_dense(adata.X[:min(10, adata.n_obs)])
    has_floats = not np.all(X_sample == np.floor(X_sample))
    return adata.n_obs < 50_000 and has_floats


# ── Standard pseudobulk path ──────────────────────────────────────────────────

def pseudobulk_aggregate(X, labels: np.ndarray, unique_perts: list) -> np.ndarray:
    """Sum raw counts per perturbation using sparse matrix multiply."""
    pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
    cell_idx    = np.array([pert_to_idx.get(l, -1) for l in labels])
    valid       = cell_idx >= 0
    cell_idx    = cell_idx[valid]
    n_perts     = len(unique_perts)
    n_valid     = int(valid.sum())

    indicator = sp.csr_matrix(
        (np.ones(n_valid, dtype=np.float32), (cell_idx, np.arange(n_valid))),
        shape=(n_perts, n_valid),
    )
    X_valid = X[valid]
    if not sp.issparse(X_valid):
        X_valid = sp.csr_matrix(X_valid)

    print(f"  Matrix multiply: ({n_perts} × {n_valid}) @ ({n_valid} × {X.shape[1]}) ...")
    return (indicator @ X_valid).toarray().astype(np.float32)


def lognorm_zscore(pb: np.ndarray) -> np.ndarray:
    """log1p(CPM) normalise pseudobulk sums, then z-score across perturbations."""
    lib = pb.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    pb_norm = np.log1p(pb / lib * 1e6).astype(np.float64)
    return _zscore(pb_norm)


# ── Cell-cycle regression path ────────────────────────────────────────────────

def lognorm_sparse(X_raw) -> sp.csr_matrix:
    """
    Sparse log1p(CPM) normalisation.  Zeros remain zero (log1p(0)=0), so
    sparsity is fully preserved — avoids densifying the 2.66M × 37K matrix.
    """
    if not sp.issparse(X_raw):
        X_raw = sp.csr_matrix(X_raw)
    X = X_raw.astype(np.float32).copy()
    lib_sizes = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    lib_sizes[lib_sizes == 0] = 1.0
    scale = (1e6 / lib_sizes).astype(np.float32)
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X


def score_cell_cycle(X_lognorm: sp.csr_matrix,
                     gene_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-cell S_score and G2M_score as the mean log-normalised expression
    of the Tirosh et al. 2015 S-phase and G2M-phase gene sets respectively.
    Only genes detected in the data are used.
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    s_idx    = [gene_idx[g] for g in S_GENES   if g in gene_idx]
    g2m_idx  = [gene_idx[g] for g in G2M_GENES if g in gene_idx]
    print(f"  Cell-cycle genes found: {len(s_idx)}/{len(S_GENES)} S-phase, "
          f"{len(g2m_idx)}/{len(G2M_GENES)} G2M-phase")

    s_scores   = np.asarray(X_lognorm[:, s_idx].mean(axis=1)).ravel().astype(np.float32)
    g2m_scores = np.asarray(X_lognorm[:, g2m_idx].mean(axis=1)).ravel().astype(np.float32)
    return s_scores, g2m_scores


def pseudobulk_mean_cc_corrected(
    X_lognorm: sp.csr_matrix,
    labels: np.ndarray,
    unique_perts: list,
    s_scores: np.ndarray,
    g2m_scores: np.ndarray,
) -> np.ndarray:
    """
    Pseudobulk mean of log-normalised expression with cell-cycle regression.

    Key identity (linearity of means):
        mean_p(X_corrected) = mean_p(X) − β_S · mean_p(S) − β_G2M · mean_p(G2M)

    where β_S, β_G2M come from fitting X ~ 1 + S_score + G2M_score globally
    across all cells.  This avoids materialising the full per-cell correction
    matrix (would be ~394 GB for 2.66M × 37K).
    """
    pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
    n_perts     = len(unique_perts)
    n_genes     = X_lognorm.shape[1]

    cell_idx = np.array([pert_to_idx.get(l, -1) for l in labels])
    valid    = cell_idx >= 0
    cidx_v   = cell_idx[valid]
    n_valid  = int(valid.sum())

    # ── Pseudobulk mean of log-normalised expression ──────────────────────
    indicator = sp.csr_matrix(
        (np.ones(n_valid, dtype=np.float32), (cidx_v, np.arange(n_valid))),
        shape=(n_perts, n_valid),
    )
    X_valid = X_lognorm[valid]

    print(f"  Pseudobulk matrix multiply (lognorm) ...")
    pb_sum    = (indicator @ X_valid).toarray().astype(np.float64)   # n_perts × n_genes
    counts    = np.bincount(cidx_v, minlength=n_perts).astype(np.float64)
    inv_cnt   = np.where(counts > 0, 1.0 / counts, 0.0)
    pb_mean   = pb_sum * inv_cnt[:, None]                             # n_perts × n_genes

    # ── Global regression coefficients ────────────────────────────────────
    print("  Fitting global cell-cycle regression (3 × n_genes) ...")
    s_v   = s_scores[valid].astype(np.float64)
    g2m_v = g2m_scores[valid].astype(np.float64)
    Z     = np.column_stack([np.ones(n_valid), s_v, g2m_v])          # n_valid × 3

    ZtZ_inv = np.linalg.inv(Z.T @ Z)                                 # 3 × 3
    # sparse(n_genes × n_valid) @ dense(n_valid × 3) — scipy handles this efficiently
    ZtX     = (X_valid.T @ Z).T                                       # 3 × n_genes
    beta    = ZtZ_inv @ ZtX                                           # 3 × n_genes

    # ── Per-perturbation mean cell-cycle scores ────────────────────────────
    pb_s   = np.bincount(cidx_v, weights=s_v,   minlength=n_perts) * inv_cnt
    pb_g2m = np.bincount(cidx_v, weights=g2m_v, minlength=n_perts) * inv_cnt

    # ── Apply correction (keeps mean expression, removes cell-cycle variance) ──
    pb_corrected = pb_mean - pb_s[:, None] * beta[1] - pb_g2m[:, None] * beta[2]

    return pb_corrected.astype(np.float32)


# ── Shared z-score ────────────────────────────────────────────────────────────

def _zscore(pb_norm: np.ndarray) -> np.ndarray:
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
    parser.add_argument("--regress-cell-cycle", action="store_true",
                        help="Regress out S_score and G2M_score at single-cell level "
                             "before pseudobulking (Tirosh et al. 2015 gene sets). "
                             "Output is saved as iPSC_KOLF2_pseudobulk_cc_corrected.h5ad "
                             "unless --output is specified.")
    args = parser.parse_args()

    in_path = args.input
    if args.output:
        out_path = args.output
    elif args.regress_cell_cycle:
        out_path = in_path.parent / OUTPUT_NAME_CC
    else:
        out_path = in_path.parent / OUTPUT_NAME

    if not in_path.exists():
        print(
            f"[ERROR] Input file not found: {in_path}\n\n"
            "Download instructions:\n"
            "  1. Visit https://plus.figshare.com/articles/dataset/27261219\n"
            "  2. Download the processed h5ad file\n"
            f"  3. Save it to: {in_path}\n"
            "  4. Re-run this script"
        )
        return

    if out_path.exists():
        print(f"[skip] {out_path.name} already exists. Delete it to reprocess.")
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

    raw = adata.layers.get("counts", adata.X)

    if already_pb:
        # Already pseudobulk — cell-cycle regression not applicable
        if args.regress_cell_cycle:
            print("[warn] Data appears to be already pseudobulked; "
                  "--regress-cell-cycle requires single-cell input. "
                  "Proceeding without regression.")
        unique_perts = list(pert_labels.values)
        pb = to_dense(raw)
        print(f"  {len(unique_perts)} perturbation rows (already pseudobulk)")
        print("  log1p(CPM) normalising + z-scoring ...")
        pb_z = lognorm_zscore(pb)

    elif args.regress_cell_cycle:
        # ── Cell-cycle regression path ────────────────────────────────────
        counts_df = pd.Series(pert_labels.values).value_counts()
        unique_perts = [p for p, n in counts_df.items() if n >= MIN_CELLS]
        print(f"  Perturbations with ≥{MIN_CELLS} cells: {len(unique_perts):,}")

        print("  Sparse log1p(CPM) normalising single cells ...")
        X_lognorm = lognorm_sparse(raw)

        print("  Scoring cell cycle ...")
        s_scores, g2m_scores = score_cell_cycle(X_lognorm, gene_names)
        print(f"  S_score:   mean={s_scores.mean():.4f}, std={s_scores.std():.4f}")
        print(f"  G2M_score: mean={g2m_scores.mean():.4f}, std={g2m_scores.std():.4f}")

        pb = pseudobulk_mean_cc_corrected(
            X_lognorm, pert_labels.values, unique_perts, s_scores, g2m_scores
        )

        print("  Z-scoring across perturbations ...")
        pb_z = _zscore(pb.astype(np.float64))

    else:
        # ── Standard path ─────────────────────────────────────────────────
        counts_df = pd.Series(pert_labels.values).value_counts()
        unique_perts = [p for p, n in counts_df.items() if n >= MIN_CELLS]
        print(f"  Perturbations with ≥{MIN_CELLS} cells: {len(unique_perts):,}")
        pb = pseudobulk_aggregate(raw, pert_labels.values, np.array(unique_perts))
        print("  log1p(CPM) normalising + z-scoring ...")
        pb_z = lognorm_zscore(pb)

    # ── Standardise control label and save ───────────────────────────────────
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

    cc_flag = " --regress-cell-cycle" if args.regress_cell_cycle else ""
    out_cl  = "iPSC_cc" if args.regress_cell_cycle else "iPSC"
    print(f"\nNext steps (update CELL_LINE_FILES in 02 / 15 if using cc-corrected output):")
    print(f"  python scripts/02_compute_delta_z.py --cell-line {out_cl}")
    print(f"  python scripts/10_rank_all_pairs.py  --cell-line {out_cl}")
    print(f"  python scripts/15_per_pair_pvalues.py --cell-line {out_cl} --all-pairs")


if __name__ == "__main__":
    main()
