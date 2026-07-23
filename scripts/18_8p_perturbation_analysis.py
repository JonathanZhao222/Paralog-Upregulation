"""
18_8p_perturbation_analysis.py
------------------------------
For each gene in the chromosome 8p common deletion region, checks whether a
CRISPRi knockdown is available in the iPSC Perturb-seq data and computes its
full transcriptional perturbation signature (Δz across all detected genes).

Optionally correlates each signature against bulk RNA-seq logFC values from
8p-deleted vs normal cells to generate scatter plots equivalent to Fig. 16 in
the original 8p analysis pipeline.

Outputs (saved to results/8p_deletion/ and figures/8p_deletion/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  availability.csv        — which 8p genes are in the iPSC KD library
  signatures.csv          — Δz per gene per 8p KD (rows=genes, cols=8p KDs)
  01_scatter_*.pdf        — one scatter plot per available KD (if --rna-seq given)
  02_summary_correlations.pdf — r values for all available genes (if --rna-seq given)

Usage
~~~~~
  # Availability check + signatures only
  python scripts/18_8p_perturbation_analysis.py

  # Also generate RNA-seq correlation scatter plots
  python scripts/18_8p_perturbation_analysis.py \\
      --rna-seq data/raw/8p_rnaseq_logfc.csv

  # Use cell-cycle corrected iPSC data
  python scripts/18_8p_perturbation_analysis.py \\
      --input data/raw/iPSC_KOLF2_pseudobulk_cc_corrected.h5ad

RNA-seq CSV format
~~~~~~~~~~~~~~~~~~
  The file passed to --rna-seq should have at minimum two columns:
    gene_name   — HGNC gene symbol
    logFC       — log2 fold change (8p-deleted vs normal)
  Any additional columns are ignored.
"""

import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

DEFAULT_INPUT = DATA_DIR / "iPSC_KOLF2_pseudobulk_normalized.h5ad"
CTRL_LABEL    = "non-targeting"

# ── 8p deletion gene list ─────────────────────────────────────────────────────
GENES_8P = [
    "DLGAP2",
    "TDRP",
    "KBTBD11-OT1",   # overlapping transcript — may not be a KD target
    "CSMD1",
    "ERICH1",
    "ARHGEF10",
    "KBTBD11",
    "MYOM2",
    "CLN8",
]

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font_scale=1.0)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
COLOUR_DOTS    = "#1f77b4"   # blue scatter dots + regression line
COLOUR_STATS   = "#d62728"   # red stats annotation
COLOUR_SUMMARY = "#1f77b4"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_ipsc(path: Path):
    """
    Load iPSC pseudobulk h5ad.
    Returns (X, pert_labels, gene_names, ctrl_mean).
      X            : float64 array (n_perts × n_genes)
      pert_labels  : list[str]
      gene_names   : list[str]
      ctrl_mean    : float64 array (n_genes,)
    """
    print(f"Loading {path.name} ...")
    adata = ad.read_h5ad(path)
    print(f"  {adata.n_obs} perturbations × {adata.n_vars} genes")

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan

    pert_labels = list(adata.obs.index)
    gene_names  = list(adata.var["gene_name"])

    ctrl_rows = [i for i, l in enumerate(pert_labels) if l == CTRL_LABEL]
    if not ctrl_rows:
        raise ValueError(f"No '{CTRL_LABEL}' rows found in obs.index.")
    ctrl_mean = np.nanmean(X[ctrl_rows, :], axis=0)

    n_ctrl = len(ctrl_rows)
    n_perts = len([l for l in pert_labels if l != CTRL_LABEL])
    print(f"  {n_ctrl} control rows, {n_perts} unique perturbations")

    return X, pert_labels, gene_names, ctrl_mean


# ── Availability ──────────────────────────────────────────────────────────────

def check_availability(genes_8p: list[str], pert_labels: list[str]) -> pd.DataFrame:
    """Report which 8p genes have a CRISPRi knockdown in the iPSC library."""
    pert_set = set(pert_labels)
    rows = []
    for g in genes_8p:
        avail = g in pert_set
        rows.append({"gene": g, "in_iPSC_library": avail})
    df = pd.DataFrame(rows)
    n_avail = df["in_iPSC_library"].sum()
    print(f"\n8p gene availability in iPSC Perturb-seq library:")
    for _, r in df.iterrows():
        status = "✓" if r["in_iPSC_library"] else "✗"
        print(f"  {status}  {r['gene']}")
    print(f"\n  {n_avail}/{len(genes_8p)} genes available")
    return df


# ── Signatures ────────────────────────────────────────────────────────────────

def compute_signatures(
    genes_avail: list[str],
    X: np.ndarray,
    pert_labels: list[str],
    gene_names: list[str],
    ctrl_mean: np.ndarray,
) -> pd.DataFrame:
    """
    For each available 8p gene, compute Δz = mean(z-score under KD) − mean(z-score under ctrl)
    across every detected gene.  Returns a DataFrame (rows=genes, cols=8p KD genes).
    """
    pert_idx = {}
    for i, l in enumerate(pert_labels):
        if l not in pert_idx:
            pert_idx[l] = []
        pert_idx[l].append(i)

    sigs = {}
    for gene in genes_avail:
        rows = pert_idx[gene]
        kd_mean    = np.nanmean(X[rows, :], axis=0)
        delta_z    = kd_mean - ctrl_mean
        sigs[gene] = delta_z

    df = pd.DataFrame(sigs, index=gene_names)
    df.index.name = "gene"
    dups = df.index.duplicated(keep="first").sum()
    if dups:
        df = df[~df.index.duplicated(keep="first")]
    return df


# ── Correlation with RNA-seq ──────────────────────────────────────────────────

def load_rnaseq(path: Path) -> pd.Series:
    """
    Load bulk RNA-seq logFC file.
    Accepts several common column-name conventions:
      gene_name / logFC        (script default)
      Expressed_Gene_Symbol / lfc   (Feng et al. Cell Genomics 2026)
      gene / log2FoldChange    (DESeq2 default)
    """
    df = pd.read_csv(path)

    # Detect gene column
    gene_col = None
    for c in ["gene_name", "Expressed_Gene_Symbol", "gene_symbol", "gene", "Gene"]:
        if c in df.columns:
            gene_col = c
            break

    # Detect logFC column
    lfc_col = None
    for c in ["logFC", "lfc", "log2FoldChange", "LFC", "log2FC"]:
        if c in df.columns:
            lfc_col = c
            break

    if gene_col is None or lfc_col is None:
        raise ValueError(
            f"Cannot find gene or logFC columns in RNA-seq file.\n"
            f"Found columns: {list(df.columns)}\n"
            f"Expected a gene column (e.g. gene_name, Expressed_Gene_Symbol) "
            f"and a logFC column (e.g. logFC, lfc, log2FoldChange)."
        )

    print(f"  Using columns: gene='{gene_col}', logFC='{lfc_col}'")
    s = df.set_index(gene_col)[lfc_col]
    dups = s.index.duplicated(keep="first").sum()
    if dups:
        print(f"  Dropping {dups} duplicate gene symbols (keeping first occurrence)")
        s = s[~s.index.duplicated(keep="first")]
    return s


def scatter_one(
    gene: str,
    sig: pd.Series,
    rna_lfc: pd.Series,
    ax: plt.Axes,
) -> tuple[float, int] | None:
    """
    Scatter plot of RNA-seq logFC (x) vs perturbation lfc (y) for one 8p KD gene.
    Matches the original figure style: blue dots, blue regression line, red stats top-left.
    Returns (Pearson r, n_genes), or None if <20 common genes.
    """
    common = sig.index.intersection(rna_lfc.index)
    if len(common) < 20:
        return None

    x = rna_lfc.loc[common].values.astype(float)
    y = sig.loc[common].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return None

    r, p = pearsonr(x, y)

    ax.scatter(x, y, s=2, alpha=0.25, color=COLOUR_DOTS, linewidths=0, rasterized=True)
    m, b = np.polyfit(x, y, 1)
    xline = np.array([x.min(), x.max()])
    ax.plot(xline, m * xline + b, color=COLOUR_DOTS, linewidth=1.5)

    ax.set_title(gene, fontsize=10, fontweight="bold")
    ax.set_xlabel("logFC", fontsize=8)
    ax.set_ylabel("perturb_lfc", fontsize=8)

    p_str = f"{p:.3e}"
    ax.text(0.05, 0.95,
            f"r = {r:.3f}\np = {p_str}\nn = {len(x):,}",
            transform=ax.transAxes, fontsize=7,
            va="top", color=COLOUR_STATS)
    sns.despine(ax=ax)
    return r, len(x)


def plot_scatter_grid(sigs: pd.DataFrame, rna_lfc: pd.Series,
                      figures_dir: Path, n_rna_total: int) -> dict[str, float]:
    """One scatter per available gene on a 4-column grid. Returns {gene: r}."""
    genes = list(sigs.columns)
    ncols = min(4, len(genes))
    nrows = (len(genes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).ravel()

    r_values = {}
    n_common = None
    for i, gene in enumerate(genes):
        result = scatter_one(gene, sigs[gene], rna_lfc, axes[i])
        if result is not None:
            r, n = result
            r_values[gene] = r
            n_common = n

    for j in range(len(genes), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"RNA-seq correlation of {len(genes)} deleted-region genes "
        "available as Perturb-seq targets",
        fontsize=11, fontweight="bold", y=1.01,
    )

    if n_common is not None:
        fig.text(
            0.5, -0.02,
            f"Out of {n_rna_total:,} genes detected in 8p RNA-seq, "
            f"{n_common:,} expressed genes also detected in Perturb-seq",
            ha="center", fontsize=8, style="italic",
        )

    plt.tight_layout()
    out = figures_dir / "01_8p_kd_vs_rnaseq_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return r_values


def plot_summary_bar(r_values: dict[str, float], figures_dir: Path) -> None:
    """Bar chart of Pearson r for each available 8p gene."""
    genes  = list(r_values.keys())
    rs     = [r_values[g] for g in genes]
    colours = [COLOUR_DOTS if r >= 0 else COLOUR_STATS for r in rs]

    order = np.argsort(rs)[::-1]
    genes_s = [genes[i] for i in order]
    rs_s    = [rs[i]    for i in order]
    cols_s  = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(5, len(genes) * 0.9), 4))
    ax.bar(range(len(genes_s)), rs_s, color=cols_s, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(genes_s)))
    ax.set_xticklabels(genes_s, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r  (KD Δz vs 8p RNA-seq logFC)", fontsize=10)
    ax.set_title("Transcriptional correlation of 8p KD signatures with 8p deletion RNA-seq",
                 fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = figures_dir / "02_8p_kd_correlation_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="iPSC pseudobulk h5ad (default: iPSC_KOLF2_pseudobulk_normalized.h5ad)",
    )
    parser.add_argument(
        "--rna-seq", type=Path, default=None,
        metavar="CSV",
        help="Optional: CSV with 'gene_name' and 'logFC' columns from 8p RNA-seq. "
             "If provided, generates correlation scatter plots.",
    )
    args = parser.parse_args()

    results_dir = ROOT / "results" / "8p_deletion"
    figures_dir = ROOT / "figures" / "8p_deletion"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        print(f"[ERROR] iPSC h5ad not found: {args.input}")
        print("Run: python scripts/17_preprocess_ipsc.py")
        return

    X, pert_labels, gene_names, ctrl_mean = load_ipsc(args.input)

    # ── Availability ─────────────────────────────────────────────────────────
    avail_df = check_availability(GENES_8P, pert_labels)
    avail_df.to_csv(results_dir / "availability.csv", index=False)
    print(f"Saved → results/8p_deletion/availability.csv")

    genes_avail = avail_df.loc[avail_df["in_iPSC_library"], "gene"].tolist()
    if not genes_avail:
        print("\nNo 8p genes found in the iPSC KD library. "
              "Check that the h5ad obs.index contains gene symbol labels.")
        return

    # ── Perturbation signatures ───────────────────────────────────────────────
    print(f"\nComputing perturbation signatures for {len(genes_avail)} genes ...")
    sigs = compute_signatures(genes_avail, X, pert_labels, gene_names, ctrl_mean)
    sigs.to_csv(results_dir / "signatures.csv")
    print(f"Saved → results/8p_deletion/signatures.csv  "
          f"({len(gene_names)} genes × {len(genes_avail)} KDs)")

    # Print top upregulated and downregulated genes per KD
    for gene in genes_avail:
        s = sigs[gene].dropna().sort_values(ascending=False)
        top_up  = ", ".join(s.head(5).index.tolist())
        top_dn  = ", ".join(s.tail(5).index.tolist())
        print(f"\n  {gene} KD:")
        print(f"    Top upregulated:   {top_up}")
        print(f"    Top downregulated: {top_dn}")

    # ── Optional: RNA-seq correlation ─────────────────────────────────────────
    if args.rna_seq is None:
        print(
            "\nNo --rna-seq file provided — skipping correlation plots.\n"
            "To generate them, run:\n"
            "  python scripts/18_8p_perturbation_analysis.py "
            "--rna-seq data/raw/8p_rnaseq_logfc.csv\n"
            "Expected CSV columns: gene_name, logFC"
        )
        return

    print(f"\nLoading RNA-seq logFC from {args.rna_seq} ...")
    rna_lfc = load_rnaseq(args.rna_seq)
    n_rna_total = len(rna_lfc)
    print(f"  {n_rna_total:,} genes with logFC values")

    r_values = plot_scatter_grid(sigs, rna_lfc, figures_dir, n_rna_total)

    if r_values:
        r_df = pd.DataFrame(
            {"gene": list(r_values.keys()), "pearson_r": list(r_values.values())}
        ).sort_values("pearson_r", ascending=False)
        r_df.to_csv(results_dir / "correlations.csv", index=False)
        print(f"\nCorrelation summary:")
        print(r_df.to_string(index=False))
        plot_summary_bar(r_values, figures_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
