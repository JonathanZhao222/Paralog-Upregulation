"""
19_8p_gsea.py
-------------
Preranked GSEA for each 8p gene knockdown signature.

For each KD gene in results/8p_deletion/signatures.csv:
  - Ranks all expressed genes by delta-z score
  - Runs GSEA preranked against MSigDB Hallmarks and GO Biological Process
  - Saves per-gene results to results/8p_deletion/gsea/

Generates a summary dot plot across all KD genes:
  - Rows: gene sets significant in at least one KD (FDR < 0.25)
  - Columns: 8p KD genes
  - Dot size: -log10(FDR), capped at 4
  - Dot colour: NES (blue = negative, red = positive)

Usage:
    python scripts/19_8p_gsea.py
    python scripts/19_8p_gsea.py --gene-sets MSigDB_Hallmark_2020 GO_Biological_Process_2023
    python scripts/19_8p_gsea.py --fdr 0.1
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gseapy as gp

ROOT        = Path(__file__).resolve().parent.parent
SIG_CSV     = ROOT / "results" / "8p_deletion" / "signatures.csv"
RESULTS_DIR = ROOT / "results" / "8p_deletion" / "gsea"
FIGURES_DIR = ROOT / "figures" / "8p_deletion"

DEFAULT_GENE_SETS = [
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
]



FDR_THRESHOLD  = 0.25   # GSEA standard threshold
MIN_SIZE       = 15     # minimum gene set size
MAX_SIZE       = 500    # maximum gene set size
PERMUTATIONS   = 1000


def run_gsea_for_gene(gene: str, ranked: pd.Series,
                      gene_sets: list[str], out_dir: Path) -> pd.DataFrame:
    """Run preranked GSEA for one KD gene. Returns combined results DataFrame."""
    ranked = ranked.dropna().sort_values(ascending=False)
    ranked = ranked[~ranked.index.duplicated(keep="first")]
    # gseapy requires unique gene names

    all_results = []
    for gs in gene_sets:
        try:
            pre = gp.prerank(
                rnk=ranked,
                gene_sets=gs,
                min_size=MIN_SIZE,
                max_size=MAX_SIZE,
                permutation_num=PERMUTATIONS,
                outdir=None,
                seed=42,
                verbose=False,
                threads=1,
            )
            res = pre.res2d.copy()
            res.insert(0, "gene_set_library", gs)
            res.insert(0, "kd_gene", gene)
            all_results.append(res)
        except Exception as e:
            print(f"    [{gene}] {gs}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_dir / f"gsea_{gene}.csv", index=False)
    return combined


def plot_dotplot(all_res: pd.DataFrame, genes: list[str],
                 fdr_thresh: float, figures_dir: Path) -> None:
    """
    Dot plot: rows = gene sets, columns = KD genes.
    Dot size = -log10(FDR), colour = NES.
    Only shows gene sets with FDR < fdr_thresh in at least one KD.
    """
    # Detect column names robustly (gseapy output: Term, NES, FDR q-val)
    cols = all_res.columns.tolist()
    fdr_col  = next((c for c in cols if "fdr" in c.lower()), None)
    nes_col  = next((c for c in cols if c.lower() == "nes"), None)
    term_col = next((c for c in cols if c.lower() == "term"), None) \
               or next((c for c in cols if c.lower() == "name"), None)

    if not all([fdr_col, nes_col, term_col]):
        raise ValueError(
            f"Cannot find required columns in GSEA results.\n"
            f"Found: {cols}\nNeed: FDR, NES, Term"
        )

    all_res = all_res.rename(columns={fdr_col: "fdr", nes_col: "nes", term_col: "term"})

    # Ensure numeric
    all_res["fdr"] = pd.to_numeric(all_res["fdr"], errors="coerce")
    all_res["nes"] = pd.to_numeric(all_res["nes"], errors="coerce")
    all_res = all_res.dropna(subset=["fdr", "nes", "term"])

    sig_terms = (
        all_res[all_res["fdr"] < fdr_thresh]["term"].unique()
    )
    if len(sig_terms) == 0:
        print(f"  No gene sets pass FDR < {fdr_thresh} in any KD gene.")
        print("  Lowering threshold to show top 20 terms by best FDR ...")
        top_per_gene = (
            all_res.groupby("kd_gene")
            .apply(lambda x: x.nsmallest(5, "fdr"))
            .reset_index(drop=True)
        )
        sig_terms = top_per_gene["term"].unique()

    plot_df = all_res[all_res["term"].isin(sig_terms)].copy()

    # Pivot to matrices
    nes_mat = plot_df.pivot_table(index="term", columns="kd_gene",
                                  values="nes", aggfunc="first")
    fdr_mat = plot_df.pivot_table(index="term", columns="kd_gene",
                                  values="fdr", aggfunc="first")

    # Reorder columns to match genes list order
    gene_cols = [g for g in genes if g in nes_mat.columns]
    nes_mat = nes_mat.reindex(columns=gene_cols)
    fdr_mat = fdr_mat.reindex(columns=gene_cols)

    # Sort rows by mean abs NES
    row_order = nes_mat.abs().mean(axis=1).sort_values(ascending=False).index
    nes_mat = nes_mat.loc[row_order]
    fdr_mat = fdr_mat.loc[row_order]

    nrows, ncols = nes_mat.shape
    fig_h = max(4, nrows * 0.35 + 2)
    fig_w = max(5, ncols * 1.2 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    nes_lim = max(2.0, np.nanmax(np.abs(nes_mat.values)))
    norm = mcolors.TwoSlopeNorm(vmin=-nes_lim, vcenter=0, vmax=nes_lim)
    cmap = plt.cm.RdBu_r

    # Build arrays for a single vectorised scatter call
    xs, ys, sizes, colours = [], [], [], []
    for r, term in enumerate(nes_mat.index):
        for c, gene in enumerate(gene_cols):
            nes = nes_mat.loc[term, gene]
            fdr = fdr_mat.loc[term, gene]
            if pd.isna(nes) or pd.isna(fdr):
                continue
            xs.append(c)
            ys.append(r)
            sizes.append(min(-np.log10(max(fdr, 1e-10)), 4) * 40)
            colours.append(cmap(norm(nes)))

    if xs:
        ax.scatter(xs, ys, s=sizes, color=colours,
                   edgecolors="grey", linewidths=0.3, zorder=3)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(gene_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(nes_mat.index, fontsize=7)
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, color="lightgrey", zorder=0)
    ax.set_title("GSEA: 8p gene knockdown signatures", fontsize=11,
                 fontweight="bold", pad=10)

    # Colour bar (NES)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("NES", fontsize=8)

    # Size legend using Line2D markers to avoid empty-scatter matplotlib bug
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="grey", markeredgecolor="grey",
                      markersize=np.sqrt(min(-np.log10(fdr_val), 4) * 40),
                      label=label)
        for fdr_val, label in [(0.25, "FDR=0.25"), (0.05, "FDR=0.05"), (0.01, "FDR=0.01")]
    ]
    ax.legend(handles=handles, title="-log₁₀(FDR)", fontsize=7,
              title_fontsize=7, loc="upper left", bbox_to_anchor=(1.15, 1.0))

    plt.tight_layout()
    out = figures_dir / "03_8p_gsea_dotplot.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gene-sets", nargs="+", default=DEFAULT_GENE_SETS,
        help="EnrichR/MSigDB gene set library names to query"
    )
    parser.add_argument(
        "--fdr", type=float, default=FDR_THRESHOLD,
        help="FDR threshold for dot plot display (default 0.25)"
    )
    parser.add_argument(
        "--permutations", type=int, default=PERMUTATIONS,
        help="Number of GSEA permutations (default 1000)"
    )
    args = parser.parse_args()

    if not SIG_CSV.exists():
        raise FileNotFoundError(
            f"{SIG_CSV} not found. Run 18_8p_perturbation_analysis.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading signatures from {SIG_CSV} ...")
    sigs = pd.read_csv(SIG_CSV, index_col=0)
    genes = list(sigs.columns)
    print(f"  {len(genes)} KD genes x {len(sigs):,} expressed genes")
    print(f"  Gene sets: {args.gene_sets}")
    print(f"  Permutations: {args.permutations}\n")

    all_results = []
    for gene in genes:
        print(f"Running GSEA: {gene} ...")
        ranked = sigs[gene].dropna()
        res = run_gsea_for_gene(gene, ranked, args.gene_sets, RESULTS_DIR)
        if not res.empty:
            all_results.append(res)
            n_sig = (pd.to_numeric(res.get("fdr", res.get("FDR q-val", pd.Series())),
                                   errors="coerce") < args.fdr).sum()
            print(f"  {len(res)} gene sets tested, {n_sig} pass FDR < {args.fdr}")

    if not all_results:
        print("No GSEA results produced.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = RESULTS_DIR / "gsea_all_genes.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined results saved to {combined_path}")

    print("\nGenerating dot plot ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_dotplot(combined, genes, args.fdr, FIGURES_DIR)

    print("\nDone.")



if __name__ == "__main__":
    main()
