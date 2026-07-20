"""
16_cn_expression_correlation.py
--------------------------------
For every gene that appears in paralog pairs (sig or non-sig), computes
the Pearson correlation between:
  1. Gene copy number vs RNA expression  (CCLE, ~980 cell lines)
  2. Gene copy number vs protein abundance (ProCan, ~947 cell lines)

across CCLE cell lines.  Then compares the correlation distributions for:
  - Paralog genes from the 27 significant K562 pairs
  - Paralog genes from all non-significant pairs

Outputs:
  results/cn_expression_correlation.csv
  figures/cross_cell_line/16_cn_rna_protein_correlation.pdf

Usage:
    python scripts/16_cn_expression_correlation.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, mannwhitneyu, spearmanr

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DIFF_DIR = Path("/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Difference/data")
CHROM_DIR = Path("/Users/jonathanzhao/Desktop/Sheltzer Lab/Chromosome Compensation")

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

COLOUR_SIG = "#d62728"
COLOUR_NS  = "#1f77b4"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_ccle_gene(col: str) -> str:
    """'TSPAN6 (7105)' → 'TSPAN6'"""
    return col.split(" (")[0]


def parse_procan_gene(col: str) -> str:
    """'Q9Y651;SOX21_HUMAN' → 'SOX21'"""
    return col.split(";")[1].replace("_HUMAN", "")


def _pval_str(p: float) -> str:
    if p < 1e-10:
        exp = int(np.floor(np.log10(p)))
        man = p / 10**exp
        return f"p = {man:.1f}×10^{exp}"
    return f"p = {p:.2e}"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_cn() -> pd.DataFrame:
    print("Loading copy number data ...")
    cn = pd.read_csv(DIFF_DIR / "OmicsAbsoluteCNGene.csv", index_col=0)
    cn.columns = [parse_ccle_gene(c) for c in cn.columns]
    return cn


def load_rna() -> pd.DataFrame:
    print("Loading RNA expression data ...")
    rna = pd.read_csv(DATA_DIR / "CCLE_expression.csv", index_col=0)
    rna.columns = [parse_ccle_gene(c) for c in rna.columns]
    return rna


def load_protein() -> pd.DataFrame:
    print("Loading ProCan protein data ...")
    prot = pd.read_csv(
        CHROM_DIR / "ProCan_protein_matrix_8498_averaged.txt",
        sep="\t", index_col=0,
    )
    # Parse gene symbols from column names
    prot.columns = [parse_procan_gene(c) for c in prot.columns]
    # Convert SIDM index → ACH IDs
    model = pd.read_csv(DIFF_DIR / "Model.csv")[["ModelID", "SangerModelID"]].dropna()
    sidm_to_ach = dict(zip(model["SangerModelID"], model["ModelID"]))
    prot.index = [sidm_to_ach.get(idx.split(";")[0], None) for idx in prot.index]
    prot = prot[prot.index.notna()].copy()
    prot.index = prot.index.astype(str)
    return prot


def load_sig_genes() -> tuple[set, set]:
    """Returns (sig_dep_genes, sig_paralog_genes) for K562 sig pairs."""
    ranked = pd.read_csv(ROOT / "results" / "K562" / "all_pairs_ranked.csv")
    sig = ranked[ranked["group"] == "Aneuploidy vulnerability"]
    return set(sig["dep_gene"]), set(sig["paralog_gene"])


def load_nonsig_genes() -> tuple[set, set]:
    """Returns (nonsig_dep_genes, nonsig_paralog_genes) from K562 non-sig."""
    ns = pd.read_csv(ROOT / "results" / "K562" / "nonsig_results.csv")
    return set(ns["dep_gene"]), set(ns["paralog_gene"])


# ── Correlation computation ───────────────────────────────────────────────────

def compute_correlations(
    cn: pd.DataFrame,
    other: pd.DataFrame,
    genes: list[str],
    label: str,
    min_obs: int = 50,
) -> pd.DataFrame:
    """
    For each gene, compute Pearson r between CN and `other` (RNA or protein)
    across their common cell lines.  Returns DataFrame with columns:
      gene, pearson_r, spearman_r, n_obs, p_pearson
    """
    # Align on cell lines
    common_cells = cn.index.intersection(other.index)
    cn_sub    = cn.loc[common_cells]
    other_sub = other.loc[common_cells]

    records = []
    for gene in genes:
        if gene not in cn_sub.columns or gene not in other_sub.columns:
            records.append({"gene": gene, "pearson_r": np.nan,
                            "spearman_r": np.nan, "n_obs": 0, "p_pearson": np.nan})
            continue

        x = cn_sub[gene].values.astype(float)
        y = other_sub[gene].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_obs:
            records.append({"gene": gene, "pearson_r": np.nan,
                            "spearman_r": np.nan, "n_obs": int(mask.sum()), "p_pearson": np.nan})
            continue

        xm, ym = x[mask], y[mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr, pp   = pearsonr(xm, ym)
            sr, _    = spearmanr(xm, ym)
        records.append({"gene": gene, "pearson_r": pr, "spearman_r": sr,
                        "n_obs": int(mask.sum()), "p_pearson": pp})

    df = pd.DataFrame(records)
    df[f"{label}_pearson_r"]  = df["pearson_r"]
    df[f"{label}_spearman_r"] = df["spearman_r"]
    df[f"{label}_n_obs"]      = df["n_obs"]
    return df[["gene", f"{label}_pearson_r", f"{label}_spearman_r", f"{label}_n_obs"]]


# ── Plotting ──────────────────────────────────────────────────────────────────

def violin_comparison(
    ax: plt.Axes,
    sig_vals: np.ndarray,
    ns_vals: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    sig_clean = sig_vals[np.isfinite(sig_vals)]
    ns_clean  = ns_vals[np.isfinite(ns_vals)]

    data = pd.DataFrame({
        "r": np.concatenate([sig_clean, ns_clean]),
        "group": (["Significant\nparalogs"] * len(sig_clean) +
                  ["Non-significant\nparalogs"] * len(ns_clean)),
    })

    palette = {"Significant\nparalogs": COLOUR_SIG, "Non-significant\nparalogs": COLOUR_NS}
    sns.violinplot(data=data, x="group", y="r", hue="group", palette=palette,
                   inner="box", cut=0, density_norm="width", ax=ax, alpha=0.85,
                   legend=False)
    ax.axhline(0, color="black", lw=0.8, ls="--", zorder=3)

    # Overlay individual sig-pair points
    jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(sig_clean))
    ax.scatter(np.zeros(len(sig_clean)) + jitter, sig_clean,
               color="black", s=22, zorder=5, alpha=0.7, linewidths=0)

    # Mann-Whitney test
    if len(sig_clean) >= 3 and len(ns_clean) >= 3:
        _, p = mannwhitneyu(sig_clean, ns_clean, alternative="two-sided")
        ax.text(0.5, 1.02, f"Mann-Whitney  {_pval_str(p)}",
                transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=10)
    sns.despine(ax=ax)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(42)

    cn   = load_cn()
    rna  = load_rna()
    prot = load_protein()

    sig_dep, sig_para  = load_sig_genes()
    ns_dep,  ns_para   = load_nonsig_genes()

    # All unique genes to score (paralog universe)
    all_para_genes = sorted((sig_para | ns_para) & set(cn.columns))
    print(f"\n{len(all_para_genes)} unique paralog genes to score")

    # ── Correlations ─────────────────────────────────────────────────────────
    print("\nComputing CN vs RNA correlations ...")
    rna_corr  = compute_correlations(cn, rna,  all_para_genes, "rna")

    print("Computing CN vs protein correlations ...")
    prot_corr = compute_correlations(cn, prot, all_para_genes, "protein")

    # Merge
    result = rna_corr.merge(prot_corr, on="gene")

    # Label groups
    result["group"] = result["gene"].apply(
        lambda g: "significant" if g in sig_para else "non-significant"
    )

    # Save
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "cn_expression_correlation.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Summary stats ────────────────────────────────────────────────────────
    for grp in ["significant", "non-significant"]:
        sub = result[result["group"] == grp]
        print(f"\n[{grp}]  n={len(sub)}")
        for col in ["rna_pearson_r", "protein_pearson_r"]:
            vals = sub[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}  median={vals.median():.3f}  n={len(vals)}")

    # ── Figures ──────────────────────────────────────────────────────────────
    fig_dir = ROOT / "figures" / "cross_cell_line"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sig_rna  = result.loc[result["group"] == "significant",     "rna_pearson_r"].values
    ns_rna   = result.loc[result["group"] == "non-significant",  "rna_pearson_r"].values
    sig_prot = result.loc[result["group"] == "significant",     "protein_pearson_r"].values
    ns_prot  = result.loc[result["group"] == "non-significant",  "protein_pearson_r"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    violin_comparison(
        axes[0], sig_rna, ns_rna,
        title="Copy number vs RNA expression\n(Pearson r across CCLE cell lines)",
        ylabel="Pearson r  (CN vs RNA)",
    )
    violin_comparison(
        axes[1], sig_prot, ns_prot,
        title="Copy number vs protein abundance\n(Pearson r across CCLE cell lines)",
        ylabel="Pearson r  (CN vs protein)",
    )

    fig.suptitle(
        "Copy number–expression coupling:\n"
        "Significant paralog genes vs non-significant paralog background",
        fontsize=12, y=1.02,
    )

    plt.tight_layout()
    out_fig = fig_dir / "16_cn_rna_protein_correlation.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_fig}")

    # ── Density plot: where do sig paralogs fall in the non-sig distribution? ─
    fig_d, ax_d = plt.subplots(figsize=(9, 5))

    ns_rna_clean  = ns_rna[np.isfinite(ns_rna)]
    sig_rna_clean = sig_rna[np.isfinite(sig_rna)]

    # Non-sig KDE
    sns.kdeplot(ns_rna_clean, ax=ax_d, color=COLOUR_NS, fill=True,
                alpha=0.3, linewidth=2, label=f"Non-significant paralogs (n={len(ns_rna_clean):,})")

    # Sig paralogs as vertical lines with gene labels
    sig_result = result[result["group"] == "significant"].dropna(subset=["rna_pearson_r"])
    y_max = ax_d.get_ylim()[1] if ax_d.get_ylim()[1] > 0 else 1
    ax_d.set_ylim(bottom=0)

    for _, row in sig_result.iterrows():
        ax_d.axvline(row["rna_pearson_r"], color=COLOUR_SIG, lw=1.2, alpha=0.7, zorder=3)

    # Dummy handle for legend
    from matplotlib.lines import Line2D
    sig_handle = Line2D([0], [0], color=COLOUR_SIG, lw=1.5,
                        label=f"Significant paralogs (n={len(sig_rna_clean)})")
    ax_d.legend(handles=[ax_d.get_legend_handles_labels()[0][0], sig_handle],
                fontsize=9, frameon=False)

    ax_d.set_xlabel("Pearson r  (copy number vs RNA expression)", fontsize=11)
    ax_d.set_ylabel("Density", fontsize=11)
    ax_d.set_title("Distribution of CN–RNA coupling across paralog genes\n"
                   "Significant paralogs marked individually", fontsize=11)
    sns.despine(ax=ax_d)
    plt.tight_layout()
    out_d = fig_dir / "16_cn_rna_density.pdf"
    fig_d.savefig(out_d, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_d}")

    # ── Raw CN vs expression scatter (sig genes highlighted) ─────────────────
    common_rna  = cn.index.intersection(rna.index)
    common_prot = cn.index.intersection(prot.index)

    CN_MAX = 10  # clip x-axis to diploid/mild amplification range
    CN_BINS = np.linspace(0, CN_MAX, 12)
    BIN_CENTRES = 0.5 * (CN_BINS[:-1] + CN_BINS[1:])

    def bin_median(cn_vals, expr_vals, bins):
        """Median expression per CN bin; NaN if fewer than 5 observations."""
        meds = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (cn_vals >= lo) & (cn_vals < hi)
            meds.append(np.nanmedian(expr_vals[mask]) if mask.sum() >= 5 else np.nan)
        return np.array(meds)

    for label, other, common_cells, fname in [
        ("RNA expression (log2 TPM)", rna,  common_rna,  "16_cn_vs_rna_scatter.pdf"),
        ("Protein abundance",         prot, common_prot, "16_cn_vs_protein_scatter.pdf"),
    ]:
        cn_sub    = cn.loc[common_cells]
        other_sub = other.loc[common_cells]

        fig3, ax3 = plt.subplots(figsize=(8, 6))

        # Non-sig background — median trend per bin across all non-sig genes
        ns_genes_avail = list(ns_para & set(cn_sub.columns) & set(other_sub.columns))
        ns_cn_all   = np.concatenate([cn_sub[g].values.astype(float)   for g in ns_genes_avail])
        ns_expr_all = np.concatenate([other_sub[g].values.astype(float) for g in ns_genes_avail])
        valid = np.isfinite(ns_cn_all) & np.isfinite(ns_expr_all) & (ns_cn_all <= CN_MAX)
        ns_med = bin_median(ns_cn_all[valid], ns_expr_all[valid], CN_BINS)
        ax3.plot(BIN_CENTRES, ns_med, color=COLOUR_NS, lw=2.5, ls="--",
                 label=f"Non-sig median (n={len(ns_genes_avail):,} genes)", zorder=2)
        ax3.fill_between(BIN_CENTRES, ns_med, alpha=0.12, color=COLOUR_NS, zorder=1)

        # Sig genes — median trend per bin across all sig genes
        sig_genes_avail = list(sig_para & set(cn_sub.columns) & set(other_sub.columns))
        sig_cn_all   = np.concatenate([cn_sub[g].values.astype(float)   for g in sig_genes_avail])
        sig_expr_all = np.concatenate([other_sub[g].values.astype(float) for g in sig_genes_avail])
        valid_s = np.isfinite(sig_cn_all) & np.isfinite(sig_expr_all) & (sig_cn_all <= CN_MAX)
        sig_med = bin_median(sig_cn_all[valid_s], sig_expr_all[valid_s], CN_BINS)
        ax3.plot(BIN_CENTRES, sig_med, color=COLOUR_SIG, lw=2.5,
                 label=f"Sig median (n={len(sig_genes_avail)} genes)", zorder=4)
        ax3.fill_between(BIN_CENTRES, sig_med, alpha=0.15, color=COLOUR_SIG, zorder=3)

        ax3.set_xlim(0, CN_MAX)
        ax3.set_xlabel("Absolute copy number", fontsize=11)
        ax3.set_ylabel(label, fontsize=11)
        ax3.set_title(f"Copy number vs {label}\nacross CCLE cell lines  (CN ≤ {CN_MAX})",
                      fontsize=11)
        ax3.legend(fontsize=10, frameon=False)
        sns.despine(ax=ax3)
        plt.tight_layout()
        out3 = fig_dir / fname
        fig3.savefig(out3, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out3}")

    # ── Per-gene scatter: CN-RNA vs CN-protein, coloured by group ────────────
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    ns_sub  = result[result["group"] == "non-significant"].dropna(subset=["rna_pearson_r","protein_pearson_r"])
    sig_sub = result[result["group"] == "significant"].dropna(subset=["rna_pearson_r","protein_pearson_r"])

    ax2.scatter(ns_sub["rna_pearson_r"], ns_sub["protein_pearson_r"],
                color=COLOUR_NS, s=12, alpha=0.35, linewidths=0,
                label=f"Non-significant paralogs (n={len(ns_sub):,})")
    ax2.scatter(sig_sub["rna_pearson_r"], sig_sub["protein_pearson_r"],
                color=COLOUR_SIG, s=55, alpha=0.9, linewidths=0.5, edgecolors="black",
                zorder=5, label=f"Significant paralogs (n={len(sig_sub)})")

    # Label sig genes
    for _, row in sig_sub.iterrows():
        ax2.annotate(row["gene"],
                     xy=(row["rna_pearson_r"], row["protein_pearson_r"]),
                     xytext=(5, 3), textcoords="offset points",
                     fontsize=7, color=COLOUR_SIG)

    ax2.axhline(0, color="grey", lw=0.7, ls="--")
    ax2.axvline(0, color="grey", lw=0.7, ls="--")
    ax2.set_xlabel("Pearson r  (copy number vs RNA expression)", fontsize=11)
    ax2.set_ylabel("Pearson r  (copy number vs protein abundance)", fontsize=11)
    ax2.set_title("CN–expression coupling across CCLE cell lines\n"
                  "(each point = one paralog gene)", fontsize=11)
    ax2.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax2)
    plt.tight_layout()
    out_fig2 = fig_dir / "16_cn_rna_vs_protein_scatter.pdf"
    fig2.savefig(out_fig2, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_fig2}")


if __name__ == "__main__":
    main()
