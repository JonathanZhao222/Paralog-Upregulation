"""
20_paralog_compensation_test.py
--------------------------------
Tests whether paralog genes show compensatory upregulation when their
partner gene is knocked down — genome-wide, across all KD experiments.

This is distinct from the sig vs non-sig comparison in 03_compare_visualize.py.
Instead of asking "do the 37 curated pairs show upregulation?", this asks:
"putting all labels aside, when any gene is KD'd, do its paralogs tend to
go up more than non-paralog genes?"

Two analyses
~~~~~~~~~~~~
Analysis B — Rank-sum meta-analysis (fast, ~2 min)
  For every KD gene with ≥1 paralog measured in the transcriptome:
    - Split Δz vector into paralog genes vs all other genes
    - Mann-Whitney U test → rank-biserial correlation (effect size)
  Aggregate effect sizes across all KD genes.
  If compensation is real: distribution of effect sizes shifts > 0.

Analysis A — GSEA-style enrichment (slower, ~30-60 min)
  For every KD gene with ≥MIN_PARALOG_GSEA paralogs measured:
    - Build custom gene set = {paralogs of KD gene}
    - Run preranked GSEA on full Δz-ranked gene list
    - Extract NES
  Aggregate NES values across all KD genes.
  If compensation is real: distribution of NES shifts > 0.

Outputs (results/paralog_compensation/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  paralog_table.csv          — Ensembl genome-wide paralog pairs (cached)
  ranksum_results.csv        — per-KD rank-biserial correlation + MW p-value
  gsea_results.csv           — per-KD NES (Analysis A only)
  figures/20_ranksum_distribution.pdf
  figures/20_gsea_nes_distribution.pdf
  figures/20_effect_vs_nparalogs.pdf

Usage (Sherlock)
~~~~~~~~~~~~~~~~
  # Analysis B only (fast):
  sbatch --partition=normal --mem=64G --time=1:00:00 --wrap="
    ml python/3.12.1
    source /scratch/groups/sheltzer/paralog_env/bin/activate
    python3 scripts/20_paralog_compensation_test.py --cell-line iPSC
  "

  # Both analyses:
  sbatch --partition=normal --mem=64G --time=4:00:00 --wrap="
    ml python/3.12.1
    source /scratch/groups/sheltzer/paralog_env/bin/activate
    python3 scripts/20_paralog_compensation_test.py --cell-line iPSC --gsea
  "
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_1samp
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "paralog_compensation"
FIGURES_DIR = ROOT / "figures" / "paralog_compensation"
CTRL_LABEL  = "non-targeting"

MIN_PARALOGS_RANKSUM = 1    # minimum paralogs for Analysis B
MIN_PARALOGS_GSEA    = 3    # minimum paralogs for Analysis A (need signal)
GSEA_PERMUTATIONS    = 100  # fewer permutations for speed across 1000s of KDs

CELL_LINE_FILES = {
    "iPSC":    "iPSC_KOLF2_pseudobulk_normalized.h5ad",
    "iPSC_cc": "iPSC_KOLF2_pseudobulk_cc_corrected.h5ad",
    "K562":    "K562_gwps_normalized_bulk_01.h5ad",
    "rpe1":    "rpe1_normalized_bulk_01.h5ad",
}

DIRECT_INDEX = {"iPSC", "iPSC_cc"}


# ── Step 1: Ensembl paralog table ─────────────────────────────────────────────

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
BIOMART_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="external_gene_name"/>
    <Attribute name="hsapiens_paralog_associated_gene_name"/>
    <Attribute name="hsapiens_paralog_perc_id"/>
    <Attribute name="hsapiens_paralog_perc_id_r1"/>
  </Dataset>
</Query>"""


def fetch_paralog_table(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Loading cached paralog table from {cache_path} ...")
        df = pd.read_csv(cache_path)
    else:
        print("Downloading Ensembl human paralog table from BioMart ...")
        r = requests.get(BIOMART_URL, params={"query": BIOMART_XML}, timeout=300)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), sep="\t")
        df.columns = ["gene", "paralog", "pct_id_query", "pct_id_target"]
        df = df.dropna(subset=["gene", "paralog"])
        df = df[df["gene"] != df["paralog"]]          # remove self-pairs
        df = df[df["paralog"].str.strip() != ""]
        df.to_csv(cache_path, index=False)
        print(f"  {len(df):,} paralog pairs downloaded → cached at {cache_path}")

    # Build lookup: gene → set of paralogs
    return df


def build_paralog_lookup(df: pd.DataFrame) -> dict[str, set]:
    lookup: dict[str, set] = {}
    for gene, paralog in zip(df["gene"], df["paralog"]):
        lookup.setdefault(gene, set()).add(paralog)
        lookup.setdefault(paralog, set()).add(gene)
    return lookup


# ── Step 2: Load h5ad and compute Δz ─────────────────────────────────────────

def load_delta_z(cell_line: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
      delta_z_df: DataFrame (n_perts × n_genes) of Δz values
      gene_names: list of gene names (columns)
    """
    h5ad_path = DATA_DIR / CELL_LINE_FILES[cell_line]
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"{h5ad_path} not found.\n"
            f"On Sherlock this should be at:\n"
            f"  /scratch/groups/sheltzer/paralog_upregulation/data/raw/"
            f"{CELL_LINE_FILES[cell_line]}"
        )

    print(f"Loading {h5ad_path.name} ...")
    adata = ad.read_h5ad(h5ad_path)

    if cell_line in DIRECT_INDEX:
        pert_labels = list(adata.obs.index)
    else:
        pert_labels = [idx.split("_")[1] for idx in adata.obs.index]

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan

    gene_names = list(adata.var["gene_name"])
    print(f"  {X.shape[0]} perturbation rows × {X.shape[1]} genes")

    # Build per-perturbation mean
    from collections import defaultdict
    rows_by_pert: dict[str, list] = defaultdict(list)
    for i, label in enumerate(pert_labels):
        rows_by_pert[label].append(i)

    print("  Computing per-perturbation mean expression ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pert_mean = {
            label: np.nanmean(X[rows, :], axis=0)
            for label, rows in rows_by_pert.items()
        }

    ctrl_vec = pert_mean.get(CTRL_LABEL)
    if ctrl_vec is None:
        raise ValueError(f"No '{CTRL_LABEL}' rows found.")

    # Compute Δz for every non-control perturbation
    kd_genes = [p for p in pert_mean if p != CTRL_LABEL]
    print(f"  Computing Δz for {len(kd_genes):,} KD genes ...")
    dz_matrix = np.stack(
        [pert_mean[g] - ctrl_vec for g in kd_genes], axis=0
    ).astype(np.float32)   # shape: (n_kd, n_genes)

    df = pd.DataFrame(dz_matrix, index=kd_genes, columns=gene_names)
    return df, gene_names


# ── Step 3: Analysis B — rank-sum meta-analysis ───────────────────────────────

def run_ranksum(dz_df: pd.DataFrame, paralog_lookup: dict[str, set]) -> pd.DataFrame:
    """
    For each KD gene with ≥1 paralog in the transcriptome, compute
    Mann-Whitney rank-biserial correlation between paralog Δz and non-paralog Δz.
    Returns DataFrame with one row per KD gene.
    """
    gene_set = set(dz_df.columns)
    results = []

    kd_genes = list(dz_df.index)
    print(f"\nAnalysis B: rank-sum test for {len(kd_genes):,} KD genes ...")

    for kd in tqdm(kd_genes, desc="rank-sum"):
        paralogs = paralog_lookup.get(kd, set()) & gene_set
        if len(paralogs) < MIN_PARALOGS_RANKSUM:
            continue

        row = dz_df.loc[kd]
        para_vals  = row[list(paralogs)].values.astype(float)
        other_vals = row[~row.index.isin(paralogs)].values.astype(float)

        # Drop NaN
        para_vals  = para_vals[np.isfinite(para_vals)]
        other_vals = other_vals[np.isfinite(other_vals)]
        if len(para_vals) == 0 or len(other_vals) == 0:
            continue

        stat, pval = mannwhitneyu(para_vals, other_vals, alternative="two-sided")
        n1, n2 = len(para_vals), len(other_vals)
        # Rank-biserial correlation: effect size in [-1, 1]
        rbc = 2 * stat / (n1 * n2) - 1

        results.append({
            "kd_gene":          kd,
            "n_paralogs":       n1,
            "n_other":          n2,
            "rank_biserial_r":  rbc,
            "mw_pval":          pval,
            "para_median_dz":   float(np.median(para_vals)),
            "other_median_dz":  float(np.median(other_vals)),
            "delta_median":     float(np.median(para_vals) - np.median(other_vals)),
        })

    df = pd.DataFrame(results)
    print(f"  {len(df):,} KD genes with ≥{MIN_PARALOGS_RANKSUM} paralog(s) tested")
    return df


# ── Step 4: Analysis A — GSEA per KD gene ─────────────────────────────────────

def run_gsea_analysis(dz_df: pd.DataFrame,
                      paralog_lookup: dict[str, set]) -> pd.DataFrame:
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed — skipping Analysis A. "
              "Install with: pip install gseapy")
        return pd.DataFrame()

    gene_set = set(dz_df.columns)
    results  = []

    eligible = [
        g for g in dz_df.index
        if len(paralog_lookup.get(g, set()) & gene_set) >= MIN_PARALOGS_GSEA
    ]
    print(f"\nAnalysis A: GSEA for {len(eligible):,} KD genes "
          f"(≥{MIN_PARALOGS_GSEA} paralogs) ...")

    for kd in tqdm(eligible, desc="GSEA"):
        paralogs = list(paralog_lookup.get(kd, set()) & gene_set)
        ranked   = dz_df.loc[kd].dropna().sort_values(ascending=False)
        ranked   = ranked[~ranked.index.duplicated(keep="first")]

        try:
            pre = gp.prerank(
                rnk=ranked,
                gene_sets={kd: paralogs},
                min_size=MIN_PARALOGS_GSEA,
                max_size=len(gene_set),
                permutation_num=GSEA_PERMUTATIONS,
                outdir=None,
                seed=42,
                verbose=False,
                threads=1,
            )
            res = pre.res2d
            if res.empty:
                continue
            row = res.iloc[0]
            results.append({
                "kd_gene":    kd,
                "n_paralogs": len(paralogs),
                "nes":        float(row.get("NES", np.nan)),
                "nom_pval":   float(row.get("NOM p-val", np.nan)),
                "fdr":        float(row.get("FDR q-val", np.nan)),
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    print(f"  {len(df):,} KD genes with GSEA results")
    return df


# ── Step 5: Figures ───────────────────────────────────────────────────────────

def plot_ranksum(ranksum_df: pd.DataFrame, cell_line: str,
                 figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: distribution of rank-biserial correlations
    ax = axes[0]
    rbc = ranksum_df["rank_biserial_r"].dropna()
    ax.hist(rbc, bins=60, color="#1f77b4", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.axvline(rbc.mean(), color="#d62728", linewidth=1.5, linestyle="-",
               label=f"mean = {rbc.mean():.4f}")
    # One-sample t-test against 0
    t, p = ttest_1samp(rbc.dropna(), 0)
    ax.set_title(f"Rank-biserial correlation\n(paralogs vs non-paralogs)\np={p:.2e}",
                 fontsize=9)
    ax.set_xlabel("Rank-biserial r", fontsize=8)
    ax.set_ylabel("Number of KD genes", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 2: Δz median paralog vs non-paralog (paired)
    ax = axes[1]
    delta = ranksum_df["delta_median"].dropna()
    ax.hist(delta, bins=60, color="#2ca02c", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.axvline(delta.mean(), color="#d62728", linewidth=1.5,
               label=f"mean = {delta.mean():.4f}")
    t2, p2 = ttest_1samp(delta.dropna(), 0)
    ax.set_title(f"Median Δz: paralogs − non-paralogs\np={p2:.2e}", fontsize=9)
    ax.set_xlabel("Δ median Δz", fontsize=8)
    ax.set_ylabel("Number of KD genes", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 3: effect size vs number of paralogs
    ax = axes[2]
    ax.scatter(ranksum_df["n_paralogs"], ranksum_df["rank_biserial_r"],
               s=4, alpha=0.3, color="#1f77b4", linewidths=0)
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Number of paralogs", fontsize=8)
    ax.set_ylabel("Rank-biserial r", fontsize=8)
    ax.set_title("Effect size vs paralog count", fontsize=9)
    sns.despine(ax=ax)

    fig.suptitle(
        f"Genome-wide paralog compensation test — {cell_line}\n"
        f"n = {len(rbc):,} KD genes with ≥{MIN_PARALOGS_RANKSUM} paralog(s)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    out = figures_dir / "20_ranksum_distribution.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_gsea(gsea_df: pd.DataFrame, cell_line: str, figures_dir: Path) -> None:
    if gsea_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    nes = gsea_df["nes"].dropna()
    ax.hist(nes, bins=50, color="#ff7f0e", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.axvline(nes.mean(), color="#d62728", linewidth=1.5,
               label=f"mean NES = {nes.mean():.3f}")
    _, p = ttest_1samp(nes.dropna(), 0)
    ax.set_title(
        f"GSEA NES distribution — {cell_line}\n"
        f"n={len(nes):,} KD genes  p={p:.2e} (t-test vs 0)",
        fontsize=9
    )
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=8)
    ax.set_ylabel("Number of KD genes", fontsize=8)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = figures_dir / "20_gsea_nes_distribution.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-line", choices=list(CELL_LINE_FILES.keys()),
                        default="iPSC")
    parser.add_argument("--gsea", action="store_true",
                        help="Also run Analysis A (GSEA per KD gene, slower)")
    parser.add_argument("--min-pct-id", type=float, default=0.0,
                        help="Minimum paralog percent identity to include (default 0 = all)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override path to directory containing h5ad files")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    global DATA_DIR
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
        print(f"Using data dir: {DATA_DIR}")

    # Step 1: Paralog table
    cache = RESULTS_DIR / "paralog_table.csv"
    para_df = fetch_paralog_table(cache)

    if args.min_pct_id > 0:
        min_id = args.min_pct_id
        before = len(para_df)
        para_df = para_df[
            (para_df["pct_id_query"].fillna(0) >= min_id) |
            (para_df["pct_id_target"].fillna(0) >= min_id)
        ]
        print(f"  Filtered to pct_id ≥ {min_id}: {len(para_df):,}/{before:,} pairs")

    paralog_lookup = build_paralog_lookup(para_df)
    print(f"  {len(paralog_lookup):,} genes with at least one paralog")

    # Step 2: Load Δz
    t0 = time.time()
    dz_df, _ = load_delta_z(args.cell_line)
    print(f"  Δz matrix loaded in {time.time()-t0:.1f}s  "
          f"shape: {dz_df.shape[0]:,} × {dz_df.shape[1]:,}")

    # Step 3: Analysis B
    ranksum_df = run_ranksum(dz_df, paralog_lookup)
    ranksum_path = RESULTS_DIR / f"ranksum_results_{args.cell_line}.csv"
    ranksum_df.to_csv(ranksum_path, index=False)
    print(f"  Saved → {ranksum_path}")

    rbc = ranksum_df["rank_biserial_r"].dropna()
    if rbc.empty:
        print("  WARNING: no KD genes had paralogs in the transcriptome — "
              "check that paralog gene names match h5ad var['gene_name'].")
        return
    t_stat, p_val = ttest_1samp(rbc, 0)
    print(f"\n  === Analysis B summary [{args.cell_line}] ===")
    print(f"  KD genes tested:        {len(ranksum_df):,}")
    print(f"  Mean rank-biserial r:   {rbc.mean():.5f}")
    print(f"  Median rank-biserial r: {rbc.median():.5f}")
    print(f"  t-test vs 0:            t={t_stat:.3f}, p={p_val:.3e}")
    frac_pos = (rbc > 0).mean()
    print(f"  Fraction r > 0:         {frac_pos:.3f}")

    plot_ranksum(ranksum_df, args.cell_line, FIGURES_DIR)

    # Step 4: Analysis A (optional)
    if args.gsea:
        gsea_df = run_gsea_analysis(dz_df, paralog_lookup)
        if not gsea_df.empty:
            gsea_path = RESULTS_DIR / f"gsea_results_{args.cell_line}.csv"
            gsea_df.to_csv(gsea_path, index=False)
            print(f"  Saved → {gsea_path}")
            nes = gsea_df["nes"].dropna()
            if nes.empty:
                print("  WARNING: all GSEA NES values are NaN.")
            else:
                t2, p2 = ttest_1samp(nes, 0)
                print(f"\n  === Analysis A summary [{args.cell_line}] ===")
                print(f"  KD genes tested:  {len(gsea_df):,}")
                print(f"  Mean NES:         {nes.mean():.4f}")
                print(f"  t-test vs 0:      t={t2:.3f}, p={p2:.3e}")
                plot_gsea(gsea_df, args.cell_line, FIGURES_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
