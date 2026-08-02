"""
21_8p_functional_similarity.py
-------------------------------
Tests whether the chromosome 8p common-deletion-region genes are
"functionally similar" to each other, per Perturb-seq CRISPRi KD phenotypes —
i.e. whether their knockdown transcriptional signatures resemble each other
more than expected by chance, the way paralogs or protein-complex partners do
in the KOLF2.1J atlas paper (Nourreddine, Doctor et al., Nat Biotechnol 2026).

This extends 18_8p_perturbation_analysis.py, which only compared each 8p gene's
KD signature against bulk RNA-seq. Here we build the genome-scale
perturbation-perturbation correlation matrix (same construction as the paper's
`create_corrmatrix`, mode='perturb') and ask three questions about the 8p
gene set {TDRP, ERICH1, ARHGEF10, KBTBD11, CLN8} (the 5/9 common-region genes
with a CRISPRi KD in the iPSC library):

  1. Do the 5 8p KD signatures correlate with each other more than random
     gene sets of the same size? (permutation null on mean pairwise r)
  2. Do they fall in the same HDBSCAN cluster more than chance?
     (same clustering recipe as the paper's Fig 2b: metric='precomputed' on
     1-correlation, min_cluster_size=4, cluster_selection_method='eom',
     min_samples=1; permutation null on cluster co-membership)
  3. How does their pairwise correlation compare to known CORUM
     protein-complex partners vs. random gene pairs? (same benchmark as the
     paper's `benchmark_corum_vs_noncorum_pairs`)

Feature-selection caveat
~~~~~~~~~~~~~~~~~~~~~~~~
The paper's own clustering is run on a DEG + HVG feature subset built by
`replogle_pipeline()`, which requires per-cell counts and a DESeq2 DEG table
we don't have locally (we only have the pseudobulk z-normalised h5ad from
17_preprocess_ipsc.py). As an approximation we restrict the correlation
matrix to the top `--top-pct-hvgs` most-variable genes (default 10%, matching
the paper's own `pct_hvgs=10` call) but do NOT filter perturbations by DEG
count/effect size. Confirmed empirically: without HVG restriction, HDBSCAN
collapses the full atlas into ~2 clusters; with it, ~600+ clusters emerge,
consistent with the paper's fine-grained structure.

Outputs (results/8p_deletion/ and figures/8p_deletion/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  global_hdbscan_clusters.csv        — cluster ID for every perturbation in the atlas
  8p_cluster_membership.csv          — cluster ID for the 5 available 8p genes
  8p_pairwise_correlations.csv       — 5x5 correlation matrix among 8p KD signatures
  mean_corr_permutation_test.csv     — null distribution summary + p-value
  cluster_coclustering_permutation_test.csv — same, for cluster co-membership
  corum_benchmark_summary.csv        — Mann-Whitney stats + 8p pair percentiles
  figures/06_8p_pairwise_correlation_heatmap.pdf
  figures/07_8p_corum_calibration.pdf

Usage
~~~~~
  python scripts/21_8p_functional_similarity.py
  python scripts/21_8p_functional_similarity.py --top-pct-hvgs 0.05 --n-permutations 20000
  python scripts/21_8p_functional_similarity.py --force-recompute   # ignore cached correlation matrix
"""

import argparse
import itertools
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.cluster import HDBSCAN

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "raw"
CACHE_DIR   = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results" / "8p_deletion"
FIGURES_DIR = ROOT / "figures" / "8p_deletion"

H5AD_PATH   = DATA_DIR / "iPSC_KOLF2_pseudobulk_normalized.h5ad"
CORUM_PATH  = DATA_DIR / "corum_humanComplexes.txt"
CTRL_LABEL  = "non-targeting"

# The 5 common 8p-deletion-region genes with a CRISPRi KD in the iPSC library
# (of the 9-gene common region checked in 18_8p_perturbation_analysis.py;
# DLGAP2, KBTBD11-OT1, CSMD1, MYOM2 have no KD available).
GENES_8P_AVAILABLE = ["TDRP", "ERICH1", "ARHGEF10", "KBTBD11", "CLN8"]

MIN_CLUSTER_SIZE = 4
CLUSTER_SELECTION_METHOD = "eom"
MIN_SAMPLES = 1

CORUM_MIN_FRACTION = 0.66   # matches paper's get_corum_perturbation_pairs_from_list default

sns.set_theme(style="ticks", font_scale=1.0)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


# ── Step 1: Load atlas + build (cached) HVG-restricted correlation matrix ────

def load_pseudobulk() -> tuple[pd.DataFrame, list[str]]:
    """Load the iPSC pseudobulk h5ad, drop genes with any NaN entries."""
    print(f"Loading {H5AD_PATH.name} ...")
    with h5py.File(H5AD_PATH, "r") as f:
        X = f["X"][:]
        idx = [x.decode() if isinstance(x, bytes) else x for x in f["obs"]["_index"][:]]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in f["var"]["gene_name"][:]]

    nan_per_col = np.isnan(X).sum(axis=0)
    keep = nan_per_col == 0
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} genes with NaN entries (of {X.shape[1]})")
    X = X[:, keep].astype(np.float32)
    gene_names = list(np.array(gene_names)[keep])

    df = pd.DataFrame(X, index=idx, columns=gene_names)
    print(f"  {df.shape[0]:,} perturbations x {df.shape[1]:,} genes")
    return df, gene_names


def build_perturb_corrs(pb: pd.DataFrame, top_pct_hvgs: float, force: bool) -> pd.DataFrame:
    """
    Perturbation-perturbation Pearson correlation matrix, restricted to the
    top `top_pct_hvgs` most-variable genes across perturbations.
    Mirrors the paper's create_corrmatrix(mode='perturb'), minus the DEG-based
    perturbation filter (see module docstring).
    Cached to data/raw/cache/ since it's a ~1GB float64 matrix (gitignored).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"ipsc_perturb_corrs_hvg{int(top_pct_hvgs*100)}.npz"

    if cache_path.exists() and not force:
        print(f"Loading cached correlation matrix from {cache_path} ...")
        npz = np.load(cache_path, allow_pickle=True)
        return pd.DataFrame(npz["corr"], index=npz["labels"], columns=npz["labels"])

    print(f"Selecting top {top_pct_hvgs:.0%} most-variable genes ...")
    variances = pb.values.var(axis=0)
    n_top = int(len(variances) * top_pct_hvgs)
    top_idx = np.argsort(variances)[::-1][:n_top]
    print(f"  Using {n_top:,} genes")

    print("Computing perturbation-perturbation correlation matrix ...")
    corr = np.corrcoef(pb.values[:, top_idx])
    corr_df = pd.DataFrame(corr, index=pb.index, columns=pb.index)

    np.savez_compressed(cache_path, corr=corr, labels=np.array(pb.index))
    print(f"  Cached -> {cache_path}")
    return corr_df


# ── Step 2: HDBSCAN clustering + 8p lookup ────────────────────────────────────

def run_hdbscan(perturb_corrs: pd.DataFrame) -> pd.Series:
    print("Running HDBSCAN on 1-correlation distance ...")
    dist = np.clip(1 - perturb_corrs.values, 0, None).astype(np.float64)
    np.fill_diagonal(dist, 0)
    clusterer = HDBSCAN(
        metric="precomputed",
        min_cluster_size=MIN_CLUSTER_SIZE,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        min_samples=MIN_SAMPLES,
        copy=True,
    )
    labels = clusterer.fit_predict(dist)
    labels_series = pd.Series(labels, index=perturb_corrs.index, name="cluster_id")
    n_clusters = labels_series.max() + 1
    n_noise = (labels_series == -1).sum()
    print(f"  {n_clusters} clusters found, {n_noise:,}/{len(labels_series):,} perturbations unclustered (noise)")
    return labels_series


# ── Step 3: Permutation nulls ─────────────────────────────────────────────────

def mean_pairwise_corr(genes: list[str], corr_mat: pd.DataFrame) -> float:
    pairs = list(itertools.combinations(genes, 2))
    vals = [corr_mat.loc[g1, g2] for g1, g2 in pairs]
    return float(np.mean(vals))


def permutation_test_mean_corr(
    target_genes: list[str], perturb_corrs: pd.DataFrame,
    n_permutations: int, seed: int = 42,
) -> tuple[float, np.ndarray, float]:
    """Null: mean pairwise correlation of random gene sets of the same size."""
    print(f"\nPermutation test 1: mean pairwise correlation (n={n_permutations:,}) ...")
    rng = np.random.default_rng(seed)
    all_genes = perturb_corrs.index.to_numpy()
    k = len(target_genes)

    observed = mean_pairwise_corr(target_genes, perturb_corrs)

    corr_vals = perturb_corrs.values
    label_to_idx = {g: i for i, g in enumerate(perturb_corrs.index)}
    pair_idx = np.array(list(itertools.combinations(range(k), 2)))

    null_dist = np.empty(n_permutations, dtype=np.float64)
    n_all = len(all_genes)
    for i in range(n_permutations):
        sample_idx = rng.choice(n_all, size=k, replace=False)
        sub = corr_vals[np.ix_(sample_idx, sample_idx)]
        null_dist[i] = sub[np.triu_indices(k, k=1)].mean()

    p_value = float((null_dist >= observed).sum() + 1) / (n_permutations + 1)
    print(f"  Observed mean pairwise r = {observed:.4f}")
    print(f"  Null mean = {null_dist.mean():.4f}, std = {null_dist.std():.4f}")
    print(f"  Empirical p-value (one-sided, greater) = {p_value:.4g}")
    return observed, null_dist, p_value


def permutation_test_coclustering(
    target_genes: list[str], cluster_labels: pd.Series,
    n_permutations: int, seed: int = 42,
) -> tuple[int, np.ndarray, float]:
    """Null: number of co-clustered pairs (same non-noise cluster) among random gene sets."""
    print(f"\nPermutation test 2: cluster co-membership (n={n_permutations:,}) ...")
    rng = np.random.default_rng(seed)
    all_genes = cluster_labels.index.to_numpy()
    labels_arr = cluster_labels.values
    label_to_idx = {g: i for i, g in enumerate(cluster_labels.index)}
    k = len(target_genes)

    def n_coclustered(idxs: np.ndarray) -> int:
        labs = labels_arr[idxs]
        count = 0
        for a, b in itertools.combinations(range(k), 2):
            if labs[a] != -1 and labs[a] == labs[b]:
                count += 1
        return count

    target_idx = np.array([label_to_idx[g] for g in target_genes])
    observed = n_coclustered(target_idx)

    n_all = len(all_genes)
    null_dist = np.empty(n_permutations, dtype=np.int64)
    for i in range(n_permutations):
        sample_idx = rng.choice(n_all, size=k, replace=False)
        null_dist[i] = n_coclustered(sample_idx)

    p_value = float((null_dist >= observed).sum() + 1) / (n_permutations + 1)
    print(f"  Observed co-clustered pairs = {observed} / {k*(k-1)//2}")
    print(f"  Null mean = {null_dist.mean():.4f}")
    print(f"  Empirical p-value (one-sided, greater) = {p_value:.4g}")
    return observed, null_dist, p_value


# ── Step 4: CORUM benchmark ───────────────────────────────────────────────────

def load_corum_complexes(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", usecols=["complex_id", "complex_name", "subunits_gene_name"])
    complexes = {}
    for _, row in df.iterrows():
        genes = [g.strip() for g in str(row["subunits_gene_name"]).split(";") if g.strip()]
        if len(genes) >= 2:
            complexes[row["complex_id"]] = genes
    print(f"Loaded {len(complexes):,} CORUM complexes (>=2 members)")
    return complexes


def get_corum_pairs(
    perturbations: list[str], corum_complexes: dict[str, list[str]],
    min_fraction: float = CORUM_MIN_FRACTION, seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Same logic as the paper's get_corum_perturbation_pairs_from_list, but
    negative pairs are drawn by rejection sampling (not full enumeration of
    all C(N,2) pairs) since N~11,673 makes exhaustive enumeration impractical
    locally (paper ran this on an HPC node with much larger RAM).
    """
    perturbs = sorted(set(perturbations))
    pert_set = set(perturbs)

    filtered = {}
    for cid, members in corum_complexes.items():
        if len(set(members)) == 1:
            continue
        present = [g for g in members if g in pert_set]
        if members and (len(present) / len(members)) >= min_fraction:
            filtered[cid] = present
    print(f"  {len(filtered)} CORUM complexes present (>= {min_fraction:.0%} members in KD library)")

    same_complex_pairs = set()
    for members in filtered.values():
        for g1, g2 in itertools.combinations(sorted(set(members)), 2):
            same_complex_pairs.add((g1, g2))
    same_complex_pairs = sorted(same_complex_pairs)
    n_pos = len(same_complex_pairs)
    print(f"  {n_pos} CORUM same-complex pairs")

    rng = random.Random(seed)
    pos_set = set(same_complex_pairs)
    negative_pairs = set()
    attempts = 0
    max_attempts = n_pos * 200 + 10_000
    while len(negative_pairs) < n_pos and attempts < max_attempts:
        g1, g2 = rng.sample(perturbs, 2)
        pair = tuple(sorted((g1, g2)))
        if pair not in pos_set:
            negative_pairs.add(pair)
        attempts += 1
    negative_pairs = sorted(negative_pairs)
    print(f"  {len(negative_pairs)} random non-CORUM pairs sampled")
    return same_complex_pairs, negative_pairs


def corum_benchmark(
    perturb_corrs: pd.DataFrame, corum_pairs: list[tuple[str, str]],
    noncorum_pairs: list[tuple[str, str]], target_genes: list[str],
) -> dict:
    corum_vals = np.array([perturb_corrs.loc[g1, g2] for g1, g2 in corum_pairs])
    noncorum_vals = np.array([perturb_corrs.loc[g1, g2] for g1, g2 in noncorum_pairs])

    u_stat, p_value = mannwhitneyu(corum_vals, noncorum_vals, alternative="two-sided")
    print(f"\nCORUM vs non-CORUM Mann-Whitney U: U={u_stat:.1f}, p={p_value:.4g}")
    print(f"  CORUM pairs:     mean r = {corum_vals.mean():.4f}, median r = {np.median(corum_vals):.4f}")
    print(f"  Non-CORUM pairs: mean r = {noncorum_vals.mean():.4f}, median r = {np.median(noncorum_vals):.4f}")

    target_pairs = list(itertools.combinations(target_genes, 2))
    target_vals = np.array([perturb_corrs.loc[g1, g2] for g1, g2 in target_pairs])

    pct_vs_noncorum = [float((noncorum_vals <= v).mean()) for v in target_vals]
    pct_vs_corum = [float((corum_vals <= v).mean()) for v in target_vals]

    print("\n  8p gene-pair correlations vs. CORUM/non-CORUM background:")
    for (g1, g2), v, pn, pc in zip(target_pairs, target_vals, pct_vs_noncorum, pct_vs_corum):
        print(f"    {g1}-{g2}: r={v:.4f}  (percentile vs non-CORUM={pn:.1%}, vs CORUM={pc:.1%})")

    return {
        "u_stat": u_stat, "p_value": p_value,
        "corum_vals": corum_vals, "noncorum_vals": noncorum_vals,
        "target_pairs": target_pairs, "target_vals": target_vals,
        "pct_vs_noncorum": pct_vs_noncorum, "pct_vs_corum": pct_vs_corum,
    }


# ── Figures ────────────────────────────────────────────────────────────────

def plot_pairwise_heatmap(sub_corr: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(sub_corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", square=True, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("8p common-region gene KD signatures\npairwise correlation", fontsize=10)
    plt.tight_layout()
    out = figures_dir / "06_8p_pairwise_correlation_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_corum_calibration(bench: dict, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.kdeplot(bench["noncorum_vals"], label="Non-CORUM pairs", color="#e74c3c",
                fill=True, alpha=0.25, ax=ax)
    sns.kdeplot(bench["corum_vals"], label="CORUM pairs", color="#2ecc71",
                fill=True, alpha=0.25, ax=ax)
    for (g1, g2), v in zip(bench["target_pairs"], bench["target_vals"]):
        ax.axvline(v, color="#1f77b4", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(np.nan, color="#1f77b4", linewidth=1.2, linestyle="--",
               label="8p gene pairs")  # dummy for legend

    ax.set_xlabel("Pairwise correlation (perturbation-perturbation)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"8p gene-pair correlations vs. CORUM calibration\n"
        f"Mann-Whitney CORUM vs non-CORUM p={bench['p_value']:.2e}",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = figures_dir / "07_8p_corum_calibration.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-pct-hvgs", type=float, default=0.10,
                        help="Fraction of most-variable genes to use as clustering features (default 0.10, matches paper's pct_hvgs=10)")
    parser.add_argument("--n-permutations", type=int, default=10000,
                        help="Number of permutations for null-distribution tests (default 10000)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore cached correlation matrix and recompute")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not H5AD_PATH.exists():
        print(f"[ERROR] {H5AD_PATH} not found. Run scripts/17_preprocess_ipsc.py first.")
        return
    if not CORUM_PATH.exists():
        print(f"[ERROR] {CORUM_PATH} not found.")
        return

    # ── Step 1: correlation matrix ─────────────────────────────────────────
    pb = load_pseudobulk()[0]
    missing = [g for g in GENES_8P_AVAILABLE if g not in pb.index]
    if missing:
        print(f"[ERROR] Expected 8p genes missing from library: {missing}")
        return

    perturb_corrs = build_perturb_corrs(pb, args.top_pct_hvgs, args.force_recompute)

    # ── Step 2: HDBSCAN clustering + 8p lookup ─────────────────────────────
    cluster_labels = run_hdbscan(perturb_corrs)
    cluster_labels.to_frame().to_csv(RESULTS_DIR / "global_hdbscan_clusters.csv")
    print(f"Saved -> results/8p_deletion/global_hdbscan_clusters.csv")

    membership = cluster_labels.loc[GENES_8P_AVAILABLE].to_frame()
    membership.index.name = "gene"
    membership.to_csv(RESULTS_DIR / "8p_cluster_membership.csv")
    print(f"\n8p gene cluster membership:\n{membership}")
    print(f"Saved -> results/8p_deletion/8p_cluster_membership.csv")

    # ── Step 3: pairwise correlations among 8p genes ───────────────────────
    sub_corr = perturb_corrs.loc[GENES_8P_AVAILABLE, GENES_8P_AVAILABLE]
    sub_corr.to_csv(RESULTS_DIR / "8p_pairwise_correlations.csv")
    print(f"\n8p pairwise correlation matrix:\n{sub_corr.round(4)}")
    print(f"Saved -> results/8p_deletion/8p_pairwise_correlations.csv")
    plot_pairwise_heatmap(sub_corr, FIGURES_DIR)

    # ── Step 4: permutation tests ───────────────────────────────────────────
    obs_corr, null_corr, p_corr = permutation_test_mean_corr(
        GENES_8P_AVAILABLE, perturb_corrs, args.n_permutations)
    pd.DataFrame({
        "observed_mean_pairwise_r": [obs_corr],
        "null_mean": [null_corr.mean()],
        "null_std": [null_corr.std()],
        "p_value": [p_corr],
        "n_permutations": [args.n_permutations],
    }).to_csv(RESULTS_DIR / "mean_corr_permutation_test.csv", index=False)
    print("Saved -> results/8p_deletion/mean_corr_permutation_test.csv")

    obs_clust, null_clust, p_clust = permutation_test_coclustering(
        GENES_8P_AVAILABLE, cluster_labels, args.n_permutations)
    pd.DataFrame({
        "observed_coclustered_pairs": [obs_clust],
        "total_pairs": [len(GENES_8P_AVAILABLE) * (len(GENES_8P_AVAILABLE) - 1) // 2],
        "null_mean": [null_clust.mean()],
        "p_value": [p_clust],
        "n_permutations": [args.n_permutations],
    }).to_csv(RESULTS_DIR / "cluster_coclustering_permutation_test.csv", index=False)
    print("Saved -> results/8p_deletion/cluster_coclustering_permutation_test.csv")

    # ── Step 5: CORUM benchmark ─────────────────────────────────────────────
    corum_complexes = load_corum_complexes(CORUM_PATH)
    corum_pairs, noncorum_pairs = get_corum_pairs(
        list(perturb_corrs.index), corum_complexes)
    bench = corum_benchmark(perturb_corrs, corum_pairs, noncorum_pairs, GENES_8P_AVAILABLE)

    bench_rows = []
    for (g1, g2), v, pn, pc in zip(bench["target_pairs"], bench["target_vals"],
                                    bench["pct_vs_noncorum"], bench["pct_vs_corum"]):
        bench_rows.append({
            "gene_1": g1, "gene_2": g2, "pearson_r": v,
            "percentile_vs_noncorum": pn, "percentile_vs_corum": pc,
        })
    summary_df = pd.DataFrame(bench_rows)
    summary_df.attrs["mannwhitney_u"] = bench["u_stat"]
    summary_df.attrs["mannwhitney_p"] = bench["p_value"]
    summary_df.to_csv(RESULTS_DIR / "corum_benchmark_summary.csv", index=False)
    with open(RESULTS_DIR / "corum_benchmark_summary.csv", "a") as f:
        f.write(f"\n# Mann-Whitney CORUM vs non-CORUM: U={bench['u_stat']:.1f}, p={bench['p_value']:.4g}\n")
        f.write(f"# CORUM pairs mean r={bench['corum_vals'].mean():.4f}, "
                f"non-CORUM pairs mean r={bench['noncorum_vals'].mean():.4f}\n")
    print("Saved -> results/8p_deletion/corum_benchmark_summary.csv")

    plot_corum_calibration(bench, FIGURES_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
