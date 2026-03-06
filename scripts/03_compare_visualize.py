"""
03_compare_visualize.py
-----------------------
Loads the Δz results from 02_compute_delta_z.py and:

  1. Runs statistical tests comparing sig vs non-sig Δz distributions.
  2. Saves a summary CSV.
  3. Generates four publication-quality figures.

Statistical tests
~~~~~~~~~~~~~~~~~
  (a) One-sample t-test  : are Δz values for sig pairs > 0?
  (b) Mann-Whitney U test : are Δz values for sig pairs > non-sig pairs?
      (one-sided, H1: sig > non-sig)

Figures produced
~~~~~~~~~~~~~~~~
  figures/01_sig_delta_z_barplot.pdf  – per-pair Δz for the 37 sig pairs,
                                        sorted, annotated with paralog gene name
  figures/02_sig_vs_nonsig_violin.pdf – violin + box comparing both groups
  figures/03_ecdf_comparison.pdf      – empirical CDF of Δz, both groups
  figures/04_delta_z_vs_identity.pdf  – Δz vs sequence identity (sig pairs)

Usage:
    python scripts/03_compare_visualize.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_1samp
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RESULTS  = ROOT / "results"
FIGURES  = ROOT / "figures"

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # editable text in PDF

COLOUR = {"Significant": "#d62728", "Non-significant": "#1f77b4"}
ALPHA  = 0.85


# ── Load ──────────────────────────────────────────────────────────────────────
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    sig_path = RESULTS / "sig_results.csv"
    ns_path  = RESULTS / "nonsig_results.csv"
    for p in (sig_path, ns_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run 02_compute_delta_z.py first.")

    sig   = pd.read_csv(sig_path)
    nonsig = pd.read_csv(ns_path)

    sig    = sig[sig["testable"] == True].dropna(subset=["delta_z"]).copy()
    nonsig = nonsig.dropna(subset=["delta_z"]).copy()

    sig["group"]    = "Significant"
    nonsig["group"] = "Non-significant"

    print(f"Loaded: {len(sig)} testable sig pairs, {len(nonsig):,} non-sig directions")
    return sig, nonsig


# ── Statistics ────────────────────────────────────────────────────────────────
def run_stats(sig: pd.DataFrame, nonsig: pd.DataFrame) -> dict:
    z_sig = sig["delta_z"].values
    z_ns  = nonsig["delta_z"].values

    # (a) One-sample t-test: mean Δz > 0 for sig pairs?
    t_stat, t_p = ttest_1samp(z_sig, popmean=0, alternative="greater")

    # (b) Mann-Whitney U: sig Δz > non-sig Δz?
    u_stat, u_p = mannwhitneyu(z_sig, z_ns, alternative="greater")

    stats = {
        "n_sig":         len(z_sig),
        "n_nonsig":      len(z_ns),
        "mean_sig":      float(np.mean(z_sig)),
        "median_sig":    float(np.median(z_sig)),
        "frac_pos_sig":  float(np.mean(z_sig > 0)),
        "mean_nonsig":   float(np.mean(z_ns)),
        "median_nonsig": float(np.median(z_ns)),
        "frac_pos_nonsig": float(np.mean(z_ns > 0)),
        "ttest_t":       float(t_stat),
        "ttest_p":       float(t_p),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(u_p),
    }

    print("\n=== Statistical results ===")
    print(f"  Significant pairs      n={stats['n_sig']}")
    print(f"    mean Δz   = {stats['mean_sig']:.4f}")
    print(f"    median Δz = {stats['median_sig']:.4f}")
    print(f"    fraction Δz > 0 = {stats['frac_pos_sig']:.2f}")
    print(f"    One-sample t-test (Δz > 0) : t={t_stat:.3f}, p={t_p:.3e}")
    print(f"\n  Non-significant pairs  n={stats['n_nonsig']:,}")
    print(f"    mean Δz   = {stats['mean_nonsig']:.4f}")
    print(f"    median Δz = {stats['median_nonsig']:.4f}")
    print(f"    fraction Δz > 0 = {stats['frac_pos_nonsig']:.2f}")
    print(f"\n  Mann-Whitney U (sig > non-sig): U={u_stat:.0f}, p={u_p:.3e}")

    return stats


def _pval_str(p: float) -> str:
    if p < 1e-300:
        return "p < 1×10⁻³⁰⁰"
    exp = int(np.floor(np.log10(p)))
    man = p / 10**exp
    return f"p = {man:.1f}×10^{exp}"


# ── Figure 1 — per-pair barplot ───────────────────────────────────────────────
def plot_barplot(sig: pd.DataFrame) -> None:
    df = sig.sort_values("delta_z", ascending=False).reset_index(drop=True)
    colours = [COLOUR["Significant"] if v >= 0 else COLOUR["Non-significant"]
               for v in df["delta_z"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(df)), df["delta_z"], color=colours,
           edgecolor="white", linewidth=0.4, alpha=ALPHA)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=3)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["dep_gene"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δz  (paralog z-score: KD − control)", fontsize=11)
    ax.set_title(
        f"Paralog upregulation upon dep_gene CRISPRi knockdown\n"
        f"Significant pairs (n={len(df)}), Replogle 2022 K562 gwps",
        fontsize=11,
    )

    # Label each bar with the paralog gene name
    for i, row in df.iterrows():
        y_off = 0.04 if row["delta_z"] >= 0 else -0.04
        va    = "bottom" if row["delta_z"] >= 0 else "top"
        ax.text(i, row["delta_z"] + y_off, row["paralog_gene"],
                ha="center", va=va, fontsize=6.5, rotation=90, color="black")

    sns.despine(ax=ax)
    plt.tight_layout()
    out = FIGURES / "01_sig_delta_z_barplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


# ── Figure 2 — violin comparison ─────────────────────────────────────────────
def plot_violin(sig: pd.DataFrame, nonsig: pd.DataFrame, stats: dict) -> None:
    # Subsample non-sig to keep plot rendering fast (5 000 pts is plenty for a violin)
    ns_sample = nonsig.sample(min(5000, len(nonsig)), random_state=42)
    combined  = pd.concat([sig[["delta_z", "group"]], ns_sample[["delta_z", "group"]]],
                           ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.violinplot(
        data=combined, x="group", y="delta_z",
        palette=COLOUR, inner="box", cut=0,
        density_norm="width", ax=ax, alpha=ALPHA,
    )
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=3)

    # Overlay individual sig points
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(sig))
    ax.scatter(
        np.zeros(len(sig)) + jitter, sig["delta_z"],
        color="black", s=18, zorder=5, alpha=0.7, linewidths=0,
    )

    # Annotate with Mann-Whitney p-value
    ax.text(
        0.5, 1.02, f"Mann-Whitney U  {_pval_str(stats['mannwhitney_p'])}",
        transform=ax.transAxes, ha="center", fontsize=10, style="italic",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Δz  (paralog z-score: KD − control)", fontsize=11)
    ax.set_title("Paralog upregulation: significant vs non-significant pairs", fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = FIGURES / "02_sig_vs_nonsig_violin.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


# ── Figure 3 — ECDF ───────────────────────────────────────────────────────────
def plot_ecdf(sig: pd.DataFrame, nonsig: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, df in [("Non-significant", nonsig), ("Significant", sig)]:
        vals = np.sort(df["delta_z"].values)
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, ecdf, label=f"{label}  (n={len(vals):,})",
                color=COLOUR[label], linewidth=2.2)

    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Δz  (paralog z-score: KD − control)", fontsize=11)
    ax.set_ylabel("Cumulative proportion", fontsize=11)
    ax.set_title("Empirical CDF of paralog Δz", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = FIGURES / "03_ecdf_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


# ── Figure 4 — Δz vs sequence identity ───────────────────────────────────────
def plot_identity_scatter(sig: pd.DataFrame) -> None:
    df = sig.dropna(subset=["delta_z", "mean_identical_score"])
    if df.empty:
        print("No data for identity scatter — skipping figure 4.")
        return

    abs_max = df["delta_z"].abs().max()
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["mean_identical_score"], df["delta_z"],
        c=df["delta_z"], cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
        s=70, edgecolors="black", linewidths=0.5, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="Δz", shrink=0.8)

    for _, row in df.iterrows():
        ax.text(
            row["mean_identical_score"] + 0.6, row["delta_z"],
            row["dep_gene"], fontsize=7, va="center",
        )

    ax.axhline(0, color="grey", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Mean pairwise sequence identity (%)", fontsize=11)
    ax.set_ylabel("Δz  (paralog z-score: KD − control)", fontsize=11)
    ax.set_title("Δz vs sequence identity for significant pairs", fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = FIGURES / "04_delta_z_vs_identity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    sig, nonsig = load()
    stats = run_stats(sig, nonsig)

    # Save summary statistics
    summary_path = RESULTS / "summary_comparison.csv"
    pd.DataFrame([stats]).to_csv(summary_path, index=False)
    print(f"\nSaved summary → {summary_path.name}")

    print("\nGenerating figures ...")
    plot_barplot(sig)
    plot_violin(sig, nonsig, stats)
    plot_ecdf(sig, nonsig)
    plot_identity_scatter(sig)

    print(f"\nAll figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
