"""
12_flipped_analysis.py
----------------------
Scatter plot of Δz (dep_gene KD → paralog expression) vs Δz_flipped
(paralog KD → dep_gene expression) for significant pairs.

Points in the upper-right quadrant show reciprocal upregulation — knocking
down either gene causes the other to go up, suggesting bidirectional
transcriptional compensation.

Output: figures/{cell_line}/06_flipped_scatter.pdf

Usage:
    python scripts/12_flipped_analysis.py --cell-line K562
    python scripts/12_flipped_analysis.py --all
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_1samp, mannwhitneyu, pearsonr

ROOT = Path(__file__).resolve().parent.parent

CELL_LINES = ["K562", "K562_essential", "rpe1", "HCT116", "HEK293T",
              "melanoma", "cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr", "neuron"]

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

QUAD_COLOURS = {
    "reciprocal_up":   "#d62728",   # both positive — upper right
    "reciprocal_down": "#1f77b4",   # both negative — lower left
    "asymmetric":      "#888888",   # one positive, one negative
}

QUAD_LABELS = {
    "reciprocal_up":   "Both ↑  (reciprocal upregulation)",
    "reciprocal_down": "Both ↓  (reciprocal downregulation)",
    "asymmetric":      "Asymmetric",
}


def quadrant(dz, dz_flip):
    if dz > 0 and dz_flip > 0:
        return "reciprocal_up"
    if dz < 0 and dz_flip < 0:
        return "reciprocal_down"
    return "asymmetric"


def run(cell_line: str) -> None:
    path = ROOT / "results" / cell_line / "sig_results.csv"
    if not path.exists():
        print(f"[skip] {cell_line}: sig_results.csv not found.")
        return

    df = pd.read_csv(path)
    if "delta_z_flipped" not in df.columns:
        print(f"[skip] {cell_line}: no delta_z_flipped column — re-run 02_compute_delta_z.py.")
        return

    df = df[df["testable"] == True].dropna(subset=["delta_z", "delta_z_flipped"]).copy()

    if len(df) < 2:
        print(f"[skip] {cell_line}: only {len(df)} pair(s) with both Δz and Δz_flipped — "
              "need to re-run 02_compute_delta_z.py.")
        return

    df["quad"] = df.apply(lambda r: quadrant(r["delta_z"], r["delta_z_flipped"]), axis=1)

    # ── Statistics ────────────────────────────────────────────────────────────
    dz      = df["delta_z"].values
    dz_flip = df["delta_z_flipped"].values

    # 1. One-sample t-test: is mean(delta_z_flipped) > 0?
    t_stat, t_p = ttest_1samp(dz_flip, popmean=0, alternative="greater")

    # 2. Pearson correlation between original and flipped directions
    r_val, r_p = pearsonr(dz, dz_flip)

    # 3. Mann-Whitney: flipped sig pairs vs nonsig background
    ns_path = ROOT / "results" / cell_line / "nonsig_results.csv"
    mw_p = np.nan
    if ns_path.exists():
        ns = pd.read_csv(ns_path).dropna(subset=["delta_z"])
        _, mw_p = mannwhitneyu(dz_flip, ns["delta_z"].values, alternative="greater")

    stats = {
        "cell_line":              cell_line,
        "n_pairs":                len(df),
        "n_reciprocal_up":        (df["quad"] == "reciprocal_up").sum(),
        "n_reciprocal_down":      (df["quad"] == "reciprocal_down").sum(),
        "n_asymmetric":           (df["quad"] == "asymmetric").sum(),
        "mean_delta_z_flipped":   float(np.mean(dz_flip)),
        "ttest_flipped_gt0_p":    float(t_p),
        "pearson_r":              float(r_val),
        "pearson_p":              float(r_p),
        "mannwhitney_vs_nonsig_p": float(mw_p),
    }

    print(f"\n[{cell_line}] {len(df)} pairs with both directions:")
    for q, label in QUAD_LABELS.items():
        print(f"  {label}: {(df['quad']==q).sum()}")
    print(f"\n  Mean Δz_flipped = {stats['mean_delta_z_flipped']:.4f}")
    print(f"  t-test (flipped > 0):        p = {t_p:.3e}")
    print(f"  Pearson r (orig vs flipped): r = {r_val:.3f},  p = {r_p:.3e}")
    print(f"  Mann-Whitney vs nonsig:      p = {mw_p:.3e}")

    out_stats = ROOT / "results" / cell_line / "flipped_stats.csv"
    pd.DataFrame([stats]).to_csv(out_stats, index=False)
    print(f"  Saved stats → {out_stats}")

    figures_dir = ROOT / "figures" / cell_line
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Quadrant shading
    xlim_pad = max(df["delta_z"].abs().max(), df["delta_z_flipped"].abs().max()) * 1.35
    ax.axhline(0, color="lightgrey", lw=0.8, zorder=1)
    ax.axvline(0, color="lightgrey", lw=0.8, zorder=1)
    ax.fill_between([-xlim_pad, 0], 0, xlim_pad,  alpha=0.04, color=QUAD_COLOURS["reciprocal_down"], zorder=0)
    ax.fill_between([0, xlim_pad],  0, xlim_pad,  alpha=0.06, color=QUAD_COLOURS["reciprocal_up"],   zorder=0)
    ax.fill_between([-xlim_pad, 0], -xlim_pad, 0, alpha=0.06, color=QUAD_COLOURS["reciprocal_down"], zorder=0)
    ax.fill_between([0, xlim_pad],  -xlim_pad, 0, alpha=0.04, color=QUAD_COLOURS["reciprocal_up"],   zorder=0)

    # Points
    for q, colour in QUAD_COLOURS.items():
        sub = df[df["quad"] == q]
        ax.scatter(sub["delta_z"], sub["delta_z_flipped"],
                   color=colour, s=70, edgecolors="black",
                   linewidths=0.5, zorder=4, label=f"{QUAD_LABELS[q]} (n={len(sub)})")

    # Labels
    for _, r in df.iterrows():
        ax.text(r["delta_z"] + xlim_pad * 0.02, r["delta_z_flipped"],
                f"{r['dep_gene']}→{r['paralog_gene']}",
                fontsize=6.5, va="center", color="black")

    # Quadrant annotations
    ax.text( xlim_pad * 0.97,  xlim_pad * 0.97, "Reciprocal\nupregulation",
             ha="right", va="top",   fontsize=8, color=QUAD_COLOURS["reciprocal_up"],   style="italic")
    ax.text(-xlim_pad * 0.97, -xlim_pad * 0.97, "Reciprocal\ndownregulation",
             ha="left",  va="bottom", fontsize=8, color=QUAD_COLOURS["reciprocal_down"], style="italic")

    # p-value annotation box
    def fmt_p(p):
        if np.isnan(p):   return "n/a"
        if p < 0.001:     return f"{p:.2e}"
        return f"{p:.3f}"

    stats_text = (
        f"t-test (flipped > 0):   p = {fmt_p(t_p)}\n"
        f"Pearson r = {r_val:.2f}  (p = {fmt_p(r_p)})\n"
        f"Mann-Whitney vs bkg:   p = {fmt_p(mw_p)}"
    )
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))

    ax.set_xlim(-xlim_pad, xlim_pad)
    ax.set_ylim(-xlim_pad, xlim_pad)
    ax.set_xlabel("Δz  (dep_gene KD → paralog expression)", fontsize=11)
    ax.set_ylabel("Δz flipped  (paralog KD → dep_gene expression)", fontsize=11)
    ax.set_title(f"Bidirectional paralog compensation — {cell_line}\n"
                 f"Significant aneuploid vulnerability pairs (n={len(df)})", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_aspect("equal")
    sns.despine(ax=ax)
    plt.tight_layout()

    out = figures_dir / "06_flipped_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell-line", choices=CELL_LINES)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cell_lines = CELL_LINES if args.all else [args.cell_line]
    for cl in cell_lines:
        run(cl)


if __name__ == "__main__":
    main()
