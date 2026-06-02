"""
10_rank_all_pairs.py
--------------------
Ranks ALL paralog pairs (significant + non-significant) by Δz for a given
cell line and shows where the 37 aneuploidy vulnerability pairs fall in
that ranking.

Outputs (saved to results/{cell_line}/ and figures/{cell_line}/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  all_pairs_ranked.csv          — full ranked table with sig/nonsig label
  05_rank_all_pairs.pdf         — rank plot with sig pairs highlighted

Usage:
    python scripts/10_rank_all_pairs.py --cell-line K562
    python scripts/10_rank_all_pairs.py --all
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import percentileofscore

ROOT = Path(__file__).resolve().parent.parent

CELL_LINES = ["K562", "K562_essential", "rpe1", "HCT116", "HEK293T", "melanoma",
              "cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr", "neuron"]

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def run(cell_line: str) -> None:
    results_dir = ROOT / "results" / cell_line
    figures_dir = ROOT / "figures" / cell_line
    figures_dir.mkdir(parents=True, exist_ok=True)

    sig_path  = results_dir / "sig_results.csv"
    ns_path   = results_dir / "nonsig_results.csv"
    if not sig_path.exists() or not ns_path.exists():
        print(f"[skip] {cell_line}: run 02_compute_delta_z.py first.")
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    sig = pd.read_csv(sig_path)
    sig = sig[sig["testable"] == True].dropna(subset=["delta_z"]).copy()
    sig["group"] = "Aneuploidy vulnerability"

    ns = pd.read_csv(ns_path).dropna(subset=["delta_z"]).copy()
    ns["group"] = "Non-significant background"

    # Use only the columns present in both; add p-value columns from sig if available
    pval_cols = [c for c in ["empirical_pval", "empirical_fdr"] if c in sig.columns]
    shared_cols = ["dep_gene", "paralog_gene", "mean_identical_score", "delta_z", "group"]
    sig_out = sig[shared_cols + pval_cols].copy()
    ns_out  = ns[shared_cols].copy()
    for c in pval_cols:
        ns_out[c] = float("nan")
    combined = pd.concat([sig_out, ns_out], ignore_index=True)
    combined = combined.sort_values("delta_z", ascending=False).reset_index(drop=True)
    combined["rank"] = combined.index + 1
    combined["rank_pct"] = combined["rank"] / len(combined) * 100

    # ── Stats ─────────────────────────────────────────────────────────────────
    all_dz = combined["delta_z"].values
    sig_rows = combined[combined["group"] == "Aneuploidy vulnerability"].copy()
    print(f"\n[{cell_line}] Total pairs ranked: {len(combined):,}")
    print(f"  Aneuploidy vulnerability pairs: {len(sig_rows)}")
    for _, r in sig_rows.sort_values("delta_z", ascending=False).iterrows():
        pct = percentileofscore(all_dz, r["delta_z"], kind="rank")
        print(f"    {r['dep_gene']:10s} → {r['paralog_gene']:10s}  "
              f"Δz={r['delta_z']:+.3f}  rank {int(r['rank'])}/{len(combined)}  "
              f"top {100-pct:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = results_dir / "all_pairs_ranked.csv"
    combined.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

    # ── Figure — top 100 pairs only ───────────────────────────────────────────
    top100 = combined.head(100).copy()
    top100_sig = top100[top100["group"] == "Aneuploidy vulnerability"]
    top100_bg  = top100[top100["group"] == "Non-significant background"]

    fig, ax = plt.subplots(figsize=(13, 5))

    # Grey bars for non-sig pairs
    ax.bar(top100_bg["rank"], top100_bg["delta_z"],
           color="#b0b0b0", edgecolor="white", linewidth=0.3, zorder=2,
           label=f"Non-significant background (top 100)")

    # Red bars for sig pairs
    ax.bar(top100_sig["rank"], top100_sig["delta_z"],
           color="#d62728", edgecolor="white", linewidth=0.3, zorder=3,
           label=f"Aneuploidy vulnerability (n={len(top100_sig)})")

    ax.axhline(0, color="black", lw=0.8, ls="--")

    # Labels on sig pairs — offset above bar using data range
    dz_range = top100["delta_z"].max() - top100["delta_z"].min()
    for _, r in top100_sig.iterrows():
        ax.text(r["rank"], r["delta_z"] + dz_range * 0.02,
                f"{r['dep_gene']}→{r['paralog_gene']}",
                ha="center", va="bottom", fontsize=7, rotation=45, color="#d62728")

    ax.set_xlabel("Rank among all paralog pairs (1 = highest Δz)", fontsize=11)
    ax.set_ylabel("Δz  (paralog z-score: KD − control)", fontsize=11)
    ax.set_title(f"Top 100 paralog pairs by Δz — {cell_line}\n"
                 f"Aneuploidy vulnerability pairs in red  "
                 f"(total ranked: {len(combined):,})", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_fig = figures_dir / "05_rank_all_pairs.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_fig}")


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
