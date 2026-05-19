"""
09_cd4t_temporal.py
-------------------
Generates a grouped bar chart for CD4+ T cell Δz values across three
stimulation states: resting, 8-hour stimulated, 48-hour stimulated.

For each significant dep_gene → paralog_gene pair, three adjacent bars
are shown (one per timepoint), colour-coded by timepoint. Pairs are
sorted by their resting Δz value.

Output: figures/cd4t_temporal/01_temporal_dynamics.pdf

Usage:
    python scripts/09_cd4t_temporal.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures" / "cd4t_temporal"

TIMEPOINTS = ["cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr"]
LABELS     = ["Resting", "8 hr", "48 hr"]
COLOURS    = ["#2c7bb6", "#fdae61", "#d7191c"]  # blue → orange → red


sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def load_timepoint(cell_line: str) -> pd.DataFrame:
    path = ROOT / "results" / cell_line / "sig_results.csv"
    df = pd.read_csv(path)
    df = df[df["testable"] == True].dropna(subset=["delta_z"])
    df["pair"] = df["dep_gene"] + " → " + df["paralog_gene"]
    return df.set_index("pair")[["delta_z"]].rename(columns={"delta_z": cell_line})


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Merge all three timepoints on pair label
    dfs = [load_timepoint(cl) for cl in TIMEPOINTS]
    merged = dfs[0].join(dfs[1], how="outer").join(dfs[2], how="outer")
    merged.columns = LABELS

    # Only keep pairs present in at least 2 timepoints; sort by resting Δz
    merged = merged[merged.notna().sum(axis=1) >= 2]
    merged = merged.sort_values(LABELS[0], ascending=False)
    print(f"Pairs with ≥2 timepoints: {len(merged)}")

    n_pairs  = len(merged)
    n_groups = len(LABELS)
    bar_w    = 0.25
    group_gap = 0.85          # centre-to-centre spacing between pair groups
    offsets  = np.array([-bar_w, 0, bar_w])

    fig, ax = plt.subplots(figsize=(max(14, n_pairs * 0.6), 6))

    centres = np.arange(n_pairs) * group_gap

    for t_idx, (label, colour) in enumerate(zip(LABELS, COLOURS)):
        vals = merged[label].values.astype(float)
        xs   = centres + offsets[t_idx]
        bar_colours = [colour if not np.isnan(v) else "none" for v in vals]
        ax.bar(xs, np.nan_to_num(vals), width=bar_w,
               color=bar_colours, edgecolor="white", linewidth=0.3,
               label=label, zorder=3)

    ax.axhline(0, color="black", lw=0.9, ls="--", zorder=4)
    ax.set_xticks(centres)
    ax.set_xticklabels(merged.index, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Δz  (paralog z-score: KD − control)", fontsize=11)
    ax.set_title("Paralog upregulation across CD4+ T cell activation states\n"
                 "Significant pairs sorted by resting Δz", fontsize=11)
    ax.legend(title="Timepoint", fontsize=9, title_fontsize=9,
              frameon=False, loc="upper right")
    sns.despine(ax=ax)
    plt.tight_layout()

    out = FIGURES_DIR / "01_temporal_dynamics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
