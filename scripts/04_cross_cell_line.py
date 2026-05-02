"""
04_cross_cell_line.py
---------------------
Compares Δz results across cell lines for significant paralog pairs.

Requires results/{cell_line}/sig_results.csv to exist for each cell line.
Run 02_compute_delta_z.py --all first.

Figures produced (saved to figures/cross_cell_line/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  01_delta_z_scatter.pdf   – Δz K562 vs Δz RPE1, one point per pair
  02_delta_z_heatmap.pdf   – heatmap of Δz: rows=pairs, cols=cell lines

Usage:
    python scripts/04_cross_cell_line.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures" / "cross_cell_line"

# Cell lines to compare — add more here as data becomes available
CELL_LINES = ["K562", "K562_essential", "rpe1", "HCT116", "HEK293T", "melanoma",
              "cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr"]

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


# ── Load and merge ─────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    dfs = []
    for cl in CELL_LINES:
        path = ROOT / "results" / cl / "sig_results.csv"
        if not path.exists():
            print(f"[skip] {cl}: results not found — run 02_compute_delta_z.py --cell-line {cl}")
            continue
        df = pd.read_csv(path)
        df = df[df["testable"] == True].dropna(subset=["delta_z"])
        df = df[["dep_gene", "paralog_gene", "mean_identical_score", "delta_z"]].copy()
        df = df.rename(columns={"delta_z": f"delta_z_{cl}"})
        dfs.append(df)

    if len(dfs) < 2:
        raise RuntimeError(f"Need results for at least 2 cell lines (found {len(dfs)}). "
                           "Run 02_compute_delta_z.py for more cell lines first.")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["dep_gene", "paralog_gene", "mean_identical_score"], how="outer")

    print(f"Merged {len(merged)} unique sig pairs across {len(dfs)} cell lines.")
    return merged


# ── Figure 1 — pairwise scatter grid ─────────────────────────────────────────
def plot_scatter(merged: pd.DataFrame, available_cls: list[str]) -> None:
    dz_cols = [f"delta_z_{cl}" for cl in available_cls]
    pairs = [(available_cls[i], available_cls[j])
             for i in range(len(available_cls))
             for j in range(i + 1, len(available_cls))]

    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 5 * nrows),
                              squeeze=False)

    for idx, (cl_x, cl_y) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        x_col, y_col = f"delta_z_{cl_x}", f"delta_z_{cl_y}"
        df = merged.dropna(subset=[x_col, y_col])

        if df.empty:
            ax.set_visible(False)
            continue

        lim = max(df[[x_col, y_col]].abs().max()) * 1.15
        ax.scatter(df[x_col], df[y_col], color="#2c7bb6", edgecolors="black",
                   linewidths=0.5, s=50, zorder=3, alpha=0.85)
        for _, row in df.iterrows():
            ax.text(row[x_col] + lim * 0.02, row[y_col],
                    row["dep_gene"], fontsize=6.5, va="center")

        ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.9, ls="--", zorder=1)
        ax.axhline(0, color="lightgrey", lw=0.6)
        ax.axvline(0, color="lightgrey", lw=0.6)

        corr = df[x_col].corr(df[y_col])
        ax.text(0.05, 0.95, f"r = {corr:.2f}  (n={len(df)})",
                transform=ax.transAxes, fontsize=9, va="top")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Δz  ({cl_x})", fontsize=10)
        ax.set_ylabel(f"Δz  ({cl_y})", fontsize=10)
        ax.set_aspect("equal")
        sns.despine(ax=ax)

    # Hide unused subplots
    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Paralog Δz consistency across cell lines (significant pairs)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "01_delta_z_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Figure 2 — heatmap ────────────────────────────────────────────────────────
def plot_heatmap(merged: pd.DataFrame, available_cls: list[str]) -> None:
    dz_cols = [f"delta_z_{cl}" for cl in available_cls]
    # Only keep columns that actually exist in the merged dataframe
    dz_cols = [c for c in dz_cols if c in merged.columns]
    if not dz_cols:
        print("No Δz columns found for heatmap — skipping.")
        return

    df = merged.dropna(subset=dz_cols, how="all").copy()
    df["pair"] = df["dep_gene"] + " → " + df["paralog_gene"]
    heat = df.set_index("pair")[dz_cols].copy()
    heat.columns = [c.replace("delta_z_", "") for c in dz_cols]
    heat = heat.sort_values(heat.columns[0], ascending=False)

    abs_max = heat.abs().max().max()
    fig, ax = plt.subplots(figsize=(2 + len(dz_cols) * 1.2, max(6, len(heat) * 0.35)))
    sns.heatmap(heat, ax=ax, cmap="RdBu_r", center=0,
                vmin=-abs_max, vmax=abs_max,
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Δz", "shrink": 0.6})
    ax.set_xlabel("Cell line", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Paralog Δz across cell lines\n(significant pairs)", fontsize=11)
    plt.tight_layout()

    out = FIGURES_DIR / "02_delta_z_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_all()

    # Only work with cell lines that are present as columns
    available_cls = [cl for cl in CELL_LINES
                     if f"delta_z_{cl}" in merged.columns]
    print(f"Cell lines with results: {available_cls}")

    plot_scatter(merged, available_cls)
    plot_heatmap(merged, available_cls)

    out = ROOT / "results" / "cross_cell_line_delta_z.csv"
    merged.to_csv(out, index=False)
    print(f"Saved merged results → {out}")


if __name__ == "__main__":
    main()
