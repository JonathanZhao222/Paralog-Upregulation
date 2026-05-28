"""
11_gsea.py
----------
Pathway enrichment analysis (via Enrichr REST API) for paralog genes
showing the strongest upregulation upon dep_gene knockdown.

Two modes:
  --mode sig   (default) — genes from the 37 significant aneuploid pairs with mean Δz ≥ min_dz
  --mode top   — genes from ALL pairs (sig + nonsig) with Δz ≥ dz-threshold in ANY cell line;
                 intended to identify pathways enriched among the most highly upregulated paralogs
                 regardless of whether the dep_gene was a curated aneuploid vulnerability

Gene sets queried: KEGG_2021_Human, GO_Biological_Process_2023, Reactome_2022

Outputs (saved to results/gsea/ or results/gsea_top/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  upregulated_paralogs.txt      — gene list submitted
  enrichr_<gene_set>.csv        — enrichment results per database
  figures/gsea[_top]/01_enrichment_dotplot.pdf

Usage:
    python scripts/11_gsea.py                         # sig pairs, mean Δz ≥ 0
    python scripts/11_gsea.py --min-dz 0.5            # sig pairs, mean Δz ≥ 0.5
    python scripts/11_gsea.py --mode top --dz-threshold 2.0   # all pairs with Δz > 2 anywhere
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CELL_LINES = ["K562", "K562_essential", "rpe1", "HCT116", "HEK293T", "melanoma",
              "cd4t_rest", "cd4t_stim8hr", "cd4t_stim48hr", "neuron"]

GENE_SETS = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
    "Reactome_2022",
]

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


# ── Collect paralog genes by mean Δz across all pairs (sig + nonsig) ──────────
def collect_top_dz(dz_threshold: float) -> tuple[list[str], pd.DataFrame]:
    """
    For every (dep_gene → paralog_gene) direction in sig and nonsig results,
    compute the mean Δz across all cell lines where that direction was tested.
    Return paralog genes whose mean Δz ≥ dz_threshold.

    This is intentionally broader than collect_mean_dz() (which uses only the
    37 significant aneuploid pairs); here we include all paralog pairs.
    """
    rows = []
    for cl in CELL_LINES:
        for fname in ("sig_results.csv", "nonsig_results.csv"):
            path = ROOT / "results" / cl / fname
            if not path.exists():
                continue
            df = pd.read_csv(path).dropna(subset=["delta_z"])
            if "testable" in df.columns:
                df = df[df["testable"] == True]
            for _, r in df.iterrows():
                rows.append({
                    "paralog_gene": r["paralog_gene"],
                    "dep_gene":     r["dep_gene"],
                    "cell_line":    cl,
                    "delta_z":      r["delta_z"],
                })
    if not rows:
        raise RuntimeError("No results found — run 02_compute_delta_z.py first.")

    all_dz = pd.DataFrame(rows)

    # Mean Δz per paralog_gene across all tested directions and cell lines
    summary = (all_dz.groupby("paralog_gene")
               .agg(mean_dz=("delta_z", "mean"),
                    max_dz=("delta_z", "max"),
                    n_pairs=("delta_z", "count"))
               .reset_index()
               .sort_values("mean_dz", ascending=False))

    selected = summary[summary["mean_dz"] >= dz_threshold]
    gene_list = selected["paralog_gene"].tolist()
    print(f"Paralog genes with mean Δz ≥ {dz_threshold} (all pairs, all cell lines): {len(gene_list)}")
    if gene_list:
        print(f"  {gene_list}")
    return gene_list, summary


# ── Collect mean Δz per paralog_gene across all cell lines ────────────────────
def collect_mean_dz(min_dz: float) -> pd.DataFrame:
    rows = []
    for cl in CELL_LINES:
        path = ROOT / "results" / cl / "sig_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["testable"] == True].dropna(subset=["delta_z"])
        for _, r in df.iterrows():
            rows.append({"paralog_gene": r["paralog_gene"],
                         "dep_gene":     r["dep_gene"],
                         "cell_line":    cl,
                         "delta_z":      r["delta_z"]})
    if not rows:
        raise RuntimeError("No sig_results found — run 02_compute_delta_z.py first.")

    all_dz = pd.DataFrame(rows)
    mean_dz = (all_dz.groupby("paralog_gene")["delta_z"]
               .mean()
               .reset_index()
               .rename(columns={"delta_z": "mean_delta_z"})
               .sort_values("mean_delta_z", ascending=False))

    n_cell_lines = (all_dz.groupby("paralog_gene")["cell_line"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"cell_line": "n_cell_lines"}))
    mean_dz = mean_dz.merge(n_cell_lines, on="paralog_gene")

    gene_list = mean_dz[mean_dz["mean_delta_z"] >= min_dz]["paralog_gene"].tolist()
    print(f"Paralog genes with mean Δz ≥ {min_dz}: {len(gene_list)}")
    print(f"  {gene_list}")
    return mean_dz, gene_list


# ── Enrichr API ───────────────────────────────────────────────────────────────
def enrichr_submit(genes: list[str], description: str = "query") -> str:
    # Enrichr expects multipart/form-data
    files = {
        "list":        (None, "\n".join(genes)),
        "description": (None, description),
    }
    resp = requests.post(f"{ENRICHR_BASE}/addList", files=files, timeout=30)
    resp.raise_for_status()
    return resp.json()["userListId"]


def enrichr_results(user_list_id: str, gene_set: str) -> pd.DataFrame:
    url = f"{ENRICHR_BASE}/enrich?userListId={user_list_id}&backgroundType={gene_set}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get(gene_set, [])
    if not data:
        return pd.DataFrame()
    cols = ["rank", "term", "p_value", "z_score", "combined_score",
            "overlapping_genes", "adj_p_value", "old_p_value", "old_adj_p_value"]
    df = pd.DataFrame(data, columns=cols[:len(data[0])])
    df["gene_set"] = gene_set
    df["overlapping_genes"] = df["overlapping_genes"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_dotplot(all_results: pd.DataFrame, figures_dir: Path, title: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    top = (all_results[all_results["adj_p_value"] < 0.05]
           .sort_values("combined_score", ascending=False)
           .groupby("gene_set")
           .head(10)
           .reset_index(drop=True))

    if top.empty:
        print("No terms with adj_p_value < 0.05 — lowering threshold to p < 0.1 for plot.")
        top = (all_results[all_results["p_value"] < 0.1]
               .sort_values("combined_score", ascending=False)
               .groupby("gene_set")
               .head(10)
               .reset_index(drop=True))

    if top.empty:
        print("No enriched terms to plot.")
        return

    top["-log10_p"] = -np.log10(top["p_value"].clip(lower=1e-300))
    top["term_short"] = top["term"].str[:55]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.35)))
    sc = ax.scatter(top["-log10_p"], top["term_short"],
                    c=top["combined_score"], cmap="YlOrRd",
                    s=80, edgecolors="grey", linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label="Combined score", shrink=0.6)
    ax.axvline(-np.log10(0.05), color="grey", lw=0.8, ls="--", label="p=0.05")
    ax.set_xlabel("-log₁₀(p-value)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()

    out = figures_dir / "01_enrichment_dotplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Shared enrichment runner ──────────────────────────────────────────────────
def run_enrichment(gene_list: list[str], out_dir: Path, figures_dir: Path,
                   description: str, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if len(gene_list) < 3:
        print(f"[warn] Only {len(gene_list)} genes — enrichment unlikely to be informative.")

    gene_list_path = out_dir / "gene_list.txt"
    gene_list_path.write_text("\n".join(gene_list))
    print(f"Gene list saved → {gene_list_path}")

    print(f"\nSubmitting {len(gene_list)} genes to Enrichr ...")
    try:
        user_list_id = enrichr_submit(gene_list, description=description)
        print(f"  Enrichr list ID: {user_list_id}")
    except Exception as e:
        print(f"  ERROR: Could not connect to Enrichr: {e}")
        return

    all_results = []
    for gs in GENE_SETS:
        print(f"  Fetching {gs} ...")
        time.sleep(0.5)
        try:
            df = enrichr_results(user_list_id, gs)
            if not df.empty:
                df.to_csv(out_dir / f"enrichr_{gs}.csv", index=False)
                print(f"    {len(df)} terms; top hit: {df.iloc[0]['term']} "
                      f"(p={df.iloc[0]['p_value']:.3e})")
                all_results.append(df)
        except Exception as e:
            print(f"    ERROR fetching {gs}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(out_dir / "enrichr_all.csv", index=False)
        plot_dotplot(combined, figures_dir, title)
        sig_terms = combined[combined["adj_p_value"] < 0.05]
        print(f"\nSignificant terms (adj_p < 0.05): {len(sig_terms)}")
        if not sig_terms.empty:
            print(sig_terms[["gene_set", "term", "adj_p_value",
                              "overlapping_genes"]].to_string(index=False))
    else:
        print("No enrichment results returned.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sig", "top"], default="sig",
                        help="'sig': use significant pairs only (default); "
                             "'top': use all pairs with Δz ≥ dz-threshold")
    parser.add_argument("--min-dz", type=float, default=0.0,
                        help="[sig mode] Minimum mean Δz to include a paralog gene (default: 0.0)")
    parser.add_argument("--dz-threshold", type=float, default=2.0,
                        help="[top mode] Δz threshold; include any paralog seen above this "
                             "in any cell line (default: 2.0)")
    args = parser.parse_args()

    if args.mode == "top":
        out_dir     = ROOT / "results" / "gsea_top"
        figures_dir = ROOT / "figures" / "gsea_top"
        out_dir.mkdir(parents=True, exist_ok=True)
        gene_list, summary_df = collect_top_dz(args.dz_threshold)
        summary_df.to_csv(out_dir / "mean_dz_all_pairs.csv", index=False)
        title = (f"Enrichr pathway enrichment\n"
                 f"All-pair paralogs with mean Δz ≥ {args.dz_threshold} "
                 f"(n={len(gene_list)} genes)")
        run_enrichment(gene_list, out_dir, figures_dir,
                       description=f"top_dz_paralogs_ge{args.dz_threshold}",
                       title=title)
    else:
        out_dir     = ROOT / "results" / "gsea"
        figures_dir = ROOT / "figures" / "gsea"
        mean_dz, gene_list = collect_mean_dz(args.min_dz)
        mean_dz.to_csv(out_dir / "mean_dz_per_paralog.csv", index=False)
        title = (f"Enrichr pathway enrichment\n"
                 f"Significant aneuploid pairs — paralog genes with mean Δz ≥ {args.min_dz}")
        run_enrichment(gene_list, out_dir, figures_dir,
                       description="upregulated_paralogs",
                       title=title)


if __name__ == "__main__":
    main()
