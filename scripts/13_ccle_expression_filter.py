"""
13_ccle_expression_filter.py
----------------------------
Downloads the CCLE 2019 expression matrix (log2 TPM+1, protein-coding genes)
from Figshare, extracts baseline expression for the paralog genes in our
sig_37_paralog.xlsx and non_sig_paralog.xlsx lists, and saves a compact
reference file used by 02_compute_delta_z.py to filter lowly expressed genes.

CCLE source
~~~~~~~~~~~
  DepMap / CCLE 2019 (Figshare article 11384241)
  File: CCLE_expression.csv  (~300 MB)
  Rows: DepMap cell line IDs (ACH-XXXXXX)
  Cols: gene symbols + Entrez IDs, e.g. "NRAS (4893)"
  Values: log2(TPM + 1)

Cell line mapping
~~~~~~~~~~~~~~~~~
  K562       → ACH-000551   (chronic myelogenous leukemia)
  rpe1       → ACH-000634   (hTERT RPE-1, retinal pigment epithelial)
  K562_essential uses the same K562 expression profile.
  CD4T / neuron / melanoma are not in CCLE (primary or iPSC-derived).

Output
~~~~~~
  data/raw/ccle_expression_paralogs.csv
    Columns: gene, K562_log2tpm, rpe1_log2tpm
    One row per unique paralog gene across all pairs.

Usage:
    python scripts/13_ccle_expression_filter.py

  After this runs, re-run:
    python scripts/02_compute_delta_z.py --all
  to add paralog_ccle_log2tpm to sig_results.csv and nonsig_results.csv.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

CCLE_ARTICLE_ID = 11384241
CCLE_FILENAME   = "CCLE_expression.csv"
CCLE_DEST       = DATA_DIR / "CCLE_expression.csv"
OUT_PATH        = DATA_DIR / "ccle_expression_paralogs.csv"

# Known DepMap IDs for the cell lines we analyse
CELL_LINE_DEPMAP = {
    "K562":           "ACH-000551",
    "K562_essential": "ACH-000551",   # same cell line
    "rpe1":           "ACH-000634",
}

CHUNK_SIZE = 1 << 20  # 1 MB

SIG_XL    = DATA_DIR / "sig_37_paralog.xlsx"
NONSIG_XL = DATA_DIR / "non_sig_paralog.xlsx"


def get_download_url() -> str:
    resp = requests.get(
        f"https://api.figshare.com/v2/articles/{CCLE_ARTICLE_ID}/files",
        timeout=30,
    )
    resp.raise_for_status()
    for f in resp.json():
        if f["name"] == CCLE_FILENAME:
            return f["download_url"]
    raise RuntimeError(f"'{CCLE_FILENAME}' not found in Figshare article {CCLE_ARTICLE_ID}")


def download_ccle() -> None:
    if CCLE_DEST.exists():
        print(f"[skip] {CCLE_FILENAME} already exists.")
        return
    url = get_download_url()
    print(f"Downloading {CCLE_FILENAME} (~300 MB) ...")
    resp = requests.get(url, stream=True, timeout=3600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(CCLE_DEST, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=CCLE_FILENAME, ncols=80,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            fh.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved → {CCLE_DEST}")


def collect_paralog_genes() -> list[str]:
    sig    = pd.read_excel(SIG_XL)
    nonsig = pd.read_excel(NONSIG_XL)
    genes  = set(sig["paralog_gene"].dropna())
    genes |= set(sig["dep_gene"].dropna())
    genes |= set(nonsig["para_gene_1"].dropna())
    genes |= set(nonsig["para_gene_2"].dropna())
    return sorted(genes)


def build_reference() -> None:
    print("Collecting paralog genes from pair lists ...")
    target_genes = collect_paralog_genes()
    print(f"  {len(target_genes)} unique genes to look up")

    print(f"Loading {CCLE_FILENAME} ...")
    ccle = pd.read_csv(CCLE_DEST, index_col=0)
    print(f"  {ccle.shape[0]} cell lines × {ccle.shape[1]} genes")

    # Strip Entrez ID from column names: "NRAS (4893)" → "NRAS"
    ccle.columns = [c.split(" (")[0] for c in ccle.columns]

    # Verify target cell lines are present
    for cl, depmap_id in CELL_LINE_DEPMAP.items():
        if depmap_id not in ccle.index:
            print(f"  [warn] {cl} ({depmap_id}) not found in CCLE index — skipping.")

    # Build output: one row per gene, one column per cell line
    records = []
    missing = []
    for gene in target_genes:
        row = {"gene": gene}
        if gene not in ccle.columns:
            missing.append(gene)
            for cl in CELL_LINE_DEPMAP:
                row[f"{cl}_log2tpm"] = np.nan
        else:
            for cl, depmap_id in CELL_LINE_DEPMAP.items():
                if depmap_id in ccle.index:
                    row[f"{cl}_log2tpm"] = float(ccle.loc[depmap_id, gene])
                else:
                    row[f"{cl}_log2tpm"] = np.nan
        records.append(row)

    if missing:
        print(f"  {len(missing)} genes not found in CCLE: {missing[:10]}{'...' if len(missing)>10 else ''}")

    out = pd.DataFrame(records)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved reference → {OUT_PATH}")
    print(f"  {len(out)} genes, columns: {out.columns.tolist()}")

    # Summary
    threshold = 1.0  # log2(TPM+1) > 1 ≈ TPM > 1
    for cl in CELL_LINE_DEPMAP:
        col = f"{cl}_log2tpm"
        if col in out.columns:
            expressed = (out[col] > threshold).sum()
            print(f"  {cl}: {expressed}/{len(out)} paralog genes expressed (log2TPM > {threshold})")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_ccle()
    build_reference()
    print("\nNext: python scripts/02_compute_delta_z.py --all")
    print("  This will add paralog_ccle_log2tpm to sig_results.csv.")
    print("  Then use: python scripts/03_compare_visualize.py --cell-line K562 --min-expr 1.0")


if __name__ == "__main__":
    main()
