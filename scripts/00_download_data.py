"""
00_download_data.py
-------------------
Downloads Perturb-seq normalized pseudobulk datasets.

Sources
~~~~~~~
  Replogle 2022 (Figshare 20029387) — pseudobulk, ready to use directly:
    K562_gwps_normalized_bulk_01.h5ad       K562 genome-wide (~357 MB)
    K562_essential_normalized_bulk_01.h5ad  K562 essential genes (~80 MB)
    rpe1_normalized_bulk_01.h5ad            RPE1 near-diploid (~91 MB)

  X-Atlas/Orion 2025 (Figshare 29190726) — single-cell, needs preprocessing:
    HCT116_filtered_dual_guide_cells.h5ad   HCT116 colorectal cancer
    HEK293T_filtered_dual_guide_cells.h5ad  HEK293T kidney
    → Run 05_preprocess_xatlas.py after downloading these.

  GSE291147 (GEO, Nature 2026) — CRISPRi Perturb-seq in melanoma cells:
    GSE291147_Dual_omics_RNA_gene_count_matrix.RDS   RNA counts (genes × cells)
    GSE291147_Dual_omics_RNA_cell_metadata.csv.gz    Cell metadata / pert labels
    GSE291147_Dual_omics_sgRNA_count_matrix.RDS      sgRNA assignments per cell
    → Run 06_convert_rds.R then 06_preprocess_gse291147.py after downloading.

  Marson/Dann 2025 (S3) — genome-scale CRISPRi Perturb-seq in primary CD4+ T cells:
    GWCD4i.pseudobulk_merged.h5ad   Pseudobulk UMI counts (44.6 GB)
    → Run 07_preprocess_cd4t.py after downloading.
    → Best downloaded on Sherlock (fast internet, large scratch storage).

Usage:
    # Download Replogle datasets only (fast, ready to use)
    python scripts/00_download_data.py --replogle

    # Download X-Atlas datasets only (large single-cell files)
    python scripts/00_download_data.py --xatlas

    # Download GSE291147 melanoma Perturb-seq data
    python scripts/00_download_data.py --geo

    # Download CD4+ T cell Perturb-seq pseudobulk (44.6 GB — use on Sherlock)
    python scripts/00_download_data.py --cd4t

    # Download everything
    python scripts/00_download_data.py --all
"""

import argparse
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK_SIZE = 1 << 20  # 1 MB chunks

REPLOGLE_ARTICLE_ID = 20029387
REPLOGLE_TARGETS = [
    "K562_gwps_normalized_bulk_01.h5ad",
    "K562_essential_raw_bulk_01.h5ad",   # no normalized_bulk exists; we z-score in preprocessing
    "rpe1_normalized_bulk_01.h5ad",
]

XATLAS_ARTICLE_ID = 29190726
XATLAS_TARGETS = [
    "HCT116_filtered_dual_guide_cells.h5ad",
    "HEK293T_filtered_dual_guide_cells.h5ad",
]

# GEO accession GSE291147 — CRISPRi Perturb-seq in melanoma cells (Nature 2026)
GEO_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE291nnn/GSE291147/suppl/"
)
GEO_TARGETS = [
    "GSE291147_Dual_omics_RNA_gene_count_matrix.RDS",
    "GSE291147_Dual_omics_RNA_cell_metadata.csv.gz",
    "GSE291147_Dual_omics_sgRNA_count_matrix.RDS",
]

# Marson/Dann 2025 — genome-scale CRISPRi Perturb-seq in primary CD4+ T cells
CD4T_S3_URL = (
    "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com"
    "/marson2025_data/GWCD4i.pseudobulk_merged.h5ad"
)
CD4T_FILENAME = "GWCD4i.pseudobulk_merged.h5ad"


# ── Helpers ───────────────────────────────────────────────────────────────────
def list_figshare_files(article_id: int) -> list[dict]:
    url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def stream_download(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=3600)  # 1-hour timeout for large files
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=dest.name, ncols=80,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            fh.write(chunk)
            bar.update(len(chunk))


def download_geo_files(base_url: str, targets: list[str]) -> None:
    """Download files directly from GEO FTP via HTTPS."""
    for name in targets:
        dest = DATA_DIR / name
        if dest.exists():
            print(f"[skip] {name} already exists.")
            continue
        url = base_url + name
        print(f"Downloading {name} ...")
        try:
            stream_download(url, dest)
            print(f"  Saved to {dest}\n")
        except requests.HTTPError as e:
            print(f"  ERROR: {e}")
            print(f"  Try manually: wget '{url}' -O '{dest}'")


def download_from_article(article_id: int, targets: list[str]) -> None:
    print(f"Querying Figshare article {article_id} ...")
    try:
        files = list_figshare_files(article_id)
    except requests.HTTPError as e:
        print(f"ERROR fetching file list: {e}")
        sys.exit(1)

    available = {f["name"]: f for f in files}
    print("\nAvailable files in article:")
    for f in files:
        print(f"  {f['name']:<60s}  {f['size'] / 1e9:.2f} GB")
    print()

    for target_name in targets:
        dest = DATA_DIR / target_name
        if dest.exists():
            print(f"[skip] {target_name} already exists.")
            continue
        if target_name not in available:
            print(f"[warn] '{target_name}' not found in article — skipping.")
            continue
        f = available[target_name]
        print(f"Downloading {target_name} ({f['size'] / 1e9:.2f} GB) ...")
        stream_download(f["download_url"], dest)
        print(f"  Saved to {dest}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--replogle", action="store_true",
                       help="Download Replogle 2022 pseudobulk files (fast, ready to use)")
    group.add_argument("--xatlas", action="store_true",
                       help="Download X-Atlas/Orion single-cell files (large, needs preprocessing)")
    group.add_argument("--geo", action="store_true",
                       help="Download GSE291147 melanoma CRISPRi Perturb-seq files from GEO")
    group.add_argument("--cd4t", action="store_true",
                       help="Download CD4+ T cell pseudobulk h5ad from S3 (44.6 GB — use on Sherlock)")
    group.add_argument("--all", action="store_true",
                       help="Download all files from all sources")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.replogle or args.all:
        print("=== Replogle 2022 ===")
        download_from_article(REPLOGLE_ARTICLE_ID, REPLOGLE_TARGETS)

    if args.xatlas or args.all:
        print("=== X-Atlas/Orion 2025 ===")
        download_from_article(XATLAS_ARTICLE_ID, XATLAS_TARGETS)
        print("NOTE: Run 'python scripts/05_preprocess_xatlas.py' next to convert to pseudobulk.")

    if args.geo or args.all:
        print("=== GSE291147 (melanoma CRISPRi Perturb-seq, Nature 2026) ===")
        download_geo_files(GEO_BASE_URL, GEO_TARGETS)
        print("NOTE: Convert RDS files first, then preprocess:")
        print("      conda install -c conda-forge r-base r-matrix")
        print("      Rscript scripts/06_convert_rds.R")
        print("      python scripts/06_preprocess_gse291147.py")

    if args.cd4t or args.all:
        print("=== CD4+ T cell Perturb-seq (Marson/Dann 2025) ===")
        dest = DATA_DIR / CD4T_FILENAME
        if dest.exists():
            print(f"[skip] {CD4T_FILENAME} already exists.")
        else:
            print(f"Downloading {CD4T_FILENAME} (44.6 GB) ...")
            print("  Tip: run this on Sherlock for faster download speeds.")
            stream_download(CD4T_S3_URL, dest)
            print(f"  Saved to {dest}")
        print("NOTE: Run 'python scripts/07_preprocess_cd4t.py' next to produce per-condition h5ad files.")

    print("Done.")


if __name__ == "__main__":
    main()
