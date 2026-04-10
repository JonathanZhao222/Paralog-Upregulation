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

Usage:
    # Download Replogle datasets only (fast, ready to use)
    python scripts/00_download_data.py --replogle

    # Download X-Atlas datasets only (large single-cell files)
    python scripts/00_download_data.py --xatlas

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


# ── Helpers ───────────────────────────────────────────────────────────────────
def list_figshare_files(article_id: int) -> list[dict]:
    url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def stream_download(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc=dest.name, ncols=80,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            fh.write(chunk)
            bar.update(len(chunk))


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
    group.add_argument("--all", action="store_true",
                       help="Download all files from both sources")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.replogle or args.all:
        print("=== Replogle 2022 ===")
        download_from_article(REPLOGLE_ARTICLE_ID, REPLOGLE_TARGETS)

    if args.xatlas or args.all:
        print("=== X-Atlas/Orion 2025 ===")
        download_from_article(XATLAS_ARTICLE_ID, XATLAS_TARGETS)
        print("NOTE: Run 'python scripts/05_preprocess_xatlas.py' next to convert to pseudobulk.")

    print("Done.")


if __name__ == "__main__":
    main()
