"""
00_download_data.py
-------------------
Downloads the Replogle et al. 2022 K562 genome-wide Perturb-seq
normalized pseudobulk dataset from Figshare (article 20029387).

File downloaded:
    K562_gwps_normalized_bulk_01.h5ad
    - One row per perturbation (pseudobulk-aggregated)
    - Expression values are gemgroup z-normalised
    - ~10,000 CRISPRi perturbations + non-targeting controls

Usage:
    python scripts/00_download_data.py
"""

import sys
import requests
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
FIGSHARE_ARTICLE_ID = 20029387
TARGET_FILENAME = "K562_gwps_normalized_bulk_01.h5ad"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK_SIZE = 1 << 20  # 1 MB chunks


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


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / TARGET_FILENAME

    if dest.exists():
        print(f"[skip] {TARGET_FILENAME} already exists at:\n  {dest}")
        return

    print(f"Querying Figshare article {FIGSHARE_ARTICLE_ID} ...")
    try:
        files = list_figshare_files(FIGSHARE_ARTICLE_ID)
    except requests.HTTPError as e:
        print(f"ERROR fetching file list: {e}")
        sys.exit(1)

    print("\nAvailable files in article:")
    for f in files:
        print(f"  {f['name']:<55s}  {f['size'] / 1e9:.2f} GB")

    target = next((f for f in files if f["name"] == TARGET_FILENAME), None)
    if target is None:
        print(f"\nERROR: '{TARGET_FILENAME}' not found in the article.")
        print("Check the filename against the list above and update TARGET_FILENAME.")
        sys.exit(1)

    size_gb = target["size"] / 1e9
    print(f"\nDownloading {TARGET_FILENAME} ({size_gb:.2f} GB) ...")
    print(f"Destination: {dest}\n")

    stream_download(target["download_url"], dest)
    print(f"\nDownload complete. Saved to:\n  {dest}")


if __name__ == "__main__":
    main()
