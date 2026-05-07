"""
Downloads the OpenDroneMap Caliterra drone dataset.

77 aerial JPEG images with GPS coordinates embedded in EXIF.
Source: https://github.com/OpenDroneMap/odm_data_caliterra
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://github.com/OpenDroneMap/odm_data_caliterra/archive/master.zip"
DATASET_SHA256 = None  # not pinned; checked on first successful download
IMAGES_DIR = Path(__file__).parent.parent / "data" / "caliterra"
ZIP_CACHE = Path(__file__).parent.parent / "data" / "caliterra_raw.zip"


def download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            bar = "#" * (pct // 2)
            print(f"\r  [{bar:<50}] {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def extract_images(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".jpg", ".jpeg")):
                data = zf.read(name)
                dest = out_dir / Path(name).name
                dest.write_bytes(data)
                count += 1
    return count


def main() -> None:
    if IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.jpg")):
        imgs = list(IMAGES_DIR.glob("*.jpg"))
        print(f"Dataset already present: {len(imgs)} images in {IMAGES_DIR}")
        return

    if not ZIP_CACHE.exists():
        try:
            download_with_progress(DATASET_URL, ZIP_CACHE)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Using cached zip: {ZIP_CACHE}")

    print(f"Extracting images to {IMAGES_DIR} ...")
    count = extract_images(ZIP_CACHE, IMAGES_DIR)
    print(f"Extracted {count} images.")

    if count == 0:
        print("ERROR: no images found in archive", file=sys.stderr)
        ZIP_CACHE.unlink(missing_ok=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
