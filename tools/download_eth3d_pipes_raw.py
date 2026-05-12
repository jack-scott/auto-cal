"""
Download the ETH3D "pipes" raw DSLR dataset (distorted JPEGs).

Source:  https://www.eth3d.net/data/pipes_dslr_jpg.7z
Content: DSLR images of an indoor industrial pipe scene (distorted) + COLMAP calibration.

The archive is expected to extract to:
  data/eth3d_pipes/pipes/images/dslr_images_jpg/          (JPEG images)
  data/eth3d_pipes/pipes/dslr_calibration_jpg/            (COLMAP OPENCV calibration)

Run:
    pixi run pipes-download-raw
"""

import urllib.request
from pathlib import Path

import py7zr

URL = "https://www.eth3d.net/data/pipes_dslr_jpg.7z"
DATA_DIR = Path(__file__).parent.parent / "data"
ARCHIVE = DATA_DIR / "pipes_dslr_jpg.7z"
EXTRACT_DIR = DATA_DIR / "eth3d_pipes"


def download() -> None:
    if ARCHIVE.exists():
        print(f"Archive already exists: {ARCHIVE}")
        return
    print(f"Downloading {URL} ...")
    last_pct = [-1]

    def _progress(count, block_size, total):
        pct = min(100, int(count * block_size * 100 / total))
        if pct != last_pct[0] and pct % 10 == 0:
            print(f"  {pct}%", flush=True)
            last_pct[0] = pct

    urllib.request.urlretrieve(URL, ARCHIVE, reporthook=_progress)
    print(f"Saved to {ARCHIVE}  ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")


def extract() -> None:
    marker = EXTRACT_DIR / "pipes" / "images" / "dslr_images"
    if marker.exists() and any(marker.glob("*.JPG")):
        print(f"Already extracted: {marker}")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {EXTRACT_DIR} ...")
    with py7zr.SevenZipFile(ARCHIVE, mode="r") as z:
        z.extractall(path=EXTRACT_DIR)
    print("Done.")


def verify() -> None:
    img_dir = EXTRACT_DIR / "pipes" / "images" / "dslr_images"
    cal_dir = EXTRACT_DIR / "pipes" / "dslr_calibration_jpg"

    imgs = sorted(img_dir.glob("*.JPG")) if img_dir.exists() else []
    print(f"Images found:      {len(imgs)} in {img_dir}")

    if cal_dir.exists():
        print(f"Calibration found: {cal_dir}")
        for f in sorted(cal_dir.iterdir()):
            print(f"  {f.name}")
    else:
        print(f"WARNING: calibration directory not found at {cal_dir}")
        print("  Check archive contents — may need a separate calibration download.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download()
    extract()
    verify()


if __name__ == "__main__":
    main()
