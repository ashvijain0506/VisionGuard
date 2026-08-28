"""Explore the MVTec AD cable category before model training.

Expected dataset location:
    data/mvtec_ad/cable/

Run:
    python src/explore_cable.py
"""

from collections import Counter
from pathlib import Path


ROOT = Path("data/mvtec_ad/cable")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    if not ROOT.exists():
        print(f"Cable dataset not found at: {ROOT}")
        print("Download MVTec AD and place it under data/mvtec_ad/")
        return

    for split in ["train", "test"]:
        split_dir = ROOT / split
        classes = sorted(
            path.name for path in split_dir.iterdir() if path.is_dir()
        )
        print(f"\n{split.upper()}")
        print("-" * 40)
        total = 0
        counts = Counter()

        for class_name in classes:
            count = count_images(split_dir / class_name)
            counts[class_name] = count
            total += count
            print(f"{class_name:30s} {count:4d} images")

        print(f"Total: {total} images")


if __name__ == "__main__":
    main()
