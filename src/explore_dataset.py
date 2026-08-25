"""Quick dataset inspection for VisionGuard.

Run after downloading MVTec AD and placing it under data/mvtec_ad/:
    python src/explore_dataset.py
"""

from pathlib import Path


ROOT = Path("data/mvtec_ad")


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.rglob("*.png") if p.is_file())


def main() -> None:
    if not ROOT.exists():
        print(f"Dataset directory not found: {ROOT}")
        print("Download MVTec AD and extract it to data/mvtec_ad/")
        return

    categories = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    print(f"Categories found: {len(categories)}")

    for category in categories:
        category_dir = ROOT / category
        train_count = count_images(category_dir / "train")
        test_count = count_images(category_dir / "test")
        print(f"{category:20s} train={train_count:4d} test={test_count:4d}")


if __name__ == "__main__":
    main()
