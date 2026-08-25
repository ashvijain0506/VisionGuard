"""Dataset utilities for VisionGuard.

MVTec AD is organized as:
    category/train/good/*.png
    category/test/good/*.png
    category/test/<defect_type>/*.png

This module provides a small PyTorch Dataset that converts the directory
structure into binary labels:
    0 = normal
    1 = defective
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class MVTecBinaryDataset(Dataset):
    """Binary normal-vs-defective dataset for one MVTec AD category."""

    def __init__(self, root_dir: str, category: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        split_dir = self.root_dir / category / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        if split == "train":
            self._add_images(split_dir / "good", label=0)
        elif split == "test":
            self._add_images(split_dir / "good", label=0)
            for defect_dir in split_dir.iterdir():
                if defect_dir.is_dir() and defect_dir.name != "good":
                    self._add_images(defect_dir, label=1)
        else:
            raise ValueError("split must be 'train' or 'test'")

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

    def _add_images(self, directory: Path, label: int) -> None:
        if not directory.exists():
            return
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
