"""MVTec AD dataset loader for multi-class cable defect detection."""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class MVTecCableDataset(Dataset):
    """Load the MVTec AD cable category for multi-class classification."""

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        cable_dir = self.root_dir / "cable"
        split_dir = cable_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        class_names = sorted(
            directory.name for directory in split_dir.iterdir() if directory.is_dir()
        )
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}

        for class_name, label in self.class_to_idx.items():
            class_dir = split_dir / class_name
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
