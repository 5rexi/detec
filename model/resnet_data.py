"""Dataset utilities for PPE classifier training."""

import os
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class PPEClassificationDataset(Dataset):
    def __init__(self, root: str, class_names: List[str], transform=None):
        self.samples = []
        self.transform = transform

        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, class_id))

        if not self.samples:
            raise RuntimeError(f"No samples found in {root}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        image = Image.open(file_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
