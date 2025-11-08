from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset


def maxpool_2x(arr):
    return arr.reshape(
        arr.shape[0] // 2,
        2,
        arr.shape[1] // 2,
        2,
    ).max(axis=(1, 3))


def load_image(path_jpg, convert):
    img = np.array(PIL.Image.open(path_jpg).convert(convert))

    if True:
        # scale down 2x, because images are too large for my GPU
        if convert == "RGB":
            img = np.array(
                PIL.Image.fromarray(img).resize(
                    (img.shape[1] // 2, img.shape[0] // 2), PIL.Image.Resampling.BOX
                )
            )
        elif convert == "1":
            # max pooling
            img = maxpool_2x(img)

    return img


class CrackSegmentationDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _mask_to_probs(self, mask):
        # convert binary mask (W,H) to (2,W,H)
        probs = torch.zeros((2, mask.shape[0], mask.shape[1]))
        probs[0] = (~mask).float()
        probs[1] = (mask).float()
        return probs

    def __getitem__(self, idx):
        # Item (image: (C,W,H), mask: (W,H))
        image_path, mask_path = self.paths[idx]
        image = (
            torch.tensor(load_image(image_path, "RGB")).permute(2, 0, 1).float() / 255.0
        )  # (C,W,H), 0..1
        mask = torch.tensor(load_image(mask_path, "1"))
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, self._mask_to_probs(mask)


def list_examples(path: Path):
    # dataset structure:
    # images: $path/images/xxx
    # masks:  $path/masks/xxx
    # return tuples (image_path, mask_path)
    images_path = path / "images"
    masks_path = path / "masks"
    examples = []
    for img_file in images_path.iterdir():
        mask_file = masks_path / img_file.name
        if mask_file.exists():
            examples.append((img_file, mask_file))

    return examples
