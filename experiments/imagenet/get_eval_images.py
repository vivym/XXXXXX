import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def main():
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )

    val_path = Path("/home/ym/datasets/imagenet/val")
    val_dataset = ImageFolder(val_path, transform=transform)

    for i, (img, label) in tqdm(enumerate(val_dataset)):
        img.save(f"eval_images/{i:08d}.jpeg")


if __name__ == "__main__":
    main()
