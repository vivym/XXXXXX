import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder


def main():
    val_path = Path("/home/ym/datasets/imagenet/val")
    val_dataset = ImageFolder(val_path, loader=lambda path: path)

    predictions_list = []
    for file_path in Path("results").glob("predictions_*.pth"):
        predictions = torch.load(file_path, map_location="cpu")
        predictions_list.append(predictions)
    assert len(predictions_list) == 30

    # N, num_samples, num_classes
    predictions = torch.stack(predictions_list, dim=1)

    # saliency_maps_list = []
    # for file_path in Path("results").glob("saliency_maps_*.pth"):
    #     saliency_maps = torch.load(file_path, map_location="cpu")
    #     saliency_maps = np.concatenate(saliency_maps, axis=0)
    #     saliency_maps_list.append(saliency_maps)

    # # N, num_samples, 1, H, W
    # saliency_maps = np.stack(saliency_maps_list, axis=1)
    # print(saliency_maps.shape)

    for i, (path, label) in enumerate(val_dataset):
        # num_samples, num_classes
        prediction = predictions[i]
        # num_samples, 1, H, W
        saliency_map = []
        for j in range(30):
            saliency_map.append(
                torch.load(
                    f"results/saliency_maps_{j:02d}_{i:08d}.pth",
                    map_location="cpu",
                ),
            )
        saliency_map = np.stack(saliency_map, axis=0)

        with open(f"json_results/{i:08d}.json") as f:
            json.dump({
                "path": path,
                "pred": prediction.argmax(1).tolist(),
                "label": label,
                "softmax": prediction.tolist(),
                "saliency_map": saliency_map.tolist(),
            }, f)


if __name__ == "__main__":
    main()
