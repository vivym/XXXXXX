import json
from pathlib import Path

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
    print(predictions.shape)

    saliency_maps_list = []
    for file_path in Path("results/saliency_maps_*.pth"):
        saliency_maps = torch.load(file_path, map_location="cpu")
        saliency_maps_list.append(saliency_maps)

    # N, num_samples, H, W
    saliency_maps = torch.stack(saliency_maps_list, dim=1)

    for i, (path, label) in enumerate(val_dataset):
        # num_samples, num_classes
        prediction = predictions[i]
        # num_samples, H, W
        saliency_map = saliency_maps[i]

        with open("json_results/{i:08d}.json") as f:
            json.dump({
                "path": path,
                "pred": prediction.argmax(1).tolist(),
                "label": label,
                "softmax": prediction.tolist(),
                "saliency_map": saliency_map.tolist(),
            }, f)


if __name__ == "__main__":
    main()
