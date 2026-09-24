"""Minimal container, CUDA, dataset, and model smoke test."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset, get_data_transforms
from inference_benchmark import build_model


MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"


def main():
    assert torch.cuda.is_available(), "CUDA is not visible"
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    image_transform, gt_transform = get_data_transforms(224, 196)
    train = ImageFolder(MVTEC_ROOT / "bottle" / "train", transform=image_transform)
    test = MVTecDataset(
        root=MVTEC_ROOT / "bottle",
        transform=image_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    assert len(train) > 0 and len(test) > 0
    image, _ = train[0]
    test[0]
    print(f"MVTec AD loaded: {len(train)} bottle train, {len(test)} bottle test")

    model = build_model("dinov2reg_vit_base_14", 6, "cuda:0")
    with torch.inference_mode():
        encoded, decoded, gather_loss = model(image.unsqueeze(0).cuda())
    assert encoded and decoded and gather_loss.ndim == 0
    print("Model initialized; one 196x196 forward pass succeeded")


if __name__ == "__main__":
    main()
