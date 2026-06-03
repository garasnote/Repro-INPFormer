"""
Download MVTec-AD from HuggingFace (Voxel51/mvtec-ad) and reorganize
into original folder structure expected by INP-Former.

Usage:
    pip install huggingface_hub
    huggingface-cli login  # if needed
    python download_mvtec_hf.py --output ./data/mvtec_anomaly_detection

Target structure:
    mvtec_anomaly_detection/
      bottle/
        train/good/001.png
        test/broken_large/001.png
        ground_truth/broken_large/001_mask.png
      cable/
        ...
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/mvtec_anomaly_detection")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache dir (default: ~/.cache/huggingface)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(">>> Downloading Voxel51/mvtec-ad from HuggingFace...")
    repo_path = Path(snapshot_download(
        "Voxel51/mvtec-ad",
        repo_type="dataset",
        cache_dir=args.cache_dir,
    ))
    print(f">>> Downloaded to: {repo_path}")

    samples_json = repo_path / "samples.json"
    if not samples_json.exists():
        raise FileNotFoundError(f"samples.json not found in {repo_path}")

    with open(samples_json) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", data.get("data", []))

    copied = 0
    skipped = 0

    for sample in samples:
        category = sample["category"]["label"]
        defect = sample["defect"]["label"]
        split = sample["split"]
        src_img = repo_path / sample["filepath"]

        if not src_img.exists():
            skipped += 1
            continue

        ext = src_img.suffix
        img_stem = src_img.stem

        if split == "train":
            dst_dir = output / category / "train" / "good"
        else:
            dst_dir = output / category / "test" / defect

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_img = dst_dir / f"{img_stem}{ext}"

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Copy ground truth mask if present
        mask_info = sample.get("defect_mask")
        if mask_info and mask_info.get("mask_path"):
            src_mask = repo_path / mask_info["mask_path"]
            if src_mask.exists():
                gt_dir = output / category / "ground_truth" / defect
                gt_dir.mkdir(parents=True, exist_ok=True)
                dst_mask = gt_dir / f"{img_stem}_mask{src_mask.suffix}"
                if not dst_mask.exists():
                    shutil.copy2(src_mask, dst_mask)

        copied += 1

    print(f">>> Reorganized {copied} samples into {output}")
    if skipped:
        print(f">>> Skipped {skipped} (missing source files)")

    # Verify structure
    categories = sorted([d.name for d in output.iterdir() if d.is_dir()])
    print(f">>> Categories ({len(categories)}): {', '.join(categories)}")
    for cat in categories:
        train_count = len(list((output / cat / "train" / "good").glob("*"))) if (output / cat / "train" / "good").exists() else 0
        test_dirs = [d.name for d in (output / cat / "test").iterdir() if d.is_dir()] if (output / cat / "test").exists() else []
        print(f"    {cat}: {train_count} train, test defects: {test_dirs}")


if __name__ == "__main__":
    main()
