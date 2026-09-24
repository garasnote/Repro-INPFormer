"""
Download MVTec-AD from HuggingFace (Voxel51/mvtec-ad) and reorganize
into original folder structure expected by INP-Former.

Usage:
    python scripts/download_mvtec_hf.py

Target structure:
    data/mvtec_ad/
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "data" / ".cache" / "huggingface"
EXPECTED_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "mvtec_ad")
    parser.add_argument("--cache-dir", type=Path,
                        default=DEFAULT_CACHE,
                        help="Hugging Face cache directory")
    args = parser.parse_args()

    output = args.output.resolve()
    staging = output.with_name(f"{output.name}.partial")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    staging.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Downloading Voxel51/mvtec-ad from HuggingFace...")
    repo_path = Path(snapshot_download(
        "Voxel51/mvtec-ad",
        repo_type="dataset",
        cache_dir=str(args.cache_dir),
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
            dst_dir = staging / category / "train" / "good"
        else:
            dst_dir = staging / category / "test" / defect

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_img = dst_dir / f"{img_stem}{ext}"

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Copy ground truth mask if present
        mask_info = sample.get("defect_mask")
        if mask_info and mask_info.get("mask_path"):
            src_mask = repo_path / mask_info["mask_path"]
            if src_mask.exists():
                gt_dir = staging / category / "ground_truth" / defect
                gt_dir.mkdir(parents=True, exist_ok=True)
                dst_mask = gt_dir / f"{img_stem}_mask{src_mask.suffix}"
                if not dst_mask.exists():
                    shutil.copy2(src_mask, dst_mask)
            else:
                skipped += 1

        copied += 1

    print(f">>> Reorganized {copied} samples into {staging}")
    if skipped:
        raise RuntimeError(f"Validation failed: {skipped} source images or masks are missing")

    # Verify structure
    categories = sorted(d.name for d in staging.iterdir() if d.is_dir())
    if set(categories) != EXPECTED_CATEGORIES:
        missing = sorted(EXPECTED_CATEGORIES - set(categories))
        extra = sorted(set(categories) - EXPECTED_CATEGORIES)
        raise RuntimeError(f"Category validation failed; missing={missing}, extra={extra}")
    print(f">>> Categories ({len(categories)}): {', '.join(categories)}")
    for cat in categories:
        train_dir = staging / cat / "train" / "good"
        test_dir = staging / cat / "test"
        gt_dir = staging / cat / "ground_truth"
        if not train_dir.is_dir() or not any(train_dir.iterdir()):
            raise RuntimeError(f"No training images found for {cat}")
        if not test_dir.is_dir() or not any(test_dir.iterdir()):
            raise RuntimeError(f"No test images found for {cat}")
        if not gt_dir.is_dir() or not any(gt_dir.iterdir()):
            raise RuntimeError(f"No ground-truth masks found for {cat}")
        train_count = len(list(train_dir.glob("*")))
        test_dirs = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
        print(f"    {cat}: {train_count} train, test defects: {test_dirs}")

    staging.rename(output)
    if args.cache_dir.resolve() == DEFAULT_CACHE.resolve():
        shutil.rmtree(args.cache_dir)
    print(f">>> Verified MVTec AD dataset: {output}")


if __name__ == "__main__":
    main()
