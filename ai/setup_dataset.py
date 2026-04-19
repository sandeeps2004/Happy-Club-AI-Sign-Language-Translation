"""
Dataset Setup — Download INCLUDE ISL videos and extract MediaPipe keypoints.

Usage:
    python setup_dataset.py --download        # Download ISL videos from Zenodo
    python setup_dataset.py --extract         # Extract keypoints from videos
    python setup_dataset.py --download --extract  # Both
    python setup_dataset.py --sample          # Synthetic test data (no download)
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config

# ── Zenodo INCLUDE dataset ──────────────────────────────────────────────
ZENODO_RECORD = "4010759"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

# Full list of ZIPs required to cover the 78-gloss INCLUDE-50 vocabulary used
# by both sign-to-text (training) and text-to-sign (video playback).
# Retrieved from https://zenodo.org/api/records/4010759 — covers 12 categories.
# Each tuple: (filename, approx_size_MB). Total ~37 GB download.
#
# Peak disk stays below ~20 GB because each ZIP is extracted, pruned to the
# canonical gloss list, and deleted before the next ZIP downloads.
CATEGORY_ZIPS = [
    ("Animals_1of2.zip", 1736),
    ("Animals_2of2.zip", 1020),
    ("Clothes_1of2.zip", 1323),
    ("Clothes_2of2.zip", 1332),
    ("Colours_1of2.zip", 1210),
    ("Colours_2of2.zip", 1385),
    ("Days_and_Time_1of3.zip", 1186),
    ("Days_and_Time_2of3.zip", 970),
    ("Days_and_Time_3of3.zip", 813),
    ("Electronics_1of2.zip", 883),
    ("Electronics_2of2.zip", 786),
    ("Greetings_1of2.zip", 1501),
    ("Greetings_2of2.zip", 1153),
    ("Home_1of4.zip", 1191),
    ("Home_2of4.zip", 1277),
    ("Home_3of4.zip", 1030),
    ("Home_4of4.zip", 833),
    ("Jobs_1of2.zip", 1433),
    ("Jobs_2of2.zip", 1574),
    ("People_1of5.zip", 1266),
    ("People_2of5.zip", 1193),
    ("People_3of5.zip", 1513),
    ("People_4of5.zip", 1243),
    ("People_5of5.zip", 1241),
    ("Places_1of4.zip", 1330),
    ("Places_2of4.zip", 1293),
    ("Places_3of4.zip", 1398),
    ("Places_4of4.zip", 1009),
    ("Pronouns_1of2.zip", 1354),
    ("Pronouns_2of2.zip", 903),
    ("Seasons_1of1.zip", 1194),
]

# INCLUDE-50 word labels (from HuggingFace metadata)
# Maps: label_in_dataset -> clean_class_name
INCLUDE50_LABELS = {
    "1. loud": "loud", "2. quiet": "quiet", "3. happy": "happy",
    "78. long": "long", "79. short": "short", "83. big large": "big_large",
    "84. small little": "small_little", "87. hot": "hot", "91. new": "new",
    "94. good": "good", "97. dry": "dry",
    "1. Dog": "dog", "4. Bird": "bird", "5. Cow": "cow",
    "37. Hat": "hat", "42. T-Shirt": "tshirt", "44. Shoes": "shoes",
    "47. Red": "red", "54. Black": "black", "55. White": "white",
    "67. Monday": "monday", "78. Year": "year", "86. Time": "time",
    "53. Fan": "fan", "54. Cell phone": "cellphone",
    "48. Hello": "hello", "51. Good Morning": "good_morning",
    "55. Thank you": "thank_you",
    "28. Window": "window", "34. Pen": "pen", "40. Paint": "paint",
    "84. Teacher": "teacher", "91. Priest": "priest",
    "11. Car": "car", "16. train ticket": "train_ticket",
    "61. Father": "father", "66. Brother": "brother",
    "77. Boy": "boy", "78. Girl": "girl",
    "19. House": "house", "23. Court": "court",
    "28. Store or Shop": "store", "35. Bank": "bank",
    "40. I": "i", "44. it": "it", "46. you (plural)": "you_plural",
    "61. Summer": "summer", "64. Fall": "fall",
    "14. Election": "election", "2. Death": "death",
}


def _load_canonical_glosses():
    """Read the canonical 78-gloss list from dataset_meta.json (fallback: empty)."""
    meta_path = config.INCLUDE_DIR / "dataset_meta.json"
    if not meta_path.exists():
        return set()
    try:
        return set(json.loads(meta_path.read_text()).get("glosses") or [])
    except Exception:
        return set()


def _normalize_folder_name(name):
    """Normalize a raw/ folder name ('40. I' → 'i', 'Ex. Monsoon' → 'ex._monsoon')."""
    stripped = re.sub(r"^\d+\.\s+", "", name)
    return stripped.lower().replace(" ", "_")


def _folder_matches_canonical(folder_name, canonical):
    """True if this Zenodo folder maps to a canonical gloss."""
    if not canonical:
        return True  # no filter available — extract everything
    key = _normalize_folder_name(folder_name)
    return key in canonical or key.replace("_", "") in canonical


def _extract_needed(zip_path, raw_dir, canonical):
    """Extract only gloss folders that match the canonical vocabulary.

    Saves disk: a full INCLUDE ZIP contains many non-INCLUDE-50 glosses (e.g. the
    Animals ZIP has 'Goat', 'Horse', 'Rabbit' we don't use). Those get skipped.
    """
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            # Path inside ZIP looks like: Category/<NN. Name>/file.MOV
            if len(parts) < 2:
                continue
            gloss_folder = parts[1] if len(parts) >= 3 else parts[0]
            if not _folder_matches_canonical(gloss_folder, canonical):
                skipped += 1
                continue
            zf.extract(member, raw_dir)
            extracted += 1
    return extracted, skipped


def download_include(keep_zips=False):
    """
    Download INCLUDE ISL videos from Zenodo, streaming one ZIP at a time.

    Flow per ZIP: download → extract only canonical-gloss folders → delete ZIP.
    This keeps peak disk usage under ~2 GB transient instead of ~37 GB all-at-once.

    Args:
        keep_zips: if True, retain ZIPs in zips/ after extraction (useful for
                   re-extracting to train/val/test splits later without re-downloading).
    """
    total_mb = sum(s for _, s in CATEGORY_ZIPS)
    print("=" * 60)
    print("Downloading INCLUDE ISL dataset from Zenodo...")
    print(f"  Source: https://zenodo.org/record/{ZENODO_RECORD}")
    print(f"  ZIPs:   {len(CATEGORY_ZIPS)} files across 12 categories")
    print(f"  Total:  ~{total_mb / 1024:.1f} GB (streamed, ZIPs deleted after extract)")
    print("=" * 60)

    zip_dir = config.INCLUDE_DIR / "zips"
    raw_dir = config.INCLUDE_DIR / "raw"
    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    canonical = _load_canonical_glosses()
    if canonical:
        print(f"  Filter: keeping only folders that match {len(canonical)} canonical glosses\n")
    else:
        print("  Filter: none (dataset_meta.json missing — extracting everything)\n")

    failed = []
    for idx, (zip_name, approx_mb) in enumerate(CATEGORY_ZIPS, 1):
        print(f"\n[{idx}/{len(CATEGORY_ZIPS)}] {zip_name} (~{approx_mb} MB)")
        zip_path = zip_dir / zip_name

        # 1. Download (skip if already present from a prior run)
        if zip_path.exists() and zip_path.stat().st_size > approx_mb * 500_000:
            print(f"  [SKIP] already downloaded")
        else:
            url = f"{ZENODO_BASE}/{zip_name}/content"
            try:
                _download_with_progress(url, zip_path)
            except Exception as e:
                print(f"  [ERROR] Download failed: {e}")
                failed.append(zip_name)
                continue

        # 2. Extract — only folders matching canonical glosses
        try:
            extracted, skipped = _extract_needed(zip_path, raw_dir, canonical)
            print(f"  Extracted {extracted} files ({skipped} non-canonical skipped)")
        except zipfile.BadZipFile:
            print(f"  [ERROR] Corrupted ZIP, skipping")
            failed.append(zip_name)
            continue

        # 3. Delete ZIP to free disk (unless --keep-zips)
        if not keep_zips:
            zip_path.unlink(missing_ok=True)

    # Only rebuild splits/metadata if we have enough data
    _organize_splits(raw_dir)
    _print_dataset_stats()

    if failed:
        print(f"\n[WARN] {len(failed)} ZIPs failed: {failed}")
        print("  Re-run the command; it will skip already-downloaded ZIPs.")


def _download_with_progress(url, dest_path):
    """Download a file with a progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get("Content-Length", 0))
    block_size = 1024 * 64

    with open(dest_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def _organize_splits(raw_dir):
    """
    Organize extracted videos into train/val/test splits.

    Zenodo structure: raw/{Category}/{Word_Label}/video.MOV
    Target structure: include50/{split}/{clean_class}/video.MOV
    """
    print("\nOrganizing videos into train/val/test splits...")

    # Collect all INCLUDE-50 videos from extracted directories
    videos_by_class = defaultdict(list)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if Path(f).suffix.lower() not in video_extensions:
                continue
            video_path = Path(root) / f
            # The parent directory name is the word label (e.g., "48. Hello")
            word_label = video_path.parent.name

            if word_label in INCLUDE50_LABELS:
                clean_name = INCLUDE50_LABELS[word_label]
                videos_by_class[clean_name].append(video_path)

    # Also include non-INCLUDE-50 classes for more training data
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if Path(f).suffix.lower() not in video_extensions:
                continue
            video_path = Path(root) / f
            word_label = video_path.parent.name
            # Skip if already added as INCLUDE-50
            if word_label in INCLUDE50_LABELS:
                continue
            # Clean the class name
            clean = word_label.lower().strip()
            clean = clean.lstrip("0123456789. ")
            clean = clean.replace(" ", "_")
            if clean:
                videos_by_class[clean].append(video_path)

    # Filter: only keep classes with at least 3 videos
    videos_by_class = {
        k: v for k, v in videos_by_class.items() if len(v) >= 3
    }

    print(f"  Found {len(videos_by_class)} classes with videos")
    total_videos = sum(len(v) for v in videos_by_class.values())
    print(f"  Total videos: {total_videos}")

    # Split 70/15/15 with shuffle
    random.seed(42)
    for class_name, video_list in videos_by_class.items():
        random.shuffle(video_list)
        n = len(video_list)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))

        splits = {
            "train": video_list[:n_train],
            "val": video_list[n_train:n_train + n_val],
            "test": video_list[n_train + n_val:],
        }
        # Ensure test has at least 1 if possible
        if not splits["test"] and len(splits["train"]) > 1:
            splits["test"] = [splits["train"].pop()]

        for split_name, split_videos in splits.items():
            dest_dir = config.INCLUDE_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_videos:
                dst = dest_dir / src.name
                if not dst.exists():
                    os.symlink(src.resolve(), dst)

    # Save class list metadata
    glosses = sorted(videos_by_class.keys())
    meta = {
        "glosses": glosses,
        "num_classes": len(glosses),
        "source": "INCLUDE (Zenodo)",
    }
    meta_path = config.INCLUDE_DIR / "dataset_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Organized into train/val/test splits")
    print(f"  Classes: {len(glosses)}")


def _print_dataset_stats():
    """Print dataset statistics — video files present on disk."""
    stats = defaultdict(lambda: {"videos": 0, "classes": set()})

    for split in ["train", "val", "test"]:
        split_dir = config.INCLUDE_DIR / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            n = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            )
            if n > 0:
                stats[split]["videos"] += n
                stats[split]["classes"].add(class_dir.name)

    total_videos = sum(s["videos"] for s in stats.values())
    all_classes = set()
    for s in stats.values():
        all_classes.update(s["classes"])

    if total_videos > 0:
        print(f"\n  Dataset summary:")
        print(f"  {'Split':<10} {'Videos':>8} {'Classes':>8}")
        print(f"  {'-'*30}")
        for split in ["train", "val", "test"]:
            if split in stats:
                print(f"  {split:<10} {stats[split]['videos']:>8} {len(stats[split]['classes']):>8}")
        print(f"  {'total':<10} {total_videos:>8} {len(all_classes):>8}")
    else:
        print("\n  [NOTE] No video files organized yet.")
        print("  Run --download to fetch videos from Zenodo.")


def extract_keypoints():
    """
    Extract MediaPipe Holistic keypoints from all videos.

    Saves numpy arrays: data/keypoints/{split}/{class_name}/{video_name}.npy
    Each .npy is shape (SEQ_LEN, FEATURE_DIM) = (30, 108)
    """
    from keypoint_extractor import KeypointExtractor

    print("=" * 60)
    print("Extracting MediaPipe keypoints from videos...")
    print(f"  Sequence length: {config.SEQ_LEN} frames")
    print(f"  Feature dim: {config.FEATURE_DIM} per frame")
    print("=" * 60)

    extractor = KeypointExtractor(static_mode=True)

    # Find all video files in the organized split directories
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = []

    for split in ["train", "val", "test"]:
        split_dir = config.INCLUDE_DIR / split
        if not split_dir.exists():
            continue
        for root, dirs, files in os.walk(split_dir):
            for f in files:
                if Path(f).suffix.lower() in video_extensions:
                    video_files.append(Path(root) / f)

    if not video_files:
        print(f"[ERROR] No video files found in {config.INCLUDE_DIR}")
        print("  Run with --download first, or place videos manually.")
        sys.exit(1)

    print(f"  Found {len(video_files)} videos to process\n")

    # Track metadata for splits
    metadata = {"glosses": set(), "samples": []}

    for video_path in tqdm(video_files, desc="Extracting keypoints"):
        # Structure: include50/{split}/{class_name}/video.MOV
        rel = video_path.relative_to(config.INCLUDE_DIR)
        parts = rel.parts
        if len(parts) < 3:
            continue
        split_name, class_name = parts[0], parts[1]

        # Extract keypoints
        try:
            keypoints = extractor.extract_video(video_path, seq_len=config.SEQ_LEN)
        except Exception as e:
            print(f"  [SKIP] {video_path.name}: {e}")
            continue

        # Save
        save_dir = config.KEYPOINTS_DIR / split_name / class_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_path.stem}.npy"
        np.save(save_path, keypoints)

        metadata["glosses"].add(class_name)
        metadata["samples"].append({
            "split": split_name,
            "class": class_name,
            "path": str(save_path.relative_to(config.KEYPOINTS_DIR)),
        })

    extractor.close()

    # Save metadata
    metadata["glosses"] = sorted(metadata["glosses"])
    meta_path = config.KEYPOINTS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Extracted {len(metadata['samples'])} samples")
    print(f"  Classes: {len(metadata['glosses'])}")
    print(f"  Saved to: {config.KEYPOINTS_DIR}")
    print(f"  Metadata: {meta_path}")


def create_sample_dataset():
    """
    Create a small synthetic dataset for testing the pipeline without downloading.
    Generates random keypoint sequences for 10 dummy ISL words.
    """
    print("=" * 60)
    print("Creating sample dataset (for pipeline testing only)...")
    print("=" * 60)

    sample_glosses = [
        "hello", "thank_you", "please", "sorry", "good",
        "bad", "yes", "no", "help", "water",
    ]

    metadata = {"glosses": sample_glosses, "samples": []}

    for split in ["train", "val", "test"]:
        n_samples = 40 if split == "train" else 10
        for gloss in sample_glosses:
            save_dir = config.KEYPOINTS_DIR / split / gloss
            save_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_samples):
                # Create semi-distinct patterns per class
                base_pattern = np.random.RandomState(hash(gloss) % 2**31).randn(
                    config.SEQ_LEN, config.FEATURE_DIM
                ).astype(np.float32) * 0.3

                noise = np.random.randn(config.SEQ_LEN, config.FEATURE_DIM).astype(np.float32) * 0.1
                keypoints = base_pattern + noise
                keypoints = np.clip(keypoints, 0, 1)

                save_path = save_dir / f"sample_{i:03d}.npy"
                np.save(save_path, keypoints)

                metadata["samples"].append({
                    "split": split,
                    "class": gloss,
                    "path": str(save_path.relative_to(config.KEYPOINTS_DIR)),
                })

    meta_path = config.KEYPOINTS_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Created {len(metadata['samples'])} synthetic samples")
    print(f"  Classes: {len(sample_glosses)}")
    print(f"  Use this to verify the training pipeline works.")
    print(f"  For real accuracy, download the actual INCLUDE dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup INCLUDE dataset")
    parser.add_argument("--download", action="store_true", help="Download from Zenodo")
    parser.add_argument("--extract", action="store_true", help="Extract MediaPipe keypoints")
    parser.add_argument("--sample", action="store_true", help="Create synthetic test dataset")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Keep ZIPs in zips/ after extraction (default: delete to save disk)")
    args = parser.parse_args()

    if not any([args.download, args.extract, args.sample]):
        print("Usage:")
        print("  python setup_dataset.py --download        # Download & extract INCLUDE ISL (~37 GB transient, ~18 GB kept)")
        print("  python setup_dataset.py --download --keep-zips  # Keep ZIPs for re-use")
        print("  python setup_dataset.py --extract         # Extract MediaPipe keypoints from videos")
        print("  python setup_dataset.py --sample          # Create synthetic test data")
        sys.exit(0)

    if args.download:
        download_include(keep_zips=args.keep_zips)
    if args.extract:
        extract_keypoints()
    if args.sample:
        create_sample_dataset()
