"""
Vocabulary index for text-to-sign.

Scans the INCLUDE-50 RAW videos at ai/data/include50/raw/<Category>/<NN. Name>/
once at first access, normalizes folder names against the canonical 79-gloss
vocabulary (from dataset_meta.json), and caches {gloss: Path} for the process.

The scan is built off raw/ rather than train/ because train/ holds only small
keypoint-only placeholders — the full-size MOVs live in raw/.

Used by:
  - text_to_glosses.py — inlines the gloss list into the Gemini prompt
  - apps/text_to_sign/views.py — resolves a gloss to a playable MP4 URL
"""

import json
import logging
import re
from pathlib import Path
from threading import Lock

from .constants import DATA_DIR

logger = logging.getLogger(__name__)

_RAW_DIR = DATA_DIR / "include50" / "raw"
_META_PATH = DATA_DIR / "include50" / "dataset_meta.json"
_VIDEO_EXTENSIONS = {".mov", ".mp4", ".webm", ".avi"}
_MIN_VIDEO_BYTES = 500_000  # filter out ~4KB corrupted placeholder files

# Strip a "NN. " prefix from folder names ("40. I" → "I", "Ex. Monsoon" stays).
_FOLDER_PREFIX_RE = re.compile(r"^\d+\.\s+")

_lock = Lock()
_cache = None  # tuple[list[str], dict[str, Path]] | None


def _normalize_folder(name):
    """Normalize a raw/ folder name to a canonical gloss key."""
    stripped = _FOLDER_PREFIX_RE.sub("", name)
    return stripped.lower().replace(" ", "_")


def _load_canonical_glosses():
    """Canonical 79-gloss list — source of truth for valid vocab."""
    if not _META_PATH.exists():
        logger.warning("dataset_meta.json missing at %s", _META_PATH)
        return []
    try:
        data = json.loads(_META_PATH.read_text())
        return list(data.get("glosses") or [])
    except Exception as e:
        logger.warning("Failed to read dataset_meta.json: %s", e)
        return []


def _pick_best_clip(gloss_dir):
    """Pick the first video clip in gloss_dir that's large enough to be valid."""
    clips = sorted(
        p for p in gloss_dir.iterdir()
        if p.suffix.lower() in _VIDEO_EXTENSIONS
    )
    for c in clips:
        try:
            if c.stat().st_size >= _MIN_VIDEO_BYTES:
                return c
        except OSError:
            continue
    return None


def _scan():
    """Walk raw/<Category>/<NN. Name>/ and build the gloss → MOV-path map."""
    canonical = set(_load_canonical_glosses())
    if not canonical:
        return [], {}
    if not _RAW_DIR.exists():
        logger.warning("Raw video directory missing: %s", _RAW_DIR)
        return [], {}

    gloss_to_path = {}
    for category in sorted(_RAW_DIR.iterdir()):
        if not category.is_dir():
            continue
        for gloss_dir in sorted(category.iterdir()):
            if not gloss_dir.is_dir():
                continue
            key = _normalize_folder(gloss_dir.name)
            # Handle compact forms like "cell phone" → "cellphone"
            if key not in canonical and key.replace("_", "") in canonical:
                key = key.replace("_", "")
            if key not in canonical or key in gloss_to_path:
                continue
            clip = _pick_best_clip(gloss_dir)
            if clip is not None:
                gloss_to_path[key] = clip

    missing = canonical - set(gloss_to_path.keys())
    if missing:
        logger.info("Glosses without playable video: %s", sorted(missing))
    return sorted(gloss_to_path.keys()), gloss_to_path


def get_vocab():
    """Return (glosses, gloss_to_path) — cached after first call.

    gloss_to_path maps each gloss to the absolute Path of its source MOV.
    Callers should NOT serve these paths directly — run them through
    video_transcode.get_playable_mp4() first.
    """
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _scan()
                logger.info("Vocab indexed: %d glosses", len(_cache[0]))
    return _cache


def resolve_source_path(gloss):
    """Return the absolute source-MOV Path for a gloss, or None if not in vocab."""
    gloss = (gloss or "").lower().strip()
    if not gloss or "/" in gloss or ".." in gloss:
        return None
    _, mapping = get_vocab()
    return mapping.get(gloss)
