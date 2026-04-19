"""
On-demand MOV → MP4 transcoding for text-to-sign playback.

The INCLUDE-50 raw videos are Canon DSLR .MOV files. Chrome/Firefox refuse to
decode those containers, so we remux each clip to MP4 on first access:

    ffmpeg -i <clip>.MOV -c copy -movflags +faststart <clip>.mp4

`-c copy` does NOT re-encode — it just rewrites the container (fast, lossless).
`+faststart` moves the `moov` atom to the front so the browser can start
playback before the full file arrives.

Cached outputs live in media/text_to_sign_cache/<gloss>.mp4 and survive across
restarts. If remux fails (e.g., source missing moov atom), we fall back to a
fast H.264 re-encode which handles nearly any malformed input.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from threading import Lock

from django.conf import settings

from .vocab_index import resolve_source_path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(settings.MEDIA_ROOT) / "text_to_sign_cache"
CACHE_URL_PREFIX = f"{settings.MEDIA_URL}text_to_sign_cache/"

_FFMPEG = shutil.which("ffmpeg")
_locks = {}  # per-gloss locks prevent duplicate ffmpeg jobs for the same clip
_dict_lock = Lock()


def _gloss_lock(gloss):
    with _dict_lock:
        if gloss not in _locks:
            _locks[gloss] = Lock()
        return _locks[gloss]


def _run_ffmpeg(args, timeout=60):
    """Run ffmpeg with a hard timeout. Returns True on success."""
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg failed: %s", result.stderr.decode(errors="replace")[-400:])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out: %s", " ".join(args))
        return False
    except Exception as e:
        logger.warning("ffmpeg error: %s", e)
        return False


def _transcode(src, dst):
    """Remux first (fast, lossless). Fall back to re-encode if remux fails."""
    if not _FFMPEG:
        logger.error("ffmpeg not found on PATH — cannot transcode %s", src)
        return False

    tmp = dst.with_suffix(dst.suffix + ".part")
    # Remux attempt: -c copy keeps the same streams, just changes container.
    # -f mp4 forces the format (tmp file has .part extension ffmpeg can't infer from).
    ok = _run_ffmpeg([
        _FFMPEG, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c", "copy",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ])
    if not ok:
        # Re-encode fallback: handles sources with broken moov atom, odd codecs, etc.
        logger.info("Remux failed for %s — re-encoding", src.name)
        ok = _run_ffmpeg([
            _FFMPEG, "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",  # broadest browser compatibility
            "-an",  # drop audio — not relevant for sign clips
            "-movflags", "+faststart",
            "-f", "mp4",
            str(tmp),
        ], timeout=120)

    if ok and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(dst)
        return True

    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False


def get_playable_mp4_url(gloss):
    """Return a media URL for a playable MP4 of this gloss, or None.

    Transcodes on first call (per gloss), then serves from cache.
    Safe to call concurrently — a per-gloss lock serializes ffmpeg jobs.
    """
    gloss = (gloss or "").lower().strip()
    if not gloss:
        return None

    src = resolve_source_path(gloss)
    if src is None:
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / f"{gloss}.mp4"

    if dst.exists() and dst.stat().st_size > 0:
        return f"{CACHE_URL_PREFIX}{gloss}.mp4"

    with _gloss_lock(gloss):
        # Re-check after acquiring lock (another thread may have finished)
        if dst.exists() and dst.stat().st_size > 0:
            return f"{CACHE_URL_PREFIX}{gloss}.mp4"
        if _transcode(src, dst):
            return f"{CACHE_URL_PREFIX}{gloss}.mp4"

    return None
