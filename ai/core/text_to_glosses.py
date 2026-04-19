"""
Text-to-Sign translator — converts English text into a playable ISL gloss sequence.

Two-step Gemini pipeline:
  1. TEXT_TO_GLOSSES_PROMPT   — English → vocab-constrained glosses + skipped + mappings
  2. TRANSLATION_PREVIEW_PROMPT — that JSON → one-line user-facing preview

Both steps fall back to deterministic local behavior if the Gemini API is
unavailable, so the feature still works (with reduced quality) without a key.
"""

import json
import logging
import re

from decouple import config as env_config

from .constants import (
    GEMINI_MODEL,
    TEXT_TO_GLOSSES_PROMPT,
    TRANSLATION_PREVIEW_PROMPT,
)
from .video_transcode import get_playable_mp4_url
from .vocab_index import get_vocab

logger = logging.getLogger(__name__)

_GEMINI_API_KEY = env_config("GEMINI_API_KEY", default="")

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _gemini_call(prompt):
    """Send a prompt to Gemini and return the raw text, or None on failure."""
    if not _GEMINI_API_KEY:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=_GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text or None
    except ImportError:
        logger.warning("google-genai not installed. pip install -U google-genai")
        return None
    except Exception as e:
        logger.warning("Gemini API error: %s", e)
        return None


def _parse_glosses_json(raw, vocab_set):
    """Strip fences, parse JSON, validate against vocab. Returns dict or None."""
    if not raw:
        return None
    cleaned = _JSON_FENCE.sub("", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Gemini returned non-JSON: %s | raw=%r", e, raw[:200])
        return None

    glosses = [g.lower() for g in data.get("glosses", []) if isinstance(g, str)]
    skipped = [s for s in data.get("skipped", []) if isinstance(s, str)]
    mappings = {
        k: v for k, v in (data.get("mappings") or {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }

    # Defensive filter: drop glosses Gemini hallucinated outside the vocab.
    valid = [g for g in glosses if g in vocab_set]
    dropped = [g for g in glosses if g not in vocab_set]
    if dropped:
        logger.info("Dropped out-of-vocab glosses from Gemini: %s", dropped)
        skipped.extend(dropped)

    return {"glosses": valid, "skipped": skipped, "mappings": mappings}


def _local_fallback_glosses(text, vocab_set):
    """No-Gemini fallback: token-match user text against vocab. No reorder, no synonyms."""
    tokens = re.findall(r"[a-zA-Z_]+", text.lower())
    glosses, skipped = [], []
    for tok in tokens:
        if tok in vocab_set:
            glosses.append(tok)
        else:
            skipped.append(tok)
    return {"glosses": glosses, "skipped": skipped, "mappings": {}}


def _local_preview(result):
    """No-Gemini preview string."""
    glosses = result.get("glosses") or []
    skipped = result.get("skipped") or []
    if not glosses and skipped:
        return "No words from your sentence are in the ISL vocabulary."
    if not glosses:
        return "Nothing to sign."
    body = f"Signing: {' '.join(glosses)}."
    if skipped:
        body += f" Skipped (not in vocabulary): {', '.join(skipped)}."
    return body


def translate(text):
    """
    Translate English text into a playable ISL sequence.

    Returns a dict:
      {
        "glosses":  [str],            # ordered ISL words (SOV), all in vocab
        "skipped":  [str],            # original-text words with no mapping
        "mappings": {str: str},       # transparency: synonym → gloss
        "videos":   [{gloss, url}],   # ready-to-play queue for the browser
        "preview":  str,              # one-line UI summary
      }
    """
    glosses_list, _ = get_vocab()
    vocab_set = set(glosses_list)

    if not text or not text.strip():
        return {
            "glosses": [], "skipped": [], "mappings": {},
            "videos": [], "preview": "Enter text to translate.",
        }

    # Step 1 — text → glosses
    prompt1 = TEXT_TO_GLOSSES_PROMPT.format(
        vocabulary=json.dumps(glosses_list),
        text=text.strip(),
    )
    raw1 = _gemini_call(prompt1)
    parsed = _parse_glosses_json(raw1, vocab_set) if raw1 else None
    if parsed is None:
        logger.info("Falling back to local token-match for: %r", text[:80])
        parsed = _local_fallback_glosses(text, vocab_set)

    # Resolve each gloss to a playable MP4 URL (transcodes on first hit).
    videos = []
    unplayable = []
    for g in parsed["glosses"]:
        url = get_playable_mp4_url(g)
        if url:
            videos.append({"gloss": g, "url": url})
        else:
            unplayable.append(g)
    if unplayable:
        parsed["skipped"].extend(unplayable)

    # Step 2 — preview
    prompt2 = TRANSLATION_PREVIEW_PROMPT.format(result=json.dumps(parsed))
    preview = _gemini_call(prompt2) or _local_preview(parsed)

    return {**parsed, "videos": videos, "preview": preview}
