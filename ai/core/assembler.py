"""
Sentence Assembler — converts ISL glosses to English via Gemini API.

Falls back to rule-based assembly if Gemini is unavailable.
"""

import json
import logging

from decouple import config as env_config

from .constants import (
    GEMINI_MODEL,
    SENTENCE_ASSEMBLY_PROMPT,
)

logger = logging.getLogger(__name__)

# Read once at import time from .env / environment
_GEMINI_API_KEY = env_config("GEMINI_API_KEY", default="")


def _try_gemini(glosses):
    """Call Google Gemini API for sentence assembly."""
    if not _GEMINI_API_KEY:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=_GEMINI_API_KEY)
        prompt = SENTENCE_ASSEMBLY_PROMPT.format(glosses=json.dumps(glosses))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        sentence = response.text.strip()
        return sentence if sentence else None
    except ImportError:
        logger.warning("google-genai not installed. pip install -U google-genai")
        return None
    except Exception as e:
        logger.warning("Gemini API error: %s", e)
        return None


def _simple_assembly(glosses):
    """Rule-based fallback when Gemini is unavailable."""
    if not glosses:
        return ""

    words = [g.lower().replace("_", " ") for g in glosses]

    question_words = {"what", "where", "when", "why", "how", "who", "which"}
    is_question = words[-1] in question_words if words else False

    if is_question:
        q_word = words.pop()
        words.insert(0, q_word)

    sentence = " ".join(words).capitalize()
    return sentence + ("?" if is_question else ".")


def assemble_sentence(glosses_list):
    """
    Convert ISL glosses into a natural English sentence.

    Tries Gemini API first, falls back to rule-based.
    """
    if not glosses_list:
        return ""

    result = _try_gemini(glosses_list)
    if result:
        return result

    return _simple_assembly(glosses_list)
