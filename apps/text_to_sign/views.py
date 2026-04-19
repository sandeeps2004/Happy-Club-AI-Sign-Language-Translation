"""Text-to-Sign views: translator page + JSON translate endpoint.

Videos are served by Django's media pipeline — the translate endpoint returns
URLs under MEDIA_URL/text_to_sign_cache/, populated on first hit by
ai/core/video_transcode.py.
"""

import json
import logging

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .inference import get_vocab, translate

logger = logging.getLogger(__name__)


# Category groupings for the vocabulary preview. Only signs actually present
# in the indexed vocab are shown — keeps the UI honest when entries are missing.
_VOCAB_CATEGORIES = [
    ("Pronouns", ["i", "you", "he", "she", "it"]),
    ("Greetings", ["hello", "how_are_you", "alright", "good_morning",
                   "good_afternoon", "good_evening", "good_night", "thank_you"]),
    ("People & Jobs", ["mother", "father", "parent", "son", "daughter",
                       "teacher", "student", "doctor", "patient", "lawyer",
                       "waiter", "secretary", "priest"]),
    ("Places & Home", ["city", "house", "restaurant", "train_station",
                       "street_or_road", "bedroom", "bed", "chair", "table",
                       "door", "window", "dream"]),
    ("Colors", ["red", "green", "blue", "yellow", "orange", "pink", "brown",
                "grey", "black", "white", "colour"]),
    ("Days & Seasons", ["today", "monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday", "spring", "summer",
                        "fall", "winter", "season", "ex._monsoon"]),
    ("Clothing", ["dress", "shirt", "skirt", "suit", "hat"]),
    ("Animals", ["dog", "cat", "cow", "bird", "fish"]),
    ("Electronics", ["cellphone", "computer", "clock", "lamp", "fan"]),
]

# Sample prompts the user can click to try — built only from in-vocab words
# so they're guaranteed to produce video.
_EXAMPLE_PROMPTS = [
    "Hello, how are you",
    "Good morning father",
    "I am a student",
    "Red dress and blue shirt",
    "Today is Monday",
    "Thank you doctor",
]


@login_required
def translator_view(request):
    """Render the text-to-sign translator page."""
    glosses, _ = get_vocab()
    available = set(glosses)
    categories = [
        {"name": name, "items": [w for w in words if w in available]}
        for name, words in _VOCAB_CATEGORIES
    ]
    return render(request, "text_to_sign/translator.html", {
        "vocab_size": len(glosses),
        "vocab_categories": categories,
        "example_prompts": _EXAMPLE_PROMPTS,
    })


@login_required
@require_POST
def translate_endpoint(request):
    """POST {text: str} → {glosses, skipped, mappings, videos, preview}."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return HttpResponseBadRequest("Invalid JSON body")

    text = (payload.get("text") or "").strip()
    if not text:
        return JsonResponse({
            "glosses": [], "skipped": [], "mappings": {},
            "videos": [], "preview": "Enter text to translate.",
        })
    if len(text) > 500:
        return HttpResponseBadRequest("Text too long (max 500 chars)")

    result = translate(text)
    return JsonResponse(result)
