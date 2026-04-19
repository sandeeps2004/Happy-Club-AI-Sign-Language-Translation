"""
Text-to-Sign Inference Engine.

Thin integration layer — imports shared pipeline from ai.core.
All translation, vocab, and transcoding logic lives in ai/core/ to avoid
duplication between the standalone ai/ utilities and the Django app.
"""

import sys
from pathlib import Path

_ai_parent = str(Path(__file__).resolve().parent.parent.parent)
if _ai_parent not in sys.path:
    sys.path.insert(0, _ai_parent)

from ai.core import (  # noqa: E402, F401
    get_vocab,
    translate,
)
