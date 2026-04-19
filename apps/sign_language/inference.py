"""
Sign Language Inference Engine.

Thin integration layer — imports shared pipeline from ai.core.
All model code, preprocessing, and detection logic lives in ai/core/
to avoid duplication between the standalone demo and Django app.
"""

import sys
from pathlib import Path

# Ensure ai/ is on the import path
_ai_parent = str(Path(__file__).resolve().parent.parent.parent)
if _ai_parent not in sys.path:
    sys.path.insert(0, _ai_parent)

from ai.core import (  # noqa: E402, F401
    CONFIDENCE_THRESHOLD,
    FEATURE_DIM,
    FEATURE_DIM_FINAL,
    SEQ_LEN,
    KeypointExtractor,
    SignDetector,
    SignLanguageLSTM,
    add_velocity_features,
    assemble_sentence,
    compute_hand_velocity,
    normalize_sequence,
)
from ai.core.model import load_model as _load_model  # noqa: E402

# Re-export load_model as get_model for backward compat with consumers.py
get_model = _load_model


def predict_sign(sequence):
    """
    Run inference on a completed sign sequence.

    Args:
        sequence: np.array of shape (SEQ_LEN, FEATURE_DIM) = (30, 108)

    Returns:
        (word, confidence) or (None, 0.0)
    """
    import numpy as np
    import torch

    model, glosses, device = get_model()
    if model is None:
        return None, 0.0

    processed = normalize_sequence(sequence)
    processed = add_velocity_features(processed)

    x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    conf = confidence.item()
    idx = pred_idx.item()

    if conf >= CONFIDENCE_THRESHOLD and idx < len(glosses):
        return glosses[idx], conf
    return None, conf
