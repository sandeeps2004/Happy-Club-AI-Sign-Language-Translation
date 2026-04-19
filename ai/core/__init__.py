"""
Shared AI core — single source of truth for inference pipeline.
Used by both ai/ (training/demo) and apps/sign_language/ (Django inference).
"""
from .assembler import assemble_sentence
from .constants import *  # noqa: F401,F403
from .detector import SignDetector
from .extractor import KeypointExtractor
from .model import SignLanguageLSTM, load_model
from .preprocessing import (
    add_velocity_features, compute_hand_velocity,
    normalize_keypoints, normalize_sequence,
)
from .text_to_glosses import translate
from .vocab_index import get_vocab, resolve_source_path

__all__ = [
    "KeypointExtractor", "SignLanguageLSTM", "SignDetector",
    "load_model", "assemble_sentence",
    "normalize_keypoints", "normalize_sequence",
    "add_velocity_features", "compute_hand_velocity",
    "translate", "get_vocab", "resolve_source_path",
]
