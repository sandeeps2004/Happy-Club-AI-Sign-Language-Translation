"""
Happy Club Pipeline — Configuration
All paths, model hyperparameters, and constants live here.

Shared constants (paths, model params, MediaPipe, LLM) are defined in
ai/core/constants.py and re-exported here for backward compatibility.
Training-specific and dataset-specific settings are defined below.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `ai.core` is importable
# (needed when running standalone scripts like `cd ai && python demo.py`)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.core.constants import *  # noqa: F401,F403

# ── Training-specific paths ────────────────────────────────────────────────
INCLUDE_DIR = DATA_DIR / "include50"   # noqa: F405
KEYPOINTS_DIR = DATA_DIR / "keypoints"  # noqa: F405

# Created at runtime
for d in [DATA_DIR, INCLUDE_DIR, KEYPOINTS_DIR, CHECKPOINT_DIR]:  # noqa: F405
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────
HF_DATASET_ID = "ai4bharat/INCLUDE"  # HuggingFace dataset
DATASET_VARIANT = "include50"         # 50 most common ISL words
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
PATIENCE = 10               # Early stopping patience
SCHEDULER_FACTOR = 0.5      # LR reduction factor
SCHEDULER_PATIENCE = 5      # LR reduction patience
FRAME_SKIP = 1              # Process every Nth frame (1 = all frames)
