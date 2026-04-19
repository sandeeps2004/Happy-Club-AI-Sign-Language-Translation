"""
Shared constants for the ISL inference pipeline.

Single source of truth used by both ai/ (training/demo) and
apps/sign_language/ (Django inference).
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
AI_DIR = Path(__file__).resolve().parent.parent  # ai/
DATA_DIR = AI_DIR / "data"
CHECKPOINT_DIR = AI_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "isl_lstm_best.pt"

# ── Keypoint Layout ────────────────────────────────────────────────────────
# We extract 54 keypoints: 12 upper body + 21 left hand + 21 right hand
# Coordinates are normalized (x, y) in [0, 1] range.
#
# Extracted from RTMPose WholeBody (COCO-WholeBody 133-keypoint format):
#   Upper body: COCO indices [0,1,2,3,4,5,6,7,8,9,10,11]
#     = nose, left_eye, right_eye, left_ear, right_ear,
#       left_shoulder, right_shoulder, left_elbow, right_elbow,
#       left_wrist, right_wrist, left_hip
#   Left hand:  COCO indices 91-111 (21 keypoints)
#   Right hand: COCO indices 112-132 (21 keypoints)
NUM_POSE_LANDMARKS = 12    # upper body subset
NUM_HAND_LANDMARKS = 21    # per hand
NUM_KEYPOINTS = NUM_POSE_LANDMARKS + (NUM_HAND_LANDMARKS * 2)  # 54 total
KEYPOINT_DIM = 2           # x, y coordinates (normalized 0-1)
FEATURE_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 108 features per frame

# Shoulder indices within our 12-landmark pose array (for normalization)
# Index 5 = left_shoulder, Index 6 = right_shoulder
SHOULDER_LEFT_IDX = 5
SHOULDER_RIGHT_IDX = 6

# ── Sequence ───────────────────────────────────────────────────────────────
SEQ_LEN = 30               # Number of frames per sign (padded/truncated)

# ── Feature Engineering ───────────────────────────────────────────────────
USE_VELOCITY = True         # Append frame-to-frame velocity features
# With velocity: FEATURE_DIM_FINAL = 108 (pos) + 108 (vel) = 216
FEATURE_DIM_FINAL = FEATURE_DIM * 2 if USE_VELOCITY else FEATURE_DIM

# ── Model Hyperparameters ─────────────────────────────────────────────────
HIDDEN_SIZE = 128           # LSTM hidden units
NUM_LAYERS = 2              # Stacked LSTM layers
NUM_CLASSES = 79            # INCLUDE dataset — 79 ISL word classes
DROPOUT = 0.3               # Dropout between LSTM layers
BIDIRECTIONAL = True        # Bidirectional LSTM

# ── Sign Boundary Detection ───────────────────────────────────────────────
VELOCITY_THRESHOLD = 0.02   # Hand-only velocity below this = "not signing"
PAUSE_FRAMES = 8            # Consecutive still frames to trigger word boundary
MIN_SIGN_FRAMES = 10        # Minimum frames to consider a valid sign (filters junk micro-segments)
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept a prediction

# ── LLM ───────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3-flash-preview"

SENTENCE_ASSEMBLY_PROMPT = """You are a sign language interpreter assistant.
Given a sequence of Indian Sign Language (ISL) glosses (words recognized from signing),
convert them into a natural, grammatically correct English sentence.

ISL follows Subject-Object-Verb (SOV) order and drops articles/prepositions.
For example:
- Glosses: ["I", "school", "go"] → "I go to school."
- Glosses: ["tomorrow", "rain", "come"] → "It will rain tomorrow."
- Glosses: ["you", "name", "what"] → "What is your name?"
- Glosses: ["hello", "how", "you"] → "Hello, how are you?"

Now convert these glosses into a natural English sentence:
Glosses: {glosses}

Respond with ONLY the English sentence, nothing else."""


# ── Text-to-Sign Prompts ──────────────────────────────────────────────────
# Two-step pipeline: (1) English text → ISL glosses, (2) glosses → preview.

TEXT_TO_GLOSSES_PROMPT = """You are an Indian Sign Language (ISL) translator.
Convert an English sentence into a sequence of ISL glosses for video playback.

You can ONLY use glosses from this fixed vocabulary (case-insensitive, exact match):
{vocabulary}

Rules — apply in order:
1. REORDER to ISL grammar: Subject-Object-Verb (SOV). Move verbs to the end.
   Drop articles (a, an, the), prepositions (to, in, on, at, of), and auxiliaries (is, am, are, was, will, do).
2. MAP each remaining content word to the closest gloss in the vocabulary above:
   - Exact match wins (lowercase, underscores allowed: "thank_you").
   - Otherwise pick a clear synonym from the vocabulary (e.g., "meal" → "food" if "food" exists, "instructor" → "teacher" if it exists).
   - Be strict: only map if the meaning is genuinely close. Do not stretch ("pizza" → "food" is too loose — skip it).
3. SKIP words that have no exact match and no clear synonym in the vocabulary. Add the original word to the "skipped" list.
4. PRESERVE the SOV-reordered position when emitting glosses (do not re-sort alphabetically).

Examples (assume vocabulary contains: i, you, school, go, food, eat, today, what, name):
- "I am going to school." → {{"glosses": ["i", "school", "go"], "skipped": [], "mappings": {{}}}}
- "What is your name?" → {{"glosses": ["you", "name", "what"], "skipped": [], "mappings": {{}}}}
- "I had a meal today." → {{"glosses": ["i", "today", "food", "eat"], "skipped": [], "mappings": {{"meal": "food", "had": "eat"}}}}
- "I ate pizza yesterday." → {{"glosses": ["i", "eat"], "skipped": ["pizza", "yesterday"], "mappings": {{"ate": "eat"}}}}

User sentence: {text}

Respond with ONLY a single valid JSON object matching this schema, no markdown fences, no prose:
{{"glosses": [string], "skipped": [string], "mappings": {{original: gloss}}}}"""


TRANSLATION_PREVIEW_PROMPT = """You are a UI assistant for an ISL translator.
Given the structured translation result below, write ONE short, friendly sentence (max 25 words)
that tells the user what will be signed and what was dropped, so they can confirm before playback.

Format guidance:
- If nothing was skipped: "Signing: <glosses joined with spaces>."
- If some words were skipped: "Signing: <glosses>. Skipped (not in vocabulary): <skipped words>."
- If everything was skipped (empty glosses): "No words from your sentence are in the ISL vocabulary."
- Do not invent words. Do not add emojis. Do not add quotation marks around individual glosses.

Translation result (JSON):
{result}

Respond with ONLY the single sentence, no prefix, no JSON, no markdown."""
