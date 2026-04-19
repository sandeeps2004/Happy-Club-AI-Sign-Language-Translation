# Happy Club — ISL Sign Language → Text Pipeline (Proof of Concept)

## What this does

Real-time Indian Sign Language word recognition via webcam → sentence assembly via LLM.

**Pipeline:** Webcam → MediaPipe Holistic (543 landmarks) → LSTM classifier → Word buffer → LLM sentence assembly

## Architecture

```
[Webcam] → [MediaPipe Holistic] → [Keypoint Sequence Buffer]
                                          ↓
                               [Sign Boundary Detector]
                               (hand velocity < threshold = pause)
                                          ↓
                               [LSTM Word Classifier]
                               (trained on INCLUDE-50 ISL dataset)
                                          ↓
                               [Word Accumulator Buffer]
                               ["hello", "how", "you"]
                                          ↓
                               [LLM Sentence Assembly]
                               "Hello, how are you?"
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the INCLUDE dataset

```bash
python setup_dataset.py --download
```

This downloads INCLUDE-50 (50 ISL word classes, ~1200 videos) from HuggingFace
and extracts MediaPipe keypoints automatically.

### 3. Train the LSTM model

```bash
python train.py
```

Trains a 3-layer LSTM on the extracted keypoints. Takes ~10-15 min on CPU, ~2 min on GPU.

### 4. Run the live demo

```bash
python demo.py
```

Opens webcam, recognizes ISL signs in real-time, assembles sentences.

### 5. Run without webcam (test mode)

```bash
python demo.py --test
```

Runs inference on sample videos from the dataset to verify the pipeline works.

## Project Structure

```
happy_club_pipeline/
├── README.md
├── requirements.txt
├── config.py              # All constants and paths
├── setup_dataset.py       # Download INCLUDE + extract keypoints
├── model.py               # LSTM model definition
├── train.py               # Training loop
├── keypoint_extractor.py  # MediaPipe landmark extraction
├── sign_detector.py       # Real-time sign boundary detection
├── sentence_assembler.py  # LLM word→sentence conversion
├── demo.py                # Main entry point
├── data/                  # Created by setup_dataset.py
│   ├── include50/         # Raw videos
│   └── keypoints/         # Extracted .npy files
└── checkpoints/           # Saved model weights
```

## Key Design Decisions

1. **MediaPipe Holistic** over OpenPose: Runs real-time on CPU, no GPU required for inference
2. **LSTM** over Transformer: Simpler, faster to train on small dataset, good enough for 50 classes
3. **Sign boundary via velocity**: When hand movement drops below threshold for 300ms, treat as word boundary
4. **LLM for sentence assembly**: Converts raw ISL gloss order to natural English grammar
5. **INCLUDE-50** over full INCLUDE: 50 most common words = sufficient for demo, faster training

## Requirements

- Python 3.9+
- Webcam (for live demo)
- ~2GB disk space (dataset + model)
- GPU optional (training is fast enough on CPU for INCLUDE-50)
