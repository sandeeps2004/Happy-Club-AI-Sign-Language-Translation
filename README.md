# Happy Club - AI Sign Language Translation

Real-time Indian Sign Language (ISL) recognition system that translates webcam video of sign language into natural English sentences. Built with Django, WebSockets, a BiLSTM deep learning model, and RTMPose for skeleton tracking.

## How It Works

```
Webcam -> WebSocket -> RTMPose (54 keypoints) -> Sign Boundary Detector -> BiLSTM Classifier -> Gemini LLM -> English Sentence
```

1. **Webcam frames** are streamed from the browser to the server via WebSocket (JPEG, ~20-30 fps)
2. **RTMPose WholeBody** extracts 54 keypoints per frame (12 upper body + 21 left hand + 21 right hand)
3. **Sign Boundary Detector** segments the continuous stream into individual signs using hand velocity — when hands pause, a sign boundary is triggered
4. **BiLSTM + Attention** classifies each 30-frame sign segment into one of 79 ISL word classes
5. **Gemini API** assembles the recognized ISL glosses (SOV order) into a grammatically correct English sentence

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Alpine.js, Tailwind CSS, HTML5 Canvas |
| **Backend** | Django 6.0, Django Channels, Daphne (ASGI) |
| **WebSocket** | Django Channels with `InMemoryChannelLayer` |
| **Pose Estimation** | RTMPose WholeBody via rtmlib (ONNX, 133 keypoints) |
| **Deep Learning** | PyTorch — Bidirectional LSTM with Attention Pooling |
| **Sentence Assembly** | Google Gemini 2.5 Flash (with rule-based fallback) |
| **Auth** | Custom User model with email login, RBAC |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **Templating** | Django Templates + HTMX + widget-tweaks |

## Project Structure

```
ai-sign-language/
|
|-- ai/                          # AI pipeline (training, demo, shared core)
|   |-- core/                    # Shared inference modules (used by both ai/ and apps/)
|   |   |-- assembler.py         # ISL gloss -> English sentence (Gemini + fallback)
|   |   |-- constants.py         # All hyperparameters & thresholds (single source of truth)
|   |   |-- detector.py          # Sign boundary detector (velocity-based segmentation)
|   |   |-- extractor.py         # RTMPose keypoint extraction (54 keypoints x 2 coords)
|   |   |-- model.py             # BiLSTM + Attention classifier (thread-safe singleton)
|   |   |-- preprocessing.py     # Normalization, velocity features, hand velocity
|   |-- checkpoints/             # Trained model weights (.pt files)
|   |-- data/                    # INCLUDE dataset (keypoint .npy files)
|   |-- charts/                  # Training visualization outputs
|   |-- train.py                 # Training script (INCLUDE-50 dataset)
|   |-- setup_dataset.py         # Download & extract INCLUDE ISL dataset from Zenodo
|   |-- demo.py                  # Standalone OpenCV demo (no Django)
|   |-- visualize.py             # Training metrics visualization
|   |-- config.py                # AI-specific config
|
|-- apps/                        # Django applications
|   |-- accounts/                # Custom User model, email auth, registration, profile
|   |-- core/                    # Dashboard, middleware, logging, base models, template tags
|   |-- sign_language/           # Real-time interpreter app
|       |-- consumers.py         # WebSocket consumer (frame processing + inference)
|       |-- inference.py         # Thin import layer from ai/core/
|       |-- models.py            # InterpretationSession & Prediction (DB persistence)
|       |-- routing.py           # WebSocket URL routing
|       |-- views.py             # Interpreter page view
|       |-- admin.py             # Django admin registration
|
|-- config/                      # Django project settings
|   |-- settings.py              # Main settings (reads from .env)
|   |-- urls.py                  # Root URL configuration
|   |-- asgi.py                  # ASGI application (HTTP + WebSocket routing)
|   |-- wsgi.py                  # WSGI fallback
|
|-- templates/                   # Django templates
|   |-- layouts/                 # Base templates (base.html, auth_base.html)
|   |-- sign_language/           # Interpreter UI (Alpine.js + Canvas landmark drawing)
|   |-- accounts/                # Login, register, profile, password reset
|   |-- core/                    # Dashboard, 404, 500
|   |-- includes/                # Sidebar, topbar, pagination, command palette
|   |-- components/              # Reusable UI components
|
|-- services/                    # Service layer (email, etc.)
|-- static/                      # Static assets (JS, CSS)
|-- tests/                       # Test suite
|-- media/                       # User uploads
|-- requirements.txt             # Python dependencies
|-- manage.py                    # Django management
|-- .env.example                 # Environment variable template
```

## AI Model Details

### Architecture

```
Input (30 x 216) -> LayerNorm -> BiLSTM (2 layers, 128 hidden) -> Attention Pooling -> FC(256->128) -> ReLU -> Dropout(0.3) -> FC(128->79) -> Softmax
```

- **Input**: 30 frames x 216 features (108 keypoint positions + 108 velocity deltas)
- **Output**: 79 ISL word classes (INCLUDE dataset)
- **Normalization**: Shoulder-centered, shoulder-distance scaled, per-hand min-max
- **Sign Segmentation**: Hand-only velocity tracking with configurable pause detection thresholds

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `SEQ_LEN` | 30 | Frames per sign (padded or uniformly sampled) |
| `FEATURE_DIM` | 108 | 54 keypoints x 2 coordinates (x, y) |
| `FEATURE_DIM_FINAL` | 216 | Position (108) + Velocity (108) |
| `HIDDEN_SIZE` | 128 | LSTM hidden units |
| `NUM_LAYERS` | 2 | Stacked BiLSTM layers |
| `NUM_CLASSES` | 79 | ISL word vocabulary |
| `VELOCITY_THRESHOLD` | 0.02 | Hand-only velocity below this = pause |
| `PAUSE_FRAMES` | 8 | Consecutive still frames to trigger word boundary |
| `MIN_SIGN_FRAMES` | 10 | Minimum frames for a valid sign segment |
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum softmax confidence to accept a prediction |

### Dataset

Trained on the [INCLUDE ISL dataset](https://zenodo.org/records/4010759) — a large-scale Indian Sign Language dataset with video recordings of 263 signs across multiple categories.

## Setup

### Prerequisites

- Python 3.12+
- Webcam (for real-time inference)

### Installation

```bash
# Clone
git clone https://github.com/sandeeps2004/Happy-Club-AI-Sign-Language-Translation.git
cd cd Happy-Club-AI-Sign-Language-Translation

# Virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Edit .env — set SECRET_KEY, GEMINI_API_KEY, etc.
```

### Environment Variables

```bash
# Required
SECRET_KEY=your-secret-key-here
GEMINI_API_KEY=your-gemini-api-key    # For sentence assembly (optional — falls back to rule-based)

# Optional
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
TIME_ZONE=UTC

# Database (defaults to SQLite)
# DB_ENGINE=django.db.backends.postgresql
# DB_NAME=sign_language_db
# DB_USER=postgres
# DB_PASSWORD=password
# DB_HOST=localhost
# DB_PORT=5432
```

### Database Setup

```bash
python manage.py migrate
python manage.py createsuperuser
```

### AI Model Setup

```bash
cd ai/

# Option A: Download INCLUDE dataset and train
python setup_dataset.py --download --extract
python train.py --epochs 50

# Option B: Use pre-trained checkpoint
# Place isl_lstm_best.pt in ai/checkpoints/
```

### Run

```bash
# Development server (Daphne ASGI — required for WebSocket support)
python manage.py runserver

# Or with Daphne directly
daphne -b 0.0.0.0 -p 8000 config.asgi:application
```

Open http://localhost:8000 and navigate to the Sign Language Interpreter.

## Usage

1. **Start Camera** — enables webcam; landmarks are extracted server-side via RTMPose
2. **Start Recording** — begins sign detection; perform ISL signs in front of the camera
3. **Pause briefly** (~0.3s) between each sign — the system uses hand velocity to detect word boundaries
4. **Stop Recording** — triggers sentence assembly from all recognized words
5. The assembled English sentence appears in the results panel

## WebSocket Protocol

### Client -> Server

```json
{"type": "frame", "image": "<base64_jpeg>"}
{"type": "start_recording"}
{"type": "stop_recording"}
{"type": "clear"}
```

### Server -> Client

```json
{"type": "model_info", "num_classes": 79, "glosses": [...], "device": "cpu"}
{"type": "frame_result", "landmarks": {...}, "status": {...}, "prediction": {...}}
{"type": "prediction", "word": "hello", "confidence": 0.95, "buffer": ["hello"]}
{"type": "sentence", "text": "Hello, how are you?", "glosses": ["hello", "how", "you"]}
{"type": "recording_started"}
{"type": "recording_stopped"}
{"type": "cleared"}
{"type": "error", "message": "..."}
```

## Standalone Demo (No Django)

```bash
cd ai/
python demo.py
```

Runs a real-time OpenCV window with live landmark overlay and sign detection — useful for testing the AI pipeline without the web server.

## Testing

```bash
pytest tests/
```

## License

MIT License

Copyright (c) 2026 Sandeep Sahu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

**Happy Club - AI Sign Language Translation**

