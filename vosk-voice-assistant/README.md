# Vosk Voice Assistant

A privacy-focused, offline voice assistant with multiple speech recognition approaches optimized for quality and performance.

## 🎯 Overview

This project provides several voice assistant implementations, each with different trade-offs between speed, accuracy, and hallucination filtering:

| Script | STT Engine | VAD | Best For |
|--------|-----------|-----|----------|
| `voice_assistant_vosk_simple.py` | Vosk | Vosk built-in | Fast, simple, basic filtering |
| `voice_assistant_vosk_improved.py` | Vosk | Vosk built-in | Confidence-based filtering |
| `voice_assistant_vosk_fast.py` | Vosk | Vosk built-in | Aggressive anti-hallucination |
| `voice_assistant_hybrid.py` | Faster-Whisper | Vosk | **Best accuracy, no hallucinations** ⭐ |

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install vosk pyaudio piper-tts faster-whisper sounddevice numpy

# Install system audio (Linux)
sudo pacman -S portaudio  # Manjaro/Arch
sudo apt install portaudio19-dev  # Ubuntu/Debian
```

### Basic Usage

```bash
# Simple Vosk-only (fastest)
python voice_assistant_vosk_simple.py

# With hallucination filtering (balanced)
python voice_assistant_vosk_fast.py

# Best accuracy: Vosk VAD + Whisper STT (recommended)
python voice_assistant_hybrid.py
├── requirements.txt            # Python dependencies
├── PERFORMANCE_SUMMARY.md      # Detailed benchmarks
└── models/
    ├── vosk-model-small-en-us-0.15/  # STT model
    └── piper/
        ├── en_US-lessac-medium.onnx      # TTS model
        └── en_US-lessac-medium.onnx.json # TTS config
```
