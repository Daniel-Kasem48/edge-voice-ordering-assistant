# Models Directory

This directory contains machine learning models for the voice assistant. **Models are NOT included in the repository** and must be generated locally.

## 📁 Directory Structure

```
models/
├── README.md                           # This file
├── restaurant_intent_small.bin         # Generated - 4.8 MB
├── restaurant_intent_medium.bin        # Generated - 19 MB
└── piper/                              # Optional - TTS voice
    ├── en_US-amy-medium.onnx          # Downloaded - 60 MB
    └── en_US-amy-medium.onnx.json     # Downloaded - 1 KB
```

## 🚀 How to Generate Models

### Intent Classification Models (Required)

These models are **required** for the assistant to understand user intent.

```bash
# From the jetson-voice-assistant/ directory
python train_two_models.py --data training_data_fasttext.txt
```

**Output**:
- `restaurant_intent_small.bin` - 4.8 MB, 85.85% accuracy, optimized for Jetson
- `restaurant_intent_medium.bin` - 19 MB, 86.79% accuracy, higher quality

**Training Time**: ~10 seconds on modern hardware

### TTS Voice Model (Optional)

This model is **optional** and only needed if you want voice output.

```bash
# Create directory
mkdir -p models/piper

# Download Amy voice (natural, high quality)
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

**Download Time**: 2-5 minutes (depending on connection)

## 📊 Model Details

### Small Model (restaurant_intent_small.bin)

- **Size**: 4.81 MB
- **Accuracy**: 85.85%
- **Inference**: < 1ms
- **Best For**: Jetson Nano, resource-constrained devices
- **Word N-grams**: Bigrams (2)
- **Dimensions**: 50

### Medium Model (restaurant_intent_medium.bin)

- **Size**: 19.14 MB
- **Accuracy**: 86.79%
- **Inference**: < 1ms
- **Best For**: Jetson Orin, development machines
- **Word N-grams**: Trigrams (3)
- **Dimensions**: 100

### Amy Voice (en_US-amy-medium.onnx)

- **Size**: 60 MB
- **Sample Rate**: 22050 Hz
- **Quality**: High (natural sounding)
- **Language**: American English
- **Speed**: Real-time on Jetson Orin and Jetson Nano

## ✅ Verification

After generating/downloading models:

```bash
# Check intent models exist
ls -lh models/*.bin

# Expected output:
# -rw-r--r-- 1 user user 4.8M Jan 23 12:00 restaurant_intent_small.bin
# -rw-r--r-- 1 user user  19M Jan 23 12:00 restaurant_intent_medium.bin

# Check TTS model (if downloaded)
ls -lh models/piper/*.onnx

# Expected output:
# -rw-r--r-- 1 user user  60M Jan 23 12:00 en_US-amy-medium.onnx
```

## 🔒 Git Ignore

Models are excluded from git via `.gitignore`:

```
models/*.bin
models/*.onnx
models/piper/*.onnx
models/piper/*.json
```

**Why?** Model files are large (5-60 MB) and can be regenerated/downloaded quickly.

## 🔄 When to Regenerate Models

Regenerate intent models when:

- ✅ You modify training data
- ✅ You add new intents
- ✅ You update the menu significantly
- ✅ Accuracy drops below 85%
- ✅ New patterns aren't recognized

**Command**:
```bash
python train_two_models.py --data training_data_fasttext.txt
```

## 🎯 Model Selection Guide

### For Development (Laptop/Desktop)

Use **medium model** for best accuracy (automatically selected by assistant)

### For Production (Jetson Nano 4GB)

Use **small model** for lower memory

### For Production (Jetson Orin 8GB+)

Use **medium model** for best accuracy (recommended)

## 📈 Training Data Stats

Current training data (`training_data_fasttext.txt`):

- **Total Examples**: 703
- **Intents**: 16
- **Split**: 597 train / 106 validation
- **Top Intents**:
  - `add_to_cart`: 165 examples
  - `query_ingredients`: 78 examples
  - `query_item`: 71 examples
  - `remove_from_cart`: 49 examples
  - `search_menu`: 48 examples

## 🔗 External Resources

- **FastText**: https://fasttext.cc/
- **Piper TTS**: https://github.com/rhasspy/piper
- **Piper Voices**: https://huggingface.co/rhasspy/piper-voices

---

**Last Updated**: January 23, 2026  
**Model Version**: 1.0  
**Training Data**: 703 examples, 16 intents
