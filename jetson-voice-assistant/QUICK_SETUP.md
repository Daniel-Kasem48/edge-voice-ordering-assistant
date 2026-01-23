# Quick Setup Reference

## 🚀 Initial Setup (First Time)

```bash
# 1. Clone and setup
git clone <repo-url>
cd jetson-voice-assistant
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install fasttext-wheel

# 3. Train models (REQUIRED - models not in git)
python train_two_models.py --data training_data_fasttext.txt

# 4. Test
python run.py
```

---

## 📦 What's Included vs What You Generate

### ✅ Included in Repository
- `training_data_fasttext.txt` (703 examples)
- `validation_data_fasttext.txt` (validation set)
- `data/menu.json` (sample menu with ingredients)
- `generate_training_from_menu.py` (training data generator)
- `train_two_models.py` (model trainer)
- All source code in `src/`

### ❌ NOT Included (You Must Generate)
- `models/restaurant_intent_small.bin` (4.8 MB)
- `models/restaurant_intent_medium.bin` (19 MB)
- `models/piper/*.onnx` (TTS voice models, optional)

**Why?** Model files are too large for git (5-60 MB each). Training takes 5-10 seconds.

---

## 🔄 Common Workflows

### Add New Intent

```bash
# 1. Edit generate_training_from_menu.py
#    Add your _generate_new_intent() method

# 2. Regenerate training data
python generate_training_from_menu.py --menu data/menu.json --output training_data_fasttext.txt

# 3. Retrain models
python train_two_models.py --data training_data_fasttext.txt

# 4. Add handler in src/assistant.py
#    def _handle_new_intent(self, text, intent): ...

# 5. Test
python run.py
```

### Update Menu

```bash
# 1. Edit data/menu.json
#    Add/modify items (include ingredients!)

# 2. Regenerate training data (picks up new items)
python generate_training_from_menu.py --menu data/menu.json --output training_data_fasttext.txt

# 3. Retrain models
python train_two_models.py --data training_data_fasttext.txt

# 4. Test
python run.py
```

### Download TTS Models (Optional)

```bash
mkdir -p models/piper
cd models/piper

# Amy voice (natural, high quality, 60MB, 22kHz)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

cd ../..
pip install piper-tts

# Test with TTS
python run_with_tts.py
```

---

## 🧪 Testing Commands

```bash
# Test text mode
python run.py

# Test with TTS
python run_with_tts.py

# Test specific query
python -c "from src.assistant import TextAssistant; a = TextAssistant('demo_restaurant'); print(a.process_input('what is in pizza'))"

# Check model accuracy
python train_two_models.py --data training_data_fasttext.txt
# Look for "Accuracy: XX.XX%"

# Test intent classification
python -c "
import fasttext
model = fasttext.load_model('models/restaurant_intent_medium.bin')
label, conf = model.predict('show me pizza ingredients')
print(f'{label[0].replace(\"__label__\", \"\")} - {conf[0]:.3f}')
"

# Count training examples per intent
grep "query_ingredients" training_data_fasttext.txt | wc -l
```

---

## 📊 Expected Model Stats

After running `python train_two_models.py --data training_data_fasttext.txt`:

```
✅ Small model saved: models/restaurant_intent_small.bin
   Size: 4.81 MB
   Accuracy: 85.85%
   Inference: < 1ms

✅ Medium model saved: models/restaurant_intent_medium.bin
   Size: 19.14 MB
   Accuracy: 86.79%
   Inference: < 1ms
```

**Training Data**: 703 examples, 16 intents, 597 train / 106 validation

---

## 🐛 Quick Fixes

### Models not found
```bash
python train_two_models.py --data training_data_fasttext.txt
ls -lh models/*.bin
```

### FastText import error
```bash
pip install fasttext-wheel
```

### Low accuracy
```bash
# Check training data
wc -l training_data_fasttext.txt  # Should be 703+
grep "__label__" training_data_fasttext.txt | head -n 5

# Retrain
python train_two_models.py --data training_data_fasttext.txt
```

### Intent not recognized
```bash
# Test classification
python -c "
import fasttext
model = fasttext.load_model('models/restaurant_intent_medium.bin')
queries = ['show me pizza', 'what is in burger', 'add 2 coffees']
for q in queries:
    label, conf = model.predict(q)
    print(f'{q:30s} -> {label[0]:30s} ({conf[0]:.3f})')
"
```

---

## 💾 File Sizes

```
training_data_fasttext.txt         ~40 KB (text)
validation_data_fasttext.txt       ~8 KB (text)
restaurant_intent_small.bin        4.8 MB (generated)
restaurant_intent_medium.bin       19 MB (generated)
en_US-amy-medium.onnx             60 MB (optional download)
```

**Total Repository Size** (without models): ~50 KB  
**Total with Models**: ~85 MB

---

## 🎯 Production Checklist

Before deploying to Jetson:

- [ ] Trained both models locally
- [ ] Tested all intents (`python run.py`)
- [ ] Verified accuracy (> 85%)
- [ ] Tested ingredient queries
- [ ] Tested cart operations
- [ ] Updated menu.json with your items
- [ ] All menu items have ingredients
- [ ] Tested on Jetson hardware
- [ ] Verified models/*.bin exist
- [ ] Downloaded TTS models (if using voice)
- [ ] Tested TTS voice quality

---

## 📦 Deployment Package

Copy these to Jetson:

```bash
# Required files
src/                              # All source code
data/                             # Menu and config
requirements.txt                  # Dependencies
run.py                            # Text mode runner
run_with_tts.py                   # TTS mode runner (optional)
training_data_fasttext.txt        # For retraining if needed
train_two_models.py               # For retraining if needed

# Generated on Jetson (or copy if same arch)
models/restaurant_intent_small.bin          # 4.8 MB
models/restaurant_intent_medium.bin         # 19 MB (optional)
models/piper/en_US-amy-medium.onnx         # 60 MB (if using TTS)
models/piper/en_US-amy-medium.onnx.json    # 1 KB
```

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Initial setup | 5 min |
| Install dependencies | 2-5 min |
| Train models | 10 sec |
| Download TTS models | 2-5 min |
| Test text mode | 1 min |
| Add new intent | 10-15 min |
| Update menu | 5 min |
| **Total first-time setup** | **15-30 min** |

---

## 🔗 Quick Links

- Full Setup Guide: `SETUP_GUIDE.md`
- TTS Integration: `PIPER_TTS_INTEGRATION.md`
- Main README: `README.md`
- Model Info: `models/README.md`
