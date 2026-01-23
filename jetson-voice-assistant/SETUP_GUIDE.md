# Jetson Voice Assistant - Developer Setup Guide

Complete setup instructions for developers to get the voice assistant running from scratch.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Model Setup](#model-setup)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)
7. [Development Workflow](#development-workflow)

---

## 🔧 Prerequisites

### Hardware Requirements

- **Minimum**: Jetson Orin Nano 4GB
- **Recommended**: Jetson Orin Nano 8GB or higher
- **Alternative**: x86_64 Linux laptop (for development/testing)

### Software Requirements

- Ubuntu 20.04+ / Jetson Linux 35.x+
- Python 3.8 - 3.11
- Git
- 2GB free disk space (for models)

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
cd /path/to/your/projects
git clone <your-repo-url>
cd jetson-voice-assistant
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

#### For Text-Only Mode (Testing)

```bash
pip install -r requirements.txt
```

#### For Intent Classification (Required for Production)

```bash
pip install fasttext-wheel>=0.9.2
```

**Note**: On ARM64 (Jetson), use `fasttext-wheel` instead of `fasttext` for better compatibility.

#### For Full Voice Mode (Optional)

```bash
# Install Piper TTS
pip install piper-tts>=1.2.0

# Install audio dependencies (if needed)
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio sounddevice soundfile
```

---

## 🤖 Model Setup

**Important**: Model files are NOT included in the repository due to size. You must generate them locally.

### Step 1: Training Data Overview

The training data is already included in the repository:

```
training_data_fasttext.txt    # 703 examples, 16 intents
validation_data_fasttext.txt  # Validation set
```

**Training Data Structure**:
```
__label__greeting hi there
__label__add_to_cart i want 2 coffees
__label__query_ingredients what's in the pizza
__label__get_price how much is the burger
...
```

### Step 2: Train Intent Classification Models

Run the training script to generate both small and medium models:

```bash
python train_two_models.py --data training_data_fasttext.txt
```

**Expected Output**:
```
✅ Small model saved: models/restaurant_intent_small.bin
   Size: 4.81 MB
   Accuracy: 85.85%

✅ Medium model saved: models/restaurant_intent_medium.bin
   Size: 19.14 MB
   Accuracy: 86.79%
```

**Models Generated**:
- `models/restaurant_intent_small.bin` - 4.81 MB, optimized for Jetson
- `models/restaurant_intent_medium.bin` - 19.14 MB, higher accuracy

### Step 3: Download Piper TTS Models (Optional)

For natural voice output, download Piper TTS models:

```bash
# Create models directory
mkdir -p models/piper

# Download Amy voice (natural, high quality, 22kHz)
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

cd ../..
```

**Voice Model**:
- **Amy** (60MB, 22kHz) - Natural, high quality voice

---

## ⚙️ Configuration

### Menu Configuration

The assistant loads menu from `data/menu.json`. This file defines:

- Menu items (name, price, category, description, **ingredients**)
- Categories (Main Course, Beverages, Desserts, etc.)
- Options and modifiers
- Tags for search

**Example Menu Item**:
```json
{
  "name": "Margherita Pizza",
  "price": 12.99,
  "category": "Main Course",
  "description": "Classic Italian pizza",
  "ingredients": [
    "pizza dough",
    "tomato sauce",
    "mozzarella cheese",
    "fresh basil",
    "olive oil"
  ],
  "tags": ["vegetarian", "pizza", "italian"],
  "options": [...]
}
```

### Tenant Configuration

Multi-tenant support via `data/tenants/` directory (optional):

```
data/tenants/
├── demo_restaurant/
│   └── config.yaml
└── pizza_place/
    └── config.yaml
```

**Tenant Config Example** (`data/tenants/demo_restaurant/config.yaml`):
```yaml
tenant_id: demo_restaurant
tenant_name: Demo Restaurant
wake_word_phrase: jarvis
menu_source:
  type: file
  path: data/menu.json
```

---

## 🧪 Testing

### Step 1: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8-3.11

# Check dependencies
pip list | grep -E "rapidfuzz|fasttext|pyyaml"

# Verify models exist
ls -lh models/*.bin
```

### Step 2: Test Text Mode

```bash
python run.py
```

**Expected Interaction**:
```
🚀 Initializing Text Assistant for tenant: demo_restaurant
✅ Loaded FastText model: models/restaurant_intent_medium.bin
✅ Using menu-trained medium model (19MB)
✅ Assistant ready for: Demo Restaurant

You: hi
🤖 Hey! Welcome to Demo Restaurant. What can I get you?

You: show me pizza
🤖 For pizza, we have Margherita, Pepperoni, BBQ Chicken, Hawaiian.

You: what's in margherita pizza
🤖 Margherita Pizza contains: pizza dough, tomato sauce, mozzarella cheese, fresh basil, olive oil.

You: add 2 margherita pizzas
🤖 Added 2x Margherita Pizza ($25.98) to your cart.

You: checkout
🤖 Ready to checkout! Your cart total: $25.98. Confirm? (yes/no)
```

### Step 3: Test with TTS (if installed)

```bash
python run_with_tts.py
```

The assistant will speak responses using Piper TTS.

### Step 4: Run Test Suite

```bash
# Test speech formatting
python test_speech_formatting.py

# Test ingredient queries
python -c "
from src.assistant import TextAssistant
assistant = TextAssistant('demo_restaurant')
print(assistant.process_input('what\\'s in the cheeseburger'))
"
```

---

## 🔍 Troubleshooting

### Issue: Models Not Found

**Error**: `FileNotFoundError: models/restaurant_intent_medium.bin`

**Solution**:
```bash
# Train models first
python train_two_models.py --data training_data_fasttext.txt
```

### Issue: FastText Import Error

**Error**: `ModuleNotFoundError: No module named 'fasttext'`

**Solution**:
```bash
# On ARM64 (Jetson)
pip install fasttext-wheel

# On x86_64
pip install fasttext-wheel
# OR
pip install fasttext
```

### Issue: Low Intent Accuracy

**Problem**: Intent classifier not recognizing queries

**Solutions**:

1. **Retrain models**:
   ```bash
   python train_two_models.py --data training_data_fasttext.txt
   ```

2. **Check training data**:
   ```bash
   grep "query_ingredients" training_data_fasttext.txt | wc -l
   # Should show 78+ examples
   ```

3. **Test specific patterns**:
   ```bash
   python -c "
   import fasttext
   model = fasttext.load_model('models/restaurant_intent_medium.bin')
   label, conf = model.predict('show me pizza ingredients')
   print(f'{label[0]} - {conf[0]:.3f}')
   "
   ```

### Issue: Piper TTS Not Speaking

**Error**: No audio output

**Solutions**:

1. **Check TTS models exist**:
   ```bash
   ls -lh models/piper/*.onnx
   ```

2. **Test audio system**:
   ```bash
   speaker-test -t wav -c 2
   ```

3. **Check Piper installation**:
   ```bash
   pip list | grep piper
   python -c "import piper; print(piper.__version__)"
   ```

### Issue: Context Manager Overriding Intent

**Problem**: Ingredient queries misclassified when in conversation context

**Status**: Known issue - context manager can override high-confidence intents

**Workaround**: Query ingredients in fresh conversation state

---

## 🛠️ Development Workflow

### Adding New Intents

1. **Add training examples** to `generate_training_from_menu.py`:
   ```python
   def _generate_new_intent(self, items: List[str]) -> List[Tuple[str, str]]:
       patterns = [
           "pattern with {item}",
           "another pattern",
       ]
       data = []
       for pattern in patterns:
           if '{item}' in pattern:
               for item in items:
                   data.append((pattern.format(item=item), "new_intent"))
           else:
               data.append((pattern, "new_intent"))
       return data
   ```

2. **Regenerate training data**:
   ```bash
   python generate_training_from_menu.py --menu data/menu.json --output training_data_fasttext.txt
   ```

3. **Retrain models**:
   ```bash
   python train_two_models.py --data training_data_fasttext.txt
   ```

4. **Add intent handler** in `src/assistant.py`:
   ```python
   def _handle_new_intent(self, text: str, intent: IntentResult) -> str:
       """Handle new intent"""
       # Your logic here
       return "Response message"
   
   # Register in handlers dict
   handlers = {
       ...
       "new_intent": self._handle_new_intent,
   }
   ```

### Updating Menu

1. **Edit menu file**: `data/menu.json`

2. **Ensure ingredients are included**:
   ```json
   {
     "name": "New Item",
     "ingredients": ["ingredient1", "ingredient2", ...]
   }
   ```

3. **Regenerate training data** (picks up new item names):
   ```bash
   python generate_training_from_menu.py --menu data/menu.json --output training_data_fasttext.txt
   ```

4. **Retrain models**:
   ```bash
   python train_two_models.py --data training_data_fasttext.txt
   ```

### Customizing Voice

Edit `src/output/piper_tts.py`:

```python
# Voice tuning parameters
self.length_scale = 1.05  # Speed (lower = faster)
self.noise_scale = 0.667   # Expressiveness
self.noise_w_scale = 0.8   # Phoneme variation
self.sentence_silence = 0.35  # Pause between sentences (seconds)
```

### Testing Changes

```bash
# Quick test
python run.py

# Test specific query
python -c "
from src.assistant import TextAssistant
assistant = TextAssistant('demo_restaurant')
print(assistant.process_input('your test query here'))
"

# Test with TTS
python run_with_tts.py
```

---

## 📊 Model Training Details

### Training Data Statistics

- **Total Examples**: 703
- **Intents**: 16
- **Train/Valid Split**: 597/106 (85/15)

**Intent Distribution**:
```
add_to_cart          165 examples
query_ingredients     78 examples
query_item            71 examples
remove_from_cart      49 examples
search_menu           48 examples
get_price             42 examples
checkout              35 examples
confirm_yes           34 examples
confirm_no            30 examples
view_cart             28 examples
clear_cart            24 examples
modify_item           23 examples
help                  20 examples
greeting              19 examples
goodbye               19 examples
view_menu             18 examples
```

### FastText Hyperparameters

**Small Model** (Jetson-optimized):
```python
lr=0.1              # Learning rate
epoch=50            # Training epochs
wordNgrams=2        # Bigrams
dim=50              # Word vector dimensions
loss='softmax'      # Multiclass classification
```

**Medium Model** (balanced):
```python
lr=0.1
epoch=50
wordNgrams=3        # Trigrams (better context)
dim=100             # Higher dimensions
loss='softmax'
```

### Model Performance

| Model | Size | Accuracy | Precision | Recall | Inference |
|-------|------|----------|-----------|--------|-----------|
| Small | 4.81 MB | 85.85% | 0.8585 | 0.8585 | < 1ms |
| Medium | 19.14 MB | 86.79% | 0.8679 | 0.8679 | < 1ms |

---

## 🎯 Feature Checklist

### Implemented Features

- ✅ Intent classification (FastText, 86.79% accuracy)
- ✅ Menu search with fuzzy matching
- ✅ Cart management (add, remove, view, clear, checkout)
- ✅ Ingredient queries ("what's in X", "how is X made")
- ✅ Price queries
- ✅ Context tracking (conversation state)
- ✅ Ambiguity detection (asks for clarification)
- ✅ Natural TTS voice (Piper with Amy voice)
- ✅ Speech formatting (humanized responses)
- ✅ Dynamic vocabulary extraction
- ✅ Multi-tenant support

### Testing Requirements Before Deployment

- [ ] Train intent models on your machine
- [ ] Test all 16 intents with your menu
- [ ] Verify ingredient queries work
- [ ] Test cart operations (add/remove/checkout)
- [ ] Test ambiguous queries (e.g., "pizza" with multiple pizzas)
- [ ] Test voice output (TTS) if using voice mode
- [ ] Test on target hardware (Jetson)
- [ ] Verify model file sizes are acceptable
- [ ] Test inference speed (should be < 10ms)

---

## 📚 Additional Resources

### Documentation Files

- `README.md` - Quick start and overview
- `PIPER_TTS_INTEGRATION.md` - TTS setup and configuration
- `TTS_INTEGRATION_SUMMARY.md` - Voice implementation details
- `models/README.md` - Model information

### Key Source Files

- `src/assistant.py` - Main assistant logic, intent handlers
- `src/nlp/lightweight_intent_classifier.py` - Intent classification
- `src/menu/menu_matcher.py` - Fuzzy menu search
- `src/menu/vocabulary.py` - Dynamic vocabulary extraction
- `src/output/piper_tts.py` - TTS engine
- `src/output/speech_formatter.py` - Natural speech formatting
- `generate_training_from_menu.py` - Training data generator
- `train_two_models.py` - Model training script

### Training Scripts

```bash
# Generate training data from menu
python generate_training_from_menu.py --menu data/menu.json --output training_data_fasttext.txt

# Train both models
python train_two_models.py --data training_data_fasttext.txt

# Test specific query
python -c "from src.assistant import TextAssistant; a = TextAssistant('demo_restaurant'); print(a.process_input('test query'))"
```

---

## 🚀 Quick Start Checklist

For new developers:

- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Install fasttext (`pip install fasttext-wheel`)
- [ ] Train intent models (`python train_two_models.py --data training_data_fasttext.txt`)
- [ ] Verify models exist (`ls models/*.bin`)
- [ ] Test text mode (`python run.py`)
- [ ] (Optional) Download Piper TTS models
- [ ] (Optional) Test with TTS (`python run_with_tts.py`)

**Time estimate**: 15-30 minutes (excluding model downloads)

---

## 💡 Tips for Success

1. **Always retrain models** after changing training data or menu
2. **Test incrementally** - don't change multiple things at once
3. **Check intent confidence** - use `python run.py` to see confidence scores
4. **Use medium model** for development, small model for production
5. **Keep ingredients updated** in menu.json for accurate ingredient queries
6. **Monitor model file sizes** - they should NOT be committed to git
7. **Test on target hardware** - Jetson performance differs from laptop

---

## 🐛 Known Issues

1. **Context manager can override intents**: When in `awaiting_item_selection` context, ingredient queries may be misclassified as `get_price`. Workaround: Ask ingredient questions in fresh conversation.

2. **Fuzzy matching edge cases**: "pepperoni pizza" may match both "Pepperoni Pizza" (100%) and "BBQ Chicken Pizza" (85%) due to shared "pizza" word. System handles this by asking for clarification.

3. **Model files not in git**: Intentional - models are 5-20MB each and should be generated locally.

---

## 📞 Support

For issues or questions:

1. Check this guide first
2. Review troubleshooting section
3. Check training data quality
4. Retrain models
5. Test with simplified queries
6. Review logs and confidence scores

---

**Last Updated**: January 23, 2026  
**Version**: 1.0  
**Tested On**: Jetson Orin Nano, Ubuntu 22.04, Python 3.10
