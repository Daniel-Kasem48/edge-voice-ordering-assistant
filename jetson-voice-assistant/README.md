# Restaurant Voice Assistant for Jetson# Restaurant Voice Assistant for Jetson



Production-ready voice assistant with menu-agnostic architecture and ingredient query support.Production-ready voice assistant with menu-agnostic architecture.



## ⚠️ Important: Model Setup Required## 🚀 Quick Start



**Models are NOT included in the repository.** You must generate them before running the assistant.```bash

# 1. Install dependencies

### Quick Setup (5 minutes)pip install -r requirements.txt



```bash# 2. Run the assistant

# 1. Install dependenciespython3 run.py

pip install -r requirements.txt```

pip install fasttext-wheel

That's it! The assistant will load the menu and start listening for commands.

# 2. Train intent models (REQUIRED - takes ~10 seconds)

python train_two_models.py --data training_data_fasttext.txt## 📋 Features



# 3. Run the assistant- ✅ **FastText Intent Classification** - 99% accuracy, 4.79 MB model

python run.py- ✅ **Dynamic Menu Loading** - JSON file or REST API

```- ✅ **Menu-Agnostic** - Works with any restaurant's menu

- ✅ **Context Tracking** - Remembers conversation state

## 📚 Documentation- ✅ **Cart Management** - Add, remove, view, checkout

- ✅ **Fuzzy Search** - Handles typos and variations

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions for developers- ✅ **Jetson Optimized** - Low memory, fast inference

- **[QUICK_SETUP.md](QUICK_SETUP.md)** - Quick reference with copy-paste commands

- **[models/README.md](models/README.md)** - Model generation and download guide## 🧪 Usage Examples

- **[PIPER_TTS_INTEGRATION.md](PIPER_TTS_INTEGRATION.md)** - Voice output setup (optional)

```

## 🚀 Quick StartYou: show me drinks

Assistant: Here are our drinks: Coffee, Tea, Soda, ...

```bash

# 1. Clone repositoryYou: add 2 coffees

git clone <repo-url>Assistant: Added 2x Coffee ($8.00) to your cart

cd jetson-voice-assistant

You: what's my total

# 2. Create virtual environmentAssistant: Your cart total is $8.00

python3 -m venv venv

source venv/bin/activateYou: checkout

Assistant: Processing order... Thank you!

# 3. Install dependencies```

pip install -r requirements.txt

pip install fasttext-wheel## 📊 Model Comparison



# 4. Train intent models (REQUIRED)| Model | Size | Accuracy | Best For |

python train_two_models.py --data training_data_fasttext.txt|-------|------|----------|----------|

| **Small** | **4.79 MB** | **99.21%** | ✅ Jetson Nano/Orin |

# 5. Test text mode| Medium | 19.12 MB | 99.21% | Server/Desktop |

python run.py

# 6. Optional: Download TTS models and test voice
# See SETUP_GUIDE.md for TTS setup
python run_with_tts.py
```

## 📋 Features

- ✅ **FastText Intent Classification** - 86.79% accuracy, 19 MB model
- ✅ **Ingredient Queries** - "what's in the pizza", "how is burger made"
- ✅ **Dynamic Menu Loading** - JSON file or REST API
- ✅ **Menu-Agnostic** - Works with any restaurant's menu
- ✅ **Context Tracking** - Remembers conversation state
- ✅ **Cart Management** - Add, remove, view, checkout
- ✅ **Fuzzy Search** - Handles typos and variations
- ✅ **Ambiguity Detection** - Asks for clarification when needed
- ✅ **Natural TTS Voice** - Piper with Amy voice (optional)
- ✅ **Jetson Optimized** - Low memory, fast inference

## 🧪 Usage Examples

```
You: hi
Assistant: Hey! Welcome to Demo Restaurant. What can I get you?

You: show me pizza
Assistant: For pizza, we have Margherita, Pepperoni, BBQ Chicken, Hawaiian.

You: what's in margherita pizza
Assistant: Margherita Pizza contains: pizza dough, tomato sauce, 
           mozzarella cheese, fresh basil, olive oil.

You: add 2 margherita pizzas
Assistant: Added 2x Margherita Pizza ($25.98) to your cart.

You: what's my total
Assistant: Your cart total: $25.98 (including tax)

You: checkout
Assistant: Ready to checkout! Confirm? (yes/no)
```

## 📊 Model Comparison

| Model | Size | Accuracy | Inference | Best For |
|-------|------|----------|-----------|----------|
| **Small** | **4.8 MB** | **85.85%** | **< 1ms** | ✅ Jetson Nano |
| **Medium** | **19 MB** | **86.79%** | **< 1ms** | ✅ Jetson Orin (recommended) |

## 🎯 What's Included vs Generated

### ✅ Included in Repository
- Training data (703 examples, 16 intents)
- Training scripts
- All source code
- Sample menu with ingredients
- Documentation

### ❌ NOT Included (Must Generate Locally)
- `models/restaurant_intent_small.bin` (4.8 MB)
- `models/restaurant_intent_medium.bin` (19 MB)
- `models/piper/en_US-amy-medium.onnx` (TTS voice, optional, 60 MB)

**Why?** Model files are too large for git. Training takes only 10 seconds.

## 🔧 Requirements

- Python 3.8 - 3.11
- 2GB free disk space (for models)
- Jetson Orin Nano 4GB+ (or x86_64 Linux for development)

## 📦 Dependencies

```bash
# Core (required)
pip install rapidfuzz pyyaml numpy fasttext-wheel

# Optional: TTS
pip install piper-tts pyaudio sounddevice
```

See `requirements.txt` for complete list.

## 🛠️ Development Workflow

### Adding New Intents

1. Edit `generate_training_from_menu.py`
2. Regenerate training data: `python generate_training_from_menu.py --menu data/menu.json --output training_data_fasttext.txt`
3. Retrain models: `python train_two_models.py --data training_data_fasttext.txt`
4. Add handler in `src/assistant.py`
5. Test: `python run.py`

### Updating Menu

1. Edit `data/menu.json` (include ingredients!)
2. Regenerate training data (picks up new item names)
3. Retrain models
4. Test

See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for detailed workflows.

## 🐛 Troubleshooting

### Models not found error

```bash
# Train models first
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
# Retrain
python train_two_models.py --data training_data_fasttext.txt
```

See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for complete troubleshooting guide.

## 📈 Training Data

- **Total**: 703 examples
- **Intents**: 16 (greeting, add_to_cart, query_ingredients, get_price, etc.)
- **Split**: 597 train / 106 validation
- **Accuracy**: 85.85% (small), 86.79% (medium)

## 🎤 Supported Intents

- `greeting` - "hi", "hello"
- `add_to_cart` - "add 2 pizzas", "I want coffee"
- `query_ingredients` - "what's in pizza", "how is burger made"
- `get_price` - "how much is the coffee"
- `search_menu` - "show me drinks"
- `view_cart` - "what's in my cart"
- `checkout` - "checkout", "complete order"
- `remove_from_cart` - "remove pizza"
- `clear_cart` - "clear my cart"
- And more... (16 total)

## 🔗 Additional Resources

- **FastText**: https://fasttext.cc/
- **Piper TTS**: https://github.com/rhasspy/piper
- **Piper Voices**: https://huggingface.co/rhasspy/piper-voices

---

**Last Updated**: January 23, 2026  
**Version**: 1.0  
**Tested On**: Jetson Orin Nano, Ubuntu 22.04, Python 3.10
