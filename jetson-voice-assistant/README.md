# Restaurant Voice Assistant for Jetson

Production-ready voice assistant with menu-agnostic architecture.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the assistant
python3 run.py
```

That's it! The assistant will load the menu and start listening for commands.

## 📋 Features

- ✅ **FastText Intent Classification** - 99% accuracy, 4.79 MB model
- ✅ **Dynamic Menu Loading** - JSON file or REST API
- ✅ **Menu-Agnostic** - Works with any restaurant's menu
- ✅ **Context Tracking** - Remembers conversation state
- ✅ **Cart Management** - Add, remove, view, checkout
- ✅ **Fuzzy Search** - Handles typos and variations
- ✅ **Jetson Optimized** - Low memory, fast inference

## 🧪 Usage Examples

```
You: show me drinks
Assistant: Here are our drinks: Coffee, Tea, Soda, ...

You: add 2 coffees
Assistant: Added 2x Coffee ($8.00) to your cart

You: what's my total
Assistant: Your cart total is $8.00

You: checkout
Assistant: Processing order... Thank you!
```

## 📊 Model Comparison

| Model | Size | Accuracy | Best For |
|-------|------|----------|----------|
| **Small** | **4.79 MB** | **99.21%** | ✅ Jetson Nano/Orin |
| Medium | 19.12 MB | 99.21% | Server/Desktop |
