# Offline Voice Assistant with Tool Calling

A fully offline voice assistant that runs on laptop or RK3588S, featuring wake word detection, speech recognition, LLM reasoning, and tool calling capabilities.

## Features

- 🎤 Wake word detection ("Hey Mycroft" and others)
- 🗣️ Speech-to-text using Whisper
- 🧠 LLM reasoning with Phi-3 or Functionary
- 🔧 Tool calling for actions (shopping cart, etc.)
- 🔊 Text-to-speech with Piper
- 🚀 Fully offline operation
- 💻 Optimized for both laptop and embedded devices

## Quick Start

1. **Setup environment:**
   ```bash
   ./scripts/setup.sh
   ```

2. **Build dependencies:**
   ```bash
   ./scripts/build.sh
   ```

3. **Download models:**
   ```bash
   ./scripts/download_models.sh
   ```

4. **Run the assistant:**
   ```bash
   python src/main.py
   ```

## Voice Commands

- **Shopping**: "Add milk to my cart", "Check if you have eggs"
- **Search**: "Search for bread", "What dairy products do you have?"
- **Cart**: "What's in my cart?", "Show me my total"

## Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- ~10GB disk space for models
- Working microphone and speakers

## Architecture

```
Wake Word → ASR (Whisper) → LLM (Phi-3) → Tool Execution → TTS (Piper) → Audio Output
```

## Models Used

- **ASR**: Whisper base/small
- **LLM**: Phi-3 Mini (Q4) or Functionary-Small
- **TTS**: Piper en_US-amy-medium
- **Wake Word**: OpenWakeWord models