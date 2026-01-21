# Vosk Voice Assistant - Complete Documentation

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
```

---

## 📚 Implementation Guides

### 1. Simple Vosk (`voice_assistant_vosk_simple.py`)

Fastest implementation with basic filtering.

**Architecture:**
```
Mic → Vosk (STT+VAD) → Echo → Piper TTS
```

**Usage:**
```bash
# Small model (fast)
python voice_assistant_vosk_simple.py

# Large model (better accuracy)
python voice_assistant_vosk_simple.py --model models/vosk-model-en-us-0.22-lgraph
```

**Pros:**
- ✅ Fast (~50ms latency)
- ✅ Simple codebase
- ✅ Real-time partial transcription

**Cons:**
- ⚠️ May hallucinate with background noise

---

### 2. Improved Vosk (`voice_assistant_vosk_improved.py`)

Adds confidence-based filtering to reduce hallucinations.

**Usage:**
```bash
# Default (70% confidence, min 1 word)
python voice_assistant_vosk_improved.py

# Stricter filtering
python voice_assistant_vosk_improved.py --min-confidence 0.8 --min-words 2
```

**Output Example:**
```
📝 You: "hello there" [conf: 0.92]
🚫 Filtered: "the" (reason: low_confidence (0.43))
```

---

### 3. Fast with Anti-Hallucination (`voice_assistant_vosk_fast.py`)

Aggressive multi-stage filtering for clean transcription.

**Filtering Stages:**
1. **Word count check**: Minimum 2 words (configurable)
2. **Confidence score**: Minimum 75% (configurable)
3. **Hallucination word detection**: Blocks common filler words
4. **Filler word ratio**: Rejects if >50% are fillers
5. **Pattern detection**: Blocks repeated hallucinations

**Usage:**
```bash
# Balanced (recommended)
python voice_assistant_vosk_fast.py

# Very strict
python voice_assistant_vosk_fast.py --min-confidence 0.85 --min-words 3
```

**Filtered Words:**
```python
HALLUCINATION_WORDS = {
    'the', 'a', 'and', 'of', 'to', 'in', 'is', 'it', 'that'
}
```

**Output Example:**
```
🎤 hello how are you
📝 You: "hello how are you" ✓86%
💬 Assistant: "You said: hello how are you"

🚫 "the" → too_short (1 words, need 2)
🚫 "the a" → starts_with_hallucinations
```

---

### 4. Hybrid: Vosk VAD + Whisper STT (`voice_assistant_hybrid.py`) ⭐

**Best accuracy** - Uses Vosk for speech detection, Whisper for transcription.

**Architecture:**
```
Mic → Vosk (VAD only) → Buffer Audio → Whisper (STT) → Echo → Piper TTS
```

**How It Works:**
1. Vosk detects when you start speaking (real-time)
2. Audio is buffered while you speak
3. When Vosk detects silence (end of speech), buffer is sent to Whisper
4. Whisper transcribes with high accuracy
5. Result is spoken via Piper TTS

**Usage:**
```bash
# Default (Whisper small)
python voice_assistant_hybrid.py

# Faster (lower accuracy)
python voice_assistant_hybrid.py --whisper-model tiny

# Better accuracy (slower)
python voice_assistant_hybrid.py --whisper-model medium

# Custom Vosk model for VAD
python voice_assistant_hybrid.py --vosk-model models/vosk-model-small-en-us-0.15
```

**Parameters:**
- `--vosk-model`: Path to Vosk model (used only for VAD)
- `--whisper-model`: tiny/base/small/medium/large
- `--whisper-device`: cpu or cuda
- `--whisper-compute`: int8/float16/float32

**Performance:**
- Vosk VAD: ~instant (real-time detection)
- Whisper small transcription: 200-500ms
- Total latency: 300-600ms from speech end to response

**Output Example:**
```
🎤 Detecting speech...
🔄 Processing with Whisper...
📝 You: "i want to drink some tea" (350ms)
💬 Assistant: "You said: i want to drink some tea"
🔊 TTS: 150ms + 1800ms

👂 Listening…
```

**Why Hybrid is Best:**
- ✅ **No hallucinations**: Vosk filters silence before Whisper sees it
- ✅ **Excellent accuracy**: Whisper's superior transcription quality
- ✅ **Fast detection**: Vosk detects speech instantly (no delay)
- ✅ **Best of both worlds**: Speed + Accuracy

---

## 📊 Performance Comparison

| Metric | Simple Vosk | Fast Vosk | Hybrid (Vosk+Whisper) |
|--------|-------------|-----------|----------------------|
| **Latency** | ~50ms | ~50ms | ~300-500ms |
| **Accuracy** | Good | Good | **Excellent** |
| **Hallucinations** | Common | Rare | **Very Rare** |
| **CPU Usage** | Low | Low | Medium |
| **Model Size** | 40-128 MB | 40-128 MB | 500 MB - 1.5 GB |
| **Load Time** | 0.4-1.2s | 0.4-1.2s | 1.5-2.5s |

**Recommendation Matrix:**

| Use Case | Recommended Script |
|----------|-------------------|
| **Speed priority** | `voice_assistant_vosk_simple.py` |
| **Balanced (recommended)** | `voice_assistant_vosk_fast.py` |
| **Accuracy priority** | `voice_assistant_hybrid.py` ⭐ |
| **Low-power devices** | `voice_assistant_vosk_fast.py` (small model) |
| **Noisy environment** | `voice_assistant_hybrid.py` |

---

## 🔧 Troubleshooting

### Issue: Hallucinations (Random Words)

**Symptoms:**
- Transcribes "the", "a", "i" when silent
- Repeats words randomly
- Low confidence scores

**Solutions:**

1. **Use stricter filtering:**
```bash
python voice_assistant_vosk_fast.py --min-confidence 0.85 --min-words 3
```

2. **Switch to hybrid mode (best solution):**
```bash
python voice_assistant_hybrid.py
```

3. **Use larger Vosk model:**
```bash
python voice_assistant_vosk_fast.py --model models/vosk-model-en-us-0.22-lgraph
```

4. **Reduce background noise:**
- Move away from fans, AC units
- Close windows
- Use headset microphone instead of laptop mic

---

### Issue: Too Slow (Takes Too Long to Respond)

**Symptoms:**
- Long pause after speaking
- Feels unresponsive

**Solutions:**

1. **Use smaller Whisper model:**
```bash
python voice_assistant_hybrid.py --whisper-model tiny
```

2. **Use Vosk-only (fastest):**
```bash
python voice_assistant_vosk_fast.py
```

3. **Check CPU usage:**
```bash
htop  # Look for other processes consuming CPU
```

---

### Issue: No Audio Input / Not Detecting Voice

**Symptoms:**
- Nothing happens when speaking
- No partial transcription shown

**Solutions:**

1. **Check PulseAudio default source:**
```bash
# List all sources
pactl list sources short

# Set default microphone
pactl set-default-source alsa_input.pci-0000_03_00.6.HiFi__Mic1__source
```

2. **Test microphone:**
```bash
# Record 5 seconds
parecord --channels=1 --rate=16000 test.wav
# Speak for 5 seconds, then Ctrl+C

# Play back
paplay test.wav
```

3. **Check PyAudio devices:**
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"{i}: {info['name']}")
```

---

## 🗂️ Project Structure

```
vosk-voice-assistant/
├── voice_assistant_edge.py          # Original implementation (wake-word + LLM)
├── voice_assistant_vosk_simple.py   # Simple Vosk-only
├── voice_assistant_vosk_improved.py # With confidence filtering
├── voice_assistant_vosk_fast.py     # Aggressive anti-hallucination
├── voice_assistant_hybrid.py        # Vosk VAD + Whisper STT ⭐
├── requirements.txt                 # Python dependencies
├── README.md                        # Quick reference
├── DOCUMENTATION.md                 # This file (complete guide)
└── models/
    ├── vosk-model-small-en-us-0.15/
    ├── vosk-model-en-us-0.22-lgraph/
    └── piper/
        ├── en_US-lessac-medium.onnx
        └── en_US-lessac-medium.onnx.json
```

---

## 🧠 Models

### Vosk Models
Download from: https://alphacephei.com/vosk/models

**Available Models:**
1. **vosk-model-small-en-us-0.15** (40 MB)
   - Fast loading (~0.4s)
   - Good for VAD and basic transcription
   - May have hallucinations with background noise

2. **vosk-model-en-us-0.22-lgraph** (128 MB)
   - Better accuracy
   - More robust to noise
   - Slower loading (~1.2s)
   - **Recommended for production**

### Faster-Whisper Models
Automatically downloaded to `~/.cache/huggingface/hub/`

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **tiny** | ~75 MB | Fastest | Lowest | Testing, very low resources |
| **base** | ~150 MB | Fast | Decent | Speed priority |
| **small** | ~500 MB | Good | Good | **Recommended balance** ⭐ |
| **medium** | ~1.5 GB | Slow | High | Accuracy priority |
| **large** | ~3 GB | Slowest | Best | Maximum accuracy |

**Compute Types:**
- **int8**: Fastest, lowest memory (CPU) - **Recommended**
- **float16**: GPU-optimized (CUDA only)
- **float32**: Highest quality, slowest

### Piper TTS Model
Located in `models/piper/`

**Current Model:**
- `en_US-lessac-medium.onnx` (~60 MB)
- Natural female voice
- Fast synthesis (~150ms for typical sentence)
- 22.05kHz output sample rate

---

## 🎯 Architecture Inspiration

This project draws inspiration from production speech-to-text tools:

### Handy (https://github.com/cjpais/Handy)
Desktop app for speech-to-text with:
- **Architecture**: VAD first, then transcribe (what we're doing)
- **VAD**: Silero (ML-based, lightweight)
- **STT**: Whisper or Parakeet V3
- **Key insight**: Separate VAD from transcription for better control

### ChatGPT Voice Mode
- **Architecture**: WebRTC streaming to server-side VAD + Whisper
- **VAD**: Server-side (Silero or similar)
- **STT**: Whisper Large (server-side)
- **Key insight**: Real-time streaming with immediate feedback

### Our Implementation
- **Architecture**: Vosk VAD → Whisper STT (best offline approach)
- **VAD**: Vosk built-in (ML-based, works offline)
- **STT**: Faster-Whisper (CPU-optimized, offline)
- **Advantage**: Fully offline, no network required, privacy-focused

---

## 🚀 Future Improvements

### Potential Enhancements

1. **Switch to Silero VAD**
   - Lighter weight than Vosk (just VAD, not full STT)
   - More accurate silence detection
   - Used by production tools (Handy, etc.)
   - Package: `pip install silero-vad`

2. **Add Parakeet V3 Option**
   - CPU-optimized alternative to Whisper
   - ~5x realtime speed
   - Auto language detection
   - Better for low-end hardware

3. **GPU Acceleration**
   - CUDA support for Whisper (already available with `--whisper-device cuda`)
   - Metal support for macOS
   - Significant speedup on supported hardware

4. **Re-enable LLM Integration**
   - Add intent detection (TinyLlama or Qwen2.5-3B)
   - Tool calling (cart management, product search)
   - Smart responses instead of echo

5. **Wake Word Detection**
   - Optional wake-word gating ("Hey assistant")
   - Hands-free activation
   - Lower power consumption (only process after wake word)

6. **Multi-language Support**
   - Download models for other languages
   - Auto language detection (Whisper supports this)

---

## 📝 Development Notes

### Design Decisions

**Why PyAudio instead of sounddevice?**
- Better PulseAudio integration on Linux
- More reliable device routing
- Vosk documentation examples use PyAudio
- Fixes audio device issues on Linux

**Why Vosk for VAD?**
- Built-in ML-based VAD (no manual RMS thresholds)
- Works offline
- Fast and reliable
- Provides word-level confidence scores
- Lightweight for VAD-only use

**Why Faster-Whisper?**
- CPU-optimized (CTranslate2 backend)
- Quantization support (int8)
- Same accuracy as OpenAI Whisper
- Much faster inference
- Active maintenance

**Why Piper TTS?**
- High-quality neural voices
- Fast synthesis (~150ms)
- Fully offline
- ONNX format (portable)
- Better than traditional TTS (eSpeak, Festival)

**Why Hybrid Approach?**
- Separates concerns: detection vs transcription
- Vosk's VAD eliminates hallucinations from silence
- Whisper gets only real speech to transcribe
- Best balance of speed and accuracy for offline use

---

## 🎤 Audio Stack (Linux)

### Audio Flow
```
Hardware Mic → ALSA → PulseAudio → PortAudio → PyAudio → Python
```

### Important Notes
- **PyAudio** respects PulseAudio routing better than sounddevice on Linux
- Set default source: `pactl set-default-source <source-name>`
- List sources: `pactl list sources short`
- Monitor sources are loopback (not microphones)

### Audio Parameters
- **Sample rate**: 16kHz (required by Vosk and Whisper)
- **Format**: 16-bit signed integer PCM (paInt16)
- **Channels**: 1 (mono)
- **Chunk size**: 4000 samples (~250ms at 16kHz)

---

## 🙏 Acknowledgments

- **Vosk** - Alpha Cephei for excellent offline STT
- **Whisper** - OpenAI for revolutionary speech recognition
- **Faster-Whisper** - Guillaume Klein for CPU-optimized implementation
- **Piper** - Rhasspy project for high-quality TTS
- **Handy** - CJ Pais for architecture inspiration and best practices
- **Stack Overflow** - Community for Vosk real-time recognition patterns

---

## 📄 License

MIT License - Feel free to use, modify, and distribute.

---

## 💡 Key Takeaways

1. **Hybrid approach is best** for offline voice assistants
   - Vosk for fast speech detection (VAD)
   - Whisper for accurate transcription
   - Piper for natural TTS

2. **Separate VAD from transcription** for better control
   - Prevents hallucinations from silence
   - Allows mixing different engines
   - Production tools use this pattern

3. **Offline doesn't mean low quality**
   - Whisper rivals commercial services
   - Piper TTS sounds natural
   - Privacy-focused by design

4. **Multiple implementations serve different needs**
   - Speed: Vosk-only
   - Balance: Vosk with filtering
   - Accuracy: Hybrid Vosk+Whisper ⭐

---

**For questions or issues, refer to the Troubleshooting section or review the implementation that best fits your needs.**
