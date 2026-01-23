# Piper TTS Integration

Production-ready text-to-speech using [Piper](https://github.com/rhasspy/piper) for natural voice output.

## Features

- ✅ **Natural Speech Formatting**: Converts "$12.95" → "12 dollars and 95 cents"
- ✅ **Context-Aware**: Different formatting for cart, menu, confirmations, options
- ✅ **Fast & Lightweight**: Piper runs efficiently on Jetson and other embedded devices
- ✅ **Auto-Detection**: Automatically detects response type from content

## Installation

### 1. Install Piper

```bash
# Install piper-tts
pip install piper-tts

# Or build from source for Jetson:
git clone https://github.com/rhasspy/piper.git
cd piper/src/python
pip install -e .
```

### 2. Download Voice Model

```bash
# Create models directory
mkdir -p models/piper

# Download English voice (lessac-medium recommended)
cd models/piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json

# Other voice options:
# - en_US-lessac-low.onnx (faster, lower quality)
# - en_US-lessac-high.onnx (slower, higher quality)
# - en_US-libritts-high.onnx (different voice)
```

### 3. Test Audio Output

```bash
# Test piper installation
echo "Hello, this is a test" | piper --model models/piper/en_US-lessac-medium.onnx --output-raw | aplay -r 22050 -f S16_LE -t raw -

# Test speech formatter
python3 test_speech_formatting.py
```

## Usage

### Option 1: Enable TTS in Assistant (Recommended)

Update `run.py` to use Piper TTS:

```python
from src.output.piper_tts import ProductionTTSEngine

# Initialize assistant
assistant = TextAssistant(tenant_id="demo_restaurant")

# Replace text output with TTS
assistant.output_engine = ProductionTTSEngine(
    model_path="models/piper/en_US-lessac-medium.onnx",
    enable_formatting=True
)

# Run assistant
assistant.run()
```

### Option 2: Manual TTS Control

```python
from src.output.piper_tts import PiperTTS, ProductionTTSEngine
from src.output.speech_formatter import SpeechFormatter

# Basic Piper TTS
tts = PiperTTS("models/piper/en_US-lessac-medium.onnx")
tts.speak("Hello, welcome to Demo Restaurant!")

# With speech formatting
formatter = SpeechFormatter()
text = "Your total is $12.95 including tax."
formatted = formatter.format_for_speech(text)
tts.speak(formatted)  # "Your total is 12 dollars and 95 cents including tax"

# Production TTS (TTS + Formatter combined)
engine = ProductionTTSEngine()
engine.speak("Your total is $12.95", response_type='cart')
```

## Speech Formatting Examples

The `SpeechFormatter` makes responses natural for TTS:

| Raw Text | Formatted for TTS |
|----------|-------------------|
| `"Welcome! 🤖"` | `"Welcome!"` |
| `"Total: $12.95"` | `"Total: 12 dollars and 95 cents"` |
| `"• Coffee - $3.50"` | `"Coffee, 3.50 dollars"` |
| `"What size? (S, M, L)"` | `"What size? You can choose S, M, or L"` |
| `"(Size: Large)"` | `"with Large size"` |

### Auto-Detection

The assistant automatically detects response types:

```python
# Automatically detected as 'cart' response
response = "Your cart: Burger - $12.95. Total: $12.95"
assistant.respond(response)  # Auto-formats currency

# Automatically detected as 'menu' response  
response = "Our pizza include: • Margherita ($10.99)"
assistant.respond(response)  # Auto-formats menu listing

# Automatically detected as 'option' response
response = "What size? (Small, Medium, Large)"
assistant.respond(response)  # "You can choose Small, Medium, or Large"
```

## Performance

### Piper Models Comparison

| Model | Size | Speed (RTF) | Quality | Recommended |
|-------|------|-------------|---------|-------------|
| `en_US-lessac-low` | ~15MB | 0.1-0.2 | Good | Embedded devices |
| `en_US-lessac-medium` | ~26MB | 0.2-0.4 | Very Good | **Recommended** |
| `en_US-lessac-high` | ~63MB | 0.5-0.8 | Excellent | High-end devices |

RTF (Real-Time Factor): Lower is faster. 0.2 means 1 second of audio generates in 0.2 seconds.

### Jetson Performance

- **Jetson Orin Nano**: ~0.3 RTF with medium model (real-time)
- **Jetson Xavier NX**: ~0.4 RTF with medium model (real-time)
- **Jetson Nano**: ~0.8 RTF with low model (near real-time)

## Troubleshooting

### No Audio Output

```bash
# Check audio devices
aplay -l

# Test audio
speaker-test -t wav -c 2

# Set default audio device
export AUDIODEV=hw:0,0  # Adjust to your device
```

### Piper Not Found

```bash
# Check piper installation
which piper

# Install manually
pip install piper-tts --user
export PATH="$HOME/.local/bin:$PATH"
```

### Model Not Found

```bash
# Check model exists
ls -lh models/piper/

# Re-download if missing
cd models/piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json
```

### Choppy Audio

1. Use a smaller model (`en_US-lessac-low.onnx`)
2. Reduce sample rate to 16000 Hz
3. Enable audio buffering:
   ```python
   tts = PiperTTS(model_path="...", sample_rate=16000)
   ```

## API Reference

### PiperTTS

```python
class PiperTTS:
    def __init__(
        self, 
        model_path: str = "models/piper/en_US-lessac-medium.onnx",
        output_device: Optional[str] = None,
        sample_rate: int = 22050
    )
    
    def speak(self, text: str, blocking: bool = True) -> None
    def speak_to_file(self, text: str, output_path: str) -> None
```

### ProductionTTSEngine

```python
class ProductionTTSEngine:
    def __init__(
        self,
        model_path: str = "models/piper/en_US-lessac-medium.onnx",
        enable_formatting: bool = True
    )
    
    def speak(self, text: str, response_type: str = 'general') -> None
    def enable(self) -> None
    def disable(self) -> None
    def toggle(self) -> bool
```

### SpeechFormatter

```python
class SpeechFormatter:
    @staticmethod
    def format_for_speech(text: str) -> str
    
    @staticmethod
    def format_cart_for_speech(cart_text: str) -> str
    
    @staticmethod
    def format_menu_for_speech(menu_text: str) -> str
    
    @staticmethod
    def format_confirmation_for_speech(text: str) -> str
    
    @staticmethod
    def format_option_prompt(text: str) -> str
```

## Next Steps

1. **Test TTS**: Run `test_speech_formatting.py` to see formatted output
2. **Enable in Assistant**: Update `run.py` to use `ProductionTTSEngine`
3. **Choose Voice**: Download different voices from Piper releases
4. **Optimize**: Adjust model size based on your device performance

## Resources

- [Piper GitHub](https://github.com/rhasspy/piper)
- [Available Voices](https://github.com/rhasspy/piper/releases)
- [Piper Documentation](https://github.com/rhasspy/piper/blob/master/README.md)
