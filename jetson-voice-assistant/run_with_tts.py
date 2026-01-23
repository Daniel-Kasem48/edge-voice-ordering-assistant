#!/usr/bin/env python3
"""
Run assistant with Piper TTS enabled
Speaks responses through audio output
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Change to script directory so relative paths work
os.chdir(Path(__file__).parent)

# Check if piper model exists (prefer amy, fallback to lessac)
model_path = "models/piper/en_US-amy-medium.onnx"
sample_rate = 22050
if not os.path.exists(model_path):
    model_path = "models/piper/en_US-lessac-medium.onnx"
    sample_rate = 16000
    
if not os.path.exists(model_path):
    print("❌ Piper model not found!")
    print(f"   Expected: models/piper/en_US-amy-medium.onnx")
    print()
    print("To download:")
    print("  mkdir -p models/piper")
    print("  cd models/piper")
    print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx")
    print("  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json")
    sys.exit(1)

from output.piper_tts import PiperTTS
from output.speech_formatter import SpeechFormatter
from assistant import TextAssistant


class TTSOutputEngine:
    """Output engine that prints AND speaks via Piper TTS"""
    
    def __init__(self, model_path: str, sample_rate: int = 22050, prefix: str = "🤖 "):
        self.prefix = prefix
        self.tts = PiperTTS(
            model_path=model_path,
            sample_rate=sample_rate,
            length_scale=1.05,      # Slightly slower for natural pacing
            noise_scale=0.667,      # Expressive voice
            noise_w_scale=0.8,      # Varied phoneme timing
            sentence_silence=0.35   # Natural pauses
        )
        self.formatter = SpeechFormatter()
        self.tts_enabled = True
    
    def speak(self, text: str):
        """Print text and speak it via TTS"""
        # Print to console
        print(f"{self.prefix}{text}")
        
        # Speak via TTS if enabled
        if self.tts_enabled and text.strip():
            try:
                # Format for natural speech
                speech_text = self.formatter.format_for_speech(text)
                self.tts.speak(speech_text)
            except Exception as e:
                print(f"  [TTS Error: {e}]")
    
    def toggle_tts(self) -> bool:
        """Toggle TTS on/off"""
        self.tts_enabled = not self.tts_enabled
        return self.tts_enabled


def main():
    """Run assistant with TTS"""
    print("="*60)
    print("🎤 Restaurant Assistant with Piper TTS")
    print("="*60)
    
    # Use amy (natural voice) or fallback to lessac
    print(f"Using model: {model_path}")
    print(f"Sample rate: {sample_rate}Hz")
    
    # Initialize assistant
    assistant = TextAssistant(tenant_id="demo_restaurant")
    
    # Replace output engine with TTS-enabled engine
    try:
        tts_output = TTSOutputEngine(model_path, sample_rate=sample_rate)
        assistant.output_engine = tts_output
        print("✅ TTS engine ready - responses will be spoken!")
        print()
    except Exception as e:
        print(f"⚠️  TTS initialization failed: {e}")
        print("   Falling back to text-only mode")
        print()
    
    # Run assistant
    assistant.run()


if __name__ == "__main__":
    main()
