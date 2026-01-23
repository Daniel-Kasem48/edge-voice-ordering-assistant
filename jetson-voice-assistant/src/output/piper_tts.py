"""
Piper TTS Engine - Production-ready text-to-speech with Piper
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional


class PiperTTS:
    """Piper text-to-speech engine with natural voice tuning"""
    
    def __init__(
        self,
        model_path: str = "models/piper/en_US-amy-medium.onnx",
        output_device: Optional[str] = None,
        sample_rate: int = 22050,  # amy uses 22kHz (higher quality)
        # Voice tuning for more natural speech
        length_scale: float = 1.05,  # Slightly slower = more natural (1.0 = normal)
        noise_scale: float = 0.667,  # Voice variation (higher = more expressive)
        noise_w_scale: float = 0.8,  # Phoneme width variation
        sentence_silence: float = 0.35  # Pause between sentences
    ):
        """
        Initialize Piper TTS with natural voice settings
        
        Args:
            model_path: Path to Piper ONNX model
            output_device: Audio output device (None for default)
            sample_rate: Audio sample rate
            length_scale: Speed control (>1 = slower, <1 = faster)
            noise_scale: Voice expressiveness (0.333-1.0)
            noise_w_scale: Phoneme duration variation
            sentence_silence: Seconds of pause after sentences
        """
        self.model_path = Path(model_path)
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.sentence_silence = sentence_silence
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found at {model_path}. "
                f"Download from: https://github.com/rhasspy/piper/releases"
            )
        
        # Check if piper is installed (--help works, --version doesn't)
        try:
            result = subprocess.run(['piper', '--help'], capture_output=True)
            # piper --help returns exit code 0
        except FileNotFoundError:
            raise RuntimeError(
                "Piper not installed. Install with: pip install piper-tts"
            )
    
    def speak(self, text: str, blocking: bool = True) -> None:
        """
        Speak text using Piper TTS with natural voice settings
        
        Args:
            text: Text to speak
            blocking: Wait for speech to complete (default True)
        """
        if not text or not text.strip():
            return
        
        # Build piper command with voice tuning
        cmd = [
            'piper',
            '--model', str(self.model_path),
            '--length-scale', str(self.length_scale),
            '--noise-scale', str(self.noise_scale),
            '--noise-w-scale', str(self.noise_w_scale),
            '--sentence-silence', str(self.sentence_silence),
            '--output-raw'  # Output raw audio to stdout
        ]
        
        if self.output_device:
            cmd.extend(['--output_device', self.output_device])
        
        try:
            # Pipe text to piper and play with aplay
            piper_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Play audio with aplay
            aplay_process = subprocess.Popen(
                ['aplay', '-r', str(self.sample_rate), '-f', 'S16_LE', '-t', 'raw', '-'],
                stdin=piper_process.stdout,
                stderr=subprocess.PIPE
            )
            
            # Send text to piper
            piper_process.stdin.write(text.encode('utf-8'))
            piper_process.stdin.close()
            
            if blocking:
                # Wait for completion
                piper_process.wait()
                aplay_process.wait()
            
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def speak_to_file(self, text: str, output_path: str) -> None:
        """
        Speak text and save to WAV file
        
        Args:
            text: Text to speak
            output_path: Path to save WAV file
        """
        if not text or not text.strip():
            return
        
        cmd = [
            'piper',
            '--model', str(self.model_path),
            '--output_file', output_path
        ]
        
        try:
            process = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"TTS Error: {e.stderr.decode()}")
            raise


class ProductionTTSEngine:
    """
    Production-ready TTS engine with Piper and speech formatting
    Uses natural voice tuning for human-like speech
    """
    
    def __init__(
        self,
        model_path: str = "models/piper/en_US-amy-medium.onnx",
        enable_formatting: bool = True,
        # Natural voice settings (passed to PiperTTS)
        length_scale: float = 1.05,     # 5% slower for natural pacing
        noise_scale: float = 0.667,     # More voice expressiveness
        noise_w_scale: float = 0.8,     # Varied phoneme timing
        sentence_silence: float = 0.35, # Natural pauses between sentences
        sample_rate: int = 22050        # Amy uses 22kHz
    ):
        """
        Initialize production TTS engine with natural voice
        
        Args:
            model_path: Path to Piper model
            enable_formatting: Enable speech formatting (removes $, bullets, etc.)
            length_scale: Speed control (>1 = slower, more natural)
            noise_scale: Voice expressiveness (0.333-1.0)
            noise_w_scale: Phoneme duration variation
            sentence_silence: Seconds of pause after sentences
        """
        from src.output.speech_formatter import SpeechFormatter
        
        self.tts = PiperTTS(
            model_path=model_path,
            sample_rate=sample_rate,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            sentence_silence=sentence_silence
        )
        self.formatter = SpeechFormatter() if enable_formatting else None
        self.enabled = True
    
    def speak(self, text: str, response_type: str = 'general') -> None:
        """
        Speak text with automatic formatting
        
        Args:
            text: Raw text from assistant
            response_type: Type of response (general, cart, menu, confirmation, option)
        """
        if not self.enabled:
            return
        
        # Format for speech if formatter enabled
        if self.formatter:
            if response_type == 'cart':
                speech_text = self.formatter.format_cart_for_speech(text)
            elif response_type == 'menu':
                speech_text = self.formatter.format_menu_for_speech(text)
            elif response_type == 'confirmation':
                speech_text = self.formatter.format_confirmation_for_speech(text)
            elif response_type == 'option':
                speech_text = self.formatter.format_option_prompt(text)
            else:
                speech_text = self.formatter.format_for_speech(text)
        else:
            speech_text = text
        
        # Speak
        self.tts.speak(speech_text)
    
    def enable(self):
        """Enable TTS"""
        self.enabled = True
    
    def disable(self):
        """Disable TTS"""
        self.enabled = False
    
    def toggle(self):
        """Toggle TTS on/off"""
        self.enabled = not self.enabled
        return self.enabled
