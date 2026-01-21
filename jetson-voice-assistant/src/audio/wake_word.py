"""
Wake Word Detection using Picovoice Porcupine
Optimized for Jetson Orin Nano (ARM64)
"""

import os
import struct
import threading
import queue
from typing import Callable, Optional
import numpy as np

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    print("⚠️  Porcupine not available. Install with: pip install pvporcupine")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available. Install with: pip install pyaudio")


class WakeWordDetector:
    """
    Wake word detection using Picovoice Porcupine
    Supports built-in keywords and custom keyword files
    """
    
    # Built-in Porcupine keywords
    BUILTIN_KEYWORDS = [
        "alexa", "americano", "blueberry", "bumblebee", "computer",
        "grapefruit", "grasshopper", "hey google", "hey siri", "jarvis",
        "ok google", "picovoice", "porcupine", "terminator"
    ]
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        keyword: str = "computer",
        custom_keyword_path: Optional[str] = None,
        sensitivity: float = 0.5,
        on_wake_word: Optional[Callable[[], None]] = None
    ):
        """
        Initialize wake word detector
        
        Args:
            access_key: Picovoice access key (get from console.picovoice.ai)
            keyword: Built-in keyword to use (if not using custom)
            custom_keyword_path: Path to custom .ppn keyword file
            sensitivity: Detection sensitivity (0.0 - 1.0)
            on_wake_word: Callback function when wake word detected
        """
        if not PORCUPINE_AVAILABLE:
            raise ImportError("Porcupine is not installed")
        
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio is not installed")
        
        self.access_key = access_key or os.environ.get("PICOVOICE_ACCESS_KEY", "")
        self.keyword = keyword
        self.custom_keyword_path = custom_keyword_path
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.on_wake_word = on_wake_word
        
        self.porcupine = None
        self.audio = None
        self.stream = None
        
        self._running = False
        self._thread = None
        self._audio_queue = queue.Queue()
    
    def _create_porcupine(self):
        """Create Porcupine instance"""
        if self.custom_keyword_path and os.path.exists(self.custom_keyword_path):
            # Use custom keyword file
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[self.custom_keyword_path],
                sensitivities=[self.sensitivity]
            )
        else:
            # Use built-in keyword
            keyword = self.keyword.lower()
            if keyword not in self.BUILTIN_KEYWORDS:
                print(f"⚠️  Unknown keyword '{keyword}', using 'computer'")
                keyword = "computer"
            
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[keyword],
                sensitivities=[self.sensitivity]
            )
        
        return self.porcupine
    
    def _create_audio_stream(self):
        """Create PyAudio stream"""
        self.audio = pyaudio.PyAudio()
        
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
            stream_callback=self._audio_callback
        )
        
        return self.stream
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - puts audio data in queue"""
        self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _detection_loop(self):
        """Main detection loop running in thread"""
        print(f"🎤 Listening for wake word: '{self.keyword}'...")
        
        while self._running:
            try:
                # Get audio data from queue
                audio_data = self._audio_queue.get(timeout=1.0)
                
                # Convert to int16 array
                pcm = struct.unpack_from(
                    "h" * self.porcupine.frame_length,
                    audio_data
                )
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print(f"🔔 Wake word detected!")
                    if self.on_wake_word:
                        self.on_wake_word()
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self._running:
                    print(f"❌ Detection error: {e}")
    
    def start(self):
        """Start wake word detection"""
        if self._running:
            return
        
        try:
            self._create_porcupine()
            self._create_audio_stream()
            
            self._running = True
            self.stream.start_stream()
            
            self._thread = threading.Thread(target=self._detection_loop, daemon=True)
            self._thread.start()
            
            print(f"✅ Wake word detector started (keyword: '{self.keyword}')")
            
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to start wake word detector: {e}")
    
    def stop(self):
        """Stop wake word detection"""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        print("🛑 Wake word detector stopped")
    
    def is_running(self) -> bool:
        """Check if detector is running"""
        return self._running
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class SimpleWakeWordDetector:
    """
    Simple wake word detector using energy-based VAD
    Fallback when Porcupine is not available
    Listens for any speech activity as a "wake word"
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 500.0,
        on_wake_word: Optional[Callable[[], None]] = None
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.on_wake_word = on_wake_word
        
        self.audio = None
        self.stream = None
        self._running = False
        self._thread = None
    
    def _detection_loop(self):
        """Detection loop using energy-based VAD"""
        chunk_size = 1024
        silence_chunks = 0
        max_silence_chunks = 10
        speech_detected = False
        
        print("🎤 Listening for speech (simple VAD)...")
        
        while self._running:
            try:
                audio_data = self.stream.read(chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate energy
                energy = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
                
                if energy > self.energy_threshold:
                    if not speech_detected:
                        speech_detected = True
                        print("🔔 Speech detected!")
                        if self.on_wake_word:
                            self.on_wake_word()
                    silence_chunks = 0
                else:
                    silence_chunks += 1
                    if silence_chunks > max_silence_chunks:
                        speech_detected = False
                        
            except Exception as e:
                if self._running:
                    print(f"❌ Detection error: {e}")
    
    def start(self):
        """Start detection"""
        if self._running:
            return
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        
        print("✅ Simple wake word detector started")
    
    def stop(self):
        """Stop detection"""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("🛑 Simple wake word detector stopped")


def create_wake_word_detector(
    access_key: Optional[str] = None,
    keyword: str = "computer",
    sensitivity: float = 0.5,
    on_wake_word: Optional[Callable[[], None]] = None,
    use_simple_fallback: bool = True
) -> WakeWordDetector:
    """
    Factory function to create appropriate wake word detector
    Falls back to simple VAD if Porcupine unavailable
    """
    if PORCUPINE_AVAILABLE and access_key:
        return WakeWordDetector(
            access_key=access_key,
            keyword=keyword,
            sensitivity=sensitivity,
            on_wake_word=on_wake_word
        )
    elif use_simple_fallback:
        print("⚠️  Using simple VAD fallback (Porcupine not configured)")
        return SimpleWakeWordDetector(on_wake_word=on_wake_word)
    else:
        raise RuntimeError("Porcupine not available and fallback disabled")


if __name__ == "__main__":
    import time
    
    def on_wake():
        print(">>> Wake word callback triggered!")
    
    # Test with simple detector (no API key needed)
    detector = SimpleWakeWordDetector(on_wake_word=on_wake)
    
    try:
        detector.start()
        print("Listening... Press Ctrl+C to stop")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()
