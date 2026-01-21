import subprocess
import tempfile
import numpy as np
import wave
import json
import os

class WhisperASR:
    def __init__(self, model_size="base", whisper_cpp_path="lib/whisper.cpp"):
        self.whisper_cpp_path = whisper_cpp_path
        self.model_size = model_size
        self.model_path = f"{whisper_cpp_path}/models/ggml-{model_size}.bin"
        self.executable = f"{whisper_cpp_path}/main"
        
        # Verify paths exist
        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"whisper.cpp executable not found at {self.executable}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")
            
    def transcribe(self, audio_data, language="en"):
        """Transcribe audio using whisper.cpp"""
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            self._save_wav(tmp_wav.name, audio_data, 16000)
            
            try:
                # Run whisper.cpp
                cmd = [
                    self.executable,
                    "-m", self.model_path,
                    "-f", tmp_wav.name,
                    "-l", language,
                    "--no-timestamps",
                    "-nt"  # No timestamps
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Parse output - whisper.cpp outputs directly to stdout
                    text = result.stdout.strip()
                    # Remove any whisper.cpp metadata lines
                    lines = [l for l in text.split('\n') if not l.startswith('[')]
                    return ' '.join(lines).strip()
                else:
                    print(f"Whisper error: {result.stderr}")
                    return ""
            finally:
                os.unlink(tmp_wav.name)
                
    def _save_wav(self, filename, audio_data, sample_rate):
        """Save audio data to WAV file"""
        # Ensure audio is in correct format
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
            
        with wave.open(filename, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())