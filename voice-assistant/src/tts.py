import subprocess
import numpy as np
import wave
import tempfile
import os
import shutil

class PiperTTS:
    def __init__(self, model_path="models/piper/en_US-amy-medium.onnx"):
        self.model_path = model_path
        self.config_path = model_path + ".json"
        
        # Check if piper is installed
        if not shutil.which("piper"):
            raise RuntimeError("Piper not found. Install with: pip install piper-tts")
            
        # Verify model files exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model not found at {self.model_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Piper config not found at {self.config_path}")
            
    def synthesize(self, text):
        """Synthesize speech from text"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            cmd = [
                "piper",
                "--model", self.model_path,
                "--config", self.config_path,
                "--output_file", tmp_wav.name
            ]
            
            try:
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, text=True)
                                         
                stdout, stderr = process.communicate(input=text)
                
                if process.returncode == 0:
                    # Read the WAV file
                    with wave.open(tmp_wav.name, 'rb') as wav:
                        frames = wav.readframes(wav.getnframes())
                        sample_rate = wav.getframerate()
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
                    return audio, sample_rate
                else:
                    print(f"TTS error: {stderr}")
                    return None, None
            finally:
                if os.path.exists(tmp_wav.name):
                    os.unlink(tmp_wav.name)