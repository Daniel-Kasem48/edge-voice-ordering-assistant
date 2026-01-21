"""
Speech-to-Text Engine using whisper.cpp
Optimized for Jetson Orin Nano with GPU acceleration
"""

import os
import subprocess
import tempfile
import wave
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    text: str
    duration_seconds: float
    processing_time_seconds: float
    confidence: float = 1.0
    language: str = "en"


class WhisperCppEngine:
    """
    Speech-to-text using whisper.cpp
    Optimized for Jetson with CUDA support
    """
    
    # Available models (smallest to largest)
    MODELS = {
        "tiny": "ggml-tiny.bin",
        "tiny.en": "ggml-tiny.en.bin",
        "base": "ggml-base.bin",
        "base.en": "ggml-base.en.bin",
        "small": "ggml-small.bin",
        "small.en": "ggml-small.en.bin",
        "medium": "ggml-medium.bin",
        "medium.en": "ggml-medium.en.bin",
        "large": "ggml-large.bin"
    }
    
    def __init__(
        self,
        model_path: str = "models/whisper/ggml-base.en.bin",
        whisper_cpp_path: str = "whisper.cpp/main",
        use_gpu: bool = True,
        language: str = "en",
        threads: int = 4,
        beam_size: int = 5
    ):
        """
        Initialize whisper.cpp engine
        
        Args:
            model_path: Path to ggml model file
            whisper_cpp_path: Path to whisper.cpp main executable
            use_gpu: Enable CUDA acceleration
            language: Language code (en, es, fr, etc.)
            threads: Number of CPU threads
            beam_size: Beam search size for decoding
        """
        self.model_path = Path(model_path)
        self.whisper_cpp_path = Path(whisper_cpp_path)
        self.use_gpu = use_gpu
        self.language = language
        self.threads = threads
        self.beam_size = beam_size
        
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate whisper.cpp installation"""
        if not self.whisper_cpp_path.exists():
            raise FileNotFoundError(
                f"whisper.cpp executable not found at {self.whisper_cpp_path}\n"
                "Build whisper.cpp with CUDA support:\n"
                "  git clone https://github.com/ggerganov/whisper.cpp\n"
                "  cd whisper.cpp && make WHISPER_CUDA=1"
            )
        
        if not self.model_path.exists():
            print(f"⚠️  Model not found at {self.model_path}")
            print("Download with: ./models/download-ggml-model.sh base.en")
    
    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to WAV file (16kHz, mono, 16-bit)
            
        Returns:
            TranscriptionResult with transcribed text
        """
        start_time = time.time()
        
        # Build command
        cmd = [
            str(self.whisper_cpp_path),
            "-m", str(self.model_path),
            "-f", audio_path,
            "-l", self.language,
            "-t", str(self.threads),
            "--beam-size", str(self.beam_size),
            "--no-timestamps",
            "-otxt"  # Output format: text only
        ]
        
        if self.use_gpu:
            cmd.append("--gpu")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"❌ whisper.cpp error: {result.stderr}")
                return TranscriptionResult(
                    text="",
                    duration_seconds=0,
                    processing_time_seconds=time.time() - start_time
                )
            
            # Parse output
            text = result.stdout.strip()
            
            # Get audio duration
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
            
            return TranscriptionResult(
                text=text,
                duration_seconds=duration,
                processing_time_seconds=time.time() - start_time,
                language=self.language
            )
            
        except subprocess.TimeoutExpired:
            return TranscriptionResult(
                text="",
                duration_seconds=0,
                processing_time_seconds=time.time() - start_time
            )
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe audio data directly
        
        Args:
            audio_data: Numpy array of audio samples
            sample_rate: Sample rate (should be 16000)
            
        Returns:
            TranscriptionResult with transcribed text
        """
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Ensure correct format
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Write WAV file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            # Transcribe
            return self.transcribe_file(temp_path)
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)


class AudioRecorder:
    """
    Audio recorder with Voice Activity Detection (VAD)
    Records speech until silence is detected
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        vad_aggressiveness: int = 2,
        silence_threshold_ms: int = 800,
        max_recording_seconds: float = 10.0
    ):
        """
        Initialize audio recorder
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            vad_aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            silence_threshold_ms: Silence duration to stop recording (ms)
            max_recording_seconds: Maximum recording duration
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice not installed")
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold_ms = silence_threshold_ms
        self.max_recording_seconds = max_recording_seconds
        
        # Initialize VAD
        if WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
        else:
            self.vad = None
            print("⚠️  WebRTC VAD not available, using energy-based detection")
    
    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech"""
        if self.vad:
            try:
                return self.vad.is_speech(audio_chunk, self.sample_rate)
            except:
                return True
        else:
            # Fallback: energy-based detection
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            return energy > 500
    
    def record_speech(self) -> Tuple[np.ndarray, float]:
        """
        Record speech with VAD
        
        Returns:
            Tuple of (audio_data, duration_seconds)
        """
        print("🎙️  Recording... (speak now)")
        
        chunk_duration_ms = 30  # 30ms chunks for VAD
        chunk_samples = int(self.sample_rate * chunk_duration_ms / 1000)
        
        recorded_chunks = []
        silence_chunks = 0
        max_silence_chunks = int(self.silence_threshold_ms / chunk_duration_ms)
        max_chunks = int(self.max_recording_seconds * 1000 / chunk_duration_ms)
        
        speech_started = False
        chunk_count = 0
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal silence_chunks, speech_started, chunk_count
            
            if status:
                print(f"⚠️  Audio status: {status}")
            
            audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            is_speech = self._is_speech(audio_bytes)
            
            if is_speech:
                speech_started = True
                silence_chunks = 0
            else:
                if speech_started:
                    silence_chunks += 1
            
            if speech_started:
                recorded_chunks.append(indata.copy())
            
            chunk_count += 1
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=chunk_samples,
            callback=audio_callback
        ):
            while True:
                sd.sleep(chunk_duration_ms)
                
                # Stop conditions
                if speech_started and silence_chunks >= max_silence_chunks:
                    print("🔇 Silence detected, stopping...")
                    break
                
                if chunk_count >= max_chunks:
                    print("⏱️  Max duration reached, stopping...")
                    break
        
        if not recorded_chunks:
            return np.array([], dtype=np.float32), 0.0
        
        # Concatenate all chunks
        audio_data = np.concatenate(recorded_chunks, axis=0)
        duration = len(audio_data) / self.sample_rate
        
        print(f"✅ Recorded {duration:.2f} seconds of audio")
        
        return audio_data, duration


class STTEngine:
    """
    High-level Speech-to-Text engine
    Combines recording and transcription
    """
    
    def __init__(
        self,
        model_path: str = "models/whisper/ggml-base.en.bin",
        whisper_cpp_path: str = "whisper.cpp/main",
        use_gpu: bool = True,
        language: str = "en"
    ):
        self.whisper = WhisperCppEngine(
            model_path=model_path,
            whisper_cpp_path=whisper_cpp_path,
            use_gpu=use_gpu,
            language=language
        )
        
        self.recorder = AudioRecorder()
    
    def listen_and_transcribe(self) -> TranscriptionResult:
        """
        Record speech and transcribe it
        
        Returns:
            TranscriptionResult with transcribed text
        """
        # Record audio
        audio_data, duration = self.recorder.record_speech()
        
        if duration < 0.3:
            return TranscriptionResult(
                text="",
                duration_seconds=duration,
                processing_time_seconds=0
            )
        
        # Convert to int16 for whisper
        audio_int16 = (audio_data[:, 0] * 32767).astype(np.int16)
        
        # Transcribe
        return self.whisper.transcribe_audio(audio_int16)


# Alternative: Use faster-whisper (Python-native, CUDA-accelerated)
class FasterWhisperEngine:
    """
    Alternative STT using faster-whisper Python library
    Better integration, but requires more memory
    """
    
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        Initialize faster-whisper engine
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu)
            compute_type: Compute type (float16, int8, float32)
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with:\n"
                "pip install faster-whisper"
            )
        
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.recorder = AudioRecorder()
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio data"""
        start_time = time.time()
        
        # faster-whisper expects float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32767.0
        
        segments, info = self.model.transcribe(audio_data, language="en")
        
        text = " ".join([segment.text for segment in segments])
        
        return TranscriptionResult(
            text=text.strip(),
            duration_seconds=info.duration,
            processing_time_seconds=time.time() - start_time,
            language=info.language
        )
    
    def listen_and_transcribe(self) -> TranscriptionResult:
        """Record and transcribe speech"""
        audio_data, duration = self.recorder.record_speech()
        
        if duration < 0.3:
            return TranscriptionResult(
                text="",
                duration_seconds=duration,
                processing_time_seconds=0
            )
        
        return self.transcribe_audio(audio_data[:, 0])


if __name__ == "__main__":
    # Test audio recording
    print("Testing audio recording with VAD...")
    
    recorder = AudioRecorder(
        silence_threshold_ms=800,
        max_recording_seconds=10.0
    )
    
    audio, duration = recorder.record_speech()
    
    print(f"Recorded audio shape: {audio.shape}")
    print(f"Duration: {duration:.2f}s")
    
    # Save for testing
    if len(audio) > 0 and SOUNDFILE_AVAILABLE:
        sf.write("test_recording.wav", audio, 16000)
        print("Saved to test_recording.wav")
