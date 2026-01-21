#!/usr/bin/env python3
"""
Voice assistant using Silero VAD + Faster-Whisper STT.

This is the production-grade approach used by tools like Handy:
- Silero VAD: Lightweight, accurate speech detection (ML-based)
- Faster-Whisper: High-quality transcription
- Piper: Fast TTS

Advantages over Vosk VAD:
- Silero is purpose-built for VAD (lighter, more accurate)
- Better silence detection
- No hallucinations from background noise
- Used by production speech-to-text tools
"""
import pyaudio
import time
import os
import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice
import sounddevice as sd
import requests
from pathlib import Path

class SileroVAD:
    """Silero VAD using torch hub (official method)"""
    
    def __init__(self, threshold=0.5, sample_rate=16000):
        """
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            sample_rate: Audio sample rate (must be 16000 for Silero)
        """
        import torch
        self.threshold = threshold
        self.sample_rate = sample_rate
        
        # Load Silero VAD from torch hub
        print("📥 Loading Silero VAD from torch.hub...")
        torch.set_num_threads(1)  # Optimize for CPU
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False  # Use JIT model
        )
        self.model.eval()
        
        # Get utilities
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        # Reset states
        self._reset_states()
        
        # State tracking for start/end detection
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        
        # Parameters (in samples at 16kHz)
        self.min_silence_samples = int(0.5 * sample_rate)  # 500ms
        self.min_speech_samples = int(0.25 * sample_rate)  # 250ms
        self.speech_pad_samples = int(0.03 * sample_rate)  # 30ms
    
    def _reset_states(self):
        """Reset model states"""
        import torch
        self._h = torch.zeros((2, 1, 64))
        self._c = torch.zeros((2, 1, 64))
    
    def __call__(self, audio_chunk):
        """
        Process audio chunk and detect speech.
        
        Args:
            audio_chunk: numpy array of float32 audio samples [-1, 1]
        
        Returns:
            (speech_prob, dict) where dict has 'start' or 'end' key if speech state changed
        """
        import torch
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk)
        
        # Run VAD
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        # Track current sample position
        self.current_sample += len(audio_chunk)
        
        # State machine for speech start/end detection
        result = None
        if speech_prob >= self.threshold:
            if not self.triggered:
                # Speech started
                self.triggered = True
                result = {'start': self.current_sample}
            self.temp_end = 0
        else:
            if self.triggered:
                # Potential speech end, but wait for min_silence_duration
                if self.temp_end == 0:
                    self.temp_end = self.current_sample
                
                if self.current_sample - self.temp_end >= self.min_silence_samples:
                    # Speech ended
                    self.triggered = False
                    temp_end = self.temp_end
                    self.temp_end = 0
                    result = {'end': temp_end}
        
        return speech_prob, result


class VoiceAssistantSileroWhisper:
    def __init__(self, 
                 whisper_model="small",
                 whisper_device="cpu",
                 whisper_compute="int8",
                 vad_threshold=0.5,
                 debug=False):
        """
        Args:
            whisper_model: Faster-Whisper model size (tiny/base/small/medium/large)
            whisper_device: cpu or cuda
            whisper_compute: int8, float16, float32
            vad_threshold: Silero VAD threshold (0.0-1.0, default 0.5)
            debug: Print debug info (VAD probabilities, audio levels)
        """
        print("="*60)
        print("🤖 VOICE ASSISTANT (Silero VAD + Whisper STT)")
        print("="*60)
        
        self.debug = debug
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Audio setup
        self.sample_rate = 16000
        
        # Load Silero VAD
        print("\n📡 Loading Silero VAD (torch.hub)...")
        start = time.time()
        self.vad = SileroVAD(threshold=vad_threshold, sample_rate=self.sample_rate)
        print(f"✅ Silero VAD loaded in {time.time()-start:.2f}s")
        
        # VAD parameters
        self.vad_threshold = vad_threshold
        
        # Load Faster-Whisper (for transcription)
        print(f"\n🎯 Loading Faster-Whisper ({whisper_model}, {whisper_device}, {whisper_compute})...")
        start = time.time()
        self.whisper_model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute
        )
        print(f"✅ Whisper loaded in {time.time()-start:.2f}s")
        
        # Load Piper TTS
        print("\n🔊 Loading Piper TTS...")
        start = time.time()
        piper_path = os.path.join(script_dir, "models/piper/en_US-lessac-medium.onnx")
        self.piper_voice = PiperVoice.load(piper_path)
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        print(f"\n⚙️  Settings:")
        print(f"   - VAD threshold: {self.vad_threshold}")
        print(f"   - Sample rate: {self.sample_rate} Hz")
        print(f"   - Whisper model: {whisper_model}")
        print("\n✅ Ready!")
    
    def _transcribe_audio(self, audio_data):
        """
        Transcribe audio using Faster-Whisper.
        
        Args:
            audio_data: numpy array of int16 audio samples
        
        Returns:
            transcribed text or None
        """
        # Convert int16 to float32 normalized to [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Transcribe with Whisper
        segments, info = self.whisper_model.transcribe(
            audio_float,
            language="en",
            beam_size=3,
            best_of=3,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            )
        )
        
        # Collect all text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        text = " ".join(text_parts).strip()
        return text if text else None
    
    def speak(self, text):
        """Synthesize and play using Piper"""
        start = time.time()
        
        audio_chunks = []
        for chunk in self.piper_voice.synthesize(text):
            audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        
        synth_time = int((time.time() - start) * 1000)
        
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
            audio_float = audio_data.astype(np.float32) / 32767
            
            play_start = time.time()
            sd.play(audio_float, 22050)
            sd.wait()
            play_time = int((time.time() - play_start) * 1000)
            
            return synth_time, play_time
        
        return synth_time, 0
    
    def run(self):
        print("\n🎤 Silero VAD detects speech → Whisper transcribes")
        print("Speak naturally. Ctrl+C to exit.\n")
        
        # Setup PyAudio
        p = pyaudio.PyAudio()
        chunk_size = 1536  # Silero works with 512, 1536, or 4096 samples (96ms chunks)
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
            stream.start_stream()
            
            print("👂 Listening…\n")
            
            # State tracking
            speech_buffer = []
            is_speaking = False
            speech_start_time = None
            chunk_count = 0
            
            while True:
                # Read audio chunk
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Convert to float32 for VAD
                audio_float32 = audio_chunk.astype(np.float32) / 32768.0
                
                # Calculate RMS level (for debug)
                rms = np.sqrt(np.mean(audio_float32 ** 2))
                
                # Run VAD
                speech_prob, speech_dict = self.vad(audio_float32)
                
                # Debug output every 20 chunks (~2 seconds)
                if self.debug and chunk_count % 20 == 0:
                    print(f"[DEBUG] RMS: {rms:.4f} | VAD prob: {speech_prob:.3f} | Threshold: {self.vad.threshold}")
                
                chunk_count += 1
                
                if speech_dict:
                    if 'start' in speech_dict:
                        # Speech started
                        is_speaking = True
                        speech_start_time = time.time()
                        speech_buffer = []
                        print("\r🎤 Speaking...", end='', flush=True)
                    
                    elif 'end' in speech_dict:
                        # Speech ended
                        is_speaking = False
                        
                        if speech_buffer:
                            print("\r🔄 Processing with Whisper...          ", flush=True)
                            
                            # Concatenate all buffered audio
                            full_audio = np.concatenate(speech_buffer)
                            
                            # Calculate speech duration
                            speech_duration = len(full_audio) / self.sample_rate
                            
                            # Only process if speech was longer than 0.3 seconds
                            if speech_duration >= 0.3:
                                # Transcribe with Whisper
                                start = time.time()
                                text = self._transcribe_audio(full_audio)
                                whisper_time = int((time.time() - start) * 1000)
                                
                                if text:
                                    print(f"📝 You: \"{text}\"")
                                    print(f"   ⏱️  Speech: {speech_duration:.1f}s | Whisper: {whisper_time}ms")
                                    
                                    # Echo response
                                    response = f"You said: {text}"
                                    print(f"💬 Assistant: \"{response}\"")
                                    
                                    synth_ms, play_ms = self.speak(response)
                                    print(f"🔊 TTS: {synth_ms}ms + {play_ms}ms\n")
                                else:
                                    print("🚫 No speech detected by Whisper\n")
                            else:
                                print(f"⏭️  Skipped (too short: {speech_duration:.2f}s)\n")
                            
                            # Clear buffer
                            speech_buffer = []
                        
                        print("👂 Listening…\n")
                
                # Buffer audio during speech
                if is_speaking:
                    speech_buffer.append(audio_chunk)
        
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Voice Assistant with Silero VAD + Whisper STT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Faster-Whisper model size"
    )
    parser.add_argument(
        "--whisper-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper"
    )
    parser.add_argument(
        "--whisper-compute",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for Whisper"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold (0.0-1.0, higher = stricter)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (VAD probabilities, audio levels)"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantSileroWhisper(
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute=args.whisper_compute,
        vad_threshold=args.vad_threshold,
        debug=args.debug
    )
    assistant.run()


if __name__ == "__main__":
    main()
