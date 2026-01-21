#!/usr/bin/env python3
"""
Hybrid voice assistant:
- Vosk for speech detection (VAD only, no transcription)
- Faster-Whisper for accurate transcription
- Piper for TTS

This combines Vosk's reliable VAD with Whisper's superior accuracy.
"""
import pyaudio
import json
import time
import os
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
from faster_whisper import WhisperModel
from piper import PiperVoice
import sounddevice as sd
import io
import wave

# Reduce Vosk's verbose logging
SetLogLevel(-1)

class VoiceAssistantHybrid:
    def __init__(self, 
                 vosk_model_path="models/vosk-model-small-en-us-0.15",
                 whisper_model="small",
                 whisper_device="cpu",
                 whisper_compute="int8",
                 silence_duration=1.0):
        """
        Args:
            vosk_model_path: Path to Vosk model (used only for VAD)
            whisper_model: Faster-Whisper model size (tiny/base/small/medium)
            whisper_device: cpu or cuda
            whisper_compute: int8, float16, float32
            silence_duration: Seconds of silence before processing (default 1.0)
        """
        print("="*50)
        print("🤖 VOICE ASSISTANT (Hybrid: Vosk VAD + Whisper STT)")
        print("="*50)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load Vosk model (for VAD only)
        print("\n📡 Loading Vosk (VAD only)...")
        start = time.time()
        if not os.path.isabs(vosk_model_path):
            vosk_model_path = os.path.join(script_dir, vosk_model_path)
        self.vosk_model = Model(vosk_model_path)
        print(f"✅ Vosk VAD loaded in {time.time()-start:.2f}s")
        
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
        
        # Audio setup
        self.sample_rate = 16000
        self.chunk_size = 4000
        self.silence_duration = silence_duration
        self.silence_threshold_chunks = int(silence_duration * self.sample_rate / self.chunk_size)
        
        print(f"\n⏱️  Silence threshold: {silence_duration}s (~{self.silence_threshold_chunks} chunks)")
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
        print("\n🎤 Hybrid mode: Vosk detects speech → Whisper transcribes")
        print("Speak naturally. Ctrl+C to exit.\n")
        
        # Create Vosk recognizer (VAD only)
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        recognizer.SetWords(False)  # Don't need word details from Vosk
        
        # Setup PyAudio
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            stream.start_stream()
            
            print("👂 Listening…\n")
            
            # Buffer to accumulate audio for Whisper
            audio_buffer = []
            silence_chunks = 0  # Track consecutive silent chunks
            speech_started = False  # Track if we've detected speech
            last_speech_time = 0  # Track when we last detected speech
            
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Store audio for later transcription
                audio_buffer.append(audio_chunk)
                
                # Calculate RMS (audio energy level) to supplement VAD
                rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
                has_audio_energy = rms > 300  # Threshold for detecting audio energy
                
                # Check if Vosk has a final result (end of utterance)
                has_final = recognizer.AcceptWaveform(data)
                
                # Get partial result to check if speech is ongoing
                partial_result = json.loads(recognizer.PartialResult())
                partial_text = partial_result.get('partial', '').strip()
                
                # Debug: Print RMS and partial text when there's activity
                if rms > 100 or len(partial_text) > 0:
                    print(f"\r[DEBUG] RMS: {int(rms)}, Partial: '{partial_text}', Energy: {has_audio_energy}  ", end='', flush=True)
                
                # Primary detection: Vosk detects partial text
                # Secondary: High audio energy (only when speech already started)
                vosk_speaking = len(partial_text) > 0
                energy_speaking = speech_started and has_audio_energy
                
                is_speaking = vosk_speaking or energy_speaking
                
                if is_speaking:
                    # Speech is ongoing
                    if not speech_started:
                        speech_started = True
                        print(f"\r🎤 Detecting speech...", end='', flush=True)
                    silence_chunks = 0
                    last_speech_time = time.time()
                
                elif speech_started:
                    # Was speaking, now silence
                    silence_chunks += 1
                    
                    # Check if enough silence has passed OR timeout (5 seconds max)
                    time_since_speech = time.time() - last_speech_time
                    should_process = (silence_chunks >= self.silence_threshold_chunks) or (time_since_speech > 5.0)
                    
                    if should_process:
                        print("\r🔄 Processing with Whisper...                    ")
                        
                        # Get final result if available
                        if has_final:
                            final_result = json.loads(recognizer.Result())
                        
                        # Concatenate all buffered audio
                        full_audio = np.concatenate(audio_buffer)
                        
                        # Transcribe with Whisper
                        start = time.time()
                        text = self._transcribe_audio(full_audio)
                        whisper_time = int((time.time() - start) * 1000)
                        
                        if text:
                            print(f"📝 You: \"{text}\" ({whisper_time}ms)")
                            
                            # Echo response
                            response = f"You said: {text}"
                            print(f"💬 Assistant: \"{response}\"")
                            
                            synth_ms, play_ms = self.speak(response)
                            print(f"🔊 TTS: {synth_ms}ms + {play_ms}ms\n")
                        else:
                            print("🚫 No speech detected by Whisper\n")
                        
                        # Reset state
                        audio_buffer = []
                        silence_chunks = 0
                        speech_started = False
                        print("👂 Listening…\n")
                
                # Keep only last 16 seconds of audio to prevent memory overflow
                if len(audio_buffer) > 400:  # 400 chunks * 4000 samples = 16 seconds
                    audio_buffer = audio_buffer[-400:]
        
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Hybrid Voice Assistant (Vosk VAD + Whisper STT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--vosk-model",
        default="models/vosk-model-small-en-us-0.15",
        help="Path to Vosk model (used only for VAD)"
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
        "--silence-duration",
        type=float,
        default=1.0,
        help="Seconds of silence before processing speech (prevents interruption)"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantHybrid(
        vosk_model_path=args.vosk_model,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute=args.whisper_compute,
        silence_duration=args.silence_duration
    )
    assistant.run()


if __name__ == "__main__":
    main()
