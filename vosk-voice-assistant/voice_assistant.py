#!/usr/bin/env python3
"""
Voice Echo with preloaded Piper TTS for speed.
"""
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import time
import sys

# Import Piper directly (not CLI)
from piper import PiperVoice

SetLogLevel(-1)

class OptimizedEcho:
    def __init__(self):
        print("="*50)
        print("🎙️  OPTIMIZED VOICE ECHO")
        print("="*50)
        
        self.sr = 16000
        
        # Load Vosk
        print("\nLoading Vosk STT...")
        start = time.time()
        self.vosk_model = Model("models/vosk-model-small-en-us-0.15")
        print(f"✅ Vosk loaded in {time.time()-start:.2f}s")
        
        # Pre-load Piper voice (this is the key optimization!)
        print("Loading Piper TTS...")
        start = time.time()
        self.piper_voice = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        print("\n✅ Ready!\n")
    
    def record(self, duration=3):
        print(f"🎤 Recording {duration}s...")
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype=np.int16)
        sd.wait()
        return audio.flatten()
    
    def transcribe(self, audio):
        start = time.time()
        rec = KaldiRecognizer(self.vosk_model, self.sr)
        rec.SetWords(True)  # Enable word-level info for better accuracy
        
        # Process in chunks for better recognition
        chunk_size = 4000
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            rec.AcceptWaveform(chunk.tobytes())
        
        result = json.loads(rec.FinalResult())
        return result.get('text', '').strip(), int((time.time() - start) * 1000)
    
    def speak(self, text):
        """Synthesize and play using preloaded Piper"""
        start = time.time()
        
        # Synthesize to audio - AudioChunk has .audio_int16_bytes property
        audio_chunks = []
        for chunk in self.piper_voice.synthesize(text):
            audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        
        synth_time = int((time.time() - start) * 1000)
        
        if audio_chunks:
            # Combine and play
            audio_data = np.concatenate(audio_chunks)
            audio_float = audio_data.astype(np.float32) / 32767
            
            play_start = time.time()
            sd.play(audio_float, 22050)  # Piper outputs at 22050Hz
            sd.wait()
            play_time = int((time.time() - play_start) * 1000)
            
            return synth_time, play_time
        
        return synth_time, 0
    
    def run(self):
        print("Press Enter to record, Ctrl+C to exit\n")
        
        try:
            while True:
                input(">>> Press Enter...")
                
                # Record
                audio = self.record(2)
                
                # Transcribe
                text, stt_ms = self.transcribe(audio)
                
                if text:
                    print(f"📝 \"{text}\" (STT: {stt_ms}ms)")
                    
                    # Speak
                    synth_ms, play_ms = self.speak(text)
                    total = stt_ms + synth_ms + play_ms
                    
                    print(f"🔊 Synth: {synth_ms}ms, Play: {play_ms}ms")
                    print(f"⏱️  Total: {total}ms ({total/1000:.1f}s)\n")
                else:
                    print("❌ No speech detected\n")
                
        except KeyboardInterrupt:
            print("\n👋 Bye!")

if __name__ == "__main__":
    OptimizedEcho().run()
