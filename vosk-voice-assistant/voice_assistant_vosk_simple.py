#!/usr/bin/env python3
"""
Simple always-on voice assistant using Vosk for real-time STT.
Based on: https://stackoverflow.com/questions/51525691/realtime-offline-speech-recognition-in-python

This uses Vosk's built-in VAD which is much more reliable than manual RMS thresholds.
"""
import pyaudio
import json
import time
import os
from vosk import Model, KaldiRecognizer
from piper import PiperVoice
import sounddevice as sd
import numpy as np

class VoiceAssistantVosk:
    def __init__(self, vosk_model_path="models/vosk-model-small-en-us-0.15"):
        print("="*50)
        print("🤖 VOICE ASSISTANT (VOSK)")
        print("="*50)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load Vosk model
        print("\nLoading Vosk STT model...")
        start = time.time()
        if not os.path.isabs(vosk_model_path):
            vosk_model_path = os.path.join(script_dir, vosk_model_path)
        self.vosk_model = Model(vosk_model_path)
        print(f"✅ Vosk loaded in {time.time()-start:.2f}s")
        
        # Load Piper TTS
        print("Loading Piper TTS...")
        start = time.time()
        piper_path = os.path.join(script_dir, "models/piper/en_US-lessac-medium.onnx")
        self.piper_voice = PiperVoice.load(piper_path)
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        # Audio setup
        self.sample_rate = 16000
        self.chunk_size = 4000  # Vosk recommends 4000 samples per chunk
        
        print("\n✅ Ready!")
    
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
        print("\nAlways-listening mode (Vosk VAD). Speak naturally. Ctrl+C to exit.\n")
        
        # Create recognizer
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        recognizer.SetWords(True)  # Get word-level timestamps
        
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
            
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                if recognizer.AcceptWaveform(data):
                    # Final result (end of utterance detected by Vosk's VAD)
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text:
                        print(f"📝 You: \"{text}\"")
                        
                        # Echo response (LLM disabled for STT quality focus)
                        response = f"You said: {text}"
                        print(f"💬 Assistant: \"{response}\"")
                        
                        synth_ms, play_ms = self.speak(response)
                        print(f"🔊 TTS: {synth_ms}ms synth + {play_ms}ms play\n")
                        
                        print("👂 Listening…\n")
                else:
                    # Partial result (speech in progress)
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get('partial', '')
                    if partial_text:
                        # Show live transcription
                        print(f"\r🎤 {partial_text}", end='', flush=True)
        
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Assistant with Vosk")
    parser.add_argument(
        "--model",
        default="models/vosk-model-small-en-us-0.15",
        help="Path to Vosk model directory"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantVosk(vosk_model_path=args.model)
    assistant.run()


if __name__ == "__main__":
    main()
