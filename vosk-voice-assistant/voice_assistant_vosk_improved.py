#!/usr/bin/env python3
"""
Improved voice assistant with hallucination filtering.
Addresses common Vosk issues: background noise triggering, phantom words, over-sensitivity.
"""
import pyaudio
import json
import time
import os
from vosk import Model, KaldiRecognizer, SetLogLevel
from piper import PiperVoice
import sounddevice as sd
import numpy as np

# Reduce Vosk's verbose logging
SetLogLevel(-1)

class VoiceAssistantVoskImproved:
    def __init__(self, 
                 vosk_model_path="models/vosk-model-small-en-us-0.15",
                 min_confidence=0.7,
                 min_words=1):
        """
        Args:
            vosk_model_path: Path to Vosk model
            min_confidence: Minimum confidence score (0-1) to accept transcription
            min_words: Minimum number of words to accept (filters single-letter hallucinations)
        """
        print("="*50)
        print("🤖 VOICE ASSISTANT (VOSK - Anti-Hallucination)")
        print("="*50)
        
        self.min_confidence = min_confidence
        self.min_words = min_words
        
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
        self.chunk_size = 4000
        
        print(f"\n⚙️  Settings:")
        print(f"   - Min confidence: {self.min_confidence}")
        print(f"   - Min words: {self.min_words}")
        print("\n✅ Ready!")
    
    def _is_valid_transcription(self, result):
        """
        Filter out hallucinations using multiple heuristics.
        
        Args:
            result: Vosk result dict with 'text' and 'result' (word-level details)
        
        Returns:
            (is_valid: bool, reason: str)
        """
        text = result.get('text', '').strip()
        
        # Check 1: Empty text
        if not text:
            return False, "empty"
        
        # Check 2: Word count
        words = text.split()
        if len(words) < self.min_words:
            return False, f"too_short ({len(words)} words)"
        
        # Check 3: Confidence score (from word-level results)
        word_results = result.get('result', [])
        if word_results:
            confidences = [w.get('conf', 0) for w in word_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if avg_confidence < self.min_confidence:
                return False, f"low_confidence ({avg_confidence:.2f})"
        
        # Check 4: Repeated single characters (common hallucination pattern)
        if len(words) == 1 and len(words[0]) <= 2:
            return False, "single_char_word"
        
        return True, "valid"
    
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
        print("\nAlways-listening mode with hallucination filtering. Ctrl+C to exit.\n")
        
        # Create recognizer with word-level timestamps for confidence scores
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        recognizer.SetWords(True)
        
        # Optional: Configure Vosk parameters (requires model support)
        # These help reduce false positives
        config_str = json.dumps({
            "silence_timeout": 2.0,  # Longer silence before finalizing (reduces premature triggers)
        })
        # Note: Not all models support configuration, so we don't set it by default
        
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
            
            print("👂 Listening… (speak clearly, avoid background noise)\n")
            
            rejected_count = 0
            
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                if recognizer.AcceptWaveform(data):
                    # Final result (end of utterance detected by Vosk's VAD)
                    result = json.loads(recognizer.Result())
                    
                    # Validate transcription
                    is_valid, reason = self._is_valid_transcription(result)
                    
                    if is_valid:
                        text = result.get('text', '').strip()
                        
                        # Calculate actual confidence
                        word_results = result.get('result', [])
                        if word_results:
                            confidences = [w.get('conf', 0) for w in word_results]
                            avg_conf = sum(confidences) / len(confidences)
                            conf_str = f" [conf: {avg_conf:.2f}]"
                        else:
                            conf_str = ""
                        
                        print(f"\n📝 You: \"{text}\"{conf_str}")
                        
                        # Reset rejected counter
                        if rejected_count > 0:
                            rejected_count = 0
                        
                        # Echo response
                        response = f"You said: {text}"
                        print(f"💬 Assistant: \"{response}\"")
                        
                        synth_ms, play_ms = self.speak(response)
                        print(f"🔊 TTS: {synth_ms}ms synth + {play_ms}ms play\n")
                        
                        print("👂 Listening…\n")
                    else:
                        # Rejected hallucination
                        rejected_count += 1
                        text = result.get('text', '').strip()
                        print(f"\r🚫 Filtered: \"{text}\" (reason: {reason})", end='', flush=True)
                        
                        # If too many rejections, give user feedback
                        if rejected_count >= 3:
                            print(f"\n⚠️  Tip: Speak more clearly or reduce background noise\n")
                            rejected_count = 0
                
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
    parser = argparse.ArgumentParser(
        description="Voice Assistant with Vosk (Anti-Hallucination)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="models/vosk-model-small-en-us-0.15",
        help="Path to Vosk model directory"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence score (0-1) to accept transcription"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum number of words to accept (filters single-letter hallucinations)"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantVoskImproved(
        vosk_model_path=args.model,
        min_confidence=args.min_confidence,
        min_words=args.min_words
    )
    assistant.run()


if __name__ == "__main__":
    main()
