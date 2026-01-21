#!/usr/bin/env python3
"""
Fast voice assistant with aggressive hallucination filtering.
- Stricter confidence thresholds
- Faster timeout (less waiting)
- Filters common hallucinated words ("the", "a", "i")
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

class VoiceAssistantVoskFast:
    # Common hallucinated words to filter out (only most problematic ones)
    HALLUCINATION_WORDS = {
        'the', 'a', 'and', 'of', 'to', 'in', 'is', 'it', 'that'
    }
    
    def __init__(self, 
                 vosk_model_path="models/vosk-model-en-us-0.22-lgraph",
                 min_confidence=0.75,  # Balanced: 75%
                 min_words=2):          # Require at least 2 words
        """
        Args:
            vosk_model_path: Path to Vosk model
            min_confidence: Minimum confidence score (0-1)
            min_words: Minimum number of words (filters short hallucinations)
        """
        print("="*50)
        print("🤖 VOICE ASSISTANT (VOSK - Fast & Filtered)")
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
        print(f"   - Filtered words: {', '.join(sorted(self.HALLUCINATION_WORDS)[:5])}...")
        print("\n✅ Ready!")
    
    def _is_valid_transcription(self, result):
        """
        Filter out hallucinations using multiple heuristics.
        """
        text = result.get('text', '').strip()
        
        # Check 1: Empty text
        if not text:
            return False, "empty"
        
        # Check 2: Word count
        words = text.split()
        if len(words) < self.min_words:
            return False, f"too_short ({len(words)} words, need {self.min_words})"
        
        # Check 3: Single hallucinated word (even if repeated)
        unique_words = set(w.lower() for w in words)
        if len(unique_words) == 1 and list(unique_words)[0] in self.HALLUCINATION_WORDS:
            return False, f"repeated_hallucination"
        
        # Check 4: Starts with common hallucination and has low word count
        # Only filter if it's JUST hallucination words at the start
        if len(words) <= 3 and words[0].lower() in self.HALLUCINATION_WORDS:
            # Check if first 2 words are both hallucinations
            if len(words) >= 2 and words[1].lower() in self.HALLUCINATION_WORDS:
                return False, f"starts_with_hallucinations"
        
        # Check 5: Confidence score (from word-level results)
        word_results = result.get('result', [])
        if word_results:
            confidences = [w.get('conf', 0) for w in word_results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if avg_confidence < self.min_confidence:
                return False, f"low_confidence ({avg_confidence:.2f} < {self.min_confidence})"
        
        # Check 6: Too many hallucination words (>50% of utterance)
        hallucination_count = sum(1 for w in words if w.lower() in self.HALLUCINATION_WORDS)
        if hallucination_count > len(words) * 0.5:
            return False, f"too_many_filler_words ({hallucination_count}/{len(words)})"
        
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
        print("\n⚡ Fast mode with aggressive filtering. Speak clearly!\n")
        
        # Create recognizer
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        recognizer.SetWords(True)
        
        # Set max alternatives to 0 for faster processing
        recognizer.SetMaxAlternatives(0)
        
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
            
            rejected_count = 0
            last_partial = ""
            
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                if recognizer.AcceptWaveform(data):
                    # Final result
                    result = json.loads(recognizer.Result())
                    
                    # Clear partial display
                    if last_partial:
                        print("\r" + " " * (len(last_partial) + 10), end='\r')
                        last_partial = ""
                    
                    # Validate transcription
                    is_valid, reason = self._is_valid_transcription(result)
                    
                    if is_valid:
                        text = result.get('text', '').strip()
                        
                        # Calculate confidence
                        word_results = result.get('result', [])
                        if word_results:
                            confidences = [w.get('conf', 0) for w in word_results]
                            avg_conf = sum(confidences) / len(confidences)
                            conf_str = f" ✓{avg_conf:.0%}"
                        else:
                            conf_str = ""
                        
                        print(f"📝 You: \"{text}\"{conf_str}")
                        
                        rejected_count = 0
                        
                        # Echo response
                        response = f"You said: {text}"
                        print(f"💬 Assistant: \"{response}\"")
                        
                        synth_ms, play_ms = self.speak(response)
                        print(f"🔊 {synth_ms}ms + {play_ms}ms\n")
                        
                        print("👂 Listening…\n")
                    else:
                        # Rejected
                        rejected_count += 1
                        text = result.get('text', '').strip()
                        
                        # Only show first few rejections to avoid spam
                        if rejected_count <= 2:
                            print(f"🚫 \"{text}\" → {reason}")
                        
                        if rejected_count == 5:
                            print(f"⚠️  Too many rejections. Try speaking louder/clearer.\n")
                            rejected_count = 0
                
                else:
                    # Partial result
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get('partial', '')
                    
                    # Only show if meaningful (not just hallucination words)
                    words = partial_text.split()
                    non_hallucination = [w for w in words if w.lower() not in self.HALLUCINATION_WORDS]
                    
                    if len(non_hallucination) > 0 or len(words) > 3:
                        display = f"🎤 {partial_text}"
                        print(f"\r{display}", end='', flush=True)
                        last_partial = display
                    elif last_partial:
                        # Clear if only hallucination words now
                        print("\r" + " " * len(last_partial), end='\r', flush=True)
                        last_partial = ""
        
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fast Voice Assistant with Aggressive Filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="models/vosk-model-en-us-0.22-lgraph",
        help="Path to Vosk model directory"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.80,
        help="Minimum confidence score (0-1)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum number of words to accept"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantVoskFast(
        vosk_model_path=args.model,
        min_confidence=args.min_confidence,
        min_words=args.min_words
    )
    assistant.run()


if __name__ == "__main__":
    main()
