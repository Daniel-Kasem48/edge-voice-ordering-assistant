#!/usr/bin/env python3
"""
Voice Assistant with NeMo ASR + Production Intent Classifier
"""

import os
import sys
import time
import ctypes
import numpy as np
import json
import warnings
import logging

# Suppress ALL stderr from C libraries (ALSA, JACK, etc.)
# This must be done BEFORE importing pyaudio
from ctypes import CDLL, c_int, c_char_p

# Suppress ALSA errors
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = CDLL("libasound.so.2")
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# Suppress JACK errors by redirecting stderr during PyAudio init
import io
from contextlib import redirect_stderr

# Suppress all Python logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for name in ['nemo_logger', 'nemo', 'pytorch_lightning', 'torch', 'nemo.collections']:
    logging.getLogger(name).setLevel(logging.CRITICAL)

# Now import audio libraries with stderr suppressed
_stderr = sys.stderr
sys.stderr = io.StringIO()
import pyaudio
sys.stderr = _stderr

import torch
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)

# Suppress NeMo import logs
logging.getLogger().setLevel(logging.CRITICAL)
import nemo.collections.asr as nemo_asr
from piper.voice import PiperVoice

from production_intent_classifier import ProductionIntentClassifier


class VoiceAssistantWithIntent:
    def __init__(
        self,
        vosk_model_path="models/vosk-model-small-en-us-0.15",
        piper_model_path="models/piper/en_US-lessac-medium.onnx",
        nemo_model="stt_en_quartznet15x5",
        sample_rate=16000
    ):
        self.sample_rate = sample_rate
        
        # Shopping cart state
        self.cart = {}
        
        print("🔧 Initializing Voice Assistant...")
        
        # Load Vosk for VAD only
        print(f"📥 Loading Vosk VAD model...")
        start = time.time()
        self.vosk_model = VoskModel(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
        self.recognizer.SetWords(False)
        print(f"   ✅ Vosk loaded in {time.time() - start:.2f}s")
        
        # Load NeMo ASR
        print(f"📥 Loading NeMo ASR ({nemo_model})...")
        start = time.time()
        self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=nemo_model)
        self.asr_model.eval()
        print(f"   ✅ NeMo loaded in {time.time() - start:.2f}s")
        
        # Load Production Intent Classifier
        print("📥 Loading Intent Classifier...")
        start = time.time()
        self.intent_classifier = ProductionIntentClassifier()
        print(f"   ✅ Intent Classifier loaded in {time.time() - start:.2f}s")
        
        # Load Piper TTS
        print(f"📥 Loading Piper TTS...")
        start = time.time()
        self.tts_voice = PiperVoice.load(piper_model_path)
        print(f"   ✅ Piper TTS loaded in {time.time() - start:.2f}s")
        
        print("✅ Voice Assistant ready!\n")
    
    def _transcribe_audio(self, audio_data):
        """Transcribe audio using NeMo ASR"""
        import tempfile
        import soundfile as sf
        
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Save to temp file (NeMo requires file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_float, self.sample_rate)
            temp_path = f.name
        
        try:
            # Transcribe with NeMo
            transcriptions = self.asr_model.transcribe([temp_path])
            # NeMo returns Hypothesis objects, extract text
            if transcriptions:
                result = transcriptions[0]
                # Handle both string and Hypothesis object
                if hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)
                return text.strip() if text.strip() else None
            return None
        finally:
            os.unlink(temp_path)
    
    def _execute_intent(self, intent):
        """Execute the classified intent and return response"""
        intent_name = intent.name
        entities = intent.entities
        
        if intent_name == "add_to_cart":
            # Check for multiple items
            if 'items' in entities:
                added = []
                for product, quantity in entities['items']:
                    if product in self.cart:
                        self.cart[product] += quantity
                    else:
                        self.cart[product] = quantity
                    added.append(f"{quantity} {product}")
                return f"Added {', '.join(added)} to your cart."
            else:
                # Single item
                product = entities.get('product', 'item')
                quantity = entities.get('quantity', 1)
                
                if product in self.cart:
                    self.cart[product] += quantity
                else:
                    self.cart[product] = quantity
                
                return f"Added {quantity} {product} to your cart."
        
        elif intent_name == "remove_from_cart":
            product = entities.get('product')
            if not product:
                return "What would you like to remove?"
            
            if product in self.cart:
                del self.cart[product]
                return f"Removed {product} from your cart."
            else:
                return f"{product} is not in your cart."
        
        elif intent_name == "view_cart":
            if not self.cart:
                return "Your cart is empty."
            
            items = []
            for product, quantity in self.cart.items():
                items.append(f"{quantity} {product}")
            return "Your cart contains: " + ", ".join(items)
        
        elif intent_name == "clear_cart":
            self.cart.clear()
            return "Your cart has been cleared."
        
        elif intent_name == "checkout":
            if not self.cart:
                return "Your cart is empty. Add some items first."
            
            items = []
            for product, quantity in self.cart.items():
                items.append(f"{quantity} {product}")
            response = "Processing your order for: " + ", ".join(items) + ". Thank you!"
            self.cart.clear()
            return response
        
        elif intent_name == "search_products":
            product = entities.get('product', 'items')
            return f"Here are our available {product}. What would you like to order?"
        
        elif intent_name == "get_time":
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."
        
        elif intent_name == "get_date":
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            return f"Today is {current_date}."
        
        elif intent_name == "greeting":
            return "Hello! Welcome to our coffee shop. What would you like to order?"
        
        elif intent_name == "help":
            return "You can add items to cart, view cart, remove items, search products, or checkout. What would you like to do?"
        
        elif intent_name == "goodbye":
            return "Goodbye! Have a great day!"
        
        else:
            return "I didn't understand that. Can you rephrase?"
    
    def _synthesize_speech(self, text):
        """Synthesize speech using Piper TTS"""
        audio_chunks = []
        for chunk in self.tts_voice.synthesize(text):
            audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        
        # Concatenate all chunks
        audio_data = np.concatenate(audio_chunks)
        return audio_data
    
    def _play_audio(self, audio_data, sample_rate=22050):
        """Play audio through speakers"""
        _stderr = sys.stderr
        sys.stderr = io.StringIO()
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
        sys.stderr = _stderr
        stream.write(audio_data.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def listen_and_respond(self, silence_duration=1.0):
        """Listen for speech, transcribe, classify intent, and respond."""
        _stderr = sys.stderr
        sys.stderr = io.StringIO()
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
            input=True, input_device_index=4, frames_per_buffer=4000
        )
        sys.stderr = _stderr
        
        print("🎤 Listening...")
        
        frames = []
        silence_chunks = 0
        silence_threshold_chunks = int(silence_duration * self.sample_rate / 4000)
        speech_detected = False
        speech_start_time = None
        
        try:
            while True:
                data = stream.read(4000, exception_on_overflow=False)
                frames.append(data)
                
                # Check if speech is detected
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if result.get("text", "").strip():
                        speech_detected = True
                        silence_chunks = 0
                        if speech_start_time is None:
                            speech_start_time = time.time()
                            print("🎤 Detecting speech...")
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    if partial.get("partial", "").strip():
                        if not speech_detected:
                            speech_detected = True
                            speech_start_time = time.time()
                            print("🎤 Detecting speech...")
                        silence_chunks = 0
                    else:
                        if speech_detected:
                            silence_chunks += 1
                
                # Check if we should process the audio
                if speech_start_time:
                    time_since_start = time.time() - speech_start_time
                    should_process = (silence_chunks >= silence_threshold_chunks) or (time_since_start > 10.0)
                    
                    if should_process:
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return None
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if not speech_detected:
            print("⚠️  No speech detected")
            return None
        
        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        # Transcribe with NeMo (suppress progress bar)
        print("🔄 Transcribing...")
        
        transcription = self._transcribe_audio(audio_data)
        
        if not transcription:
            print("⚠️  No transcription")
            return None
        
        print(f"📝 \"{transcription}\"")
        
        # Classify intent
        intent = self.intent_classifier.classify(transcription)
        
        print(f"🎯 {intent.name} ({intent.confidence:.0%})", end="")
        if intent.entities:
            print(f" → {intent.entities}")
        else:
            print()
        
        # Execute intent and get response
        response = self._execute_intent(intent)
        print(f"💬 \"{response}\"")
        
        # Synthesize and play response
        audio = self._synthesize_speech(response)
        self._play_audio(audio)
        
        print("✅ Done!\n")
        return {
            'transcription': transcription,
            'intent': intent.name,
            'confidence': intent.confidence,
            'entities': intent.entities,
            'response': response
        }


def main():
    # Initialize assistant
    assistant = VoiceAssistantWithIntent()
    
    print("="*70)
    print("🎙️  VOICE ASSISTANT WITH INTENT CLASSIFICATION")
    print("="*70)
    print("Commands you can try:")
    print("  - 'I want two coffees'")
    print("  - 'add a muffin'")
    print("  - 'show me my cart'")
    print("  - 'remove the coffee'")
    print("  - 'checkout'")
    print("  - 'what time is it?'")
    print("  - 'hello'")
    print("\nPress Ctrl+C to exit")
    print("="*70 + "\n")
    
    # Continuous listening loop
    try:
        while True:
            result = assistant.listen_and_respond()
            
            # Check for goodbye intent
            if result and result['intent'] == 'goodbye':
                print("👋 Goodbye!")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
