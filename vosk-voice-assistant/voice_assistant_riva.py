#!/usr/bin/env python3
"""
Voice Assistant with NVIDIA Riva ASR API + Production Intent Classifier
Uses cloud-based Riva for superior accuracy
"""

import os
import sys
import time
import ctypes
import numpy as np
import json
import warnings
import logging
import requests
import tempfile
import soundfile as sf

# Suppress ALL stderr from C libraries (ALSA, JACK, etc.)
from ctypes import CDLL, c_int, c_char_p

try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = CDLL("libasound.so.2")
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# Suppress logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
_stderr = sys.stderr
sys.stderr = io.StringIO()
import pyaudio
sys.stderr = _stderr

from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)

from piper.voice import PiperVoice
from production_intent_classifier import ProductionIntentClassifier


class RivaASRClient:
    """Client for NVIDIA Riva ASR API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # NVIDIA NIM Parakeet ASR endpoint (updated format)
        self.endpoint = "https://integrate.api.nvidia.com/v1/asr/transcriptions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        print("✅ Riva API client initialized")
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using NVIDIA Riva API
        
        Args:
            audio_data: numpy array of int16 audio samples
            sample_rate: sample rate in Hz
        
        Returns:
            Transcribed text or None
        """
        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Save to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_float, sample_rate)
                temp_path = f.name
            
            try:
                # Read audio file
                with open(temp_path, 'rb') as audio_file:
                    files = {
                        'file': ('audio.wav', audio_file, 'audio/wav')
                    }
                    
                    # Optional: specify model (defaults to Parakeet)
                    data = {
                        'model': 'nvidia/parakeet-ctc-1.1b-asr',  # or 'nvidia/parakeet-rnnt-1.1b'
                        'language': 'en'
                    }
                    
                    # Make API request
                    response = requests.post(
                        self.endpoint,
                        headers=self.headers,
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Riva API returns: {"text": "transcribed text"}
                        text = result.get('text', '')
                        return text.strip() if text.strip() else None
                    else:
                        print(f"❌ Riva API error: {response.status_code}")
                        print(f"   Response: {response.text}")
                        return None
            
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        
        except Exception as e:
            print(f"❌ Riva transcription error: {e}")
            return None


class VoiceAssistantRiva:
    """Voice Assistant using NVIDIA Riva ASR API"""
    
    def __init__(
        self,
        riva_api_key: str,
        vosk_model_path="models/vosk-model-small-en-us-0.15",
        piper_model_path="models/piper/en_US-lessac-medium.onnx",
        sample_rate=16000
    ):
        self.sample_rate = sample_rate
        self.cart = {}
        
        print("🔧 Initializing Voice Assistant with Riva API...")
        
        # Load Vosk for VAD only
        print(f"📥 Loading Vosk VAD...")
        start = time.time()
        self.vosk_model = VoskModel(vosk_model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
        self.recognizer.SetWords(False)
        print(f"   ✅ Vosk loaded in {time.time() - start:.2f}s")
        
        # Initialize Riva API client
        print(f"📥 Connecting to NVIDIA Riva API...")
        self.riva_client = RivaASRClient(riva_api_key)
        
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
        
        print("\n✅ Voice Assistant ready!\n")
    
    def _transcribe_audio(self, audio_data):
        """Transcribe audio using Riva API"""
        return self.riva_client.transcribe_audio(audio_data, self.sample_rate)
    
    def _execute_intent(self, intent):
        """Execute the classified intent and return response"""
        intent_name = intent.name
        entities = intent.entities
        
        if intent_name == "add_to_cart":
            # Check for multiple items
            if 'items' in entities:
                added = []
                for item in entities['items']:
                    # item is a CartItem namedtuple with product and quantity
                    product = item.product
                    quantity = item.quantity
                    
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
        """Listen for speech, transcribe with Riva, classify intent, and respond."""
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
        
        # Transcribe with Riva API
        print("🔄 Transcribing with Riva API...")
        
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
    # Get Riva API key from environment variable or prompt
    api_key = os.environ.get('NVIDIA_API_KEY')
    
    if not api_key:
        print("="*70)
        print("🔑 NVIDIA Riva API Key Required")
        print("="*70)
        print("Get your API key from: https://build.nvidia.com/")
        print("\nYou can either:")
        print("1. Set environment variable: export NVIDIA_API_KEY='your-key-here'")
        print("2. Enter it below")
        print("="*70 + "\n")
        
        api_key = input("Enter your NVIDIA API Key: ").strip()
        
        if not api_key:
            print("❌ API key is required!")
            sys.exit(1)
    
    # Initialize assistant
    assistant = VoiceAssistantRiva(riva_api_key=api_key)
    
    print("="*70)
    print("🎙️  VOICE ASSISTANT (NVIDIA Riva API + Intent Classification)")
    print("="*70)
    print("✅ Cloud-based Riva ASR - Superior accuracy")
    print("✅ Parakeet-CTC-1.1B model")
    print("✅ No local GPU required")
    print("\nCommands you can try:")
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
