#!/usr/bin/env python3
"""
Voice Assistant with NVIDIA Riva Realtime WebSocket API
Uses cloud-based Riva for superior accuracy with WebSocket streaming
"""

import os
import sys
import time
import json
import base64
import asyncio
import warnings
import logging
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import websockets
import pyaudio
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)

from piper.voice import PiperVoice
from production_intent_classifier import ProductionIntentClassifier


class RivaRealtimeASRClient:
    """Client for NVIDIA Riva Realtime WebSocket API"""
    
    def __init__(self, token: str, function_id: str):
        self.token = token
        self.function_id = function_id
        self.ws_url = "wss://api.nvcf.nvidia.com/v2/nvcf/streaming/pexec"
        self.websocket = None
        self.transcription_queue = asyncio.Queue()
        
    async def connect(self):
        """Connect to Riva WebSocket server"""
        try:
            # Construct WebSocket URL with authentication
            full_url = f"{self.ws_url}?token={self.token}&functionId={self.function_id}"
            
            print("📡 Connecting to NVIDIA Riva API...")
            self.websocket = await websockets.connect(full_url)
            print("✅ Connected to Riva API")
            
            # Send streaming configuration
            config = {
                "streamingConfig": {
                    "config": {
                        "encoding": 1,  # LINEAR_PCM
                        "sampleRateHertz": 16000,
                        "languageCode": "en-US",
                        "maxAlternatives": 1,
                        "audioChannelCount": 1,
                        "enableAutomaticPunctuation": True
                    },
                    "interimResults": True
                }
            }
            
            await self.websocket.send(json.dumps(config))
            print("✅ Sent streaming configuration")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def send_audio(self, audio_data: np.ndarray):
        """Send audio data to Riva API"""
        try:
            # Convert to int16 bytes
            audio_bytes = audio_data.tobytes()
            
            # Base64 encode
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Send audio content
            message = {
                "audioContent": audio_base64
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"❌ Send audio error: {e}")
    
    async def receive_transcriptions(self):
        """Receive transcription results from Riva API"""
        try:
            async for message in self.websocket:
                result = json.loads(message)
                
                # Check for different result types
                if "results" in result:
                    for res in result["results"]:
                        if "alternatives" in res:
                            for alt in res["alternatives"]:
                                transcript = alt.get("transcript", "")
                                is_final = res.get("isFinal", False)
                                
                                if transcript:
                                    await self.transcription_queue.put({
                                        "text": transcript,
                                        "is_final": is_final
                                    })
                
        except websockets.exceptions.ConnectionClosed:
            print("⚠️  WebSocket connection closed")
        except Exception as e:
            print(f"❌ Receive error: {e}")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()


class VoiceAssistantRivaWebSocket:
    """Voice assistant using NVIDIA Riva Realtime WebSocket API"""
    
    def __init__(self, token: str, function_id: str):
        print("🔧 Initializing Voice Assistant with Riva WebSocket API...")
        
        # Initialize Vosk VAD
        print("📥 Loading Vosk VAD...")
        start = time.time()
        self.vosk_model = VoskModel("models/vosk-model-small-en-us-0.15")
        self.vad_recognizer = KaldiRecognizer(self.vosk_model, 16000)
        print(f"   ✅ Vosk loaded in {time.time() - start:.2f}s")
        
        # Initialize Riva API client
        self.riva_client = RivaRealtimeASRClient(token, function_id)
        
        # Initialize Intent Classifier
        print("📥 Loading Intent Classifier...")
        start = time.time()
        self.intent_classifier = ProductionIntentClassifier()
        print(f"   ✅ Intent Classifier loaded in {time.time() - start:.2f}s")
        
        # Initialize Piper TTS
        print("📥 Loading Piper TTS...")
        start = time.time()
        self.tts_model = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
        print(f"   ✅ Piper TTS loaded in {time.time() - start:.2f}s")
        
        # Audio setup
        self.sample_rate = 16000
        self.chunk_size = 4000  # 0.25s chunks
        
        # Cart state
        self.cart = []
        
        print("\n✅ Voice Assistant ready!\n")
    
    async def run(self):
        """Main async run loop"""
        # Connect to Riva API
        if not await self.riva_client.connect():
            print("❌ Failed to connect to Riva API")
            return
        
        # Start receiving transcriptions in background
        receive_task = asyncio.create_task(self.riva_client.receive_transcriptions())
        
        # Start audio streaming
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("🎤 Listening...")
        
        current_utterance = []
        speech_detected = False
        silence_count = 0
        max_silence = 20  # ~1 second of silence
        
        try:
            while True:
                # Read audio
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # VAD detection with Vosk
                if self.vad_recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.vad_recognizer.Result())
                    if result.get("text"):
                        speech_detected = True
                        silence_count = 0
                
                # Send to Riva API
                if speech_detected:
                    await self.riva_client.send_audio(audio_np)
                    current_utterance.append(audio_np)
                    
                    # Check for end of utterance (silence)
                    result = json.loads(self.vad_recognizer.PartialResult())
                    if not result.get("partial"):
                        silence_count += 1
                    else:
                        silence_count = 0
                    
                    # Process utterance if silence detected
                    if silence_count >= max_silence:
                        # Get final transcription from queue
                        transcript = await self._get_final_transcription()
                        
                        if transcript:
                            print(f"\n🗣️  You: {transcript}")
                            
                            # Process intent
                            intent_result = self.intent_classifier.classify(transcript)
                            response = self._execute_intent(intent_result)
                            
                            print(f"🤖 Assistant: {response}")
                            self._speak(response)
                        
                        # Reset
                        current_utterance = []
                        speech_detected = False
                        silence_count = 0
                        self.vad_recognizer = KaldiRecognizer(self.vosk_model, 16000)
                        print("\n🎤 Listening...")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            await self.riva_client.close()
            receive_task.cancel()
    
    async def _get_final_transcription(self, timeout=2.0) -> Optional[str]:
        """Get final transcription from queue"""
        try:
            # Wait a bit for final result
            await asyncio.sleep(0.5)
            
            final_text = ""
            while not self.riva_client.transcription_queue.empty():
                result = await asyncio.wait_for(
                    self.riva_client.transcription_queue.get(),
                    timeout=timeout
                )
                if result["is_final"]:
                    final_text = result["text"]
                    break
            
            return final_text.strip() if final_text else None
            
        except asyncio.TimeoutError:
            return None
    
    def _execute_intent(self, intent_result):
        """Execute the classified intent"""
        intent_name = intent_result.intent.name
        entities = intent_result.intent.entities
        
        if intent_name == "add_to_cart":
            return self._handle_add_to_cart(entities)
        elif intent_name == "remove_from_cart":
            return self._handle_remove_from_cart(entities)
        elif intent_name == "view_cart":
            return self._handle_view_cart()
        elif intent_name == "clear_cart":
            return self._handle_clear_cart()
        elif intent_name == "checkout":
            return self._handle_checkout()
        elif intent_name == "search_products":
            query = entities.get('query', [''])[0]
            return f"Searching for products matching: {query}"
        elif intent_name == "get_time":
            from datetime import datetime
            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        elif intent_name == "get_date":
            from datetime import datetime
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
        elif intent_name == "greeting":
            return "Hello! How can I help you with your shopping today?"
        elif intent_name == "help":
            return "You can ask me to add items to cart, view cart, remove items, or checkout. What would you like to do?"
        elif intent_name == "goodbye":
            return "Goodbye! Have a great day!"
        else:
            return "I'm not sure what you mean. Can you try rephrasing that?"
    
    def _handle_add_to_cart(self, entities):
        """Handle add to cart intent"""
        items = entities.get('items', [])
        
        if not items:
            return "What would you like to add to your cart?"
        
        added = []
        for item in items:
            self.cart.append({'product': item.product, 'quantity': item.quantity})
            added.append(f"{item.quantity} {item.product}")
        
        items_str = ", ".join(added)
        return f"Added {items_str} to your cart."
    
    def _handle_remove_from_cart(self, entities):
        """Handle remove from cart intent"""
        items = entities.get('items', [])
        
        if not items:
            return "What would you like to remove from your cart?"
        
        removed = []
        for item in items:
            for cart_item in self.cart[:]:
                if item.product.lower() in cart_item['product'].lower():
                    self.cart.remove(cart_item)
                    removed.append(cart_item['product'])
                    break
        
        if removed:
            return f"Removed {', '.join(removed)} from your cart."
        return "I couldn't find those items in your cart."
    
    def _handle_view_cart(self):
        """Handle view cart intent"""
        if not self.cart:
            return "Your cart is empty."
        
        items = [f"{item['quantity']} {item['product']}" for item in self.cart]
        return f"Your cart contains: {', '.join(items)}"
    
    def _handle_clear_cart(self):
        """Handle clear cart intent"""
        count = len(self.cart)
        self.cart = []
        return f"Cleared {count} items from your cart."
    
    def _handle_checkout(self):
        """Handle checkout intent"""
        if not self.cart:
            return "Your cart is empty. Add some items first."
        
        items = [f"{item['quantity']} {item['product']}" for item in self.cart]
        total_items = sum(item['quantity'] for item in self.cart)
        
        self.cart = []
        return f"Processing checkout for {total_items} items: {', '.join(items)}. Thank you for your order!"
    
    def _speak(self, text: str):
        """Convert text to speech using Piper"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
            
            for audio_bytes in self.tts_model.synthesize_stream_raw(text):
                stream.write(audio_bytes)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"⚠️  TTS error: {e}")


async def main():
    """Main entry point"""
    print("=" * 70)
    print("🎙️  VOICE ASSISTANT (NVIDIA Riva WebSocket API)")
    print("=" * 70)
    print("✅ Cloud-based Riva ASR - Superior accuracy")
    print("✅ Real-time WebSocket streaming")
    print()
    
    # Get credentials
    token = os.environ.get('NVIDIA_RIVA_TOKEN')
    function_id = os.environ.get('NVIDIA_RIVA_FUNCTION_ID')
    
    if not token:
        token = input("Enter your NVIDIA Riva token: ").strip()
    
    if not function_id:
        function_id = input("Enter your NVIDIA Riva function ID: ").strip()
    
    if not token or not function_id:
        print("❌ Token and function ID are required")
        return
    
    print("\nCommands you can try:")
    print("  - 'I want two coffees'")
    print("  - 'add a muffin'")
    print("  - 'show me my cart'")
    print("  - 'remove the coffee'")
    print("  - 'checkout'")
    print("  - 'what time is it?'")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 70)
    print()
    
    assistant = VoiceAssistantRivaWebSocket(token, function_id)
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
