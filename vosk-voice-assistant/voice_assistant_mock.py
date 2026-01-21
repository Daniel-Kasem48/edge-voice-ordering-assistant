#!/usr/bin/env python3
"""
Fast Voice Assistant with mock LLM for testing the flow.
No actual LLM - just hardcoded responses to test tool calling.
"""
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import time
import sys
from typing import Dict, List, Callable, Any
import re

# Import Piper directly (not CLI)
from piper import PiperVoice

SetLogLevel(-1)


class ToolRegistry:
    """Simple tool registry for LLM function calling"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register(self, name: str, description: str, parameters: List[str]):
        """Register a tool with its definition"""
        def decorator(func: Callable):
            self.tools[name] = func
            return func
        return decorator
    
    def execute(self, name: str, arguments: Dict) -> str:
        """Execute a tool by name"""
        if name not in self.tools:
            return f"Error: Tool '{name}' not found"
        
        try:
            return self.tools[name](**arguments)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"


class MockLLM:
    """Mock LLM for testing - pattern matches user intent"""
    
    def process(self, user_text: str, tools: ToolRegistry) -> str:
        """Simple pattern matching to simulate LLM reasoning"""
        user_text = user_text.lower()
        
        # Time queries
        if any(word in user_text for word in ["time", "clock"]):
            return tools.execute("get_time", {})
        
        # Date queries
        if any(word in user_text for word in ["date", "today", "day"]):
            return tools.execute("get_date", {})
        
        # Weather queries
        if "weather" in user_text:
            # Try to extract location
            location = "New York"  # Default
            words = user_text.split()
            if "in" in words:
                idx = words.index("in")
                if idx + 1 < len(words):
                    location = words[idx + 1].title()
            return tools.execute("get_weather", {"location": location})
        
        # Reminder
        if "remind" in user_text:
            # Simple extraction
            task = "something"
            time_str = "5 minutes"
            
            if "to" in user_text:
                parts = user_text.split("to", 1)
                if len(parts) > 1:
                    task_part = parts[1].split("in")[0].strip()
                    task = task_part if task_part else "something"
            
            if "in" in user_text:
                parts = user_text.split("in", 1)
                if len(parts) > 1:
                    time_str = parts[1].strip()
            
            return tools.execute("set_reminder", {"task": task, "time": time_str})
        
        # Greeting
        if any(word in user_text for word in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        
        # Gratitude
        if any(word in user_text for word in ["thank", "thanks"]):
            return "You're welcome!"
        
        # Default response
        return "I understand. I can help with time, date, weather, and reminders. What would you like to know?"


class VoiceAssistantMock:
    def __init__(self):
        print("="*50)
        print("🤖 VOICE ASSISTANT (MOCK LLM)")
        print("="*50)
        
        self.sr = 16000
        self.tools = ToolRegistry()
        self.llm = MockLLM()
        
        # Register default tools
        self._register_tools()
        
        # Load Vosk
        print("\nLoading Vosk STT...")
        start = time.time()
        self.vosk_model = Model("models/vosk-model-small-en-us-0.15")
        print(f"✅ Vosk loaded in {time.time()-start:.2f}s")
        
        # Pre-load Piper voice
        print("Loading Piper TTS...")
        start = time.time()
        self.piper_voice = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        print("\n✅ Ready! (Using mock LLM for testing)\n")
    
    def _register_tools(self):
        """Register available tools"""
        
        @self.tools.register(
            name="get_time",
            description="Get the current time",
            parameters=[]
        )
        def get_time() -> str:
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            print(f"🔧 [TOOL] get_time() -> {current_time}")
            return f"The current time is {current_time}"
        
        @self.tools.register(
            name="get_date",
            description="Get today's date",
            parameters=[]
        )
        def get_date() -> str:
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            print(f"🔧 [TOOL] get_date() -> {current_date}")
            return f"Today is {current_date}"
        
        @self.tools.register(
            name="get_weather",
            description="Get weather for a location",
            parameters=["location"]
        )
        def get_weather(location: str) -> str:
            print(f"🔧 [TOOL] get_weather(location='{location}')")
            # Mock response - replace with actual API
            return f"The weather in {location} is sunny and 72 degrees"
        
        @self.tools.register(
            name="set_reminder",
            description="Set a reminder",
            parameters=["task", "time"]
        )
        def set_reminder(task: str, time: str) -> str:
            print(f"🔧 [TOOL] set_reminder(task='{task}', time='{time}')")
            return f"Reminder set: {task} in {time}"
    
    def _call_llm(self, user_message: str) -> str:
        """Call mock LLM"""
        start = time.time()
        
        response = self.llm.process(user_message, self.tools)
        
        llm_ms = int((time.time() - start) * 1000)
        print(f"🧠 Mock LLM: {llm_ms}ms")
        
        return response
    
    def record(self, duration=3):
        print(f"🎤 Recording {duration}s...")
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype=np.int16)
        sd.wait()
        return audio.flatten()
    
    def transcribe(self, audio):
        start = time.time()
        rec = KaldiRecognizer(self.vosk_model, self.sr)
        rec.SetWords(True)
        
        chunk_size = 4000
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            rec.AcceptWaveform(chunk.tobytes())
        
        result = json.loads(rec.FinalResult())
        return result.get('text', '').strip(), int((time.time() - start) * 1000)
    
    def speak(self, text):
        """Synthesize and play using preloaded Piper"""
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
        print("Press Enter to speak, Ctrl+C to exit")
        print("\nTry saying:")
        print("  - What time is it?")
        print("  - What's the weather in Paris?")
        print("  - Remind me to call mom in 5 minutes")
        print("  - What's the date?\n")
        
        try:
            while True:
                input(">>> Press Enter...")
                
                # 1. Record audio
                audio = self.record(3)
                
                # 2. Speech-to-Text
                user_text, stt_ms = self.transcribe(audio)
                
                if not user_text:
                    print("❌ No speech detected\n")
                    continue
                
                print(f"📝 You: \"{user_text}\" (STT: {stt_ms}ms)")
                
                # 3. LLM Processing (mock)
                response_text = self._call_llm(user_text)
                print(f"💬 Assistant: \"{response_text}\"")
                
                # 4. Text-to-Speech
                synth_ms, play_ms = self.speak(response_text)
                
                # 5. Stats
                total = stt_ms + synth_ms + play_ms
                print(f"🔊 TTS: {synth_ms}ms synth + {play_ms}ms play")
                print(f"⏱️  Total: {total}ms ({total/1000:.1f}s)\n")
                
        except KeyboardInterrupt:
            print("\n👋 Bye!")


if __name__ == "__main__":
    assistant = VoiceAssistantMock()
    assistant.run()
