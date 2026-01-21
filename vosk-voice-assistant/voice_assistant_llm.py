#!/usr/bin/env python3
"""
Fast Voice Assistant with LLM and tool calling.
Supports both local (Ollama) and cloud (OpenAI) LLMs.
"""
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import time
import sys
from typing import Dict, List, Callable, Any
import os

# Import Piper directly (not CLI)
from piper import PiperVoice

SetLogLevel(-1)


class ToolRegistry:
    """Simple tool registry for LLM function calling"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict] = []
    
    def register(self, name: str, description: str, parameters: Dict):
        """Register a tool with its definition"""
        def decorator(func: Callable):
            self.tools[name] = func
            self.tool_definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            })
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


class VoiceAssistant:
    def __init__(self, llm_backend="ollama", model_name="phi3"):
        print("="*50)
        print("🤖 VOICE ASSISTANT WITH LLM")
        print("="*50)
        
        self.sr = 16000
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.tools = ToolRegistry()
        
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
        
        # Initialize LLM
        print(f"Initializing {llm_backend.upper()} LLM ({model_name})...")
        self._init_llm()
        
        print("\n✅ Ready!\n")
    
    def _register_tools(self):
        """Register available tools"""
        
        @self.tools.register(
            name="get_time",
            description="Get the current time",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        def get_time() -> str:
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            print(f"🔧 [TOOL] get_time() -> {current_time}")
            return f"The current time is {current_time}"
        
        @self.tools.register(
            name="get_weather",
            description="Get weather information for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }
        )
        def get_weather(location: str) -> str:
            print(f"🔧 [TOOL] get_weather(location='{location}')")
            # Mock response - replace with actual API
            return f"The weather in {location} is sunny and 72°F"
        
        @self.tools.register(
            name="set_reminder",
            description="Set a reminder for later",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to be reminded about"
                    },
                    "time": {
                        "type": "string",
                        "description": "When to remind (e.g., '5 minutes', '2 hours')"
                    }
                },
                "required": ["task", "time"]
            }
        )
        def set_reminder(task: str, time: str) -> str:
            print(f"🔧 [TOOL] set_reminder(task='{task}', time='{time}')")
            return f"Reminder set: {task} in {time}"
    
    def _init_llm(self):
        """Initialize LLM backend"""
        if self.llm_backend == "ollama":
            try:
                import requests
                # Test Ollama connection
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print(f"✅ Ollama connected")
                else:
                    print("⚠️  Ollama not running - start with: ollama serve")
            except Exception as e:
                print(f"⚠️  Ollama error: {e}")
        
        elif self.llm_backend == "openai":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print(f"✅ OpenAI connected")
            except Exception as e:
                print(f"⚠️  OpenAI error: {e}")
    
    def _call_llm(self, user_message: str) -> str:
        """Call LLM with tool support"""
        start = time.time()
        
        if self.llm_backend == "ollama":
            response_text = self._call_ollama(user_message)
        elif self.llm_backend == "openai":
            response_text = self._call_openai(user_message)
        else:
            response_text = "Error: Unknown LLM backend"
        
        llm_ms = int((time.time() - start) * 1000)
        print(f"🧠 LLM: {llm_ms}ms")
        
        return response_text
    
    def _call_ollama(self, user_message: str) -> str:
        """Call Ollama with tool calling"""
        import requests
        
        # Ollama doesn't support native function calling yet,
        # so we use prompt engineering
        tools_description = "\n".join([
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in self.tools.tool_definitions
        ])
        
        system_prompt = f"""You are a helpful voice assistant. You have access to these tools:

{tools_description}

If the user's request requires a tool, respond with JSON in this format:
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}

Otherwise, respond naturally in conversational text."""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                llm_response = response.json()["response"].strip()
                
                # Check if LLM wants to use a tool
                if llm_response.startswith("{") and "tool" in llm_response:
                    try:
                        tool_call = json.loads(llm_response)
                        tool_name = tool_call["tool"]
                        tool_args = tool_call.get("arguments", {})
                        
                        # Execute tool
                        tool_result = self.tools.execute(tool_name, tool_args)
                        return tool_result
                    except:
                        pass  # If JSON parsing fails, return as-is
                
                return llm_response
            else:
                return "Sorry, I couldn't process that request."
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_openai(self, user_message: str) -> str:
        """Call OpenAI with native function calling"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": user_message}
                ],
                tools=self.tools.tool_definitions,
                tool_choice="auto",
                max_tokens=150
            )
            
            message = response.choices[0].message
            
            # Check if LLM wants to call a tool
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                tool_result = self.tools.execute(tool_name, tool_args)
                return tool_result
            
            return message.content
        
        except Exception as e:
            return f"Error: {str(e)}"
    
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
        print("Press Enter to speak, Ctrl+C to exit\n")
        
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
                
                # 3. LLM Processing
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Assistant with LLM")
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        default="phi3",
        help="Model name (default: phi3 for ollama, gpt-4o-mini for openai)"
    )
    
    args = parser.parse_args()
    
    # Set default OpenAI model if not specified
    if args.backend == "openai" and args.model == "phi3":
        args.model = "gpt-4o-mini"
    
    assistant = VoiceAssistant(llm_backend=args.backend, model_name=args.model)
    assistant.run()


if __name__ == "__main__":
    main()
