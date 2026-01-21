#!/usr/bin/env python3
"""
Voice Assistant with Qwen2.5-3B for better reasoning
Uses the same architecture but with larger, smarter LLM
"""

import sounddevice as sd
import numpy as np
import time
import re
from piper.voice import PiperVoice


class ToolRegistry:
    """Simple tool registry for function calling"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name, description, parameters):
        """Decorator to register a tool"""
        def decorator(func):
            self.tools[name] = {
                'function': func,
                'description': description,
                'parameters': parameters
            }
            return func
        return decorator
    
    def execute(self, name, args):
        """Execute a registered tool"""
        if name in self.tools:
            return self.tools[name]['function'](**args)
        raise ValueError(f"Unknown tool: {name}")
    
    def list_tools(self):
        """List all registered tools"""
        return list(self.tools.keys())


class QwenLLM:
    """Qwen2.5-3B LLM wrapper using llama-cpp-python with GGUF models"""
    
    def __init__(self, model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf"):
        print(f"Loading {model_path}...")
        start = time.time()
        
        from llama_cpp import Llama
        
        # Check if model exists
        import os
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found at: {model_path}")
            print("\nDownload with:")
            print(f"  wget -O {model_path} https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load GGUF model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Larger context for Qwen2.5
            n_threads=4,
            verbose=False,
        )
        
        print(f"✅ Qwen2.5-3B loaded in {time.time()-start:.2f}s")
    
    def generate(self, prompt: str, max_tokens: int = 80) -> str:
        """Generate response"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temp for more focused responses
            top_p=0.9,
            stop=["<|user|>", "<|assistant|>", "\n\nQ:", "\nQ:"],
            echo=False,
        )
        
        return response['choices'][0]['text'].strip()


class VoiceAssistantQwen:
    def __init__(self, model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf"):
        print("="*50)
        print("🤖 VOICE ASSISTANT (QWEN2.5-3B)")
        print("="*50)
        
        # Get script directory for relative paths
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.sr = 16000
        self.tools = ToolRegistry()
        
        # Register default tools
        self._register_tools()
        
    # Load Faster-Whisper
        print("\nLoading Faster-Whisper STT...")
        start = time.time()
        from faster_whisper import WhisperModel
    # Use small model for a better accuracy/speed balance
    self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print(f"✅ Faster-Whisper loaded in {time.time()-start:.2f}s")
        
        # Pre-load Piper voice
        print("Loading Piper TTS...")
        start = time.time()
        piper_path = os.path.join(script_dir, "models/piper/en_US-lessac-medium.onnx")
        self.piper_voice = PiperVoice.load(piper_path)
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        # Initialize Qwen LLM
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
        self.llm = QwenLLM(model_path)
        
        print("\n✅ Ready!\n")
    
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
            return f"Weather in {location}: sunny, 72 degrees"
        
        @self.tools.register(
            name="set_reminder",
            description="Set a reminder",
            parameters=["task", "time"]
        )
        def set_reminder(task: str, time: str) -> str:
            print(f"🔧 [TOOL] set_reminder(task='{task}', time='{time}')")
            return f"Reminder set: {task} in {time}"
        
        # POS-specific tools
        @self.tools.register(
            name="add_to_cart",
            description="Add item to shopping cart",
            parameters=["product", "quantity"]
        )
        def add_to_cart(product: str, quantity: str = "1") -> str:
            print(f"🔧 [TOOL] add_to_cart(product='{product}', quantity='{quantity}')")
            return f"Added {quantity} {product} to cart"
        
        @self.tools.register(
            name="remove_from_cart",
            description="Remove item from shopping cart",
            parameters=["product"]
        )
        def remove_from_cart(product: str) -> str:
            print(f"🔧 [TOOL] remove_from_cart(product='{product}')")
            return f"Removed {product} from cart"
        
        @self.tools.register(
            name="view_cart",
            description="View shopping cart contents",
            parameters=[]
        )
        def view_cart() -> str:
            print(f"🔧 [TOOL] view_cart()")
            return "Cart: 2x Coffee ($5.00), 1x Muffin ($3.50). Total: $13.50"
        
        @self.tools.register(
            name="search_products",
            description="Search for products",
            parameters=["query"]
        )
        def search_products(query: str) -> str:
            print(f"🔧 [TOOL] search_products(query='{query}')")
            results = {
                "coffee": "Found: Espresso ($3), Latte ($4.50), Cappuccino ($4)",
                "tea": "Found: Green Tea ($2.50), Black Tea ($2.50), Chai ($3)",
                "muffin": "Found: Blueberry Muffin ($3.50), Chocolate Muffin ($3.50)",
                "pasta": "Found: Spaghetti ($8), Penne ($7.50), Fettuccine ($8.50)",
            }
            for key in results:
                if key in query.lower():
                    return results[key]
            return f"Found products matching '{query}': Coffee, Tea, Pastries, Pasta"
        
        @self.tools.register(
            name="checkout",
            description="Complete purchase and checkout",
            parameters=[]
        )
        def checkout() -> str:
            print(f"🔧 [TOOL] checkout()")
            return "Processing checkout. Total: $13.50. Payment method?"
        
        @self.tools.register(
            name="clear_cart",
            description="Clear all items from cart",
            parameters=[]
        )
        def clear_cart() -> str:
            print(f"🔧 [TOOL] clear_cart()")
            return "Cart cleared"
    
    def _call_llm(self, user_message: str) -> str:
        """Call Qwen LLM with tool support"""
        start = time.time()
        
        # Optimized prompt for Qwen2.5
        prompt = f"""You are a helpful POS assistant. Respond with JSON for actions or natural text for conversation.

Available tools (respond with exact JSON format):
- {{"function":"get_time"}} - current time
- {{"function":"get_date"}} - today's date
- {{"function":"view_cart"}} - show cart
- {{"function":"checkout"}} - complete purchase
- {{"function":"clear_cart"}} - empty cart
- {{"function":"add_to_cart","product":"Coffee","quantity":"2"}} - add items
- {{"function":"remove_from_cart","product":"Coffee"}} - remove items
- {{"function":"search_products","query":"pasta"}} - search products

Examples:
User: what time is it?
Assistant: {{"function":"get_time"}}

User: add 2 coffees
Assistant: {{"function":"add_to_cart","product":"Coffee","quantity":"2"}}

User: do you have pasta?
Assistant: {{"function":"search_products","query":"pasta"}}

User: show my cart
Assistant: {{"function":"view_cart"}}

User: hello
Assistant: Hello! How can I help you today?

User: {user_message}
Assistant:"""
        
        response = self.llm.generate(prompt, max_tokens=50).strip()
        
        llm_ms = int((time.time() - start) * 1000)
        print(f"🧠 LLM: {llm_ms}ms")
        print(f"   Raw response: {response}")
        
        # Try to parse JSON for valid tool calls only
        if response.startswith("{") and "function" in response:
            try:
                import json
                # Extract ONLY the first complete JSON object
                json_str = response
                
                # Find first complete JSON object
                brace_count = 0
                json_end = -1
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    json_str = json_str[:json_end]
                    print(f"   📦 Extracted JSON: {json_str}")
                
                tool_call = json.loads(json_str)
                func_name = tool_call.get("function")
                
                # Validate: functions without parameters shouldn't have extra fields
                no_param_functions = ["get_time", "get_date", "view_cart", "checkout", "clear_cart"]
                if func_name in no_param_functions:
                    if len(tool_call) > 1:
                        print(f"   ⚠️  Function {func_name} should have no parameters, ignoring extras")
                        tool_call = {"function": func_name}
                
                # Execute registered tools
                if func_name == "get_time":
                    return self.tools.execute("get_time", {})
                elif func_name == "get_date":
                    return self.tools.execute("get_date", {})
                elif func_name == "get_weather":
                    location = tool_call.get("location", "New York")
                    return self.tools.execute("get_weather", {"location": location})
                elif func_name == "set_reminder":
                    task = tool_call.get("task", "something")
                    time_str = tool_call.get("time", "later")
                    return self.tools.execute("set_reminder", {"task": task, "time": time_str})
                elif func_name == "add_to_cart":
                    product = tool_call.get("product", "item")
                    quantity = tool_call.get("quantity", "1")
                    return self.tools.execute("add_to_cart", {"product": product, "quantity": quantity})
                elif func_name == "remove_from_cart":
                    product = tool_call.get("product", "item")
                    return self.tools.execute("remove_from_cart", {"product": product})
                elif func_name == "view_cart":
                    return self.tools.execute("view_cart", {})
                elif func_name == "search_products":
                    query = tool_call.get("query", "")
                    return self.tools.execute("search_products", {"query": query})
                elif func_name == "checkout":
                    return self.tools.execute("checkout", {})
                elif func_name == "clear_cart":
                    return self.tools.execute("clear_cart", {})
                else:
                    print(f"   ⚠️  Unknown function: {func_name}")
                    return "I'm not sure how to help with that."
            except Exception as e:
                print(f"   ⚠️  JSON parse error: {e}")
                pass
        
        # Return natural language response
        return response
    
    def record(self, duration=3):
        print(f"🎤 Recording {duration}s...")
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype=np.int16)
        sd.wait()
        return audio.flatten()
    
    def transcribe(self, audio):
        """Transcribe audio using Faster-Whisper"""
        start = time.time()
        
        # Convert int16 to float32
        audio_float = audio.astype(np.float32) / 32768.0
        
        # Transcribe
        segments, info = self.whisper_model.transcribe(
            audio_float,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect all segments
        text = " ".join([segment.text for segment in segments]).strip()
        
        elapsed_ms = int((time.time() - start) * 1000)
        return text, elapsed_ms
    
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
                
                print(f'📝 You: "{user_text}" (STT: {stt_ms}ms)')
                
                # 3. LLM processing with tool calling
                total_start = time.time()
                response_text = self._call_llm(user_text)
                
                print(f'💬 Assistant: "{response_text}"')
                
                # 4. Text-to-Speech
                synth_ms, play_ms = self.speak(response_text)
                print(f"🔊 TTS: {synth_ms}ms synth + {play_ms}ms play")
                
                # Total time (excluding recording)
                total_ms = int((time.time() - total_start) * 1000)
                print(f"⏱️  Total: {total_ms}ms ({total_ms/1000:.1f}s)\n")
                
        except KeyboardInterrupt:
            print("\n👋 Bye!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Assistant with Qwen2.5-3B")
    parser.add_argument(
        "--model",
        default="models/qwen2.5-3b-instruct-q4_k_m.gguf",
        help="Path to Qwen GGUF model file"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantQwen(model_path=args.model)
    assistant.run()


if __name__ == "__main__":
    main()
