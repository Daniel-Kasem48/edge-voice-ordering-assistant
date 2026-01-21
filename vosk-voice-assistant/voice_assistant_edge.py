#!/usr/bin/env python3
"""
Fast Voice Assistant with tiny LLM for edge devices (RK3588/Jetson Nano).
Uses Faster-Whisper for optimized STT.
"""
import sounddevice as sd
import numpy as np
import json
import time
import sys
import subprocess
import select
from typing import Dict, List, Callable, Any
import re

# Import Piper directly (not CLI)
from piper import PiperVoice


class ToolRegistry:
    """Simple tool registry for LLM function calling"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: List[str] = []
    
    def register(self, name: str, description: str, parameters: List[str]):
        """Register a tool with its definition"""
        def decorator(func: Callable):
            self.tools[name] = func
            param_str = ", ".join(parameters) if parameters else ""
            self.tool_descriptions.append(f"{name}({param_str}): {description}")
            return func
        return decorator
    
    def get_tools_prompt(self) -> str:
        """Get formatted tools description for prompt"""
        return "\n".join([f"- {desc}" for desc in self.tool_descriptions])
    
    def execute(self, name: str, arguments: Dict) -> str:
        """Execute a tool by name"""
        if name not in self.tools:
            return f"Error: Tool '{name}' not found"
        
        try:
            return self.tools[name](**arguments)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"


class TinyLLM:
    """Lightweight LLM wrapper using llama-cpp-python with GGUF models"""
    
    def __init__(self, model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        print(f"Loading {model_path}...")
        start = time.time()
        
        from llama_cpp import Llama
        
        # Check if model exists, if not provide download instructions
        import os
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found at: {model_path}")
            print("\nDownload with:")
            print(f"  mkdir -p models")
            print(f"  curl -L -o {model_path} 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'")
            print("\nModel size: ~669MB (4-bit quantized)")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load GGUF model (already compressed to ~600MB)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,  # Context window
            n_threads=4,  # CPU threads
            verbose=False,
        )
        
        print(f"✅ LLM loaded in {time.time()-start:.2f}s")
    
    def generate(self, prompt: str, max_tokens: int = 80) -> str:
        """Generate response"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["<|user|>", "<|assistant|>", "\n\n", "\nQ:", "Q:"],
            echo=False,
        )
        
        return response['choices'][0]['text'].strip()


class VoiceAssistantEdge:
    def __init__(self, model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        print("="*50)
        print("🤖 VOICE ASSISTANT (EDGE)")
        print("="*50)
        
        # Get script directory for relative paths
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.sr = 16000
        self.use_parec = True  # Use PulseAudio native recording (respects pactl settings)
        self.tools = ToolRegistry()

        # Audio device selection
        # Try actual ALSA mic devices, with fallback to system default
        try:
            self.input_device = None
            # Try device 2 first (often the working internal mic on laptops)
            for idx in [2, 1]:
                try:
                    dev = sd.query_devices(idx)
                    name = str(dev.get("name", "")).lower()
                    if ("mic" in name or "analog" in name) and "monitor" not in name:
                        if dev.get("max_input_channels", 0) > 0:
                            # Quick test if device is actually available
                            test_stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', device=idx, blocksize=480)
                            test_stream.close()
                            self.input_device = idx
                            break
                except Exception:
                    continue
        except Exception:
            self.input_device = None

        if self.input_device is not None:
            try:
                print(f"🎙️  Using input device: {self.input_device} ({sd.query_devices(self.input_device)['name']})")
            except Exception:
                print(f"🎙️  Using input device: {self.input_device}")
        else:
            try:
                default_in = sd.default.device[0]
                print(f"🎙️  Using system default: device {default_in} ({sd.query_devices(default_in)['name']})")
            except Exception:
                print("🎙️  Using system default input device")

        # ChatGPT-style always-on listening config (STT quality first)
        # We do light VAD/energy detection to segment utterances, then run Whisper once per utterance.
        self.listen_chunk_ms = 30  # audio callback chunk size
        self.max_utterance_s = 12.0
        self.min_utterance_s = 0.35
        self.end_silence_s = 0.85
        self.start_trigger_s = 0.10
        # Base RMS threshold; we also auto-calibrate a noise floor and use max(base, noise*mult)
        self.vad_rms_threshold = 0.20  # fixed threshold for your mic's ambient level
        self.vad_noise_multiplier = 1.5  # lower = more sensitive (speech must be 1.5x louder than ambient)
        self.vad_hangover_s = 0.20  # keep recording briefly after dropping below threshold
        self.vad_calibration_s = 0.0  # disable auto-calibration (use fixed threshold)
        self.vad_calibration_s = 1.0  # measure ambient RMS at start of each listen
        self.vad_debug = True  # set False once it's working well
        
        # Register default tools
        self._register_tools()

        # Load Faster-Whisper
        print("\nLoading Faster-Whisper STT...")
        start = time.time()
        from faster_whisper import WhisperModel
        # Use small model for better accuracy than base (larger + slower)
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print(f"✅ Faster-Whisper loaded in {time.time()-start:.2f}s")
        
        # Pre-load Piper voice
        print("Loading Piper TTS...")
        start = time.time()
        piper_path = os.path.join(script_dir, "models/piper/en_US-lessac-medium.onnx")
        self.piper_voice = PiperVoice.load(piper_path)
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        # Initialize tiny LLM (DISABLED for speed testing)
        # if not os.path.isabs(model_path):
        #     model_path = os.path.join(script_dir, model_path)
        # self.llm = TinyLLM(model_path)
        print("⚠️  LLM disabled - echo mode only")
        
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
            # Mock response - replace with actual API
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
            # Mock cart - in production, this would interact with your cart system
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
            # Mock cart contents - replace with actual cart data
            return "Cart: 2x Coffee ($5.00), 1x Muffin ($3.50). Total: $13.50"
        
        @self.tools.register(
            name="search_products",
            description="Search for products",
            parameters=["query"]
        )
        def search_products(query: str) -> str:
            print(f"🔧 [TOOL] search_products(query='{query}')")
            # Mock search results - replace with actual search
            results = {
                "coffee": "Found: Espresso ($3), Latte ($4.50), Cappuccino ($4)",
                "tea": "Found: Green Tea ($2.50), Black Tea ($2.50), Chai ($3)",
                "muffin": "Found: Blueberry Muffin ($3.50), Chocolate Muffin ($3.50)",
            }
            for key in results:
                if key in query.lower():
                    return results[key]
            return f"Found products matching '{query}': Coffee, Tea, Pastries"
        
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
    
    def _parse_tool_call(self, text: str) -> tuple:
        """Parse tool call from LLM response"""
        # Look for pattern: TOOL: function_name(arg1="value1", arg2="value2")
        pattern = r'TOOL:\s*(\w+)\((.*?)\)'
        match = re.search(pattern, text)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Parse arguments
            args = {}
            if args_str.strip():
                # Simple parsing: key="value" or key='value'
                arg_pattern = r'(\w+)=["\']([^"\']+)["\']'
                for arg_match in re.finditer(arg_pattern, args_str):
                    args[arg_match.group(1)] = arg_match.group(2)
            
            return tool_name, args
        
        return None, None
    
    def _call_llm(self, user_message: str) -> str:
        """Call tiny LLM with tool support"""
        start = time.time()
        
        # LLM DISABLED - Just echo back
        response = f"You said: {user_message}"
        
        print(f"🧠 LLM: DISABLED")
        print(f"   Echo: {response}")
        
        return response
        
        llm_ms = int((time.time() - start) * 1000)
        print(f"🧠 LLM: {llm_ms}ms")
        print(f"   Raw response: {response}")
        
        # Try to parse JSON for valid tool calls only
        if response.startswith("{") and "function" in response:
            try:
                import json
                # Extract ONLY the first JSON object, ignore everything after
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
                    # Check if there are unexpected parameters
                    if len(tool_call) > 1:  # More than just "function" key
                        print(f"   ⚠️  Function {func_name} should have no parameters, ignoring extras")
                        tool_call = {"function": func_name}  # Clean it up
                
                # Only execute if it's a registered tool
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
                    # Invalid function name - treat as natural response
                    print(f"   ⚠️  Unknown function: {func_name}, using natural response")
                    return "I'm not sure how to help with that. Try asking about time, weather, or reminders."
            except Exception as e:
                print(f"   ⚠️  JSON parse error: {e}")
                pass  # If JSON parsing fails, return response as-is
        
        # Return natural language response
        return response
    
    def record(self, duration=3):
        print(f"🎤 Recording {duration}s...")
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype=np.int16)
        sd.wait()
        return audio.flatten()

    def _record_silent(self, duration: float) -> np.ndarray:
        """Record audio without printing (used for always-on wake word listening)."""
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype=np.int16)
        sd.wait()
        return audio.flatten()

    def _rms(self, audio_float: np.ndarray) -> float:
        if audio_float.size == 0:
            return 0.0
        # Stable RMS
        return float(np.sqrt(np.mean(np.square(audio_float), dtype=np.float64)))

    def _capture_utterance(self) -> np.ndarray | None:
        """Capture one speech utterance using a lightweight RMS-based VAD.

        Returns int16 mono audio at self.sr (16kHz), or None if nothing was captured.
        """
        if self.use_parec:
            return self._capture_utterance_parec()
        else:
            return self._capture_utterance_sounddevice()

    def _capture_utterance_parec(self) -> np.ndarray | None:
        """Capture using parec (PulseAudio) with VAD."""
        # Start parec subprocess
        proc = subprocess.Popen([
            'parec',
            '--rate=16000',
            '--channels=1',
            '--format=s16le',
            '--latency-msec=30'
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        chunk_size = int(self.sr * (self.listen_chunk_ms / 1000.0)) * 2  # 2 bytes per sample
        start_trigger_frames = int(self.sr * self.start_trigger_s)
        end_silence_frames = int(self.sr * self.end_silence_s)
        hangover_frames = int(self.sr * self.vad_hangover_s)
        max_frames = int(self.sr * self.max_utterance_s)

        # No calibration with parec (real mic should have low noise floor)
        dyn_threshold = 0.02

        ring: list[np.ndarray] = []
        ring_len = 0
        frames: list[np.ndarray] = []
        recorded = 0
        speaking = False
        above_count = 0
        below_count = 0
        hangover = 0
        tick = 0

        try:
            while True:
                raw = proc.stdout.read(chunk_size)
                if not raw:
                    break
                
                chunk_i16 = np.frombuffer(raw, dtype=np.int16)
                chunk = chunk_i16.astype(np.float32) / 32768.0
                rms = self._rms(chunk)
                is_voice = rms >= dyn_threshold

                if self.vad_debug:
                    tick += 1
                    if tick % 16 == 0:
                        mode = "SPEAK" if speaking else "WAIT"
                        print(f"   🔊 rms={rms:.5f} thr={dyn_threshold:.5f} voice={int(is_voice)} {mode}")

                ring.append(chunk_i16)
                ring_len += chunk_i16.shape[0]
                while ring_len > int(self.sr * 0.25):
                    dropped = ring.pop(0)
                    ring_len -= dropped.shape[0]

                if not speaking:
                    if is_voice:
                        above_count += chunk_i16.shape[0]
                    else:
                        above_count = 0

                    if above_count >= start_trigger_frames:
                        speaking = True
                        frames.extend(ring)
                        recorded = sum(f.shape[0] for f in frames)
                        ring.clear()
                        ring_len = 0
                        below_count = 0
                        hangover = 0
                else:
                    frames.append(chunk_i16)
                    recorded += chunk_i16.shape[0]

                    if is_voice:
                        below_count = 0
                        hangover = hangover_frames
                    else:
                        below_count += chunk_i16.shape[0]
                        hangover = max(hangover - chunk_i16.shape[0], 0)

                    if below_count >= end_silence_frames and hangover == 0:
                        break

                    if recorded >= max_frames:
                        break

        finally:
            proc.terminate()
            proc.wait()

        if not frames:
            return None

        audio_i16 = np.concatenate(frames)
        dur = audio_i16.shape[0] / self.sr
        if dur < self.min_utterance_s:
            return None

        return audio_i16

    def _capture_utterance_sounddevice(self) -> np.ndarray | None:
        """Original sounddevice-based capture (fallback)."""
        chunk_frames = int(self.native_sr * (self.listen_chunk_ms / 1000.0))
        start_trigger_frames = int(self.native_sr * self.start_trigger_s)
        end_silence_frames = int(self.native_sr * self.end_silence_s)
        hangover_frames = int(self.native_sr * self.vad_hangover_s)
        max_frames = int(self.native_sr * self.max_utterance_s)

        # Calibrate ambient noise RMS so threshold adapts per environment/mic gain.
        noise_rms = 0.0
        if self.vad_calibration_s > 0:
            calib_chunks = max(1, int(self.vad_calibration_s / (self.listen_chunk_ms / 1000.0)))
            calib_rms_vals: list[float] = []
            try:
                with sd.InputStream(
                    samplerate=self.native_sr,
                    channels=1,
                    dtype="float32",
                    blocksize=chunk_frames,
                ) as stream:
                    for _ in range(calib_chunks):
                        data, _overflowed = stream.read(chunk_frames)
                        chunk = data[:, 0]
                        calib_rms_vals.append(self._rms(chunk))
                        time.sleep(0.001)
            except Exception:
                calib_rms_vals = []
            if calib_rms_vals:
                # Clamp to sane range and use median.
                calib_rms_vals = [min(max(v, 0.0), 1.0) for v in calib_rms_vals]
                calib_rms_vals.sort()
                noise_rms = calib_rms_vals[len(calib_rms_vals) // 2]

        # If calibration looks wrong (e.g., near-1.0 constant), ignore it.
        if noise_rms >= 0.5 or self.vad_calibration_s == 0.0:
            noise_rms = 0.0

        dyn_threshold = max(self.vad_rms_threshold, noise_rms * self.vad_noise_multiplier)
        if self.vad_debug:
            print(f"   🔎 VAD calib: noise_rms={noise_rms:.5f} base={self.vad_rms_threshold:.5f} dyn={dyn_threshold:.5f}")

        ring: list[np.ndarray] = []
        ring_len = 0

        frames: list[np.ndarray] = []
        recorded = 0
        speaking = False
        above_count = 0
        below_count = 0
        hangover = 0

        tick = 0

        def callback(indata, _frames, _time, status):
            nonlocal ring, ring_len, frames, recorded, speaking, above_count, below_count, hangover
            nonlocal tick
            if status:
                # Ignore minor callback status warnings
                pass

            # Some backends can return float32 slightly outside [-1..1]; clamp to keep RMS meaningful.
            chunk = np.clip(indata[:, 0], -1.0, 1.0).copy()
            rms = self._rms(chunk)
            is_voice = rms >= dyn_threshold

            if self.vad_debug:
                tick += 1
                # Print about 2x per second (30ms blocks -> ~33/sec)
                if tick % 16 == 0:
                    mode = "SPEAK" if speaking else "WAIT"
                    print(f"   🔊 rms={rms:.5f} thr={dyn_threshold:.5f} voice={int(is_voice)} {mode}")

            # Keep a short pre-roll buffer so we don't clip the first phoneme.
            ring.append(chunk)
            ring_len += chunk.shape[0]
            while ring_len > int(self.native_sr * 0.25):
                dropped = ring.pop(0)
                ring_len -= dropped.shape[0]

            if not speaking:
                if is_voice:
                    above_count += chunk.shape[0]
                else:
                    above_count = 0

                if above_count >= start_trigger_frames:
                    speaking = True
                    # flush pre-roll
                    frames.extend(ring)
                    recorded = sum(f.shape[0] for f in frames)
                    ring.clear()
                    ring_len = 0
                    below_count = 0
                    hangover = 0
            else:
                frames.append(chunk)
                recorded += chunk.shape[0]

                if is_voice:
                    below_count = 0
                    hangover = hangover_frames
                else:
                    below_count += chunk.shape[0]
                    hangover = max(hangover - chunk.shape[0], 0)

                # Stop when we have enough trailing silence (and hangover elapsed)
                if below_count >= end_silence_frames and hangover == 0:
                    raise sd.CallbackStop

                if recorded >= max_frames:
                    raise sd.CallbackStop

        try:
            with sd.InputStream(
                samplerate=self.native_sr,
                channels=1,
                dtype="float32",
                blocksize=chunk_frames,
                callback=callback,
                device=self.input_device,
            ):
                # Block until callback stops
                while True:
                    time.sleep(0.05)
        except sd.CallbackStop:
            pass

        if not frames:
            return None

        audio = np.clip(np.concatenate(frames), -1.0, 1.0)
        dur = audio.shape[0] / self.native_sr
        if dur < self.min_utterance_s:
            return None

        # Resample from native_sr (48kHz) to 16kHz for Whisper
        from scipy import signal
        num_samples_16k = int(len(audio) * self.sr / self.native_sr)
        audio = signal.resample(audio, num_samples_16k)

        # Convert float32 [-1..1] to int16
        audio_i16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        return audio_i16
    
    def transcribe(self, audio):
        """Transcribe audio using Faster-Whisper"""
        start = time.time()
        
        # Convert int16 to float32 as required by Faster-Whisper
        audio_float = audio.astype(np.float32) / 32768.0
        
        # Transcribe
        segments, info = self.whisper_model.transcribe(
            audio_float,
            language="en",
            # STT quality first; bump beam a little (costs some CPU)
            beam_size=3,
            best_of=3,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=650,
                speech_pad_ms=200,
            ),
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
        print("Always-listening mode: talk naturally. Ctrl+C to exit.\n")

        try:
            while True:
                print("\n👂 Listening…")

                audio = self._capture_utterance()
                if audio is None:
                    continue

                # Speech-to-Text (one shot per utterance)
                user_text, stt_ms = self.transcribe(audio)

                if not user_text:
                    print("❌ No speech detected")
                    continue

                print(f"📝 You: \"{user_text}\" (STT: {stt_ms}ms)")

                # For now you only care about STT quality.
                # Keep the loop conversational but lightweight.
                response_text = self._call_llm(user_text)
                print(f"💬 Assistant: \"{response_text}\"")

                synth_ms, play_ms = self.speak(response_text)

                total = stt_ms + synth_ms + play_ms
                print(f"🔊 TTS: {synth_ms}ms synth + {play_ms}ms play")
                print(f"⏱️  Total: {total}ms ({total/1000:.1f}s)")

        except KeyboardInterrupt:
            print("\n👋 Bye!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Assistant for Edge Devices")
    parser.add_argument(
        "--model",
        default="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        help="Path to GGUF model file"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantEdge(model_path=args.model)
    assistant.run()


if __name__ == "__main__":
    main()
