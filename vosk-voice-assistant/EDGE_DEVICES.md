# Voice Assistant for Edge Devices (RK3588/Jetson Nano)

Optimized voice assistant with tiny LLM that runs on ARM devices with limited resources.

## Supported Devices
- ✅ Firefly RK3588
- ✅ Jetson Nano
- ✅ Raspberry Pi 4/5
- ✅ Other ARM boards with 4GB+ RAM

## Model Options

### TinyLlama (Default - Recommended)
- **Size**: 1.1B parameters (~2.2GB)
- **RAM**: 3-4GB
- **Speed**: 2-5s inference on RK3588
- **Quality**: Good for simple tasks

### Phi-2 (Better Quality)
- **Size**: 2.7B parameters (~5.4GB)
- **RAM**: 6-8GB
- **Speed**: 5-10s inference on RK3588
- **Quality**: Excellent reasoning

### Qwen2-0.5B (Fastest)
- **Size**: 0.5B parameters (~1GB)
- **RAM**: 2GB
- **Speed**: <1s inference
- **Quality**: Basic but fast

## Performance Estimates

### RK3588 (6 TOPS NPU)
| Component | Latency | Notes |
|-----------|---------|-------|
| STT (Vosk) | 500-700ms | CPU |
| LLM (TinyLlama) | 2000-5000ms | CPU |
| TTS (Piper) | 60-170ms | CPU |
| **Total** | **3-6 seconds** | Without NPU acceleration |

### Jetson Nano (128 CUDA cores)
| Component | Latency | Notes |
|-----------|---------|-------|
| STT (Vosk) | 500-700ms | CPU |
| LLM (TinyLlama) | 3000-7000ms | CPU (can use GPU) |
| TTS (Piper) | 60-170ms | CPU |
| **Total** | **4-8 seconds** | Can improve with GPU |

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model (First Run)
The model downloads automatically on first run (~2.2GB for TinyLlama).

Or pre-download:
```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
```

### 3. Run
```bash
# Default: TinyLlama
python voice_assistant_edge.py

# Use Phi-2 (needs more RAM)
python voice_assistant_edge.py --model microsoft/phi-2

# Use Qwen2 (fastest)
python voice_assistant_edge.py --model Qwen/Qwen2-0.5B-Instruct
```

## Memory Requirements

| Model | Disk Space | RAM Usage | Minimum RAM |
|-------|------------|-----------|-------------|
| TinyLlama-1.1B | 2.2GB | 3-4GB | 4GB |
| Qwen2-0.5B | 1GB | 2GB | 3GB |
| Phi-2 | 5.4GB | 6-8GB | 8GB |

## Optimizations for ARM

1. **CPU-only inference** - No GPU required
2. **Float32** - Better compatibility on ARM
3. **Low memory mode** - Efficient loading
4. **Short responses** - Max 80 tokens
5. **Simple tool parsing** - Regex-based, no JSON overhead

## Tool Calling

Tools work via simple prompt engineering:

**User**: "What time is it?"
→ LLM outputs: `TOOL: get_time()`
→ System executes tool
→ Response: "The current time is 2:30 PM"

**Available Tools**:
- `get_time()` - Current time
- `get_date()` - Today's date  
- `get_weather(location="City")` - Weather info (mock)
- `set_reminder(task="...", time="...")` - Set reminder (mock)

## Adding Custom Tools

Edit `voice_assistant_edge.py` in `_register_tools()`:

```python
@self.tools.register(
    name="my_tool",
    description="What it does",
    parameters=["param1", "param2"]
)
def my_tool(param1: str, param2: str) -> str:
    print(f"🔧 [TOOL] my_tool({param1}, {param2})")
    # Your logic here
    return "Result"
```

The LLM will call it as: `TOOL: my_tool(param1="value1", param2="value2")`

## Performance Tips

### For RK3588:
1. Use TinyLlama or Qwen2-0.5B
2. Consider RKNN toolkit for NPU acceleration (advanced)
3. Close other applications

### For Jetson Nano:
1. Use TinyLlama with CUDA:
   ```python
   self.device = "cuda" if torch.cuda.is_available() else "cpu"
   ```
2. Use JetPack optimizations
3. Consider TensorRT (advanced)

### For Any Device:
1. Use quantized models (4-bit):
   ```bash
   pip install bitsandbytes
   ```
   Then modify code to use `load_in_4bit=True`
2. Reduce `max_tokens` to 50 for faster responses
3. Use Vosk small model (already used)

## Comparison with Cloud LLM

| Aspect | Edge (TinyLlama) | Cloud (GPT-4) |
|--------|------------------|---------------|
| Latency | 3-6s | 2-3s |
| Quality | Good | Excellent |
| Privacy | 100% offline | Sends to cloud |
| Cost | Free | $$ per call |
| Internet | Not needed | Required |

## Troubleshooting

**Out of Memory**: Use smaller model (Qwen2-0.5B) or add swap:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Too Slow**: 
- Use Qwen2-0.5B
- Reduce max_tokens to 50
- Consider quantization

**Model Download Fails**:
```bash
# Set cache dir with more space
export HF_HOME=/path/to/storage
```

## Example Usage

```bash
$ python voice_assistant_edge.py

==================================================
🤖 VOICE ASSISTANT (EDGE)
==================================================

Loading Vosk STT...
✅ Vosk loaded in 0.42s
Loading Piper TTS...
✅ Piper loaded in 0.93s
Loading TinyLlama/TinyLlama-1.1B-Chat-v1.0...
✅ LLM loaded in 8.45s

✅ Ready!

Press Enter to speak, Ctrl+C to exit

>>> Press Enter...
🎤 Recording 3s...
📝 You: "what time is it" (STT: 564ms)
🔧 [TOOL] get_time() -> 02:30 PM
🧠 LLM: 2847ms
💬 Assistant: "The current time is 02:30 PM"
🔊 TTS: 147ms synth + 1008ms play
⏱️  Total: 4566ms (4.6s)
```
