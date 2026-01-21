# Voice Assistant with LLM

Extended voice assistant with LLM reasoning and tool calling capabilities.

## Features
- 🎙️ **Fast STT**: Vosk (500-700ms)
- 🔊 **Fast TTS**: Piper (60-170ms)
- 🧠 **LLM Reasoning**: Ollama (local) or OpenAI (cloud)
- 🔧 **Tool Calling**: Execute functions based on user intent
- ⚡ **Total Latency**: ~2-4s with LLM layer

## Tools Available
1. **get_time** - Returns current time
2. **get_weather** - Gets weather for a location (mock for now)
3. **set_reminder** - Sets a reminder (mock for now)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose LLM Backend

#### Option A: Ollama (Local - Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Download a model (in another terminal)
ollama pull phi3       # 2.3GB, fast
# or
ollama pull llama3.2   # 2GB, good quality
```

#### Option B: OpenAI (Cloud)
```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### With Ollama (local):
```bash
python voice_assistant_llm.py --backend ollama --model phi3
```

### With OpenAI:
```bash
python voice_assistant_llm.py --backend openai --model gpt-4o-mini
```

### Simple echo (no LLM):
```bash
python voice_assistant.py
```

## Example Interactions

**User**: "What time is it?"
- 🔧 Tool: `get_time()` → executes
- 🗣️ Assistant: "The current time is 2:30 PM"

**User**: "What's the weather in New York?"
- 🔧 Tool: `get_weather(location='New York')` → executes
- 🗣️ Assistant: "The weather in New York is sunny and 72°F"

**User**: "Remind me to call mom in 5 minutes"
- 🔧 Tool: `set_reminder(task='call mom', time='5 minutes')` → executes
- 🗣️ Assistant: "Reminder set: call mom in 5 minutes"

**User**: "Tell me a joke"
- 🧠 LLM: generates response (no tool)
- 🗣️ Assistant: "Why did the programmer quit his job? Because he didn't get arrays!"

## Adding Custom Tools

Edit `voice_assistant_llm.py` and add to `_register_tools()`:

```python
@self.tools.register(
    name="my_tool",
    description="What this tool does",
    parameters={
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param_name"]
    }
)
def my_tool(param_name: str) -> str:
    print(f"🔧 [TOOL] my_tool(param_name='{param_name}')")
    # Your logic here
    return "Tool result"
```

## Performance

| Component | Latency |
|-----------|---------|
| STT | 500-700ms |
| LLM (Ollama phi3) | 500-1500ms |
| LLM (OpenAI gpt-4o-mini) | 300-800ms |
| TTS | 60-170ms |
| Playback | 500-1500ms |
| **Total** | **2-4 seconds** |

## Architecture

```
User Speech → Vosk STT → LLM (with tools) → Piper TTS → Audio Out
                ↓                ↓
           Text (500ms)    Response + Tool Calls
                              (500-1500ms)
```

## Notes

- Tool calls are printed to console with 🔧 prefix
- Ollama requires ~3GB RAM for phi3
- OpenAI requires internet and API key
- All tools are currently mocks - implement real logic as needed
