# Vosk Models

This directory should contain Vosk speech recognition models. Models are not included in git due to their large size (300+ MB).

## Download Models

### Small Model (68MB) - Faster, less accurate
```bash
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
```

### Large Model (205MB) - Better accuracy
```bash
cd models
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
unzip vosk-model-en-us-0.22-lgraph.zip
rm vosk-model-en-us-0.22-lgraph.zip
```

## Piper TTS Model

For text-to-speech:
```bash
cd models
mkdir -p piper
cd piper
wget https://github.com/rhasspy/piper/releases/download/v0.0.2/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v0.0.2/en_US-lessac-medium.onnx.json
```

## Model Structure

```
models/
├── vosk-model-small-en-us-0.15/     # 68MB - Fast
├── vosk-model-en-us-0.22-lgraph/    # 205MB - Accurate
└── piper/
    ├── en_US-lessac-medium.onnx      # TTS model
    └── en_US-lessac-medium.onnx.json # TTS config
```

## More Models

Browse all available models at:
- Vosk: https://alphacephei.com/vosk/models
- Piper: https://github.com/rhasspy/piper/releases
