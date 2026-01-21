#!/usr/bin/env python3
"""
Hybrid voice assistant:
- Vosk for speech detection (VAD only, no transcription)
- NVIDIA NeMo for accurate transcription
- Piper for TTS

This combines Vosk's reliable VAD with NeMo's efficient ASR.
"""
import pyaudio
import json
import time
import os
import numpy as np
from vosk import Model, KaldiRecognizer, SetLogLevel
import nemo.collections.asr as nemo_asr
from piper import PiperVoice
import sounddevice as sd
import torch

# Reduce Vosk's verbose logging
SetLogLevel(-1)


class VoiceAssistantNemo:
    def __init__(self, 
                 vosk_model_path="models/vosk-model-small-en-us-0.15",
                 nemo_model="stt_en_quartznet15x5",
                 silence_duration=1.0):
        """
        Args:
            vosk_model_path: Path to Vosk model (used only for VAD)
            nemo_model: NeMo ASR model name (from NGC or local path)
                Options: stt_en_quartznet15x5, stt_en_jasper10x5dr
            silence_duration: Seconds of silence before processing (default 1.0)
        """
        print("="*50)
        print("🤖 VOICE ASSISTANT (Hybrid: Vosk VAD + NeMo ASR)")
        print("="*50)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load Vosk model (for VAD only)
        print("\n📡 Loading Vosk (VAD only)...")
        start = time.time()
        if not os.path.isabs(vosk_model_path):
            vosk_model_path = os.path.join(script_dir, vosk_model_path)
        self.vosk_model = Model(vosk_model_path)
        print(f"✅ Vosk VAD loaded in {time.time()-start:.2f}s")
        
        # Load NeMo ASR model
        print(f"\n🎯 Loading NeMo ASR ({nemo_model})...")
        start = time.time()
        
        # Check if it's a local path or NGC model name
        if os.path.exists(nemo_model):
            self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(nemo_model)
        else:
            # Download from NGC (first time will take a while)
            self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(
                model_name=nemo_model
            )
        
        # Put model in eval mode
        self.asr_model.eval()
        
        # Move to CPU (or CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_model = self.asr_model.to(device)
        self.device = device
        
        print(f"✅ NeMo ASR loaded in {time.time()-start:.2f}s (device: {device})")
        
        # Load Piper TTS
        print("\n🔊 Loading Piper TTS...")
        start = time.time()
        piper_path = os.path.join(script_dir, "models/piper/en_US-lessac-medium.onnx")
        self.piper_voice = PiperVoice.load(piper_path)
        print(f"✅ Piper loaded in {time.time()-start:.2f}s")
        
        # Audio setup
        self.sample_rate = 16000
        self.chunk_size = 4000
        self.silence_duration = silence_duration
        self.silence_threshold_chunks = int(silence_duration * self.sample_rate / self.chunk_size)
        
        print(f"\n⏱️  Silence threshold: {silence_duration}s (~{self.silence_threshold_chunks} chunks)")
        print("\n✅ Ready!")
    
    def _transcribe_audio(self, audio_data):
        """
        Transcribe audio using NeMo ASR.
        
        Args:
            audio_data: numpy array of int16 audio samples
        
        Returns:
            transcribed text or None
        """
        # Convert int16 to float32 normalized to [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Transcribe with NeMo (returns list of Hypothesis objects)
        with torch.no_grad():
            transcriptions = self.asr_model.transcribe(
                audio=[audio_float],
                batch_size=1
            )
        
        # Extract text from Hypothesis object
        if isinstance(transcriptions, list) and len(transcriptions) > 0:
            hypothesis = transcriptions[0]
            # Check if it has a 'text' attribute (Hypothesis object)
            if hasattr(hypothesis, 'text'):
                text = hypothesis.text.strip()
            else:
                text = str(hypothesis).strip()
        else:
            text = None
        
        return text if text else None
    
    def speak(self, text):
        """Synthesize and play using Piper"""
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
        print("\n🎤 Hybrid mode: Vosk detects speech → NeMo transcribes")
        print("Speak naturally. Ctrl+C to exit.\n")
        
        # Create Vosk recognizer (VAD only)
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        recognizer.SetWords(False)  # Don't need word details from Vosk
        
        # Setup PyAudio
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=4,  # Use PulseAudio device
                frames_per_buffer=self.chunk_size
            )
            stream.start_stream()
            
            print("👂 Listening…\n")
            
            # Buffer to accumulate audio for NeMo
            audio_buffer = []
            silence_chunks = 0  # Track consecutive silent chunks
            speech_started = False  # Track if we've detected speech
            last_speech_time = 0  # Track when we last detected speech
            speech_start_time = 0  # Track when speech first started
            
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Store audio for later transcription
                audio_buffer.append(audio_chunk)
                
                # Check if Vosk has a final result (end of utterance)
                has_final = recognizer.AcceptWaveform(data)
                
                # Get partial result to check if speech is ongoing
                partial_result = json.loads(recognizer.PartialResult())
                partial_text = partial_result.get('partial', '').strip()
                
                # Detect speech: either partial text exists OR we just had final result
                is_speaking = len(partial_text) > 0
                
                if is_speaking:
                    # Speech is ongoing
                    if not speech_started:
                        speech_started = True
                        speech_start_time = time.time()
                        print(f"\r🎤 Detecting speech...", end='', flush=True)
                    silence_chunks = 0
                    last_speech_time = time.time()
                
                # Always increment silence counter when speech has started
                # This ensures we eventually process even if Vosk gets stuck
                if speech_started and not is_speaking:
                    silence_chunks += 1
                    
                # Check if we should process (silence OR absolute timeout from speech start)
                if speech_started:
                    time_since_speech = time.time() - last_speech_time
                    time_since_start = time.time() - speech_start_time
                    should_process = (silence_chunks >= self.silence_threshold_chunks) or (time_since_start > 10.0)
                    
                    if should_process:
                        print("\r🔄 Processing with NeMo...                    ")
                        
                        # Get final result if available
                        if has_final:
                            final_result = json.loads(recognizer.Result())
                        
                        # Concatenate all buffered audio
                        full_audio = np.concatenate(audio_buffer)
                        
                        # Transcribe with NeMo
                        start = time.time()
                        text = self._transcribe_audio(full_audio)
                        nemo_time = int((time.time() - start) * 1000)
                        
                        if text:
                            print(f"📝 You: \"{text}\" ({nemo_time}ms)")
                            
                            # Echo response
                            response = f"You said: {text}"
                            print(f"💬 Assistant: \"{response}\"")
                            
                            synth_ms, play_ms = self.speak(response)
                            print(f"🔊 TTS: {synth_ms}ms + {play_ms}ms\n")
                        else:
                            print("🚫 No speech detected by NeMo\n")
                        
                        # Reset state
                        audio_buffer = []
                        silence_chunks = 0
                        speech_started = False
                        print("👂 Listening…\n")
                
                # Keep only last 16 seconds of audio to prevent memory overflow
                if len(audio_buffer) > 400:  # 400 chunks * 4000 samples = 16 seconds
                    audio_buffer = audio_buffer[-400:]
        
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Hybrid Voice Assistant (Vosk VAD + NeMo ASR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--vosk-model",
        default="models/vosk-model-small-en-us-0.15",
        help="Path to Vosk model (used only for VAD)"
    )
    parser.add_argument(
        "--nemo-model",
        default="stt_en_quartznet15x5",
        help="NeMo ASR model name. Options: stt_en_quartznet15x5, stt_en_jasper10x5dr"
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.0,
        help="Seconds of silence before processing speech (prevents interruption)"
    )
    
    args = parser.parse_args()
    
    assistant = VoiceAssistantNemo(
        vosk_model_path=args.vosk_model,
        nemo_model=args.nemo_model,
        silence_duration=args.silence_duration
    )
    assistant.run()


if __name__ == "__main__":
    main()
