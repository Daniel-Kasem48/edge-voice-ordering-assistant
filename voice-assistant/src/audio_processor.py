import numpy as np
import sounddevice as sd
import queue
import threading
import webrtcvad
import collections
import time

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_duration_ms=30, device=None):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.audio_queue = queue.Queue()
        self.device = device
        self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        
        # Ring buffer for VAD
        self.ring_buffer = collections.deque(maxlen=10)
        
        print(f"Audio device: {self.get_device_info()}")
        
    def get_device_info(self):
        """Get current audio device info"""
        if self.device is None:
            return sd.query_devices(kind='input')['name']
        return sd.query_devices(self.device, 'input')['name']
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio warning: {status}")
        self.audio_queue.put(indata.copy().flatten())
        
    def start_stream(self):
        """Start audio input stream"""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=self.chunk_size,
            device=self.device
        )
        self.stream.start()
        print("Audio stream started")
        
    def stop_stream(self):
        """Stop audio stream"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
    def get_audio_chunk(self, timeout=0.1):
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def is_speech(self, audio_chunk):
        """Check if chunk contains speech"""
        # Convert float32 to int16 for VAD
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Ensure correct size for VAD (must be 10, 20, or 30 ms)
        expected_size = int(self.sample_rate * self.chunk_duration_ms / 1000) * 2
        if len(audio_int16.tobytes()) == expected_size:
            try:
                return self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
            except:
                return False
        return False
        
    def record_until_silence(self, pre_buffer_chunks=5, silence_chunks=10):
        """Record audio until silence is detected"""
        audio_buffer = []
        pre_buffer = collections.deque(maxlen=pre_buffer_chunks)
        silence_counter = 0
        recording = False
        
        print("Listening... (speak now)")
        
        while True:
            chunk = self.get_audio_chunk()
            if chunk is None:
                continue
                
            is_speech = self.is_speech(chunk)
            
            if not recording:
                pre_buffer.append(chunk)
                if is_speech:
                    recording = True
                    # Add pre-buffer to recording
                    audio_buffer.extend(pre_buffer)
                    audio_buffer.append(chunk)
                    print("Speech detected, recording...")
            else:
                audio_buffer.append(chunk)
                if is_speech:
                    silence_counter = 0
                else:
                    silence_counter += 1
                    if silence_counter >= silence_chunks:
                        print("Silence detected, stopping recording")
                        break
                        
            # Timeout after 10 seconds
            if len(audio_buffer) > self.sample_rate * 10 / self.chunk_size:
                print("Max recording length reached")
                break
                
        return np.concatenate(audio_buffer) if audio_buffer else None