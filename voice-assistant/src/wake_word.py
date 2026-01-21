from openwakeword import Model
import numpy as np
import os

class WakeWordDetector:
    def __init__(self, model_names=["hey_mycroft"], threshold=0.5):
        """Initialize wake word detector"""
        # Get available models
        available_models = Model.list_models()
        print(f"Available wake word models: {available_models}")
        
        # Use intersection of requested and available models
        models_to_use = list(set(model_names) & set(available_models))
        if not models_to_use:
            print(f"Warning: No requested models available. Using default.")
            models_to_use = available_models[:1]  # Use first available
            
        print(f"Loading wake word models: {models_to_use}")
        self.model = Model(wakeword_models=models_to_use)
        self.threshold = threshold
        self.detection_callback = None
        
    def process_audio(self, audio_chunk):
        """Process audio chunk for wake word"""
        predictions = self.model.predict(audio_chunk)
        
        for wake_word, score in predictions.items():
            if score > self.threshold:
                if self.detection_callback:
                    self.detection_callback(wake_word, score)
                return True, wake_word, score
        return False, None, 0
        
    def set_callback(self, callback):
        """Set callback for wake word detection"""
        self.detection_callback = callback