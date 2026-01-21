"""
Text-based Input Engine
For testing without audio hardware
"""

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class TranscriptionResult:
    """Result from text input (simulates STT output)"""
    text: str
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0
    confidence: float = 1.0
    language: str = "en"


class TextInputEngine:
    """
    Text-based input for testing
    Simulates STT by reading from stdin
    """
    
    def __init__(self, prompt: str = "You: "):
        self.prompt = prompt
    
    def listen_and_transcribe(self) -> TranscriptionResult:
        """
        Get text input from user
        
        Returns:
            TranscriptionResult with user's text
        """
        start_time = time.time()
        
        try:
            text = input(self.prompt).strip()
        except (EOFError, KeyboardInterrupt):
            text = ""
        
        return TranscriptionResult(
            text=text,
            duration_seconds=0,
            processing_time_seconds=time.time() - start_time,
            confidence=1.0,
            language="en"
        )


class TextWakeWordDetector:
    """
    Text-based wake word detector for testing
    User types the wake word to activate
    """
    
    def __init__(self, wake_phrase: str = "hey assistant"):
        self.wake_phrase = wake_phrase.lower()
        self._callback = None
    
    def set_callback(self, callback):
        """Set callback for when wake word detected"""
        self._callback = callback
    
    def check_wake_word(self, text: str) -> bool:
        """
        Check if text contains wake word
        
        Args:
            text: User input text
            
        Returns:
            True if wake word detected
        """
        if self.wake_phrase in text.lower():
            if self._callback:
                self._callback()
            return True
        return False
    
    def wait_for_wake_word(self) -> str:
        """
        Wait for user to type wake word
        Returns the text after wake word
        """
        print(f"💤 Waiting for wake word: '{self.wake_phrase}'")
        print("   (Type your command with or without the wake phrase)")
        
        while True:
            try:
                text = input("You: ").strip()
                
                if not text:
                    continue
                
                # Check for wake word
                lower_text = text.lower()
                if self.wake_phrase in lower_text:
                    # Remove wake phrase and return remaining text
                    idx = lower_text.find(self.wake_phrase)
                    remaining = text[idx + len(self.wake_phrase):].strip()
                    if remaining:
                        return remaining
                    # Wake word only, get next input
                    print("🔔 Listening...")
                    return input("You: ").strip()
                else:
                    # Allow commands without wake word for convenience
                    return text
                    
            except (EOFError, KeyboardInterrupt):
                return ""


class TextOutputEngine:
    """
    Text-based output (simulates TTS)
    """
    
    def __init__(self, prefix: str = "🤖 Assistant: "):
        self.prefix = prefix
    
    def speak(self, text: str):
        """Print response text"""
        print(f"{self.prefix}{text}")
    
    def speak_async(self, text: str):
        """Same as speak for text mode"""
        self.speak(text)


if __name__ == "__main__":
    # Test text input
    engine = TextInputEngine()
    
    print("Text Input Engine Test")
    print("Type 'quit' to exit\n")
    
    while True:
        result = engine.listen_and_transcribe()
        
        if result.text.lower() == "quit":
            break
        
        print(f"  → Got: '{result.text}'")
