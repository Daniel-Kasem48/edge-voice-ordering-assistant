"""
Lightweight Intent Classifier with FastText
Optimized for edge devices (Jetson) with CPU/laptop testing support
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from difflib import SequenceMatcher
import re

from .intent_classifier import IntentResult


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    model_type: str
    inference_time_ms: float
    memory_usage_mb: float
    accuracy: Optional[float] = None


class BaseIntentModel(ABC):
    """Base class for intent classification models"""
    
    @abstractmethod
    def predict(self, text: str, top_k: int = 3) -> IntentResult:
        """Predict intent from text"""
        pass
    
    @abstractmethod
    def load(self, model_path: str):
        """Load model from path"""
        pass
    
    @abstractmethod
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        pass


class FastTextIntentModel(BaseIntentModel):
    """
    FastText-based intent classifier
    - Very lightweight (< 10MB models typical)
    - Fast inference (< 1ms on CPU)
    - Good for edge devices
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = None
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load FastText model"""
        try:
            import fasttext
        except ImportError:
            raise ImportError("FastText not installed. Run: pip install fasttext-wheel")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = fasttext.load_model(model_path)
        self.model_path = model_path
        print(f"✅ Loaded FastText model: {model_path}")
    
    def predict(self, text: str, top_k: int = 3) -> IntentResult:
        """Predict intent using FastText"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Clean text (basic preprocessing only)
        text = text.lower().strip()
        
        try:
            # Predict - handle numpy 2.0 compatibility
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Use k=1 first to avoid tuple unpacking issues
                labels, probs = self.model.predict(text, k=1)
                
                # Now get more if needed
                if top_k > 1:
                    labels, probs = self.model.predict(text, k=top_k)
            
            # Parse results - handle both tuple and array returns
            intents = {}
            
            # Convert to lists if needed
            if hasattr(labels, 'tolist'):
                labels = labels.tolist() if hasattr(labels.tolist, '__call__') else [str(labels)]
            elif not isinstance(labels, (list, tuple)):
                labels = [labels]
                
            if hasattr(probs, 'tolist'):
                probs = probs.tolist() if hasattr(probs.tolist, '__call__') else [float(probs)]
            elif not isinstance(probs, (list, tuple)):
                probs = [float(probs)]
            
            for label, prob in zip(labels, probs):
                intent = str(label).replace("__label__", "")
                intents[intent] = float(prob)
            
            top_intent = list(intents.keys())[0] if intents else "unknown"
            top_confidence = list(intents.values())[0] if intents else 0.0
            
            return IntentResult(
                intent=top_intent,
                confidence=top_confidence,
                all_intents=intents
            )
        except Exception as e:
            # If prediction fails, return unknown
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                all_intents={"unknown": 1.0}
            )
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        if self.model_path and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)
        return 0.0


class LightweightIntentClassifier:
    """
    FastText-based intent classifier with optional rule-based fallback
    Optimized for edge devices
    """
    
    def __init__(
        self,
        model_type: str = "fasttext",
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        fallback_to_rules: bool = True,
        vocabulary = None  # Optional MenuVocabulary for dynamic keyword detection
    ):
        """
        Initialize lightweight intent classifier
        
        Args:
            model_type: Must be "fasttext" (only supported type)
            model_path: Path to FastText model file (.bin)
            use_gpu: Ignored (FastText is CPU-only)
            fallback_to_rules: Use rule-based classifier as fallback
            vocabulary: Optional MenuVocabulary for dynamic food keyword detection
        """
        self.model_type = model_type
        self.model = None
        self.rule_classifier = None
        self.vocabulary = vocabulary
        
        # Initialize model
        if model_path:
            self._load_model(model_type, model_path, use_gpu)
        
        # Setup fallback
        if fallback_to_rules:
            from .intent_classifier import RuleBasedIntentClassifier
            self.rule_classifier = RuleBasedIntentClassifier()
            print("✅ Rule-based fallback ready")
    
    def _load_model(self, model_type: str, model_path: str, use_gpu: bool):
        """Load FastText model"""
        if model_type == "fasttext":
            self.model = FastTextIntentModel(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Only 'fasttext' is supported.")
    
    def _should_use_rule_based(self, text: str, model_result: IntentResult) -> bool:
        """Check if we should use rule-based classifier instead"""
        # Use rules if model confidence is low
        if model_result.confidence < 0.7:
            return True
        
        # Use rules if the result doesn't make sense contextually
        text_lower = text.lower()
        
        # Check for specific misclassifications using DYNAMIC food keywords from vocabulary
        food_keywords = []
        if self.vocabulary:
            # Extract all food-related terms from vocabulary
            food_keywords.extend([cat.lower() for cat in self.vocabulary.categories])
            food_keywords.extend([tag.lower() for tag in self.vocabulary.tags])
            # Get first word of each item name as keyword
            food_keywords.extend([name.split()[0].lower() for name in self.vocabulary.all_item_names if name])
            # Remove duplicates and very short words
            food_keywords = list(set([kw for kw in food_keywords if len(kw) > 3]))
        
        has_food_keyword = any(keyword in text_lower for keyword in food_keywords) if food_keywords else False
        
        # If text mentions food but classified as greeting, use rules
        if has_food_keyword and model_result.intent == 'greeting':
            return True
        
        # If asking "what do you have for X", it should be search_menu
        if re.search(r'what.*have.*for', text_lower) and model_result.intent != 'search_menu':
            return True
        
        return False
    
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify intent using ML model only - no regex fallback"""
        if self.model:
            try:
                model_result = self.model.predict(text, top_k)
                
                # Only use rule-based if model explicitly fails or returns very low confidence
                # This ensures we trust the ML model even with medium confidence
                if self.rule_classifier and (model_result.confidence < 0.30 or model_result.intent == "unknown"):
                    print(f"  [Very low confidence ({model_result.confidence:.2f}) - checking rules]")
                    rule_result = self.rule_classifier.classify(text, top_k)
                    # Only override if rules have strong match
                    if rule_result.confidence > 0.7:
                        return rule_result
                
                # Trust the ML model for everything else
                return model_result
            except Exception as e:
                print(f"⚠️  Model prediction failed: {e}")
                if self.rule_classifier:
                    print("📋 Falling back to rule-based classifier")
                    return self.rule_classifier.classify(text, top_k)
                else:
                    raise
        elif self.rule_classifier:
            # No ML model available - use rules
            return self.rule_classifier.classify(text, top_k)
        else:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                all_intents={"unknown": 1.0}
            )
    
    def benchmark(self, test_phrases: List[str]) -> BenchmarkResult:
        """Benchmark model performance"""
        if not self.model:
            return BenchmarkResult(
                model_type="none",
                inference_time_ms=0,
                memory_usage_mb=0
            )
        
        # Measure inference time
        times = []
        for phrase in test_phrases:
            start = time.time()
            _ = self.model.predict(phrase)
            times.append((time.time() - start) * 1000)  # ms
        
        avg_time = np.mean(times)
        model_size = self.model.get_model_size_mb()
        
        return BenchmarkResult(
            model_type=self.model_type,
            inference_time_ms=avg_time,
            memory_usage_mb=model_size
        )


if __name__ == "__main__":
    # Test the lightweight classifier
    print("Testing Lightweight Intent Classifier\n" + "="*50)
    
    # Test with rule-based fallback
    classifier = LightweightIntentClassifier(
        model_type="fasttext",
        model_path=None,  # No model, will use rules
        fallback_to_rules=True
    )
    
    test_phrases = [
        "I want a cheeseburger",
        "add two pizzas to my order",
        "remove the salad",
        "what's in my cart",
        "checkout please",
        "how much is the burger",
        "show me the menu",
        "hello there",
        "thank you goodbye"
    ]
    
    print("\n📊 Classification Results:")
    print("-" * 50)
    
    for phrase in test_phrases:
        result = classifier.classify(phrase)
        print(f"'{phrase}'")
        print(f"  → Intent: {result.intent} ({result.confidence:.2f})")
        if len(result.all_intents) > 1:
            other_intents = [(i, c) for i, c in result.all_intents.items() if i != result.intent]
            if other_intents:
                print(f"  → Also considered: {other_intents[0][0]} ({other_intents[0][1]:.2f})")
        print()
    
    # Benchmark
    print("\n⚡ Benchmark Results:")
    print("-" * 50)
    bench = classifier.benchmark(test_phrases)
    print(f"Model Type: {bench.model_type}")
    print(f"Avg Inference Time: {bench.inference_time_ms:.2f} ms")
    print(f"Model Size: {bench.memory_usage_mb:.2f} MB")