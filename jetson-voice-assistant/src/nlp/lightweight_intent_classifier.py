"""
Lightweight Intent Classifier with FastText and DistilBERT options
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
    
    def _correct_typos(self, text: str) -> str:
        """Simple typo correction for common restaurant terms"""
        # Common typos and their corrections
        typo_corrections = {
            'deserrt': 'dessert',
            'desserrt': 'dessert',
            'desrt': 'dessert',
            'icecream': 'ice cream',
            'ice-cream': 'ice cream',
            'scholocate': 'chocolate',
            'choclate': 'chocolate',
            'chocolat': 'chocolate',
            'burgur': 'burger',
            'buger': 'burger',
            'piza': 'pizza',
            'pizzza': 'pizza',
            'cofee': 'coffee',
            'coffe': 'coffee',
            'frys': 'fries',
            'fris': 'fries',
            'vegeterian': 'vegetarian',
            'vegitarian': 'vegetarian',
            'vegiterian': 'vegetarian',
            'hawaian': 'hawaiian',
            'hawain': 'hawaiian',
            'margherita': 'margherita',
            'margarita': 'margherita',
            'peperoni': 'pepperoni',
            'pepperoni': 'pepperoni',
        }
        
        # Apply corrections
        words = text.lower().split()
        corrected_words = []
        for word in words:
            corrected_words.append(typo_corrections.get(word, word))
        
        corrected_text = ' '.join(corrected_words)
        
        # Fix common grammar issues
        # "do you pizzas" -> "do you have pizzas"
        corrected_text = re.sub(r'\b(do you|can you|could you)\s+(pizza|burger|drink|coffee|dessert|salad)s?\b', 
                               r'\1 have \2s', corrected_text)
        
        return corrected_text
    
    def predict(self, text: str, top_k: int = 3) -> IntentResult:
        """Predict intent using FastText"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Clean and correct text
        text = text.lower().strip()
        corrected_text = self._correct_typos(text)
        
        # Try both original and corrected text
        texts_to_try = [corrected_text]
        if corrected_text != text:
            texts_to_try.append(text)
        
        best_result = None
        best_confidence = 0.0
        
        for try_text in texts_to_try:
            try:
                # Predict - handle numpy 2.0 compatibility
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # Use k=1 first to avoid tuple unpacking issues
                    labels, probs = self.model.predict(try_text, k=1)
                    
                    # Now get more if needed
                    if top_k > 1:
                        labels, probs = self.model.predict(try_text, k=top_k)
                
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
                
                # Keep best result
                if top_confidence > best_confidence:
                    best_confidence = top_confidence
                    best_result = IntentResult(
                        intent=top_intent,
                        confidence=top_confidence,
                        all_intents=intents
                    )
            except Exception as e:
                # Numpy or fasttext compatibility issue - continue to next variant or fallback
                continue
        
        return best_result if best_result else IntentResult(
            intent="unknown",
            confidence=0.0,
            all_intents={"unknown": 1.0}
        )
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        if self.model_path and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)
        return 0.0


class DistilBERTIntentModel(BaseIntentModel):
    """
    DistilBERT-based intent classifier
    - Lightweight BERT (65MB base model)
    - Better accuracy than FastText
    - Suitable for Jetson with GPU
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.use_gpu = use_gpu
        self.device = None
        self.label_map = {}
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load DistilBERT model"""
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            import torch
        except ImportError:
            raise ImportError("Transformers not installed. Run: pip install transformers torch")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Set device
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label map
        label_map_path = Path(model_path) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r") as f:
                self.label_map = json.load(f)
        
        self.model_path = model_path
        print(f"✅ Loaded DistilBERT model on {self.device}: {model_path}")
    
    def predict(self, text: str, top_k: int = 3) -> IntentResult:
        """Predict intent using DistilBERT"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch not installed")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(probs[0])))
        
        # Map to intents
        intents = {}
        for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
            intent = self.label_map.get(str(idx), f"class_{idx}")
            intents[intent] = float(prob)
        
        top_intent = list(intents.keys())[0] if intents else "unknown"
        top_confidence = list(intents.values())[0] if intents else 0.0
        
        return IntentResult(
            intent=top_intent,
            confidence=top_confidence,
            all_intents=intents
        )
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        if self.model_path and os.path.exists(self.model_path):
            total_size = 0
            for file in Path(self.model_path).glob("**/*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024 * 1024)
        return 0.0


class ONNXIntentModel(BaseIntentModel):
    """
    ONNX-optimized intent classifier
    - Convert PyTorch/TF models to ONNX
    - Optimized for edge inference
    - Works well on Jetson
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        self.model_path = None
        self.label_map = {}
        self.tokenizer = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.model_path = model_path
        
        # Load label map
        label_map_path = Path(model_path).parent / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r") as f:
                self.label_map = json.load(f)
        
        # Load tokenizer config
        tokenizer_path = Path(model_path).parent / "tokenizer_config.json"
        if tokenizer_path.exists():
            # Simple tokenizer for demo
            self.tokenizer = self._simple_tokenizer
        
        print(f"✅ Loaded ONNX model: {model_path}")
    
    def _simple_tokenizer(self, text: str) -> np.ndarray:
        """Simple tokenizer for demo (replace with proper tokenizer)"""
        # Convert text to token IDs (simplified)
        words = text.lower().split()
        # Mock token IDs (in practice, use proper tokenizer)
        token_ids = [hash(w) % 10000 for w in words[:128]]
        # Pad to 128
        token_ids += [0] * (128 - len(token_ids))
        return np.array([token_ids], dtype=np.int64)
    
    def predict(self, text: str, top_k: int = 3) -> IntentResult:
        """Predict intent using ONNX"""
        if not self.session:
            raise RuntimeError("Model not loaded")
        
        # Tokenize (simplified)
        input_ids = self._simple_tokenizer(text)
        
        # Run inference
        inputs = {self.session.get_inputs()[0].name: input_ids}
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Get top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        # Map to intents
        intents = {}
        for idx in top_indices:
            intent = self.label_map.get(str(idx), f"class_{idx}")
            intents[intent] = float(probs[idx])
        
        top_intent = list(intents.keys())[0] if intents else "unknown"
        top_confidence = list(intents.values())[0] if intents else 0.0
        
        return IntentResult(
            intent=top_intent,
            confidence=top_confidence,
            all_intents=intents
        )
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        if self.model_path and os.path.exists(self.model_path):
            return os.path.getsize(self.model_path) / (1024 * 1024)
        return 0.0


class LightweightIntentClassifier:
    """
    Main classifier that can use different lightweight models
    Falls back to rule-based if no model available
    """
    
    def __init__(
        self,
        model_type: str = "fasttext",
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        fallback_to_rules: bool = True
    ):
        """
        Initialize lightweight intent classifier
        
        Args:
            model_type: "fasttext", "distilbert", or "onnx"
            model_path: Path to model file/directory
            use_gpu: Use GPU if available (for DistilBERT)
            fallback_to_rules: Use rule-based classifier as fallback
        """
        self.model_type = model_type
        self.model = None
        self.rule_classifier = None
        
        # Initialize model
        if model_path:
            self._load_model(model_type, model_path, use_gpu)
        
        # Setup fallback
        if fallback_to_rules:
            from .intent_classifier import RuleBasedIntentClassifier
            self.rule_classifier = RuleBasedIntentClassifier()
            print("✅ Rule-based fallback ready")
    
    def _load_model(self, model_type: str, model_path: str, use_gpu: bool):
        """Load specified model type"""
        if model_type == "fasttext":
            self.model = FastTextIntentModel(model_path)
        elif model_type == "distilbert":
            self.model = DistilBERTIntentModel(model_path, use_gpu)
        elif model_type == "onnx":
            self.model = ONNXIntentModel(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _should_use_rule_based(self, text: str, model_result: IntentResult) -> bool:
        """Check if we should use rule-based classifier instead"""
        # Use rules if model confidence is low
        if model_result.confidence < 0.7:
            return True
        
        # Use rules if the result doesn't make sense contextually
        text_lower = text.lower()
        
        # Check for specific misclassifications
        food_keywords = ['dessert', 'burger', 'pizza', 'drink', 'coffee', 'salad', 'fries', 'cake', 'ice cream', 'chocolate']
        has_food_keyword = any(keyword in text_lower for keyword in food_keywords)
        
        # If text mentions food but classified as greeting, use rules
        if has_food_keyword and model_result.intent == 'greeting':
            return True
        
        # If asking "what do you have for X", it should be search_menu
        if re.search(r'what.*have.*for', text_lower) and model_result.intent != 'search_menu':
            return True
        
        return False
    
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify intent with improved typo tolerance"""
        if self.model:
            try:
                model_result = self.model.predict(text, top_k)
                
                # Check if we should use rule-based instead
                if self.rule_classifier and self._should_use_rule_based(text, model_result):
                    print(f"  [Low confidence ({model_result.confidence:.2f}) or contextual mismatch - using rules]")
                    rule_result = self.rule_classifier.classify(text, top_k)
                    # If rule-based has a strong match, use it
                    if rule_result.confidence > 0.5:
                        return rule_result
                    # Otherwise, stick with model if it had some confidence
                    elif model_result.confidence > 0.5:
                        return model_result
                    else:
                        return rule_result
                
                return model_result
            except Exception as e:
                print(f"⚠️  Model prediction failed: {e}")
                if self.rule_classifier:
                    print("📋 Falling back to rule-based classifier")
                    return self.rule_classifier.classify(text, top_k)
                else:
                    raise
        elif self.rule_classifier:
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