"""
Intent Classifier using FastText
Lightweight and fast for edge deployment
"""

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class IntentResult:
    """Result from intent classification"""
    intent: str
    confidence: float
    all_intents: Dict[str, float]


class FastTextIntentClassifier:
    """
    Intent classifier using FastText
    For production use with trained model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize FastText classifier
        
        Args:
            model_path: Path to trained FastText model (.bin)
        """
        try:
            import fasttext
        except ImportError:
            raise ImportError("fasttext not installed. pip install fasttext-wheel")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = fasttext.load_model(model_path)
    
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """
        Classify intent of text
        
        Args:
            text: Input text
            top_k: Number of top intents to return
            
        Returns:
            IntentResult with predicted intent
        """
        # Clean text
        text = self._preprocess(text)
        
        # Predict
        labels, probs = self.model.predict(text, k=top_k)
        
        # Parse labels (FastText format: __label__intent_name)
        intents = {}
        for label, prob in zip(labels, probs):
            intent = label.replace("__label__", "")
            intents[intent] = float(prob)
        
        top_intent = list(intents.keys())[0] if intents else "unknown"
        top_confidence = list(intents.values())[0] if intents else 0.0
        
        return IntentResult(
            intent=top_intent,
            confidence=top_confidence,
            all_intents=intents
        )
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for classification"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text


class RuleBasedIntentClassifier:
    """
    Rule-based intent classifier
    No ML model required - uses keyword matching
    Good for testing and simple deployments
    """
    
    # Intent patterns (keywords/phrases -> intent)
    INTENT_PATTERNS = {
        "remove_from_cart": [
            r"\b(remove|delete|take off|take out|cancel|no more|don't want)\b",
            r"\b(remove|delete)\b.*\b(from|order|cart)\b",
            r"^remove\b",  # Starts with remove
        ],
        "add_to_cart": [
            r"\b(add|get|i want|i'd like|give me|order|can i have|i'll have|i need)\b",
            r"\b(one|two|three|four|five|1|2|3|4|5)\b.*\b(burger|pizza|salad|drink|fries|coffee)\b",
            r"^(add|get|order)\b",  # Starts with add/get/order
            r"^(burger|pizza|coffee|salad|fries|coke|pepsi|cake|latte)s?$",  # Single item names
        ],
        "view_cart": [
            r"\b(what's in my|(?!clear\b|empty\b|reset\b)show|view|check|see)\b.*\b(cart|order|bag)\b",
            r"\b((?!clear\b|empty\b|reset\b)show|view|check|see)\s+(me\s+)?(my\s+)?(cart|order|bag)\b",
            r"\bmy (cart|order|bag)\b",
            r"\bwhat do i have\b",
            r"^show cart$",
            r"^(cart|my cart|my order)$",
        ],
        "checkout": [
            r"\b(checkout|check out|pay|complete|finish|done)\b",
            r"\b(that's all|that will be all|ready to pay)\b",
            r"\bi'm done\b",
        ],
        "clear_cart": [
            r"\b(clear|empty|reset|start over)\b.*\b(cart|order)\b",
            r"\bcancel (my |the )?(entire |whole )?order\b"
        ],
        "search_menu": [
            r"\b(show|display|see|view)\b.*\b(menu|options)\b",
            r"\b(what|which)\b.*\b(burger|pizza|drink|salad|dessert|side)s?\b.*\b(have|available|got)\b",
            r"\b(do you have|you have|you got)\b.*\b(burger|pizza|drink|salad|dessert)\b",
            r"\b(search|find|look for)\b",
            r"\bwhat kind of\b",
            r"^(menu|breakfast|lunch|dinner)$",  # Single word menu requests
            r"^(drinks?|desserts?|burgers?|pizzas?|salads?|sides?)$",  # Single category words
            r"\b(want to|looking for|need).*\b(drink|eat|food|something to drink|something to eat)\b",  # General food/drink requests
            r"\b(drink|food|eat).*\bsomething\b",  # "drink something", "eat something"
            r"\bwhat.*\b(dessert|drink|food|burger|pizza|salad)\b",  # "what desserts", "what drinks"
        ],
        "get_price": [
            r"\b(how much|what's the price|cost|price of)\b",
            r"\bhow much (is|are|does)\b"
        ],
        "modify_item": [
            r"\b(change|modify|make it)\b(?!.*\b(pizza|burger|salad)\b)",  # Only if not followed by item
        ],
        "help": [
            r"\b(help|what can you do|commands|options)\b",
            r"\bhow (do i|can i|does this)\b"
        ],
        "greeting": [
            r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b",
            r"^(hi|hello|hey)$"
        ],
        "goodbye": [
            r"\b(bye|goodbye|see you|thanks|thank you|done)\b",
            r"\bhave a (good|nice)\b"
        ]
    }
    
    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize rule-based classifier
        
        Args:
            custom_patterns: Additional patterns to add/override
        """
        self.patterns = dict(self.INTENT_PATTERNS)
        
        if custom_patterns:
            for intent, patterns in custom_patterns.items():
                if intent in self.patterns:
                    self.patterns[intent].extend(patterns)
                else:
                    self.patterns[intent] = patterns
        
        # Compile regex patterns
        self._compiled = {}
        for intent, patterns in self.patterns.items():
            self._compiled[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """
        Classify intent using pattern matching
        
        Args:
            text: Input text
            top_k: Number of top intents to return
            
        Returns:
            IntentResult with predicted intent
        """
        text = text.lower().strip()

        # Fast-path for ambiguous overlaps: "clear my cart" can match both clear_cart and view_cart.
        # Make "clear/empty/reset" semantics win deterministically.
        if re.search(r"\b(clear|empty|reset|start over)\b.*\b(cart|order|bag)\b", text):
            return IntentResult(
                intent="clear_cart",
                confidence=1.0,
                all_intents={"clear_cart": 1.0}
            )
        
        # Score each intent based on pattern matches
        scores = {}
        for intent, patterns in self._compiled.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text):
                    score += 0.5  # Each pattern match adds to score
                    # Debug logging for specific cases
                    if text in ["drinks", "desserts", "burgers", "pizzas", "salads"]:
                        print(f"  [Pattern matched for '{text}': {pattern.pattern} → {intent}]")
            
            if score > 0:
                scores[intent] = min(score, 1.0)  # Cap at 1.0
        
        # If multiple intents match, prioritize certain ones
        # NOTE: order matters; e.g. "clear my cart" can also hit view_cart patterns.
        priority_intents = ["remove_from_cart", "clear_cart", "view_cart", "checkout", "search_menu"]
        for priority in priority_intents:
            if priority in scores and scores[priority] >= 0.5:
                scores[priority] += 0.3  # Boost priority intents
        
        if not scores:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                all_intents={"unknown": 1.0}
            )
        
        # Sort by score
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        total = sum(s for _, s in sorted_intents)
        normalized = {i: min(s, 1.0) for i, s in sorted_intents[:top_k]}
        
        return IntentResult(
            intent=sorted_intents[0][0],
            confidence=min(sorted_intents[0][1], 1.0),
            all_intents=normalized
        )


class GuardrailIntentClassifier:
    """Small, production-friendly ruleset.

    This intentionally covers only a handful of high-impact commands that should
    be reliable when the ML model is uncertain.

    Keep this short. Anything not here should be handled by the ML classifier.
    """

    PATTERNS = {
        "confirm_yes": [r"^(yes|yep|yeah|correct|right|sure|ok|okay)$"],
        "confirm_no": [r"^(no|nope|nah|incorrect|wrong)$"],
        "help": [r"\b(help|what can you do|commands|options)\b"],
        "goodbye": [r"\b(bye|goodbye|see you|thanks|thank you|done)\b"],
        "greeting": [r"^(hi|hello|hey)$", r"\b(good morning|good afternoon|good evening)\b"],
        "clear_cart": [r"\b(clear|empty|reset|start over)\b.*\b(cart|order|bag|basket)\b"],
        "checkout": [r"\b(checkout|check out|pay|complete|finish)\b"],
        "view_cart": [
            r"\b(show|view|see|what's in|what is in|display)\b.*\b(cart|order|bag|basket)\b",
            r"\b(total|bill|check|receipt|what('?s| is) my total|how much do i owe)\b",
        ],
    }

    def __init__(self):
        self._compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.PATTERNS.items()
        }

    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        text = text.lower().strip()

        for intent in [
            "confirm_yes",
            "confirm_no",
            "clear_cart",
            "checkout",
            "view_cart",
            "help",
            "greeting",
            "goodbye",
        ]:
            for pattern in self._compiled.get(intent, []):
                if pattern.search(text):
                    return IntentResult(intent=intent, confidence=1.0, all_intents={intent: 1.0})

        return IntentResult(intent="unknown", confidence=0.0, all_intents={"unknown": 1.0})


class IntentClassifier:
    """
    Unified intent classifier interface
    Uses FastText if available, falls back to rules
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_rules_fallback: bool = True
    ):
        """
        Initialize intent classifier
        
        Args:
            model_path: Path to FastText model (optional)
            use_rules_fallback: Use rule-based classifier as fallback
        """
        self.fasttext_classifier = None
        self.rule_classifier = None
        
        # Try FastText first
        if model_path and os.path.exists(model_path):
            try:
                self.fasttext_classifier = FastTextIntentClassifier(model_path)
                print(f"✅ Loaded FastText model: {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load FastText model: {e}")
        
        # Setup rule-based fallback
        if use_rules_fallback:
            self.rule_classifier = RuleBasedIntentClassifier()
            print("✅ Rule-based classifier ready")
    
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify intent of text"""
        if self.fasttext_classifier:
            return self.fasttext_classifier.classify(text, top_k)
        elif self.rule_classifier:
            return self.rule_classifier.classify(text, top_k)
        else:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                all_intents={}
            )


if __name__ == "__main__":
    # Test rule-based classifier
    classifier = RuleBasedIntentClassifier()
    
    test_phrases = [
        "I want a cheeseburger",
        "add two pizzas to my order",
        "remove the salad",
        "what's in my cart",
        "checkout please",
        "how much is the burger",
        "make it large",
        "hello",
        "thank you goodbye",
        "what kind of drinks do you have"
    ]
    
    print("Intent Classification Test\n" + "="*50)
    
    for phrase in test_phrases:
        result = classifier.classify(phrase)
        print(f"'{phrase}'")
        print(f"  → Intent: {result.intent} ({result.confidence:.2f})")
        print()
