"""Production intent routing.

Goal: Keep production behavior robust without maintaining a huge regex surface.

Strategy:
- Use an ML model (FastText) as the primary signal.
- Use a tiny "guardrail" ruleset for critical commands (yes/no/help/checkout/etc).
- If the model is low-confidence or obviously wrong (e.g., food term -> greeting),
  ask guardrails or fall back.

This stays CPU/RAM friendly for Jetson-class devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

from .intent_classifier import IntentResult, GuardrailIntentClassifier


@dataclass
class IntentRouterConfig:
    model_confidence_threshold: float = 0.50  # Lower threshold - trust the model more
    disallow_greeting_when_food: bool = True


class IntentRouter:
    def __init__(
        self,
        model_classifier,
        guardrails: Optional[GuardrailIntentClassifier] = None,
        config: Optional[IntentRouterConfig] = None,
        vocabulary = None,  # Optional MenuVocabulary for dynamic food keyword detection
    ):
        self.model_classifier = model_classifier
        self.guardrails = guardrails or GuardrailIntentClassifier()
        self.config = config or IntentRouterConfig()
        self.vocabulary = vocabulary

    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        # 1) Hard guardrails first for critical, low-latency commands only.
        # Keep this minimal - only yes/no/help/checkout
        g = self.guardrails.classify(text, top_k=top_k)
        if g.intent != "unknown":
            return g

        # 2) Use FastText model for everything else - NO REGEX FALLBACK
        model_result = self.model_classifier.classify(text, top_k=top_k)

        # 3) Trust the model even with lower confidence
        # If confidence is very low (<0.30), mark as unknown
        if model_result.confidence < 0.30:
            return IntentResult(intent="unknown", confidence=model_result.confidence, all_intents=model_result.all_intents)

        # 4) Sanity check: greeting intent when food keywords present (DYNAMIC)
        if self.config.disallow_greeting_when_food and model_result.intent == "greeting":
            text_lower = text.lower()
            if self._looks_like_food_request(text_lower):
                # Override greeting -> unknown, let assistant handle it
                return IntentResult(intent="unknown", confidence=model_result.confidence, all_intents=model_result.all_intents)

        return model_result
    
    def _looks_like_food_request(self, text_lower: str) -> bool:
        """Check if text looks like a food request using DYNAMIC keywords from vocabulary"""
        # Build dynamic food keywords from vocabulary
        food_keywords = []
        if self.vocabulary:
            # Extract all food-related terms from vocabulary
            food_keywords.extend([cat.lower() for cat in self.vocabulary.categories])
            food_keywords.extend([tag.lower() for tag in self.vocabulary.tags])
            # Get first word of each item name as keyword
            food_keywords.extend([name.split()[0].lower() for name in self.vocabulary.all_item_names if name])
            # Remove duplicates and very short words
            food_keywords = list(set([kw for kw in food_keywords if len(kw) > 3]))
        
        # Check for food keywords
        if any(k in text_lower for k in food_keywords):
            return True
        
        # Check for action patterns (these are still valid - they're not hardcoded item names)
        if re.search(r"\b(add|order|get|i want|i need|give me|can i have|i'll have)\b", text_lower):
            return True
        if re.search(r"\b(menu|options|what do you have|available)\b", text_lower):
            return True
        return False


# Deprecated - kept for backward compatibility but not used
_FOOD_KEYWORDS = ()  # Removed hardcoded keywords, now dynamic from vocabulary
