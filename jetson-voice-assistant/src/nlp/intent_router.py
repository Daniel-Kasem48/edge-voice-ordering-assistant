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
    model_confidence_threshold: float = 0.70
    disallow_greeting_when_food: bool = True


class IntentRouter:
    def __init__(
        self,
        model_classifier,
        guardrails: Optional[GuardrailIntentClassifier] = None,
        config: Optional[IntentRouterConfig] = None,
    ):
        self.model_classifier = model_classifier
        self.guardrails = guardrails or GuardrailIntentClassifier()
        self.config = config or IntentRouterConfig()

    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        # 1) Hard guardrails first for critical, low-latency commands.
        g = self.guardrails.classify(text, top_k=top_k)
        if g.intent != "unknown":
            return g

        # 2) Model classification.
        model_result = self.model_classifier.classify(text, top_k=top_k)

        # 3) Low confidence: return model result, but mark unknown if super low.
        if model_result.confidence < self.config.model_confidence_threshold:
            return model_result

        # 4) Cheap contextual sanity checks.
        if self.config.disallow_greeting_when_food and model_result.intent == "greeting":
            text_lower = text.lower()
            if _looks_like_food_request(text_lower):
                # Let downstream handling treat it as unknown -> assistant can suggest menu/search.
                return IntentResult(intent="unknown", confidence=model_result.confidence, all_intents=model_result.all_intents)

        return model_result


_FOOD_KEYWORDS = (
    "dessert",
    "burger",
    "pizza",
    "drink",
    "coffee",
    "salad",
    "fries",
    "cake",
    "ice cream",
    "chocolate",
)


def _looks_like_food_request(text_lower: str) -> bool:
    if any(k in text_lower for k in _FOOD_KEYWORDS):
        return True
    if re.search(r"\b(add|order|get|i want|i need|give me|can i have|i'll have)\b", text_lower):
        return True
    if re.search(r"\b(menu|options|what do you have|available)\b", text_lower):
        return True
    return False
