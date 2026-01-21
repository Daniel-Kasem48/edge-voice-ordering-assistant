"""
Context-Aware Intent Classifier
Enhances FastText model input with contextual information
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .intent_classifier import IntentResult, IntentClassifier
from ..state.context_manager import ContextManager, ConversationState


@dataclass
class ContextualInput:
    """Structured input with context for the classifier"""
    raw_input: str
    enriched_input: str
    context_features: Dict[str, Any]


class ContextAwareClassifier:
    """Wraps FastText classifier with context-aware preprocessing"""
    
    def __init__(self, base_classifier: IntentClassifier, context_manager: ContextManager):
        self.base_classifier = base_classifier
        self.context_manager = context_manager
        
        # Context feature templates
        self.context_templates = {
            ConversationState.BROWSING_MENU: "[BROWSING] ",
            ConversationState.ADDING_ITEMS: "[ORDERING] ",
            ConversationState.CONFIRMING_ORDER: "[CONFIRMING] ",
            ConversationState.CHECKOUT: "[CHECKOUT] ",
            ConversationState.ASKING_FOR_DETAILS: "[DETAILS] "
        }
        
        # Referential resolution templates
        self.referential_expansions = {
            "that one": "item",
            "those": "items", 
            "it": "item",
            "them": "items",
            "this": "item",
            "these": "items"
        }
    
    def _enrich_input_with_context(self, text: str) -> ContextualInput:
        """Enrich user input with contextual information"""
        context_hints = self.context_manager.get_context_hints()
        enriched = text
        features = {}
        
        # 1. Add conversation state prefix
        state = self.context_manager.context.state
        if state in self.context_templates:
            enriched = self.context_templates[state] + enriched
            features['state_prefix'] = self.context_templates[state]
        
        # 2. Expand referential phrases with context
        text_lower = text.lower()
        for ref_phrase, expansion in self.referential_expansions.items():
            if ref_phrase in text_lower:
                # Add contextual expansion based on last mentioned item/category
                if context_hints['last_item']:
                    enriched = enriched.replace(ref_phrase, f"{expansion} {context_hints['last_item']}")
                    features['expanded_reference'] = context_hints['last_item']
                elif context_hints['last_category']:
                    enriched = enriched.replace(ref_phrase, f"{context_hints['last_category']} {expansion}")
                    features['expanded_reference'] = context_hints['last_category']
        
        # 3. Add recent intent context for follow-ups
        if context_hints['last_intent'] and self._is_follow_up(text):
            enriched = f"[FOLLOWUP:{context_hints['last_intent']}] {enriched}"
            features['followup_intent'] = context_hints['last_intent']
        
        # 4. Add awaiting response context
        if context_hints['awaiting_response']:
            enriched = f"[AWAITING:{context_hints['awaiting_response']}] {enriched}"
            features['awaiting'] = context_hints['awaiting_response']
        
        # 5. Add conversation history summary
        if self.context_manager.context.conversation_history:
            # Get last 3 intents for pattern recognition
            recent_intents = [turn['intent'] for turn in self.context_manager.context.conversation_history[-3:]]
            if recent_intents:
                enriched = f"[HISTORY:{','.join(recent_intents)}] {enriched}"
                features['recent_intents'] = recent_intents
        
        return ContextualInput(
            raw_input=text,
            enriched_input=enriched,
            context_features=features
        )
    
    def _is_follow_up(self, text: str) -> bool:
        """Check if input is likely a follow-up"""
        follow_up_indicators = [
            "more", "else", "another", "also", "and", "plus",
            "what about", "how about", "show me"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in follow_up_indicators)
    
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify with context enhancement"""
        # First check if we should use context
        if not self.context_manager.should_use_context_for_intent(text):
            # Direct classification without context
            return self.base_classifier.classify(text, top_k)
        
        # Enrich input with context
        contextual_input = self._enrich_input_with_context(text)
        
        # Classify the enriched input
        base_result = self.base_classifier.classify(contextual_input.enriched_input, top_k)
        
        # Apply context-based post-processing
        enhanced_intent, enhanced_confidence = self.context_manager.enhance_intent_with_context(
            text, base_result.intent, base_result.confidence
        )
        
        # If context significantly changed the intent, adjust confidence
        if enhanced_intent != base_result.intent:
            # Blend the confidence based on how much context influenced the decision
            context_weight = 0.3  # How much to trust context vs model
            final_confidence = (base_result.confidence * (1 - context_weight) + 
                               enhanced_confidence * context_weight)
            
            return IntentResult(
                intent=enhanced_intent,
                confidence=final_confidence,
                all_intents={
                    enhanced_intent: final_confidence,
                    base_result.intent: base_result.confidence * 0.7,  # Downweight original
                    **{k: v * 0.5 for k, v in base_result.all_intents.items() 
                       if k not in [enhanced_intent, base_result.intent]}
                }
            )
        
        return base_result
    
    def get_debug_info(self, text: str) -> Dict[str, Any]:
        """Get detailed debugging information about classification"""
        contextual_input = self._enrich_input_with_context(text)
        base_result = self.base_classifier.classify(contextual_input.enriched_input)
        enhanced_intent, enhanced_confidence = self.context_manager.enhance_intent_with_context(
            text, base_result.intent, base_result.confidence
        )
        
        return {
            "raw_input": text,
            "enriched_input": contextual_input.enriched_input,
            "context_features": contextual_input.context_features,
            "base_intent": base_result.intent,
            "base_confidence": base_result.confidence,
            "enhanced_intent": enhanced_intent,
            "enhanced_confidence": enhanced_confidence,
            "context_state": self.context_manager.context.state.value,
            "last_intent": self.context_manager.context.last_intent,
            "conversation_length": len(self.context_manager.context.conversation_history)
        }