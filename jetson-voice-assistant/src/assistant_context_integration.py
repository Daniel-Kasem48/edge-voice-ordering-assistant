"""
Helper to integrate enhanced context into existing assistant
"""

from nlp.context_aware_classifier import ContextAwareClassifier
from state.enhanced_context_manager import EnhancedContextManager
from nlp.intent_classifier import IntentClassifier
from typing import Optional


def upgrade_assistant_with_context(assistant):
    """Upgrade existing assistant instance with enhanced context"""
    
    # Replace context manager with enhanced version
    old_context = assistant.context_manager.context if hasattr(assistant, 'context_manager') else None
    assistant.context_manager = EnhancedContextManager()
    
    # Restore old context if exists
    if old_context:
        assistant.context_manager.context = old_context
    
    # Wrap intent classifier with context-aware version
    if hasattr(assistant, 'intent_classifier'):
        assistant.intent_classifier = ContextAwareClassifier(
            base_classifier=assistant.intent_classifier,
            context_manager=assistant.context_manager
        )
    
    # Add debug method
    def _debug_classification(self, text: str):
        """Debug helper to see context enhancement in action"""
        if hasattr(self.intent_classifier, 'get_debug_info'):
            debug_info = self.intent_classifier.get_debug_info(text)
            print("\n[Context Debug Info]")
            print(f"  Raw: {debug_info['raw_input']}")
            print(f"  Enriched: {debug_info['enriched_input']}")
            print(f"  Features: {debug_info['context_features']}")
            print(f"  Base Intent: {debug_info['base_intent']} ({debug_info['base_confidence']:.2f})")
            print(f"  Enhanced: {debug_info['enhanced_intent']} ({debug_info['enhanced_confidence']:.2f})")
            print(f"  Context State: {debug_info['context_state']}")
            
            # Show conversation summary
            summary = self.context_manager.get_conversation_summary()
            if summary.dominant_intents:
                print(f"  Dominant Intents: {summary.dominant_intents}")
            if summary.mentioned_items:
                print(f"  Recent Items: {summary.mentioned_items[:3]}")
    
    # Bind the debug method to assistant instance
    assistant._debug_classification = _debug_classification.__get__(assistant)
    
    print("✨ Assistant upgraded with enhanced context capabilities!")
    print("  - Context-aware classification enabled")
    print("  - Conversation summarization active")
    print("  - Debug with: assistant._debug_classification('your text')")
    
    return assistant


# Example usage in existing code:
# 
# from src.assistant import TextAssistant
# from src.assistant_context_integration import upgrade_assistant_with_context
# 
# # Create normal assistant
# assistant = TextAssistant(config)
# 
# # Upgrade with enhanced context
# assistant = upgrade_assistant_with_context(assistant)
# 
# # Now the assistant will use enhanced context for classification