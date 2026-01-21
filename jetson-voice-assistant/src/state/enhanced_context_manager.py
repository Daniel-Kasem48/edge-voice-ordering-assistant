"""
Enhanced Context Manager with Conversation Summarization
Provides richer context for intent classification
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter
import re
from .context_manager import ContextManager, ConversationContext, ConversationState


@dataclass
class ConversationSummary:
    """Summary of conversation patterns"""
    dominant_intents: List[Tuple[str, int]]  # (intent, count)
    mentioned_items: List[str]
    mentioned_categories: List[str]
    total_items_ordered: int
    conversation_duration: timedelta
    last_successful_order: Optional[str]
    common_patterns: List[str]


class EnhancedContextManager(ContextManager):
    """Extended context manager with advanced features"""
    
    def __init__(self):
        super().__init__()
        self.item_history: List[str] = []
        self.category_history: List[str] = []
        self.successful_orders: List[Dict[str, Any]] = []
        self.failed_intents: List[Tuple[str, str]] = []  # (input, intent)
        
        # Pattern recognition for better context understanding
        self.item_patterns = re.compile(
            r'\b(pizza|burger|drink|coffee|latte|cola|coke|pepsi|water|'
            r'salad|fries|dessert|cake|ice cream|sandwich|wrap)\b',
            re.IGNORECASE
        )
        self.quantity_patterns = re.compile(
            r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b',
            re.IGNORECASE
        )
        self.modifier_patterns = re.compile(
            r'\b(large|small|medium|hot|cold|iced|extra|no|without|add|remove)\b',
            re.IGNORECASE
        )
    
    def update_context(self, user_input: str, intent: str, response: str):
        """Enhanced context update with pattern extraction"""
        super().update_context(user_input, intent, response)
        
        # Extract and store additional patterns
        self._extract_advanced_entities(user_input, intent, response)
        
        # Track successful vs failed intents
        if "sorry" in response.lower() or "didn't understand" in response.lower():
            self.failed_intents.append((user_input, intent))
        elif intent == "add_to_cart" and "added" in response.lower():
            self.successful_orders.append({
                "timestamp": datetime.now(),
                "input": user_input,
                "response": response
            })
    
    def _extract_advanced_entities(self, user_input: str, intent: str, response: str):
        """Extract advanced entities from conversation"""
        # Extract items
        items = self.item_patterns.findall(user_input)
        if items:
            self.item_history.extend(items)
            # Keep last 20 items
            self.item_history = self.item_history[-20:]
        
        # Extract categories from successful searches
        if intent == "search_menu":
            categories = ["pizza", "burger", "drink", "dessert", "salad", "side"]
            for cat in categories:
                if cat in user_input.lower():
                    self.category_history.append(cat)
                    self.category_history = self.category_history[-10:]
                    break
        
        # Extract quantities and modifiers for richer context
        quantities = self.quantity_patterns.findall(user_input)
        modifiers = self.modifier_patterns.findall(user_input)
        
        # Store in current context
        if hasattr(self.context, '_extra_data'):
            self.context._extra_data['last_quantities'] = quantities
            self.context._extra_data['last_modifiers'] = modifiers
        else:
            self.context._extra_data = {
                'last_quantities': quantities,
                'last_modifiers': modifiers
            }
    
    def get_conversation_summary(self) -> ConversationSummary:
        """Generate a summary of the conversation"""
        if not self.context.conversation_history:
            return ConversationSummary(
                dominant_intents=[],
                mentioned_items=[],
                mentioned_categories=[],
                total_items_ordered=0,
                conversation_duration=timedelta(0),
                last_successful_order=None,
                common_patterns=[]
            )
        
        # Calculate dominant intents
        intent_counter = Counter(turn['intent'] for turn in self.context.conversation_history)
        dominant_intents = intent_counter.most_common(3)
        
        # Get unique items and categories
        mentioned_items = list(dict.fromkeys(self.item_history))  # Preserve order, remove duplicates
        mentioned_categories = list(dict.fromkeys(self.category_history))
        
        # Calculate duration
        first_turn = self.context.conversation_history[0]['timestamp']
        last_turn = self.context.conversation_history[-1]['timestamp']
        duration = last_turn - first_turn
        
        # Get last successful order
        last_order = self.successful_orders[-1]['response'] if self.successful_orders else None
        
        # Identify common patterns
        patterns = []
        if len([i for i, _ in dominant_intents if i == "search_menu"]) > 2:
            patterns.append("browsing_multiple_categories")
        if len(self.failed_intents) > 2:
            patterns.append("having_difficulty")
        if "checkout" in [i for i, _ in dominant_intents]:
            patterns.append("ready_to_order")
        
        return ConversationSummary(
            dominant_intents=dominant_intents,
            mentioned_items=mentioned_items[:5],  # Top 5
            mentioned_categories=mentioned_categories,
            total_items_ordered=len(self.successful_orders),
            conversation_duration=duration,
            last_successful_order=last_order,
            common_patterns=patterns
        )
    
    def get_enriched_context_hints(self) -> Dict[str, Any]:
        """Get enriched context hints with summary"""
        base_hints = self.get_context_hints()
        summary = self.get_conversation_summary()
        
        # Add enriched information
        base_hints.update({
            "dominant_intent": summary.dominant_intents[0][0] if summary.dominant_intents else None,
            "mentioned_items": summary.mentioned_items,
            "mentioned_categories": summary.mentioned_categories,
            "total_ordered": summary.total_items_ordered,
            "conversation_patterns": summary.common_patterns,
            "failed_attempts": len(self.failed_intents),
            "session_duration": str(summary.conversation_duration),
            "frequent_items": Counter(self.item_history).most_common(3)
        })
        
        # Add extra data if available
        if hasattr(self.context, '_extra_data'):
            base_hints['extra_data'] = self.context._extra_data
        
        return base_hints
    
    def should_suggest_item(self) -> Optional[str]:
        """Suggest an item based on conversation history"""
        if not self.item_history:
            return None
        
        # If user has been browsing a lot without ordering
        summary = self.get_conversation_summary()
        if ("browsing_multiple_categories" in summary.common_patterns and 
            summary.total_items_ordered == 0):
            # Suggest a popular item from their browsing history
            item_counts = Counter(self.item_history)
            if item_counts:
                return item_counts.most_common(1)[0][0]
        
        return None
    
    def get_contextual_prompt_enhancement(self) -> str:
        """Generate a context summary string for model input enhancement"""
        hints = self.get_enriched_context_hints()
        
        prompt_parts = []
        
        # Add conversation state
        if hints['state'] != 'idle':
            prompt_parts.append(f"State: {hints['state']}")
        
        # Add recent items
        if hints['mentioned_items']:
            prompt_parts.append(f"Recent items: {', '.join(hints['mentioned_items'][:3])}")
        
        # Add patterns
        if hints['conversation_patterns']:
            prompt_parts.append(f"Patterns: {', '.join(hints['conversation_patterns'])}")
        
        # Add failed attempts indicator
        if hints['failed_attempts'] > 2:
            prompt_parts.append("Multiple failed attempts")
        
        return " | ".join(prompt_parts) if prompt_parts else ""