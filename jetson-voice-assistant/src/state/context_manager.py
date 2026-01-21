"""
Context Manager for Voice Assistant
Tracks conversation context and state for better multi-turn conversations
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum


class ConversationState(Enum):
    """Current conversation state"""
    IDLE = "idle"
    BROWSING_MENU = "browsing_menu"
    ADDING_ITEMS = "adding_items"
    CONFIRMING_ORDER = "confirming_order"
    CHECKOUT = "checkout"
    ASKING_FOR_DETAILS = "asking_for_details"
    AWAITING_OPTION_SELECTION = "awaiting_option_selection"
    AWAITING_ITEM_CONFIRMATION = "awaiting_item_confirmation"


@dataclass
class PendingItem:
    """Item waiting for options/confirmation before adding to cart"""
    item_id: str
    item_name: str
    base_price: float
    quantity: int = 1
    pending_options: List[Dict[str, Any]] = field(default_factory=list)  # Options still to select
    selected_options: Dict[str, Any] = field(default_factory=dict)  # Selected option values
    current_option_index: int = 0  # Which option we're asking about


@dataclass
class ConversationContext:
    """Context information for current conversation"""
    state: ConversationState = ConversationState.IDLE
    last_intent: Optional[str] = None
    last_category_mentioned: Optional[str] = None
    last_item_mentioned: Optional[str] = None
    awaiting_response_for: Optional[str] = None
    pending_item: Optional[PendingItem] = None  # Item waiting for options
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_turn(self, user_input: str, intent: str, assistant_response: str):
        """Add a conversation turn to history"""
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "intent": intent,
            "assistant_response": assistant_response
        })
        self.last_activity = datetime.now()
        self.last_intent = intent
        
        # Keep only last 10 turns
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def is_stale(self, timeout_minutes: int = 5) -> bool:
        """Check if context is stale"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def reset(self):
        """Reset context"""
        self.state = ConversationState.IDLE
        self.last_intent = None
        self.last_category_mentioned = None
        self.last_item_mentioned = None
        self.awaiting_response_for = None
        self.pending_item = None
        self.conversation_history.clear()


class ContextManager:
    """Manages conversation context for better understanding"""
    
    def __init__(self):
        self.context = ConversationContext()
        self.context_timeout_minutes = 5
    
    def set_pending_item(self, item_id: str, item_name: str, base_price: float, options: List[Dict[str, Any]]):
        """Set an item that needs options selected before adding to cart"""
        # Filter to only mandatory options
        mandatory_options = [opt for opt in options if opt.get('is_mandatory', False)]
        
        self.context.pending_item = PendingItem(
            item_id=item_id,
            item_name=item_name,
            base_price=base_price,
            pending_options=mandatory_options,
            selected_options={},
            current_option_index=0
        )
        self.context.state = ConversationState.AWAITING_OPTION_SELECTION
        self.context.awaiting_response_for = "option_selection"
    
    def get_current_option_prompt(self) -> Optional[str]:
        """Get the prompt for the current pending option"""
        if not self.context.pending_item:
            return None
        
        pending = self.context.pending_item
        if pending.current_option_index >= len(pending.pending_options):
            return None  # All options selected
        
        option = pending.pending_options[pending.current_option_index]
        option_name = option.get('name', 'Option')
        values = option.get('option_values', [])
        
        if values:
            choices = ", ".join([v['name'] for v in values])
            return f"What {option_name.lower()} would you like? ({choices})"
        return f"What {option_name.lower()} would you like?"
    
    def select_option(self, value_name: str) -> bool:
        """Select an option value for the pending item. Returns True if more options needed."""
        if not self.context.pending_item:
            return False
        
        pending = self.context.pending_item
        if pending.current_option_index >= len(pending.pending_options):
            return False
        
        option = pending.pending_options[pending.current_option_index]
        option_id = option.get('id', f'opt_{pending.current_option_index}')
        
        # Find matching option value
        for val in option.get('option_values', []):
            if val['name'].lower() == value_name.lower():
                pending.selected_options[option_id] = {
                    'option_name': option.get('name'),
                    'value_id': val.get('id'),
                    'value_name': val['name'],
                    'price': val.get('price', 0.0)
                }
                pending.current_option_index += 1
                
                # Check if more options needed
                if pending.current_option_index < len(pending.pending_options):
                    return True  # More options to select
                else:
                    self.context.state = ConversationState.AWAITING_ITEM_CONFIRMATION
                    self.context.awaiting_response_for = "item_confirmation"
                    return False  # All options selected
        
        return True  # Option not matched, still waiting
    
    def get_pending_item_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of pending item with selected options"""
        if not self.context.pending_item:
            return None
        
        pending = self.context.pending_item
        total_price = pending.base_price
        options_text = []
        
        for opt_id, opt_val in pending.selected_options.items():
            total_price += opt_val.get('price', 0.0)
            options_text.append(opt_val['value_name'])
        
        # Multiply by quantity
        total_price *= pending.quantity
        
        return {
            'item_id': pending.item_id,
            'item_name': pending.item_name,
            'quantity': pending.quantity,
            'options': pending.selected_options,
            'options_text': ", ".join(options_text) if options_text else None,
            'total_price': total_price
        }
    
    def clear_pending_item(self):
        """Clear the pending item"""
        self.context.pending_item = None
        self.context.state = ConversationState.IDLE
        self.context.awaiting_response_for = None
    
    def has_pending_options(self) -> bool:
        """Check if there are pending options to select"""
        if not self.context.pending_item:
            return False
        pending = self.context.pending_item
        return pending.current_option_index < len(pending.pending_options)
    
    def update_context(self, user_input: str, intent: str, response: str):
        """Update context based on conversation"""
        # Reset if stale
        if self.context.is_stale(self.context_timeout_minutes):
            self.context.reset()
        
        # Add to history
        self.context.add_turn(user_input, intent, response)
        
        # Update state based on intent
        self._update_state_from_intent(intent, user_input, response)
        
        # Extract entities
        self._extract_entities(user_input, intent, response)
    
    def _update_state_from_intent(self, intent: str, user_input: str, response: str):
        """Update conversation state based on intent"""
        state_transitions = {
            "search_menu": ConversationState.BROWSING_MENU,
            "add_to_cart": ConversationState.ADDING_ITEMS,
            "checkout": ConversationState.CHECKOUT,
            "view_cart": ConversationState.ADDING_ITEMS,
            "greeting": ConversationState.IDLE,
        }
        
        if intent in state_transitions:
            self.context.state = state_transitions[intent]
        
        # Check for questions in response
        if "?" in response:
            if "confirm" in response.lower() or "complete" in response.lower():
                self.context.state = ConversationState.CONFIRMING_ORDER
                self.context.awaiting_response_for = "confirmation"
            elif any(word in response.lower() for word in ["what would you like", "what can i get"]):
                self.context.awaiting_response_for = "item_selection"
    
    def _extract_entities(self, user_input: str, intent: str, response: str):
        """Extract entities from conversation"""
        # Extract categories
        categories = ["pizza", "burger", "drink", "dessert", "salad", "side"]
        for cat in categories:
            if cat in user_input.lower():
                self.context.last_category_mentioned = cat
                break
        
        # Extract specific items mentioned
        # This could be enhanced with actual menu item matching
        if intent == "add_to_cart" and "Added" in response:
            # Extract item from response
            import re
            match = re.search(r"Added (.+?) to your order", response)
            if match:
                self.context.last_item_mentioned = match.group(1)
    
    def get_context_hints(self) -> Dict[str, Any]:
        """Get context hints for intent classification and response generation"""
        return {
            "state": self.context.state.value,
            "last_intent": self.context.last_intent,
            "last_category": self.context.last_category_mentioned,
            "last_item": self.context.last_item_mentioned,
            "awaiting_response": self.context.awaiting_response_for,
            "pending_item": self.context.pending_item,
            "previous_turn": self.context.conversation_history[-1] if self.context.conversation_history else None
        }
    
    def should_use_context_for_intent(self, user_input: str) -> bool:
        """Check if context should influence intent classification"""
        # Use context for short/ambiguous inputs
        if len(user_input.split()) <= 2:
            return True
        
        # Use context if we're waiting for a response
        if self.context.awaiting_response_for:
            return True
        
        # Use context for pronouns/references
        context_triggers = ["it", "that", "this", "them", "those", "more", "another", "same", "one", "ones"]
        return any(trigger in user_input.lower() for trigger in context_triggers)
    
    def enhance_intent_with_context(self, user_input: str, predicted_intent: str, confidence: float) -> tuple[str, float]:
        """Enhance intent prediction using context"""
        user_lower = user_input.lower().strip()
        
        # Handle confirmations
        if self.context.awaiting_response_for == "confirmation":
            if user_lower in ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm"]:
                return "confirm_yes", 0.95
            elif user_lower in ["no", "nope", "cancel", "nevermind"]:
                return "confirm_no", 0.95
        
        # Handle referential phrases after showing menu items
        if self.context.last_intent == "search_menu" and self.context.last_category_mentioned:
            # Patterns like "the X one", "X ones", "that one", etc.
            referential_patterns = ["one", "ones", "that", "those", "this", "these"]
            if any(pattern in user_lower for pattern in referential_patterns):
                # Check if it contains an item descriptor
                words = user_lower.split()
                # If it's just "that one" or similar, need more context
                if len(words) <= 2:
                    return "add_to_cart", 0.85
                # If it has descriptor like "vegetarian ones"
                else:
                    return "add_to_cart", 0.9
        
        # Handle follow-up queries
        if self.context.last_intent == "search_menu" and self.context.last_category_mentioned:
            # If user just says an item name after browsing a category
            if len(user_lower.split()) <= 3 and predicted_intent in ["unknown", "greeting"]:
                # Check if it could be an item name or descriptor
                item_words = ["vegetarian", "pepperoni", "margherita", "hawaiian", "bbq", 
                            "chicken", "classic", "cheese", "bacon", "large", "small", 
                            "coffee", "latte", "cola", "coke", "pepsi"]
                if any(word in user_lower for word in item_words):
                    return "add_to_cart", 0.85
        
        # Handle "more" or "what else" after search_menu
        if self.context.last_intent == "search_menu":
            if any(word in user_lower for word in ["more", "else", "other", "another"]):
                return "search_menu", 0.9
        
        # If we're in checkout state, "yes" means confirm
        if self.context.state == ConversationState.CHECKOUT:
            if user_lower in ["yes", "yeah", "yep", "sure"]:
                return "confirm_yes", 0.95
        
        return predicted_intent, confidence
    
    def get_contextual_response_prefix(self) -> Optional[str]:
        """Get a contextual prefix for responses"""
        if self.context.last_category_mentioned and self.context.state == ConversationState.BROWSING_MENU:
            return f"From our {self.context.last_category_mentioned} menu, "
        
        if self.context.state == ConversationState.CONFIRMING_ORDER:
            return "To confirm your order, "
        
        return None