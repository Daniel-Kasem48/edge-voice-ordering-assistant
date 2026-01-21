"""
Dialog Manager for Multi-turn Conversations
Handles stateful interactions, especially for item customization
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

from menu.menu_provider import MenuItem
from config.tenant_loader import Option, OptionValue


class DialogState(Enum):
    """Current state of the dialog"""
    IDLE = "idle"
    ITEM_SELECTION = "item_selection"
    OPTION_SELECTION = "option_selection"
    QUANTITY_SELECTION = "quantity_selection"
    CONFIRMATION = "confirmation"
    COMPLETE = "complete"


@dataclass
class OrderInProgress:
    """Represents an order being built through dialog"""
    item: MenuItem
    selected_options: Dict[str, OptionValue] = field(default_factory=dict)
    quantity: int = 1
    pending_options: List[Option] = field(default_factory=list)
    current_option: Optional[Option] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_next_required_option(self) -> Optional[Option]:
        """Get next mandatory option that needs selection"""
        for option in self.item.options:
            if option.is_mandatory and option.name not in self.selected_options:
                return option
        return None
    
    def get_next_optional_option(self) -> Optional[Option]:
        """Get next optional option that could be selected"""
        for option in self.item.options:
            if not option.is_mandatory and option.name not in self.selected_options:
                return option
        return None
    
    def is_complete(self) -> bool:
        """Check if all required options are selected"""
        return self.get_next_required_option() is None


@dataclass
class DialogContext:
    """Conversation context"""
    state: DialogState = DialogState.IDLE
    current_order: Optional[OrderInProgress] = None
    last_mentioned_items: List[MenuItem] = field(default_factory=list)
    last_query: str = ""
    turn_count: int = 0
    
    def reset(self):
        """Reset dialog to idle state"""
        self.state = DialogState.IDLE
        self.current_order = None
        self.turn_count = 0


class DialogManager:
    """
    Manages multi-turn conversations for ordering
    
    Example flow:
        User: "I want pizza"
        Bot: "Which pizza? We have Test Pizza and Margherita Pizza"
        User: "test pizza"
        Bot: "What size? small ($7), medium ($8), or large ($9)?"
        User: "large"
        Bot: "Added large Test Pizza. Anything else?"
    """
    
    def __init__(self, menu_provider):
        self.menu_provider = menu_provider
        self.contexts: Dict[str, DialogContext] = {}
    
    def get_or_create_context(self, session_id: str) -> DialogContext:
        """Get or create dialog context for session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = DialogContext()
        return self.contexts[session_id]
    
    def handle_item_selection(self, items: List[MenuItem], query: str, session_id: str) -> tuple[DialogState, str]:
        """
        Handle item selection with disambiguation if needed
        
        Returns:
            (next_state, response_text)
        """
        context = self.get_or_create_context(session_id)
        
        if not items:
            return DialogState.IDLE, "I couldn't find that item. What would you like to order?"
        
        if len(items) == 1:
            # Single item found - start customization
            item = items[0]
            context.current_order = OrderInProgress(item=item)
            context.last_mentioned_items = items
            
            # Check if item has mandatory options
            if item.options and any(opt.is_mandatory for opt in item.options):
                context.state = DialogState.OPTION_SELECTION
                return self._handle_next_option(context)
            else:
                # No mandatory options - complete order
                context.state = DialogState.COMPLETE
                return DialogState.COMPLETE, f"Adding {item.name} to your order. No customization needed."
        
        else:
            # Multiple items - need disambiguation
            context.last_mentioned_items = items
            context.state = DialogState.ITEM_SELECTION
            
            item_list = "\n".join([f"• {item.name} (${item.price})" for item in items[:5]])
            return DialogState.ITEM_SELECTION, f"Which one would you like?\n{item_list}"
    
    def _handle_next_option(self, context: DialogContext) -> tuple[DialogState, str]:
        """Handle the next required option selection"""
        if not context.current_order:
            return DialogState.IDLE, "No order in progress."
        
        next_option = context.current_order.get_next_required_option()
        
        if not next_option:
            # All required options selected - check for optional ones
            optional_option = context.current_order.get_next_optional_option()
            if optional_option:
                # Ask about optional option
                context.current_order.current_option = optional_option
                return self._ask_optional_option(context, optional_option)
            else:
                # All options (required + optional) handled
                context.state = DialogState.COMPLETE
                return DialogState.COMPLETE, self._build_order_summary(context.current_order)
        
        # Ask about next option
        context.current_order.current_option = next_option
        
        # Build option prompt
        option_text = f"What {next_option.name.lower()} would you like?"
        
        # Show available values
        values = []
        for value in next_option.option_values:
            if value.price > 0:
                values.append(f"{value.name} (+${value.price})")
            else:
                values.append(value.name)
        
        option_text += f"\nOptions: {', '.join(values)}"
        
        # Suggest default if available
        default = next((v for v in next_option.option_values if v.is_default), None)
        if default:
            option_text += f"\n(Default: {default.name})"
        
        return DialogState.OPTION_SELECTION, option_text
    
    def handle_option_selection(self, user_input: str, session_id: str) -> tuple[DialogState, str]:
        """Handle user's option selection"""
        context = self.get_or_create_context(session_id)
        
        if context.state != DialogState.OPTION_SELECTION or not context.current_order:
            return DialogState.IDLE, "No option selection in progress."
        
        current_option = context.current_order.current_option
        if not current_option:
            return self._handle_next_option(context)
        
        # Check for skip/no for optional options
        skip_words = ["no", "nope", "skip", "none", "nothing", "pass"]
        if user_input.lower().strip() in skip_words and not current_option.is_mandatory:
            # Skip this optional option - move to next
            return self._handle_next_option(context)
        
        # Find matching option value
        selected_value = self._match_option_value(user_input, current_option)
        
        if not selected_value:
            # For optional options, suggest skipping
            if not current_option.is_mandatory:
                return DialogState.OPTION_SELECTION, f"I didn't understand. Please choose from: {', '.join([v.name for v in current_option.option_values])} or say 'no' to skip."
            else:
                return DialogState.OPTION_SELECTION, f"I didn't understand. Please choose from: {', '.join([v.name for v in current_option.option_values])}"
        
        # Store selection
        context.current_order.selected_options[current_option.name] = selected_value
        
        # Move to next option
        return self._handle_next_option(context)
    
    def _match_option_value(self, user_input: str, option: Option) -> Optional[OptionValue]:
        """Match user input to option value"""
        user_input_lower = user_input.lower().strip()
        
        # Exact match
        for value in option.option_values:
            if value.name.lower() == user_input_lower:
                return value
        
        # Partial match
        for value in option.option_values:
            if user_input_lower in value.name.lower() or value.name.lower() in user_input_lower:
                return value
        
        # Common aliases
        size_aliases = {
            "s": "small", "m": "medium", "l": "large", "xl": "extra large",
            "sm": "small", "med": "medium", "lg": "large"
        }
        
        if user_input_lower in size_aliases:
            normalized = size_aliases[user_input_lower]
            for value in option.option_values:
                if value.name.lower() == normalized:
                    return value
        
        return None
    
    def _ask_optional_option(self, context: DialogContext, option: Option) -> tuple[DialogState, str]:
        """Ask about optional option with ability to skip"""
        # Build option prompt
        option_text = f"Would you like to add {option.name.lower()}?"
        
        # Show available values
        values = []
        for value in option.option_values:
            if value.price > 0:
                values.append(f"{value.name} (+${value.price})")
            else:
                values.append(value.name)
        
        option_text += f"\nOptions: {', '.join(values)}"
        option_text += "\nSay the option you want, or 'no' to skip."
        
        context.state = DialogState.OPTION_SELECTION
        return DialogState.OPTION_SELECTION, option_text
    
    def _build_order_summary(self, order: OrderInProgress) -> str:
        """Build order completion message"""
        item_desc = order.item.name
        
        # Add selected options
        if order.selected_options:
            options = []
            for opt_name, opt_value in order.selected_options.items():
                options.append(f"{opt_value.name}")
            item_desc = f"{' '.join(options)} {item_desc}"
        
        # Calculate price
        total_price = order.item.price
        for opt_value in order.selected_options.values():
            total_price += opt_value.price - opt_value.discount_amount
        
        total_price *= order.quantity
        
        return f"Perfect! Adding {item_desc} to your order (${total_price:.2f}). Anything else?"
    
    def extract_size_from_input(self, user_input: str) -> Optional[str]:
        """Extract size specification from input like 'large pizza' or 'pizza large'"""
        sizes = ["small", "medium", "large", "extra large", "xl", "sm", "med", "lg"]
        input_lower = user_input.lower()
        
        for size in sizes:
            if size in input_lower:
                # Normalize abbreviations
                if size in ["s", "sm"]:
                    return "small"
                elif size in ["m", "med"]:
                    return "medium"
                elif size in ["l", "lg"]:
                    return "large"
                elif size in ["xl", "extra large"]:
                    return "extra large"
                return size
        
        return None