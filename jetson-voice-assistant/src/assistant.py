"""
Text-based POS Voice Assistant
Multi-tenant restaurant order assistant (text mode for testing)
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.tenant_loader import TenantLoader, TenantConfig
from input_output.text_io import TextInputEngine, TextWakeWordDetector, TextOutputEngine, TranscriptionResult
from nlp.intent_classifier import IntentClassifier, IntentResult
from nlp.intent_router import IntentRouter
from nlp.menu_matcher import MenuMatcher, ParsedOrder
from state.cart import ShoppingCart, CartManager
from state.context_manager import ContextManager
from menu.vocabulary import MenuVocabulary, MenuVocabularyBuilder, VocabularyMatcher
from output.speech_formatter import SpeechFormatter


class TextAssistant:
    """
    Text-based restaurant POS assistant
    Processes orders through text input/output
    """
    
    def __init__(
        self,
        tenant_id: str = "demo_restaurant",
        tenants_path: str = "data/tenants"
    ):
        """
        Initialize the assistant
        
        Args:
            tenant_id: Tenant configuration to load
            tenants_path: Path to tenant configurations
        """
        print(f"🚀 Initializing Text Assistant for tenant: {tenant_id}")
        
        # Load tenant configuration
        self.tenant_loader = TenantLoader(tenants_path)
        self.config = self.tenant_loader.load_tenant(tenant_id)
        
        # Initialize components
        self.input_engine = TextInputEngine(prompt="You: ")
        self.output_engine = TextOutputEngine(prefix="🤖 ")
        self.wake_detector = TextWakeWordDetector(
            wake_phrase=self.config.wake_word_phrase
        )
        
        # NLP components
        # Try to use FastText model if available, otherwise fall back to rules
        try:
            from nlp.lightweight_intent_classifier import LightweightIntentClassifier
            from nlp.intent_router import IntentRouterConfig
            # Check for restaurant-optimized model first (priority order)
            model_options = [
                ("models/restaurant_intent_medium.bin", "menu-trained medium model (19MB)"),
                ("models/restaurant_intent_small.bin", "menu-trained small model (5MB)"),
                ("models/restaurant_intent_optimized.bin", "optimized model (20MB)"),
                ("models/restaurant_intent.bin", "restaurant model"),
                ("models/fasttext_intent.bin", "FastText model"),
            ]
            
            model_loaded = False
            for model_path, model_name in model_options:
                if os.path.exists(model_path):
                    model_classifier = LightweightIntentClassifier(
                        model_type="fasttext",
                        model_path=model_path,
                        fallback_to_rules=True
                    )
                    self.intent_classifier = IntentRouter(
                        model_classifier=model_classifier,
                        config=IntentRouterConfig(model_confidence_threshold=0.70)
                    )
                    print(f"  ✅ Using {model_name}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                self.intent_classifier = IntentClassifier(use_rules_fallback=True)
                print("  📋 Using rule-based intent classifier")
        except ImportError:
            self.intent_classifier = IntentClassifier(use_rules_fallback=True)
            print("  📋 Using rule-based intent classifier")
            
        self.menu_matcher = MenuMatcher(self.config)
        
        # Build vocabulary dynamically from menu (no hardcoded item names)
        menu_items = [self._item_to_dict(item) for item in self.config.menu_items]
        self.vocabulary = MenuVocabularyBuilder(menu_items).build()
        self.vocab_matcher = VocabularyMatcher(self.vocabulary)
        print(f"  📖 Vocabulary: {len(self.vocabulary.categories)} categories, {len(self.vocabulary.tags)} tags")
        
        # Cart management
        self.cart_manager = CartManager()
        self.current_session = "default"
        
        # Context management
        self.context_manager = ContextManager()
        
        # Speech formatting for TTS
        self.speech_formatter = SpeechFormatter()
        
        # State
        self.running = False
        self.awaiting_confirmation = False
        self.pending_action: Optional[Dict[str, Any]] = None
        
        print(f"✅ Assistant ready for: {self.config.tenant_name}")
    
    def _item_to_dict(self, item) -> Dict:
        """Convert menu item object to dict for vocabulary building"""
        return {
            'name': getattr(item, 'name', ''),
            'category': getattr(item, 'category', ''),
            'tags': getattr(item, 'tags', []),
            'options': [
                {
                    'name': getattr(opt, 'name', ''),
                    'option_values': [
                        {'name': getattr(v, 'name', '')} 
                        for v in getattr(opt, 'option_values', [])
                    ]
                }
                for opt in getattr(item, 'options', [])
            ],
            'description': getattr(item, 'description', ''),
        }
    
    @property
    def cart(self) -> ShoppingCart:
        """Get current session's cart"""
        return self.cart_manager.get_cart(self.current_session, self.config.tenant_id)
    
    def respond(self, message: str, response_type: str = 'auto'):
        """
        Send response to user with TTS formatting
        
        Args:
            message: Raw response text
            response_type: Type of response (auto, general, cart, menu, confirmation, option)
        """
        # Auto-detect response type from content if not specified
        if response_type == 'auto':
            response_type = self._detect_response_type(message)
        
        # Format for natural speech
        if response_type == 'cart':
            formatted = self.speech_formatter.format_cart_for_speech(message)
        elif response_type == 'menu':
            formatted = self.speech_formatter.format_menu_for_speech(message)
        elif response_type == 'confirmation':
            formatted = self.speech_formatter.format_confirmation_for_speech(message)
        elif response_type == 'option':
            formatted = self.speech_formatter.format_option_prompt(message)
        else:
            formatted = self.speech_formatter.format_for_speech(message)
        
        self.output_engine.speak(formatted)
    
    def _detect_response_type(self, message: str) -> str:
        """Auto-detect response type from message content"""
        message_lower = message.lower()
        
        # Check for cart-related content
        if any(word in message_lower for word in ['your cart', 'total', 'including tax', 'cart is empty']):
            return 'cart'
        
        # Check for menu listing
        if any(word in message_lower for word in ['our menu', 'we have', 'available', 'include:']):
            return 'menu'
        
        # Check for option prompts
        if any(word in message_lower for word in ['what size', 'which option', 'choose from', 'select']):
            return 'option'
        
        # Check for confirmations
        if any(word in message_lower for word in ['confirm', 'ready to checkout', 'would you like to complete']):
            return 'confirmation'
        
        return 'general'
    
    def process_intent(self, text: str, intent: IntentResult) -> str:
        """
        Process user intent and return response
        
        Args:
            text: Original user text
            intent: Classified intent
            
        Returns:
            Response message
        """
        intent_name = intent.intent
        
        # Intent handlers
        handlers = {
            "add_to_cart": self._handle_add_to_cart,
            "remove_from_cart": self._handle_remove_from_cart,
            "view_cart": self._handle_view_cart,
            "clear_cart": self._handle_clear_cart,
            "checkout": self._handle_checkout,
            "search_menu": self._handle_search_menu,
            "view_menu": self._handle_search_menu,  # Alias for search_menu
            "query_item": self._handle_query_item,
            "query_ingredients": self._handle_query_ingredients,
            "get_price": self._handle_get_price,
            "modify_item": self._handle_modify_item,
            "help": self._handle_help,
            "greeting": self._handle_greeting,
            "goodbye": self._handle_goodbye,
            "confirm_yes": self._handle_confirm_yes,
            "confirm_no": self._handle_confirm_no,
        }
        
        handler = handlers.get(intent_name, self._handle_unknown)
        return handler(text, intent)
    
    def _handle_add_to_cart(self, text: str, intent: IntentResult) -> str:
        """Handle add to cart intent"""
        # Check context for referential phrases
        context = self.context_manager.get_context_hints()
        text_lower = text.lower().strip().rstrip('?')  # Remove trailing question mark
        
        # Handle referential phrases with context
        # Also handle single descriptors after showing a menu
        if context['last_category'] and (any(word in text_lower for word in ["one", "ones", "that", "those"]) or len(text_lower.split()) <= 2):
            # Try to find a matching item in the last category using fuzzy search
            # instead of hardcoded descriptors
            results = self.menu_matcher.search_menu(text_lower, limit=3)
            if results:
                # Filter to items in the last category
                category_results = [r for r in results if r[0].category.lower() == context['last_category'].lower()]
                if category_results:
                    best_match = category_results[0][0]
                    text = best_match.name
                    print(f"  [Context: matched '{text_lower}' → '{text}' in category {context['last_category']}]")
        
        order = self.menu_matcher.parse_order(text)
        
        if not order.items:
            # Check if this is a generic "add" without specific item
            generic_add_phrases = ["add another", "add more", "add something", "add item", "another item", "something else", "order more"]
            if any(phrase in text.lower() for phrase in generic_add_phrases):
                categories = set(item.category for item in self.config.menu_items)
                cat_list = ", ".join(categories)
                return f"Sure! We have {cat_list}. What would you like to add?"
            
            # Disambiguation: if we have plausible candidates, ask instead of failing.
            candidates = self.menu_matcher.get_item_candidates(text, limit=2)
            if len(candidates) >= 2:
                (a, a_score), (b, b_score) = candidates[0], candidates[1]
                if a_score >= 70 and (a_score - b_score) <= 6:
                    return (
                        f"Did you mean '{a.name}' or '{b.name}'? "
                        "Reply with the item name."
                    )
            elif len(candidates) == 1 and candidates[0][1] >= 70:
                a, _ = candidates[0]
                return f"Did you mean '{a.name}'? Reply with the item name."

            # Try to find what they might want
            results = self.menu_matcher.search_menu(text, limit=3)
            if results:
                suggestions = ", ".join([r[0].name for r in results])
                return f"I couldn't find that exact item. Did you mean: {suggestions}?"
            return "I couldn't find that item on our menu. Try 'help' to see what's available."
        
        # Check if any item has mandatory options that need selection
        for matched_item in order.items:
            item = matched_item.item
            mandatory_options = [opt for opt in item.options if opt.is_mandatory]
            
            # Check if options are already selected (e.g., "large coffee" already has size)
            already_selected_option_ids = {so.option.id for so in matched_item.selected_options}
            unselected_mandatory = [opt for opt in mandatory_options if opt.id not in already_selected_option_ids]
            
            if unselected_mandatory:
                # Need to ask for options - trigger options flow
                return self._start_options_flow(item, matched_item.quantity)
        
        # No mandatory options or all already selected - add items to cart
        added = []
        for item in order.items:
            self.cart.add_item(item)
            added.append(item.get_description())
        
        added_text = ", ".join(added)
        total = self.cart.get_total()
        
        return f"Added {added_text} to your order. Your total is now ${total:.2f}."
    
    def _start_options_flow(self, item, quantity: int = 1) -> str:
        """Start the options selection flow for an item with mandatory options"""
        mandatory_options = [opt for opt in item.options if opt.is_mandatory]
        
        options_data = []
        for opt in mandatory_options:
            opt_data = {
                'id': opt.id,
                'name': opt.name,
                'type': opt.type,
                'is_mandatory': opt.is_mandatory,
                'option_values': [
                    {'id': v.id, 'name': v.name, 'price': v.price, 'is_default': v.is_default}
                    for v in opt.option_values
                ]
            }
            options_data.append(opt_data)
        
        self.context_manager.set_pending_item(
            item_id=item.id,
            item_name=item.name,
            base_price=item.price,
            options=options_data
        )
        
        # Store quantity in pending item
        if self.context_manager.context.pending_item:
            self.context_manager.context.pending_item.quantity = quantity
        
        # Ask for first option
        option_prompt = self.context_manager.get_current_option_prompt()
        return f"Got it, {item.name}! {option_prompt}"
    
    def _handle_remove_from_cart(self, text: str, intent: IntentResult) -> str:
        """Handle remove from cart intent"""
        if self.cart.is_empty():
            return "Your cart is empty, there's nothing to remove."
        
        # Check for "last item" or "last one"
        if "last" in text.lower():
            removed = self.cart.remove_last()
            if removed:
                return f"Removed {removed.matched_item.get_description()} from your order."
        
        # Extract item name from "remove X" or "remove the X"
        # Remove command words to get just the item name
        text_clean = text.lower()
        for word in ['remove', 'delete', 'take off', 'take out', 'cancel', 'the', 'my']:
            text_clean = text_clean.replace(word, '')
        text_clean = text_clean.strip()
        
        if not text_clean:
            return "What would you like to remove? Say the item name or 'remove last item'."
        
        # Try to find item in cart by fuzzy matching the name
        removed = self.cart.remove_by_name(text_clean)
        if removed:
            total = self.cart.get_total()
            if self.cart.is_empty():
                return f"Removed {removed.matched_item.get_description()}. Your cart is now empty."
            return f"Removed {removed.matched_item.get_description()}. Your total is now ${total:.2f}."
        
        return f"I couldn't find '{text_clean}' in your cart. Your items are: {', '.join([item.matched_item.item.name for item in self.cart.items])}"
    
    def _handle_view_cart(self, text: str, intent: IntentResult) -> str:
        """Handle view cart intent"""
        if self.cart.is_empty():
            return "Your cart is empty. Would you like to order something?"
        return self.cart.get_voice_summary()
    
    def _handle_clear_cart(self, text: str, intent: IntentResult) -> str:
        """Handle clear cart intent"""
        if self.cart.is_empty():
            return "Your cart is already empty."
        
        count = self.cart.get_item_count()
        self.cart.clear()
        return f"Cart cleared. Removed {count} item{'s' if count != 1 else ''}."
    
    def _handle_checkout(self, text: str, intent: IntentResult) -> str:
        """Handle checkout intent"""
        if self.cart.is_empty():
            return "Your cart is empty. Add some items before checking out."
        
        summary = self.cart.get_summary()
        return f"Ready to checkout!\n\n{summary}\n\nWould you like to complete this order? (say 'yes' to confirm)"
    
    def _check_ingredient_query(self, text: str) -> Optional[str]:
        """
        Check if user is asking about ingredients.
        Patterns: "what's in X", "ingredients in X", "does X have Y", "is there Y in X"
        Returns response if ingredient query, None otherwise.
        """
        # Patterns for ingredient queries
        ingredient_patterns = [
            r"what'?s?\s+in\s+(?:the\s+)?(.+)",
            r"what\s+are\s+the\s+ingredients\s+(?:in|of|for)\s+(?:the\s+)?(.+)",
            r"ingredients\s+(?:in|of|for)\s+(?:the\s+)?(.+)",
            r"show\s+(?:me\s+)?(?:the\s+)?(.+?)\s+ingredients",  # "show me pizza ingredients"
            r"(?:the\s+)?(.+?)\s+ingredients",  # "pizza ingredients"
            r"how\s+(?:is\s+)?(?:the\s+)?(.+?)\s+(?:is\s+)?made",  # "how pizza is made"
            r"how\s+do\s+you\s+make\s+(?:the\s+)?(.+)",  # "how do you make pizza"
            r"what'?s?\s+(?:the\s+)?(.+?)\s+made\s+(?:of|with|from)",  # "what's pizza made of"
            r"does\s+(?:the\s+)?(.+?)\s+have\s+(.+)",
            r"is\s+there\s+(.+?)\s+in\s+(?:the\s+)?(.+)",
            r"(?:does|do)\s+(?:the\s+)?(.+?)\s+contain\s+(.+)",
        ]
        
        for pattern in ingredient_patterns:
            match = re.match(pattern, text)
            if match:
                groups = match.groups()
                
                # Two group patterns: "does X have Y" - check specific ingredient
                if len(groups) == 2:
                    item_query = groups[0].strip() if "have" in pattern or "contain" in pattern else groups[1].strip()
                    ingredient_query = groups[1].strip() if "have" in pattern or "contain" in pattern else groups[0].strip()
                    return self._answer_ingredient_check(item_query, ingredient_query)
                
                # Single group pattern: "what's in X" - list all ingredients
                else:
                    item_query = groups[0].strip()
                    return self._list_ingredients(item_query)
        
        return None
    
    def _list_ingredients(self, item_query: str) -> str:
        """List all ingredients in an item"""
        # Try to find the item
        order = self.menu_matcher.parse_order(item_query)
        
        if order.items:
            item = order.items[0].item
            
            # Check if there are multiple high-confidence matches
            results = self.menu_matcher.search_menu(item_query, limit=10)
            high_matches = [(r[0], r[1]) for r in results if r[1] >= 85]  # Stricter threshold
            
            # If multiple matches with similar scores, ask for clarification
            # BUT: if the top match is significantly better (>10% difference), use it
            if len(high_matches) > 1:
                top_score = high_matches[0][1]
                second_score = high_matches[1][1]
                
                # If top match is significantly better, use it
                if top_score - second_score >= 10:
                    # Clear winner - proceed with single match
                    pass
                else:
                    # Similar scores - check if they're all from same category
                    categories = set([m[0].category for m in high_matches])
                    if len(categories) == 1:
                        # Multiple items in same category
                        items_list = ", ".join([m[0].name for m in high_matches])
                        return f"We have several {item.category} options: {items_list}. Which one would you like to know about?"
            
            # Single match - show ingredients
            if hasattr(item, 'ingredients') and item.ingredients:
                ingredients_text = ", ".join(item.ingredients)
                return f"{item.name} contains: {ingredients_text}."
            else:
                return f"{item.name} doesn't have ingredient information available."
        
        # Try fuzzy search
        results = self.menu_matcher.search_menu(item_query, limit=5)
        if results and results[0][1] > 70:  # High confidence match
            # Check for multiple similar matches
            high_matches = [(r[0], r[1]) for r in results if r[1] >= results[0][1] - 10]
            if len(high_matches) > 1:
                items_list = ", ".join([m[0].name for m in high_matches])
                return f"Which one? We have: {items_list}"
            
            item = results[0][0]
            if hasattr(item, 'ingredients') and item.ingredients:
                ingredients_text = ", ".join(item.ingredients)
                return f"{item.name} contains: {ingredients_text}."
            else:
                return f"{item.name} doesn't have ingredient information available."
        
        return f"I couldn't find '{item_query}' on the menu. Could you try asking about a specific item?"
    
    def _answer_ingredient_check(self, item_query: str, ingredient_query: str) -> str:
        """Check if an item contains a specific ingredient"""
        # Find the item
        order = self.menu_matcher.parse_order(item_query)
        item = None
        
        if order.items:
            item = order.items[0].item
        else:
            # Try fuzzy search
            results = self.menu_matcher.search_menu(item_query, limit=1)
            if results and results[0][1] > 70:
                item = results[0][0]
        
        if not item:
            return f"I couldn't find '{item_query}' on the menu."
        
        # Check if item has ingredients
        if not hasattr(item, 'ingredients') or not item.ingredients:
            return f"{item.name} doesn't have ingredient information available."
        
        # Check if ingredient is in the item
        ingredients_lower = [ing.lower() for ing in item.ingredients]
        ingredient_query_lower = ingredient_query.lower()
        
        # Check for exact or partial match
        found = False
        for ing in ingredients_lower:
            if ingredient_query_lower in ing or ing in ingredient_query_lower:
                found = True
                break
        
        if found:
            return f"Yes, {item.name} contains {ingredient_query}."
        else:
            # List what it does have
            ingredients_text = ", ".join(item.ingredients[:5])  # First 5 ingredients
            if len(item.ingredients) > 5:
                ingredients_text += f", and {len(item.ingredients)-5} more"
            return f"No, {item.name} doesn't contain {ingredient_query}. It has: {ingredients_text}."
    
    def _handle_query_ingredients(self, text: str, intent: IntentResult) -> str:
        """Handle ingredient queries (what's in X?, does X have Y?)"""
        text_lower = text.lower()
        
        # Try to extract ingredient query using patterns
        response = self._check_ingredient_query(text_lower)
        if response:
            return response
        
        # If no specific pattern matched, try to find any menu item mentioned in the text
        # This handles cases like "I want burger, show ingredients"
        order = self.menu_matcher.parse_order(text_lower)
        if order.items:
            item = order.items[0].item
            
            # Check if there are multiple matches - if so, ask user to clarify
            results = self.menu_matcher.search_menu(text_lower, limit=10)
            high_matches = [(r[0], r[1]) for r in results if r[1] >= 80]
            
            # If multiple high-confidence matches, list them
            if len(high_matches) > 1:
                # Check if they're all from the same category
                categories = set([m[0].category for m in high_matches])
                if len(categories) == 1:
                    # Multiple items in same category - list them
                    items_list = ", ".join([m[0].name for m in high_matches[:5]])
                    return f"We have several {item.category} options: {items_list}. Which one would you like to know about?"
            
            # Single match or clear winner - show ingredients
            if hasattr(item, 'ingredients') and item.ingredients:
                ingredients_text = ", ".join(item.ingredients)
                return f"{item.name} has {ingredients_text}."
            else:
                return f"{item.name} doesn't have ingredient information available."
        
        # Try fuzzy search for item name
        results = self.menu_matcher.search_menu(text_lower, limit=5)
        if results and results[0][1] > 60:
            item = results[0][0]
            
            # Check for multiple similar matches
            high_matches = [(r[0], r[1]) for r in results if r[1] >= results[0][1] - 10]
            if len(high_matches) > 1:
                items_list = ", ".join([m[0].name for m in high_matches])
                return f"Which one? {items_list}"
            
            if hasattr(item, 'ingredients') and item.ingredients:
                ingredients_text = ", ".join(item.ingredients)
                return f"{item.name} has {ingredients_text}."
        
        # If no item specified, ask which item they want to know about
        return "Which item would you like to know the ingredients for?"
    
    def _handle_query_item(self, text: str, intent: IntentResult) -> str:
        """Handle query about item availability (do you have X?) or ingredients (what's in X?)"""
        text_lower = text.lower()
        
        # Check if asking about ingredients
        ingredient_query = self._check_ingredient_query(text_lower)
        if ingredient_query:
            return ingredient_query
        
        # First check if asking about a tag (vegetarian, breakfast, gluten-free, etc.)
        matched_tag = self.vocab_matcher.match_tag(text_lower)
        if matched_tag:
            # Redirect to search_menu for tag-based filtering
            return self._handle_search_menu(text, intent)
        
        # Check if asking about a category
        categories = {}
        for item in self.config.menu_items:
            cat = item.category.lower()
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # Check for category match (singular or plural)
        for cat, items in categories.items():
            cat_singular = cat.rstrip('s')
            if cat in text_lower or cat_singular in text_lower:
                # List all items in this category
                items_text = ", ".join([i.name for i in items])
                if len(items) > 5:
                    items_text = ", ".join([i.name for i in items[:5]]) + f" and {len(items)-5} more"
                return f"Yes! We have these {cat}: {items_text}. Which one would you like?"
        
        # Try to parse what they're asking about
        order = self.menu_matcher.parse_order(text)
        
        if order.items:
            # Found exact match
            item = order.items[0].item
            return self._offer_item_with_options(item)
        
        # Try fuzzy search with higher limit
        results = self.menu_matcher.search_menu(text, limit=5)
        if results:
            # Filter to only high-confidence matches (score > 60)
            good_matches = [(item, score) for item, score in results if score > 60]
            
            if len(good_matches) == 1:
                item = good_matches[0][0]
                return self._offer_item_with_options(item)
            elif good_matches:
                # Multiple matches - list them all
                items_text = ", ".join([r[0].name for r in good_matches])
                return f"We have: {items_text}. Which one would you like?"
            else:
                # Low confidence matches - still suggest them
                items_text = ", ".join([r[0].name for r in results[:3]])
                return f"I'm not sure exactly, but we have: {items_text}. Is that what you're looking for?"
        
        # Not found - suggest alternatives or browse menu
        return "I don't see that on our menu. Say 'menu' to see what we have, or try describing what you're looking for."
    
    def _offer_item_with_options(self, item) -> str:
        """Offer an item, checking if it has mandatory options"""
        # Check if item has mandatory options
        mandatory_options = [opt for opt in item.options if opt.is_mandatory]
        
        if mandatory_options:
            # Set up pending item with options
            options_data = []
            for opt in mandatory_options:
                opt_data = {
                    'id': opt.id,
                    'name': opt.name,
                    'type': opt.type,
                    'is_mandatory': opt.is_mandatory,
                    'option_values': [
                        {'id': v.id, 'name': v.name, 'price': v.price, 'is_default': v.is_default}
                        for v in opt.option_values
                    ]
                }
                options_data.append(opt_data)
            
            self.context_manager.set_pending_item(
                item_id=item.id,
                item_name=item.name,
                base_price=item.price,
                options=options_data
            )
            
            # Ask for first option
            option_prompt = self.context_manager.get_current_option_prompt()
            return f"Sure, {item.name}! {option_prompt}"
        else:
            # No mandatory options - ask if they want to add it
            return f"Yes, we have {item.name} for ${item.price:.2f}. Would you like to add it to your order?"
    
    def _handle_option_selection(self, text: str) -> str:
        """Handle option selection for pending item"""
        text_lower = text.lower().strip()
        
        # Check for cancel/nevermind
        if text_lower in ["cancel", "nevermind", "never mind", "no", "forget it"]:
            item_name = self.context_manager.context.pending_item.item_name if self.context_manager.context.pending_item else "that"
            self.context_manager.clear_pending_item()
            return f"No problem, cancelled {item_name}. What else can I get you?"
        
        # Try to match the option value
        pending = self.context_manager.context.pending_item
        if not pending:
            return "I'm not sure what you're selecting. Can you start over?"
        
        current_option = pending.pending_options[pending.current_option_index]
        option_values = current_option.get('option_values', [])
        
        # Try to find matching option value (fuzzy match)
        matched_value = None
        for val in option_values:
            val_name = val['name'].lower()
            if val_name == text_lower or val_name in text_lower or text_lower in val_name:
                matched_value = val['name']
                break
        
        if not matched_value:
            # Didn't understand - re-prompt
            choices = ", ".join([v['name'] for v in option_values])
            return f"Sorry, I didn't catch that. Please choose: {choices}"
        
        # Select the option
        has_more = self.context_manager.select_option(matched_value)
        
        if has_more:
            # More options to select
            next_prompt = self.context_manager.get_current_option_prompt()
            return f"Got it, {matched_value}! {next_prompt}"
        else:
            # All options selected - ask for confirmation
            summary = self.context_manager.get_pending_item_summary()
            if summary['options_text']:
                return f"So that's a {summary['options_text']} {summary['item_name']} for ${summary['total_price']:.2f}. Add it to your order?"
            else:
                return f"So that's a {summary['item_name']} for ${summary['total_price']:.2f}. Add it to your order?"
    
    def _handle_pending_item_confirmation(self, text: str) -> str:
        """Handle confirmation of pending item with selected options"""
        text_lower = text.lower().strip()
        
        # Check for yes/no
        yes_words = ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "add it", "yes please"]
        no_words = ["no", "nope", "cancel", "nevermind", "never mind", "forget it"]
        
        if any(word in text_lower for word in yes_words):
            # Add the item to cart
            summary = self.context_manager.get_pending_item_summary()
            if not summary:
                self.context_manager.clear_pending_item()
                return "Something went wrong. Let's start over - what would you like?"
            
            # Find the original menu item
            item = self.config.find_menu_item(summary['item_name'])
            if item:
                from nlp.menu_matcher import MatchedItem, SelectedOption
                from config.tenant_loader import Option, OptionValue
                
                # Convert selected options dict to SelectedOption objects
                selected_options = []
                for opt_id, opt_data in summary['options'].items():
                    # Find the original option from the item
                    for item_opt in item.options:
                        if item_opt.id == opt_id:
                            # Find the selected value
                            for val in item_opt.option_values:
                                if val.id == opt_data.get('value_id') or val.name == opt_data.get('value_name'):
                                    selected_options.append(SelectedOption(
                                        option=item_opt,
                                        selected_values=[val]
                                    ))
                                    break
                            break
                
                matched = MatchedItem(
                    item=item,
                    quantity=summary.get('quantity', 1),
                    modifiers=[],
                    toppings=[],
                    selected_options=selected_options,
                    match_score=100.0
                )
                self.cart.add_item(matched)
                
                total = self.cart.get_total()
                self.context_manager.clear_pending_item()
                
                qty_text = f"{summary['quantity']}x " if summary.get('quantity', 1) > 1 else ""
                if summary['options_text']:
                    return f"Added {qty_text}{summary['options_text']} {summary['item_name']} to your order. Your total is now ${total:.2f}."
                else:
                    return f"Added {qty_text}{summary['item_name']} to your order. Your total is now ${total:.2f}."
            else:
                self.context_manager.clear_pending_item()
                return "Something went wrong finding that item. Please try again."
        
        elif any(word in text_lower for word in no_words):
            item_name = self.context_manager.context.pending_item.item_name if self.context_manager.context.pending_item else "that"
            self.context_manager.clear_pending_item()
            return f"No problem, cancelled {item_name}. What else can I get you?"
        
        else:
            # Didn't understand
            return "Just say 'yes' to add it or 'no' to cancel."
    
    def _handle_search_menu(self, text: str, intent: IntentResult) -> str:
        """Handle menu search intent"""
        # Check for follow-up context
        context = self.context_manager.get_context_hints()
        if context['last_category'] and any(word in text.lower() for word in ['more', 'else', 'other']):
            text = f"show me more {context['last_category']}s"
        
        # Build category and tag indexes
        categories = {}
        for item in self.config.menu_items:
            cat = item.category.lower()
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        text_lower = text.lower()
        
        # Check for tag-based filtering (breakfast, vegetarian, gluten-free, etc.)
        matched_tag = self.vocab_matcher.match_tag(text_lower)
        if matched_tag:
            # Filter items by tag
            matching_items = []
            for item in self.config.menu_items:
                item_tags = getattr(item, 'tags', []) or []
                # Convert to lowercase for comparison
                item_tags_lower = [t.lower() for t in item_tags]
                if matched_tag.lower() in item_tags_lower:
                    matching_items.append(item)
            
            if matching_items:
                # Group by category for better display
                items_by_cat = {}
                for item in matching_items:
                    cat = item.category
                    if cat not in items_by_cat:
                        items_by_cat[cat] = []
                    items_by_cat[cat].append(item.name)
                
                # Format response
                if len(items_by_cat) == 1:
                    cat = list(items_by_cat.keys())[0]
                    items_text = ", ".join(items_by_cat[cat][:5])
                    if len(items_by_cat[cat]) > 5:
                        items_text += f" and {len(items_by_cat[cat])-5} more"
                    return f"Our {matched_tag} {cat} options: {items_text}."
                else:
                    parts = []
                    for cat, items in list(items_by_cat.items())[:3]:
                        items_text = ", ".join(items[:3])
                        parts.append(f"{cat}: {items_text}")
                    return f"Our {matched_tag} options include: {'; '.join(parts)}. What would you like?"
            else:
                return f"Sorry, we don't have any {matched_tag} items right now."
        
        # Check for "without" or "no" modifiers
        filter_out = []
        has_filter = "without" in text_lower or text_lower.startswith("no ")
        
        if has_filter:
            # Extract filter word(s) directly from "without X" pattern
            import re
            filter_match = re.search(r'\b(?:without|no)\s+(\w+)', text_lower)
            if filter_match:
                filter_out.append(filter_match.group(1))
        
        # Find matching category
        for cat, items in categories.items():
            # Check both singular and plural forms
            cat_singular = cat.rstrip('s')
            if cat in text_lower or cat_singular in text_lower or f"{cat_singular}s" in text_lower:
                # Filter items if needed
                if filter_out:
                    filtered_items = []
                    for item in items:
                        item_name_lower = item.name.lower()
                        
                        # Check if filter word is in item NAME (e.g., "chicken" in "bbq chicken pizza")
                        name_has_unwanted = any(fw in item_name_lower for fw in filter_out)
                        
                        # Also check default toppings
                        toppings_have_unwanted = any(
                            any(fw in topping.lower() for fw in filter_out)
                            for topping in item.default_toppings
                        )
                        
                        if not name_has_unwanted and not toppings_have_unwanted:
                            filtered_items.append(item.name)
                    
                    if filtered_items:
                        items_text = ", ".join(filtered_items)
                        filter_text = " or ".join(filter_out)
                        return f"Our {cat} without {filter_text}: {items_text}. Say 'add' followed by the name to order!"
                    else:
                        return f"All our {cat} can be customized. Just say 'add a {cat[:-1] if cat.endswith('s') else cat} without {filter_out[0]}'."
                
                # No filter, show all
                items_text = ", ".join([i.name for i in items[:5]])
                if len(items) > 5:
                    items_text += f" and {len(items)-5} more"
                return f"Our {cat} include: {items_text}."
        
        # Check if asking about general menu
        if "menu" in text_lower or "show" in text_lower or text_lower in ["breakfast", "lunch", "dinner"]:
            cat_list = ", ".join(categories.keys())
            sample_items = []
            for cat, items in list(categories.items())[:3]:
                sample_items.append(f"{cat}: {items[0].name}")
            samples = ", ".join(sample_items)
            return f"We have {cat_list}. For example: {samples}. What would you like?"
        
        # General menu overview
        cat_list = ", ".join(categories.keys())
        return f"We have {cat_list}. What would you like to know more about?"
    
    def _handle_get_price(self, text: str, intent: IntentResult) -> str:
        """Handle price inquiry"""
        order = self.menu_matcher.parse_order(text)
        
        if order.items:
            item = order.items[0]
            return f"The {item.item.name} is ${item.item.price:.2f}."
        
        # Try fuzzy search
        results = self.menu_matcher.search_menu(text, limit=1)
        if results:
            item, score = results[0]
            return f"The {item.name} is ${item.price:.2f}."
        
        return "Which item would you like to know the price of?"
    
    def _handle_modify_item(self, text: str, intent: IntentResult) -> str:
        """Handle modify item intent"""
        text_lower = text.lower()
        
        # Check if this is a generic "add more" / "add another" without specific item
        generic_add_phrases = ["add another", "add more", "add something", "add item", "another item", "something else"]
        if any(phrase in text_lower for phrase in generic_add_phrases):
            categories = set(item.category for item in self.config.menu_items)
            cat_list = ", ".join(categories)
            return f"Sure! We have {cat_list}. What would you like to add?"
        
        # Check if this is actually an order without explicit "add" keyword
        # Example: "pizza without onion" should be treated as add
        order = self.menu_matcher.parse_order(text)
        
        if order.items:
            # This is an order, handle as add_to_cart
            return self._handle_add_to_cart(text, intent)
        
        if self.cart.is_empty():
            return "Your cart is empty. Add an item first, then modify it."
        
        # Get a sample item dynamically from menu
        sample_item = self.vocabulary.all_item_names[0] if self.vocabulary.all_item_names else "item"
        return f"Modifications are applied when you add items. Try: 'add a {sample_item}'"
    
    def _handle_help(self, text: str, intent: IntentResult) -> str:
        """Handle help request - uses dynamic examples from menu"""
        # Get dynamic examples from vocabulary
        help_text = self.vocab_matcher.get_help_examples()
        return f"I can help you order! Try saying:\n{help_text}"
    
    def _handle_greeting(self, text: str, intent: IntentResult) -> str:
        """Handle greeting"""
        return f"Hello! Welcome to {self.config.tenant_name}. What can I get for you today?"
    
    def _handle_goodbye(self, text: str, intent: IntentResult) -> str:
        """Handle goodbye"""
        if not self.cart.is_empty():
            return f"Thanks for your order! Your total is ${self.cart.get_total():.2f}. Have a great day!"
        return "Thanks for visiting! Come back soon!"
    
    def _handle_confirm_yes(self, text: str, intent: IntentResult) -> str:
        """Handle confirmation yes"""
        context = self.context_manager.get_context_hints()
        
        if context['awaiting_response'] == 'confirmation' and not self.cart.is_empty():
            # Process checkout
            total = self.cart.get_total()
            items = self.cart.get_item_count()
            self.cart.clear()
            return f"Perfect! Your order of {items} items for ${total:.2f} has been confirmed. It will be ready soon!"
        
        return "Great! What would you like to order?"
    
    def _handle_confirm_no(self, text: str, intent: IntentResult) -> str:
        """Handle confirmation no"""
        context = self.context_manager.get_context_hints()
        
        if context['awaiting_response'] == 'confirmation':
            return "No problem! Feel free to modify your order or add more items."
        
        return "Alright, let me know if you need anything else."
    
    def _handle_unknown(self, text: str, intent: IntentResult) -> str:
        """Handle unknown intent"""
        text_lower = text.lower().strip()
        
        # Handle follow-up questions like "only this?", "anything else?", "more options?"
        follow_up_phrases = ["only this", "that's it", "that all", "anything else", "more options", 
                            "what else", "other options", "is that all", "just these"]
        if any(phrase in text_lower for phrase in follow_up_phrases):
            context = self.context_manager.get_context_hints()
            if context.get('last_category'):
                # Show more from the same category
                return self._handle_search_menu(f"show me {context['last_category']}", intent)
            else:
                # Show menu overview
                categories = set(item.category for item in self.config.menu_items)
                cat_list = ", ".join(categories)
                return f"We have {cat_list}. What category would you like to see?"
        
        # Check if it matches a tag from the menu vocabulary (vegetarian, breakfast, etc.)
        matched_tag = self.vocab_matcher.match_tag(text_lower)
        if matched_tag:
            # Treat as search menu with tag filter
            return self._handle_search_menu(text, intent)
        
        # Check if it matches a category from the menu vocabulary (dynamic, not hardcoded)
        matched_categories = self.vocab_matcher.match_category(text_lower)
        if matched_categories:
            # Treat as search menu
            return self._handle_search_menu(text, intent)
        
        # Check if it's a request for items with filters (without/no)
        if "without" in text_lower or text_lower.startswith("no "):
            # Treat as search menu
            return self._handle_search_menu(text, intent)
        
        # Try to match as an add to cart (single item name)
        order = self.menu_matcher.parse_order(text)
        if order.items:
            return self._handle_add_to_cart(text, intent)
        
        # Try fuzzy search to suggest alternatives
        results = self.menu_matcher.search_menu(text, limit=3)
        if results:
            suggestions = ", ".join([r[0].name for r in results])
            return f"Did you mean: {suggestions}? Or say 'help' to see all commands."
        
        return "I'm not sure what you mean. Say 'help' to see the menu or 'menu' to see what we have."
    
    def process_input(self, text: str) -> str:
        """
        Process user input and return response
        
        Args:
            text: User input text
            
        Returns:
            Response message
        """
        if not text.strip():
            return ""
        
        # Check if we're waiting for an option selection
        if self.context_manager.has_pending_options():
            return self._handle_option_selection(text)
        
        # Check if we're waiting for item confirmation
        context_hints = self.context_manager.get_context_hints()
        if context_hints['awaiting_response'] == 'item_confirmation':
            return self._handle_pending_item_confirmation(text)
        
        # Classify intent
        intent = self.intent_classifier.classify(text)
        
        # Enhance with context if needed
        if self.context_manager.should_use_context_for_intent(text):
            enhanced_intent, enhanced_confidence = self.context_manager.enhance_intent_with_context(
                text, intent.intent, intent.confidence
            )
            if enhanced_intent != intent.intent:
                print(f"  [Context enhanced: {intent.intent} → {enhanced_intent}]")
                intent.intent = enhanced_intent
                intent.confidence = enhanced_confidence
        
        # Log for debugging
        print(f"  [Intent: {intent.intent} ({intent.confidence:.2f})]")
        
        # Get context hints
        context_hints = self.context_manager.get_context_hints()
        if context_hints['awaiting_response']:
            print(f"  [Context: awaiting {context_hints['awaiting_response']}]")
        
        # Process intent and get response
        response = self.process_intent(text, intent)
        
        # Update context
        self.context_manager.update_context(text, intent.intent, response)
        
        return response
    
    def run(self):
        """Run the assistant in interactive mode"""
        self.running = True
        
        print("\n" + "="*60)
        print(f"  {self.config.tenant_name} - Voice Assistant")
        print("="*60)
        print(f"Wake phrase: '{self.config.wake_word_phrase}' (optional)")
        print("Type 'quit' or 'exit' to stop")
        print("Type 'help' for commands")
        print("="*60 + "\n")
        
        self.respond(f"Welcome to {self.config.tenant_name}! How can I help you?")
        
        while self.running:
            try:
                # Get input
                result = self.input_engine.listen_and_transcribe()
                text = result.text.strip()
                
                if not text:
                    continue
                
                # Check for exit
                if text.lower() in ["quit", "exit", "q"]:
                    self.respond("Goodbye!")
                    self.running = False
                    break
                
                # Remove wake phrase if present
                wake_lower = self.config.wake_word_phrase.lower()
                if text.lower().startswith(wake_lower):
                    text = text[len(wake_lower):].strip()
                
                # Process and respond
                response = self.process_input(text)
                if response:
                    self.respond(response)
                    
            except KeyboardInterrupt:
                print("\n")
                self.respond("Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.respond("Sorry, something went wrong. Please try again.")
    
    def close(self):
        """Cleanup resources"""
        self.running = False
        print("Assistant closed.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-based POS Assistant")
    parser.add_argument(
        "--tenant",
        default="demo_restaurant",
        help="Tenant ID to load"
    )
    parser.add_argument(
        "--tenants-path",
        default="data/tenants",
        help="Path to tenant configurations"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(Path(__file__).parent.parent)
    
    # Create and run assistant
    assistant = TextAssistant(
        tenant_id=args.tenant,
        tenants_path=args.tenants_path
    )
    
    try:
        assistant.run()
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
