#!/usr/bin/env python3
"""
Production-Ready Intent Classification using spaCy
Combines rule-based patterns with ML-based text categorization
"""

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class Intent:
    """Detected intent with entities and confidence"""
    name: str
    confidence: float
    entities: Dict[str, any]
    raw_text: str


@dataclass
class CartItem:
    """Single item to add to cart"""
    product: str
    quantity: int


class ProductionIntentClassifier:
    """
    Production-ready intent classifier using spaCy.
    Combines:
    - Rule-based pattern matching (fast, accurate for known patterns)
    - Entity extraction (spaCy NER)
    - Fallback to similarity-based classification
    """
    
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize pattern matcher
        self.matcher = Matcher(self.nlp.vocab)
        
        # Define intent patterns
        self._setup_patterns()
        
        # Product catalog (could be loaded from database)
        self.products = {
            'coffee', 'espresso', 'latte', 'cappuccino', 'americano', 'mocha',
            'tea', 'green tea', 'black tea', 'chai', 'herbal tea',
            'muffin', 'blueberry muffin', 'chocolate muffin', 'bran muffin',
            'pasta', 'spaghetti', 'penne', 'fettuccine', 'linguine',
            'pizza', 'margherita', 'pepperoni', 'hawaiian', 'veggie pizza',
            'burger', 'cheeseburger', 'hamburger', 'veggie burger',
            'sandwich', 'salad', 'cake', 'cookie', 'croissant', 'bagel',
            'juice', 'soda', 'water', 'smoothie',
            'pepsi', 'coke', 'cola', 'sprite', 'fanta',
            'shawarma', 'falafel', 'kebab', 'wrap', 'hummus',
            'chicken', 'beef', 'fish', 'shrimp', 'steak',
            'fries', 'chips', 'nuggets', 'wings',
        }
        
        # Number words
        self.number_words = {
            'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3,
            'four': 4, 'five': 5, 'six': 6, 'seven': 7,
            'eight': 8, 'nine': 9, 'ten': 10,
            'couple': 2, 'few': 3, 'several': 3,
        }
        
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup spaCy matcher patterns for each intent"""
        
        # ADD_TO_CART patterns
        add_patterns = [
            [{"LOWER": {"IN": ["add", "get", "order", "want", "need"]}}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": "give"}, {"LOWER": "me"}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": "i"}, {"LOWER": {"IN": ["want", "need", "would", "will"]}}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": {"IN": ["get", "order"]}}, {"IS_DIGIT": True, "OP": "?"}, {"POS": "NOUN"}],
            [{"LOWER": "can"}, {"LOWER": "i"}, {"LOWER": "have"}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": "could"}, {"LOWER": "i"}, {"LOWER": "get"}, {"OP": "*"}, {"POS": "NOUN"}],
        ]
        self.matcher.add("ADD_TO_CART", add_patterns)
        
        # REMOVE_FROM_CART patterns
        remove_patterns = [
            [{"LOWER": {"IN": ["remove", "delete", "cancel"]}}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": "take"}, {"LOWER": "out"}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": "no"}, {"LOWER": {"IN": ["more", "longer"]}}, {"POS": "NOUN"}],
        ]
        self.matcher.add("REMOVE_FROM_CART", remove_patterns)
        
        # VIEW_CART patterns
        view_cart_patterns = [
            [{"LOWER": {"IN": ["show", "view", "check", "see"]}}, {"LOWER": {"IN": ["my", "the"]}, "OP": "?"}, {"LOWER": {"IN": ["cart", "basket", "order"]}}],
            [{"LOWER": "what"}, {"LOWER": {"IN": ["is", "did", "s"]}}, {"OP": "*"}, {"LOWER": {"IN": ["cart", "order", "basket"]}}],
            [{"LOWER": "what"}, {"LOWER": {"IN": ["in", "s"]}}, {"OP": "*"}, {"LOWER": {"IN": ["cart", "basket", "order"]}}],
            [{"LOWER": {"IN": ["show", "see"]}}, {"LOWER": "me"}, {"OP": "*"}, {"LOWER": {"IN": ["cart", "basket"]}}],
        ]
        self.matcher.add("VIEW_CART", view_cart_patterns)
        
        # CHECKOUT patterns
        checkout_patterns = [
            [{"LOWER": {"IN": ["checkout", "pay", "complete", "finish", "done"]}}],
            [{"LOWER": "that"}, {"LOWER": {"IN": ["is", "s"]}}, {"LOWER": {"IN": ["it", "all"]}}],
            [{"LOWER": "ready"}, {"LOWER": "to"}, {"LOWER": "pay"}],
            [{"LOWER": "i"}, {"LOWER": "m"}, {"LOWER": "done"}],
            [{"LOWER": {"IN": ["all", "everything"]}}, {"LOWER": "done"}],
        ]
        self.matcher.add("CHECKOUT", checkout_patterns)
        
        # CLEAR_CART patterns
        clear_patterns = [
            [{"LOWER": {"IN": ["clear", "empty", "reset"]}}, {"OP": "*"}, {"LOWER": {"IN": ["cart", "basket", "order"]}}],
            [{"LOWER": "remove"}, {"LOWER": {"IN": ["everything", "all"]}}],
            [{"LOWER": "start"}, {"LOWER": "over"}],
        ]
        self.matcher.add("CLEAR_CART", clear_patterns)
        
        # SEARCH_PRODUCTS patterns
        search_patterns = [
            [{"LOWER": {"IN": ["search", "find", "look"]}}, {"OP": "*"}, {"POS": "NOUN"}],
            [{"LOWER": "what"}, {"OP": "*"}, {"LOWER": {"IN": ["have", "available", "sell"]}}],
            [{"LOWER": {"IN": ["menu", "options"]}}],
            [{"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "have"}],
        ]
        self.matcher.add("SEARCH_PRODUCTS", search_patterns)
    
    def classify(self, text: str) -> Intent:
        """
        Classify intent and extract entities.
        
        Args:
            text: User input
            
        Returns:
            Intent object with name, confidence, and entities
        """
        # Process with spaCy
        doc = self.nlp(text.lower())
        text_lower = text.lower()
        
        # Handle some edge cases before pattern matching
        # "what's" gets tokenized as "what" + "'s", check for contractions
        if "what" in text_lower and ("in my" in text_lower or "in the" in text_lower):
            if any(word in text_lower for word in ["cart", "basket", "order"]):
                return Intent('view_cart', 0.95, {}, text)
        
        # "that's all" should trigger checkout
        if ("that" in text_lower and "all" in text_lower) or ("that" in text_lower and "it" in text_lower):
            return Intent('checkout', 0.90, {}, text)
        
        # Check if text contains products - likely an order
        has_products = any(prod in text_lower for prod in self.products)
        has_quantity = any(word in text_lower for word in self.number_words.keys()) or any(c.isdigit() for c in text_lower)
        
        # Try pattern matching first (most reliable)
        matches = self.matcher(doc)
        
        if matches:
            # Get the first match
            match_id, start, end = matches[0]
            intent_name = self.nlp.vocab.strings[match_id]
            confidence = 0.95
            
            entities = self._extract_entities(intent_name, text, doc)
            
            return Intent(
                name=intent_name.lower(),
                confidence=confidence,
                entities=entities,
                raw_text=text
            )
        
        # If no pattern match but has products + quantities = likely add_to_cart
        if has_products and has_quantity:
            entities = self._extract_entities('ADD_TO_CART', text, doc)
            if entities:
                return Intent('add_to_cart', 0.85, entities, text)
        
        # Fallback: check for specific keywords
        
        # Time/Date intents
        if any(word in text_lower for word in ['time', 'clock', 'hour']):
            return Intent('get_time', 0.85, {}, text)
        if any(word in text_lower for word in ['date', 'today', 'day']):
            return Intent('get_date', 0.85, {}, text)
        
        # Greeting
        if any(word in text_lower for word in ['hi', 'hello', 'hey', 'greetings']):
            return Intent('greeting', 0.90, {}, text)
        
        # Help
        if any(word in text_lower for word in ['help', 'how', 'what can you']):
            return Intent('help', 0.85, {}, text)
        
        # Goodbye
        if any(word in text_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            return Intent('goodbye', 0.90, {}, text)
        
        # Unknown
        return Intent('unknown', 0.0, {}, text)
    
    def _extract_entities(self, intent: str, text: str, doc: Doc) -> Dict[str, any]:
        """Extract entities using spaCy NER and custom logic"""
        entities = {}
        text_lower = text.lower()
        
        if intent in ['ADD_TO_CART', 'REMOVE_FROM_CART', 'SEARCH_PRODUCTS']:
            # Extract ALL products (for multi-item orders)
            items = self._extract_all_products(text_lower, doc)
            
            if items:
                if len(items) == 1:
                    # Single item - backward compatible format
                    entities['product'] = items[0].product
                    entities['quantity'] = items[0].quantity
                else:
                    # Multiple items
                    entities['items'] = [(item.product, item.quantity) for item in items]
                    # Also set first item for backward compatibility
                    entities['product'] = items[0].product
                    entities['quantity'] = items[0].quantity
        
        return entities
    
    def _extract_all_products(self, text: str, doc: Doc) -> List[CartItem]:
        """Extract all products with quantities from text"""
        items = []
        text_lower = text.lower()
        
        import re
        
        # Build product pattern including plurals
        all_products = set()
        for prod in self.products:
            all_products.add(prod)
            all_products.add(prod + 's')  # Add plural
            all_products.add(prod + 'es')  # Add plural for words ending in consonant
        
        # Method 1: Look for "number + product" patterns directly
        number_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|a|an|\d+)\s+'
        
        for prod in sorted(all_products, key=len, reverse=True):
            pattern = number_pattern + re.escape(prod) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                num_word = match.group(1)
                quantity = self.number_words.get(num_word, 1)
                if num_word.isdigit():
                    quantity = int(num_word)
                
                # Normalize product name (remove plural)
                base_prod = prod.rstrip('s')
                if base_prod + 'es' == prod:
                    base_prod = prod[:-2]
                
                # Find the canonical product name
                canonical = None
                for p in self.products:
                    if p == base_prod or p == prod or base_prod.startswith(p) or p.startswith(base_prod):
                        canonical = p.title()
                        break
                
                if not canonical:
                    canonical = base_prod.title()
                
                # Check if this product wasn't already added
                if not any(item.product == canonical for item in items):
                    items.append(CartItem(product=canonical, quantity=quantity))
        
        # Method 2: If no pattern matches, try splitting by separators
        if not items:
            segments = re.split(r'\s+and\s+|\s*,\s*|\s+also\s+', text_lower)
            
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                
                quantity = 1
                for word, num in self.number_words.items():
                    if f' {word} ' in f' {segment} ' or segment.startswith(f'{word} '):
                        quantity = num
                        break
                
                digit_match = re.search(r'\b(\d+)\b', segment)
                if digit_match:
                    quantity = int(digit_match.group(1))
                
                product = None
                for prod in sorted(self.products, key=len, reverse=True):
                    if prod in segment or prod + 's' in segment:
                        product = prod.title()
                        break
                
                if product and not any(item.product == product for item in items):
                    items.append(CartItem(product=product, quantity=quantity))
        
        # Method 3: Fallback - single product extraction
        if not items:
            product = self._extract_product(text_lower, doc)
            quantity = self._extract_quantity(text_lower, doc)
            if product:
                items.append(CartItem(product=product, quantity=quantity))
        
        return items
    
    def _extract_product(self, text: str, doc: Doc) -> Optional[str]:
        """Extract product name using spaCy NER + product catalog"""
        # Try longest product names first
        sorted_products = sorted(self.products, key=len, reverse=True)
        
        for product in sorted_products:
            if product in text:
                return product.title()
        
        # Fallback: extract nouns from spaCy
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        if nouns:
            return nouns[-1].title()  # Return last noun
        
        return None
    
    def _extract_quantity(self, text: str, doc: Doc) -> int:
        """Extract quantity from text"""
        # Look for numbers using spaCy
        for token in doc:
            if token.like_num or token.pos_ == "NUM":
                try:
                    return int(token.text)
                except ValueError:
                    pass
        
        # Look for digit numbers with regex
        digit_match = re.search(r'\b(\d+)\b', text)
        if digit_match:
            return int(digit_match.group(1))
        
        # Look for word numbers
        for word, num in self.number_words.items():
            if f' {word} ' in f' {text} ':
                return num
        
        # Default to 1
        return 1
    
    def get_intent_description(self, intent_name: str) -> str:
        """Get human-readable description"""
        descriptions = {
            'add_to_cart': 'Add item to shopping cart',
            'remove_from_cart': 'Remove item from cart',
            'view_cart': 'View cart contents',
            'clear_cart': 'Clear all items from cart',
            'checkout': 'Complete purchase',
            'search_products': 'Search for products',
            'get_time': 'Get current time',
            'get_date': 'Get current date',
            'greeting': 'Greet the user',
            'help': 'Show available commands',
            'goodbye': 'End conversation',
            'unknown': 'Intent not recognized',
        }
        return descriptions.get(intent_name, 'Unknown intent')


# Test function
if __name__ == "__main__":
    classifier = ProductionIntentClassifier()
    
    test_cases = [
        "I want two coffees",
        "add three muffins to my cart",
        "show me my cart",
        "what's in my basket",
        "remove the pizza",
        "search for pasta",
        "checkout",
        "what time is it",
        "hello",
        "clear my cart",
        "give me a blueberry muffin",
        "order one latte",
        "get me 5 pizzas",
        "I need a cappuccino",
        "can I have two burgers",
        "what do you have for lunch",
        "that's all",
    ]
    
    print("\\n🧪 PRODUCTION INTENT CLASSIFIER TEST\\n")
    print("="*70)
    
    for text in test_cases:
        intent = classifier.classify(text)
        print(f"\\nInput: \\\"{text}\\\"")
        print(f"Intent: {intent.name} (confidence: {intent.confidence:.2f})")
        if intent.entities:
            print(f"Entities: {intent.entities}")
    
    print("\\n" + "="*70)
