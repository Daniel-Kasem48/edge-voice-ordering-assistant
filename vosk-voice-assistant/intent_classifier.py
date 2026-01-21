#!/usr/bin/env python3
"""
Intent Classification and Entity Extraction for Voice Assistant
Uses pattern matching + regex for fast, offline intent detection
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Intent:
    """Detected intent with entities"""
    name: str
    confidence: float
    entities: Dict[str, any]
    raw_text: str


class IntentClassifier:
    """
    Fast pattern-based intent classifier for e-commerce voice commands.
    No ML models needed - uses regex patterns and keyword matching.
    """
    
    def __init__(self):
        # Define intent patterns (intent_name: list of regex patterns)
        self.intent_patterns = {
            # Shopping cart operations
            'add_to_cart': [
                r'\b(add|put|get|order|i want|i need|give me)\b.*\b(coffee|tea|muffin|pasta|pizza|burger|sandwich|salad|latte|cappuccino|espresso)',
                r'\b(get|order)\s+(?:me\s+)?(\d+|a|an|one|two|three|four|five)?\s*(coffee|tea|muffin|pasta|pizza|latte)',
                r'\b(can i (have|get)|i\'d like)\b.*\b(coffee|tea|muffin|pasta|pizza|latte)',
            ],
            'remove_from_cart': [
                r'\b(remove|delete|take out|cancel)\b.*\b(coffee|tea|muffin|pasta|pizza|burger)',
                r'\b(don\'t want|no more)\b.*\b(coffee|tea|muffin|pasta)',
            ],
            'view_cart': [
                r'\b(show|view|what\'s in|check|see)\s+(my\s+)?(cart|basket|order)',
                r'\b(show|what\'s in)\s+my\s+(cart|basket)',
                r'\bwhat did i order\b',
                r'\bwhat (do|did) i (get|order|have)',
                r'\b(my cart|my basket)\b',
            ],
            'clear_cart': [
                r'\b(clear|empty|remove everything|cancel all|start over)\b',
                r'\b(cancel|clear)\s+(my\s+)?(cart|order)',
            ],
            'checkout': [
                r'\b(checkout|check out|complete|finish|pay|done|that\'s (it|all))\b',
                r'\b(i\'m done|ready to pay|proceed)\b',
            ],
            'search_products': [
                r'\b(search|find|look for|show me|what do you have)\b.*\b(coffee|tea|food|drink|pasta|pizza)',
                r'\b(do you (have|sell)|what (coffee|tea|food|drinks))\b',
                r'\b(menu|options|what\'s available)\b',
            ],
            # Informational
            'get_time': [
                r'\b(what time|current time|time is it)\b',
            ],
            'get_date': [
                r'\b(what (day|date)|today\'s date|date is it)\b',
            ],
            'greeting': [
                r'^\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
            ],
            'help': [
                r'\b(help|what can you do|commands|how does this work)\b',
            ],
            'goodbye': [
                r'^\b(bye|goodbye|see you|exit|quit)\b',
            ],
        }
        
        # Product/item keywords for entity extraction
        self.products = [
            'coffee', 'espresso', 'latte', 'cappuccino', 'americano',
            'tea', 'green tea', 'black tea', 'chai',
            'muffin', 'blueberry muffin', 'chocolate muffin',
            'pasta', 'spaghetti', 'penne', 'fettuccine',
            'pizza', 'margherita', 'pepperoni',
            'burger', 'cheeseburger', 'hamburger',
            'sandwich', 'salad', 'cake', 'cookie',
        ]
        
        # Number word to digit mapping
        self.number_words = {
            'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 
            'four': 4, 'five': 5, 'six': 6, 'seven': 7, 
            'eight': 8, 'nine': 9, 'ten': 10,
        }
    
    def classify(self, text: str) -> Intent:
        """
        Classify user input and extract entities.
        
        Args:
            text: User's spoken/typed input
            
        Returns:
            Intent object with name, confidence, and entities
        """
        text_lower = text.lower().strip()
        
        # Match against patterns
        best_intent = None
        best_confidence = 0.0
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = 0.9  # High confidence for pattern match
                    if confidence > best_confidence:
                        best_intent = intent_name
                        best_confidence = confidence
                        break
        
        # Default to unknown if no match
        if best_intent is None:
            return Intent(
                name='unknown',
                confidence=0.0,
                entities={},
                raw_text=text
            )
        
        # Extract entities based on intent
        entities = self._extract_entities(best_intent, text_lower)
        
        return Intent(
            name=best_intent,
            confidence=best_confidence,
            entities=entities,
            raw_text=text
        )
    
    def _extract_entities(self, intent: str, text: str) -> Dict[str, any]:
        """Extract entities from text based on intent"""
        entities = {}
        
        if intent in ['add_to_cart', 'remove_from_cart', 'search_products']:
            # Extract product
            product = self._extract_product(text)
            if product:
                entities['product'] = product
            
            # Extract quantity (for add_to_cart)
            if intent == 'add_to_cart':
                quantity = self._extract_quantity(text)
                entities['quantity'] = quantity
        
        return entities
    
    def _extract_product(self, text: str) -> Optional[str]:
        """Extract product name from text"""
        # Try longer product names first (e.g., "blueberry muffin" before "muffin")
        sorted_products = sorted(self.products, key=len, reverse=True)
        
        for product in sorted_products:
            if product in text:
                return product.title()  # Capitalize for display
        
        return None
    
    def _extract_quantity(self, text: str) -> int:
        """Extract quantity from text"""
        # Look for digit numbers
        digit_match = re.search(r'\b(\d+)\s*(coffee|tea|muffin|pasta|pizza|burger|sandwich)', text)
        if digit_match:
            return int(digit_match.group(1))
        
        # Look for word numbers
        for word, num in self.number_words.items():
            pattern = rf'\b{word}\b'
            if re.search(pattern, text):
                return num
        
        # Default to 1
        return 1
    
    def get_intent_description(self, intent_name: str) -> str:
        """Get human-readable description of an intent"""
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


# Demo/test function
if __name__ == "__main__":
    classifier = IntentClassifier()
    
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
    ]
    
    print("🧪 INTENT CLASSIFIER TEST\n")
    print("="*70)
    
    for text in test_cases:
        intent = classifier.classify(text)
        print(f"\nInput: \"{text}\"")
        print(f"Intent: {intent.name} (confidence: {intent.confidence:.2f})")
        if intent.entities:
            print(f"Entities: {intent.entities}")
        print(f"Description: {classifier.get_intent_description(intent.name)}")
    
    print("\n" + "="*70)
