"""
Dynamic Menu Matcher - Works with any MenuProvider
Fuzzy matching for menu items using provider interface
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from rapidfuzz import fuzz, process

from menu.menu_provider import MenuProvider, MenuItem


@dataclass
class MatchedItem:
    """A matched menu item with quantity"""
    item: MenuItem
    quantity: int = 1
    match_score: float = 100.0
    
    def get_total_price(self) -> float:
        """Calculate total price"""
        return self.item.price * self.quantity
    
    def get_description(self) -> str:
        """Get human-readable description"""
        if self.quantity > 1:
            return f"{self.quantity}x {self.item.name}"
        return self.item.name


@dataclass 
class ParsedOrder:
    """Parsed order from natural language"""
    items: List[MatchedItem] = field(default_factory=list)
    raw_text: str = ""
    
    def get_total(self) -> float:
        """Get total price of all items"""
        return sum(item.get_total_price() for item in self.items)


class DynamicMenuMatcher:
    """
    Dynamic menu matcher - works with any MenuProvider
    No hardcoded items or categories
    """
    
    # Quantity words
    QUANTITIES = {
        "a": 1, "an": 1, "one": 1,
        "two": 2, "couple": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    
    # Size/modifier words to strip during matching
    SIZE_WORDS = ["large", "small", "medium", "big", "regular"]
    
    # Noise words to filter out
    NOISE_WORDS = [
        "very", "really", "please", "thanks", "thank", "you",
        "i", "want", "need", "like", "would", "can", "could",
        "the", "a", "an", "some", "any",
        "add", "get", "give", "gimme", "gimmie", "order",
        "me", "my", "to", "for"
    ]
    
    def __init__(self, menu_provider: MenuProvider, fuzzy_threshold: int = 70):
        """
        Initialize matcher with menu provider
        
        Args:
            menu_provider: Dynamic menu data source
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.menu = menu_provider
        self.threshold = fuzzy_threshold
        self._build_indexes()
    
    def _build_indexes(self):
        """Build search indexes from menu"""
        self.all_items = self.menu.get_all_items()
        self.all_categories = self.menu.get_categories()
        
        # Build term → item mapping for fast lookup
        self.term_to_item = {}
        for item in self.all_items:
            for term in item.get_search_terms():
                self.term_to_item[term] = item
    
    def parse_order(self, text: str) -> ParsedOrder:
        """
        Parse natural language order text
        
        Args:
            text: User input like "I want 2 coffees"
            
        Returns:
            ParsedOrder with matched items
        """
        order = ParsedOrder(raw_text=text)
        text = text.lower().strip()
        
        # Extract quantity
        quantity = self._extract_quantity(text)
        
        # Clean text for matching
        clean_text = self._clean_text_for_matching(text)
        
        if not clean_text:
            return order
        
        # Try to match item
        matched_item = self.match_item(clean_text)
        if matched_item:
            item, score = matched_item
            order.items.append(MatchedItem(
                item=item,
                quantity=quantity,
                match_score=score
            ))
        
        return order
    
    def match_item(self, text: str) -> Optional[Tuple[MenuItem, float]]:
        """
        Match text to menu item using fuzzy search
        
        Returns:
            (MenuItem, score) or None
        """
        text = text.lower().strip()
        if not text:
            return None
        
        # Try exact match first
        if text in self.term_to_item:
            return (self.term_to_item[text], 100.0)
        
        # Check if it's a category name (generic term like "coffee", "pizza")
        # If so, return cheapest/most common item in that category
        # Clean text for category matching (handle plurals)
        clean_query = self._clean_text_for_matching(text)
        resolved_cats = self.menu.resolve_category_query(clean_query)
        if resolved_cats:
            # Get all items in these categories
            cat_items = []
            for cat in resolved_cats:
                cat_items.extend(self.menu.get_items_by_category(cat))
            
            # If user says generic "coffee", return cheapest coffee
            if cat_items:
                cheapest = min(cat_items, key=lambda x: x.price)
                return (cheapest, 85.0)  # Lower score to indicate generic match
        
        # Fuzzy search across all items
        results = self.menu.search_items(text, limit=1)
        if results and results[0][1] >= self.threshold:
            return results[0]
        
        return None
    
    def search_menu(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Tuple[MenuItem, float]]:
        """
        Search menu items
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Max results
            
        Returns:
            List of (MenuItem, score) tuples
        """
        # Resolve category semantically if needed
        if category:
            resolved_cats = self.menu.resolve_category_query(category)
            if not resolved_cats:
                resolved_cats = [category]
            
            # Search across all resolved categories
            all_results = []
            for cat in resolved_cats:
                results = self.menu.search_items(query, category=cat, limit=limit)
                all_results.extend(results)
            
            # Sort by score and deduplicate
            seen = set()
            unique_results = []
            for item, score in sorted(all_results, key=lambda x: x[1], reverse=True):
                if item.id not in seen:
                    seen.add(item.id)
                    unique_results.append((item, score))
                    if len(unique_results) >= limit:
                        break
            
            return unique_results
        else:
            return self.menu.search_items(query, limit=limit)
    
    def get_category_items(self, category_query: str) -> List[MenuItem]:
        """
        Get all items in a category (semantic resolution)
        
        Args:
            category_query: User's category query like "drinks" or "desserts"
            
        Returns:
            List of menu items
        """
        resolved_categories = self.menu.resolve_category_query(category_query)
        
        items = []
        for cat in resolved_categories:
            items.extend(self.menu.get_items_by_category(cat))
        
        return items
    
    def _extract_quantity(self, text: str) -> int:
        """Extract quantity from text"""
        text_lower = text.lower()
        
        # Check for numbers
        for word, qty in self.QUANTITIES.items():
            if word in text_lower.split():
                return qty
        
        # Check for digits
        import re
        digit_match = re.search(r'\b(\d+)\b', text)
        if digit_match:
            return int(digit_match.group(1))
        
        return 1
    
    def _clean_text_for_matching(self, text: str) -> str:
        """Clean text for better matching"""
        text = text.lower()
        
        # Remove quantity words
        for word in self.QUANTITIES.keys():
            text = re.sub(rf'\b{word}\b', '', text)
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Remove size words
        for word in self.SIZE_WORDS:
            text = re.sub(rf'\b{word}\b', '', text)
        
        # Remove noise words
        for word in self.NOISE_WORDS:
            text = re.sub(rf'\b{word}\b', '', text)
        
        # Simple pluralization fix (coffees → coffee)
        text = re.sub(r'(\w+)s\b', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
