"""
Menu Provider Interface - Abstract menu layer for any restaurant
Supports JSON, YAML, API, Database menu sources
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Import Option type
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.tenant_loader import Option


@dataclass
class MenuItem:
    """Universal menu item representation"""
    id: str
    name: str
    category: str
    price: float
    description: Optional[str] = None
    aliases: List[str] = None
    modifiers: List[str] = None
    toppings: List[str] = None
    images: List[str] = None
    active: bool = True
    options: List[Option] = None
    tags: List[Dict[str, Any]] = None  # Tags for dietary, allergen, etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.modifiers is None:
            self.modifiers = []
        if self.toppings is None:
            self.toppings = []
        if self.images is None:
            self.images = []
        if self.options is None:
            self.options = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def get_search_terms(self) -> List[str]:
        """Get all terms that can match this item"""
        terms = [self.name.lower()]
        terms.extend([alias.lower() for alias in self.aliases])
        return terms


class MenuProvider(ABC):
    """Abstract base class for menu data providers"""
    
    @abstractmethod
    def get_all_items(self) -> List[MenuItem]:
        """Get all menu items"""
        pass
    
    @abstractmethod
    def get_categories(self) -> List[str]:
        """Get all category names"""
        pass
    
    @abstractmethod
    def get_items_by_category(self, category: str) -> List[MenuItem]:
        """Get items in a specific category"""
        pass
    
    @abstractmethod
    def search_items(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[MenuItem]:
        """Search items by name/alias"""
        pass
    
    @abstractmethod
    def get_item_by_id(self, item_id: str) -> Optional[MenuItem]:
        """Get item by ID"""
        pass
    
    def get_semantic_category_map(self) -> Dict[str, List[str]]:
        """
        Map common terms to actual categories
        Override this or provide in config
        """
        return self._infer_semantic_categories()
    
    def _infer_semantic_categories(self) -> Dict[str, List[str]]:
        """Auto-infer semantic category groupings"""
        categories = self.get_categories()
        semantic_map = {}
        
        # Drink-related
        drink_keywords = ['drink', 'coffee', 'tea', 'juice', 'soda', 'water', 'refresher', 'cocktail', 'beer', 'wine', 'alcohol']
        semantic_map['drinks'] = [cat for cat in categories if any(kw in cat.lower() for kw in drink_keywords)]
        
        # Food-related
        food_keywords = ['food', 'dish', 'entree', 'main', 'signature', 'special', 'barbeque', 'bbq', 'grill']
        semantic_map['food'] = [cat for cat in categories if any(kw in cat.lower() for kw in food_keywords)]
        
        # Dessert-related
        dessert_keywords = ['dessert', 'sweet', 'cake', 'cupcake', 'pastry', 'cookie']
        semantic_map['desserts'] = [cat for cat in categories if any(kw in cat.lower() for kw in dessert_keywords)]
        
        # Appetizer-related
        appetizer_keywords = ['appetizer', 'starter', 'snack', 'cracker']
        semantic_map['appetizers'] = [cat for cat in categories if any(kw in cat.lower() for kw in appetizer_keywords)]
        
        return semantic_map
    
    def resolve_category_query(self, query: str) -> List[str]:
        """
        Resolve user's category query to actual category names
        e.g., "drinks" → ["Iced & Cold Drinks", "Espresso & Coffee"]
        """
        query_lower = query.lower().strip()
        
        # Try semantic mapping first
        semantic_map = self.get_semantic_category_map()
        if query_lower in semantic_map:
            return semantic_map[query_lower]
        
        # Try exact category match
        categories = self.get_categories()
        for cat in categories:
            if cat.lower() == query_lower:
                return [cat]
        
        # Try partial match (query is substring of category)
        matches = [cat for cat in categories if query_lower in cat.lower()]
        if matches:
            return matches
        
        # Try fuzzy category name matching
        from rapidfuzz import fuzz, process
        results = process.extract(query_lower, categories, scorer=fuzz.token_sort_ratio, limit=3)
        good_matches = [cat for cat, score, _ in results if score >= 70]
        if good_matches:
            return good_matches
        
        return []
