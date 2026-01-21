"""JSON Menu Provider - Load menu from structured JSON data"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from rapidfuzz import fuzz, process

from .menu_provider import MenuProvider, MenuItem


class JSONMenuProvider(MenuProvider):
    """
    Load menu from JSON file (client format)
    
    Expected JSON structure:
    {
        "success": true,
        "result": [
            {
                "id": 563,
                "name": "Coca",
                "category": "Iced & Cold Drinks",
                "price": 1.5,
                "active": 1,
                "images": ["url..."],
                ...
            }
        ]
    }
    """
    
    def __init__(self, json_path: str, semantic_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize JSON menu provider
        
        Args:
            json_path: Path to JSON file
            semantic_categories: Optional manual semantic category mapping
        """
        self.json_path = json_path
        self._items: List[MenuItem] = []
        self._category_index: Dict[str, List[MenuItem]] = {}
        self._id_index: Dict[str, MenuItem] = {}
        self._semantic_categories = semantic_categories
        
        self._load_menu()
        self._build_indexes()
    
    def _load_menu(self):
        """Load menu from JSON file"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both {"result": [...]} and [...] formats
        items_data = data.get('result', data) if isinstance(data, dict) else data
        
        for item_data in items_data:
            # Skip inactive items
            if not item_data.get('active', True):
                continue
            
            item = MenuItem(
                id=str(item_data['id']),
                name=item_data['name'],
                category=item_data.get('category', 'Uncategorized'),
                price=float(item_data.get('price', 0)),
                description=item_data.get('description'),
                images=item_data.get('images', []),
                active=bool(item_data.get('active', True)),
                metadata=item_data
            )
            self._items.append(item)
    
    def _build_indexes(self):
        """Build search indexes"""
        for item in self._items:
            # Category index
            cat = item.category
            if cat not in self._category_index:
                self._category_index[cat] = []
            self._category_index[cat].append(item)
            
            # ID index
            self._id_index[item.id] = item
    
    def get_all_items(self) -> List[MenuItem]:
        """Get all active menu items"""
        return self._items.copy()
    
    def get_categories(self) -> List[str]:
        """Get all category names"""
        return sorted(self._category_index.keys())
    
    def get_items_by_category(self, category: str) -> List[MenuItem]:
        """Get items in a specific category"""
        return self._category_index.get(category, []).copy()
    
    def get_item_by_id(self, item_id: str) -> Optional[MenuItem]:
        """Get item by ID"""
        return self._id_index.get(item_id)
    
    def search_items(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[MenuItem]:
        """
        Search items by name using fuzzy matching
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Max results
            
        Returns:
            List of (MenuItem, score) tuples sorted by relevance
        """
        # Filter by category if specified
        if category:
            search_pool = self.get_items_by_category(category)
        else:
            search_pool = self._items
        
        if not search_pool:
            return []
        
        # Build search terms
        item_map = {}
        for item in search_pool:
            for term in item.get_search_terms():
                item_map[term] = item
        
        # Fuzzy search
        results = process.extract(
            query.lower(),
            item_map.keys(),
            scorer=fuzz.token_sort_ratio,
            limit=limit
        )
        
        # Deduplicate by item ID and return
        seen = set()
        matches = []
        for term, score, _ in results:
            item = item_map[term]
            if item.id not in seen:
                seen.add(item.id)
                matches.append((item, score))
        
        return matches
    
    def get_semantic_category_map(self) -> Dict[str, List[str]]:
        """Get semantic category mapping"""
        if self._semantic_categories:
            return self._semantic_categories
        return self._infer_semantic_categories()


def load_menu_from_json(json_path: str, semantic_categories: Optional[Dict] = None) -> JSONMenuProvider:
    """
    Convenience function to load menu from JSON
    
    Args:
        json_path: Path to JSON menu file
        semantic_categories: Optional semantic category overrides
        
    Returns:
        JSONMenuProvider instance
    """
    return JSONMenuProvider(json_path, semantic_categories)
