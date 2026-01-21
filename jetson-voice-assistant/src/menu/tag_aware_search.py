"""
Tag-Aware Menu Search
Enhances menu search with dietary preferences and allergen filtering
"""

from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re

from .menu_provider import MenuProvider, MenuItem


class TagType(Enum):
    """Types of tags for categorization"""
    DIETARY = "dietary"          # Vegan, Vegetarian, Keto, etc.
    ALLERGEN = "allergen"         # Contains Gluten, Dairy, etc.
    NUTRITIONAL = "nutritional"   # High Protein, Low Calorie, etc.
    PREPARATION = "preparation"   # Fried, Grilled, etc.
    STYLE = "style"              # Comfort Food, Bar Food, etc.
    QUALITY = "quality"          # Premium, Natural, Organic, etc.
    FLAVOR = "flavor"            # Spicy, Mild, etc.
    SERVING = "serving"          # Sharing Plate, Individual, etc.
    OTHER = "other"


@dataclass
class TagFilter:
    """Filter criteria for tags"""
    include_tags: Set[str] = None  # Tags that items MUST have
    exclude_tags: Set[str] = None  # Tags that items must NOT have
    
    def __post_init__(self):
        if self.include_tags is None:
            self.include_tags = set()
        if self.exclude_tags is None:
            self.exclude_tags = set()


class TagAwareMenuSearch:
    """Enhanced menu search with tag filtering"""
    
    # Common search patterns mapped to tags
    TAG_PATTERNS = {
        # Dietary preferences
        r"\b(vegan|plant[\s-]?based)\b": ["Vegan"],
        r"\bvegetarian\b": ["Vegetarian"],
        r"\b(gluten[\s-]?free|no[\s-]?gluten|celiac)\b": ["Gluten-Free"],
        r"\b(dairy[\s-]?free|lactose[\s-]?free|no[\s-]?dairy)\b": ["Dairy-Free"],
        r"\b(sugar[\s-]?free|no[\s-]?sugar|diabetic)\b": ["Sugar-Free"],
        r"\b(keto|ketogenic|low[\s-]?carb)\b": ["Keto-Friendly"],
        r"\b(nut[\s-]?free|no[\s-]?nuts?)\b": ["Nut-Free"],
        r"\bhalal\b": ["Halal"],
        r"\bkosher\b": ["Kosher"],
        
        # Allergen exclusions (note the negation)
        r"\b(no[\s-]?eggs?|egg[\s-]?free)\b": {"exclude": ["Contains Eggs"]},
        r"\b(no[\s-]?meat|meatless)\b": {"exclude": ["Contains Meat"]},
        r"\b(no[\s-]?seafood)\b": {"exclude": ["Contains Seafood", "Contains Shellfish"]},
        r"\b(no[\s-]?shellfish)\b": {"exclude": ["Contains Shellfish"]},
        
        # Nutritional preferences
        r"\b(high[\s-]?protein|protein[\s-]?rich)\b": ["High Protein"],
        r"\b(low[\s-]?calorie|diet|light)\b": ["Low Calorie"],
        r"\bhealthy\b": ["Low Calorie", "Natural"],
        
        # Preparation methods
        r"\b(fried|crispy)\b": ["Fried"],
        r"\b(grilled|bbq|barbecue)\b": ["Grilled"],
        r"\b(fresh|raw)\b": ["Natural"],
        
        # Style preferences
        r"\b(comfort[\s-]?food|hearty)\b": ["Comfort Food"],
        r"\b(premium|gourmet|special)\b": ["Premium"],
        r"\bspicy\b": ["Spicy"],
        r"\b(sharing|shareable|platter)\b": ["Sharing Plate"],
        
        # Temperature
        r"\b(hot|warm)\b": ["Hot Beverage"],
        r"\b(cold|iced|chilled)\b": ["Cold Beverage"]
    }
    
    def __init__(self, menu_provider: MenuProvider):
        self.menu_provider = menu_provider
        self._tag_cache = self._build_tag_cache()
    
    def _build_tag_cache(self) -> Dict[str, Set[str]]:
        """Build cache of item IDs by tag name"""
        cache = {}
        items = self.menu_provider.get_all_items()
        
        for item in items:
            if hasattr(item, 'tags') and item.tags:
                for tag in item.tags:
                    tag_name = tag.get('name', '') if isinstance(tag, dict) else str(tag)
                    if tag_name:
                        if tag_name not in cache:
                            cache[tag_name] = set()
                        cache[tag_name].add(str(item.id))
        
        return cache
    
    def extract_tag_filters(self, query: str) -> TagFilter:
        """Extract tag filters from natural language query"""
        query_lower = query.lower()
        filter_obj = TagFilter()
        
        for pattern, tags in self.TAG_PATTERNS.items():
            if re.search(pattern, query_lower):
                if isinstance(tags, dict) and "exclude" in tags:
                    # Exclusion tags
                    filter_obj.exclude_tags.update(tags["exclude"])
                elif isinstance(tags, list):
                    # Inclusion tags
                    filter_obj.include_tags.update(tags)
        
        return filter_obj
    
    def search_with_tags(
        self, 
        query: str,
        explicit_filter: Optional[TagFilter] = None
    ) -> List[MenuItem]:
        """
        Search menu items with tag filtering
        
        Args:
            query: Natural language search query
            explicit_filter: Optional explicit tag filter
            
        Returns:
            List of matching menu items
        """
        # Extract filters from query
        auto_filter = self.extract_tag_filters(query)
        
        # Combine with explicit filter if provided
        if explicit_filter:
            auto_filter.include_tags.update(explicit_filter.include_tags)
            auto_filter.exclude_tags.update(explicit_filter.exclude_tags)
        
        # Get base search results
        if query:
            # Remove tag-related words from query for base search
            clean_query = self._clean_query(query)
            if clean_query:
                results = self.menu_provider.search_items(clean_query)
            else:
                # If query only contained tag filters, get all items
                results = self.menu_provider.get_all_items()
        else:
            results = self.menu_provider.get_all_items()
        
        # Apply tag filters
        filtered_results = self._apply_tag_filters(results, auto_filter)
        
        return filtered_results
    
    def _clean_query(self, query: str) -> str:
        """Remove tag-related words from query"""
        # Words to remove
        tag_words = [
            "gluten-free", "gluten free", "no gluten",
            "dairy-free", "dairy free", "no dairy",
            "vegan", "vegetarian", "plant-based",
            "sugar-free", "sugar free", "no sugar",
            "keto", "keto-friendly", "low carb",
            "halal", "kosher",
            "high protein", "low calorie",
            "spicy", "mild",
            "fried", "grilled",
            "healthy", "organic"
        ]
        
        clean = query.lower()
        for word in tag_words:
            clean = clean.replace(word, "")
        
        # Clean up extra spaces
        clean = " ".join(clean.split())
        return clean.strip()
    
    def _apply_tag_filters(
        self, 
        items: List[MenuItem], 
        filter_obj: TagFilter
    ) -> List[MenuItem]:
        """Apply tag filters to item list"""
        filtered = []
        
        for item in items:
            if self._item_matches_filter(item, filter_obj):
                filtered.append(item)
        
        return filtered
    
    def _item_matches_filter(self, item: MenuItem, filter_obj: TagFilter) -> bool:
        """Check if item matches tag filters"""
        if not hasattr(item, 'tags') or not item.tags:
            # No tags - only include if no inclusion filters
            return len(filter_obj.include_tags) == 0
        
        # Get item's tag names
        item_tags = set()
        for tag in item.tags:
            if isinstance(tag, dict):
                item_tags.add(tag.get('name', ''))
            else:
                item_tags.add(str(tag))
        
        # Check inclusion filters (must have ALL)
        if filter_obj.include_tags:
            if not filter_obj.include_tags.issubset(item_tags):
                return False
        
        # Check exclusion filters (must have NONE)
        if filter_obj.exclude_tags:
            if filter_obj.exclude_tags.intersection(item_tags):
                return False
        
        return True
    
    def get_items_by_tag(self, tag_name: str) -> List[MenuItem]:
        """Get all items with a specific tag"""
        item_ids = self._tag_cache.get(tag_name, set())
        
        all_items = self.menu_provider.get_all_items()
        return [
            item for item in all_items 
            if str(item.id) in item_ids
        ]
    
    def get_available_tags(self) -> Dict[str, int]:
        """Get all available tags with item counts"""
        tag_counts = {}
        for tag_name, item_ids in self._tag_cache.items():
            tag_counts[tag_name] = len(item_ids)
        
        return tag_counts
    
    def suggest_tags_for_query(self, query: str) -> List[str]:
        """Suggest relevant tags based on query"""
        query_lower = query.lower()
        suggestions = []
        
        # Check for dietary preferences
        if any(word in query_lower for word in ["diet", "restriction", "allergy", "can't eat"]):
            suggestions.extend(["Gluten-Free", "Dairy-Free", "Vegan", "Vegetarian"])
        
        # Check for health-conscious
        if any(word in query_lower for word in ["healthy", "diet", "fitness", "light"]):
            suggestions.extend(["Low Calorie", "High Protein", "Sugar-Free"])
        
        # Check for style preferences
        if any(word in query_lower for word in ["share", "group", "party"]):
            suggestions.append("Sharing Plate")
        
        return list(set(suggestions))  # Remove duplicates


def create_tag_aware_search(menu_provider: MenuProvider) -> TagAwareMenuSearch:
    """Factory function to create tag-aware search"""
    return TagAwareMenuSearch(menu_provider)