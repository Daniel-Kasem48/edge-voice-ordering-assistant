"""
Menu Vocabulary - Extract all searchable terms from a menu dynamically.
No hardcoded item names, categories, or tags.
"""

import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MenuVocabulary:
    """
    Dynamically extracted vocabulary from a menu.
    Used by assistant for category matching, tag filtering, help examples, etc.
    """
    # Categories from menu (e.g., "Espresso & Coffee", "Iced Drinks")
    categories: Set[str] = field(default_factory=set)
    
    # Category aliases - common terms that map to actual categories
    # e.g., {"drinks": ["Espresso & Coffee", "Iced Drinks"], "coffee": ["Espresso & Coffee"]}
    category_aliases: Dict[str, List[str]] = field(default_factory=dict)
    
    # Tags from menu items (e.g., "Vegetarian", "Spicy", "New")
    tags: Set[str] = field(default_factory=set)
    
    # Tag aliases - common terms that map to actual tags
    # e.g., {"veggie": "Vegetarian", "hot": "Spicy"}
    tag_aliases: Dict[str, str] = field(default_factory=dict)
    
    # Ingredients from menu items (e.g., "cheese", "tomato", "beef")
    ingredients: Set[str] = field(default_factory=set)
    
    # Ingredient to items mapping - which items contain which ingredients
    # e.g., {"cheese": ["Margherita Pizza", "Cheeseburger"], "bacon": ["Bacon Burger"]}
    ingredient_to_items: Dict[str, List[str]] = field(default_factory=dict)
    
    # Sample items per category for help examples
    sample_items: Dict[str, List[str]] = field(default_factory=dict)
    
    # All item names for fuzzy matching
    all_item_names: List[str] = field(default_factory=list)
    
    # Option names (e.g., "Size", "Milk Type")
    option_names: Set[str] = field(default_factory=set)
    
    # Option values (e.g., "Small", "Medium", "Large", "Oat Milk")
    option_values: Set[str] = field(default_factory=set)


class MenuVocabularyBuilder:
    """Build vocabulary from menu data"""
    
    # Common aliases for categories (lowercase key -> pattern to match)
    CATEGORY_ALIAS_PATTERNS = {
        "drinks": ["drink", "beverage", "coffee", "tea", "juice", "smoothie", "shake"],
        "food": ["meal", "entree", "main", "dish"],
        "desserts": ["dessert", "sweet", "cake", "pastry", "cookie"],
        "snacks": ["snack", "bite", "appetizer", "starter"],
        "breakfast": ["breakfast", "morning", "brunch"],
        "lunch": ["lunch", "midday"],
        "dinner": ["dinner", "evening"],
        "sides": ["side", "extra", "add-on"],
        "salads": ["salad", "greens"],
        "sandwiches": ["sandwich", "wrap", "panini", "sub"],
        "pizza": ["pizza", "pie"],
        "burgers": ["burger", "patty"],
        "pasta": ["pasta", "noodle", "spaghetti"],
        "soup": ["soup", "broth", "chowder"],
    }
    
    # Common aliases for dietary/attribute tags
    TAG_ALIAS_PATTERNS = {
        "vegetarian": ["vegetarian", "veggie", "veg", "meat-free", "meatless"],
        "vegan": ["vegan", "plant-based", "plant based", "no animal"],
        "gluten-free": ["gluten-free", "gluten free", "no gluten", "gf", "celiac"],
        "dairy-free": ["dairy-free", "dairy free", "no dairy", "lactose-free"],
        "spicy": ["spicy", "hot", "fiery", "extra hot"],
        "mild": ["mild", "not spicy", "no spice"],
        "organic": ["organic", "bio"],
        "sugar-free": ["sugar-free", "sugar free", "no sugar", "unsweetened"],
        "keto": ["keto", "low-carb", "low carb"],
        "halal": ["halal"],
        "kosher": ["kosher"],
        "new": ["new", "latest", "just added"],
        "popular": ["popular", "best seller", "bestseller", "favorite", "favourite"],
        "special": ["special", "featured", "limited"],
    }
    
    def __init__(self, menu_items: List[Dict]):
        """
        Initialize with menu items.
        
        Args:
            menu_items: List of menu item dicts with fields like:
                - name/title: Item name
                - category: Category name
                - tags: List of tags
                - options: List of option dicts
                - description: Item description
        """
        self.items = menu_items
        
    def build(self) -> MenuVocabulary:
        """Build complete vocabulary from menu"""
        vocab = MenuVocabulary()
        
        # Extract categories
        vocab.categories = self._extract_categories()
        vocab.category_aliases = self._build_category_aliases(vocab.categories)
        
        # Extract tags
        vocab.tags = self._extract_tags()
        vocab.tag_aliases = self._build_tag_aliases(vocab.tags)
        
        # Extract ingredients
        vocab.ingredients, vocab.ingredient_to_items = self._extract_ingredients()
        
        # Extract item names
        vocab.all_item_names = self._extract_item_names()
        vocab.sample_items = self._get_sample_items_per_category()
        
        # Extract options
        vocab.option_names, vocab.option_values = self._extract_options()
        
        return vocab
    
    def _extract_categories(self) -> Set[str]:
        """Extract all unique categories from menu"""
        categories = set()
        for item in self.items:
            cat = item.get('category') or item.get('Category') or ''
            if cat:
                categories.add(cat)
        return categories
    
    def _build_category_aliases(self, categories: Set[str]) -> Dict[str, List[str]]:
        """
        Build mapping from common terms to actual category names.
        E.g., "drinks" -> ["Espresso & Coffee", "Iced & Cold Drinks"]
        """
        aliases = defaultdict(list)
        
        for category in categories:
            cat_lower = category.lower()
            
            # Check each alias pattern
            for alias, patterns in self.CATEGORY_ALIAS_PATTERNS.items():
                for pattern in patterns:
                    if pattern in cat_lower:
                        aliases[alias].append(category)
                        break
            
            # Also add the category itself and its words as aliases
            aliases[cat_lower].append(category)
            
            # Add individual words from category name
            for word in re.findall(r'\w+', cat_lower):
                if len(word) > 2:  # Skip short words like "a", "&"
                    aliases[word].append(category)
        
        # Deduplicate
        return {k: list(set(v)) for k, v in aliases.items()}
    
    def _extract_tags(self) -> Set[str]:
        """Extract all unique tags from menu items"""
        tags = set()
        for item in self.items:
            item_tags = item.get('tags') or item.get('Tags') or []
            if isinstance(item_tags, list):
                tags.update(item_tags)
            elif isinstance(item_tags, str):
                tags.add(item_tags)
        return tags
    
    def _extract_ingredients(self) -> tuple:
        """
        Extract all unique ingredients and build ingredient-to-items mapping.
        Returns (set of ingredients, dict mapping ingredient -> item names)
        """
        ingredients = set()
        ingredient_to_items = defaultdict(list)
        
        for item in self.items:
            item_name = item.get('name') or item.get('title') or ''
            item_ingredients = item.get('ingredients') or []
            
            if isinstance(item_ingredients, list):
                for ingredient in item_ingredients:
                    ingredient_lower = ingredient.lower()
                    ingredients.add(ingredient_lower)
                    ingredient_to_items[ingredient_lower].append(item_name)
                    
                    # Also add key words from multi-word ingredients
                    # e.g., "mozzarella cheese" -> also searchable by "mozzarella" or "cheese"
                    words = ingredient_lower.split()
                    for word in words:
                        if len(word) > 3:  # Skip short words like "oil"
                            if word not in ingredient_to_items:
                                ingredient_to_items[word] = []
                            if item_name not in ingredient_to_items[word]:
                                ingredient_to_items[word].append(item_name)
        
        return ingredients, dict(ingredient_to_items)
    
    def _build_tag_aliases(self, tags: Set[str]) -> Dict[str, str]:
        """
        Build mapping from common terms to actual tag names.
        E.g., "veggie" -> "Vegetarian"
        """
        aliases = {}
        
        for tag in tags:
            tag_lower = tag.lower()
            
            # First, map the tag itself
            aliases[tag_lower] = tag
            
            # Check if this tag matches any alias pattern EXACTLY
            for canonical, patterns in self.TAG_ALIAS_PATTERNS.items():
                # Only use patterns if this tag matches the canonical name
                if canonical == tag_lower:
                    # Map all alias patterns to this tag
                    for p in patterns:
                        aliases[p] = tag
                    break
        
        return aliases
    
    def _extract_item_names(self) -> List[str]:
        """Extract all item names"""
        names = []
        for item in self.items:
            name = item.get('name') or item.get('title') or item.get('Name') or ''
            if name:
                names.append(name)
        return names
    
    def _get_sample_items_per_category(self, samples_per_category: int = 3) -> Dict[str, List[str]]:
        """Get sample item names per category for help examples"""
        samples = defaultdict(list)
        
        for item in self.items:
            cat = item.get('category') or item.get('Category') or 'Other'
            name = item.get('name') or item.get('title') or ''
            
            if name and len(samples[cat]) < samples_per_category:
                samples[cat].append(name)
        
        return dict(samples)
    
    def _extract_options(self) -> tuple:
        """Extract option names and values from menu"""
        option_names = set()
        option_values = set()
        
        for item in self.items:
            options = item.get('options') or []
            for opt in options:
                name = opt.get('name') or opt.get('Name') or ''
                if name:
                    option_names.add(name)
                
                values = opt.get('option_values') or opt.get('values') or []
                for val in values:
                    if isinstance(val, dict):
                        val_name = val.get('name') or val.get('Name') or ''
                    else:
                        val_name = str(val)
                    if val_name:
                        option_values.add(val_name)
        
        return option_names, option_values


def build_vocabulary_from_menu(menu_items: List[Dict]) -> MenuVocabulary:
    """Convenience function to build vocabulary from menu items"""
    builder = MenuVocabularyBuilder(menu_items)
    return builder.build()


# Example usage for matching user input
class VocabularyMatcher:
    """Use vocabulary to match user input to menu terms"""
    
    def __init__(self, vocab: MenuVocabulary):
        self.vocab = vocab
        
    def match_category(self, text: str) -> Optional[List[str]]:
        """
        Match user text to menu categories.
        Returns list of matching category names or None.
        """
        text_lower = text.lower()
        
        # Check aliases
        for alias, categories in self.vocab.category_aliases.items():
            if alias in text_lower:
                return categories
        
        # Direct match
        for cat in self.vocab.categories:
            if cat.lower() in text_lower:
                return [cat]
        
        return None
    
    def match_tag(self, text: str) -> Optional[str]:
        """
        Match user text to menu tags.
        Returns matching tag name or None.
        """
        text_lower = text.lower()
        
        # Check aliases
        for alias, tag in self.vocab.tag_aliases.items():
            if alias in text_lower:
                return tag
        
        # Direct match
        for tag in self.vocab.tags:
            if tag.lower() in text_lower:
                return tag
        
        return None
    
    def get_category_words(self) -> List[str]:
        """Get all words that could indicate a category (for intent detection)"""
        words = set()
        words.update(self.vocab.category_aliases.keys())
        for cat in self.vocab.categories:
            words.update(cat.lower().split())
        return list(words)
    
    def get_help_examples(self, num_examples: int = 3) -> str:
        """Generate dynamic help examples from menu"""
        lines = []
        
        # Get a sample item
        sample_item = self.vocab.all_item_names[0] if self.vocab.all_item_names else "item"
        
        # Get a sample category
        sample_category = list(self.vocab.categories)[0] if self.vocab.categories else "menu"
        
        lines.append(f'• "Add a {sample_item}" - add items to your order')
        lines.append(f'• "Show me {sample_category}" - browse category')
        lines.append('• "What\'s in my cart?" - view your order')
        lines.append('• "Remove the last item" - remove items')
        lines.append('• "Checkout" - complete your order')
        
        # Add tag example if tags exist
        if self.vocab.tags:
            sample_tag = list(self.vocab.tags)[0]
            lines.append(f'• "Any {sample_tag.lower()} options?" - filter by tag')
        
        return '\n'.join(lines)
