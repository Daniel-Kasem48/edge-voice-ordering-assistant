"""
Enhanced Menu Matcher with Option Extraction
Handles complex ordering like "large pepperoni pizza with extra cheese"
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rapidfuzz import fuzz

from menu.menu_provider import MenuItem
from config.tenant_loader import Option, OptionValue
from nlp.menu_matcher import MenuMatcher, MatchedItem, SelectedOption


@dataclass
class ExtractedOptions:
    """Options extracted from natural language"""
    size: Optional[str] = None
    modifiers: List[str] = None
    quantity: Optional[int] = None
    
    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []


class EnhancedMenuMatcher(MenuMatcher):
    """
    Enhanced matcher that extracts options from natural language
    
    Examples:
        "large pizza" -> pizza + size:large
        "2 medium coffees with extra sugar" -> coffee + quantity:2 + size:medium + extra sugar
        "crispy calamari with bacon bits" -> calamari + topping:bacon bits
    """
    
    # Size patterns
    SIZE_PATTERNS = {
        'small': ['small', 'sm', 's', 'little', 'mini'],
        'medium': ['medium', 'med', 'm', 'regular', 'normal'],
        'large': ['large', 'lg', 'l', 'big'],
        'extra large': ['extra large', 'xl', 'xlarge', 'extra big', 'huge']
    }
    
    # Quantity patterns
    QUANTITY_WORDS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'a': 1, 'an': 1, 'couple': 2, 'few': 3,
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
    }
    
    def parse_order_with_options(self, text: str) -> Tuple[List[MatchedItem], ExtractedOptions]:
        """
        Parse order text and extract both items and options
        
        Returns:
            (matched_items, extracted_options)
        """
        # First extract options from text
        extracted = self._extract_options(text)
        
        # Clean text by removing option words
        clean_text = self._remove_option_words(text, extracted)
        
        # Parse items using base matcher
        order = self.parse_order(clean_text)
        
        # Apply extracted options to matched items
        for item in order.items:
            self._apply_extracted_options(item, extracted)
        
        return order.items, extracted
    
    def _extract_options(self, text: str) -> ExtractedOptions:
        """Extract options from text"""
        text_lower = text.lower()
        extracted = ExtractedOptions()
        
        # Extract quantity
        for word, num in self.QUANTITY_WORDS.items():
            if f" {word} " in f" {text_lower} " or text_lower.startswith(f"{word} "):
                extracted.quantity = num
                break
        
        # Extract size - use word boundaries to avoid false matches
        for size, patterns in self.SIZE_PATTERNS.items():
            for pattern in patterns:
                # Use word boundaries for short patterns to avoid false matches
                if len(pattern) <= 2:
                    # Check with word boundaries for single letters  
                    if re.search(rf'\b{re.escape(pattern)}\b', text_lower):
                        extracted.size = size
                        break
                else:
                    # Regular substring match for longer patterns
                    if pattern in text_lower:
                        extracted.size = size
                        break
            if extracted.size:
                break
        
        # Extract modifiers (with, no, extra, etc.)
        modifier_patterns = [
            (r'with\s+((?:\w+(?:\s+\w+)*(?:\s+and\s+)?)+)', 'with'),
            (r'no\s+(\w+(?:\s+\w+)?)', 'no'),
            (r'extra\s+(\w+(?:\s+\w+)?)', 'extra'),
            (r'add\s+(\w+(?:\s+\w+)?)', 'add'),
            (r'light\s+(\w+(?:\s+\w+)?)', 'light'),
            (r'less\s+(\w+(?:\s+\w+)?)', 'less')
        ]
        
        for pattern, prefix in modifier_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Handle "milk and sugar" type patterns
                if ' and ' in match:
                    ingredients = re.split(r'\s+and\s+', match)
                    for ingredient in ingredients:
                        ingredient = ingredient.strip()
                        if ingredient:
                            extracted.modifiers.append(f"{prefix} {ingredient}")
                else:
                    extracted.modifiers.append(f"{prefix} {match}")
        
        return extracted
    
    def _remove_option_words(self, text: str, extracted: ExtractedOptions) -> str:
        """Remove option words to get clean item name"""
        clean_text = text
        
        # Remove size words
        if extracted.size:
            for patterns in self.SIZE_PATTERNS.values():
                for pattern in patterns:
                    clean_text = re.sub(rf'\b{pattern}\b', '', clean_text, flags=re.IGNORECASE)
        
        # Remove quantity words
        if extracted.quantity:
            for word in self.QUANTITY_WORDS:
                clean_text = re.sub(rf'\b{word}\b', '', clean_text, flags=re.IGNORECASE)
        
        # Remove modifier phrases
        clean_text = re.sub(r'\b(with|no|extra|add|light|less)\s+\w+(?:\s+\w+)?', '', clean_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        clean_text = ' '.join(clean_text.split())
        
        return clean_text
    
    def _apply_extracted_options(self, matched_item: MatchedItem, extracted: ExtractedOptions):
        """Apply extracted options to matched item"""
        if not matched_item.item.options:
            return
        
        # Apply size ONLY if explicitly found in the text
        # Do NOT apply default options
        if extracted.size:
            size_option = self._find_size_option(matched_item.item)
            if size_option:
                size_value = self._find_option_value_by_name(size_option, extracted.size)
                if size_value:
                    matched_item.selected_options.append(
                        SelectedOption(option=size_option, selected_values=[size_value])
                    )
        
        # Apply quantity
        if extracted.quantity:
            matched_item.quantity = extracted.quantity
        
        # Apply modifiers/toppings
        for modifier in extracted.modifiers:
            self._apply_modifier_to_item(matched_item, modifier)
    
    def _apply_modifier_to_item(self, matched_item: MatchedItem, modifier: str):
        """Apply a single modifier to matched item"""
        modifier_lower = modifier.lower()
        
        # Extract the actual ingredient from modifier (e.g. "with milk" -> "milk")
        ingredient = modifier_lower.split()[-1]  # Get last word
        
        # Try to match to available options
        for option in matched_item.item.options:
            option_name_lower = option.name.lower()
            
            # Match by option type patterns
            if any(pattern in option_name_lower for pattern in ["add", "topping", "extra", "modifier"]):
                # Look for ingredient in option values
                for value in option.option_values:
                    value_name_lower = value.name.lower()
                    if (ingredient in value_name_lower or 
                        value_name_lower in ingredient or
                        fuzz.ratio(ingredient, value_name_lower) > 80):
                        matched_item.selected_options.append(
                            SelectedOption(option=option, selected_values=[value])
                        )
                        return  # Found match, stop looking
    
    def _find_size_option(self, item: MenuItem) -> Optional[Option]:
        """Find size option in item"""
        for option in item.options:
            if 'size' in option.name.lower():
                return option
        return None
    
    def _find_option_value_by_name(self, option: Option, name: str) -> Optional[OptionValue]:
        """Find option value by name"""
        name_lower = name.lower()
        
        # Exact match
        for value in option.option_values:
            if value.name.lower() == name_lower:
                return value
        
        # Fuzzy match
        for value in option.option_values:
            if fuzz.ratio(value.name.lower(), name_lower) > 80:
                return value
        
        return None
    
    def match_item_by_name_only(self, text: str) -> Optional[MenuItem]:
        """Match item by name, ignoring options"""
        # Remove common option words first
        clean_text = text.lower()
        
        # Remove size words
        for patterns in self.SIZE_PATTERNS.values():
            for pattern in patterns:
                clean_text = re.sub(rf'\b{pattern}\b', '', clean_text)
        
        # Remove other modifiers
        clean_text = re.sub(r'\b(with|no|extra|add|light|less)\s+\w+(?:\s+\w+)?', '', clean_text)
        clean_text = ' '.join(clean_text.split())
        
        # Now search
        results = self.search_menu(clean_text, limit=1)
        if results:
            return results[0][0]
        
        return None