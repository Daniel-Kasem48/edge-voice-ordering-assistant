"""
Menu Matcher using RapidFuzz
Fuzzy matching for menu items, modifiers, and toppings
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from rapidfuzz import fuzz, process

# Import tenant config types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.tenant_loader import TenantConfig, MenuItem, Modifier, Topping, Option, OptionValue


@dataclass
class SelectedOption:
    """Represents a selected option with its value"""
    option: Option
    selected_values: List[OptionValue] = field(default_factory=list)


@dataclass
class MatchedItem:
    """A matched menu item with modifiers and toppings"""
    item: MenuItem
    quantity: int = 1
    modifiers: List[Modifier] = field(default_factory=list)
    toppings: List[Topping] = field(default_factory=list)
    selected_options: List[SelectedOption] = field(default_factory=list)
    match_score: float = 100.0
    
    def get_total_price(self) -> float:
        """Calculate total price including modifiers, toppings, and options"""
        base_price = self.item.price
        
        # Add modifier price adjustments
        for mod in self.modifiers:
            base_price += mod.price_adjustment
        
        # Add topping prices
        for topping in self.toppings:
            base_price += topping.price
            
        # Add option value prices (minus any discounts)
        for selected_option in self.selected_options:
            for value in selected_option.selected_values:
                base_price += value.price - value.discount_amount
        
        return base_price * self.quantity
    
    def get_description(self) -> str:
        """Get human-readable description"""
        parts = []
        
        if self.quantity > 1:
            parts.append(f"{self.quantity}x")
        
        # Modifiers before item name
        for mod in self.modifiers:
            parts.append(mod.name)
        
        parts.append(self.item.name)
        
        # Options after item name
        option_descriptions = []
        for selected_option in self.selected_options:
            if selected_option.selected_values:
                value_names = [v.name for v in selected_option.selected_values]
                option_descriptions.append(f"{selected_option.option.name}: {', '.join(value_names)}")
        if option_descriptions:
            parts.append(f"({', '.join(option_descriptions)})")
        
        # Toppings after
        if self.toppings:
            topping_names = [t.name for t in self.toppings]
            parts.append("with " + ", ".join(topping_names))
        
        return " ".join(parts)


@dataclass 
class ParsedOrder:
    """Parsed order from natural language"""
    items: List[MatchedItem] = field(default_factory=list)
    raw_text: str = ""
    unmatched_parts: List[str] = field(default_factory=list)
    
    def get_total(self) -> float:
        """Get total price of all items"""
        return sum(item.get_total_price() for item in self.items)
    
    def get_summary(self) -> str:
        """Get order summary"""
        if not self.items:
            return "No items found"
        
        lines = []
        for item in self.items:
            price = item.get_total_price()
            lines.append(f"  • {item.get_description()} - ${price:.2f}")
        
        lines.append(f"  Total: ${self.get_total():.2f}")
        return "\n".join(lines)


class MenuMatcher:
    """
    Fuzzy matcher for menu items
    Uses RapidFuzz for fast approximate string matching
    """
    
    # Quantity words
    QUANTITIES = {
        "a": 1, "an": 1, "one": 1,
        "two": 2, "couple": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6
    }
    
    def __init__(
        self,
        config: TenantConfig,
        fuzzy_threshold: int = 80,  # Increased from 70 to 80 for stricter matching
        exact_match_bonus: int = 30
    ):
        """
        Initialize menu matcher
        
        Args:
            config: Tenant configuration with menu items
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            exact_match_bonus: Bonus for exact matches
        """
        self.config = config
        self.fuzzy_threshold = fuzzy_threshold
        self.exact_match_bonus = exact_match_bonus
        
        # Build lookup indexes
        self._build_indexes()
    
    def _build_indexes(self):
        """Build search indexes from config"""
        # Menu items: name and synonyms -> MenuItem
        self.item_index: Dict[str, MenuItem] = {}
        for item in self.config.menu_items:
            self.item_index[item.name.lower()] = item
            for syn in item.synonyms:
                self.item_index[syn.lower()] = item
        
        # Modifiers: name and synonyms -> Modifier
        self.modifier_index: Dict[str, Modifier] = {}
        for mod in self.config.modifiers:
            self.modifier_index[mod.name.lower()] = mod
            for syn in mod.synonyms:
                self.modifier_index[syn.lower()] = mod
        
        # Toppings: name and synonyms -> Topping
        self.topping_index: Dict[str, Topping] = {}
        for top in self.config.toppings:
            self.topping_index[top.name.lower()] = top
            for syn in top.synonyms:
                self.topping_index[syn.lower()] = top
        
        # All searchable terms
        self.all_item_terms = list(self.item_index.keys())
        self.all_modifier_terms = list(self.modifier_index.keys())
        self.all_topping_terms = list(self.topping_index.keys())

        # Phonetic index (very lightweight) for low-confidence fallback matching.
        # Maps phonetic codes -> list of terms.
        self._phonetic_item_index: Dict[str, List[str]] = {}
        for term in self.all_item_terms:
            code = self._phonetic_code(term)
            if not code:
                continue
            self._phonetic_item_index.setdefault(code, []).append(term)

    @staticmethod
    def _phonetic_code(text: str) -> str:
        """Compute a cheap phonetic-like code.

        This is intentionally simple (no extra deps) and fast on Jetson.
        It helps recover from some STT and severe typo cases by emphasizing consonant patterns.

        Notes:
        - Not a full Soundex/Metaphone implementation.
        - Works best when comparing single words or short phrases.
        """
        if not text:
            return ""

        # Keep only letters and spaces
        cleaned = re.sub(r"[^a-z\s]", "", text.lower()).strip()
        if not cleaned:
            return ""

        words = [w for w in cleaned.split() if w]
        if not words:
            return ""

        def encode_word(w: str) -> str:
            # Drop vowels after first char; collapse repeats
            vowels = set("aeiouy")
            out = []
            prev = ""
            for i, ch in enumerate(w):
                if i > 0 and ch in vowels:
                    continue
                if ch == prev:
                    continue
                prev = ch
                out.append(ch)
            # Limit length to keep codes comparable
            return "".join(out)[:6]

        # Encode first two words to keep some phrase signal ("ice cream")
        codes = [encode_word(words[0])]
        if len(words) > 1:
            codes.append(encode_word(words[1]))
        return "-".join([c for c in codes if c])
    
    def _extract_quantity(self, text: str) -> Tuple[int, str]:
        """
        Extract quantity from text
        
        Returns:
            Tuple of (quantity, remaining_text)
        """
        words = text.lower().split()
        
        for i, word in enumerate(words):
            if word in self.QUANTITIES:
                qty = self.QUANTITIES[word]
                remaining = " ".join(words[:i] + words[i+1:])
                return qty, remaining
        
        return 1, text
    
    def match_item(self, text: str) -> Optional[Tuple[MenuItem, float]]:
        """
        Find best matching menu item using multi-stage fuzzy matching
        
        Args:
            text: Text to match
            
        Returns:
            Tuple of (MenuItem, score) or None
        """
        text_lower = text.lower().strip()
        
        # Stage 1: Exact match
        if text_lower in self.item_index:
            return self.item_index[text_lower], 100.0
        
        # Stage 2: Exact word match (only for single-word inputs to avoid false positives)
        # e.g., "coffee" in "i want coffee" but not "salad" in "caesar salad"
        words_in_text = text_lower.split()
        if len(words_in_text) == 1:
            # Single word input - check if it matches any term exactly
            if text_lower in self.item_index:
                return self.item_index[text_lower], 95.0
        
        # Stage 3: Multi-word fuzzy matching
        if not self.all_item_terms:
            return None
        
        # Try token_sort_ratio first (best for word order independence)
        result_token = process.extractOne(
            text_lower,
            self.all_item_terms,
            scorer=fuzz.token_sort_ratio
        )
        
        # Try standard ratio (best for typos)
        result_ratio = process.extractOne(
            text_lower,
            self.all_item_terms,
            scorer=fuzz.ratio
        )
        
    # Pick the best result
        best_result = None
        best_score = 0
        
        if result_token and result_token[1] >= self.fuzzy_threshold:
            best_result = result_token
            best_score = result_token[1]
        
        if result_ratio and result_ratio[1] > best_score:
            best_result = result_ratio
            best_score = result_ratio[1]
        
        if best_result and best_score >= self.fuzzy_threshold:
            matched_term = best_result[0]
            return self.item_index[matched_term], best_score

        # Stage 4: Phonetic fallback (only when fuzzy is "close but not enough")
        # This keeps performance high by not running extra work on obvious matches.
        # We only attempt this when the best fuzzy score is moderately close.
        maybe_floor = max(55, self.fuzzy_threshold - 25)
        if best_result and best_score >= maybe_floor:
            code = self._phonetic_code(text_lower)
            if code and code in self._phonetic_item_index:
                # Choose best candidate among phonetic hits using a cheap scorer
                candidates = self._phonetic_item_index[code]
                phonetic_best = process.extractOne(
                    text_lower,
                    candidates,
                    scorer=fuzz.ratio
                )
                if phonetic_best and phonetic_best[1] >= maybe_floor:
                    matched_term = phonetic_best[0]
                    # Slightly penalize phonetic matches so they don't override strong fuzzy matches
                    return self.item_index[matched_term], float(min(79.0, phonetic_best[1]))
        
        return None

    def get_item_candidates(self, text: str, limit: int = 3) -> List[Tuple[MenuItem, float]]:
        """Return top candidate menu items for a text query.

        This is intended for disambiguation UX in the assistant layer.
        It stays cheap by using a single RapidFuzz scorer and then deduplicating by item.
        """
        text_lower = text.lower().strip()
        if not text_lower or not self.all_item_terms:
            return []

        results = process.extract(
            text_lower,
            self.all_item_terms,
            scorer=fuzz.token_sort_ratio,
            limit=max(5, limit * 3)
        )

        seen_ids = set()
        out: List[Tuple[MenuItem, float]] = []
        for term, score, _ in results:
            item = self.item_index[term]
            if item.id in seen_ids:
                continue
            seen_ids.add(item.id)
            out.append((item, float(score)))
            if len(out) >= limit:
                break
        return out
    
    def match_modifier(self, text: str) -> Optional[Tuple[Modifier, float]]:
        """Find best matching modifier"""
        text_lower = text.lower().strip()
        
        # Exact match
        if text_lower in self.modifier_index:
            return self.modifier_index[text_lower], 100.0
        
        # Fuzzy match
        if not self.all_modifier_terms:
            return None
        
        result = process.extractOne(
            text_lower,
            self.all_modifier_terms,
            scorer=fuzz.WRatio
        )
        
        if result and result[1] >= self.fuzzy_threshold:
            return self.modifier_index[result[0]], result[1]
        
        return None
    
    def match_topping(self, text: str) -> Optional[Tuple[Topping, float]]:
        """Find best matching topping"""
        text_lower = text.lower().strip()
        
        # Exact match
        if text_lower in self.topping_index:
            return self.topping_index[text_lower], 100.0
        
        # Fuzzy match
        if not self.all_topping_terms:
            return None
        
        result = process.extractOne(
            text_lower,
            self.all_topping_terms,
            scorer=fuzz.WRatio
        )
        
        if result and result[1] >= self.fuzzy_threshold:
            return self.topping_index[result[0]], result[1]
        
        return None
    
    def parse_order(self, text: str) -> ParsedOrder:
        """
        Parse natural language order into structured format
        
        Args:
            text: Natural language order text
            
        Returns:
            ParsedOrder with matched items
        """
        result = ParsedOrder(raw_text=text)
        text_lower = text.lower()
        
        # Remove common filler words and command words that interfere with matching
        filler_words = [
            'cup of', 'glass of', 'bottle of', 'can of', 'order of', 'plate of',
            'add', 'get', 'order', 'i want', "i'd like", 'give me', 'i need',
            'to cart', 'to order', 'to my cart', 'to my order'
        ]
        for filler in filler_words:
            text_lower = text_lower.replace(filler, ' ')

        # Normalize whitespace
        text_lower = ' '.join(text_lower.split())

        # Extract quantity
        quantity, remaining = self._extract_quantity(text_lower)

        # Remove size/temperature words from the item matching text.
        # We'll detect them later as modifiers, but keeping them here can block item matching
        # (e.g., "large coffee" should still match the menu item "coffee").
        remaining_words = [
            w for w in remaining.split()
            if w not in {'small', 'medium', 'large', 'hot', 'cold', 'iced'}
        ]
        remaining = ' '.join(remaining_words).strip()

        # Robustness for very long/noisy inputs:
        # - drop repeated intensifiers/fillers that can swamp matching
        # - keep only the last few words (the core noun phrase is typically at the end)
        if remaining:
            noise_words = {'very', 'really', 'just', 'please'}
            remaining_words = [w for w in remaining.split() if w not in noise_words]
            if len(remaining_words) > 8:
                remaining_words = remaining_words[-8:]
                
            remaining = ' '.join(remaining_words).strip()
        
        # Extract requested additions/modifiers from "with X" pattern
        requested_additions = []
        with_pattern = r'\bwith\s+(.+?)(?:\s*$)'
        with_match = re.search(with_pattern, text_lower, re.IGNORECASE)
        
        if with_match:
            additions_text = with_match.group(1)
            # Remove "with X" from remaining text
            remaining = remaining.replace(with_match.group(0), '').strip()
            
            # Split additions by "and", ","
            addition_parts = re.split(r'\s+and\s+|,\s*', additions_text)
            requested_additions = [p.strip() for p in addition_parts if p.strip()]
        
        # First, try to match item name that includes the requested additions
        # Example: "burger with bacon" should prefer "bacon burger" over "classic burger"
        best_item_match = None
        best_score = 0
        best_matched_text = ""
        
        if requested_additions:
            # Look for items that include the requested additions in their name
            for term, item in self.item_index.items():
                # Check if item name contains the base item AND the additions
                remaining_words = remaining.split()
                additions_in_name = sum(1 for add in requested_additions if add in item.name.lower())
                base_in_name = any(word in term for word in remaining_words if len(word) > 3)
                
                if additions_in_name > 0 and base_in_name:
                    score = 100 if term == remaining.strip() else 95 + (additions_in_name * 5)
                    if score > best_score:
                        best_item_match = item
                        best_score = score
                        best_matched_text = term
                        # Clear requested_additions since they're part of item name
                        requested_additions = []
        
        # If no match with additions, try a safe containment match using word boundaries.
        # This avoids substring false positives like "cheezburger" matching "burger".
        if not best_item_match:
            remaining_padded = f" {remaining} "
            for term, item in self.item_index.items():
                term_words = term.split()
                if not term:
                    continue
                # Single-word terms must match as a whole word
                if len(term_words) == 1:
                    if f" {term} " in remaining_padded:
                        score = 100 if term == remaining.strip() else 95
                        if score > best_score:
                            best_item_match = item
                            best_score = score
                            best_matched_text = term
                else:
                    # Multi-word phrase containment is safe and useful
                    if term in remaining and len(term) > 3:
                        score = 100 if term == remaining.strip() else 95
                        if score > best_score:
                            best_item_match = item
                            best_score = score
                            best_matched_text = term

        # If still no match, try fuzzy matching
        if not best_item_match:
            # For generic terms like "pizza" without modifiers, prefer first item
            if remaining.strip() in ['pizza', 'burger', 'salad', 'drink']:
                # Find simplest/cheapest item in that category
                category_items = [item for item in self.config.menu_items 
                                if remaining.strip() in item.category.lower()]
                if category_items:
                    # Sort by price (simplest usually cheapest)
                    category_items.sort(key=lambda x: x.price)
                    best_item_match = category_items[0]
                    best_score = 90
            else:
                item_match = self.match_item(remaining)
                if item_match:
                    best_item_match, best_score = item_match

        # Fallback: simple plural normalization ("pizzas" -> "pizza")
        if not best_item_match and remaining.strip().endswith('s') and len(remaining.strip()) > 3:
            singular = remaining.strip()[:-1]
            if singular in ['pizza', 'burger', 'salad', 'drink']:
                category_items = [item for item in self.config.menu_items if singular in item.category.lower()]
                if category_items:
                    category_items.sort(key=lambda x: x.price)
                    best_item_match = category_items[0]
                    best_score = 90
            else:
                item_match = self.match_item(singular)
                if item_match:
                    best_item_match, best_score = item_match
                    remaining = singular
        
        if not best_item_match:
            result.unmatched_parts.append(remaining)
            return result
        
        # Find size/temperature modifiers in the original text
        found_modifiers = []
        size_mods = ['small', 'medium', 'large', 'hot', 'cold', 'iced']
        for size in size_mods:
            if size in text_lower:
                mod_match = self.match_modifier(size)
                if mod_match and mod_match[1] >= 85:
                    found_modifiers.append(mod_match[0])
                    break  # Only one size
        
        # Convert remaining requested_additions to modifiers/toppings
        found_toppings = []
        for addition in requested_additions:
            # Try as modifier first
            mod_match = self.match_modifier(addition)
            if mod_match and mod_match[1] >= 85:
                found_modifiers.append(mod_match[0])
            else:
                # Try as topping
                top_match = self.match_topping(addition)
                if top_match and top_match[1] >= 85:
                    found_toppings.append(top_match[0])
        
        # Create matched item
        matched = MatchedItem(
            item=best_item_match,
            quantity=quantity,
            modifiers=found_modifiers,
            toppings=found_toppings,
            match_score=best_score
        )
        result.items.append(matched)
        
        return result
    
    def search_menu(self, query: str, limit: int = 5) -> List[Tuple[MenuItem, float]]:
        """
        Search menu items by query
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of (MenuItem, score) tuples
        """
        if not self.all_item_terms:
            return []
        
        results = process.extract(
            query.lower(),
            self.all_item_terms,
            scorer=fuzz.WRatio,
            limit=limit
        )
        
        # Deduplicate (same item may match on name and synonym)
        seen_items = set()
        unique_results = []
        
        for term, score, _ in results:
            item = self.item_index[term]
            if item.id not in seen_items:
                seen_items.add(item.id)
                unique_results.append((item, score))
        
        return unique_results


if __name__ == "__main__":
    # Test with sample config
    from config.tenant_loader import TenantLoader
    
    loader = TenantLoader("data/tenants")
    config = loader.load_tenant("demo_restaurant")
    
    matcher = MenuMatcher(config)
    
    test_phrases = [
        "a cheeseburger",
        "two large pepperoni pizzas",
        "burger with bacon and cheese",
        "iced coffee",
        "large fries",
        "ceasar salad",  # Misspelled
        "cheezburger with lettus",  # Misspelled
    ]
    
    print("Menu Matching Test\n" + "="*50)
    
    for phrase in test_phrases:
        order = matcher.parse_order(phrase)
        print(f"'{phrase}'")
        if order.items:
            for item in order.items:
                print(f"  → {item.get_description()} (${item.get_total_price():.2f})")
                print(f"     Score: {item.match_score:.0f}")
        else:
            print(f"  → No match found")
        print()
