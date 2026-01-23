"""
Generate training data dynamically from a restaurant's menu.
This creates restaurant-specific training data for better accuracy.
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple


class MenuBasedTrainingGenerator:
    """Generate training data using actual menu items"""
    
    def __init__(self, menu_path: str):
        self.menu = self._load_menu(menu_path)
        self.items = self._extract_items()
        self.categories = self._extract_categories()
        
    def _load_menu(self, path: str) -> Dict:
        """Load menu from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _extract_items(self) -> List[Dict]:
        """Extract all items from menu"""
        items = []
        
        # Handle different menu formats
        if isinstance(self.menu, dict):
            if 'items' in self.menu:
                items = self.menu['items']
            elif 'products' in self.menu:
                items = self.menu['products']
            elif 'menu' in self.menu:
                items = self.menu['menu']
            else:
                # Assume dict of categories
                for category, category_items in self.menu.items():
                    if isinstance(category_items, list):
                        for item in category_items:
                            if isinstance(item, dict):
                                item['category'] = category
                                items.append(item)
        elif isinstance(self.menu, list):
            items = self.menu
            
        return items
    
    def _extract_categories(self) -> List[str]:
        """Extract unique categories"""
        categories = set()
        for item in self.items:
            if 'category' in item:
                categories.add(item['category'])
        return list(categories)
    
    def _get_item_names(self) -> List[str]:
        """Get all item names and synonyms"""
        names = []
        for item in self.items:
            name = item.get('name', item.get('title', ''))
            if name:
                names.append(name.lower())
            # Also add synonyms
            for syn in item.get('synonyms', []):
                names.append(syn.lower())
        return names
    
    def generate(self) -> List[Tuple[str, str]]:
        """Generate all training data"""
        data = []
        item_names = self._get_item_names()
        
        # If no items found, use generic placeholders
        if not item_names:
            item_names = ['item', 'product', 'that', 'this', 'one']
        
        data.extend(self._generate_add_to_cart(item_names))
        data.extend(self._generate_query_item(item_names))
        data.extend(self._generate_query_ingredients(item_names))
        data.extend(self._generate_search_menu())
        data.extend(self._generate_view_menu())
        data.extend(self._generate_view_cart())
        data.extend(self._generate_remove_from_cart(item_names))
        data.extend(self._generate_clear_cart())
        data.extend(self._generate_checkout())
        data.extend(self._generate_get_price(item_names))
        data.extend(self._generate_modify_item(item_names))
        data.extend(self._generate_confirm_yes())
        data.extend(self._generate_confirm_no())
        data.extend(self._generate_greeting())
        data.extend(self._generate_goodbye())
        data.extend(self._generate_help())
        
        return data
    
    def _generate_add_to_cart(self, items: List[str]) -> List[Tuple[str, str]]:
        """Generate add_to_cart training examples"""
        patterns = [
            "i want a {item}",
            "i want {item}",
            "i'd like a {item}",
            "i'd like {item}",
            "can i get a {item}",
            "can i get {item}",
            "can i have a {item}",
            "can i have {item}",
            "give me a {item}",
            "give me {item}",
            "add a {item}",
            "add {item}",
            "add {item} to my order",
            "add {item} to cart",
            "i'll have a {item}",
            "i'll have {item}",
            "i'll have the {item}",
            "i'll take a {item}",
            "i'll take {item}",
            "one {item} please",
            "{item} please",
            "order a {item}",
            "order {item}",
            "get me a {item}",
            "get me {item}",
            "i need a {item}",
            "i need {item}",
            "gimme a {item}",
            "gimme {item}",
            "let me get a {item}",
            "let me get {item}",
            "put a {item} in my cart",
            "throw in a {item}",
            "hook me up with a {item}",
            "i'm craving a {item}",
            "i'm craving {item}",
            # With quantities
            "two {item}s",
            "2 {item}s",
            "three {item}s",
            "3 {item}s",
            "add two {item}",
            "add 2 {item}",
            "i want two {item}",
            "i want 2 {item}",
            "give me two {item}",
            "give me 2 {item}",
            # With size
            "large {item}",
            "small {item}",
            "medium {item}",
            "a large {item}",
            "a small {item}",
            "a medium {item}",
            "i want a large {item}",
            "i want a small {item}",
            "make it a large {item}",
        ]
        
        data = []
        # Sample items to avoid explosion
        sample_items = random.sample(items, min(20, len(items)))
        
        for pattern in patterns:
            # Use 2-3 random items per pattern
            for item in random.sample(sample_items, min(3, len(sample_items))):
                text = pattern.format(item=item)
                data.append((text, "add_to_cart"))
        
        return data
    
    def _generate_query_item(self, items: List[str]) -> List[Tuple[str, str]]:
        """Generate query_item training examples"""
        patterns = [
            "do you have {item}",
            "do you have a {item}",
            "do you have any {item}",
            "do you sell {item}",
            "is there {item}",
            "is there a {item}",
            "is {item} available",
            "is the {item} available",
            "got any {item}",
            "got {item}",
            "any {item}",
            "what about {item}",
            "how about {item}",
            "can i get {item} here",
            "do you still have {item}",
            "is {item} on the menu",
            "do you offer {item}",
            "what {item} do you have",
            "tell me about {item}",
            "tell me about the {item}",
            "what's in the {item}",
            "describe the {item}",
        ]
        
        # Also add category queries
        category_patterns = [
            "do you have {category}",
            "do you have any {category}",
            "what {category} do you have",
            "show me {category}",
            "any {category} options",
            "got any {category}",
        ]
        
        data = []
        sample_items = random.sample(items, min(15, len(items)))
        
        for pattern in patterns:
            for item in random.sample(sample_items, min(2, len(sample_items))):
                text = pattern.format(item=item)
                data.append((text, "query_item"))
        
        # Add category-based queries
        for pattern in category_patterns:
            for cat in self.categories[:5]:  # Limit categories
                text = pattern.format(category=cat.lower())
                data.append((text, "query_item"))
        
        # Generic availability questions
        generic = [
            "do you have vegetarian options",
            "do you have vegan options",
            "do you have gluten free options",
            "any specials today",
            "what's the special",
            "anything new",
            "what do you recommend",
            "what's popular",
            "what's good here",
        ]
        data.extend([(t, "query_item") for t in generic])
        
        return data
    
    def _generate_query_ingredients(self, items: List[str]) -> List[Tuple[str, str]]:
        """Generate query_ingredients training examples for ingredient queries"""
        patterns = [
            # "what's in X" patterns
            "what's in the {item}",
            "what's in {item}",
            "whats in the {item}",
            "what is in the {item}",
            "what is in {item}",
            
            # "ingredients" patterns
            "what are the ingredients in {item}",
            "what are the ingredients in the {item}",
            "ingredients in {item}",
            "ingredients in the {item}",
            "what ingredients are in {item}",
            "what ingredients does {item} have",
            "what ingredients does the {item} have",
            "tell me the ingredients in {item}",
            "list ingredients in {item}",
            "show me ingredients in {item}",
            "show me the {item} ingredients",  # NEW: "show me the pizza ingredients"
            "{item} ingredients",  # NEW: "pizza ingredients"
            "the {item} ingredients",  # NEW: "the pizza ingredients"
            
            # "how is X made" patterns - NEW
            "how is {item} made",
            "how is the {item} made",
            "how {item} is made",
            "how do you make {item}",
            "how do you make the {item}",
            "what's {item} made of",
            "what's the {item} made of",
            "what's {item} made with",
            "what is {item} made from",
            
            # "does X have Y" patterns
            "does {item} have cheese",
            "does the {item} have cheese",
            "does {item} contain cheese",
            "is there cheese in {item}",
            "is there cheese in the {item}",
            "does {item} have bacon",
            "does {item} have meat",
            "is {item} made with cheese",
            
            # Generic ingredient questions
            "what ingredients",
            "show ingredients",
            "tell me the ingredients",
            "what's it made of",
            "what's it made with",
            "what's in it",
            "ingredients please",
            "list the ingredients",
        ]
        
        data = []
        sample_items = random.sample(items, min(10, len(items)))
        
        # Generate with specific items
        for pattern in patterns[:35]:  # First 35 patterns have {item}
            if '{item}' in pattern:
                for item in random.sample(sample_items, min(2, len(sample_items))):
                    text = pattern.format(item=item)
                    data.append((text, "query_ingredients"))
        
        # Add generic patterns (last few without {item})
        for pattern in patterns[35:]:
            data.append((pattern, "query_ingredients"))
        
        return data
    
    def _generate_search_menu(self) -> List[Tuple[str, str]]:
        """Generate search_menu training examples"""
        patterns = [
            # Show patterns
            "show me the menu",
            "show me your menu",
            "show the menu",
            "show menu",
            "can i see the menu",
            "let me see the menu",
            "pull up the menu",
            # List patterns
            "list the menu",
            "list everything",
            "list all items",
            "list your items",
            # What patterns
            "what's on the menu",
            "what do you have",
            "what do you serve",
            "what's available",
            "what can i order",
            "what are my options",
            "what items do you have",
            # Category browsing
            "show me drinks",
            "show me food",
            "show me desserts",
            "show me snacks",
            "list drinks",
            "list food",
            "drinks please",
            "food options",
            "dessert options",
            # Browse
            "browse menu",
            "browse items",
            "see everything",
            "everything you have",
            "all options",
            "all items",
        ]
        
        # Add category-specific searches
        for cat in self.categories[:8]:
            patterns.extend([
                f"show me {cat.lower()}",
                f"list {cat.lower()}",
                f"what {cat.lower()} do you have",
                f"{cat.lower()} menu",
                f"{cat.lower()} options",
            ])
        
        return [(p, "search_menu") for p in patterns]
    
    def _generate_view_menu(self) -> List[Tuple[str, str]]:
        """Generate view_menu training examples - general menu viewing"""
        patterns = [
            "menu",
            "menu please",
            "the menu",
            "i want to see the menu",
            "open menu",
            "display menu",
            "read me the menu",
            "what's the menu",
            "menu options",
            "full menu",
            "entire menu",
            "complete menu",
            "whole menu",
            "see menu",
            "view menu",
            "check menu",
            "look at menu",
            "bring up menu",
        ]
        return [(p, "view_menu") for p in patterns]
    
    def _generate_view_cart(self) -> List[Tuple[str, str]]:
        """Generate view_cart training examples"""
        patterns = [
            "what's in my cart",
            "show my cart",
            "show cart",
            "view cart",
            "my cart",
            "what did i order",
            "what have i ordered",
            "what's my order",
            "show my order",
            "show order",
            "view order",
            "my order",
            "order summary",
            "cart summary",
            "what do i have",
            "what do i have so far",
            "recap my order",
            "read back my order",
            "tell me my order",
            "list my items",
            "items in cart",
            "current order",
            "order so far",
            "what am i getting",
            "review order",
            "review my order",
            "check my order",
            "check order",
        ]
        return [(p, "view_cart") for p in patterns]
    
    def _generate_remove_from_cart(self, items: List[str]) -> List[Tuple[str, str]]:
        """Generate remove_from_cart training examples"""
        patterns = [
            "remove {item}",
            "remove the {item}",
            "remove a {item}",
            "delete {item}",
            "delete the {item}",
            "take off {item}",
            "take off the {item}",
            "cancel the {item}",
            "cancel {item}",
            "no {item}",
            "i don't want the {item}",
            "i don't want {item}",
            "get rid of the {item}",
            "scratch the {item}",
            "forget the {item}",
            "drop the {item}",
            "actually no {item}",
            "remove {item} from cart",
            "remove {item} from my order",
            "take {item} off my order",
        ]
        
        data = []
        sample_items = random.sample(items, min(10, len(items)))
        
        for pattern in patterns:
            for item in random.sample(sample_items, min(2, len(sample_items))):
                text = pattern.format(item=item)
                data.append((text, "remove_from_cart"))
        
        # Generic removes
        generic = [
            "remove that",
            "remove the last item",
            "remove the last one",
            "delete that",
            "take that off",
            "cancel that",
            "i don't want that",
            "undo that",
            "remove one",
        ]
        data.extend([(t, "remove_from_cart") for t in generic])
        
        return data
    
    def _generate_clear_cart(self) -> List[Tuple[str, str]]:
        """Generate clear_cart training examples"""
        patterns = [
            "clear cart",
            "clear my cart",
            "empty cart",
            "empty my cart",
            "start over",
            "start fresh",
            "clear everything",
            "remove everything",
            "delete everything",
            "cancel order",
            "cancel my order",
            "cancel everything",
            "forget everything",
            "clear order",
            "clear my order",
            "reset order",
            "reset cart",
            "i want to start over",
            "let me start over",
            "begin again",
            "new order",
            "fresh start",
            "wipe cart",
            "erase order",
        ]
        return [(p, "clear_cart") for p in patterns]
    
    def _generate_checkout(self) -> List[Tuple[str, str]]:
        """Generate checkout training examples"""
        patterns = [
            "checkout",
            "check out",
            "i'm done",
            "i'm ready",
            "i'm ready to order",
            "i'm ready to pay",
            "that's all",
            "that's it",
            "that's everything",
            "nothing else",
            "complete order",
            "complete my order",
            "finish order",
            "finish my order",
            "place order",
            "place my order",
            "submit order",
            "submit my order",
            "confirm order",
            "confirm my order",
            "finalize order",
            "finalize my order",
            "done ordering",
            "finished ordering",
            "proceed to checkout",
            "go to checkout",
            "pay now",
            "ready to pay",
            "let's pay",
            "ring me up",
            "wrap it up",
            "that'll be all",
            "all done",
            "send it",
            "let's order",
        ]
        return [(p, "checkout") for p in patterns]
    
    def _generate_get_price(self, items: List[str]) -> List[Tuple[str, str]]:
        """Generate get_price training examples"""
        patterns = [
            "how much is {item}",
            "how much is a {item}",
            "how much is the {item}",
            "how much for {item}",
            "how much for a {item}",
            "what's the price of {item}",
            "what's the price of a {item}",
            "what does {item} cost",
            "what does a {item} cost",
            "price of {item}",
            "price for {item}",
            "{item} price",
            "cost of {item}",
            "{item} cost",
            "how much would {item} be",
            "what's {item} cost",
        ]
        
        data = []
        sample_items = random.sample(items, min(10, len(items)))
        
        for pattern in patterns:
            for item in random.sample(sample_items, min(2, len(sample_items))):
                text = pattern.format(item=item)
                data.append((text, "get_price"))
        
        # Generic price questions
        generic = [
            "how much is that",
            "what's the price",
            "how much",
            "what's my total",
            "what's the total",
            "total price",
            "what do i owe",
            "how much do i owe",
            "what's the damage",
            "price check",
        ]
        data.extend([(t, "get_price") for t in generic])
        
        return data
    
    def _generate_modify_item(self, items: List[str]) -> List[Tuple[str, str]]:
        """Generate modify_item training examples"""
        modifications = [
            "no onions", "no cheese", "no pickles", "no mayo",
            "extra cheese", "extra sauce", "extra spicy",
            "less salt", "less ice", "no ice",
            "with milk", "with sugar", "without sugar",
            "gluten free", "dairy free", "make it spicy",
        ]
        
        patterns = [
            "change {item} to {mod}",
            "make {item} {mod}",
            "i want {item} {mod}",
            "can i get {item} {mod}",
            "{item} but {mod}",
            "modify {item}",
            "change {item}",
            "customize {item}",
        ]
        
        data = []
        sample_items = random.sample(items, min(8, len(items)))
        
        for pattern in patterns[:4]:  # Limit patterns
            for item in random.sample(sample_items, min(2, len(sample_items))):
                mod = random.choice(modifications)
                text = pattern.format(item=item, mod=mod)
                data.append((text, "modify_item"))
        
        # Generic modifications
        generic = [
            "change that",
            "modify that",
            "can i customize",
            "make it spicy",
            "no onions on that",
            "extra cheese please",
            "hold the pickles",
            "on the side",
            "dressing on the side",
            "substitute fries",
            "swap the sides",
            "change the size",
            "make it large",
            "make it small",
            "upgrade to large",
        ]
        data.extend([(t, "modify_item") for t in generic])
        
        return data
    
    def _generate_confirm_yes(self) -> List[Tuple[str, str]]:
        """Generate confirm_yes training examples"""
        patterns = [
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "correct",
            "right",
            "that's right",
            "that's correct",
            "yes please",
            "yes that's right",
            "sounds good",
            "perfect",
            "exactly",
            "affirmative",
            "absolutely",
            "definitely",
            "of course",
            "go ahead",
            "do it",
            "confirm",
            "confirmed",
            "yes add it",
            "add it",
            "yes i want that",
            "that's what i want",
            "you got it",
            "uh huh",
            "mhm",
            "for sure",
            "totally",
            "please do",
        ]
        return [(p, "confirm_yes") for p in patterns]
    
    def _generate_confirm_no(self) -> List[Tuple[str, str]]:
        """Generate confirm_no training examples"""
        patterns = [
            "no",
            "nope",
            "nah",
            "no thanks",
            "no thank you",
            "not that",
            "not that one",
            "wrong",
            "that's wrong",
            "incorrect",
            "cancel",
            "cancel that",
            "never mind",
            "nevermind",
            "forget it",
            "forget that",
            "scratch that",
            "don't add it",
            "don't add that",
            "i don't want that",
            "that's not right",
            "that's not what i said",
            "no i said",
            "wait no",
            "hold on",
            "actually no",
            "negative",
            "nuh uh",
            "not really",
            "i changed my mind",
        ]
        return [(p, "confirm_no") for p in patterns]
    
    def _generate_greeting(self) -> List[Tuple[str, str]]:
        """Generate greeting training examples"""
        patterns = [
            "hi",
            "hello",
            "hey",
            "hey there",
            "hi there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "what's up",
            "sup",
            "yo",
            "greetings",
            "hiya",
            "hola",
            "hi how are you",
            "hello how are you",
            "hey how's it going",
        ]
        return [(p, "greeting") for p in patterns]
    
    def _generate_goodbye(self) -> List[Tuple[str, str]]:
        """Generate goodbye training examples"""
        patterns = [
            "bye",
            "goodbye",
            "bye bye",
            "see you",
            "see ya",
            "later",
            "take care",
            "have a good one",
            "have a nice day",
            "thanks bye",
            "thank you bye",
            "that's all bye",
            "i'm done bye",
            "peace",
            "peace out",
            "catch you later",
            "gotta go",
            "i'm out",
            "ciao",
        ]
        return [(p, "goodbye") for p in patterns]
    
    def _generate_help(self) -> List[Tuple[str, str]]:
        """Generate help training examples"""
        patterns = [
            "help",
            "help me",
            "i need help",
            "can you help",
            "can you help me",
            "what can you do",
            "how does this work",
            "how do i order",
            "what are my options",
            "what can i say",
            "what commands are there",
            "i'm confused",
            "i'm lost",
            "i don't understand",
            "explain",
            "instructions",
            "guide",
            "tutorial",
            "show me how",
            "teach me",
        ]
        return [(p, "help") for p in patterns]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data from menu")
    parser.add_argument("--menu", "-m", required=True, help="Path to menu JSON file")
    parser.add_argument("--output", "-o", default="data/training_data_menu.csv", 
                        help="Output CSV path")
    args = parser.parse_args()
    
    print(f"📂 Loading menu from: {args.menu}")
    generator = MenuBasedTrainingGenerator(args.menu)
    
    print(f"   Found {len(generator.items)} items")
    print(f"   Found {len(generator.categories)} categories: {generator.categories[:5]}...")
    
    # Generate data
    data = generator.generate()
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Print summary
    from collections import Counter
    intent_counts = Counter(intent for _, intent in data)
    
    print(f"\n📊 Training Data Summary:")
    print("-" * 45)
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent:22} {count:4} examples")
    print("-" * 45)
    print(f"  {'TOTAL':22} {len(data):4} examples")
    print(f"  {'Intents':22} {len(intent_counts):4}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'intent'])
        for text, intent in data:
            writer.writerow([text, intent])
    
    print(f"\n✅ Saved {len(data)} examples to {output_path}")
    print(f"\nTo train model:")
    print(f"  python train_two_models.py --data {output_path}")


if __name__ == "__main__":
    main()
