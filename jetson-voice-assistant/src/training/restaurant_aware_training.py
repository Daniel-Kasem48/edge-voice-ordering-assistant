"""
Restaurant-Aware Training Data Generator
Enhances generic training with restaurant-specific patterns
"""

import json
import csv
from typing import List, Dict, Tuple
from pathlib import Path
import random

from menu.menu_provider import MenuProvider, MenuItem


class RestaurantAwareTrainingGenerator:
    """Generates training data that combines generic intents with restaurant-specific vocabulary"""
    
    def __init__(self, menu_provider: MenuProvider):
        self.menu_provider = menu_provider
        self.base_patterns = self._load_base_patterns()
        
    def _load_base_patterns(self) -> Dict[str, List[str]]:
        """Load generic intent patterns"""
        return {
            "add_to_cart": [
                "I want {item}",
                "I'd like {item}",
                "can I get {item}",
                "give me {item}",
                "add {item}",
                "I'll have {item}",
                "make it {item}",
                "{quantity} {item} please"
            ],
            "search_menu": [
                "what {category} do you have",
                "show me {category}",
                "what's on the {category} menu",
                "do you have {category}",
                "what kind of {category}",
                "list all {category}"
            ],
            "view_cart": [
                "what's my total",
                "show my order",
                "what did I order",
                "view cart",
                "check my items"
            ],
            "checkout": [
                "check please",
                "I'm ready to pay",
                "checkout",
                "that's all",
                "complete order"
            ],
            "remove_from_cart": [
                "remove {item}",
                "cancel {item}",
                "no {item}",
                "take off the {item}"
            ]
        }
    
    def generate_training_data(self, 
                             include_restaurant_specific: bool = True,
                             samples_per_item: int = 5) -> List[Tuple[str, str]]:
        """Generate training data combining generic and restaurant-specific patterns"""
        
        training_data = []
        
        # Add generic examples
        training_data.extend(self._generate_generic_examples())
        
        if include_restaurant_specific:
            # Add restaurant-specific examples
            training_data.extend(self._generate_restaurant_examples(samples_per_item))
        
        # Shuffle for better training
        random.shuffle(training_data)
        
        return training_data
    
    def _generate_generic_examples(self) -> List[Tuple[str, str]]:
        """Generate generic training examples"""
        examples = []
        
        # Generic items
        generic_items = ["pizza", "burger", "salad", "drink", "dessert", "coffee", "sandwich"]
        generic_categories = ["drinks", "desserts", "appetizers", "entrees", "sides"]
        
        # Add to cart examples
        for pattern in self.base_patterns["add_to_cart"]:
            for item in generic_items:
                if "{item}" in pattern:
                    text = pattern.replace("{item}", item)
                    if "{quantity}" in text:
                        text = text.replace("{quantity}", random.choice(["one", "two", "2", "3"]))
                    examples.append((text, "add_to_cart"))
        
        # Search menu examples
        for pattern in self.base_patterns["search_menu"]:
            for category in generic_categories:
                if "{category}" in pattern:
                    text = pattern.replace("{category}", category)
                    examples.append((text, "search_menu"))
        
        # Fixed intent examples
        for intent in ["view_cart", "checkout"]:
            for pattern in self.base_patterns[intent]:
                examples.append((pattern, intent))
        
        return examples
    
    def _generate_restaurant_examples(self, samples_per_item: int) -> List[Tuple[str, str]]:
        """Generate restaurant-specific training examples"""
        examples = []
        
        # Get actual menu items
        menu_items = self.menu_provider.get_all_items()
        
        # Get unique categories
        categories = list(set(item.category for item in menu_items))
        
        # Generate item-specific examples
        for item in menu_items:
            # Use item name
            for i in range(samples_per_item):
                pattern = random.choice(self.base_patterns["add_to_cart"])
                if "{item}" in pattern:
                    text = pattern.replace("{item}", item.name.lower())
                    if "{quantity}" in text:
                        text = text.replace("{quantity}", random.choice(["one", "two", "2", "3"]))
                    examples.append((text, "add_to_cart"))
            
            # Use aliases
            if item.aliases:
                for alias in item.aliases[:2]:  # Limit aliases to avoid overfit
                    pattern = random.choice(self.base_patterns["add_to_cart"][:3])
                    if "{item}" in pattern:
                        text = pattern.replace("{item}", alias.lower())
                        if "{quantity}" in text:
                            text = text.replace("{quantity}", "one")
                        examples.append((text, "add_to_cart"))
        
        # Generate category-specific examples
        for category in categories:
            for pattern in self.base_patterns["search_menu"]:
                if "{category}" in pattern:
                    text = pattern.replace("{category}", category.lower())
                    examples.append((text, "search_menu"))
        
        return examples
    
    def save_to_csv(self, 
                   filename: str, 
                   include_restaurant_specific: bool = True):
        """Save training data to CSV file"""
        
        data = self.generate_training_data(include_restaurant_specific)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "intent"])  # Header
            writer.writerows(data)
        
        print(f"✅ Saved {len(data)} training examples to {filename}")
        
        # Show intent distribution
        intent_counts = {}
        for _, intent in data:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("\nIntent distribution:")
        for intent, count in sorted(intent_counts.items()):
            print(f"  {intent}: {count} examples")
    
    def generate_validation_set(self, test_size: float = 0.2) -> Tuple[List, List]:
        """Generate train/validation split"""
        
        all_data = self.generate_training_data()
        
        # Shuffle
        random.shuffle(all_data)
        
        # Split
        split_idx = int(len(all_data) * (1 - test_size))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        return train_data, val_data


def create_multi_restaurant_model(restaurant_configs: List[Dict]) -> None:
    """
    Create a model that works well across multiple restaurants
    by training on combined data from all restaurants
    """
    
    all_training_data = []
    
    for config in restaurant_configs:
        print(f"\nProcessing restaurant: {config['name']}")
        
        # Load menu for this restaurant
        menu_provider = config['menu_provider']
        
        # Generate training data
        generator = RestaurantAwareTrainingGenerator(menu_provider)
        restaurant_data = generator.generate_training_data(
            include_restaurant_specific=True,
            samples_per_item=3  # Less samples to avoid overfit
        )
        
        all_training_data.extend(restaurant_data)
    
    # Save combined training data
    output_file = "training_data_multi_restaurant.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(all_training_data)
    
    print(f"\n✅ Combined training data saved to {output_file}")
    print(f"Total examples: {len(all_training_data)}")
    
    # Now train with the combined data
    print("\nTo train the model, run:")
    print(f"python train_from_csv.py {output_file}")


# Example usage
if __name__ == "__main__":
    from menu.api_provider import APIMenuProvider
    
    # Example for single restaurant
    provider = APIMenuProvider(
        base_url="https://dummy-url.com",
        bearer_token="dummy-token",
        cache_file="menu_cache_time_based.json",
        force_refresh=False
    )
    
    generator = RestaurantAwareTrainingGenerator(provider)
    
    # Generate training data
    generator.save_to_csv(
        "training_data_restaurant_aware.csv",
        include_restaurant_specific=True
    )
    
    print("\n💡 Training approach options:")
    print("1. Generic only: Fast, works everywhere, may miss specific items")
    print("2. Restaurant-aware: Better accuracy, needs retraining for menu changes")
    print("3. Hybrid: Use generic model + dynamic menu matching (recommended)")