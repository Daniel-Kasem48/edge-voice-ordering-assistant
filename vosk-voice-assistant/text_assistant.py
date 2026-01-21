#!/usr/bin/env python3
"""
Text-Based Voice Assistant (No Audio Required)
Simple command-line interface for testing intent classification and cart operations.
"""

from production_intent_classifier import ProductionIntentClassifier
from datetime import datetime


class TextAssistant:
    """Text-based assistant with cart management"""
    
    def __init__(self):
        print("🔧 Loading Intent Classifier...")
        self.classifier = ProductionIntentClassifier()
        self.cart = {}  # {product: quantity}
        print("✅ Ready!\n")
    
    def process(self, text: str) -> str:
        """Process user input and return response"""
        intent = self.classifier.classify(text)
        
        # Handle each intent
        if intent.name == 'add_to_cart':
            return self._handle_add_to_cart(intent)
        elif intent.name == 'remove_from_cart':
            return self._handle_remove_from_cart(intent)
        elif intent.name == 'view_cart':
            return self._handle_view_cart()
        elif intent.name == 'clear_cart':
            return self._handle_clear_cart()
        elif intent.name == 'checkout':
            return self._handle_checkout()
        elif intent.name == 'search_products':
            return self._handle_search(intent)
        elif intent.name == 'get_time':
            return f"The time is {datetime.now().strftime('%I:%M %p')}"
        elif intent.name == 'get_date':
            return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
        elif intent.name == 'greeting':
            return "Hello! I'm your shopping assistant. How can I help you today?"
        elif intent.name == 'help':
            return self._get_help()
        elif intent.name == 'goodbye':
            return "Goodbye! Have a great day!"
        else:
            return f"I didn't understand that. Say 'help' for available commands."
    
    def _handle_add_to_cart(self, intent) -> str:
        """Handle add to cart with multi-item support"""
        items = intent.entities.get('items', [])
        
        if not items:
            # Fallback to single product
            product = intent.entities.get('product')
            quantity = intent.entities.get('quantity', 1)
            if product:
                items = [{'product': product, 'quantity': quantity}]
        
        if not items:
            return "What would you like to add to your cart?"
        
        responses = []
        for item in items:
            # Handle both CartItem objects and dicts
            if hasattr(item, 'product'):
                product = item.product
                quantity = item.quantity
            else:
                product = item['product']
                quantity = item['quantity']
            
            if product in self.cart:
                self.cart[product] += quantity
            else:
                self.cart[product] = quantity
            
            responses.append(f"{quantity}x {product}")
        
        return f"Added to cart: {', '.join(responses)}"
    
    def _handle_remove_from_cart(self, intent) -> str:
        """Remove item from cart"""
        product = intent.entities.get('product')
        
        if not product:
            return "What would you like to remove?"
        
        if product in self.cart:
            del self.cart[product]
            return f"Removed {product} from your cart."
        else:
            return f"{product} is not in your cart."
    
    def _handle_view_cart(self) -> str:
        """Show cart contents"""
        if not self.cart:
            return "Your cart is empty."
        
        items = [f"  • {qty}x {product}" for product, qty in self.cart.items()]
        total_items = sum(self.cart.values())
        
        return f"Your cart ({total_items} items):\n" + "\n".join(items)
    
    def _handle_clear_cart(self) -> str:
        """Clear all items from cart"""
        self.cart.clear()
        return "Cart cleared!"
    
    def _handle_checkout(self) -> str:
        """Complete the order"""
        if not self.cart:
            return "Your cart is empty. Add some items first!"
        
        items = [f"{qty}x {product}" for product, qty in self.cart.items()]
        order_summary = ", ".join(items)
        self.cart.clear()
        
        return f"Order confirmed! You ordered: {order_summary}. Thank you!"
    
    def _handle_search(self, intent) -> str:
        """Search for products"""
        product = intent.entities.get('product', 'items')
        
        # Mock search results
        available = [
            "Coffee ($3.50)", "Latte ($4.50)", "Cappuccino ($4.00)",
            "Tea ($2.50)", "Muffin ($2.00)", "Pizza ($12.00)",
            "Burger ($8.00)", "Pasta ($10.00)", "Salad ($7.00)"
        ]
        
        return f"Available items: {', '.join(available)}"
    
    def _get_help(self) -> str:
        """Return help message"""
        return """Available commands:
  • "I want 2 coffees" - Add items to cart
  • "one burger one pizza one pepsi" - Add multiple items
  • "show my cart" - View cart contents
  • "remove the coffee" - Remove item from cart
  • "clear cart" - Remove all items
  • "checkout" - Complete your order
  • "what time is it" - Get current time
  • "hello" / "goodbye" - Greetings
  • "help" - Show this message"""
    
    def run(self):
        """Main loop"""
        print("=" * 60)
        print("🛒 TEXT-BASED SHOPPING ASSISTANT")
        print("=" * 60)
        print("Type your commands (or 'quit' to exit)")
        print("Try: 'I want one burger one pizza one pepsi'")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Process and respond
                response = self.process(user_input)
                print(f"🤖 {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    assistant = TextAssistant()
    assistant.run()
