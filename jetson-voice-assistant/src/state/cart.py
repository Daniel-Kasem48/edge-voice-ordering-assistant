"""
Shopping Cart for POS Voice Assistant
Manages order state and provides cart operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json

# Import from menu_matcher
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from nlp.menu_matcher import MatchedItem


@dataclass
class CartItem:
    """Item in the shopping cart"""
    id: str
    matched_item: MatchedItem
    added_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        # Build selected options dict
        selected_options = {}
        for selected_option in self.matched_item.selected_options:
            option_name = selected_option.option.name
            value_names = [v.name for v in selected_option.selected_values]
            selected_options[option_name] = value_names
            
        return {
            "id": self.id,
            "item_name": self.matched_item.item.name,
            "quantity": self.matched_item.quantity,
            "modifiers": [m.name for m in self.matched_item.modifiers],
            "toppings": [t.name for t in self.matched_item.toppings],
            "selected_options": selected_options,
            "unit_price": self.matched_item.get_total_price() / self.matched_item.quantity,
            "total_price": self.matched_item.get_total_price(),
            "description": self.matched_item.get_description(),
            "notes": self.notes
        }


class ShoppingCart:
    """
    Shopping cart for restaurant orders
    Thread-safe cart operations
    """
    
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.items: List[CartItem] = []
        self._item_counter = 0
        self.created_at = datetime.now()
        self.customer_name: Optional[str] = None
        self.table_number: Optional[str] = None
    
    def _generate_id(self) -> str:
        """Generate unique cart item ID"""
        self._item_counter += 1
        return f"item_{self._item_counter}"
    
    def add_item(self, matched_item: MatchedItem, notes: str = "") -> CartItem:
        """
        Add item to cart
        
        Args:
            matched_item: Matched menu item
            notes: Special instructions
            
        Returns:
            CartItem added to cart
        """
        cart_item = CartItem(
            id=self._generate_id(),
            matched_item=matched_item,
            notes=notes
        )
        self.items.append(cart_item)
        return cart_item
    
    def remove_item(self, item_id: str) -> Optional[CartItem]:
        """
        Remove item from cart by ID
        
        Returns:
            Removed item or None if not found
        """
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return self.items.pop(i)
        return None
    
    def remove_by_name(self, item_name: str) -> Optional[CartItem]:
        """
        Remove item from cart by name (fuzzy)
        Removes first matching item
        
        Returns:
            Removed item or None if not found
        """
        item_name_lower = item_name.lower()
        
        for i, item in enumerate(self.items):
            if item_name_lower in item.matched_item.item.name.lower():
                return self.items.pop(i)
            # Check synonyms
            for syn in item.matched_item.item.synonyms:
                if item_name_lower in syn.lower():
                    return self.items.pop(i)
        
        return None
    
    def remove_last(self) -> Optional[CartItem]:
        """Remove the last added item"""
        if self.items:
            return self.items.pop()
        return None
    
    def update_quantity(self, item_id: str, quantity: int) -> bool:
        """
        Update item quantity
        
        Returns:
            True if updated, False if not found
        """
        for item in self.items:
            if item.id == item_id:
                item.matched_item.quantity = quantity
                return True
        return False
    
    def clear(self):
        """Clear all items from cart"""
        self.items.clear()
        self._item_counter = 0
    
    def get_subtotal(self) -> float:
        """Get subtotal before tax"""
        return sum(item.matched_item.get_total_price() for item in self.items)
    
    def get_tax(self, tax_rate: float = 0.08) -> float:
        """Calculate tax amount"""
        return self.get_subtotal() * tax_rate
    
    def get_total(self, tax_rate: float = 0.08) -> float:
        """Get total including tax"""
        return self.get_subtotal() + self.get_tax(tax_rate)
    
    def get_item_count(self) -> int:
        """Get total number of items (considering quantities)"""
        return sum(item.matched_item.quantity for item in self.items)
    
    def is_empty(self) -> bool:
        """Check if cart is empty"""
        return len(self.items) == 0
    
    def get_summary(self, include_tax: bool = True, tax_rate: float = 0.08) -> str:
        """
        Get formatted cart summary
        
        Args:
            include_tax: Include tax in summary
            tax_rate: Tax rate (default 8%)
            
        Returns:
            Formatted summary string
        """
        if self.is_empty():
            return "Your cart is empty."
        
        lines = ["Your order:"]
        
        for item in self.items:
            price = item.matched_item.get_total_price()
            lines.append(f"  • {item.matched_item.get_description()} - ${price:.2f}")
            if item.notes:
                lines.append(f"    Note: {item.notes}")
        
        lines.append("")
        lines.append(f"Subtotal: ${self.get_subtotal():.2f}")
        
        if include_tax:
            lines.append(f"Tax ({tax_rate*100:.0f}%): ${self.get_tax(tax_rate):.2f}")
            lines.append(f"Total: ${self.get_total(tax_rate):.2f}")
        
        return "\n".join(lines)
    
    def get_voice_summary(self) -> str:
        """Get summary optimized for text-to-speech"""
        if self.is_empty():
            return "Your cart is empty."
        
        count = self.get_item_count()
        total = self.get_total()
        
        if count == 1:
            item = self.items[0]
            return f"You have {item.matched_item.get_description()} for ${total:.2f} including tax."
        else:
            # Group items by name for better summary
            item_names = []
            for item in self.items:
                qty = item.matched_item.quantity
                name = item.matched_item.item.name
                if qty > 1:
                    item_names.append(f"{qty}x {name}")
                else:
                    item_names.append(name)
            
            if len(item_names) == 2:
                items_text = " and ".join(item_names)
            else:
                items_text = ", ".join(item_names[:-1]) + f" and {item_names[-1]}"
            
            return f"You have {count} items: {items_text}. Your total is ${total:.2f} including tax."
    
    def to_dict(self) -> dict:
        """Convert cart to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "customer_name": self.customer_name,
            "table_number": self.table_number,
            "items": [item.to_dict() for item in self.items],
            "subtotal": self.get_subtotal(),
            "tax": self.get_tax(),
            "total": self.get_total(),
            "item_count": self.get_item_count()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert cart to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


class CartManager:
    """
    Manages multiple shopping carts
    For multi-tenant/multi-table scenarios
    """
    
    def __init__(self):
        self.carts: Dict[str, ShoppingCart] = {}
    
    def get_cart(self, session_id: str, tenant_id: str = "default") -> ShoppingCart:
        """
        Get or create cart for session
        
        Args:
            session_id: Unique session identifier
            tenant_id: Tenant ID
            
        Returns:
            ShoppingCart for session
        """
        if session_id not in self.carts:
            self.carts[session_id] = ShoppingCart(tenant_id=tenant_id)
        return self.carts[session_id]
    
    def clear_cart(self, session_id: str):
        """Clear cart for session"""
        if session_id in self.carts:
            self.carts[session_id].clear()
    
    def delete_cart(self, session_id: str):
        """Delete cart for session"""
        if session_id in self.carts:
            del self.carts[session_id]
    
    def get_active_carts(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.carts.keys())


if __name__ == "__main__":
    # Test cart operations
    from nlp.menu_matcher import MatchedItem
    from config.tenant_loader import MenuItem
    
    # Create test item
    burger = MenuItem(
        id="burger_1",
        name="Cheeseburger",
        price=8.99,
        category="burgers",
        description="Classic beef burger with cheese"
    )
    
    matched = MatchedItem(
        item=burger,
        quantity=2
    )
    
    # Create cart and add item
    cart = ShoppingCart(tenant_id="demo_restaurant")
    cart.add_item(matched)
    
    print("Cart Test\n" + "="*50)
    print(cart.get_summary())
    print()
    print("Voice summary:", cart.get_voice_summary())
    print()
    print("JSON:", cart.to_json())
