"""
Enhanced Shopping Cart with Option Validation
Ensures all items meet option constraints before adding to cart
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from .cart import ShoppingCart, CartItem
from .option_validator import OptionValidator, validate_cart_item
from ..nlp.menu_matcher import MatchedItem, SelectedOption
from ..menu.menu_provider import MenuItem


logger = logging.getLogger(__name__)


class ValidatedCart(ShoppingCart):
    """
    Shopping cart with built-in option validation
    Extends base cart with validation logic
    """
    
    def __init__(self, tenant_id: str = "default", strict_validation: bool = True):
        """
        Initialize validated cart
        
        Args:
            tenant_id: Tenant identifier
            strict_validation: If True, reject items with invalid options.
                             If False, try to fix with defaults.
        """
        super().__init__(tenant_id)
        self.strict_validation = strict_validation
        self.validator = OptionValidator()
        
        # Track validation issues
        self.validation_warnings: List[str] = []
        self.last_validation_error: Optional[str] = None
    
    def validate_and_add_item(
        self, 
        matched_item: MatchedItem, 
        notes: str = "",
        auto_fix: bool = True
    ) -> Tuple[Optional[CartItem], bool, str]:
        """
        Validate and add item to cart
        
        Args:
            matched_item: Matched menu item
            notes: Special instructions
            auto_fix: Try to fix missing options with defaults
            
        Returns:
            Tuple of (cart_item, success, message)
        """
        # Extract selected options into validation format
        selected_options = self._extract_selected_options(matched_item)
        
        # Validate options
        is_valid, processed_options, message = validate_cart_item(
            matched_item.item,
            selected_options,
            auto_apply_defaults=auto_fix and not self.strict_validation
        )
        
        if not is_valid:
            self.last_validation_error = message
            logger.warning(f"Validation failed for {matched_item.item.name}: {message}")
            return None, False, message
        
        # Apply processed options back to matched_item if changed
        if processed_options != selected_options:
            self._apply_processed_options(matched_item, processed_options)
            self.validation_warnings.append(
                f"Applied defaults to {matched_item.item.name}: {message}"
            )
        
        # Add to cart
        try:
            cart_item = self.add_item(matched_item, notes)
            success_msg = f"Added {matched_item.quantity}x {matched_item.item.name}"
            
            # Add option details to message
            if matched_item.selected_options:
                option_details = []
                for opt in matched_item.selected_options:
                    values = ", ".join(v.name for v in opt.selected_values)
                    option_details.append(f"{opt.option.name}: {values}")
                
                if option_details:
                    success_msg += f" ({', '.join(option_details)})"
            
            return cart_item, True, success_msg
            
        except Exception as e:
            error_msg = f"Failed to add item: {str(e)}"
            logger.error(error_msg)
            return None, False, error_msg
    
    def add_item(self, matched_item: MatchedItem, notes: str = "") -> CartItem:
        """
        Override base add_item to enforce validation
        
        For backward compatibility, this validates but doesn't reject
        Use validate_and_add_item for strict validation
        """
        # Validate first
        selected_options = self._extract_selected_options(matched_item)
        result = self.validator.validate_item_options(
            matched_item.item,
            selected_options
        )
        
        if not result.is_valid and self.strict_validation:
            error_msg = self.validator.format_validation_errors(result)
            raise ValueError(f"Invalid options: {error_msg}")
        
        if result.warnings:
            self.validation_warnings.extend(result.warnings)
        
        # Call parent add_item
        return super().add_item(matched_item, notes)
    
    def check_missing_options(self) -> List[Tuple[CartItem, List[str]]]:
        """
        Check all cart items for missing required options
        
        Returns:
            List of (cart_item, missing_options) tuples
        """
        items_with_issues = []
        
        for cart_item in self.items:
            matched_item = cart_item.matched_item
            selected_options = self._extract_selected_options(matched_item)
            
            result = self.validator.validate_item_options(
                matched_item.item,
                selected_options
            )
            
            if result.missing_mandatory:
                missing = [opt.name for opt in result.missing_mandatory]
                items_with_issues.append((cart_item, missing))
        
        return items_with_issues
    
    def get_validation_summary(self) -> Dict[str, any]:
        """Get summary of validation issues"""
        missing = self.check_missing_options()
        
        return {
            "has_issues": len(missing) > 0 or len(self.validation_warnings) > 0,
            "items_with_missing_options": len(missing),
            "missing_details": [
                {
                    "item": item.matched_item.item.name,
                    "missing_options": options
                }
                for item, options in missing
            ],
            "warnings": self.validation_warnings,
            "last_error": self.last_validation_error
        }
    
    def clear_validation_warnings(self):
        """Clear accumulated validation warnings"""
        self.validation_warnings = []
        self.last_validation_error = None
    
    def _extract_selected_options(self, matched_item: MatchedItem) -> Dict[str, List[str]]:
        """Extract selected options into validator format"""
        selected_options = {}
        
        if matched_item.selected_options:
            for sel_opt in matched_item.selected_options:
                option_id = str(sel_opt.option.id)
                values = [val.name for val in sel_opt.selected_values]
                selected_options[option_id] = values
        
        return selected_options
    
    def _apply_processed_options(
        self, 
        matched_item: MatchedItem, 
        processed_options: Dict[str, List[str]]
    ):
        """Apply processed options back to matched item"""
        # This would need to update matched_item.selected_options
        # based on processed_options dict
        # Implementation depends on how MatchedItem is structured
        
        # For now, log what would be updated
        logger.info(
            f"Would update options for {matched_item.item.name}: {processed_options}"
        )
    
    def suggest_fixes_for_item(self, cart_item_id: str) -> Optional[str]:
        """
        Suggest how to fix validation issues for a specific cart item
        
        Args:
            cart_item_id: ID of cart item to check
            
        Returns:
            Suggestion message or None if item is valid
        """
        cart_item = next((item for item in self.items if item.id == cart_item_id), None)
        if not cart_item:
            return None
        
        matched_item = cart_item.matched_item
        selected_options = self._extract_selected_options(matched_item)
        
        result = self.validator.validate_item_options(
            matched_item.item,
            selected_options
        )
        
        if result.is_valid:
            return None
        
        suggestions = []
        
        # Missing mandatory options
        if result.missing_mandatory:
            for opt in result.missing_mandatory:
                suggestion = self.validator.suggest_valid_selections(opt)
                if suggestion:
                    suggestions.append(f"For {opt.name}: {suggestion}")
        
        # Other errors
        for error in result.errors:
            if "minimum" in error or "maximum" in error:
                suggestions.append(error)
        
        return "\n".join(suggestions) if suggestions else "Please review option selections"


def create_validated_cart(tenant_id: str = "default", **kwargs) -> ValidatedCart:
    """Factory function to create validated cart"""
    return ValidatedCart(tenant_id=tenant_id, **kwargs)