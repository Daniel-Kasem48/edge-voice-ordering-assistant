"""
Option Validator for Menu Items
Validates option selections against constraints (min/max selection, mandatory)
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from menu.menu_provider import MenuItem, Option


@dataclass
class ValidationResult:
    """Result of option validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_mandatory: List[Option]
    requires_defaults: List[Option]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class OptionValidator:
    """Validates option selections for menu items"""
    
    @staticmethod
    def validate_item_options(
        menu_item: MenuItem,
        selected_options: Optional[Dict[str, List[str]]] = None
    ) -> ValidationResult:
        """
        Validate selected options against item constraints
        
        Args:
            menu_item: The menu item with option definitions
            selected_options: Dict mapping option_id to list of selected values
            
        Returns:
            ValidationResult with validation status and details
        """
        if selected_options is None:
            selected_options = {}
        
        errors = []
        warnings = []
        missing_mandatory = []
        requires_defaults = []
        
        # Check each option group defined for the item
        if menu_item.options:
            for option in menu_item.options:
                option_id = str(option.id)
                selected_values = selected_options.get(option_id, [])
                
                # Validate mandatory options
                if option.is_mandatory and not selected_values:
                    # Check if there's a default value
                    default_values = [
                        val for val in option.option_values 
                        if hasattr(val, 'is_default') and val.is_default
                    ]
                    
                    if default_values:
                        requires_defaults.append(option)
                        warnings.append(
                            f"No selection for mandatory option '{option.name}', "
                            f"will use default: {default_values[0].value}"
                        )
                    else:
                        missing_mandatory.append(option)
                        errors.append(
                            f"Mandatory option '{option.name}' requires selection"
                        )
                
                # Validate selection count
                if selected_values:
                    selected_count = len(selected_values)
                    
                    # Check minimum selection
                    if hasattr(option, 'min_selection') and option.min_selection > 0:
                        if selected_count < option.min_selection:
                            errors.append(
                                f"Option '{option.name}' requires at least "
                                f"{option.min_selection} selection(s), got {selected_count}"
                            )
                    
                    # Check maximum selection
                    if hasattr(option, 'max_selection') and option.max_selection > 0:
                        if selected_count > option.max_selection:
                            errors.append(
                                f"Option '{option.name}' allows maximum "
                                f"{option.max_selection} selection(s), got {selected_count}"
                            )
                    
                    # Validate selected values exist
                    valid_values = {
                        val.value.lower() for val in option.option_values
                    } if hasattr(option, 'option_values') else set()
                    
                    for selected in selected_values:
                        if selected.lower() not in valid_values:
                            errors.append(
                                f"Invalid value '{selected}' for option '{option.name}'. "
                                f"Valid choices: {', '.join(sorted(valid_values))}"
                            )
                
                # Type-specific validation
                if hasattr(option, 'type'):
                    if option.type == 'single' and len(selected_values) > 1:
                        errors.append(
                            f"Option '{option.name}' is single-choice but "
                            f"{len(selected_values)} values were selected"
                        )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            missing_mandatory=missing_mandatory,
            requires_defaults=requires_defaults
        )
    
    @staticmethod
    def apply_defaults(
        menu_item: MenuItem,
        selected_options: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, List[str]]:
        """
        Apply default values for options that weren't selected
        
        Args:
            menu_item: The menu item with option definitions
            selected_options: Current selections
            
        Returns:
            Updated options dict with defaults applied
        """
        if selected_options is None:
            selected_options = {}
        
        result = selected_options.copy()
        
        if menu_item.options:
            for option in menu_item.options:
                option_id = str(option.id)
                
                # Skip if already has selection
                if option_id in result and result[option_id]:
                    continue
                
                # Apply defaults for mandatory options
                if option.is_mandatory:
                    default_values = []
                    
                    if hasattr(option, 'option_values'):
                        for val in option.option_values:
                            if hasattr(val, 'is_default') and val.is_default:
                                default_values.append(val.value)
                    
                    if default_values:
                        result[option_id] = default_values[:1]  # Only first default
        
        return result
    
    @staticmethod
    def get_required_options(menu_item: MenuItem) -> List[Option]:
        """Get list of mandatory options for an item"""
        if not menu_item.options:
            return []
        
        return [
            opt for opt in menu_item.options
            if opt.is_mandatory
        ]
    
    @staticmethod
    def format_validation_errors(result: ValidationResult) -> str:
        """Format validation result as user-friendly message"""
        if result.is_valid:
            return "Options are valid!"
        
        messages = []
        
        if result.missing_mandatory:
            options_list = ", ".join(opt.name for opt in result.missing_mandatory)
            messages.append(f"Please select: {options_list}")
        
        for error in result.errors:
            messages.append(error)
        
        return " ".join(messages)
    
    @staticmethod
    def suggest_valid_selections(option: Option) -> str:
        """Suggest valid selections for an option"""
        if not hasattr(option, 'option_values') or not option.option_values:
            return ""
        
        values = [val.value for val in option.option_values]
        
        if hasattr(option, 'type'):
            if option.type == 'single':
                return f"Choose one: {', '.join(values)}"
            else:
                min_sel = getattr(option, 'min_selection', 0)
                max_sel = getattr(option, 'max_selection', len(values))
                
                if min_sel > 0 and max_sel < len(values):
                    return f"Choose {min_sel}-{max_sel} from: {', '.join(values)}"
                elif min_sel > 0:
                    return f"Choose at least {min_sel} from: {', '.join(values)}"
                elif max_sel < len(values):
                    return f"Choose up to {max_sel} from: {', '.join(values)}"
                else:
                    return f"Choose from: {', '.join(values)}"
        
        return f"Options: {', '.join(values)}"


def validate_cart_item(
    menu_item: MenuItem,
    selected_options: Optional[Dict[str, List[str]]] = None,
    auto_apply_defaults: bool = True
) -> Tuple[bool, Optional[Dict[str, List[str]]], str]:
    """
    Convenient function to validate and prepare item for cart
    
    Args:
        menu_item: The menu item to validate
        selected_options: Selected options
        auto_apply_defaults: Whether to automatically apply defaults
        
    Returns:
        Tuple of (is_valid, processed_options, message)
    """
    validator = OptionValidator()
    
    # Validate current selections
    result = validator.validate_item_options(menu_item, selected_options)
    
    if result.is_valid:
        return True, selected_options, "Item ready to add"
    
    # Try applying defaults if enabled
    if auto_apply_defaults and result.requires_defaults:
        updated_options = validator.apply_defaults(menu_item, selected_options)
        # Re-validate with defaults
        new_result = validator.validate_item_options(menu_item, updated_options)
        
        if new_result.is_valid:
            message = "Applied defaults: " + ", ".join(
                f"{opt.name}={updated_options[str(opt.id)][0]}" 
                for opt in result.requires_defaults
            )
            return True, updated_options, message
    
    # Return validation errors
    error_message = validator.format_validation_errors(result)
    
    # Add suggestions for missing options
    if result.missing_mandatory:
        suggestions = []
        for opt in result.missing_mandatory:
            suggestion = validator.suggest_valid_selections(opt)
            if suggestion:
                suggestions.append(f"{opt.name}: {suggestion}")
        
        if suggestions:
            error_message += "\n" + "\n".join(suggestions)
    
    return False, selected_options, error_message