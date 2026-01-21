"""
Helper to integrate time-based menu filtering into existing assistant
"""

from datetime import time
from typing import Optional
from menu.time_aware_provider import TimeAwareMenuProvider
from config.menu_time_config import MenuTimeConfig, MealPeriod, TimeSlot


def upgrade_assistant_with_time_filtering(
    assistant,
    custom_time_config: Optional[MenuTimeConfig] = None,
    test_time: Optional[time] = None
):
    """
    Upgrade existing assistant with time-based menu filtering
    
    Args:
        assistant: The assistant instance to upgrade
        custom_time_config: Custom time configuration (optional)
        test_time: Fixed time for testing (optional)
    """
    
    # Create time configuration
    if custom_time_config is None:
        # Default configuration with common meal periods
        custom_time_config = MenuTimeConfig(
            meal_periods={
                MealPeriod.BREAKFAST: TimeSlot(time(6, 0), time(11, 0)),
                MealPeriod.LUNCH: TimeSlot(time(11, 0), time(15, 0)),
                MealPeriod.DINNER: TimeSlot(time(17, 0), time(22, 0)),
                MealPeriod.LATE_NIGHT: TimeSlot(time(22, 0), time(2, 0)),
                MealPeriod.ALL_DAY: TimeSlot(time(0, 0), time(23, 59))
            },
            menu_mappings={
                "Breakfast Menu": MealPeriod.BREAKFAST,
                "Lunch Special": MealPeriod.LUNCH,
                "Dinner": MealPeriod.DINNER,
                "Our Dinner": MealPeriod.DINNER,
                "Late Night": MealPeriod.LATE_NIGHT,
                "Snack & Beverage": MealPeriod.ALL_DAY,
                "Fast Food": MealPeriod.ALL_DAY
            }
        )
    
    # Get the current menu provider
    if hasattr(assistant, 'menu_provider'):
        base_provider = assistant.menu_provider
    elif hasattr(assistant, 'config') and hasattr(assistant.config, 'menu_provider'):
        base_provider = assistant.config.menu_provider
    else:
        print("⚠️  Could not find menu provider in assistant")
        return assistant
    
    # Create time provider function for testing
    time_provider = None
    if test_time:
        time_provider = lambda: test_time
    
    # Wrap with time-aware provider
    time_aware_provider = TimeAwareMenuProvider(
        base_provider=base_provider,
        time_config=custom_time_config,
        current_time_provider=time_provider
    )
    
    # Replace the menu provider
    if hasattr(assistant, 'menu_provider'):
        assistant.menu_provider = time_aware_provider
    if hasattr(assistant, 'config') and hasattr(assistant.config, 'menu_provider'):
        assistant.config.menu_provider = time_aware_provider
    
    # Update menu matcher if exists
    if hasattr(assistant, 'menu_matcher'):
        assistant.menu_matcher.menu_provider = time_aware_provider
    
    # Add helper methods to assistant
    def _get_current_meal_info(self):
        """Get information about current meal period"""
        periods = self.menu_provider.get_current_meal_periods()
        period_names = [p.value.replace('_', ' ').title() for p in periods]
        return period_names
    
    def _handle_meal_period_query(self, text: str):
        """Handle queries about meal periods"""
        text_lower = text.lower()
        
        # Check for current menu query
        if any(word in text_lower for word in ["what's available", "current menu", "what can i order"]):
            periods = self._get_current_meal_info()
            if periods:
                return f"We're currently serving our {', '.join(periods)} menu. What would you like?"
        
        # Check for specific meal period query
        for period in MealPeriod:
            if period.value.replace('_', ' ') in text_lower:
                items = self.menu_provider.get_items_by_meal_period(period)
                if items:
                    categories = list(set(item.category for item in items))
                    return f"Our {period.value.replace('_', ' ')} menu includes {', '.join(categories)}."
                else:
                    current_periods = self._get_current_meal_info()
                    return f"Sorry, our {period.value.replace('_', ' ')} menu is not available right now. We're currently serving {', '.join(current_periods)}."
        
        return None
    
    # Bind methods to assistant
    assistant._get_current_meal_info = _get_current_meal_info.__get__(assistant)
    assistant._handle_meal_period_query = _handle_meal_period_query.__get__(assistant)
    
    # Patch process_intent to handle meal period queries
    original_process_intent = assistant.process_intent
    
    def patched_process_intent(text, intent):
        # First check for meal period queries
        meal_response = assistant._handle_meal_period_query(text)
        if meal_response:
            return meal_response
        
        # Otherwise use original processing
        return original_process_intent(text, intent)
    
    assistant.process_intent = patched_process_intent
    
    # Display current meal period info
    current_periods = time_aware_provider.get_current_meal_periods()
    period_names = [p.value.replace('_', ' ').title() for p in current_periods]
    
    print("✨ Assistant upgraded with time-based menu filtering!")
    print(f"  - Current meal period(s): {', '.join(period_names)}")
    print(f"  - Total menus configured: {len(custom_time_config.menu_mappings)}")
    if test_time:
        print(f"  - Using test time: {test_time}")
    print("  - Ask 'what's available now?' to see current menu")
    
    return assistant