"""
Menu Time Configuration
Defines meal periods and their availability times
"""

from dataclasses import dataclass
from datetime import time
from typing import List, Dict, Optional
from enum import Enum


class MealPeriod(Enum):
    """Standard meal periods"""
    BREAKFAST = "breakfast"
    BRUNCH = "brunch"
    LUNCH = "lunch"
    DINNER = "dinner"
    LATE_NIGHT = "late_night"
    ALL_DAY = "all_day"


@dataclass
class TimeSlot:
    """Time slot definition"""
    start: time
    end: time
    
    def contains(self, check_time: time) -> bool:
        """Check if a time falls within this slot"""
        if self.start <= self.end:
            # Normal case (e.g., 6:00 - 11:00)
            return self.start <= check_time <= self.end
        else:
            # Overnight case (e.g., 22:00 - 2:00)
            return check_time >= self.start or check_time <= self.end


@dataclass
class MenuTimeConfig:
    """Configuration for menu time periods"""
    # Default meal periods
    meal_periods: Dict[MealPeriod, TimeSlot] = None
    
    # Menu name mappings to meal periods
    menu_mappings: Dict[str, MealPeriod] = None
    
    # Keywords to identify meal periods in menu names
    period_keywords: Dict[MealPeriod, List[str]] = None
    
    def __post_init__(self):
        if self.meal_periods is None:
            self.meal_periods = {
                MealPeriod.BREAKFAST: TimeSlot(time(6, 0), time(11, 0)),
                MealPeriod.BRUNCH: TimeSlot(time(10, 0), time(14, 0)),
                MealPeriod.LUNCH: TimeSlot(time(11, 0), time(15, 0)),
                MealPeriod.DINNER: TimeSlot(time(17, 0), time(22, 0)),
                MealPeriod.LATE_NIGHT: TimeSlot(time(22, 0), time(2, 0)),
                MealPeriod.ALL_DAY: TimeSlot(time(0, 0), time(23, 59))
            }
        
        if self.menu_mappings is None:
            self.menu_mappings = {}
        
        if self.period_keywords is None:
            self.period_keywords = {
                MealPeriod.BREAKFAST: ["breakfast", "morning", "brunch", "am"],
                MealPeriod.LUNCH: ["lunch", "midday", "afternoon"],
                MealPeriod.DINNER: ["dinner", "evening", "supper", "pm"],
                MealPeriod.LATE_NIGHT: ["late night", "midnight", "after hours"],
                MealPeriod.ALL_DAY: ["all day", "anytime", "regular"]
            }
    
    def get_current_meal_period(self, current_time: time) -> List[MealPeriod]:
        """Get meal periods available at current time"""
        available = []
        for period, slot in self.meal_periods.items():
            if slot.contains(current_time):
                available.append(period)
        return available if available else [MealPeriod.ALL_DAY]
    
    def infer_meal_period(self, menu_name: str) -> Optional[MealPeriod]:
        """Infer meal period from menu name"""
        menu_lower = menu_name.lower()
        
        # Check explicit mappings first
        if menu_name in self.menu_mappings:
            return self.menu_mappings[menu_name]
        
        # Check keywords
        for period, keywords in self.period_keywords.items():
            for keyword in keywords:
                if keyword in menu_lower:
                    return period
        
        return None
    
    def is_menu_available(self, menu_name: str, current_time: time) -> bool:
        """Check if a menu is available at current time"""
        period = self.infer_meal_period(menu_name)
        
        # If no period identified, assume always available
        if period is None or period == MealPeriod.ALL_DAY:
            return True
        
        # Check if current time falls within period
        time_slot = self.meal_periods.get(period)
        if time_slot:
            return time_slot.contains(current_time)
        
        return True  # Default to available


# Default configuration
DEFAULT_MENU_TIME_CONFIG = MenuTimeConfig()