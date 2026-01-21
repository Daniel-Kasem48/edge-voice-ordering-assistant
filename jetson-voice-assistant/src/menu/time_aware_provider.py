"""
Time-Aware Menu Provider
Filters menu items based on meal periods and current time
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, time
import logging

from .menu_provider import MenuProvider, MenuItem
from ..config.menu_time_config import MenuTimeConfig, MealPeriod, DEFAULT_MENU_TIME_CONFIG


logger = logging.getLogger(__name__)


class TimeAwareMenuProvider(MenuProvider):
    """
    Wrapper for menu providers that adds time-based filtering
    """
    
    def __init__(
        self,
        base_provider: MenuProvider,
        time_config: Optional[MenuTimeConfig] = None,
        current_time_provider: Optional[callable] = None
    ):
        """
        Initialize time-aware menu provider
        
        Args:
            base_provider: The underlying menu provider
            time_config: Time configuration for meal periods
            current_time_provider: Function that returns current time (for testing)
        """
        self.base_provider = base_provider
        self.time_config = time_config or DEFAULT_MENU_TIME_CONFIG
        self._current_time_provider = current_time_provider or self._get_current_time
        
        # Cache for menu period mappings
        self._menu_period_cache: Dict[int, MealPeriod] = {}
        self._items_by_menu: Dict[int, List[str]] = {}
        
        # Initialize cache
        self._build_menu_cache()
    
    def _get_current_time(self) -> time:
        """Get current time"""
        return datetime.now().time()
    
    def _build_menu_cache(self):
        """Build cache of menu periods and item mappings"""
        try:
            # Get raw menu data if available
            if hasattr(self.base_provider, 'raw_menu_data'):
                menu_data = self.base_provider.raw_menu_data
                if menu_data and 'menus' in menu_data:
                    for menu in menu_data['menus']:
                        menu_id = menu['id']
                        menu_name = menu['name']
                        
                        # Infer meal period
                        period = self.time_config.infer_meal_period(menu_name)
                        if period:
                            self._menu_period_cache[menu_id] = period
                            logger.info(f"Menu '{menu_name}' (ID: {menu_id}) mapped to {period.value}")
                        
                        # Map items to menus through categories
                        for category in menu.get('categories', []):
                            category_id = category['id']
                            # Find items in this category
                            for item in menu_data.get('items', []):
                                item_categories = item.get('category', [])
                                if any(cat['id'] == category_id for cat in item_categories):
                                    if menu_id not in self._items_by_menu:
                                        self._items_by_menu[menu_id] = []
                                    self._items_by_menu[menu_id].append(str(item['id']))
        except Exception as e:
            logger.warning(f"Could not build menu cache: {e}")
    
    def _is_item_available(self, item: MenuItem, current_time: time) -> bool:
        """Check if an item is available at current time"""
        # First check metadata for explicit availability
        if item.metadata and 'availability' in item.metadata:
            availability = item.metadata['availability']
            if isinstance(availability, dict):
                start = time.fromisoformat(availability.get('start', '00:00'))
                end = time.fromisoformat(availability.get('end', '23:59'))
                if start <= end:
                    if not (start <= current_time <= end):
                        return False
                else:  # Overnight
                    if not (current_time >= start or current_time <= end):
                        return False
        
        # Check if item belongs to a time-restricted menu
        item_id = str(item.id)
        for menu_id, item_ids in self._items_by_menu.items():
            if item_id in item_ids and menu_id in self._menu_period_cache:
                period = self._menu_period_cache[menu_id]
                if period != MealPeriod.ALL_DAY:
                    time_slot = self.time_config.meal_periods.get(period)
                    if time_slot and not time_slot.contains(current_time):
                        return False
        
        return True
    
    def get_all_items(self) -> List[MenuItem]:
        """Get all currently available items"""
        current_time = self._current_time_provider()
        all_items = self.base_provider.get_all_items()
        
        # Filter by availability
        available_items = [
            item for item in all_items
            if self._is_item_available(item, current_time)
        ]
        
        logger.info(f"Filtered {len(all_items)} items to {len(available_items)} available at {current_time}")
        return available_items
    
    def get_items_by_category(self, category: str) -> List[MenuItem]:
        """Get available items in a category"""
        current_time = self._current_time_provider()
        category_items = self.base_provider.get_items_by_category(category)
        
        # Filter by availability
        return [
            item for item in category_items
            if self._is_item_available(item, current_time)
        ]
    
    def get_items_by_meal_period(self, period: MealPeriod) -> List[MenuItem]:
        """Get all items available during a specific meal period"""
        # Find menu IDs for this period
        menu_ids = [
            menu_id for menu_id, p in self._menu_period_cache.items()
            if p == period
        ]
        
        # Get items from these menus
        period_items = []
        all_items = self.base_provider.get_all_items()
        
        for item in all_items:
            item_id = str(item.id)
            for menu_id in menu_ids:
                if menu_id in self._items_by_menu and item_id in self._items_by_menu[menu_id]:
                    period_items.append(item)
                    break
        
        return period_items
    
    def get_current_meal_periods(self) -> List[MealPeriod]:
        """Get currently active meal periods"""
        current_time = self._current_time_provider()
        return self.time_config.get_current_meal_period(current_time)
    
    def search_items(self, query: str) -> List[MenuItem]:
        """Search available items"""
        current_time = self._current_time_provider()
        search_results = self.base_provider.search_items(query)
        
        # Filter by availability
        return [
            item for item in search_results
            if self._is_item_available(item, current_time)
        ]
    
    def get_item_by_id(self, item_id: str) -> Optional[MenuItem]:
        """Get item by ID if available"""
        item = self.base_provider.get_item_by_id(item_id)
        if item:
            current_time = self._current_time_provider()
            if self._is_item_available(item, current_time):
                return item
            else:
                logger.info(f"Item {item_id} ({item.name}) not available at {current_time}")
        return None
    
    def get_semantic_categories(self) -> Dict[str, List[str]]:
        """Get semantic categories (filtered by current availability)"""
        # Get base semantic categories
        categories = self.base_provider.get_semantic_categories()
        
        # Add meal period categories
        current_periods = self.get_current_meal_periods()
        for period in current_periods:
            period_name = period.value.replace('_', ' ')
            if period_name not in categories:
                # Get items for this period
                period_items = self.get_items_by_meal_period(period)
                if period_items:
                    # Extract unique categories
                    period_categories = list(set(item.category for item in period_items))
                    categories[period_name] = period_categories
        
        return categories