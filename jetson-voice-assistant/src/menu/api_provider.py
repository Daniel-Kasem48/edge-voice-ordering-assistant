"""
API Menu Provider - Fetches menu from REST API
Supports DivPOS API format with /pos/products-db-data endpoint
"""

import requests
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from .menu_provider import MenuProvider, MenuItem

# Import Option and OptionValue from tenant_loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.tenant_loader import Option, OptionValue


class APIMenuProvider(MenuProvider):
    """
    Fetch menu from REST API endpoint with disk caching
    
    Strategy:
    1. On first init, fetch from API and save to disk
    2. On subsequent inits, load from disk (fast startup)
    3. Optionally refresh from API if cache is stale
    
    Example:
        provider = APIMenuProvider(
            base_url="https://dev.divpos.com/api/v2",
            bearer_token="eyJ0eXAi...",
            company_id="1",
            branch_id="1",
            cache_file="menu_cache.json"
        )
        items = provider.get_all_items()
    """
    
    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        company_id: str = "1",
        branch_id: str = "1",
        cache_file: str = "menu_cache.json",
        cache_ttl_hours: int = 24,  # Refresh every 24 hours
        force_refresh: bool = False
    ):
        """
        Initialize API provider
        
        Args:
            base_url: API base URL (e.g., "https://dev.divpos.com/api/v2")
            bearer_token: Authentication bearer token
            company_id: Company ID (default: "1")
            branch_id: Branch ID (default: "1")
            cache_file: Path to save cached menu JSON
            cache_ttl_hours: Cache time-to-live in hours (default: 24)
            force_refresh: Force fetch from API even if cache exists
        """
        self.base_url = base_url.rstrip('/')
        self.bearer_token = bearer_token
        self.company_id = company_id
        self.branch_id = branch_id
        self.cache_file = Path(cache_file)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Indexes
        self._items: List[MenuItem] = []
        self._categories: List[str] = []
        self._category_index: Dict[str, List[MenuItem]] = {}
        self._id_index: Dict[str, MenuItem] = {}
        
        # Try to load from disk first, then fetch if needed
        if not force_refresh and self._load_from_disk():
            print(f"✅ Loaded menu from cache: {self.cache_file}")
        else:
            print(f"🌐 Fetching menu from API...")
            self._fetch_and_save()
            print(f"✅ Menu cached to: {self.cache_file}")
        
        self._build_indexes()
    
    def _load_from_disk(self) -> bool:
        """
        Load menu from disk cache
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.cache_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is stale
            cached_time = datetime.fromisoformat(cache_data.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_time > self.cache_ttl:
                print(f"⚠️  Cache is stale (age: {datetime.now() - cached_time})")
                return False
            
            # Load menu data
            self._load_menu_from_data(cache_data.get('menu_data', {}))
            return True
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️  Failed to load cache: {e}")
            return False
    
    def _fetch_and_save(self):
        """Fetch menu from API and save to disk"""
        menu_data = self._fetch_products()
        
        # Save to disk
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'company_id': self.company_id,
            'branch_id': self.branch_id,
            'menu_data': menu_data
        }
        
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        # Load into memory
        self._load_menu_from_data(menu_data)
    
    def refresh(self):
        """Force refresh menu from API"""
        print(f"🔄 Refreshing menu from API...")
        self._fetch_and_save()
        self._build_indexes()
        print(f"✅ Menu refreshed!")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers"""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "company-id": self.company_id,
            "branch-id": self.branch_id,
            "Application-Id": "sonnet3, android version 2",
            "Application-Version": "4.0.0"
        }
    
    def _fetch_products(self) -> Dict[str, Any]:
        """
        Fetch products from API
        
        Endpoint: GET /pos/products-db-data
        Returns: Dict with keys: menus, items, categories, date
        """
        url = f"{self.base_url}/pos/products-db-data"
        
        try:
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success'):
                raise ValueError(f"API error: {data.get('message', 'Unknown error')}")
            
            return data.get('result', {})
            
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch products from API: {e}")
    
    def _load_menu_from_data(self, data: Dict[str, Any]) -> None:
        """
        Load menu from API response data
        
        API format:
        {
            "menus": [...],
            "items": [...],  # Products
            "categories": [...],
            "date": "..."
        }
        """
        if not data:
            raise ValueError("Empty menu data")
        
        # result_data has keys: menus, items, categories, date
        items_data = data.get('items', [])
        
        # Parse items
        self._items = []
        
        for item_data in items_data:
            # Skip inactive items
            if not item_data.get("active", True):
                continue
            
            # Extract category (API returns as list of dicts)
            category = "Uncategorized"
            if item_data.get("category") and isinstance(item_data["category"], list):
                if item_data["category"]:
                    category = item_data["category"][0].get("name", "Uncategorized")
            
            # Extract images
            images = []
            if item_data.get("images") and isinstance(item_data["images"], list):
                images = [img.get("path", "") for img in item_data["images"] if img.get("path")]
            
            # Parse options
            options = []
            if item_data.get("options") and isinstance(item_data["options"], list):
                for opt_data in item_data["options"]:
                    # Parse option values
                    option_values = []
                    if opt_data.get("option_values") and isinstance(opt_data["option_values"], list):
                        for val_data in opt_data["option_values"]:
                            option_value = OptionValue(
                                id=str(val_data.get("id", "")),
                                name=val_data.get("value", ""),  # Field is called "value" not "name"
                                price=float(val_data.get("price", 0)),
                                discount_amount=float(val_data.get("discount_amount", 0)),
                                is_default=bool(val_data.get("is_default", False)),
                                barcode=val_data.get("barcode"),
                                image=val_data.get("image")
                            )
                            option_values.append(option_value)
                    
                    # Create option
                    # Handle null values for min/max selection
                    min_sel = opt_data.get("min_selection")
                    max_sel = opt_data.get("max_selection")
                    
                    option = Option(
                        id=str(opt_data.get("id", "")),
                        name=opt_data.get("name", ""),
                        type=opt_data.get("type", "other"),
                        is_mandatory=bool(opt_data.get("is_mandatory", 0)),
                        min_selection=int(min_sel) if min_sel is not None else 0,
                        max_selection=int(max_sel) if max_sel is not None else 999,  # High default for unlimited
                        image=opt_data.get("image"),
                        option_values=option_values
                    )
                    options.append(option)
            
            # Parse item
            item = MenuItem(
                id=str(item_data.get("id", "")),
                name=item_data.get("name", "Unknown"),
                category=category,
                price=float(item_data.get("price", 0)),
                description=item_data.get("description") or item_data.get("short_description"),
                aliases=[],  # API doesn't provide aliases
                modifiers=[],  # Could parse from modifier_groups if needed
                toppings=[],  # Could parse from options if needed
                images=images,
                active=item_data.get("active", True),
                options=options,
                metadata=item_data
            )
            
            self._items.append(item)
    
    def _build_indexes(self):
        """Build category and ID indexes"""
        self._category_index = {}
        self._id_index = {}
        
        for item in self._items:
            # Index by category
            if item.category not in self._category_index:
                self._category_index[item.category] = []
            self._category_index[item.category].append(item)
            
            # Index by ID
            self._id_index[item.id] = item
        
        # Get unique categories
        self._categories = list(self._category_index.keys())
        
        # Infer semantic categories
        self._semantic_categories = self._infer_semantic_categories()
    
    def get_all_items(self) -> List[MenuItem]:
        """Get all menu items"""
        return self._items
    
    def get_categories(self) -> List[str]:
        """Get all category names"""
        return list(self._category_index.keys())
    
    def get_items_by_category(self, category: str) -> List[MenuItem]:
        """Get all items in a category"""
        return self._category_index.get(category, [])
    
    def get_item_by_id(self, item_id: str) -> Optional[MenuItem]:
        """Get item by ID"""
        return self._id_index.get(item_id)
    
    def search_items(self, query: str, limit: int = 5) -> List[Tuple[MenuItem, float]]:
        """
        Fuzzy search menu items
        
        Args:
            query: Search query
            limit: Max results to return
            
        Returns:
            List of (MenuItem, score) tuples
        """
        from rapidfuzz import fuzz, process
        
        query = query.lower().strip()
        if not query:
            return []
        
        # Search across item names and search terms
        candidates = []
        for item in self._items:
            for term in item.get_search_terms():
                candidates.append((term, item))
        
        # Fuzzy match
        results = process.extract(
            query,
            [term for term, _ in candidates],
            scorer=fuzz.token_sort_ratio,
            limit=limit
        )
        
        # Map back to items and deduplicate
        seen = set()
        matched_items = []
        for term, score, _ in results:
            # Find item for this term
            item = next((item for t, item in candidates if t == term), None)
            if item and item.id not in seen:
                seen.add(item.id)
                matched_items.append((item, float(score)))
        
        return matched_items[:limit]


def create_api_provider(
    base_url: str,
    store_id: str,
    bearer_token: str,
    cache_ttl: int = 300
) -> APIMenuProvider:
    """
    Factory function to create API provider
    
    Args:
        base_url: API base URL
        store_id: Store/restaurant ID  
        bearer_token: Authentication token
        cache_ttl: Cache duration in seconds (default 5 minutes)
        
    Returns:
        Configured APIMenuProvider
        
    Example:
        provider = create_api_provider(
            base_url="https://dev.divpos.com/api/v2",
            store_id="562",
            bearer_token="eyJ0eXAi..."
        )
    """
    return APIMenuProvider(
        base_url=base_url,
        store_id=store_id,
        bearer_token=bearer_token,
        cache_ttl=cache_ttl
    )
