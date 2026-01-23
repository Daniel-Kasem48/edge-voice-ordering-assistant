"""
Tenant Configuration Loader
Loads and validates tenant-specific configuration from YAML files
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class OptionValue:
    """Represents a single option value (e.g., 'Small', 'Medium', 'Large' for size)"""
    id: str
    name: str
    price: float = 0.0
    discount_amount: float = 0.0
    is_default: bool = False
    barcode: Optional[str] = None
    image: Optional[str] = None


@dataclass
class Option:
    """Represents an option group (e.g., 'Size', 'Temperature', 'Add-ons')"""
    id: str
    name: str
    type: str  # 'single_choice', 'multi_choice', 'other'
    is_mandatory: bool = False
    min_selection: int = 0
    max_selection: int = 1
    image: Optional[str] = None
    option_values: List[OptionValue] = field(default_factory=list)


@dataclass
class MenuItem:
    """Represents a menu item"""
    id: str
    name: str
    price: float
    category: str
    synonyms: List[str] = field(default_factory=list)  # aliases in YAML
    description: str = ""
    tags: List[str] = field(default_factory=list)  # dietary/meal tags (vegetarian, breakfast, etc.)
    ingredients: List[str] = field(default_factory=list)  # ingredients list for queries
    default_toppings: List[str] = field(default_factory=list)
    sizes: List[str] = field(default_factory=list)
    flavors: List[str] = field(default_factory=list)
    options: List[Option] = field(default_factory=list)
    
    def matches(self, query: str) -> bool:
        """Check if query matches this item"""
        query_lower = query.lower()
        if self.name.lower() in query_lower or query_lower in self.name.lower():
            return True
        return any(syn.lower() in query_lower or query_lower in syn.lower() 
                   for syn in self.synonyms)


@dataclass
class Modifier:
    """Represents a modifier (size, temperature, etc.)"""
    id: str
    name: str
    category: str  # size, temperature, etc.
    synonyms: List[str] = field(default_factory=list)
    price_adjustment: float = 0.0
    
    def matches(self, query: str) -> bool:
        """Check if query matches this modifier"""
        query_lower = query.lower()
        if self.name.lower() in query_lower:
            return True
        return any(syn.lower() in query_lower for syn in self.synonyms)


@dataclass
class Topping:
    """Represents a topping/addition"""
    id: str
    name: str
    price: float = 0.0
    category: str = ""  # burger_topping, pizza_topping, etc.
    synonyms: List[str] = field(default_factory=list)
    
    def matches(self, query: str) -> bool:
        """Check if query matches this topping"""
        query_lower = query.lower()
        if self.name.lower() in query_lower:
            return True
        return any(syn.lower() in query_lower for syn in self.synonyms)


@dataclass
class TenantConfig:
    """Complete tenant configuration"""
    tenant_id: str
    tenant_name: str
    language: str = "en-US"
    currency: str = "USD"
    
    # Wake word settings
    wake_word_phrase: str = "hey assistant"
    wake_word_sensitivity: float = 0.5
    custom_keyword_path: Optional[str] = None
    
    # Menu items (flat list)
    menu_items: List[MenuItem] = field(default_factory=list)
    
    # Modifiers
    modifiers: List[Modifier] = field(default_factory=list)
    
    # Toppings
    toppings: List[Topping] = field(default_factory=list)
    
    # Quantity mappings
    quantities: Dict[str, int] = field(default_factory=dict)
    
    # Response templates
    responses: Dict[str, str] = field(default_factory=dict)
    
    # Audio settings
    audio_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_items_by_category(self, category: str) -> List[MenuItem]:
        """Get menu items by category"""
        return [item for item in self.menu_items if item.category.lower() == category.lower()]
    
    def find_menu_item(self, query: str) -> Optional[MenuItem]:
        """Find menu item by name or alias"""
        for item in self.menu_items:
            if item.matches(query):
                return item
        return None
    
    def find_modifier(self, query: str) -> Optional[Modifier]:
        """Find modifier by name or alias"""
        for mod in self.modifiers:
            if mod.matches(query):
                return mod
        return None
    
    def find_topping(self, query: str) -> Optional[Topping]:
        """Find topping by name or alias"""
        for top in self.toppings:
            if top.matches(query):
                return top
        return None


class TenantLoader:
    """Loads tenant configurations from YAML files"""
    
    def __init__(self, tenants_dir: str = "data/tenants"):
        self.tenants_dir = Path(tenants_dir)
        self._cache: Dict[str, TenantConfig] = {}
    
    def load_tenant(self, tenant_id: str) -> TenantConfig:
        """Load a tenant configuration by ID"""
        if tenant_id in self._cache:
            return self._cache[tenant_id]
        
        config_path = self.tenants_dir / f"{tenant_id}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Tenant config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        config = self._parse_config(raw_config)
        self._cache[tenant_id] = config
        
        return config
    
    def _parse_json_menu(self, menu_data: Dict, tenant_info: Dict, wake_word: Dict, raw: Dict) -> TenantConfig:
        """Parse menu from JSON format (like our menu.json)"""
        # Parse menu items
        menu_items = []
        for item in menu_data.get('menu_items', []):
            # Parse options if they exist
            options = []
            for opt in item.get('options', []):
                option_values = [
                    OptionValue(
                        id=val['id'],
                        name=val['name'],
                        price=val.get('price', 0.0),
                        is_default=val.get('is_default', False)
                    )
                    for val in opt.get('option_values', [])
                ]
                options.append(Option(
                    id=opt['id'],
                    name=opt['name'],
                    type=opt.get('type', 'single_choice'),
                    is_mandatory=opt.get('is_mandatory', False),
                    min_selection=opt.get('min_selection', 0),
                    max_selection=opt.get('max_selection', 1),
                    option_values=option_values
                ))
            
            menu_items.append(MenuItem(
                id=item['id'],
                name=item['name'],
                price=item.get('price', 0.0),
                category=item.get('category', ''),
                synonyms=item.get('synonyms', []),
                description=item.get('description', ''),
                tags=item.get('tags', []),
                ingredients=item.get('ingredients', []),
                options=options
            ))
        
        # Parse modifiers
        modifiers = []
        for mod in menu_data.get('modifiers', []):
            modifiers.append(Modifier(
                id=mod['id'],
                name=mod['name'],
                category=mod.get('category', ''),
                synonyms=mod.get('synonyms', []),
                price_adjustment=mod.get('price_adjustment', 0.0)
            ))
        
        # Parse toppings
        toppings = []
        for top in menu_data.get('toppings', []):
            toppings.append(Topping(
                id=top['id'],
                name=top['name'],
                price=top.get('price', 0.0),
                category=top.get('category', ''),
                synonyms=top.get('synonyms', [])
            ))
        
        return TenantConfig(
            tenant_id=menu_data.get('tenant_id', tenant_info.get('id', 'unknown')),
            tenant_name=menu_data.get('tenant_name', tenant_info.get('name', 'Unknown Restaurant')),
            language=tenant_info.get('language', menu_data.get('language', 'en-US')),
            currency=tenant_info.get('currency', 'USD'),
            wake_word_phrase=wake_word.get('phrase', menu_data.get('wake_word', 'hey assistant')),
            wake_word_sensitivity=wake_word.get('sensitivity', 0.5),
            custom_keyword_path=wake_word.get('custom_keyword_path'),
            menu_items=menu_items,
            modifiers=modifiers,
            toppings=toppings,
            quantities=raw.get('quantities', {
                'a': 1, 'an': 1, 'one': 1,
                'two': 2, 'couple': 2,
                'three': 3, 'four': 4, 'five': 5
            }),
            responses=raw.get('responses', {}),
            audio_config=raw.get('audio', {})
        )
    
    def _parse_config(self, raw: Dict) -> TenantConfig:
        """Parse raw YAML config into TenantConfig"""
        tenant_info = raw.get('tenant_info', {})
        wake_word = raw.get('wake_word', {})
        
        # Check if menu is loaded from external JSON file
        menu_json_path = raw.get('menu_json_path')
        if menu_json_path:
            # Load menu from JSON file
            import json
            # Support both absolute and relative paths
            json_path = Path(menu_json_path)
            if not json_path.is_absolute():
                # If relative, resolve from data/ directory (parent of tenants_dir)
                json_path = self.tenants_dir.parent / menu_json_path
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    menu_data = json.load(f)
                return self._parse_json_menu(menu_data, tenant_info, wake_word, raw)
            else:
                print(f"Warning: Menu JSON file not found: {json_path}")
        
        # Parse menu items from all categories (original YAML format)
        menu_items = []
        item_counter = 0
        for category, items in raw.get('menu', {}).items():
            for item in items:
                item_counter += 1
                menu_items.append(MenuItem(
                    id=f"item_{item_counter}",
                    name=item['name'],
                    price=item.get('price', 0.0),
                    category=item.get('category', category),
                    synonyms=item.get('aliases', []),
                    description=item.get('description', ''),
                    tags=item.get('tags', []),
                    ingredients=item.get('ingredients', []),
                    default_toppings=item.get('default_toppings', []),
                    sizes=item.get('sizes', []),
                    flavors=item.get('flavors', [])
                ))
        
        # Parse modifiers
        modifiers = []
        mod_counter = 0
        for mod_type, mods in raw.get('modifiers', {}).items():
            for mod in mods:
                mod_counter += 1
                modifiers.append(Modifier(
                    id=f"mod_{mod_counter}",
                    name=mod['name'],
                    category=mod_type,
                    synonyms=mod.get('aliases', []),
                    price_adjustment=mod.get('price_modifier', mod.get('price', 0.0))
                ))
        
        # Parse toppings
        toppings = []
        top_counter = 0
        for top_type, tops in raw.get('toppings', {}).items():
            for top in tops:
                top_counter += 1
                toppings.append(Topping(
                    id=f"top_{top_counter}",
                    name=top['name'],
                    price=top.get('price', 0.0),
                    category=top_type,
                    synonyms=top.get('aliases', [])
                ))
        
        return TenantConfig(
            tenant_id=tenant_info.get('id', 'unknown'),
            tenant_name=tenant_info.get('name', 'Unknown Restaurant'),
            language=tenant_info.get('language', 'en-US'),
            currency=tenant_info.get('currency', 'USD'),
            wake_word_phrase=wake_word.get('phrase', 'hey assistant'),
            wake_word_sensitivity=wake_word.get('sensitivity', 0.5),
            custom_keyword_path=wake_word.get('custom_keyword_path'),
            menu_items=menu_items,
            modifiers=modifiers,
            toppings=toppings,
            quantities=raw.get('quantities', {
                'a': 1, 'an': 1, 'one': 1,
                'two': 2, 'couple': 2,
                'three': 3, 'four': 4, 'five': 5
            }),
            responses=raw.get('responses', {}),
            audio_config=raw.get('audio', {})
        )
    
    def list_tenants(self) -> List[str]:
        """List all available tenant IDs"""
        if not self.tenants_dir.exists():
            return []
        
        return [f.stem for f in self.tenants_dir.glob("*.yaml")]
    
    def reload_tenant(self, tenant_id: str) -> TenantConfig:
        """Force reload a tenant configuration"""
        if tenant_id in self._cache:
            del self._cache[tenant_id]
        return self.load_tenant(tenant_id)


# Convenience function
def load_tenant_config(tenant_id: str, tenants_dir: str = "data/tenants") -> TenantConfig:
    """Load tenant configuration"""
    loader = TenantLoader(tenants_dir)
    return loader.load_tenant(tenant_id)


if __name__ == "__main__":
    # Test loading
    import sys
    
    tenant_id = sys.argv[1] if len(sys.argv) > 1 else "demo_restaurant"
    
    try:
        config = load_tenant_config(tenant_id)
        print(f"✅ Loaded tenant: {config.tenant_name}")
        print(f"   Wake word: {config.wake_word_phrase}")
        print(f"   Menu items: {len(config.menu_items)}")
        print(f"   Modifiers: {len(config.modifiers)}")
        print(f"   Toppings: {len(config.toppings)}")
        print("\n   Sample items:")
        for item in config.menu_items[:5]:
            print(f"     - {item.name} (${item.price:.2f})")
    except FileNotFoundError as e:
        print(f"❌ {e}")
