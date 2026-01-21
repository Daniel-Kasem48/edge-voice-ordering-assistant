# Production Integration Guide

## Intent → Tool Mapping for Production

The `production_intent_classifier.py` detects user intents. Here's how each intent should connect to your actual backend tools/APIs in production:

### 1. **add_to_cart** Intent
**Current**: Adds to local Python dictionary  
**Production**: Should call your e-commerce backend API

```python
async def _execute_intent(self, intent):
    if intent.name == "add_to_cart":
        product = entities.get('product')
        quantity = entities.get('quantity', 1)
        
        # Call actual backend API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/cart/add",
                json={
                    "user_id": self.user_id,
                    "product_name": product,
                    "quantity": quantity
                }
            )
        return response.json()
```

### 2. **search_products** Intent
**Current**: Returns generic response  
**Production**: Should call your Weaviate/Qdrant vector search backend

```python
from src.services.enhanced_search_service import EnhancedSearchService
from src.services.qdrant_client import QdrantService

class VoiceAssistantProduction:
    def __init__(self):
        # Initialize search services
        self.search_service = EnhancedSearchService()
        # OR use Qdrant directly:
        self.qdrant = QdrantService()
    
    async def _execute_search_products(self, query: str):
        """Call actual vector search backend"""
        results = await self.search_service.search_with_learning(
            query=query,
            collection_name="products",  # or "restaurants", "menu_items"
            limit=5,
            user_id=self.user_id
        )
        
        # Format results for voice response
        products = results.get('results', [])
        if products:
            product_names = [p['name'] for p in products[:3]]
            return f"I found: {', '.join(product_names)}. What would you like?"
        else:
            return "Sorry, I couldn't find any matching products."
```

### 3. **view_cart** Intent
**Current**: Shows local cart  
**Production**: Fetch from backend

```python
async def _execute_view_cart(self):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/v1/cart/{self.user_id}"
        )
    cart_data = response.json()
    items = cart_data.get('items', [])
    
    if not items:
        return "Your cart is empty."
    
    item_descriptions = [
        f"{item['quantity']} {item['product_name']}" 
        for item in items
    ]
    return f"Your cart contains: {', '.join(item_descriptions)}"
```

### 4. **checkout** Intent
**Current**: Clears local cart  
**Production**: Process actual payment

```python
async def _execute_checkout(self):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/v1/orders/checkout",
            json={"user_id": self.user_id}
        )
    
    order_data = response.json()
    order_id = order_data.get('order_id')
    total = order_data.get('total')
    
    return f"Order {order_id} confirmed. Total is ${total:.2f}. Thank you!"
```

## Backend API Endpoints Needed

Your backend should expose these REST endpoints:

```
POST   /api/v1/cart/add           - Add item to cart
DELETE /api/v1/cart/remove        - Remove item from cart
GET    /api/v1/cart/{user_id}     - Get cart contents
DELETE /api/v1/cart/{user_id}     - Clear cart
POST   /api/v1/orders/checkout    - Process checkout
GET    /api/v1/products/search    - Search products (vector search)
```

## Search Backend Options

You have multiple search backends available in your project:

### Option 1: Enhanced Search Service (Recommended)
- Location: `src/services/enhanced_search_service.py`
- Features: Self-learning, query expansion, Qdrant vector search
- Use for: Intelligent product/restaurant search with context

### Option 2: FAISS Client
- Location: `src/services/faiss_client.py`
- Features: Fast local vector search, offline-capable
- Use for: Quick semantic search without database

### Option 3: Qdrant Direct
- Location: `src/services/qdrant_client.py`
- Features: Vector database with filters, hybrid search
- Use for: Production-grade search with persistence

## Example: Full Production Integration

```python
#!/usr/bin/env python3
"""
Voice Assistant with Production Backend Integration
"""

import asyncio
import httpx
from production_intent_classifier import ProductionIntentClassifier
from src.services.enhanced_search_service import EnhancedSearchService

class VoiceAssistantProduction:
    def __init__(self, backend_url="http://localhost:8000"):
        self.backend_url = backend_url
        self.intent_classifier = ProductionIntentClassifier()
        self.search_service = EnhancedSearchService()
        self.user_id = "voice_user_123"  # Should be from session/auth
    
    async def execute_intent(self, intent):
        """Route intent to appropriate backend service"""
        
        if intent.name == "search_products":
            query = intent.entities.get('product', intent.raw_text)
            return await self._search_products(query)
        
        elif intent.name == "add_to_cart":
            return await self._add_to_cart(
                product=intent.entities.get('product'),
                quantity=intent.entities.get('quantity', 1)
            )
        
        elif intent.name == "view_cart":
            return await self._view_cart()
        
        elif intent.name == "remove_from_cart":
            return await self._remove_from_cart(
                product=intent.entities.get('product')
            )
        
        elif intent.name == "checkout":
            return await self._checkout()
        
        # ... handle other intents
    
    async def _search_products(self, query: str) -> str:
        """Vector search using your backend"""
        results = await self.search_service.search_with_learning(
            query=query,
            collection_name="products",
            limit=5,
            user_id=self.user_id
        )
        
        products = results.get('results', [])
        if products:
            names = [p.get('name', 'item') for p in products[:3]]
            return f"We have: {', '.join(names)}. What would you like to order?"
        return "Sorry, I couldn't find that product."
    
    async def _add_to_cart(self, product: str, quantity: int) -> str:
        """Call backend cart API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.backend_url}/api/v1/cart/add",
                json={
                    "user_id": self.user_id,
                    "product_name": product,
                    "quantity": quantity
                }
            )
            response.raise_for_status()
        
        return f"Added {quantity} {product} to your cart."
    
    async def _view_cart(self) -> str:
        """Fetch cart from backend"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.backend_url}/api/v1/cart/{self.user_id}"
            )
            response.raise_for_status()
            cart = response.json()
        
        items = cart.get('items', [])
        if not items:
            return "Your cart is empty."
        
        descriptions = [f"{item['quantity']} {item['name']}" for item in items]
        return f"Your cart: {', '.join(descriptions)}"
```

## Deployment Considerations

1. **Authentication**: Add proper user authentication (JWT tokens, session IDs)
2. **Error Handling**: Handle network failures, timeouts gracefully
3. **Caching**: Cache search results to reduce latency
4. **Async/Await**: Use async throughout for better performance
5. **Monitoring**: Log all intent detections and backend calls
6. **Fallback**: Have offline mode when backend is unavailable

## Testing Backend Integration

```python
# Test search
result = await assistant._search_products("coffee")
print(result)  # Should show actual products from Weaviate/Qdrant

# Test cart operations
await assistant._add_to_cart("Latte", 2)
cart_status = await assistant._view_cart()
print(cart_status)  # Should show cart from backend
```

## Current Status

- ✅ Intent classification: Production-ready (95% confidence)
- ✅ Entity extraction: Working (products, quantities)
- ⚠️  Backend integration: Mock data (needs real API calls)
- ⚠️  Search service: Available but not connected
- ⚠️  Cart service: Local Python dict (needs backend)
- ⚠️  Checkout: Mock (needs payment integration)

## Next Steps for Production

1. **Create REST API endpoints** for cart operations in your backend
2. **Connect search_products** to your Weaviate/Qdrant search
3. **Add user authentication** (session management, JWT)
4. **Implement error handling** for network failures
5. **Add database** for cart persistence (currently in-memory)
6. **Integrate payment gateway** for checkout
