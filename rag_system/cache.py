# rag_system/cache.py
"""
Simple in-memory cache implementation to replace Redis dependency.
"""

import time
import hashlib
import json
from typing import Any, Optional, Dict, Union
from dataclasses import dataclass
from threading import Lock
import asyncio
from collections import OrderedDict


@dataclass
class CacheItem:
    """Cache item with value and expiration time."""
    value: Any
    expires_at: float
    created_at: float


class MemoryCache:
    """
    Thread-safe in-memory cache with TTL support.
    Uses LRU eviction when max_size is reached.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._lock = Lock()
    
    def _generate_key(self, key: Union[str, dict, list]) -> str:
        """Generate a consistent cache key from various input types."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (dict, list)):
            # Create deterministic hash for complex objects
            key_str = json.dumps(key, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        else:
            return str(key)
    
    def _is_expired(self, item: CacheItem) -> bool:
        """Check if cache item has expired."""
        return time.time() > item.expires_at
    
    def _evict_expired(self):
        """Remove expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self._cache.items()
            if current_time > item.expires_at
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_lru(self):
        """Remove least recently used items if cache is full."""
        while len(self._cache) >= self.max_size:
            # Remove oldest item (LRU)
            self._cache.popitem(last=False)
    
    def get(self, key: Union[str, dict, list]) -> Optional[Any]:
        """Get item from cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key not in self._cache:
                return None
            
            item = self._cache[cache_key]
            
            # Check if expired
            if self._is_expired(item):
                del self._cache[cache_key]
                return None
            
            # Move to end (mark as recently used)
            self._cache.move_to_end(cache_key)
            return item.value
    
    def set(self, key: Union[str, dict, list], value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with optional TTL."""
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Clean up expired items first
            self._evict_expired()
            
            # Evict LRU items if necessary
            if cache_key not in self._cache:
                self._evict_lru()
            
            # Create cache item
            current_time = time.time()
            item = CacheItem(
                value=value,
                expires_at=current_time + ttl,
                created_at=current_time
            )
            
            # Add/update item
            self._cache[cache_key] = item
            self._cache.move_to_end(cache_key)  # Mark as recently used
    
    def delete(self, key: Union[str, dict, list]) -> bool:
        """Delete item from cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            self._evict_expired()
            return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._evict_expired()
            current_time = time.time()
            
            stats = {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_ratio": getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1),
                "items": []
            }
            
            for key, item in self._cache.items():
                stats["items"].append({
                    "key": key,
                    "expires_in": max(0, item.expires_at - current_time),
                    "age": current_time - item.created_at
                })
            
            return stats


class AsyncMemoryCache:
    """
    Async wrapper around MemoryCache for use in async contexts.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache = MemoryCache(max_size, default_ttl)
        self._lock = asyncio.Lock()
    
    async def get(self, key: Union[str, dict, list]) -> Optional[Any]:
        """Async get item from cache."""
        async with self._lock:
            return self._cache.get(key)
    
    async def set(self, key: Union[str, dict, list], value: Any, ttl: Optional[int] = None) -> None:
        """Async set item in cache."""
        async with self._lock:
            self._cache.set(key, value, ttl)
    
    async def delete(self, key: Union[str, dict, list]) -> bool:
        """Async delete item from cache."""
        async with self._lock:
            return self._cache.delete(key)
    
    async def clear(self) -> None:
        """Async clear all items from cache."""
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        """Async get current cache size."""
        async with self._lock:
            return self._cache.size()
    
    async def stats(self) -> Dict[str, Any]:
        """Async get cache statistics."""
        async with self._lock:
            return self._cache.stats()


# Cache factory function
def create_cache(config: dict) -> Union[MemoryCache, AsyncMemoryCache]:
    """
    Create cache instance based on configuration.
    
    Args:
        config: Cache configuration dictionary
        
    Returns:
        Cache instance
    """
    cache_config = config.get("cache", {})
    
    if not cache_config.get("enabled", True):
        return None
    
    backend = cache_config.get("backend", "memory")
    max_size = cache_config.get("max_size", 1000)
    ttl = cache_config.get("ttl", 3600)
    
    if backend == "memory":
        return MemoryCache(max_size=max_size, default_ttl=ttl)
    else:
        raise ValueError(f"Unsupported cache backend: {backend}")


def create_async_cache(config: dict) -> Union[AsyncMemoryCache, None]:
    """
    Create async cache instance based on configuration.
    
    Args:
        config: Cache configuration dictionary
        
    Returns:
        Async cache instance or None if caching is disabled
    """
    cache_config = config.get("cache", {})
    
    if not cache_config.get("enabled", True):
        return None
    
    backend = cache_config.get("backend", "memory")
    max_size = cache_config.get("max_size", 1000)
    ttl = cache_config.get("ttl", 3600)
    
    if backend == "memory":
        return AsyncMemoryCache(max_size=max_size, default_ttl=ttl)
    else:
        raise ValueError(f"Unsupported cache backend: {backend}")


# Example usage in RAG system
class CachedEmbeddingService:
    """Example of how to use cache in embedding service."""
    
    def __init__(self, embedding_model, cache: Optional[MemoryCache] = None):
        self.embedding_model = embedding_model
        self.cache = cache
    
    def get_embedding(self, text: str) -> list:
        """Get embedding with caching."""
        if self.cache is None:
            return self.embedding_model.encode([text])[0]
        
        # Try to get from cache
        cached_result = self.cache.get(f"embedding:{text}")
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        embedding = self.embedding_model.encode([text])[0]
        self.cache.set(f"embedding:{text}", embedding, ttl=7200)  # 2 hours
        
        return embedding


class CachedVectorSearch:
    """Example of how to use cache in vector search."""
    
    def __init__(self, vector_store, cache: Optional[AsyncMemoryCache] = None):
        self.vector_store = vector_store
        self.cache = cache
    
    async def search(self, query: str, top_k: int = 5) -> list:
        """Search with caching."""
        if self.cache is None:
            return await self.vector_store.search(query, top_k)
        
        # Create cache key from query parameters
        cache_key = {
            "query": query,
            "top_k": top_k,
            "collection": self.vector_store.collection_name
        }
        
        # Try to get from cache
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Search and cache
        results = await self.vector_store.search(query, top_k)
        await self.cache.set(cache_key, results, ttl=1800)  # 30 minutes
        
        return results