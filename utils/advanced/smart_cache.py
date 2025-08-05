"""
Sistema de Caché Inteligente para Optimización de Memoria y Rendimiento
Implementa caching avanzado con LRU, compresión y warming strategies
"""

import asyncio
import time
import pickle
import hashlib
import logging
from typing import Any, Dict, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc
from threading import RLock
import numpy as np
import psutil
import zlib

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Estadísticas del cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    compression_ratio: float = 0.0
    total_operations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Tasa de aciertos del cache"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property 
    def memory_usage_mb(self) -> float:
        """Uso de memoria en MB"""
        return self.memory_usage_bytes / (1024 * 1024)

class SmartLRUCache:
    """
    Cache LRU inteligente con compresión automática y gestión de memoria
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_memory_mb: int = 512,
        compression_threshold: int = 1024,  # Comprimir items > 1KB
        auto_cleanup: bool = True,
        ttl_seconds: Optional[int] = 3600  # TTL de 1 hora por defecto
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        self.auto_cleanup = auto_cleanup
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._compressed_items: set = set()
        self._item_sizes: Dict[str, int] = {}
        self._lock = RLock()
        
        self.stats = CacheStats()
        
        # Cleanup automático
        if auto_cleanup:
            self._start_cleanup_task()
    
    def _compute_key(self, key: Union[str, bytes, Any]) -> str:
        """Generar clave única para cualquier objeto"""
        if isinstance(key, str):
            return key
        elif isinstance(key, bytes):
            return hashlib.md5(key).hexdigest()
        else:
            # Para objetos complejos, usar pickle + hash
            try:
                serialized = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
                return hashlib.md5(serialized).hexdigest()
            except Exception:
                return str(hash(key))
    
    def _compress_value(self, value: Any) -> bytes:
        """Comprimir valor si es beneficioso"""
        try:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized, level=6)
                compression_ratio = len(compressed) / len(serialized)
                
                if compression_ratio < 0.8:  # Solo comprimir si ahorra al menos 20%
                    logger.debug(f"Compressed item: {len(serialized)} → {len(compressed)} bytes ({compression_ratio:.2%})")
                    return compressed
            
            return serialized
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _decompress_value(self, data: bytes, is_compressed: bool) -> Any:
        """Descomprimir valor"""
        try:
            if is_compressed:
                decompressed = zlib.decompress(data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    def _estimate_memory_usage(self) -> int:
        """Estimar uso actual de memoria"""
        total_size = 0
        for key, size in self._item_sizes.items():
            total_size += size
            # Overhead estimado por entrada
            total_size += len(key) * 2 + 100  # Overhead de estructuras Python
        return total_size
    
    def _evict_lru_items(self, target_size: Optional[int] = None):
        """Evitar items LRU hasta alcanzar el tamaño objetivo"""
        if not target_size:
            target_size = int(self.max_memory_bytes * 0.8)  # Evitar hasta 80% del límite
        
        current_memory = self._estimate_memory_usage()
        evicted_count = 0
        
        while (current_memory > target_size or len(self._cache) > self.max_size) and self._cache:
            # Evitar el item menos recientemente usado
            oldest_key = next(iter(self._cache))
            self._remove_item(oldest_key)
            current_memory = self._estimate_memory_usage()
            evicted_count += 1
        
        if evicted_count > 0:
            self.stats.evictions += evicted_count
            logger.info(f"🧹 Evicted {evicted_count} LRU items, memory: {current_memory / 1024 / 1024:.1f}MB")
    
    def _remove_item(self, key: str):
        """Remover item completamente del cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        if key in self._compressed_items:
            self._compressed_items.remove(key)
        if key in self._item_sizes:
            del self._item_sizes[key]
    
    def _is_expired(self, key: str) -> bool:
        """Verificar si un item ha expirado"""
        if not self.ttl_seconds or key not in self._access_times:
            return False
        
        age = time.time() - self._access_times[key]
        return age > self.ttl_seconds
    
    def get(self, key: Union[str, Any], default: Any = None) -> Any:
        """Obtener item del cache"""
        with self._lock:
            cache_key = self._compute_key(key)
            
            # Check if exists and not expired
            if cache_key in self._cache and not self._is_expired(cache_key):
                # Move to end (most recently used)
                value = self._cache.pop(cache_key)
                self._cache[cache_key] = value
                self._access_times[cache_key] = time.time()
                
                # Decompress if needed
                is_compressed = cache_key in self._compressed_items
                result = self._decompress_value(value, is_compressed)
                
                self.stats.hits += 1
                self.stats.total_operations += 1
                return result
            
            # Cache miss or expired
            if cache_key in self._cache:
                self._remove_item(cache_key)  # Remove expired item
            
            self.stats.misses += 1
            self.stats.total_operations += 1
            return default
    
    def put(self, key: Union[str, Any], value: Any, ttl_override: Optional[int] = None):
        """Almacenar item en el cache"""
        with self._lock:
            cache_key = self._compute_key(key)
            
            # Compress value
            compressed_value = self._compress_value(value)
            is_compressed = len(compressed_value) < len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)) * 0.8
            
            # Store value
            self._cache[cache_key] = compressed_value
            self._access_times[cache_key] = time.time()
            self._item_sizes[cache_key] = len(compressed_value)
            
            if is_compressed:
                self._compressed_items.add(cache_key)
            elif cache_key in self._compressed_items:
                self._compressed_items.remove(cache_key)
            
            # Check memory limits and evict if necessary
            current_memory = self._estimate_memory_usage()
            if current_memory > self.max_memory_bytes or len(self._cache) > self.max_size:
                self._evict_lru_items()
            
            # Update stats
            self.stats.memory_usage_bytes = self._estimate_memory_usage()
            compressed_count = len(self._compressed_items)
            total_count = len(self._cache)
            self.stats.compression_ratio = (compressed_count / total_count * 100) if total_count > 0 else 0
    
    def invalidate(self, key: Union[str, Any]):
        """Invalidar item específico"""
        with self._lock:
            cache_key = self._compute_key(key)
            if cache_key in self._cache:
                self._remove_item(cache_key)
                logger.debug(f"Invalidated cache key: {cache_key}")
    
    def clear(self):
        """Limpiar todo el cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._compressed_items.clear()
            self._item_sizes.clear()
            self.stats = CacheStats()
            logger.info("🧹 Cache cleared completely")
    
    def _start_cleanup_task(self):
        """Iniciar tarea de limpieza automática"""
        def cleanup_expired():
            while True:
                try:
                    time.sleep(300)  # Cleanup cada 5 minutos
                    self._cleanup_expired_items()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-cleanup")
        executor.submit(cleanup_expired)
    
    def _cleanup_expired_items(self):
        """Limpiar items expirados"""
        if not self.ttl_seconds:
            return
        
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, access_time in self._access_times.items():
                if current_time - access_time > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_item(key)
            
            if expired_keys:
                logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache items")

class EmbeddingCache(SmartLRUCache):
    """Cache especializado para embeddings con optimizaciones específicas"""
    
    def __init__(self, max_size: int = 5000, max_memory_mb: int = 1024):
        super().__init__(
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            compression_threshold=512,  # Comprimir embeddings más agresivamente
            ttl_seconds=7200  # TTL de 2 horas para embeddings
        )
    
    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Obtener embedding del cache"""
        cache_key = f"emb:{model_name}:{hashlib.md5(text.encode()).hexdigest()}"
        result = self.get(cache_key)
        
        if result is not None and isinstance(result, np.ndarray):
            return result
        return None
    
    def put_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Almacenar embedding en cache"""
        cache_key = f"emb:{model_name}:{hashlib.md5(text.encode()).hexdigest()}"
        self.put(cache_key, embedding)
    
    def warm_cache_async(self, texts: List[str], model_name: str, embedding_func: Callable):
        """Cache warming asíncrono para embeddings"""
        async def warm_embeddings():
            for text in texts:
                if self.get_embedding(text, model_name) is None:
                    try:
                        embedding = await asyncio.get_event_loop().run_in_executor(
                            None, embedding_func, text
                        )
                        self.put_embedding(text, model_name, embedding)
                    except Exception as e:
                        logger.warning(f"Cache warming failed for text: {e}")
        
        return asyncio.create_task(warm_embeddings())

class MemoryManager:
    """Gestor inteligente de memoria del sistema"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Obtener información de memoria del sistema"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_percent": memory.percent,
            "is_warning": memory.percent / 100 > self.warning_threshold,
            "is_critical": memory.percent / 100 > self.critical_threshold
        }
    
    def should_trigger_gc(self) -> bool:
        """Determinar si se debe ejecutar garbage collection"""
        memory_info = self.get_memory_info()
        return memory_info["is_warning"]
    
    def optimize_memory(self, caches: List[SmartLRUCache] = None):
        """Optimizar uso de memoria"""
        memory_info = self.get_memory_info()
        
        if memory_info["is_critical"]:
            logger.warning(f"🚨 Critical memory usage: {memory_info['usage_percent']:.1f}%")
            
            # Aggressive cache cleanup
            if caches:
                for cache in caches:
                    original_size = len(cache._cache)
                    target_size = int(cache.max_memory_bytes * 0.5)  # Reduce to 50%
                    cache._evict_lru_items(target_size)
                    logger.info(f"🧹 Aggressive cache cleanup: {original_size} → {len(cache._cache)} items")
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"🧹 Garbage collection freed {collected} objects")
            
        elif memory_info["is_warning"]:
            logger.info(f"⚠️ High memory usage: {memory_info['usage_percent']:.1f}%")
            
            # Gentle cache cleanup
            if caches:
                for cache in caches:
                    cache._cleanup_expired_items()
            
            # Light garbage collection
            gc.collect(0)  # Only collect generation 0

# Global instances
embedding_cache = EmbeddingCache()
memory_manager = MemoryManager()

def get_embedding_cache() -> EmbeddingCache:
    """Obtener instancia global del cache de embeddings"""
    return embedding_cache

def get_memory_manager() -> MemoryManager:
    """Obtener instancia global del gestor de memoria"""
    return memory_manager