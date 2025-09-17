#!/usr/bin/env python3
"""
Parsing Cache System for L2SCA Performance Optimization
Caches parsed results to avoid recomputation on essay retries
"""

import hashlib
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CachedParseResult:
    """Structure for cached parsing results"""
    essay_hash: str
    sentences: list
    t_units: list
    clauses: list
    complex_structures: dict
    word_count: int
    cached_at: float
    ttl_seconds: int = 3600  # 1 hour default

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - self.cached_at) > self.ttl_seconds

class L2SCAParsingCache:
    """
    High-performance caching system for L2SCA parsing results
    
    Features:
    - Memory-based cache with TTL
    - Disk persistence for server restarts
    - Thread-safe operations
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, cache_dir: str = "parsing_cache", max_memory_entries: int = 1000):
        """Initialize parsing cache"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.memory_cache: Dict[str, CachedParseResult] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persistent cache
        self._load_persistent_cache()
        
        # Cleanup timer
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        logger.info(f"✅ L2SCA Parsing Cache initialized: {len(self.memory_cache)} entries loaded")
    
    def _generate_essay_hash(self, essay_text: str) -> str:
        """Generate unique hash for essay text"""
        # Normalize text for consistent hashing
        normalized = ' '.join(essay_text.strip().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    def _load_persistent_cache(self):
        """Load cache from disk on startup"""
        cache_file = self.cache_dir / "l2sca_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to CachedParseResult objects
                for essay_hash, cached_data in data.items():
                    try:
                        cached_result = CachedParseResult(**cached_data)
                        if not cached_result.is_expired():
                            self.memory_cache[essay_hash] = cached_result
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load cache entry {essay_hash}: {e}")
                
                logger.info(f"✅ Loaded {len(self.memory_cache)} valid cache entries from disk")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self):
        """Save cache to disk"""
        cache_file = self.cache_dir / "l2sca_cache.json"
        
        try:
            # Convert to serializable format
            data = {}
            for essay_hash, cached_result in self.memory_cache.items():
                if not cached_result.is_expired():
                    data[essay_hash] = asdict(cached_result)
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.rename(cache_file)
            logger.debug(f"✅ Saved {len(data)} cache entries to disk")
            
        except Exception as e:
            logger.error(f"❌ Failed to save persistent cache: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from memory cache"""
        current_time = time.time()
        
        # Only cleanup every 5 minutes
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            expired_keys = [
                key for key, cached_result in self.memory_cache.items()
                if cached_result.is_expired()
            ]
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            # Limit memory cache size
            if len(self.memory_cache) > self.max_memory_entries:
                # Remove oldest entries
                sorted_items = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].cached_at
                )
                
                excess_count = len(self.memory_cache) - self.max_memory_entries
                for key, _ in sorted_items[:excess_count]:
                    del self.memory_cache[key]
            
            self._last_cleanup = current_time
            
            if expired_keys:
                logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cached_parsing(self, essay_text: str) -> Optional[CachedParseResult]:
        """Get cached parsing results for essay"""
        essay_hash = self._generate_essay_hash(essay_text)
        
        with self._lock:
            # Cleanup expired entries periodically
            self._cleanup_expired_entries()
            
            cached_result = self.memory_cache.get(essay_hash)
            
            if cached_result and not cached_result.is_expired():
                logger.debug(f"✅ Cache hit for essay {essay_hash}")
                return cached_result
            elif cached_result:
                # Remove expired entry
                del self.memory_cache[essay_hash]
                logger.debug(f"🗑️ Removed expired cache entry {essay_hash}")
        
        return None
    
    def cache_parsing_results(self, essay_text: str, sentences: list, t_units: list, 
                            clauses: list, complex_structures: dict, 
                            ttl_seconds: int = 3600) -> bool:
        """Cache parsing results for essay"""
        essay_hash = self._generate_essay_hash(essay_text)
        word_count = len(essay_text.split())
        
        cached_result = CachedParseResult(
            essay_hash=essay_hash,
            sentences=sentences,
            t_units=t_units,
            clauses=clauses,
            complex_structures=complex_structures,
            word_count=word_count,
            cached_at=time.time(),
            ttl_seconds=ttl_seconds
        )
        
        with self._lock:
            self.memory_cache[essay_hash] = cached_result
            logger.debug(f"✅ Cached parsing results for essay {essay_hash}")
            
            # Save to disk periodically
            if len(self.memory_cache) % 10 == 0:  # Every 10 new entries
                self._save_persistent_cache()
        
        return True
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            total_entries = len(self.memory_cache)
            expired_entries = sum(1 for result in self.memory_cache.values() if result.is_expired())
            
            # Calculate size distribution
            word_counts = [result.word_count for result in self.memory_cache.values()]
            avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
            
            return {
                'total_entries': total_entries,
                'active_entries': total_entries - expired_entries,
                'expired_entries': expired_entries,
                'max_memory_entries': self.max_memory_entries,
                'memory_usage_percent': (total_entries / self.max_memory_entries) * 100,
                'average_word_count': round(avg_word_count),
                'cache_directory': str(self.cache_dir),
                'last_cleanup': self._last_cleanup
            }
    
    def clear_cache(self, keep_persistent: bool = False) -> bool:
        """Clear all cache entries"""
        with self._lock:
            try:
                self.memory_cache.clear()
                
                if not keep_persistent:
                    cache_file = self.cache_dir / "l2sca_cache.json"
                    if cache_file.exists():
                        cache_file.unlink()
                
                logger.info("✅ Cache cleared successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to clear cache: {e}")
                return False
    
    def optimize_cache_for_essay_length(self, typical_word_count: int = 250):
        """Optimize cache settings for typical essay length"""
        # Adjust TTL based on essay length
        if typical_word_count <= 200:
            default_ttl = 7200  # 2 hours for shorter essays
        elif typical_word_count <= 300:
            default_ttl = 3600  # 1 hour for standard essays
        else:
            default_ttl = 1800  # 30 minutes for longer essays
        
        # Update cache entries with optimized TTL
        with self._lock:
            for cached_result in self.memory_cache.values():
                if cached_result.word_count <= typical_word_count + 50:
                    cached_result.ttl_seconds = default_ttl
        
        logger.info(f"✅ Cache optimized for essays ~{typical_word_count} words (TTL: {default_ttl}s)")

# Global cache instance
_global_parsing_cache = None

def get_parsing_cache() -> L2SCAParsingCache:
    """Get or create global parsing cache instance"""
    global _global_parsing_cache
    if _global_parsing_cache is None:
        _global_parsing_cache = L2SCAParsingCache()
    return _global_parsing_cache