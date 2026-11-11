"""
src/cache.py
------------
Cache manager for the Intelligent QA System.

Connects to Redis (localhost:6379) and automatically falls back
to a thread-safe in-memory LRU/TTL cache when Redis is unavailable.
"""

import re
import json
import time
import hashlib
import threading
import logging
from collections import OrderedDict
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None

QUERY_TTL    = 3600    # 1 hour
CLASSIFY_TTL = 86400   # 24 hours
MEMORY_CACHE_SIZE = 2000


# ============================================================
# IN-MEMORY LRU/TTL FALLBACK CACHE
# ============================================================

class _MemoryCache:
    """Thread-safe LRU cache with TTL expiry — Redis fallback."""

    def __init__(self, max_size: int = MEMORY_CACHE_SIZE):
        self._store: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, expiry = self._store[key]
            if time.time() > expiry:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: str, ttl: int = QUERY_TTL):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time() + ttl)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def flush(self):
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "memory",
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total * 100, 2) if total > 0 else 0.0,
                "entries": len(self._store),
                "max_size": self._max_size,
            }


# ============================================================
# CACHE MANAGER
# ============================================================

class CacheManager:
    """
    Smart cache manager.
    Tries Redis; silently falls back to in-memory LRU/TTL cache.
    """

    def __init__(self):
        self._redis = None
        self._memory = _MemoryCache()
        self._using_redis = False
        self._redis_hits = 0
        self._redis_misses = 0
        self._connect_redis()

    def _connect_redis(self):
        try:
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_connect_timeout=0.05,
                socket_timeout=0.05,
                retry_on_timeout=False,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            self._using_redis = True
            logger.info("CacheManager: Redis connected at %s:%s", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            self._redis = None
            self._using_redis = False
            logger.warning("CacheManager: Redis unavailable (%s) — using in-memory fallback.", e)

    # --------------------------------------------------------
    # KEY HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = text.lower().strip()
        # Strip trailing punctuation marks like ?, ., !, ;, :
        cleaned = re.sub(r"[\s\?\.\!\,\;\:]+$", "", cleaned).strip()
        # Normalize operator spacing in arithmetic e.g. 2 + 2 or 2+2 -> standard format
        cleaned = re.sub(r"(\d+)\s*([+\-*/^=])\s*(\d+)", r"\1\2\3", cleaned)
        return " ".join(cleaned.split())

    @classmethod
    def _make_key(cls, prefix: str, text: str) -> str:
        digest = hashlib.sha256(cls._normalize(text).encode()).hexdigest()[:16]
        return f"qa_sys:v15:{prefix}:{digest}"

    # --------------------------------------------------------
    # LOW-LEVEL GET / SET
    # --------------------------------------------------------

    def _get(self, key: str) -> Optional[str]:
        if self._using_redis:
            try:
                value = self._redis.get(key)
                if value is not None:
                    self._redis_hits += 1
                else:
                    self._redis_misses += 1
                return value
            except Exception:
                self._using_redis = False
        return self._memory.get(key)

    def _set(self, key: str, value: str, ttl: int):
        if self._using_redis:
            try:
                self._redis.setex(key, ttl, value)
                return
            except Exception:
                self._using_redis = False
        self._memory.set(key, value, ttl)

    # --------------------------------------------------------
    # QUERY CACHE
    # --------------------------------------------------------

    def get_query(self, question: str) -> Optional[Dict]:
        key = self._make_key("query", question)
        raw = self._get(key)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return None

    def set_query(self, question: str, result: Dict, ttl: int = QUERY_TTL):
        key = self._make_key("query", question)
        self._set(key, json.dumps(result, ensure_ascii=False), ttl)

    # --------------------------------------------------------
    # CLASSIFICATION CACHE
    # --------------------------------------------------------

    def get_classification(self, question: str) -> Optional[Dict]:
        key = self._make_key("classify", question)
        raw = self._get(key)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return None

    def set_classification(self, question: str, result: Dict, ttl: int = CLASSIFY_TTL):
        key = self._make_key("classify", question)
        self._set(key, json.dumps(result, ensure_ascii=False), ttl)

    # --------------------------------------------------------
    # ADMIN
    # --------------------------------------------------------

    def clear(self):
        if self._using_redis:
            try:
                keys = list(self._redis.scan_iter("qa_sys:*"))
                if keys:
                    self._redis.delete(*keys)
            except Exception:
                pass
        self._memory.flush()
        self._redis_hits = 0
        self._redis_misses = 0
        logger.info("CacheManager: Cache cleared.")

    def stats(self) -> Dict[str, Any]:
        if self._using_redis:
            total = self._redis_hits + self._redis_misses
            return {
                "backend": "redis",
                "host": f"{REDIS_HOST}:{REDIS_PORT}",
                "hits": self._redis_hits,
                "misses": self._redis_misses,
                "hit_rate": round(self._redis_hits / total * 100, 2) if total > 0 else 0.0,
            }
        return self._memory.stats()

    @property
    def backend(self) -> str:
        return "redis" if self._using_redis else "memory"
