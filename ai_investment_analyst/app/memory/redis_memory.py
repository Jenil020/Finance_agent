"""
Redis-backed conversation memory & semantic cache.
Uses Upstash free tier or local Redis.
"""
import json
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logging import logger

_redis_client = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.redis_url, decode_responses=True
        )
    return _redis_client


async def save_conversation(session_id: str, messages: list, ttl: int = 86400):
    """Persist conversation history to Redis (TTL: 24h default)."""
    client = await get_redis()
    key = f"session:{session_id}:messages"
    await client.set(key, json.dumps(messages), ex=ttl)
    logger.debug(f"[Redis] Saved {len(messages)} messages for session {session_id}")


async def load_conversation(session_id: str) -> list:
    """Load conversation history from Redis."""
    client = await get_redis()
    key = f"session:{session_id}:messages"
    data = await client.get(key)
    return json.loads(data) if data else []


async def semantic_cache_get(query_hash: str) -> str | None:
    """Check semantic cache for a previously answered query."""
    client = await get_redis()
    return await client.get(f"cache:{query_hash}")


async def semantic_cache_set(query_hash: str, response: str, ttl: int = 3600):
    """Store response in semantic cache (TTL: 1h)."""
    client = await get_redis()
    await client.set(f"cache:{query_hash}", response, ex=ttl)
