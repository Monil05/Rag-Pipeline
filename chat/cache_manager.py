from collections import deque
from datetime import datetime, timezone
import logging

from chat import message_manager

logger = logging.getLogger(__name__)

_CACHE_MAXLEN = 10
_VALID_ROLES = {"user", "assistant"}
_caches = {}


def _normalize_timestamp(timestamp):
    if timestamp is None:
        return datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def get_cache(conversation_id):
    cache = _caches.get(conversation_id)
    if cache is None:
        cache = deque(maxlen=_CACHE_MAXLEN)
        _caches[conversation_id] = cache
    return cache


def append_message(conversation_id, role, content, timestamp=None):
    if role not in _VALID_ROLES:
        raise ValueError(f"Invalid role: {role!r}. Expected one of: {sorted(_VALID_ROLES)}")

    cache = get_cache(conversation_id)
    cache.append({
        "role": role,
        "content": content,
        "timestamp": _normalize_timestamp(timestamp),
    })


def clear_cache(conversation_id):
    _caches.pop(conversation_id, None)


def rebuild_cache(conversation_id):
    messages = message_manager.load_messages(conversation_id)
    cache = deque(maxlen=_CACHE_MAXLEN)

    for message in messages[-_CACHE_MAXLEN:]:
        role = message.get("role")
        if role not in _VALID_ROLES:
            logger.warning("Skipping message with invalid role while rebuilding cache for conversation_id=%s", conversation_id)
            continue

        timestamp = _normalize_timestamp(message.get("timestamp"))
        cache.append({
            "role": role,
            "content": message.get("content", ""),
            "timestamp": timestamp,
        })

    _caches[conversation_id] = cache
    return cache


def clear_all_caches():
    _caches.clear()
