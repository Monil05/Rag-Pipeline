import logging
import re
from datetime import datetime, time, timedelta, timezone

import dateparser

from chat import message_manager
from chat.cache_manager import get_cache


logger = logging.getLogger(__name__)

_VAGUE_RECENT_MARKERS = {
    "earlier",
    "previously",
    "previous",
    "before",
    "recent",
    "recently",
    "last time",
}

_RELATIVE_AGO_PATTERN = re.compile(
    r"\b(\d+)\s+(minute|minutes|hour|hours|day|days|week|weeks)\s+ago\b",
    re.IGNORECASE,
)


def handle_history_query(query, conversation_id):
    text = _normalize_query(query)

    if not text or not conversation_id:
        return _failure_response()

    cache = get_cache(conversation_id)

    explicit_window = _parse_explicit_window(text)

    if explicit_window is not None:
        label, start, end = explicit_window

        cached_messages = _get_cached_messages_for_window(cache, start, end)

        if cached_messages is not None:
            transcript = _format_transcript(cached_messages)

            if transcript:
                logger.info(
                    "Handled explicit history query from cache for conversation_id=%s",
                    conversation_id,
                )
                return _success_response(
                    transcript,
                    source="cache",
                    time_window={
                        "label": label,
                        "start": start,
                        "end": end,
                    },
                )

        messages = message_manager.load_messages(
            conversation_id,
            start=start,
            end=end,
        )
        transcript = _format_transcript(messages)

        if transcript:
            logger.info(
                "Handled explicit history query from MongoDB for conversation_id=%s",
                conversation_id,
            )
            return _success_response(
                transcript,
                source="mongodb",
                time_window={
                    "label": label,
                    "start": start,
                    "end": end,
                },
            )

        return _failure_response()

    if _has_vague_recent_marker(text):
        cached_messages = list(cache)
        transcript = _format_transcript(cached_messages)

        if transcript:
            logger.info(
                "Handled vague history query from cache for conversation_id=%s",
                conversation_id,
            )
            return _success_response(
                transcript,
                source="cache",
                mode="recent",
            )

        messages = message_manager.load_messages(conversation_id)
        recent_messages = messages[-10:]
        transcript = _format_transcript(recent_messages)

        if transcript:
            logger.info(
                "Handled vague history query from MongoDB for conversation_id=%s",
                conversation_id,
            )
            return _success_response(
                transcript,
                source="mongodb",
                mode="recent",
            )

    return _failure_response()


def _parse_explicit_window(text):
    now = datetime.now(timezone.utc)

    settings = {
        "RELATIVE_BASE": now,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": "UTC",
        "TO_TIMEZONE": "UTC",
    }

    if "yesterday" in text:
        parsed = dateparser.parse("yesterday", settings=settings)
        if parsed is None:
            return None
        start = datetime.combine(parsed.date(), time.min, tzinfo=timezone.utc)
        end = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
        return "yesterday", start, end

    if "today" in text:
        parsed = dateparser.parse("today", settings=settings)
        if parsed is None:
            return None
        start = datetime.combine(parsed.date(), time.min, tzinfo=timezone.utc)
        return "today", start, now

    if "last week" in text:
        parsed = dateparser.parse("last week", settings=settings)
        if parsed is None:
            return None
        start = _ensure_utc(parsed)
        this_monday = datetime.combine(
            now.date() - timedelta(days=now.weekday()),
            time.min,
            tzinfo=timezone.utc,
        )
        return "last week", start, this_monday

    match = _RELATIVE_AGO_PATTERN.search(text)
    if match is not None:
        expression = match.group(0)
        parsed = dateparser.parse(expression, settings=settings)
        if parsed is None:
            return None
        start = _ensure_utc(parsed)
        return expression, start, now

    return None


def _has_vague_recent_marker(text):
    tokens = set(text.split())
    return any(marker in tokens for marker in _VAGUE_RECENT_MARKERS)


def _get_cached_messages_for_window(cache, start, end):
    if not cache:
        return None

    first_timestamp = cache[0].get("timestamp")
    if first_timestamp is None:
        return None

    first_timestamp = _ensure_utc(first_timestamp)
    if start < first_timestamp:
        return None

    return _filter_messages_by_window(cache, start, end)


def _filter_messages_by_window(messages, start, end):
    filtered_messages = []

    for message in messages:
        timestamp = message.get("timestamp")
        if timestamp is None:
            continue
        timestamp = _ensure_utc(timestamp)
        if start <= timestamp <= end:
            filtered_messages.append(message)

    return filtered_messages


def _format_transcript(messages):
    lines = []

    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()

        if not content:
            continue

        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "Message"

        lines.append(f"{prefix}: {content}")

    return "\n".join(lines).strip()


def _ensure_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_query(query):
    if query is None:
        return ""
    if not isinstance(query, str):
        query = str(query)
    text = query.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _success_response(content, **metadata):
    return {
        "tool_name": "history",
        "success": True,
        "content": content,
        "metadata": metadata,
    }


def _failure_response():
    return {
        "tool_name": "history",
        "success": False,
        "content": None,
        "metadata": {},
    }