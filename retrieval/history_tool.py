import logging
from datetime import timezone

from chat import message_manager
from chat.cache_manager import get_cache


logger = logging.getLogger(__name__)


def handle_history_query(
    conversation_id,
    exact_range=False,
    start=None,
    end=None,
):
    if not conversation_id:
        return _failure_response()

    cache = get_cache(conversation_id)

    # Vague history query
    if not exact_range:
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
            )

        return _failure_response()

    # Exact time range query
    cached_messages = _get_cached_messages_for_window(
        cache,
        start,
        end,
    )

    if cached_messages is not None:
        transcript = _format_transcript(cached_messages)

        if transcript:
            logger.info(
                "Handled exact history query from cache for conversation_id=%s",
                conversation_id,
            )

            return _success_response(
                transcript,
                source="cache",
                time_window={
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
            "Handled exact history query from MongoDB for conversation_id=%s",
            conversation_id,
        )

        return _success_response(
            transcript,
            source="mongodb",
            time_window={
                "start": start,
                "end": end,
            },
        )

    return _failure_response()


def _get_cached_messages_for_window(cache, start, end):
    if not cache:
        return None

    if start is None or end is None:
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