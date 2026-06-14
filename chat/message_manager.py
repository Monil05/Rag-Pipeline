import logging
from datetime import datetime, timezone

from database.mongodb_manager import (
    get_conversations_collection,
    get_messages_collection,
)


logger = logging.getLogger(__name__)


ALLOWED_ROLES = {"user", "assistant"}


def insert_message(conversation_id, role, content):
    if role not in ALLOWED_ROLES:
        raise ValueError("role must be either 'user' or 'assistant'")

    conversations = get_conversations_collection()
    messages = get_messages_collection()

    if conversations.find_one({"conversation_id": conversation_id}) is None:
        raise RuntimeError(f"Conversation not found: {conversation_id}")

    now = datetime.now(timezone.utc)
    message = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "timestamp": now,
    }

    result = messages.insert_one(message)
    update_result = conversations.update_one(
        {"conversation_id": conversation_id},
        {"$set": {"updated_at": now}},
    )

    if update_result.matched_count == 0:
        messages.delete_one({"_id": result.inserted_id})
        raise RuntimeError(f"Conversation not found: {conversation_id}")

    logger.debug(
        "Inserted message %s for conversation %s",
        result.inserted_id,
        conversation_id,
    )
    return result.inserted_id


def load_messages(conversation_id, start=None, end=None):
    query = {
        "conversation_id": conversation_id,
    }

    if start is not None or end is not None:
        query["timestamp"] = {}

        if start is not None:
            query["timestamp"]["$gte"] = start

        if end is not None:
            query["timestamp"]["$lte"] = end

    messages = []

    for message in get_messages_collection().find(query).sort("timestamp", 1):
        message.pop("_id", None)
        messages.append(message)

    return messages


def delete_messages(conversation_id):
    result = get_messages_collection().delete_many(
        {"conversation_id": conversation_id}
    )
    logger.debug(
        "Deleted %s messages for conversation %s",
        result.deleted_count,
        conversation_id,
    )
    return result.deleted_count
