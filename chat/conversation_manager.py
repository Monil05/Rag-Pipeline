import logging
from datetime import datetime, timezone
from uuid import uuid4

from database.mongodb_manager import get_conversations_collection


logger = logging.getLogger(__name__)


def create_conversation(title):
    conversations = get_conversations_collection()
    now = datetime.now(timezone.utc)
    conversation = {
        "conversation_id": str(uuid4()),
        "title": title,
        "created_at": now,
        "updated_at": now,
    }

    conversations.insert_one(conversation)
    logger.debug("Created conversation %s", conversation["conversation_id"])
    return conversation


def list_conversations():
    conversations = get_conversations_collection()
    items = []

    for conversation in conversations.find().sort("updated_at", -1):
        conversation.pop("_id", None)
        items.append(conversation)

    return items


def load_conversation_metadata(conversation_id):
    conversation = get_conversations_collection().find_one(
        {"conversation_id": conversation_id}
    )
    if conversation is None:
        return None

    conversation.pop("_id", None)
    return conversation


def rename_conversation(conversation_id, title):
    conversations = get_conversations_collection()
    now = datetime.now(timezone.utc)
    result = conversations.update_one(
        {"conversation_id": conversation_id},
        {"$set": {"title": title, "updated_at": now}},
    )

    if result.matched_count == 0:
        raise RuntimeError(f"Conversation not found: {conversation_id}")

    logger.debug("Renamed conversation %s", conversation_id)


def delete_conversation(conversation_id):
    result = get_conversations_collection().delete_one(
        {"conversation_id": conversation_id}
    )
    if result.deleted_count == 0:
        return False

    logger.debug("Deleted conversation %s", conversation_id)
    return True
