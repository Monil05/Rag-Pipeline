import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConfigurationError as PyMongoConfigurationError


logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    pass


_client = None
_database = None
_conversations_collection = None
_messages_collection = None


def _initialize_connection():
    global _client, _database, _conversations_collection, _messages_collection

    if _client is not None:
        return

    load_dotenv()
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        raise ConfigurationError("Missing required environment variable: MONGODB_URL")

    client = MongoClient(mongo_url)

    try:
        database = client.get_default_database()
    except PyMongoConfigurationError as exc:
        raise ConfigurationError(
            "MONGODB_URL must include a database name in the URI path"
        ) from exc

    conversations_collection = database["conversations"]
    messages_collection = database["messages"]

    conversations_collection.create_index([("updated_at", -1)])
    messages_collection.create_index([("conversation_id", 1), ("timestamp", 1)])

    _client = client
    _database = database
    _conversations_collection = conversations_collection
    _messages_collection = messages_collection

    logger.debug("MongoDB connection initialized")


def get_mongo_client():
    _initialize_connection()
    return _client


def get_conversations_collection() -> Collection:
    _initialize_connection()
    return _conversations_collection


def get_messages_collection() -> Collection:
    _initialize_connection()
    return _messages_collection
