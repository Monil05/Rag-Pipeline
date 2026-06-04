import os
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models


COLLECTION_NAME = "company_knowledge"
VECTOR_SIZE = 3072


class ConfigurationError(RuntimeError):
    pass


def connect_qdrant():
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ConfigurationError("Missing required environment variable: QDRANT_URL")

    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=qdrant_url, api_key=api_key)


def create_collection_if_not_exists(client):
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def document_exists(client, document_name):
    create_collection_if_not_exists(client)
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=_document_filter(document_name),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(records)


def insert_chunks(client, chunks):
    if not chunks:
        raise RuntimeError("Qdrant insertion failed: no chunks to insert")

    create_collection_if_not_exists(client)
    points = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        payload = {"text": chunk["text"], **metadata}
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=chunk["embedding"],
                payload=payload,
            )
        )

    try:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as exc:
        raise RuntimeError(f"Qdrant insertion failed: {exc}") from exc


def delete_document(client, document_name):
    create_collection_if_not_exists(client)
    if not document_exists(client, document_name):
        return False

    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=_document_filter(document_name),
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Qdrant deletion failed: {exc}") from exc

    return True


def delete_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)


def _document_filter(document_name):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_name",
                match=models.MatchValue(value=document_name),
            )
        ]
    )
