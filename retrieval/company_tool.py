import logging
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ingestion import qdrant_manager
from ingestion.embeddings import get_embedding


logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_SCORE_THRESHOLD = 0.3
QDRANT_RETRIEVAL_LIMIT = 10
RERANKED_RESULTS_LIMIT = 4

_reranker_tokenizer = None
_reranker_model = None
_reranker_lock = threading.Lock()


def handle_company_query(query, conversation_id=None):
    del conversation_id

    query_text = _normalize_query(query)
    if not query_text:
        return _failure_response()

    embedding = get_embedding(query_text, task_type="retrieval_query")
    client = qdrant_manager.connect_qdrant()

    retrieved_points = _retrieve_points(client, embedding)

    if not retrieved_points:
        logger.info("Company retrieval returned no Qdrant matches")
        return _failure_response()

    reranked_points = _rerank_points(query_text, retrieved_points)
    
    if not reranked_points:
        logger.info("Company retrieval returned no reranked matches above threshold")
        return _failure_response()

    return _success_response(reranked_points)


def _normalize_query(query):
    if query is None:
        return ""
    if not isinstance(query, str):
        query = str(query)
    return query.strip()


def _retrieve_points(client, embedding):
    try:
        response = client.query_points(
            collection_name=qdrant_manager.COLLECTION_NAME,
            query=embedding,
            limit=QDRANT_RETRIEVAL_LIMIT,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Qdrant retrieval failed: {exc}") from exc

    points = getattr(response, "points", None)
    if points is None:
        return []
    
    return list(points)


def _rerank_points(query_text, points):
    tokenizer, model = _get_reranker()

    valid_points = []
    passages = []

    for point in points:
        payload = point.payload or {}
        text = (payload.get("text") or "").strip()

        if not text:
            continue

        valid_points.append(point)
        passages.append(text)

    if not passages:
        return []

    queries = [query_text] * len(passages)

    inputs = tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits.view(-1)

        scores = torch.sigmoid(logits).detach().cpu().tolist()
    
    all_reranked = []
    for point, score in zip(valid_points, scores):
        all_reranked.append((score, point))

    all_reranked.sort(key=lambda item: item[0], reverse=True)

    filtered = [
        (score, point)
        for score, point in all_reranked
        if score >= RERANK_SCORE_THRESHOLD
    ]
    # Normal case
    if filtered:
        return filtered[:RERANKED_RESULTS_LIMIT]
    # Fallback
    if all_reranked:
        logger.info(
            "No chunks passed rerank threshold; returning top reranked chunks because retrieval succeeded."
        )
        return all_reranked[:RERANKED_RESULTS_LIMIT]

    return []


def _get_reranker():
    global _reranker_tokenizer
    global _reranker_model

    if _reranker_tokenizer is not None and _reranker_model is not None:
        return _reranker_tokenizer, _reranker_model

    with _reranker_lock:
        if _reranker_tokenizer is not None and _reranker_model is not None:
            return _reranker_tokenizer, _reranker_model

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for company_tool reranking, but no CUDA device is available"
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL_NAME
            )
            
        except Exception as exc:
            raise RuntimeError(f"Failed to load reranker model: {exc}") from exc

        _reranker_tokenizer = tokenizer
        _reranker_model = model.to("cuda").eval()
        logger.info("Loaded company reranker model on CUDA")

    return _reranker_tokenizer, _reranker_model


def _success_response(reranked_points):
    chunks = []
    sources = []
    seen_sources = set()

    for _, point in reranked_points:
        payload = point.payload or {}

        text = (payload.get("text") or "").strip()
        if not text:
            continue

        document_name = payload.get("document_name")
        page = payload.get("page")

        chunk = {
            "text": text,
            "document_name": document_name,
            "page": page,
        }
        chunks.append(chunk)

        source_key = (document_name, page)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(
                {
                    "document_name": document_name,
                    "page": page,
                }
            )

    if not chunks:
        return _failure_response()

    return {
        "tool_name": "company",
        "success": True,
        "content": chunks,
        "metadata": {
            "sources": sources,
        },
    }



def _failure_response():
    return {
        "tool_name": "company",
        "success": False,
        "content": None,
        "metadata": {},
    }
