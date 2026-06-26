import time

import pytest
import torch

from ingestion.qdrant_manager import connect_qdrant
from retrieval.agent import _get_graph, _get_router_model, run_agent_stream
from retrieval.answer_generator import _get_model
from retrieval.company_tool import _get_reranker


MAX_PIPELINE_LATENCY_SECONDS = 5.0

TEST_CASES = [
    "what products do you offer, how are they priced?",
    "what is the pricing policy/subscription terms for your products?",
]

_latencies = []


@pytest.fixture(scope="session", autouse=True)
def warm_local_resources():
    # Warm local resources (no API calls)
    _get_router_model()
    _get_graph()
    _get_model()
    connect_qdrant()

    # Warm the reranker with one local forward pass so the first
    # benchmark doesn't pay the CUDA/model initialization cost.
    tokenizer, model = _get_reranker()

    inputs = tokenizer(
        [("warmup query", "warmup document")],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.inference_mode():
        _ = model(**inputs).logits

    yield

    if _latencies:
        average_latency = sum(_latencies) / len(_latencies)
        print(f"\nAverage latency: {average_latency:.2f} seconds")


@pytest.mark.parametrize("query", TEST_CASES)
def test_warm_rag_latency(query):
    start = time.perf_counter()

    streamed_chunks = []
    for chunk in run_agent_stream(query, conversation_id=None):
        streamed_chunks.append(str(chunk))

    latency = time.perf_counter() - start
    _latencies.append(latency)

    final_answer = "".join(streamed_chunks).strip()

    print(f"\nQuery: {query}")
    print(f"Latency: {latency:.2f} seconds")

    assert final_answer, (
        f"Expected a non-empty final answer for query={query!r}"
    )

    assert latency <= MAX_PIPELINE_LATENCY_SECONDS, (
        f"Expected latency <= {MAX_PIPELINE_LATENCY_SECONDS:.2f}s "
        f"for query={query!r}, but got {latency:.2f}s"
    )