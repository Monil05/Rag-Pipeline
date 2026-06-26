import pytest

from retrieval.company_tool import handle_company_query


TEST_CASES = [
    {
        "query": "What does ABC do as a company?",
        "expected_document": "company_overview_ABC_sample_clean_v2.pdf",
    },
    {
        "query": "What products do you offer, and how are they priced?",
        "expected_document": "product_catalog_with_pricing_ABC_detailed.pdf",
    },
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_retrieval_returns_expected_source_document(case):
    query = case["query"]
    expected_document = case["expected_document"]

    result = handle_company_query(query)

    assert result["success"] is True, (
        f"expected success=True for query={query!r}, "
        f"but got success={result['success']!r}"
    )

    assert result["content"], (
        f"expected retrieval content for query={query!r}, "
        f"but got empty content: {result['content']!r}"
    )

    sources = result.get("metadata", {}).get("sources", [])
    retrieved_documents = [source.get("document_name") for source in sources]

    assert expected_document in retrieved_documents, (
        f"expected document={expected_document!r} for query={query!r}, "
        f"but retrieved documents were {retrieved_documents!r}"
    )
