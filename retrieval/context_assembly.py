from chat.cache_manager import get_cache

HISTORY_NOTE = (
    "The following conversation history has already been filtered to match the "
    "user's requested time period as closely as possible."
)


def assemble_context(
    original_query,
    normalized_query="",
    direct_tool_output=None,
    history_tool_output=None,
    company_tool_output=None,
    conversation_id=None,
):
    original_query_text = _normalize_query(original_query)
    if not original_query_text:
        raise ValueError("original_query must not be empty")

    normalized_query_text = _normalize_query(normalized_query)

    sections = []

    history_from_cache = (
        isinstance(history_tool_output, dict)
        and history_tool_output.get("success") is True
        and history_tool_output.get("metadata", {}).get("source") == "cache"
    )

    cache_section = _format_cache_context(conversation_id)

    if not history_from_cache:
        sections.append(
            _build_section(
                "Recent Context (Cache)",
                cache_section,
            )
        )

    direct_content = _extract_successful_content(direct_tool_output)
    if direct_content:
        sections.append(_build_section("Direct Response", direct_content))

    history_content = _format_history_context(history_tool_output)
    if history_content:
        sections.append(
            _build_section(
                "Historical Conversations",
                history_content,
            )
        )

    if normalized_query_text:
        sections.append(
            _build_section(
                "Normalized Retrieval Query",
                normalized_query_text,
            )
        )

    company_content = _format_company_context(company_tool_output)
    if company_content:
        sections.append(
            _build_section(
                "Company Knowledge",
                company_content,
            )
        )

    sections.append(
        _build_section(
            "Original User Question",
            original_query_text,
        )
    )

    prompt = "\n\n---\n\n".join(sections)

    return {
        "prompt": prompt,
        "original_query": original_query_text,
        "normalized_query": normalized_query_text,
        "has_successful_tool_output": bool(
            direct_content or history_content or company_content
        ),
        "sections": {
            "recent_cache": cache_section,
            "direct": direct_content,
            "history": history_content,
            "normalized_query": normalized_query_text,
            "company": company_content,
            "original_query": original_query_text,
        },
    }


def _build_section(title, body):
    return f"{title}\n\n{body.strip()}"


def _normalize_query(query):
    if query is None:
        return ""
    if not isinstance(query, str):
        query = str(query)
    return query.strip()


def _format_cache_context(conversation_id):
    if not conversation_id:
        return "No recent cache context available."

    cache = list(get_cache(conversation_id))
    if not cache:
        return "No recent cache context available."

    lines = []

    for message in cache:
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

    if not lines:
        return "No recent cache context available."

    return "\n".join(lines)


def _extract_successful_content(tool_output):
    if not _is_successful_output(tool_output):
        return ""

    content = tool_output.get("content")

    if not content:
        return ""

    return str(content).strip()


def _format_history_context(history_tool_output):
    content = _extract_successful_content(history_tool_output)

    if not content:
        return ""

    return f"{HISTORY_NOTE}\n\n{content}"


def _format_company_context(company_tool_output):
    if not _is_successful_output(company_tool_output):
        return ""

    chunks = company_tool_output.get("content")

    if not chunks:
        return ""

    blocks = []

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        text = (chunk.get("text") or "").strip()

        if not text:
            continue

        document_name = str(
            chunk.get("document_name") or "Unknown document"
        ).strip()

        page = chunk.get("page")

        if page is None:
            source_label = f"Source: {document_name}"
        else:
            source_label = f"Source: {document_name}, Page {page}"

        blocks.append(
            f"{source_label}\n\n{text}"
        )

    if not blocks:
        return ""

    separator = "\n\n" + "-" * 50 + "\n\n"

    return separator.join(blocks)


def _is_successful_output(tool_output):
    return (
        isinstance(tool_output, dict)
        and tool_output.get("success") is True
        and tool_output.get("content") is not None
    )
