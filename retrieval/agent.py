import logging
import re

from retrieval.answer_generator import generate_answer
from retrieval.company_tool import handle_company_query
from retrieval.context_assembly import assemble_context
from retrieval.direct_tool import handle_direct_query, is_standalone_direct_query
from retrieval.history_tool import handle_history_query


logger = logging.getLogger(__name__)

DIRECT_TOOL = "direct"
HISTORY_TOOL = "history"
COMPANY_TOOL = "company"

_GREETING_WORDS = {"hi", "hello", "hey"}

_GREETING_PHRASES = {
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
}

_HELP_PHRASES = {
    "help",
    "what can you do",
    "how can you help",
    "who are you",
    "what do you do",
    "tell me what you can do",
}

_ACKNOWLEDGEMENT_PHRASES = {
    "thanks",
    "thank you",
    "thx",
    "thanks a lot",
    "thank you very much",
}

_PASSWORD_PATTERNS = (
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"\bpasscode\b", re.IGNORECASE),
    re.compile(r"\bcredentials?\b", re.IGNORECASE),
)

_HISTORY_MARKERS = {
    "today",
    "yesterday",
    "earlier",
    "previously",
    "previous",
    "before",
    "recently",
    "last",
    "talk about",
}

_FOLLOWUP_PHRASES = {
    "company",
    "product",
    "service",
    "price",
    "cost",
    "products",
    "services",
    "prices",
    "costs",
    "offer",
    "what about that",
    "tell me more",
    "explain further",
    "how does that compare",
    "and carry forward",
    "elaborate",
    "clarify",
}

_COMPARISON_MARKERS = {
    "compare",
    "comparison",
    "difference",
    "different",
    "vs",
    "contrast",
    "against",
    "current",
    "existing",
    "summarize",
    "summary",
    "discuss",
}

def run_agent(query, conversation_id=None):
    query_text = _normalize_query(query)
    if not query_text:
        raise ValueError("query must not be empty")

    selected_tools = set()

    if detect_direct_query(query_text):
        selected_tools.add(DIRECT_TOOL)

    if detect_history_query(query_text):
        selected_tools.add(HISTORY_TOOL)

    if detect_followup_query(query_text):
        selected_tools.add(COMPANY_TOOL)

    if detect_company_query(query_text):
        selected_tools.add(COMPANY_TOOL)

    if (DIRECT_TOOL in selected_tools and not is_standalone_direct_query(query_text)):
        selected_tools.add(COMPANY_TOOL)

    if not selected_tools:
        selected_tools.add(COMPANY_TOOL)

    logger.info("Selected retrieval tools: %s", sorted(selected_tools))

    direct_tool_output = None
    history_tool_output = None
    company_tool_output = None

    if DIRECT_TOOL in selected_tools:
        direct_tool_output = handle_direct_query(query_text)

    if HISTORY_TOOL in selected_tools:
        history_tool_output = handle_history_query(query_text, conversation_id)

    if COMPANY_TOOL in selected_tools:
        company_tool_output = handle_company_query(query_text, conversation_id)

    assembled_context = assemble_context(
        query=query_text,
        direct_tool_output=direct_tool_output,
        history_tool_output=history_tool_output,
        company_tool_output=company_tool_output,
        conversation_id=conversation_id,
    )

    return generate_answer(assembled_context)


def detect_direct_query(query):
    text = _normalize_for_detection(query)
    if not text:
        return False

    if _matches_any(_PASSWORD_PATTERNS, text):
        return True

    if text in _ACKNOWLEDGEMENT_PHRASES:
        return True

    if text in _HELP_PHRASES:
        return True

    tokens = text.split()
    if len(tokens) == 1 and tokens[0] in _GREETING_WORDS:
        return True

    for phrase in _GREETING_PHRASES:
        if text == phrase:
            return True
        if text.startswith(phrase):
            return True

    if tokens and tokens[0] in _GREETING_WORDS:
        return True

    return False


def detect_history_query(query):
    text = _normalize_for_detection(query)
    if not text:
        return False

    tokens = set(text.split())
    if any(marker in tokens for marker in _HISTORY_MARKERS):
        return True

    return "last week" in text or re.search(r"\bago\b", text) is not None


def detect_followup_query(query):
    text = _normalize_for_detection(query)
    if not text:
        return False

    return any(phrase in text for phrase in _FOLLOWUP_PHRASES)

def detect_company_query(query):
    text = _normalize_for_detection(query)
    if not text:
        return False

    tokens = set(text.split())

    return any(marker in tokens for marker in _COMPARISON_MARKERS)

def _normalize_query(query):
    if query is None:
        return ""
    if not isinstance(query, str):
        query = str(query)
    return query.strip()


def _normalize_for_detection(query):
    text = _normalize_query(query).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matches_any(patterns, text):
    return any(pattern.search(text) for pattern in patterns)
