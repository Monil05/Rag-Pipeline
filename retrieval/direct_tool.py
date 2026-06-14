import logging
import re


logger = logging.getLogger(__name__)

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


def handle_direct_query(query):
    text = _normalize_query(query)
    if not text:
        return _failure_response()

    if _matches_any(_PASSWORD_PATTERNS, text):
        logger.info("Handled direct password query")
        return _success_response(
            "I can't retrieve passwords or other sensitive credentials.",
            standalone=True,
        )

    if text in _ACKNOWLEDGEMENT_PHRASES:
        logger.info("Handled direct acknowledgement query")
        return _success_response(
            "You're welcome! Let me know if there's anything else I can help with.",
            standalone=True,
        )

    is_pure_greeting = _is_pure_greeting(text)
    is_compound_greeting = not is_pure_greeting and _contains_greeting(text)

    if is_pure_greeting:
        logger.info("Handled pure greeting query")
        return _success_response(
            "Hello! How can I help you today?",
            standalone=True,
        )

    if is_compound_greeting:
        logger.info("Detected greeting in compound query")
        return _success_response(
            "Hello!",
            standalone=False,
        )

    if text in _HELP_PHRASES:
        logger.info("Handled direct help query")
        return _success_response(
            "I am an AI assistant that can answer questions using company documents and previous conversations.",
            standalone=True,
        )

    return _failure_response()


def _is_pure_greeting(text):
    tokens = text.split()
    if len(tokens) == 1 and tokens[0] in _GREETING_WORDS:
        return True
    for phrase in _GREETING_PHRASES:
        if text == phrase:
            return True
    return False


def _contains_greeting(text):
    tokens = text.split()
    if tokens and tokens[0] in _GREETING_WORDS:
        return True
    for phrase in _GREETING_PHRASES:
        if text.startswith(phrase):
            return True
    return False


def _normalize_query(query):
    if query is None:
        return ""
    if not isinstance(query, str):
        query = str(query)
    text = query.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matches_any(patterns, text):
    return any(pattern.search(text) for pattern in patterns)


def _success_response(content, standalone):
    return {
        "tool_name": "direct",
        "success": True,
        "content": content,
        "metadata": {"standalone": standalone},
    }


def _failure_response():
    return {
        "tool_name": "direct",
        "success": False,
        "content": None,
        "metadata": {"standalone": False},
    }

def is_standalone_direct_query(query):
    result = handle_direct_query(query)

    return (
        result["success"]
        and result["metadata"].get("standalone", False)
    )