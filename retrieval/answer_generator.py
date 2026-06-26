import os

from dotenv import load_dotenv
import google.generativeai as genai


DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048
INSUFFICIENT_INFORMATION_RESPONSE = (
    "I do not have sufficient information to answer that question.\n\n"
    "No relevant information was found in previous conversations or the company knowledge base."
)
_model = None

def generate_answer(assembled_context):
    context = _normalize_context(assembled_context)

    if not context["has_successful_tool_output"]:
        return INSUFFICIENT_INFORMATION_RESPONSE

    prompt = _build_prompt(context)
    response_text = _call_gemini(prompt)
    if response_text:
        return response_text

    return INSUFFICIENT_INFORMATION_RESPONSE


def _normalize_context(assembled_context):
    if not isinstance(assembled_context, dict):
        raise TypeError("assembled_context must be a dict returned by context_assembly.assemble_context")

    prompt = assembled_context.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("assembled_context must include a non-empty prompt")

    return {
        "prompt": prompt.strip(),
        "has_successful_tool_output": bool(
            assembled_context.get("has_successful_tool_output")
        ),
    }


def _build_prompt(context):
    return (
        "You are a company assistant.\n"
        "Use only the provided context.\n"
        "Do not invent, assume, or guess information.\n"
        "If the context is insufficient, clearly say so.\n"
        "Prefer Company Knowledge for factual answers.\n"
        "The history section has already been filtered to the requested time period as closely as possible.\n"
        "Use conversation history and cache for context and follow-ups.\n"
        "Do not expose document names, filenames, source labels, or page numbers unless explicitly requested.\n"
        "Answer naturally in the user's language.\n\n"
        f"{context['prompt']}"
    )


def _call_gemini(prompt):
    model = _get_model()

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini answer generation failed: {exc}") from exc

    return _extract_text(response)

def _get_model():
    global _model

    if _model is not None:
        return _model

    api_key = _load_api_key()
    genai.configure(api_key=api_key)

    try:
        _model = genai.GenerativeModel(
            model_name=_get_model_name(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Gemini model: {exc}"
        ) from exc

    return _model

def _load_api_key():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")
    return api_key


def _get_model_name():
    return "gemini-3.1-flash-lite"
# 3.5 flash

def _extract_text(response):
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text).strip()

    return ""
