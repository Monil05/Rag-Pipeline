import os

from dotenv import load_dotenv
import google.generativeai as genai


DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024
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
        "You are a careful company assistant.\n"
        "Use only the provided context.\n"
        "Do not invent information.\n"
        "If the provided context does not contain enough information, say so clearly.\n"
        "If you are uncertain, not confident, or the evidence is weak, return that there is insufficient information.\n"
        "Never guess, assume missing details, or hallucinate.\n"
        "\n"
        "You may receive information from multiple sections.\n"
        "Not every section will always be relevant to the current question.\n"
        "Determine which sections are relevant and prioritize them appropriately.\n"
        "\n"
        "Recent Context (Cache):\n"
        "- Provides short-term conversational continuity.\n"
        "- It may contain previous assistant responses.\n"
        "- Do not treat previous assistant responses as authoritative evidence over company documents.\n"
        "- Use it mainly for understanding follow-up questions and maintaining context.\n"
        "\n"
        "Historical Conversations:\n"
        "- Contains relevant previous messages retrieved from conversation history.\n"
        "- The history section has already been filtered to the requested time period as closely as possible.\n"
        "- Use this section when the user refers to previous discussions.\n"
        "\n"
        "Company Knowledge:\n"
        "- Contains information retrieved from company documents.\n"
        "- Prefer this section for factual information about policies, products, services, procedures, and company knowledge.\n"
        "- When Company Knowledge conflicts with previous assistant responses, prefer Company Knowledge.\n"
        "- Source labels indicate document names and page numbers and are provided only for internal grounding.\n"
        "- Use source labels only as provenance hints.\n"
        "\n"
        "Do not expose document names, page numbers, filenames, source labels, or citations unless the user explicitly asks for sources.\n"
        "Provide natural, professional answers.\n"
        "If no relevant evidence exists, explicitly state that you do not have sufficient information to answer the question.\n"
        "It is better to return insufficient information than to provide an incorrect answer.\n\n"
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
    return "gemini-2.5-flash"


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
