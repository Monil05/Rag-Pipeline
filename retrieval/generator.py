import google.generativeai as genai
import logging
import time
logger = logging.getLogger(__name__)


from retrieval.answer_generator import (
    _build_prompt,
    _get_model,
    _normalize_context,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    INSUFFICIENT_INFORMATION_RESPONSE,
)


def stream_answer(assembled_context=None,direct_answer=None,):
    # Direct answer path
    if direct_answer is not None:
        direct_answer = str(direct_answer).strip()

        if not direct_answer:
            return

        try:
            for word in direct_answer.split():
                yield word + " "
        except Exception:
            logger.exception("Direct response streaming failed")
            yield "I encountered an error while generating the response."
        return

    # Second LLM path
    context = _normalize_context(assembled_context)

    if not context["has_successful_tool_output"]:
        yield INSUFFICIENT_INFORMATION_RESPONSE
        return

    prompt = _build_prompt(context)

    model = _get_model()

    try:
        t0 = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            stream=True,
        )
        print(f"[TIMER] Gemini answer generation (2nd call): {time.perf_counter() - t0:.3f}s")
    except Exception as exc:
        raise RuntimeError(f"Gemini answer generation failed: {exc}") from exc
    
    try:
        for chunk in response:
            text = getattr(chunk, "text", None)

            if text:
                yield str(text)
    except Exception:
        logger.exception("Streaming failed while iterating Gemini response")
        yield "I encountered an error while generating the response."