import os
import time
from dotenv import load_dotenv
import google.generativeai as genai


EMBEDDING_MODEL = "models/gemini-embedding-001"


class ConfigurationError(RuntimeError):
    pass


load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise ConfigurationError(
        "Missing required environment variable: GEMINI_API_KEY"
    )

genai.configure(api_key=_API_KEY)


def get_embedding(text, task_type="retrieval_document"):
    try:
        t0 = time.perf_counter()
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        print(f"[TIMER] Gemini Embedding API: {time.perf_counter() - t0:.3f}s")
    except Exception as exc:
        raise RuntimeError(
            f"Gemini embedding API error: {exc}"
        ) from exc

    embedding = result.get("embedding") if isinstance(result, dict) else None
    if not embedding:
        raise RuntimeError(
            "Gemini embedding API error: empty embedding returned"
        )

    return embedding
