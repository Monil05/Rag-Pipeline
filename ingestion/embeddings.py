import os

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
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
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
