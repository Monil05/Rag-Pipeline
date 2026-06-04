import os

from dotenv import load_dotenv
import google.generativeai as genai


EMBEDDING_MODEL = "models/gemini-embedding-001"


class ConfigurationError(RuntimeError):
    pass


def get_embedding(text):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ConfigurationError("Missing required environment variable: GEMINI_API_KEY")

    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini embedding API error: {exc}") from exc

    embedding = result.get("embedding") if isinstance(result, dict) else None
    if not embedding:
        raise RuntimeError("Gemini embedding API error: empty embedding returned")

    return embedding
