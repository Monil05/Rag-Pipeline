from pathlib import Path

import fitz
from docx import Document


def extract_pdf(path):
    records = []

    try:
        with fitz.open(path) as document:
            for page_index, page in enumerate(document, start=1):
                text = page.get_text().strip()
                if text:
                    records.append({"text": text, "page": page_index})
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from PDF: {exc}") from exc

    if not records:
        raise RuntimeError("Could not extract text from PDF")

    return records


def extract_docx(path):
    try:
        document = Document(path)
        text = "\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from DOCX: {exc}") from exc

    if not text:
        raise RuntimeError("Could not extract text from DOCX")

    return [{"text": text, "page": 1}]


def extract_txt(path):
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        try:
            text = Path(path).read_text(encoding="utf-8-sig").strip()
        except Exception as exc:
            raise RuntimeError(f"Could not extract text from TXT: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from TXT: {exc}") from exc

    if not text:
        raise RuntimeError("Could not extract text from TXT")

    return [{"text": text, "page": 1}]
