from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.metadata import create_metadata


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_document(page_records, document_name, source_type):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    chunk_id = 0

    for record in page_records:
        for text in splitter.split_text(record["text"]):
            clean_text = text.strip()
            if not clean_text:
                continue

            metadata = create_metadata(
                document_name=document_name,
                page=record["page"],
                chunk_id=chunk_id,
                source_type=source_type,
            )
            chunks.append({"text": clean_text, "metadata": metadata})
            chunk_id += 1

    if not chunks:
        raise RuntimeError("Could not create chunks from document text")

    return chunks
