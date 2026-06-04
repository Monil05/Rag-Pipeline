def create_metadata(document_name, page, chunk_id, source_type):
    return {
        "document_name": document_name,
        "page": page,
        "chunk_id": chunk_id,
        "source_type": source_type,
    }
