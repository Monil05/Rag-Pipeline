Ingestion Pipeline

Overview

This project implements a document ingestion pipeline for a Retrieval-Augmented Generation (RAG) system.

The pipeline processes documents from a local `docs/` folder, extracts text, generates embeddings using Google's Gemini Embedding API, and stores vectorized chunks in Qdrant for semantic search and retrieval.


 Supported File Types

* PDF (`.pdf`)
* DOCX (`.docx`)
* TXT (`.txt`)


 Features

* Document ingestion from local folder
* PDF, DOCX, and TXT support
* Metadata extraction
* Text chunking
* Gemini embedding generation
* Qdrant vector storage
* Duplicate document detection
* Document deletion
* Full collection rebuild



## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
QDRANT_URL=<your_qdrant_url>
QDRANT_API_KEY=<your_qdrant_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
```

---

## Embedding Model

The ingestion pipeline uses:

```text
models/gemini-embedding-001
```

for document embeddings.

---

## Qdrant Collection

Collection Name:

```text
company_knowledge
```

Vector Size:

```text
3072
```

Distance Metric:

```text
Cosine Similarity
```

---

## Commands

### Add Documents

Processes all supported files inside the specified folder.

```bash
python manage_docs.py add docs/
```

Example Output:

```text
Added Successfully:
✓ leave_policy.pdf
✓ reimbursement.docx
```

---

### Delete Documents

Deletes the document from both:

* Qdrant collection
* Local docs folder

Single file:

```bash
python manage_docs.py delete leave_policy.pdf
```

Multiple files:

```bash
python manage_docs.py delete leave_policy.pdf reimbursement.docx
```

Example Output:

```text
Deleted Successfully:
✓ leave_policy.pdf
```

---

### Rebuild Collection

Deletes the existing collection and reprocesses all documents from `docs/`.

```bash
python manage_docs.py rebuild
```

Use this command when:

* Changing embedding dimensions
* Modifying chunking strategy
* Re-indexing all documents

---

## Ingestion Workflow

```text
Document
    ↓
Text Extraction
    ↓
Chunking
    ↓
Gemini Embeddings
    ↓
Qdrant Storage
```

Each chunk contains:

* Chunk text
* Document name
* Metadata
* Embedding vector

---

## Duplicate Detection

Before processing a document, the pipeline checks whether the document already exists in Qdrant.

If found, the file is skipped.

Example:

```text
Skipped (Duplicate):
• leave_policy.pdf
```

---

## Notes

* The `docs/` folder contains sample documents for testing.
* API keys should never be committed to Git.
