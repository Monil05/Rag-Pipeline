# Agentic RAG Assistant

## Project Overview

Agentic RAG Assistant is a Retrieval-Augmented Generation (RAG) application designed to answer user queries using an organization's internal documents while also supporting conversational memory. The system combines a vector database, chat history, and a lightweight routing agent to determine the most appropriate source of information before generating a response.

The project consists of two major pipelines:

* **Ingestion Pipeline** – Processes company documents, generates embeddings using Google's Gemini Embedding model, and stores them in Qdrant.
* **Retrieval Pipeline** – Routes incoming user queries, retrieves relevant information from Qdrant and/or previous conversations, assembles the context, and generates a final response using Gemini.

The application provides:

* Company document search
* Multi-turn conversation support
* Previous conversation retrieval
* Multilingual query support
* Admin document management through Streamlit
* Local vector database using Qdrant
* MongoDB-based conversation persistence

---

# Technology Stack

| Component       | Technology                                       |
| --------------- | ------------------------------------------------ |
| Frontend        | Streamlit                                        |
| LLM             | Google Gemini Flash                              |
| Embeddings      | Gemini Embedding (`models/gemini-embedding-001`) |
| Vector Database | Qdrant                                           |
| Chat Database   | MongoDB                                          |
| Reranker        | BAAI/bge-reranker-base                           |
| Agent Workflow  | LangGraph                                        |
| Language        | Python                                           |

---

# Project Structure

```
RAG/
│
├── chat/
├── database/
├── docs/
├── ingestion/
├── retrieval/
├── tests/
│
├── manage_docs.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .env
```

---

# Prerequisites

Before running the project, install the following software.

## Python

Python 3.11 or newer is recommended.

---

## Docker

Qdrant runs locally using Docker.

Download Docker Desktop from:

https://www.docker.com/products/docker-desktop/

After installation, ensure Docker Desktop is running before starting the Qdrant container.

---

## MongoDB

Install MongoDB locally or use MongoDB Atlas.

Update the MongoDB connection string in the `.env` file.

---

# Installation

Clone or download the project.

```
git clone <repository-url>

cd RAG
```

---

## Create Virtual Environment

Windows

```
python -m venv venv

venv\Scripts\activate
```

macOS / Linux

```
python -m venv venv

source venv/bin/activate
```

---

## Install Python Dependencies

```
pip install -r requirements.txt
```

---

## Install PyTorch

Install the CUDA version matching your GPU.

Example (CUDA 11.8):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If CUDA is unavailable, the reranker automatically runs on the CPU. No additional configuration is required, although response latency will be higher than GPU execution.

---

# Qdrant Setup

Pull the latest Qdrant image.

```
docker pull qdrant/qdrant
```

Create a persistent Docker volume.

```
docker volume create qdrant_storage
```

Start Qdrant.

Windows (PowerShell)

```
docker run -d `
-p 6333:6333 `
-v qdrant_storage:/qdrant/storage `
--name qdrant `
qdrant/qdrant
```

macOS / Linux

```
docker run -d \
-p 6333:6333 \
-v qdrant_storage:/qdrant/storage:z \
--name qdrant \
qdrant/qdrant
```

The local Qdrant server will be available at:

```
http://localhost:6333
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

QDRANT_URL=http://localhost:6333

MONGODB_URL=YOUR_MONGODB_CONNECTION_STRING
```

---

# Preparing Company Documents

Create a folder named `docs` if it does not already exist.

```
mkdir docs
```

Copy all supported company documents into this folder.

Supported file types:

* PDF
* DOCX
* TXT

---

# Running the Ingestion Pipeline

Process all documents inside the `docs/` folder.

```
python manage_docs.py add docs/
```

This command:

* extracts document text
* splits documents into chunks
* generates Gemini embeddings
* stores vectors inside Qdrant

---

# Starting the Application

Launch Streamlit.

```
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

# Ingestion CLI Commands

## Add Documents

```
python manage_docs.py add docs/
```

Adds all supported documents from the `docs/` directory to Qdrant.

Duplicate documents are skipped automatically.

---

## Delete Documents

Delete one document.

```
python manage_docs.py delete leave_policy.pdf
```

Delete multiple documents.

```
python manage_docs.py delete leave_policy.pdf reimbursement.docx travel_policy.pdf
```

This removes:

* the vectors from Qdrant
* the corresponding document from the `docs/` folder

---

## Rebuild Collection

```
python manage_docs.py rebuild
```

Deletes the existing Qdrant collection and rebuilds it from every supported document inside `docs/`.

Use this whenever:

* changing the embedding model
* changing chunk size
* changing chunk overlap
* changing vector dimensions
* changing any ingestion configuration that affects stored embeddings

---

# Running Tests

The project includes automated tests under the `tests/` directory.

Run all tests.

```
pytest
```

Run only router tests.

```
pytest tests/test_router.py
```

Run only retrieval tests.

```
pytest tests/test_retrieval.py
```

Run only latency benchmark.

```
pytest tests/test_latency.py -s
```

The latency test measures the complete Retrieval-Augmented Generation pipeline and verifies that the response time remains within the configured performance target.

---

# Features

* Agentic Retrieval-Augmented Generation
* LangGraph-based routing
* Company knowledge retrieval using Qdrant
* Conversation history retrieval using MongoDB
* Gemini Flash answer generation
* Gemini Embedding document indexing
* BGE neural reranking
* Streaming responses
* Multilingual support
* Document management through Streamlit
* Persistent conversations
* Automated latency and retrieval tests

---

# Notes

* Docker Desktop must be running before starting the application.
* The `docs/` folder is treated as the source of truth for all company documents.
* Qdrant stores vector representations of the documents and mirrors the contents of the `docs/` folder.
* MongoDB stores conversation metadata and chat history.
* If a CUDA-capable GPU is available, the reranker automatically uses it. Otherwise, it falls back to CPU execution.
* The application automatically creates the Qdrant collection during ingestion if it does not already exist.
