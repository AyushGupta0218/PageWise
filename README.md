# Pagewise

**Ask questions across your PDFs and get answers grounded in cited passages.**

Pagewise is a document question-answering app built with FastAPI, SQLite, and OpenAI. Upload PDFs, select the documents to search, and ask questions in a simple chat interface. Each answer is grounded in retrieved passages and accompanied by source excerpts with page numbers.

## Features

- Upload and index text-based PDF documents.
- Extract text page by page and split it into overlapping chunks.
- Generate document and query embeddings with OpenAI.
- Rank passages by cosine similarity and retrieve the most relevant context.
- Generate answers using retrieved passages and recent conversation context.
- Display source filenames, page numbers, excerpts, and token usage.
- Save documents, embeddings, and conversation messages in SQLite.
- Explore the JSON API through FastAPI's interactive API page.

## How it works

1. **Upload:** Pagewise reads each PDF page and extracts selectable text.
2. **Chunk and embed:** Page text is divided into overlapping passages. OpenAI creates a vector embedding for each passage.
3. **Retrieve:** When a question arrives, Pagewise embeds it and compares it with stored passage vectors using cosine similarity.
4. **Answer:** The top matching passages and recent messages are sent to the chat model. The model is instructed to answer from the passages and cite their source markers.
5. **Show sources:** The UI displays the answer alongside the matching document excerpts and page numbers.

## Requirements

- Python 3.11 or newer
- An OpenAI API key with access to the configured embedding and chat models

## Setup

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` to your key. Keep this file private and out of GitHub. Start the app:

```sh
uvicorn app:app --reload
```

Visit <http://127.0.0.1:8000> to use Pagewise. Visit <http://127.0.0.1:8000/docs> to explore the API.

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | API key used for embeddings and answers |
| `CHAT_MODEL` | `gpt-4o-mini` | Model used to generate answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used to embed passages and questions |
| `DATABASE_PATH` | `data/pagewise.db` | Location of the SQLite database |
| `LOG_LEVEL` | `INFO` | Server log verbosity |

The server logs request status and timing, plus pipeline events for extraction, embedding, retrieval, and answer generation. Logs omit PDF text, question text, and API keys.

## Project structure

```text
pagewise/
├── app.py                 # FastAPI routes and RAG pipeline
├── static/
│   └── index.html         # Browser chat interface
├── tests/
│   └── test_core.py       # Offline core and endpoint tests
├── .env.example           # Environment variable template
├── .gitignore
├── README.md
└── requirements.txt
```

## Run tests

```sh
python -m unittest discover -s tests -v
```

The tests cover chunking, cosine similarity, retrieval ranking, page metadata, PDF parsing, and basic API routes.

## Resume description

- Built Pagewise, a PDF question-answering app with FastAPI, SQLite, and OpenAI, supporting document upload and chat-based search across selected PDFs.
- Implemented a retrieval-augmented generation pipeline with page-aware PDF extraction, overlapping chunks, vector embeddings, cosine-similarity ranking, and answers grounded in cited source passages.
- Persisted document vectors and conversation history in SQLite and added a browser interface with source excerpts, page references, token usage, and request logging.
