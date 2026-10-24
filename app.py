"""A small, explainable PDF question-answering app with retrieval-augmented generation."""
from __future__ import annotations

import json
import io
import logging
import math
import os
import time
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from openai import APIError, OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DB_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "data" / "pagewise.db")))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("docsy")
app = FastAPI(title="Pagewise", description="Ask questions about your PDFs with cited answers.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%s",
               request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page INTEGER NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
        """)


@app.on_event("startup")
def startup() -> None:
    init_db()


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on word boundaries, retaining a small overlap so context crosses chunk edges."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be between 0 and size - 1")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("openai_not_configured feature=embeddings_or_chat")
        raise HTTPException(status_code=503, detail="Set OPENAI_API_KEY in your environment to use AI features.")
    return OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def retrieve(query_embedding: list[float], document_ids: list[str], limit: int = TOP_K) -> list[dict[str, Any]]:
    with connection() as db:
        if document_ids:
            marks = ",".join("?" for _ in document_ids)
            rows = db.execute(f"SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.document_id IN ({marks})", document_ids).fetchall()
        else:
            rows = db.execute("SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.document_id").fetchall()
    scored = [{"document_id": row["document_id"], "filename": row["filename"], "page": row["page"],
               "content": row["content"], "score": cosine_similarity(query_embedding, json.loads(row["embedding"]))} for row in rows]
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    conversation_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)


@app.get("/api/documents")
def list_documents():
    with connection() as db:
        rows = db.execute("SELECT d.id, d.filename, d.created_at, COUNT(c.id) AS chunks FROM documents d LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.created_at DESC").fetchall()
    return [dict(row) for row in rows]


@app.post("/api/documents", status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning("upload_rejected reason=unsupported_extension filename=%r", file.filename)
        raise HTTPException(400, "Upload a PDF file.")
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        logger.warning("upload_rejected reason=file_too_large filename=%r bytes=%s", Path(file.filename).name, len(raw))
        raise HTTPException(413, "PDF must be smaller than 20 MB.")
    try:
        reader = PdfReader(io.BytesIO(raw))
        page_chunks = [(page_number, chunk) for page_number, page in enumerate(reader.pages, start=1)
                       for chunk in split_text(page.extract_text() or "")]
    except Exception as exc:
        logger.warning("pdf_parse_failed filename=%r bytes=%s error_type=%s", Path(file.filename).name, len(raw), type(exc).__name__)
        raise HTTPException(400, "Could not read this PDF.") from exc
    if not page_chunks:
        logger.warning("pdf_has_no_selectable_text filename=%r", Path(file.filename).name)
        raise HTTPException(422, "No selectable text found. Scanned PDFs need OCR and are not supported in this version.")
    logger.info("pdf_extracted filename=%r bytes=%s pages=%s chunks=%s", Path(file.filename).name, len(raw), len(reader.pages), len(page_chunks))
    try:
        vectors = embed_texts([text for _, text in page_chunks])
    except HTTPException:
        raise
    except APIError as exc:
        logger.exception("embedding_failed filename=%r chunks=%s error_type=%s", Path(file.filename).name, len(page_chunks), type(exc).__name__)
        raise HTTPException(502, "The embedding service failed. Check the server log and try again.") from exc
    except Exception as exc:
        logger.exception("embedding_failed filename=%r chunks=%s error_type=%s", Path(file.filename).name, len(page_chunks), type(exc).__name__)
        raise HTTPException(502, "Could not create embeddings. Check the server log and try again.") from exc
    document_id = str(uuid.uuid4())
    with connection() as db:
        db.execute("INSERT INTO documents(id, filename) VALUES (?, ?)", (document_id, Path(file.filename).name))
        db.executemany("INSERT INTO chunks(document_id, page, content, embedding) VALUES (?, ?, ?, ?)",
                       [(document_id, page, text, json.dumps(vector)) for (page, text), vector in zip(page_chunks, vectors)])
    logger.info("pdf_indexed document_id=%s filename=%r chunks=%s", document_id, Path(file.filename).name, len(page_chunks))
    return {"id": document_id, "filename": Path(file.filename).name, "chunks": len(page_chunks)}


@app.post("/api/chat")
def ask(request: AskRequest):
    client = get_client()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    logger.info("question_received conversation_id=%s selected_documents=%s", conversation_id, len(request.document_ids))
    try:
        query_vector = client.embeddings.create(model=EMBEDDING_MODEL, input=[request.question]).data[0].embedding
    except APIError as exc:
        logger.exception("query_embedding_failed conversation_id=%s error_type=%s", conversation_id, type(exc).__name__)
        raise HTTPException(502, "The embedding service failed. Check the server log and try again.") from exc
    matches = retrieve(query_vector, request.document_ids)
    if not matches:
        logger.warning("retrieval_empty conversation_id=%s selected_documents=%s", conversation_id, len(request.document_ids))
        raise HTTPException(404, "No indexed PDF chunks found. Upload a PDF first.")
    logger.info("retrieval_complete conversation_id=%s matches=%s best_score=%.4f", conversation_id, len(matches), matches[0]["score"])
    with connection() as db:
        history = db.execute("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 8", (conversation_id,)).fetchall()
    history = list(reversed(history))
    context = "\n\n".join(f"[Source {i}: {m['filename']}, page {m['page']}]\n{m['content']}" for i, m in enumerate(matches, 1))
    system = ("Answer using only the supplied PDF excerpts. If they do not contain the answer, say so. "
              "Cite factual claims with source markers like [Source 1]. Do not invent page numbers.\n\n" + context)
    try:
        completion = client.chat.completions.create(model=CHAT_MODEL, temperature=0.2, messages=[
            {"role": "system", "content": system}, *[dict(row) for row in history],
            {"role": "user", "content": request.question}])
    except APIError as exc:
        logger.exception("answer_generation_failed conversation_id=%s error_type=%s", conversation_id, type(exc).__name__)
        raise HTTPException(502, "The answer service failed. Check the server log and try again.") from exc
    answer = completion.choices[0].message.content or "I couldn't generate an answer."
    with connection() as db:
        db.executemany("INSERT INTO messages(conversation_id, role, content) VALUES (?, ?, ?)",
                       [(conversation_id, "user", request.question), (conversation_id, "assistant", answer)])
    sources = [{"filename": m["filename"], "page": m["page"], "excerpt": m["content"][:350], "score": round(m["score"], 4)} for m in matches]
    usage = completion.usage
    logger.info("answer_generated conversation_id=%s prompt_tokens=%s completion_tokens=%s", conversation_id,
                usage.prompt_tokens if usage else "unknown", usage.completion_tokens if usage else "unknown")
    return {"conversation_id": conversation_id, "answer": answer, "sources": sources,
            "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens} if usage else None}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    with connection() as db:
        rows = db.execute("SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
    return {"conversation_id": conversation_id, "messages": [dict(row) for row in rows]}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT / "static" / "index.html").read_text())
