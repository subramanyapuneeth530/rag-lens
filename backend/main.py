# Windows ChromaDB sqlite3 fix (must be first)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3-binary not installed, using system sqlite3

"""
main.py
FastAPI application for rag-lens.
Two layers of endpoints:
  - /ingest, /query, /sources, /delete  →  standard RAG operations
  - /debug/*                            →  full internal trace, all numbers exposed
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from debug import get_embeddings_for_source, get_pipeline_trace, get_token_detail
from ingest import delete_source, ingest_file, list_sources
from retriever import (
    build_prompt,
    embed_query,
    generate_answer_stream,
    retrieve_chunks,
    run_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-lens")

app = FastAPI(
    title="rag-lens API",
    description="Full-transparency RAG pipeline — every internal value exposed.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-lens"}


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    total_tokens: int
    avg_tokens_per_chunk: float
    avg_chars_per_chunk: float
    embed_model: str


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=500, ge=100, le=2000),
    chunk_overlap: int = Query(default=50, ge=0, le=300),
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = ingest_file(tmp_path, file.filename, chunk_size, chunk_overlap)
        total_tokens = sum(c.token_count for c in chunks)
        total_chars = sum(len(c.text) for c in chunks)
        n = len(chunks)
        logger.info(f"Ingested '{file.filename}' → {n} chunks, {total_tokens} tokens")
        return IngestResponse(
            message="Document ingested successfully.",
            filename=file.filename,
            chunks=n,
            total_tokens=total_tokens,
            avg_tokens_per_chunk=round(total_tokens / max(n, 1), 1),
            avg_chars_per_chunk=round(total_chars / max(n, 1), 1),
            embed_model="all-MiniLM-L6-v2",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── Sources ───────────────────────────────────────────────────────────────────

@app.get("/sources")
def sources():
    return {"sources": list_sources()}


@app.delete("/source/{filename}")
def delete(filename: str):
    try:
        count = delete_source(filename)
        return {"message": f"Deleted {count} chunks for '{filename}'.", "filename": filename}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Query (standard) ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    source_filter: str | None = None
    use_llm: bool = True


@app.post("/query")
def query(body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        trace = run_query(
            body.question,
            top_k=body.top_k,
            source_filter=body.source_filter,
            use_llm=body.use_llm,
        )
        return {
            "answer": trace.answer,
            "question": trace.question,
            "sources": [
                {"text": c.text[:300], "source": c.source, "similarity": c.similarity}
                for c in trace.retrieved_chunks
            ],
            "timing": {
                "embed_ms": trace.query_embed_ms,
                "generation_ms": trace.generation_ms,
                "total_ms": trace.total_ms,
            },
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Query stream (SSE) ────────────────────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(body: QueryRequest):
    """
    Server-Sent Events endpoint. Streams:
      1. retrieval metadata as first event
      2. LLM tokens as they arrive
      3. final timing summary
    Frontend shows tokens appearing in real time.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # Step 1: embed query + retrieve
            q_emb, q_tok, q_ms = embed_query(body.question)
            chunks = retrieve_chunks(q_emb, body.top_k, body.source_filter)
            prompt = build_prompt(body.question, chunks)

            retrieval_event = {
                "type": "retrieval",
                "query_tokens": q_tok,
                "query_embed_ms": q_ms,
                "chunks": [
                    {
                        "text": c.text,
                        "source": c.source,
                        "similarity": c.similarity,
                        "token_count": c.token_count,
                    }
                    for c in chunks
                ],
                "prompt": prompt,
            }
            yield f"data: {json.dumps(retrieval_event)}\n\n"
            await asyncio.sleep(0)

            if not body.use_llm:
                yield f"data: {json.dumps({'type': 'done', 'answer': '[LLM disabled]'})}\n\n"
                return

            # Step 2: stream LLM tokens
            full_answer = ""
            import time
            gen_start = time.perf_counter()
            for token in generate_answer_stream(prompt):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                await asyncio.sleep(0)

            gen_ms = round((time.perf_counter() - gen_start) * 1000, 2)

            # Step 3: done event with timing
            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'generation_ms': gen_ms})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Debug: full pipeline trace ────────────────────────────────────────────────

@app.post("/debug/pipeline")
def debug_pipeline(body: QueryRequest):
    """
    Runs the full RAG pipeline and returns every internal value:
    token lists, all 384 embedding dims (first 20 shown), PCA coords,
    similarity scores, exact prompt, timing per stage.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        trace = run_query(
            body.question,
            top_k=body.top_k,
            source_filter=body.source_filter,
            use_llm=body.use_llm,
        )
        return get_pipeline_trace(trace)
    except Exception as e:
        logger.error(f"Debug pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Debug: tokenize any text ──────────────────────────────────────────────────

class TokenizeRequest(BaseModel):
    text: str


@app.post("/debug/tokenize")
def debug_tokenize(body: TokenizeRequest):
    """
    Returns real BertTokenizer output for any input text.
    Shows token strings, IDs, subword flags, char offsets.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return get_token_detail(body.text)


# ── Debug: embeddings + PCA for 3D view ──────────────────────────────────────

@app.get("/debug/embeddings")
def debug_embeddings(source: str | None = Query(default=None)):
    """
    Returns all stored chunk embeddings with real PCA 3D coordinates.
    This is what feeds the Three.js vector space visualization.
    """
    return get_embeddings_for_source(source)
