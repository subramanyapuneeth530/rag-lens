"""
retriever.py
Handles query embedding, similarity search, and LLM generation.
No LangChain. Direct calls to ChromaDB and Ollama.
"""

import os
import time
from dataclasses import dataclass, field

import ollama

from ingest import get_collection, get_embed_model

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions using only the context provided below.
If the answer cannot be found in the context, say exactly:
"I could not find an answer in the provided documents."
Do not invent information. Do not reference sources outside the context.

Context:
{context}

Question: {question}

Answer:"""


# ── Data shapes ───────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    similarity: float
    embedding: list[float] = field(default_factory=list)


@dataclass
class QueryTrace:
    question: str
    query_embedding: list[float]
    query_token_count: int
    query_embed_ms: float
    retrieved_chunks: list[RetrievedChunk]
    prompt: str
    answer: str
    answer_tokens: int
    generation_ms: float
    total_ms: float
    model: str
    top_k: int


# ── Query embedding ───────────────────────────────────────────────────────────

def embed_query(question: str) -> tuple[list[float], int, float]:
    """
    Returns (embedding, token_count, elapsed_ms).
    Uses the same MiniLM model as ingest — critical for correct similarity.
    """
    model = get_embed_model()

    tokens = model.tokenizer(question, return_tensors=None, truncation=False)
    token_count = len(tokens["input_ids"])

    t0 = time.perf_counter()
    vec = model.encode(question, normalize_embeddings=True)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return vec.tolist(), token_count, elapsed_ms


# ── Similarity search ─────────────────────────────────────────────────────────

def retrieve_chunks(
    query_embedding: list[float],
    top_k: int = 3,
    source_filter: str | None = None,
) -> list[RetrievedChunk]:
    """
    Query ChromaDB with the real embedding vector.
    Returns chunks with their actual cosine similarity scores.
    """
    collection = get_collection()

    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    chunks: list[RetrievedChunk] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        # ChromaDB cosine distance = 1 - similarity (when hnsw:space = cosine)
        distance = results["distances"][0][i]
        similarity = round(1.0 - distance, 4)
        embedding = results["embeddings"][0][i] if results["embeddings"] is not None else []

        chunks.append(RetrievedChunk(
            text=results["documents"][0][i],
            source=meta.get("source", "unknown"),
            chunk_index=meta.get("chunk_index", i),
            char_start=meta.get("char_start", 0),
            char_end=meta.get("char_end", 0),
            token_count=meta.get("token_count", 0),
            similarity=similarity,
            embedding=list(embedding),
        ))

    return chunks


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Chunk {i} — {chunk.source}]\n{chunk.text}")
    context = "\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ── LLM generation ────────────────────────────────────────────────────────────

def generate_answer(prompt: str) -> tuple[str, int, float]:
    """
    Calls Ollama directly (no LangChain).
    Returns (answer_text, token_count, elapsed_ms).
    """
    client = ollama.Client(host=OLLAMA_HOST)

    t0 = time.perf_counter()
    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.1,
            "num_predict": 512,
        },
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = response["response"].strip()
    token_count = response.get("eval_count", len(answer.split()))
    return answer, token_count, elapsed_ms


def generate_answer_stream(prompt: str):
    """
    Streams tokens from Ollama. Yields string chunks as they arrive.
    Used by the SSE endpoint so the frontend can show tokens arriving live.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    stream = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        stream=True,
        options={
            "temperature": 0.1,
            "num_predict": 512,
        },
    )
    for chunk in stream:
        token = chunk.get("response", "")
        if token:
            yield token


# ── Full query pipeline ───────────────────────────────────────────────────────

def run_query(
    question: str,
    top_k: int = 3,
    source_filter: str | None = None,
    use_llm: bool = True,
) -> QueryTrace:
    """
    Complete RAG pipeline: embed query → retrieve → build prompt → generate.
    Returns a QueryTrace with every intermediate value exposed.
    """
    total_start = time.perf_counter()

    query_embedding, query_token_count, query_embed_ms = embed_query(question)
    chunks = retrieve_chunks(query_embedding, top_k, source_filter)
    prompt = build_prompt(question, chunks)

    if use_llm and chunks:
        answer, answer_tokens, generation_ms = generate_answer(prompt)
    else:
        answer = "[LLM disabled — showing retrieved chunks only]"
        answer_tokens = 0
        generation_ms = 0.0

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)

    return QueryTrace(
        question=question,
        query_embedding=query_embedding,
        query_token_count=query_token_count,
        query_embed_ms=query_embed_ms,
        retrieved_chunks=chunks,
        prompt=prompt,
        answer=answer,
        answer_tokens=answer_tokens,
        generation_ms=generation_ms,
        total_ms=total_ms,
        model=OLLAMA_MODEL,
        top_k=top_k,
    )
