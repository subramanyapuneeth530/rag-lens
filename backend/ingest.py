"""
ingest.py
Handles all document ingestion: reading, chunking, embedding, storing.
No LangChain. Every step is explicit so the debug layer can expose it.
"""

import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

_embed_model: SentenceTransformer | None = None
_chroma_client: Any | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_chroma_client() -> Any:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Chunking ──────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    index: int
    char_start: int
    char_end: int
    source: str
    token_count: int = 0
    embedding: list[float] = field(default_factory=list)
    embed_time_ms: float = 0.0


def clean_text(text: str) -> str:
    """Normalize unicode, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def split_into_chunks(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    RecursiveCharacterTextSplitter logic — no LangChain.
    Tries to split on paragraph breaks, then sentences, then words.
    Returns Chunk objects with character offsets preserved.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[Chunk] = []
    model = get_embed_model()

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        merged, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size or not current:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = part
        if current:
            merged.append(current)
        result = []
        for m in merged:
            if len(m) > chunk_size:
                result.extend(_split(m, seps[1:]))
            else:
                result.append(m)
        return result

    raw_splits = _split(text, separators)

    # Merge small splits and apply overlap
    final_texts: list[str] = []
    current = ""
    for split in raw_splits:
        if not current:
            current = split
        elif len(current) + len(split) + 1 <= chunk_size:
            current += " " + split
        else:
            final_texts.append(current)
            # overlap: take last `chunk_overlap` chars of current
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = (overlap_text + " " + split).strip() if overlap_text else split
    if current:
        final_texts.append(current)

    # Build Chunk objects with real character offsets
    search_start = 0
    for i, chunk_text in enumerate(final_texts):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        char_start = text.find(chunk_text[:40], search_start)
        if char_start == -1:
            char_start = search_start
        char_end = char_start + len(chunk_text)
        search_start = max(0, char_end - chunk_overlap)

        # Real token count from the actual tokenizer
        tokens = model.tokenizer(
            chunk_text,
            return_tensors=None,
            truncation=False,
        )
        token_count = len(tokens["input_ids"])

        chunks.append(Chunk(
            text=chunk_text,
            index=i,
            char_start=char_start,
            char_end=char_end,
            source=source,
            token_count=token_count,
        ))

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Embed each chunk with MiniLM. Stores the full 384-dim vector
    on the Chunk object so the debug layer can expose it.
    """
    model = get_embed_model()
    texts = [c.text for c in chunks]

    t0 = time.perf_counter()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    per_chunk_ms = total_ms / max(len(chunks), 1)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.tolist()
        chunk.embed_time_ms = round(per_chunk_ms, 2)

    return chunks


# ── Storage ───────────────────────────────────────────────────────────────────

def store_chunks(chunks: list[Chunk]) -> None:
    collection = get_collection()
    collection.add(
        ids=[f"{c.source}__chunk_{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=[c.embedding for c in chunks],
        metadatas=[{
            "source": c.source,
            "chunk_index": c.index,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "token_count": c.token_count,
        } for c in chunks],
    )


def delete_source(source: str) -> int:
    collection = get_collection()
    results = collection.get(where={"source": source})
    if not results["ids"]:
        raise ValueError(f"No document found: {source}")
    collection.delete(ids=results["ids"])
    return len(results["ids"])


def list_sources() -> list[dict]:
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in results["metadatas"]:
        src = meta["source"]
        if src not in seen:
            seen[src] = {"source": src, "chunk_count": 0}
        seen[src]["chunk_count"] += 1
    return sorted(seen.values(), key=lambda x: x["source"])


# ── Full ingest pipeline ──────────────────────────────────────────────────────

def ingest_file(file_path: str, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]:
    """
    Full pipeline: read → clean → chunk → embed → store.
    Returns the list of Chunk objects (with embeddings) for the debug layer.
    """
    with open(file_path, encoding="utf-8") as f:
        raw = f.read()

    text = clean_text(raw)
    if not text:
        raise ValueError("File appears to be empty after cleaning.")

    chunks = split_into_chunks(text, filename, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("Could not extract any chunks from this file.")

    chunks = embed_chunks(chunks)
    store_chunks(chunks)
    return chunks
