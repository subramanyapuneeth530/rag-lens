"""
debug.py
The core of rag-lens. Computes and returns all internal pipeline data
that doc-intelligence hides: embeddings, PCA coords, token details,
similarity scores, prompt construction, timing breakdowns.
"""


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ingest import Chunk, get_collection, get_embed_model
from retriever import QueryTrace

# ── Tokenization detail ───────────────────────────────────────────────────────

def get_token_detail(text: str) -> dict:
    """
    Returns rich token-level data using the real MiniLM tokenizer (BertTokenizer).
    This is what actually runs inside the model — not a simulation.
    """
    model = get_embed_model()
    tokenizer = model.tokenizer

    encoding = tokenizer(
        text,
        return_tensors=None,
        truncation=False,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    offset_mapping = encoding.get("offset_mapping", [])
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_list = []
    for i, (tok, id_, offset) in enumerate(zip(tokens, input_ids, offset_mapping)):
        is_special = tok in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>")
        is_subword = tok.startswith("##")
        char_start, char_end = offset if offset else (0, 0)
        token_list.append({
            "index": i,
            "token": tok,
            "id": int(id_),
            "is_special": is_special,
            "is_subword": is_subword,
            "char_start": int(char_start),
            "char_end": int(char_end),
        })

    return {
        "text": text,
        "tokens": token_list,
        "token_count": len(input_ids),
        "word_count": len(text.split()),
        "char_count": len(text),
        "tokens_per_word": round(len(input_ids) / max(len(text.split()), 1), 2),
        "within_model_limit": len(input_ids) <= 512,
        "model_limit": 512,
        "tokenizer_name": "BertTokenizer (WordPiece)",
        "vocab_size": tokenizer.vocab_size,
    }


# ── PCA projection ────────────────────────────────────────────────────────────

def compute_pca_projection(embeddings: list[list[float]], n_components: int = 3) -> list[list[float]]:
    """
    Real sklearn PCA. Takes 384-dim MiniLM embeddings → 3D coordinates.
    Standardizes first so variance across dimensions is comparable.
    Returns list of [x, y, z] coordinates.
    """
    if len(embeddings) < n_components:
        # Pad with zeros if too few points for PCA
        return {
            "coords": [[0.0, 0.0, 0.0] for _ in embeddings],
            "explained_variance": [0.0, 0.0, 0.0],
            "total_variance_explained": 0.0,
            "n_components": n_components,
            "n_embeddings": len(embeddings),
            "original_dims": len(embeddings[0]) if embeddings else 0,
        }

    arr = np.array(embeddings, dtype=np.float32)
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(arr_scaled)

    explained = pca.explained_variance_ratio_.tolist()

    # Normalize to [-1, 1] range for consistent 3D rendering
    max_abs = np.abs(coords).max()
    if max_abs > 0:
        coords = coords / max_abs

    return {
        "coords": coords.tolist(),
        "explained_variance": [round(v, 4) for v in explained],
        "total_variance_explained": round(sum(explained), 4),
        "n_components": n_components,
        "n_embeddings": len(embeddings),
        "original_dims": len(embeddings[0]),
    }


# ── Full pipeline debug trace ─────────────────────────────────────────────────

def get_pipeline_trace(
    query_trace: QueryTrace,
    source_chunks: list[Chunk] | None = None,
) -> dict:
    """
    Takes a QueryTrace (from retriever.run_query) and builds the full
    debug payload the frontend pipeline explorer consumes.
    """
    # Retrieve all stored embeddings for PCA (not just top-k)
    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])

    all_embeddings = all_data["embeddings"] if all_data["embeddings"] is not None else []
    all_texts = all_data["documents"] if all_data["documents"] is not None else []
    all_metas = all_data["metadatas"] if all_data["metadatas"] is not None else []

    pca_result = compute_pca_projection(all_embeddings) if len(all_embeddings) > 0 else {}

    # Build per-chunk pca coords mapped to chunk id
    chunk_pca = []
    if pca_result and "coords" in pca_result:
        for i, (coords, text, meta) in enumerate(zip(pca_result["coords"], all_texts, all_metas)):
            chunk_pca.append({
                "id": f"{meta['source']}__chunk_{meta['chunk_index']}",
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "text_preview": text[:120],
                "token_count": meta.get("token_count", 0),
                "x": round(coords[0], 5),
                "y": round(coords[1], 5),
                "z": round(coords[2], 5),
            })

    # Query vector PCA position (project onto same PCA space)
    query_pca = None
    if len(all_embeddings) > 0 and pca_result:
        arr = np.array(all_embeddings, dtype=np.float32)
        scaler = StandardScaler()
        arr_scaled = scaler.fit_transform(arr)
        pca = PCA(n_components=3, random_state=42)
        pca.fit(arr_scaled)
        q_arr = np.array([query_trace.query_embedding], dtype=np.float32)
        q_scaled = scaler.transform(q_arr)
        q_coords = pca.transform(q_scaled)[0]
        max_abs = np.abs(np.vstack([pca.transform(arr_scaled)])).max()
        if max_abs > 0:
            q_coords = q_coords / max_abs
        query_pca = {
            "x": round(float(q_coords[0]), 5),
            "y": round(float(q_coords[1]), 5),
            "z": round(float(q_coords[2]), 5),
        }

    # Token detail for query
    query_token_detail = get_token_detail(query_trace.question)

    # Retrieved chunks with full embedding detail
    retrieved_detail = []
    for chunk in query_trace.retrieved_chunks:
        tok_detail = get_token_detail(chunk.text)
        retrieved_detail.append({
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "token_count": chunk.token_count,
            "similarity": chunk.similarity,
            "embedding_preview": [float(v) for v in chunk.embedding[:20]],  # first 20 of 384 dims
            "embedding_norm": round(float(np.linalg.norm(chunk.embedding)), 5),
            "tokens": tok_detail["tokens"],
        })

    return {
        "stage_1_ingest": {
            "total_chunks_in_store": len(all_texts),
            "sources": list({m["source"] for m in all_metas}),
        },
        "stage_2_chunking": {
            "retrieved_chunk_count": len(query_trace.retrieved_chunks),
            "top_k": query_trace.top_k,
        },
        "stage_3_tokenization": {
            "query": query_token_detail,
        },
        "stage_4_embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "query_embed_ms": query_trace.query_embed_ms,
            "query_embedding_preview": query_trace.query_embedding[:20],
            "query_embedding_norm": round(float(np.linalg.norm(query_trace.query_embedding)), 5),
            "normalization": "L2 (cosine-ready)",
        },
        "stage_5_vector_space": {
            "pca_metadata": {k: v for k, v in pca_result.items() if k != "coords"},
            "chunk_points": chunk_pca,
            "query_point": query_pca,
        },
        "stage_6_retrieval": {
            "retrieved_chunks": retrieved_detail,
            "similarity_metric": "cosine (1 - ChromaDB distance)",
        },
        "stage_7_generation": {
            "prompt": query_trace.prompt,
            "prompt_token_estimate": len(query_trace.prompt.split()),
            "answer": query_trace.answer,
            "answer_tokens": query_trace.answer_tokens,
            "model": query_trace.model,
            "temperature": 0.1,
            "generation_ms": query_trace.generation_ms,
            "total_ms": query_trace.total_ms,
        },
    }


# ── Standalone embeddings endpoint (for 3D view without a query) ──────────────

def get_embeddings_for_source(source: str | None = None) -> dict:
    """
    Returns all stored chunk embeddings with PCA coords.
    Used by the 3D vector space view independently of any query.
    """
    collection = get_collection()
    where = {"source": source} if source else None
    results = collection.get(
        where=where,
        include=["documents", "metadatas", "embeddings"],
    )

    embeddings = results["embeddings"] if results["embeddings"] is not None else []
    texts = results["documents"] if results["documents"] is not None else []
    metas = results["metadatas"] if results["metadatas"] is not None else []

    if len(embeddings) == 0:
        return {"chunks": [], "pca": None}

    pca_result = compute_pca_projection(embeddings)
    coords = pca_result.get("coords", [])

    chunks = []
    for i, (text, meta, coord) in enumerate(zip(texts, metas, coords)):
        chunks.append({
            "id": f"{meta['source']}__chunk_{meta['chunk_index']}",
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "text": text,
            "text_preview": text[:100],
            "token_count": meta.get("token_count", 0),
            "x": round(coord[0], 5),
            "y": round(coord[1], 5),
            "z": round(coord[2], 5),
            "embedding_preview": [float(v) for v in embeddings[i][:20]],
        })

    return {
        "chunks": chunks,
        "pca": {k: v for k, v in pca_result.items() if k != "coords"},
        "total": len(chunks),
    }
