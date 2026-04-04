"""
test_api.py
Tests for all FastAPI endpoints.
All external dependencies (ChromaDB, MiniLM, Ollama) are mocked in conftest.py.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np

# ── Health ────────────────────────────────────────────────────────────────────

def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_ingest_txt_success(app_client, mock_embed_model, mock_chroma):
    mock_embed_model.encode.return_value = np.array([[0.01 * i for i in range(384)]])
    mock_embed_model.tokenizer.return_value = {"input_ids": list(range(10))}

    content = b"RAG stands for Retrieval Augmented Generation. It combines search with language models."
    r = app_client.post(
        "/ingest",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["chunks"] >= 1
    assert data["embed_model"] == "all-MiniLM-L6-v2"


def test_ingest_rejects_non_txt(app_client):
    r = app_client.post(
        "/ingest",
        files={"file": ("document.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
    )
    assert r.status_code == 400
    assert "txt" in r.json()["detail"].lower()


def test_ingest_empty_file_rejected(app_client):
    r = app_client.post(
        "/ingest",
        files={"file": ("empty.txt", io.BytesIO(b"   \n  "), "text/plain")},
    )
    assert r.status_code == 422


# ── Sources ───────────────────────────────────────────────────────────────────

def test_list_sources(app_client):
    r = app_client.get("/sources")
    assert r.status_code == 200
    assert "sources" in r.json()
    assert isinstance(r.json()["sources"], list)


def test_delete_existing_source(app_client, mock_chroma):
    r = app_client.delete("/source/sample.txt")
    assert r.status_code == 200
    assert r.json()["filename"] == "sample.txt"


def test_delete_missing_source(app_client, mock_chroma):
    mock_chroma.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    r = app_client.delete("/source/nonexistent.txt")
    assert r.status_code == 404


# ── Query ─────────────────────────────────────────────────────────────────────

def test_query_empty_rejected(app_client):
    r = app_client.post("/query", json={"question": "   "})
    assert r.status_code == 400


def test_query_without_llm(app_client):
    r = app_client.post("/query", json={"question": "What is RAG?", "top_k": 1, "use_llm": False})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert "timing" in data


def test_query_with_llm(app_client):
    with patch("retriever.generate_answer", return_value=("RAG is Retrieval Augmented Generation.", 8, 1200.0)):
        r = app_client.post("/query", json={"question": "What is RAG?", "top_k": 1, "use_llm": True})
    assert r.status_code == 200
    assert r.json()["answer"] == "RAG is Retrieval Augmented Generation."


# ── Debug endpoints ───────────────────────────────────────────────────────────

def test_debug_tokenize(app_client, mock_embed_model):
    mock_embed_model.tokenizer.side_effect = None
    tok_mock = MagicMock()
    tok_mock.__call__ = MagicMock(return_value={
        "input_ids": [101, 7632, 2003, 1996, 3231, 102],
        "offset_mapping": [(0,0),(0,4),(5,7),(8,11),(12,16),(0,0)],
    })
    tok_mock.convert_ids_to_tokens.return_value = ["[CLS]","this","is","the","test","[SEP]"]
    tok_mock.vocab_size = 30522

    with patch("debug.get_embed_model") as mock_get:
        m = MagicMock()
        m.tokenizer = tok_mock
        mock_get.return_value = m
        r = app_client.post("/debug/tokenize", json={"text": "this is the test"})

    assert r.status_code == 200
    data = r.json()
    assert "tokens" in data
    assert "token_count" in data
    assert "within_model_limit" in data


def test_debug_tokenize_empty_rejected(app_client):
    r = app_client.post("/debug/tokenize", json={"text": "  "})
    assert r.status_code == 400


def test_debug_embeddings(app_client):
    r = app_client.get("/debug/embeddings")
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data
    assert "total" in data


def test_debug_pipeline(app_client):
    with patch("main.run_query") as mock_rq, patch("main.get_pipeline_trace") as mock_pt:
        mock_trace = MagicMock()
        mock_trace.question = "What is RAG?"
        mock_trace.query_embedding = np.array([0.0] * 384)
        mock_trace.query_token_count = 6
        mock_trace.query_embed_ms = 12.0
        mock_trace.retrieved_chunks = []
        mock_trace.prompt = "test prompt"
        mock_trace.answer = "test answer"
        mock_trace.answer_tokens = 4
        mock_trace.generation_ms = 0.0
        mock_trace.total_ms = 15.0
        mock_trace.model = "llama3.2"
        mock_trace.top_k = 3
        mock_rq.return_value = mock_trace
        mock_pt.return_value = {"stage_1_ingest": {}, "stage_7_generation": {"answer": "test"}}

        r = app_client.post("/debug/pipeline", json={"question": "What is RAG?"})
    assert r.status_code == 200


# ── Ingest internals ──────────────────────────────────────────────────────────

def test_chunk_text_splits_correctly():
    from ingest import split_into_chunks
    text = ("First sentence about RAG. " * 10 + "\n\n" + "Second paragraph about embeddings. " * 10)
    with patch("ingest.get_embed_model") as mock_get:
        m = MagicMock()
        m.tokenizer.return_value = {"input_ids": list(range(8))}
        mock_get.return_value = m
        chunks = split_into_chunks(text, "test.txt", chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.source == "test.txt"
        assert c.char_start >= 0
        assert c.char_end > c.char_start
        assert len(c.text) > 0


def test_retriever_similarity_score():
    from retriever import retrieve_chunks
    chunks = retrieve_chunks([0.0] * 384, top_k=1)
    assert len(chunks) == 1
    assert 0.0 <= chunks[0].similarity <= 1.0
