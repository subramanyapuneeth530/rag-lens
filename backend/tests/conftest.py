"""
conftest.py
Shared fixtures. Mocks out ChromaDB and the embedding model so tests
run in CI without any GPU, Ollama, or ChromaDB installation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

FAKE_EMBEDDING = [0.01 * i for i in range(384)]
FAKE_CHUNK_TEXT = "This is a test chunk about RAG and vector embeddings."


@pytest.fixture(autouse=True)
def mock_embed_model():
    """Replace SentenceTransformer with a fast mock for all tests."""
    mock = MagicMock()
    mock.encode.return_value = np.array([FAKE_EMBEDDING])
    mock.tokenizer.return_value = {"input_ids": list(range(12))}
    mock.tokenizer.side_effect = None
    mock.tokenizer = MagicMock()
    mock.tokenizer.return_value = {"input_ids": list(range(12))}
    mock.tokenizer.vocab_size = 30522
    mock.tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "this", "is", "a", "test", "[SEP]"]

    with patch("ingest._embed_model", mock), \
         patch("ingest.get_embed_model", return_value=mock), \
         patch("retriever.get_embed_model", return_value=mock), \
         patch("debug.get_embed_model", return_value=mock):
        yield mock


@pytest.fixture(autouse=True)
def mock_chroma():
    """Replace ChromaDB with an in-memory mock."""
    collection = MagicMock()
    collection.add.return_value = None
    collection.get.return_value = {
        "ids": ["sample.txt__chunk_0"],
        "documents": [FAKE_CHUNK_TEXT],
        "metadatas": [{"source": "sample.txt", "chunk_index": 0, "char_start": 0, "char_end": 52, "token_count": 12}],
        "embeddings": [FAKE_EMBEDDING],
    }
    collection.query.return_value = {
        "ids": [["sample.txt__chunk_0"]],
        "documents": [[FAKE_CHUNK_TEXT]],
        "metadatas": [[{"source": "sample.txt", "chunk_index": 0, "char_start": 0, "char_end": 52, "token_count": 12}]],
        "distances": [[0.1]],
        "embeddings": [[FAKE_EMBEDDING]],
    }
    collection.delete.return_value = None

    client = MagicMock()
    client.get_or_create_collection.return_value = collection

    with patch("ingest._chroma_client", client), \
         patch("ingest.get_chroma_client", return_value=client), \
         patch("ingest.get_collection", return_value=collection), \
         patch("retriever.get_collection", return_value=collection), \
         patch("debug.get_collection", return_value=collection):
        yield collection


@pytest.fixture
def app_client(mock_embed_model, mock_chroma):
    from main import app
    return TestClient(app)
