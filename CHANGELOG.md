# Project Fixes and Changes Log

This file acts as a memory log of all the setup hurdles encountered and the fixes applied to make the `rag-lens` project work successfully on Windows. 

## 1. Missing Dependencies
- **Node.js**: The system lacked Node.js which is required to run the Vite/React frontend.
  - *Fix*: Node.js LTS (v24.14.1) was installed via `winget`.
  - *Fix*: Ran `npm install` inside the `frontend` folder to install all Vite and React dependencies.
- **Python Packages**: The backend virtual environment (`.venv`) was empty.
  - *Fix*: Ran `pip install -r requirements.txt` to install FastAPI, ChromaDB, SentenceTransformers, etc.

## 2. Frontend Build Error
- **Issue**: The Vite frontend threw an esbuild error (`Unexpected "−"`) when starting.
- **Reason**: In `frontend/src/VectorSpace.jsx` (lines 57-59), negative numbers were written using a unicode minus sign `−` (`U+2212`) instead of a standard ASCII hyphen `-`. This crashed the JavaScript bundler.
- **Fix**: Replaced the unicode minus signs with standard hyphens in the `axMat` axes array.

## 3. Backend Type Hint Evaluation Crash
- **Issue**: The backend failed to start because of a `TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'` in `backend/ingest.py`.
- **Reason**: The syntax `chromadb.Client | None` evaluates at runtime. In `chromadb` version `0.5.x`, `chromadb.Client` is a factory function, not a class. You cannot use the `|` operator on a function in Python.
- **Fix**: Modified `ingest.py` to use `Optional[Any]` instead of `chromadb.Client | None` for the `_chroma_client` typing.

## 4. Windows ChromaDB SQLite Limitation 
- **Issue**: ChromaDB requires SQLite > 3.35. Windows Python often ships with older SQLite versions, requiring `pysqlite3-binary`.
- **Reason/Fix**: The `README.md` recommended a snippet overriding SQLite with `pysqlite3`. We added a robust `try-except` block at the very top of `backend/main.py`. Because `pysqlite3-binary` isn't fully supported on Python 3.11 for Windows without compilation, it gracefully falls back to the system `sqlite3` (which in Python 3.11 is actually new enough anyway!). 

## 5. NumPy Truth Value & JSON Serialization Errors
- **Issue**: The `/debug/pipeline` and `/debug/embeddings` endpoints threw 500 Internal Server errors when navigating the dashboard or running a query.
  - "The truth value of an array with more than one element is ambiguous"
  - "Cannot convert dictionary update sequence element #0 to a sequence"
- **Reason**: 
  1. `ChromaDB` (via sentence-transformers) was occasionally returning embedding vectors as `numpy.ndarray` instead of Python `list` types. 
  2. The code had boolean truth checks like `if all_embeddings:` or `if not embeddings:`. When Python evaluates a numpy array in an `if` statement, it crashes because it's ambiguous.
  3. `FastAPI`'s `jsonable_encoder` could not natively serialize slicing a numpy array (e.g., `chunk.embedding[:20]`).
- **Fix**: 
  - Updated `backend/debug.py` to avoid implicit truth value testing on arrays. Used `if len(all_embeddings) > 0:` and explicit `is not None` checks.
  - Updated `backend/retriever.py` to verify `results["embeddings"] is not None`.
  - Converted the Numpy slices into standard python floats using list comprehensions (`[float(v) for v in chunk.embedding[:20]]`) before returning them to the frontend so that FastAPI can serialize them to JSON.

## Final Result
All backend API endpoints (`/query`, `/debug/pipeline`, `/debug/embeddings`) now execute and return `200 OK` responses end-to-end. The user interface successfully connects and visualizes the vector space.
