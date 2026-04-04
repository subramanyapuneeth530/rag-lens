@echo off
echo Creating rag-lens folder structure...

mkdir backend\tests
mkdir frontend\src
mkdir sample
mkdir .github\workflows

echo Moving backend files...
move main.py backend\
move ingest.py backend\
move retriever.py backend\
move debug.py backend\
move requirements.txt backend\
move pytest.ini backend\
move ruff.toml backend\
move conftest.py backend\tests\
move test_api.py backend\tests\

echo Moving frontend files...
move App.jsx frontend\src\
move Pipeline.jsx frontend\src\
move VectorSpace.jsx frontend\src\
move TokenViewer.jsx frontend\src\

echo Moving sample file...
move rag_explainer.txt sample\

echo Moving CI file...
move ci.yml .github\workflows\

echo Done. Now create the 4 missing files listed in README_MISSING.md
echo Then run: cd backend && python -m venv .venv
pause
