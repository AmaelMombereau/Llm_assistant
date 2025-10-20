# LLM-based Financial Assistant (FastAPI + RAG)


This project demonstrates an LLM-powered assistant for financial documents. It supports:
- Document ingestion (PDF/Text)
- Vector indexing (FAISS + MiniLM embeddings)
- RAG-style Q&A with a seq2seq LLM (Flan-T5-base)
- Clean FastAPI endpoints and Pydantic schemas
- Docker image, CI, and minimal tests


## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
