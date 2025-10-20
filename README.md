# LLM Financial Assistant (FastAPI + RAG)

This project implements an **intelligent financial assistant** powered by a **Large Language Model (LLM)** and a **Retrieval-Augmented Generation (RAG)** architecture.  
The goal is to automatically **analyze financial documents (PDF or text)**, **extract key information**, and **answer natural language questions**.

---

## Overview

- Document ingestion (PDF/Text)
- Vector indexing using FAISS + MiniLM embeddings
- RAG-style Q&A with a seq2seq LLM (Flan-T5-base)
- FastAPI backend with clean endpoints and Pydantic schemas
- Lightweight web interface for interaction
- Optional Docker image, CI, and minimal tests

---

## Project Structure

llm_fin_assistant/
┣ app/
┃ ┣ main.py # FastAPI app with /upload, /ingest, /ask
┣ data/ # Folder containing uploaded files
┣ static/
┃ ┗ index.html # Web interface
┣ requirements.txt
┗ README.md


---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AmaelMombereau/Llm_assistant.git
cd llm_fin_assistant
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
```
#### macOS / Linux
```bashsource
.venv/bin/activate
```
#### Windows PowerShell
```bash
.venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the server
The API will be available at:
http://127.0.0.1:8000

## API Endpoints

| Step | Action | Endpoint | Description |
|------|---------|-----------|-------------|
| 1 | **Upload PDFs** | `POST /upload` | Upload one or more PDF files. They are saved under `data/`. |
| 2 | **Ingest documents** | `POST /ingest` | Reads the uploaded files and indexes them (FAISS vector store). |
| 3 | **Ask a question** | `POST /ask` | Sends a question, retrieves top-k passages, and generates an answer using the LLM. |

---


## How to Use the Interface

### 1. Upload PDFs
- Drag and drop or select one or more files.
- Files are uploaded to `/upload` and stored in the `data/` folder.
- Example response:

```json
{ "saved_paths": ["data/report.pdf"] }
```
### 2. Ingest
- Click "Ingest saved files" to send:
```json
{ "paths": ["data/report.pdf"] }
```
### 3. Ask a Question
- Example: *“Summarize the article.”*
- The field `k` defines how many passages (top-k) the retriever should use:
  - `k = 4` (default) gives a good balance between context and precision.
  - Increase `k` for more detailed responses.
- Click **“Ask /ask”** and the system will display the answer and the retrieved context.
