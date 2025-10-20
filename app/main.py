from fastapi import FastAPI, HTTPException
from typing import List
from pathlib import Path

from .schemas import IngestRequest, AskRequest, AskResponse
from .utils.pdf import extract_text_from_pdf
from .utils.text import read_text_file
from .services.rag import RAGPipeline
from .services.llm import LLM
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os


app = FastAPI(title="LLM Financial Assistant")
rag = RAGPipeline()
llm = LLM()
# app/main.py


app = FastAPI(
    title="LLM Financial Assistant",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# (Facultatif mais pratique si tu ouvres index.html en file:// ou sur un autre port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # mets l'origine exacte si tu veux restreindre
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# --- UPLOAD ---
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Reçoit un ou plusieurs PDF et les sauvegarde dans data/.
    Retourne les chemins sauvegardés (à envoyer ensuite à /ingest).
    """
    saved = []
    for uf in files:
        dest = os.path.join(UPLOAD_DIR, uf.filename)
        with open(dest, "wb") as f:
            f.write(await uf.read())
        saved.append(dest)
    return {"saved_paths": saved}

# --- SERVIR L'UI ---
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

def root():
    return {"status": "ok", "message": "LLM Financial Assistant API"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    docs: List[str] = []
    for p in req.paths:
        path = Path(p)
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Path not found: {p}")
        if path.suffix.lower() == ".pdf":
            docs.append(extract_text_from_pdf(str(path)))
        else:
            docs.append(read_text_file(str(path)))
    rag.build_index(docs)
    return {"status": "indexed", "docs": len(docs)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        contexts = rag.retrieve(req.question, k=req.k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    context_block = "\n\n".join([f"[CTX {i+1}] {c}" for i, c in enumerate(contexts)])
    prompt = (
        "You are a helpful financial analyst. Answer the user's question using the CONTEXT.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {req.question}\n\n"
        "Answer concisely:"
    )
    answer = llm.generate(prompt)
    return AskResponse(answer=answer, context=contexts)

from fastapi import UploadFile, File
from typing import List
import os
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data" 
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for uf in files:
        dest = os.path.join(UPLOAD_DIR, uf.filename)
        with open(dest, "wb") as f:
            f.write(await uf.read())
        saved.append(dest)

    return {"saved_paths": saved}


