from fastapi import FastAPI, HTTPException
from typing import List
from pathlib import Path

from .schemas import IngestRequest, AskRequest, AskResponse
from .utils.pdf import extract_text_from_pdf
from .utils.text import read_text_file
from .services.rag import RAGPipeline
from .services.llm import LLM

app = FastAPI(title="LLM Financial Assistant")
rag = RAGPipeline()
llm = LLM()

@app.get("/")
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
