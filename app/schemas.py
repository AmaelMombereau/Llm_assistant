from pydantic import BaseModel, Field
from typing import List

class IngestRequest(BaseModel):
    paths: List[str] = Field(..., description="Local file paths to ingest (PDF or text)")

class AskRequest(BaseModel):
    question: str
    k: int = 4

class AskResponse(BaseModel):
    answer: str
    context: List[str]
