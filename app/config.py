from pydantic import BaseModel
from dotenv import load_dotenv
import os


load_dotenv()


class Settings(BaseModel):
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "google/flan-t5-base")
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", 32))
DEVICE: str = os.getenv("DEVICE", "cpu")
DOCS_DIR: str = os.getenv("DOCS_DIR", "data/docs")
INDEX_DIR: str = os.getenv("INDEX_DIR", "data/index")


settings = Settings()