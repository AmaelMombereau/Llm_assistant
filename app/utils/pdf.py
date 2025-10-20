from pypdf import PdfReader
from pathlib import Path


def extract_text_from_pdf(path: str) -> str:
p = Path(path)
reader = PdfReader(str(p))
content = []
for page in reader.pages:
content.append(page.extract_text() or "")
return "\n".join(content)