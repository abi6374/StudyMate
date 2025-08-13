from pathlib import Path
from app.services.pdf_loader import extract_text_chunks
import tempfile
import fitz

def make_pdf(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72,72), text)
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def test_chunking_basic():
    pdf = make_pdf("Paragraph one.\n\nParagraph two is a bit longer.")
    chunks = extract_text_chunks(pdf, "doc1", max_chunk_chars=50)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)
