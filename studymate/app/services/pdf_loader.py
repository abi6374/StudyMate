from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from dataclasses import dataclass

@dataclass
class PageChunk:
    doc_id: str
    page_number: int
    chunk_id: str
    text: str
    meta: Dict[str, Any]


def extract_text_chunks(file_path: Path, doc_id: str, max_chunk_chars: int = 1200, overlap: int = 150) -> List[PageChunk]:
    doc = fitz.open(file_path)
    chunks: List[PageChunk] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()
        if not text:
            continue
        # simple splitting by paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        buf = []
        current_len = 0
        chunk_count = 0
        def flush():
            nonlocal buf, current_len, chunk_count
            if not buf:
                return
            combined = "\n".join(buf)
            chunk_id = f"{doc_id}_{page_index}_{chunk_count}"
            chunks.append(PageChunk(
                doc_id=doc_id,
                page_number=page_index + 1,
                chunk_id=chunk_id,
                text=combined,
                meta={"source": str(file_path.name)}
            ))
            chunk_count += 1
            buf = []
            current_len = 0
        for para in paragraphs:
            if current_len + len(para) > max_chunk_chars and current_len > 0:
                flush()
                # overlap logic (reuse tail of previous chunk)
                if overlap > 0 and chunks:
                    tail = chunks[-1].text[-overlap:]
                    buf = [tail]
                    current_len = len(tail)
            buf.append(para)
            current_len += len(para)
        flush()
    return chunks
