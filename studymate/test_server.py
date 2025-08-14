#!/usr/bin/env python3
"""Simple FastAPI server for testing"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
import fitz  # PyMuPDF
from pathlib import Path
from services.simple_rag import SimpleTfIdfRAG

app = FastAPI(title="StudyMate Test API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
print("🔧 Initializing RAG engine...")
rag_engine = SimpleTfIdfRAG(chunk_size=512, chunk_overlap=50)
rag_engine.load_index()
print(f"✅ Loaded {len(rag_engine.chunks)} chunks")

class QuestionRequest(BaseModel):
    question: str
    language: Optional[str] = "en"
    response_style: Optional[str] = "comprehensive"
    model: Optional[str] = "local"

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    try:
        filename = file.filename or "unknown.pdf"
        print(f"📁 Upload attempt - filename: {filename}, content_type: {file.content_type}")
        
        if not filename.endswith('.pdf'):
            return {"error": "Only PDF files are supported"}
        
        # Create uploads directory
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = uploads_dir / filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text using PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Add to RAG engine
        doc_id = filename.replace('.pdf', '')
        chunks_added = rag_engine.add_document(
            content=text,
            doc_id=doc_id,
            filename=filename
        )
        
        # Save index
        rag_engine.save_index()
        
        print(f"✅ Successfully processed {filename} - {chunks_added} chunks added")
        
        return {
            "success": True,
            "message": f"Successfully uploaded and processed {filename}",
            "chunks_added": chunks_added,
            "total_chunks": len(rag_engine.chunks)
        }
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/api/qa/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question using the RAG engine"""
    try:
        print(f"🔍 Received question: {request.question}")
        
        # Generate answer using local RAG
        answer, citations, confidence, sources = rag_engine.generate_answer(
            query=request.question,
            language=request.language or "en",
            response_style=request.response_style or "comprehensive"
        )
        
        # Get relevant chunks for sources
        chunks = rag_engine.retrieve_relevant_chunks(request.question, top_k=3)
        
        formatted_sources = []
        for result, similarity in chunks:
            formatted_sources.append({
                "document": result.get("filename", "Unknown"),
                "chunk": result.get("chunk_id", 0),
                "relevance": f"{float(similarity):.2f}",
                "preview": result.get("content", "")[:200] + "..."
            })
        
        return {
            "answer": answer,
            "citations": citations,
            "sources": formatted_sources,
            "confidence_score": float(confidence),
            "ai_model_used": "local",
            "language_detected": request.language or "en",
            "processing_time": 1.0
        }
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "answer": "Sorry, there was an error processing your question.",
            "citations": [],
            "sources": [],
            "confidence_score": 0.0,
            "ai_model_used": "local",
            "language_detected": "en",
            "processing_time": 0.0
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "chunks": len(rag_engine.chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
