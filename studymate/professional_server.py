"""
Professional StudyMate FastAPI Server
Implements all requirements from the problem statement:
- Advanced RAG with FAISS and SentenceTransformers
- IBM Watson integration for LLM-based answers
- Professional API endpoints
- Enhanced error handling and logging
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# FastAPI and web components
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from services.advanced_rag import get_advanced_rag, AdvancedRAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StudyMate API",
    description="Advanced AI-powered PDF-based Q&A system for students",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine
rag_engine: AdvancedRAGEngine = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    
class QuestionResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    timestamp: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    upload_date: str
    chunks_count: int
    metadata: Dict[str, Any] = {}

class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_model: str
    watson_enabled: bool
    system_status: str

# Storage configuration
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global rag_engine
    try:
        logger.info("🚀 Starting StudyMate Professional Server...")
        
        # Initialize advanced RAG engine
        rag_engine = get_advanced_rag()
        
        # Load environment variables for Watson
        watson_status = "enabled" if rag_engine.watson_enabled else "disabled (using fallback)"
        logger.info(f"IBM Watson integration: {watson_status}")
        
        logger.info("✅ StudyMate server started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "StudyMate Professional API",
        "version": "2.0.0",
        "description": "Advanced AI-powered PDF-based Q&A system for students",
        "features": [
            "Semantic Search with FAISS and SentenceTransformers",
            "IBM Watson LLM integration",
            "Professional PDF text extraction",
            "Conversational Q&A interface"
        ],
        "docs": "/docs"
    }

@app.get("/api/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        stats = rag_engine.get_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "rag_engine": "advanced",
            "documents_loaded": stats["total_documents"],
            "watson_enabled": stats["watson_enabled"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/api/documents/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process a PDF document
    Implements: Accurate Text Extraction and Preprocessing
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported"
            )
        
        # Generate unique document ID
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename.replace('.pdf', '')}"
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"📄 Uploaded file: {file.filename} ({len(content)} bytes)")
        
        # Process document with advanced RAG
        success = rag_engine.add_document(
            pdf_path=str(file_path),
            doc_id=doc_id,
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "upload_date": datetime.now().isoformat()
            }
        )
        
        if success:
            return {
                "message": "Document uploaded and processed successfully",
                "document_id": doc_id,
                "filename": file.filename,
                "status": "processed",
                "features_used": [
                    "PyMuPDF text extraction",
                    "SentenceTransformer embeddings",
                    "FAISS indexing",
                    "Semantic chunking"
                ]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to process document"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about uploaded documents
    Implements: Conversational Q&A from Academic PDFs
    """
    try:
        logger.info(f"💬 Question received: {request.question}")
        
        # Process question with advanced RAG
        response = rag_engine.ask_question(request.question)
        
        return QuestionResponse(**response)
        
    except Exception as e:
        logger.error(f"Question processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )

@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Get list of all uploaded documents"""
    try:
        documents = rag_engine.get_documents()
        
        document_list = []
        for doc in documents:
            doc_info = DocumentInfo(
                id=doc['id'],
                filename=doc['metadata'].get('filename', 'Unknown'),
                upload_date=doc['added_at'],
                chunks_count=doc['chunks_count'],
                metadata=doc['metadata']
            )
            document_list.append(doc_info)
        
        return document_list
        
    except Exception as e:
        logger.error(f"Document listing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the system"""
    try:
        success = rag_engine.remove_document(doc_id)
        
        if success:
            return {"message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and status"""
    try:
        stats = rag_engine.get_stats()
        
        return SystemStats(
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks'],
            embedding_model=stats['embedding_model'],
            watson_enabled=stats['watson_enabled'],
            system_status="operational"
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )

@app.post("/api/search", response_model=Dict[str, Any])
async def semantic_search(request: QuestionRequest):
    """
    Perform semantic search without answer generation
    Implements: Semantic Search Using FAISS and Embeddings
    """
    try:
        logger.info(f"🔍 Semantic search: {request.question}")
        
        # Perform semantic search
        results = rag_engine.semantic_search(request.question, top_k=10)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk": result['chunk'],
                "score": result['score'],
                "doc_id": result['doc_id'],
                "metadata": result['metadata']
            })
        
        return {
            "query": request.question,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_method": "FAISS + SentenceTransformers"
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "status": "error"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting StudyMate Professional Server...")
    print("📚 Features:")
    print("   ✅ Advanced RAG with FAISS and SentenceTransformers")
    print("   ✅ IBM Watson LLM integration")  
    print("   ✅ Professional PDF text extraction")
    print("   ✅ Semantic search and Q&A")
    print("   ✅ RESTful API with comprehensive documentation")
    print()
    print("🌐 Server will be available at:")
    print("   • API: http://localhost:8000")
    print("   • Docs: http://localhost:8000/docs")
    print("   • Health: http://localhost:8000/api/health")
    print()
    
    uvicorn.run(
        "professional_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
