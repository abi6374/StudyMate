#!/usr/bin/env python3
"""
StudyMate FastAPI Backend Server
Serves the React frontend and provides API endpoints for PDF processing and Q&A.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add the app directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.services.pdf_loader import extract_text_chunks
    from app.services.embeddings import embed_texts
    from app.services.vector_store import FaissVectorStore, StoredChunk
    from app.services.watsonx_client import build_prompt, generate_answer
except ImportError as e:
    print(f"Warning: Could not import StudyMate services: {e}")
    print("Running in demo mode with mock responses.")


app = FastAPI(title="StudyMate API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    document_ids: List[str] = []

class QuestionResponse(BaseModel):
    answer: str
    citations: List[str]
    sources: List[Dict[str, Any]]

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class AuthResponse(BaseModel):
    user: Dict[str, str]
    token: str

# Demo data store (replace with real database)
demo_users = {
    "demo@studymate.com": {
        "id": "1",
        "name": "Demo User",
        "email": "demo@studymate.com",
        "password": "demo123"  # In real app, hash passwords!
    }
}

# Global storage (replace with proper database/session management)
user_indexes = {}
uploaded_files = []

# API Routes
@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Authenticate user login"""
    user = demo_users.get(request.email)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return AuthResponse(
        user={"id": user["id"], "name": user["name"], "email": user["email"]},
        token=f"demo-token-{user['id']}"
    )

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register new user"""
    if request.email in demo_users:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = {
        "id": str(len(demo_users) + 1),
        "name": request.name,
        "email": request.email,
        "password": request.password  # Hash in real app!
    }
    demo_users[request.email] = new_user
    
    return AuthResponse(
        user={"id": new_user["id"], "name": new_user["name"], "email": new_user["email"]},
        token=f"demo-token-{new_user['id']}"
    )

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save file (in real app, save to proper storage)
    file_path = Path("data/uploads") / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process file (mock response for demo)
    try:
        # In real implementation, use the actual PDF processing
        # chunks = extract_text_chunks(file_path, file.filename.replace('.pdf', ''))
        # ... process with embeddings and add to vector store
        pass
    except Exception as e:
        print(f"PDF processing error: {e}")
    
    # Add to uploaded files list
    uploaded_files.append({
        "id": len(uploaded_files) + 1,
        "filename": file.filename,
        "status": "processed"
    })
    
    return {"message": f"{file.filename} uploaded and processed successfully"}

@app.get("/api/documents")
async def get_documents():
    """Get list of uploaded documents"""
    return {"documents": uploaded_files}

@app.post("/api/qa/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents"""
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    
    # Mock response (in real implementation, use actual AI processing)
    mock_answer = f"Based on your uploaded documents, here's what I found regarding '{request.question}'. This response is generated from the semantic analysis of your PDFs and provides relevant information from the indexed content."
    
    mock_citations = [
        f"{uploaded_files[0]['filename']} p.12",
        f"{uploaded_files[0]['filename']} p.24"
    ]
    
    if len(uploaded_files) > 1:
        mock_citations.append(f"{uploaded_files[1]['filename']} p.7")
    
    return QuestionResponse(
        answer=mock_answer,
        citations=mock_citations,
        sources=[{"filename": f["filename"], "page": 12} for f in uploaded_files[:2]]
    )

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "StudyMate API"}

# Serve React app static files
react_build_path = Path("../studymate-frontend/build")
if react_build_path.exists():
    app.mount("/static", StaticFiles(directory=str(react_build_path / "static")), name="static")
    
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Serve index.html for all React routes
        return FileResponse(str(react_build_path / "index.html"))

if __name__ == "__main__":
    print("🚀 Starting StudyMate Server...")
    print("📚 Backend API: http://localhost:8000/api")
    print("🌐 React App: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
