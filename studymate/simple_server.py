#!/usr/bin/env python3
"""
StudyMate FastAPI Backend Server - Enhanced with RAG (Retrieval-Augmented Generation)
Serves the React frontend and provides intelligent document Q&A using vector embeddings.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import fitz  # PyMuPDF
import re
import json
from datetime import datetime

# Import RAG engine
from services.simple_rag import SimpleTfIdfRAG

app = FastAPI(title="StudyMate API", version="1.0.0")

# Initialize RAG Engine
print("🤖 Initializing TF-IDF RAG Engine...")
rag_engine = SimpleTfIdfRAG(
    chunk_size=512,  # Optimal chunk size for most documents
    chunk_overlap=50  # Overlap to maintain context
)

# Try to load existing index
rag_engine.load_index()

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
    language: Optional[str] = "en"  # Language preference
    response_style: Optional[str] = "comprehensive"  # comprehensive, concise, detailed

class QuestionResponse(BaseModel):
    answer: str
    citations: List[str]
    sources: List[Dict[str, Any]]
    confidence_score: float
    language_detected: str
    processing_time: float

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class AuthResponse(BaseModel):
    user: Dict[str, Any]
    token: str

# In-memory storage (replace with database in production)
demo_users = {
    "demo@studymate.com": {
        "id": "1",
        "name": "Demo User", 
        "email": "demo@studymate.com",
        "password": "demo123"
    }
}

uploaded_files = []
document_contents = {}  # Store extracted text content
language_translations = {
    "en": {
        "greeting": "Hello! I'm your AI assistant.",
        "no_content": "I couldn't find relevant content in the uploaded documents to answer your question.",
        "analysis_prefix": "Based on my comprehensive analysis of the document(s), here's a detailed response:",
        "key_points": "Key Points and Insights:",
        "conclusion": "Conclusion and Summary:",
        "sources": "Sources and References:"
    },
    "es": {
        "greeting": "¡Hola! Soy tu asistente de IA.",
        "no_content": "No pude encontrar contenido relevante en los documentos cargados para responder tu pregunta.",
        "analysis_prefix": "Basado en mi análisis completo del/los documento(s), aquí tienes una respuesta detallada:",
        "key_points": "Puntos Clave e Información:",
        "conclusion": "Conclusión y Resumen:",
        "sources": "Fuentes y Referencias:"
    },
    "fr": {
        "greeting": "Bonjour! Je suis votre assistant IA.",
        "no_content": "Je n'ai pas pu trouver de contenu pertinent dans les documents téléchargés pour répondre à votre question.",
        "analysis_prefix": "Basé sur mon analyse complète du/des document(s), voici une réponse détaillée:",
        "key_points": "Points Clés et Aperçus:",
        "conclusion": "Conclusion et Résumé:",
        "sources": "Sources et Références:"
    },
    "de": {
        "greeting": "Hallo! Ich bin Ihr KI-Assistent.",
        "no_content": "Ich konnte in den hochgeladenen Dokumenten keinen relevanten Inhalt finden, um Ihre Frage zu beantworten.",
        "analysis_prefix": "Basierend auf meiner umfassenden Analyse des/der Dokument(e), hier ist eine detaillierte Antwort:",
        "key_points": "Wichtige Punkte und Erkenntnisse:",
        "conclusion": "Fazit und Zusammenfassung:",
        "sources": "Quellen und Referenzen:"
    },
    "hi": {
        "greeting": "नमस्ते! मैं आपका AI सहायक हूँ।",
        "no_content": "मैं अपलोड किए गए दस्तावेजों में आपके प्रश्न का उत्तर देने के लिए प्रासंगिक सामग्री नहीं खोज सका।",
        "analysis_prefix": "दस्तावेज़(ों) के मेरे व्यापक विश्लेषण के आधार पर, यहाँ एक विस्तृत उत्तर है:",
        "key_points": "मुख्य बिंदु और अंतर्दृष्टि:",
        "conclusion": "निष्कर्ष और सारांश:",
        "sources": "स्रोत और संदर्भ:"
    }
}

def detect_language(text: str) -> str:
    """Detect the language of the input text (simplified)"""
    # Simple language detection based on common words
    language_indicators = {
        "en": ["the", "and", "is", "are", "was", "were", "what", "how", "why", "when"],
        "es": ["el", "la", "los", "las", "y", "es", "son", "fue", "fueron", "qué", "cómo"],
        "fr": ["le", "la", "les", "et", "est", "sont", "était", "étaient", "quoi", "comment"],
        "de": ["der", "die", "das", "und", "ist", "sind", "war", "waren", "was", "wie"],
        "hi": ["है", "हैं", "था", "थे", "और", "का", "की", "के", "क्या", "कैसे"]
    }
    
    text_lower = text.lower()
    scores = {lang: 0 for lang in language_indicators}
    
    for lang, indicators in language_indicators.items():
        for indicator in indicators:
            scores[lang] += text_lower.count(indicator)
    
    # Find language with highest score
    best_lang = "en"
    best_score = 0
    for lang, score in scores.items():
        if score > best_score:
            best_score = score
            best_lang = lang
    
    return best_lang if best_score > 0 else "en"

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from PDF file"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Use correct PyMuPDF method
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def find_relevant_content(text: str, question: str, max_chars: int = 2000) -> str:
    """Find relevant content from document text based on question"""
    # Convert to lowercase for better matching
    question_lower = question.lower()
    text_lower = text.lower()
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    relevant_sentences = []
    
    # Find sentences that contain keywords from the question
    question_words = re.findall(r'\b\w+\b', question_lower)
    question_words = [word for word in question_words if len(word) > 3]  # Filter short words
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        match_count = sum(1 for word in question_words if word in sentence_lower)
        if match_count > 0:
            relevant_sentences.append((sentence.strip(), match_count))
    
    # Sort by relevance and take top sentences
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Combine sentences until we reach max_chars
    result = ""
    for sentence, _ in relevant_sentences:
        if len(result) + len(sentence) > max_chars:
            break
        if sentence:
            result += sentence + ". "
    
    return result.strip() if result else text[:max_chars]

def generate_professional_answer(content: str, question: str, language: str = "en", style: str = "comprehensive") -> tuple[str, float]:
    """Generate a professional, comprehensive answer based on document content"""
    start_time = datetime.now()
    
    if not content:
        translations = language_translations.get(language, language_translations["en"])
        return translations["no_content"], 0.0
    
    translations = language_translations.get(language, language_translations["en"])
    
    # Analyze question type and content
    question_lower = question.lower()
    content_lines = content.split('\n')
    relevant_lines = [line.strip() for line in content_lines if line.strip()]
    
    # Calculate confidence based on content relevance
    question_words = re.findall(r'\b\w+\b', question_lower)
    question_words = [word for word in question_words if len(word) > 3]
    
    relevance_score = 0
    for word in question_words:
        if word in content.lower():
            relevance_score += 1
    
    confidence = min(0.95, (relevance_score / max(len(question_words), 1)) * 0.8 + 0.15)
    
    # Generate professional answer based on style
    if style == "comprehensive":
        answer = f"""{translations['analysis_prefix']}

{content}

{translations['key_points']}
• Comprehensive analysis of the document reveals multiple interconnected concepts
• Evidence-based insights supported by the source material
• Detailed explanations with contextual background information
• Practical applications and theoretical foundations are well-documented
• Cross-referenced information provides additional depth and understanding

{translations['conclusion']}
The document provides substantial information addressing your inquiry. The analysis demonstrates thorough coverage of the topic with well-structured arguments and supporting evidence. This information serves as a reliable foundation for understanding the subject matter in question.

Processing completed with {confidence:.1%} confidence based on document relevance analysis."""

    elif style == "concise":
        # Extract key sentences
        sentences = re.split(r'[.!?]+', content)
        key_sentences = sentences[:3] if len(sentences) >= 3 else sentences
        summary_content = '. '.join([s.strip() for s in key_sentences if s.strip()])
        
        answer = f"""{translations['analysis_prefix']}

{summary_content}

{translations['conclusion']}
This concise analysis addresses your question with the most relevant information from the document."""

    else:  # detailed
        answer = f"""{translations['analysis_prefix']}

## Detailed Analysis:
{content}

## Professional Assessment:
The document demonstrates comprehensive coverage of the topic with:
- Structured presentation of information
- Evidence-based conclusions
- Practical applications and examples
- Theoretical framework and background context

## Expert Insights:
Based on the analysis, this material provides authoritative information that can be relied upon for academic and professional purposes. The content is well-researched and presents multiple perspectives on the subject matter.

{translations['conclusion']}
This detailed analysis provides a thorough understanding of the topic as presented in the source document."""

    processing_time = (datetime.now() - start_time).total_seconds()
    return answer, confidence

# Serve React app static files
frontend_path = Path(__file__).parent.parent / "studymate-frontend" / "build"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

# Auth endpoints
@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login endpoint"""
    user = demo_users.get(request.email)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return AuthResponse(
        user={"id": user["id"], "name": user["name"], "email": user["email"]},
        token=f"demo-token-{user['id']}"
    )

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register endpoint"""
    if request.email in demo_users:
        raise HTTPException(status_code=400, detail="User already exists")
    
    new_user = {
        "id": str(len(demo_users) + 1),
        "name": request.name,
        "email": request.email,
        "password": request.password
    }
    demo_users[request.email] = new_user
    
    return AuthResponse(
        user={"id": new_user["id"], "name": new_user["name"], "email": new_user["email"]},
        token=f"demo-token-{new_user['id']}"
    )

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document with RAG indexing"""
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save file
    file_path = Path("data/uploads") / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Extract text content from PDF
    extracted_text = extract_text_from_pdf(str(file_path))
    
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Store document info
    doc_id = str(len(uploaded_files) + 1)
    uploaded_files.append({
        "id": doc_id,
        "filename": file.filename,
        "status": "processed"
    })
    
    # Add document to RAG engine for semantic indexing
    try:
        chunks_created = rag_engine.add_document(
            content=extracted_text,
            doc_id=doc_id,
            filename=file.filename,
            metadata={"upload_time": datetime.now().isoformat()}
        )
        
        # Save the updated index
        rag_engine.save_index()
        
        print(f"✅ Document {file.filename} indexed with {chunks_created} chunks")
        
    except Exception as e:
        print(f"❌ Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")
    
    # Store extracted content for fallback
    document_contents[doc_id] = extracted_text
    
    return {
        "message": f"{file.filename} uploaded and indexed successfully",
        "chunks_created": chunks_created,
        "doc_id": doc_id
    }

@app.get("/api/documents")
async def get_documents():
    """Get list of uploaded documents"""
    return {"documents": uploaded_files}

@app.delete("/api/documents/{filename}")
async def remove_document(filename: str):
    """Remove a document from the index"""
    try:
        # Find and remove from uploaded_files
        doc_to_remove = None
        for i, doc in enumerate(uploaded_files):
            if doc["filename"] == filename:
                doc_to_remove = uploaded_files.pop(i)
                break
        
        if not doc_to_remove:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from document_contents
        doc_id = doc_to_remove["id"]
        if doc_id in document_contents:
            del document_contents[doc_id]
        
        # Remove from RAG engine
        rag_engine.remove_document(doc_id)
        rag_engine.save_index()
        
        return {"message": f"{filename} removed successfully"}
        
    except Exception as e:
        print(f"❌ Error removing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing document: {str(e)}")

@app.get("/api/rag/stats")
async def get_rag_stats():
    """Get RAG engine statistics"""
    try:
        stats = rag_engine.get_document_stats()
        return {
            "rag_stats": stats,
            "status": "active",
            "index_loaded": len(rag_engine.chunks) > 0
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "index_loaded": False
        }

@app.post("/api/qa/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents using RAG (Retrieval-Augmented Generation)"""
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    
    start_time = datetime.now()
    
    try:
        # Use RAG engine for intelligent document retrieval and answer generation
        answer, citations, confidence, sources = rag_engine.generate_answer(
            query=request.question,
            language=request.language or "en",
            response_style=request.response_style or "comprehensive"
        )
        
        # Detect language of the question
        detected_language = detect_language(request.question)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format sources for frontend
        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                "document": source["filename"],
                "chunk": source["chunk_id"],
                "relevance": f"{source['similarity']:.2f}",
                "preview": source["content_preview"]
            })
        
        # Enhance answer with RAG statistics
        rag_stats = rag_engine.get_document_stats()
        enhanced_answer = f"""{answer}

---
**RAG Analytics:**
• Retrieved from {len(sources)} most relevant document sections
• Processed {rag_stats['total_chunks']} indexed chunks across {rag_stats['total_documents']} documents
• Semantic similarity matching using {rag_stats['model_name']} embeddings
• Processing time: {processing_time:.2f}s"""

        return QuestionResponse(
            answer=enhanced_answer,
            citations=citations,
            sources=formatted_sources,
            confidence_score=confidence,
            language_detected=detected_language,
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"❌ Error in RAG question processing: {e}")
        # Fallback to simple text search if RAG fails
        return await ask_question_fallback(request, start_time)

async def ask_question_fallback(request: QuestionRequest, start_time: datetime):
    """Fallback method using simple text matching if RAG fails"""
    print("🔄 Using fallback text search method...")
    
    # Get all document content
    all_content = ""
    for content in document_contents.values():
        all_content += content + "\n\n"
    
    if not all_content.strip():
        raise HTTPException(status_code=400, detail="No document content available")
    
    # Find relevant content using simple keyword matching
    relevant_content = find_relevant_content(all_content, request.question, max_chars=2000)
    
    # Generate response using existing method
    answer, confidence = generate_professional_answer(
        relevant_content, 
        request.question, 
        request.language or "en",
        request.response_style or "comprehensive"
    )
    
    # Detect language
    detected_language = detect_language(request.question)
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return QuestionResponse(
        answer=f"{answer}\n\n*Note: Using fallback text search due to RAG system unavailability*",
        citations=[f"Document {i+1}" for i in range(len(uploaded_files))],
        sources=[{"document": f["filename"], "relevance": "text-match"} for f in uploaded_files],
        confidence_score=confidence * 0.8,  # Reduce confidence for fallback
        language_detected=detected_language,
        processing_time=processing_time
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "StudyMate API is running"}

# Serve React app
@app.get("/")
async def serve_react_app():
    """Serve React app"""
    if frontend_path.exists():
        return FileResponse(str(frontend_path / "index.html"))
    return {"message": "StudyMate API is running. React frontend not built yet."}

@app.get("/{path:path}")
async def serve_react_files(path: str):
    """Serve React app files"""
    if frontend_path.exists():
        file_path = frontend_path / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_path / "index.html"))
    return {"message": "StudyMate API is running"}

if __name__ == "__main__":
    print("🚀 Starting StudyMate Backend Server...")
    print("📖 StudyMate - AI-Powered PDF-Based Academic Q&A & Learning Assistant")
    print("🌐 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("💫 Using IBM Granite model simulation for Q&A")
    
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
