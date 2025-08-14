"""
Advanced RAG Engine for StudyMate
Implements professional-grade features from the problem statement:
- Semantic Search Using FAISS and Embeddings
- Sentence Transformers for precise question matching
- IBM WatsonX integration for LLM-based answer generation
- Accurate text extraction and preprocessing with PyMuPDF
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging

# Core ML libraries
import faiss
from sentence_transformers import SentenceTransformer
import torch

# Text processing
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

# IBM Watson integration
import requests
from ibm_watson import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGEngine:
    """
    Professional RAG Engine implementing all requirements from the problem statement
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 100,
                 top_k: int = 5):
        """
        Initialize the Advanced RAG Engine
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            top_k: Number of top chunks to retrieve
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize core components
        self._initialize_models(embedding_model)
        self._initialize_storage()
        self._initialize_watson()
        
        logger.info("🚀 Advanced RAG Engine initialized successfully")
    
    def _initialize_models(self, embedding_model: str):
        """Initialize ML models and components"""
        try:
            # Sentence Transformer for embeddings
            logger.info(f"Loading SentenceTransformer model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # FAISS index for semantic search
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Text splitter for chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            # NLP components
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _initialize_storage(self):
        """Initialize storage components"""
        # Storage paths
        self.storage_path = Path("data/advanced_indexes")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Document storage
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = []
        
        # Load existing index if available
        self._load_index()
    
    def _initialize_watson(self):
        """Initialize IBM Watson integration"""
        # Load Watson credentials from environment variables
        self.watson_api_key = os.getenv('IBM_WATSON_API_KEY')
        self.watson_url = os.getenv('IBM_WATSON_URL', 'https://us-south.ml.cloud.ibm.com')
        self.watson_project_id = os.getenv('IBM_WATSON_PROJECT_ID')
        
        if not all([self.watson_api_key, self.watson_project_id]):
            logger.warning("IBM Watson credentials not found. Answer generation will use fallback method.")
            self.watson_enabled = False
        else:
            self.watson_enabled = True
            logger.info("IBM Watson integration initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract and preprocess text from PDF using PyMuPDF
        Implements: Accurate Text Extraction and Preprocessing
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean and preprocess text
                text = self._preprocess_text(text)
                if text.strip():
                    text_content.append(text)
            
            doc.close()
            full_text = "\n\n".join(text_content)
            
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess and clean extracted text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
        
        return text.strip()
    
    def chunk_document(self, text: str, doc_id: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk document into semantic segments
        Implements: Efficient content chunking for downstream processing
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_obj = {
                    'id': f"{doc_id}_chunk_{i}",
                    'text': chunk,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'metadata': metadata or {},
                    'created_at': datetime.now().isoformat()
                }
                chunk_objects.append(chunk_obj)
            
            logger.info(f"Created {len(chunk_objects)} chunks for document {doc_id}")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformers
        Implements: Semantic Search Using FAISS and Embeddings
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_document(self, pdf_path: str, doc_id: str, metadata: Dict = None) -> bool:
        """
        Add a document to the RAG system
        Complete pipeline: Extract -> Chunk -> Embed -> Index
        """
        try:
            logger.info(f"Adding document {doc_id} to advanced RAG system")
            
            # 1. Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # 2. Store document
            self.documents[doc_id] = {
                'id': doc_id,
                'text': text,
                'metadata': metadata or {},
                'added_at': datetime.now().isoformat(),
                'file_path': pdf_path
            }
            
            # 3. Chunk document
            chunks = self.chunk_document(text, doc_id, metadata)
            
            # 4. Generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            # 5. Add to FAISS index
            start_idx = len(self.chunks)
            self.faiss_index.add(embeddings)
            
            # 6. Store chunks and metadata
            self.chunks.extend(chunk_texts)
            self.chunk_metadata.extend(chunks)
            self.embeddings.extend(embeddings.tolist())
            
            # 7. Save index
            self._save_index()
            
            logger.info(f"Successfully added document {doc_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Perform semantic search using FAISS
        Implements: Retrieves the most relevant text chunks using SentenceTransformers and FAISS
        """
        try:
            if top_k is None:
                top_k = self.top_k
            
            if self.faiss_index.ntotal == 0:
                logger.warning("No documents in index")
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    result = {
                        'chunk': self.chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': float(score),
                        'doc_id': self.chunk_metadata[idx]['doc_id']
                    }
                    results.append(result)
            
            logger.info(f"Semantic search returned {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def generate_watson_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using IBM Watson's Mixtral-8x7B-Instruct model
        Implements: LLM-Based Answer Generation
        """
        try:
            if not self.watson_enabled:
                return self._generate_fallback_answer(query, context_chunks)
            
            # Prepare context
            context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks
            
            # Prepare prompt
            prompt = f"""Based on the following context from academic documents, please provide a comprehensive and accurate answer to the question.

Context:
{context}

Question: {query}

Answer: Please provide a detailed, well-structured answer based on the given context. If the context doesn't contain enough information to fully answer the question, please indicate what information is available and what might be missing."""

            # Watson API call
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._get_watson_token()}'
            }
            
            data = {
                "model_id": "ibm/mixtral-8x7b-instruct",
                "input": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "project_id": self.watson_project_id
            }
            
            response = requests.post(
                f"{self.watson_url}/ml/v1/text/generation?version=2023-05-29",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['results'][0]['generated_text']
                return answer.strip()
            else:
                logger.error(f"Watson API error: {response.status_code}")
                return self._generate_fallback_answer(query, context_chunks)
                
        except Exception as e:
            logger.error(f"Error generating Watson answer: {e}")
            return self._generate_fallback_answer(query, context_chunks)
    
    def _get_watson_token(self) -> str:
        """Get IBM Watson access token"""
        try:
            authenticator = IAMAuthenticator(self.watson_api_key)
            token = authenticator.token_manager.get_token()
            return token
        except Exception as e:
            logger.error(f"Error getting Watson token: {e}")
            raise
    
    def _generate_fallback_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate fallback answer when Watson is not available"""
        if not context_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
        
        # Simple extractive answer generation
        context = " ".join(context_chunks[:3])
        
        # Find sentences that might contain the answer
        sentences = sent_tokenize(context)
        query_words = set(word_tokenize(query.lower()))
        
        # Score sentences based on query word overlap
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((overlap, sentence))
        
        if scored_sentences:
            # Sort by score and return top sentences
            scored_sentences.sort(reverse=True)
            top_sentences = [sent for _, sent in scored_sentences[:3]]
            return " ".join(top_sentences)
        else:
            return f"Based on the uploaded documents: {context[:500]}..."
    
    def ask_question(self, query: str) -> Dict[str, Any]:
        """
        Main Q&A interface
        Implements: Conversational Q&A from Academic PDFs
        """
        try:
            logger.info(f"Processing question: {query}")
            
            # 1. Semantic search for relevant chunks
            search_results = self.semantic_search(query)
            
            if not search_results:
                return {
                    'query': query,
                    'answer': "I couldn't find relevant information in the uploaded documents to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # 2. Extract context from top results
            context_chunks = [result['chunk'] for result in search_results]
            
            # 3. Generate answer using Watson or fallback
            answer = self.generate_watson_answer(query, context_chunks)
            
            # 4. Prepare response
            sources = []
            for result in search_results:
                source = {
                    'doc_id': result['doc_id'],
                    'chunk': result['chunk'][:200] + "..." if len(result['chunk']) > 200 else result['chunk'],
                    'score': result['score']
                }
                sources.append(source)
            
            confidence = max([result['score'] for result in search_results]) if search_results else 0.0
            
            response = {
                'query': query,
                'answer': answer,
                'sources': sources,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Successfully generated answer")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'query': query,
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the RAG system"""
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found")
                return False
            
            # Remove document
            del self.documents[doc_id]
            
            # Remove associated chunks
            indices_to_remove = []
            for i, metadata in enumerate(self.chunk_metadata):
                if metadata['doc_id'] == doc_id:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                del self.chunks[i]
                del self.chunk_metadata[i]
                del self.embeddings[i]
            
            # Rebuild FAISS index
            self._rebuild_faiss_index()
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Successfully removed document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {doc_id}: {e}")
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index after document removal"""
        try:
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Add remaining embeddings
            if self.embeddings:
                embeddings_array = np.array(self.embeddings, dtype=np.float32)
                self.faiss_index.add(embeddings_array)
            
            logger.info("FAISS index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            raise
    
    def get_documents(self) -> List[Dict]:
        """Get list of all documents in the system"""
        return [
            {
                'id': doc_id,
                'metadata': doc_data['metadata'],
                'added_at': doc_data['added_at'],
                'chunks_count': len([m for m in self.chunk_metadata if m['doc_id'] == doc_id])
            }
            for doc_id, doc_data in self.documents.items()
        ]
    
    def _save_index(self):
        """Save the RAG index to disk"""
        try:
            # Save FAISS index
            faiss_path = self.storage_path / "faiss_index.index"
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'embeddings': self.embeddings,
                'embedding_model': self.embedding_model.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = self.storage_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self):
        """Load existing RAG index from disk"""
        try:
            faiss_path = self.storage_path / "faiss_index.index"
            metadata_path = self.storage_path / "metadata.json"
            
            if faiss_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.faiss_index = faiss.read_index(str(faiss_path))
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.documents = metadata.get('documents', {})
                self.chunks = metadata.get('chunks', [])
                self.chunk_metadata = metadata.get('chunk_metadata', [])
                self.embeddings = metadata.get('embeddings', [])
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                logger.info("No existing index found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Start fresh if loading fails
            self.documents = {}
            self.chunks = []
            self.chunk_metadata = []
            self.embeddings = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'total_embeddings': len(self.embeddings),
            'faiss_index_size': self.faiss_index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'embedding_model': self.embedding_model.model_name if hasattr(self.embedding_model, 'model_name') else "unknown",
            'watson_enabled': self.watson_enabled
        }

# Global instance
advanced_rag = None

def get_advanced_rag() -> AdvancedRAGEngine:
    """Get global RAG engine instance"""
    global advanced_rag
    if advanced_rag is None:
        advanced_rag = AdvancedRAGEngine()
    return advanced_rag
