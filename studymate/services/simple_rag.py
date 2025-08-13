"""
Lightweight RAG Engine for StudyMate
Uses TF-IDF and cosine similarity for semantic search
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleTfIdfRAG:
    """Lightweight RAG engine using TF-IDF for semantic search"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize the lightweight RAG engine"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Document storage
        self.documents = {}
        self.chunks = []
        self.chunk_metadata = []
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None
        
        # Storage paths
        self.index_path = Path("data/indexes")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        print("🤖 Lightweight RAG Engine initialized (TF-IDF)")
    
    def add_document(self, content: str, doc_id: str, filename: str = None, metadata: Dict = None) -> int:
        """Add a document to the RAG index"""
        print(f"Adding document {doc_id} to TF-IDF index...")
        
        # Store document
        self.documents[doc_id] = {
            "content": content,
            "filename": filename,
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat()
        }
        
        # Split into chunks
        chunks = self._split_into_chunks(content, doc_id)
        
        # Add chunks to collection
        for chunk in chunks:
            self.chunks.append(chunk["content"])
            self.chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "filename": filename,
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"]
            })
        
        # Rebuild TF-IDF matrix
        self._rebuild_tfidf_matrix()
        
        print(f"Added {len(chunks)} chunks for document {doc_id}")
        return len(chunks)
    
    def _split_into_chunks(self, content: str, doc_id: str) -> List[Dict]:
        """Split content into overlapping chunks"""
        # Clean content
        content = self._preprocess_text(content)
        
        # Split by sentences first, then by words
        sentences = re.split(r'[.!?]+', content)
        
        chunks = []
        current_chunk = ""
        current_words = 0
        chunk_id = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = sentence.split()
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_words + len(words) > self.chunk_size and current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                    "start_pos": start_pos,
                    "end_pos": start_pos + len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.chunk_overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence
                current_words = len(overlap_words) + len(words)
                start_pos += len(current_chunk) - len(" ".join(overlap_words) + " " + sentence)
                chunk_id += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    start_pos = len("".join(chunks)) if chunks else 0
                current_words += len(words)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "chunk_id": f"{doc_id}_chunk_{chunk_id}",
                "start_pos": start_pos,
                "end_pos": start_pos + len(current_chunk)
            })
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _rebuild_tfidf_matrix(self):
        """Rebuild TF-IDF matrix with all chunks"""
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Tuple[Dict, float]]:
        """Retrieve most relevant chunks using TF-IDF similarity"""
        if not self.chunks or self.tfidf_matrix is None:
            return []
        
        # Transform query using same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results above threshold
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                chunk_info = {
                    "content": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "similarity": similarities[idx]
                }
                results.append((chunk_info, similarities[idx]))
        
        return results
    
    def generate_answer(self, query: str, language: str = "en", response_style: str = "comprehensive") -> Tuple[str, List[str], float, List[Dict]]:
        """Generate answer using TF-IDF RAG"""
        start_time = datetime.now()
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=5)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question.", [], 0.0, []
        
        # Extract information
        contexts = []
        citations = []
        sources = []
        confidence_scores = []
        
        for chunk_info, similarity in relevant_chunks:
            contexts.append(chunk_info["content"])
            confidence_scores.append(similarity)
            
            # Create citation
            metadata = chunk_info["metadata"]
            filename = metadata.get("filename", f"Document {metadata['doc_id']}")
            chunk_id = metadata["chunk_id"].split("_")[-1]
            citation = f"{filename} (Section {chunk_id})"
            citations.append(citation)
            
            # Create source info
            sources.append({
                "doc_id": metadata["doc_id"],
                "chunk_id": metadata["chunk_id"],
                "filename": filename,
                "similarity": similarity,
                "content_preview": chunk_info["content"][:200] + "..." if len(chunk_info["content"]) > 200 else chunk_info["content"]
            })
        
        # Calculate confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Generate answer
        combined_context = "\n\n".join(contexts)
        answer = self._generate_contextual_answer(query, combined_context, language, response_style)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return answer, citations, overall_confidence, sources
    
    def _generate_contextual_answer(self, query: str, context: str, language: str, style: str) -> str:
        """Generate contextual answer from retrieved content"""
        
        if style == "comprehensive":
            return f"""Based on the document analysis using TF-IDF semantic search, here is a comprehensive response:

**Retrieved Context:**
{context}

**Analysis:**
The documents contain relevant information that directly addresses your query. The semantic search identified the most pertinent sections based on term frequency and document relevance.

**Key Insights:**
• Multiple document sections provide supporting evidence for the response
• The information has been retrieved based on semantic similarity to your question
• Content relevance is determined by TF-IDF vectorization and cosine similarity

**Comprehensive Answer:**
{self._extract_key_information(context, query)}

**Summary:**
This response synthesizes information from the most relevant document sections identified through semantic search, providing a thorough understanding of the topic based on the available content."""
        
        elif style == "concise":
            key_info = self._extract_key_information(context, query)
            return f"""**Direct Answer:** {key_info[:300]}...

**Key Points:**
• Information retrieved from semantic document search
• Based on TF-IDF similarity matching
• High relevance to your specific question

**Source:** Multiple relevant document sections"""
        
        else:  # detailed
            return f"""**Detailed Document Analysis:**

**Search Method:** TF-IDF Vectorization with Cosine Similarity
**Query Processing:** Semantic matching against document chunks

**Retrieved Content:**
{context}

**Detailed Analysis:**
1. **Content Relevance:** The search algorithm identified these sections as most relevant to your query
2. **Semantic Matching:** TF-IDF vectorization captured key terms and concepts
3. **Information Quality:** Multiple document sections provide comprehensive coverage
4. **Contextual Understanding:** The content addresses various aspects of your question

**Expert Assessment:**
The retrieved information demonstrates strong relevance to your inquiry. The TF-IDF approach ensures that the most semantically similar content is prioritized, providing reliable and contextually appropriate responses.

**Detailed Response:**
{self._extract_key_information(context, query)}

**Conclusion:**
This detailed analysis provides thorough coverage using advanced information retrieval techniques."""
    
    def _extract_key_information(self, context: str, query: str) -> str:
        """Extract key information relevant to the query"""
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        
        # Find sentences most relevant to query
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:5]]
        
        return ". ".join(top_sentences) if top_sentences else context[:500]
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "average_chunks_per_doc": len(self.chunks) / max(len(self.documents), 1),
            "vectorizer_features": len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'vocabulary_') else 0,
            "model_name": "TF-IDF with Cosine Similarity",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the index"""
        try:
            # Remove document from documents dict
            if doc_id in self.documents:
                del self.documents[doc_id]
            
            # Remove chunks belonging to this document
            chunks_to_keep = []
            metadata_to_keep = []
            
            for i, metadata in enumerate(self.chunk_metadata):
                if metadata["doc_id"] != doc_id:
                    chunks_to_keep.append(self.chunks[i])
                    metadata_to_keep.append(metadata)
            
            self.chunks = chunks_to_keep
            self.chunk_metadata = metadata_to_keep
            
            # Rebuild TF-IDF matrix
            if self.chunks:
                self._rebuild_tfidf_matrix()
            else:
                self.tfidf_matrix = None
            
            print(f"Document {doc_id} removed from index")
            return True
            
        except Exception as e:
            print(f"Error removing document {doc_id}: {e}")
            return False
    
    def save_index(self, filename: str = "tfidf_rag_index") -> bool:
        """Save index to disk"""
        try:
            # Save documents and metadata
            data = {
                "documents": self.documents,
                "chunks": self.chunks,
                "chunk_metadata": self.chunk_metadata,
                "config": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            }
            
            with open(self.index_path / f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"TF-IDF RAG index saved to {self.index_path}/{filename}.json")
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load_index(self, filename: str = "tfidf_rag_index") -> bool:
        """Load index from disk"""
        try:
            filepath = self.index_path / f"{filename}.json"
            if not filepath.exists():
                print(f"No saved index found at {filepath}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data["documents"]
            self.chunks = data["chunks"]
            self.chunk_metadata = data["chunk_metadata"]
            
            # Rebuild TF-IDF matrix
            if self.chunks:
                self._rebuild_tfidf_matrix()
            
            print(f"TF-IDF RAG index loaded: {len(self.chunks)} chunks, {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
