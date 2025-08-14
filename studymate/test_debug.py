#!/usr/bin/env python3
"""Debug script to test TF-IDF RAG functionality"""

import sys
import traceback
from services.simple_rag import SimpleTfIdfRAG

def test_rag():
    """Test RAG functionality directly"""
    try:
        print("🔧 Testing TF-IDF RAG Engine...")
        
        # Initialize RAG engine
        rag_engine = SimpleTfIdfRAG(chunk_size=512, chunk_overlap=50)
        
        # Try to load existing index
        print("📚 Loading existing index...")
        rag_engine.load_index()
        
        print(f"📊 Loaded {len(rag_engine.chunks)} chunks")
        print(f"📊 Loaded {len(rag_engine.documents)} documents")
        
        # Test query
        query = "What are the main points?"
        print(f"🔍 Testing query: '{query}'")
        
        # Test retrieve_relevant_chunks
        print("📋 Testing retrieve_relevant_chunks...")
        chunks = rag_engine.retrieve_relevant_chunks(query, top_k=5, similarity_threshold=0.0001)
        print(f"✅ Retrieved {len(chunks)} chunks")
        
        # Test generate_answer
        print("💬 Testing generate_answer...")
        answer, citations, confidence, sources = rag_engine.generate_answer(query)
        print(f"✅ Generated answer: {answer[:100]}...")
        print(f"✅ Confidence: {confidence}")
        print(f"✅ Citations: {len(citations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📋 Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag()
    if success:
        print("🎉 RAG test completed successfully!")
    else:
        print("💥 RAG test failed!")
        sys.exit(1)
