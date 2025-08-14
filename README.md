# StudyMate: Professional AI-Powered PDF-Based Q&A System for Students

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

StudyMate is an advanced AI-powered academic assistant that enables students to interact with their study materials—such as textbooks, lecture notes, and research papers—in a conversational, question-answering format. Instead of passively reading large PDF documents or relying on manual searches for specific information, users can simply upload one or more PDFs and ask natural-language questions. StudyMate responds with direct, well-contextualized answers, referenced from the source content.

## ✨ Key Features

### 🎯 **Professional Implementation Matching Problem Statement Requirements**

1. **🤖 Conversational Q&A from Academic PDFs**
   - Natural language question interface
   - Contextual answers grounded in uploaded documents
   - Interactive conversation history
   - Professional UI with Material-UI components

2. **📄 Accurate Text Extraction and Preprocessing**
   - **PyMuPDF integration** for high-quality PDF text extraction
   - Advanced text cleaning and preprocessing pipelines
   - Handles complex academic document structures
   - OCR error correction and text normalization

3. **🔍 Semantic Search Using FAISS and Embeddings**
   - **SentenceTransformers** for state-of-the-art embeddings
   - **FAISS vector database** for lightning-fast similarity search
   - Cosine similarity matching for precise question matching
   - Top-k retrieval with confidence scoring

4. **🧠 LLM-Based Answer Generation**
   - **IBM Watson Mixtral-8x7B-Instruct** model integration
   - Professional API authentication and error handling
   - Fallback answer generation for offline scenarios
   - Context-aware response generation

5. **💻 User-Friendly Local Interface**
   - **React + TypeScript** frontend with modern UI
   - **FastAPI** backend with comprehensive API documentation
   - Real-time document upload and processing
   - Source attribution and confidence scoring

## 🛠 Technologies & Architecture

### Backend Stack
- **Python 3.8+** - Core backend language
- **FastAPI** - Modern, fast web framework with automatic API docs
- **SentenceTransformers** - State-of-the-art text embeddings
- **FAISS** - Facebook AI Similarity Search for vector operations
- **PyMuPDF** - Professional PDF text extraction
- **IBM Watson ML** - Enterprise-grade LLM integration
- **LangChain** - Advanced text processing and chunking

### Frontend Stack
- **React 18** - Modern frontend framework
- **TypeScript** - Type-safe development
- **Material-UI (MUI)** - Professional component library
- **React Router** - Navigation and routing
- **Context API** - State management

### AI/ML Components
- **Embedding Model**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Vector Store**: FAISS with Inner Product similarity
- **LLM**: IBM Watson Mixtral-8x7B-Instruct
- **Text Processing**: LangChain RecursiveCharacterTextSplitter
- **NLP**: spaCy + NLTK for advanced text processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- IBM Watson Machine Learning account (for LLM features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abi6374/StudyMate.git
   cd StudyMate
   ```

2. **Run the automated setup (Recommended)**
   ```bash
   cd studymate
   python setup_professional.py
   ```

3. **Manual setup (Alternative)**
   ```bash
   # Backend setup
   cd studymate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   
   # Frontend setup
   cd ../studymate-frontend
   npm install
   ```

4. **Configure IBM Watson credentials**
   ```bash
   # Copy environment template
   cp env_template.txt .env
   
   # Edit .env file with your Watson credentials
   nano .env
   ```

5. **Start the application**
   ```bash
   # Terminal 1: Start backend
   cd studymate
   python professional_server.py
   
   # Terminal 2: Start frontend
   cd studymate-frontend
   npm start
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/health

## 🔧 Configuration

### IBM Watson Setup

1. **Create Watson Machine Learning Service**
   - Go to [IBM Cloud](https://cloud.ibm.com/catalog/services/watson-machine-learning)
   - Create a new Watson Machine Learning instance
   - Create service credentials

2. **Get Project ID**
   - Create a project in Watson Studio
   - Copy the project ID from project settings

3. **Configure Environment**
   ```bash
   IBM_WATSON_API_KEY=your_api_key_here
   IBM_WATSON_URL=https://us-south.ml.cloud.ibm.com
   IBM_WATSON_PROJECT_ID=your_project_id_here
   ```

### Advanced Configuration

```bash
# Model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=ibm/mixtral-8x7b-instruct
CHUNK_SIZE=512
CHUNK_OVERLAP=100
TOP_K_RESULTS=5

# Performance settings
UPLOAD_MAX_SIZE=50MB
LOG_LEVEL=INFO
```

## 📊 API Documentation

### Core Endpoints

```bash
# Upload document
POST /api/documents/upload
Content-Type: multipart/form-data

# Ask question
POST /api/ask
{
  "question": "What are the main concepts in this document?"
}

# List documents
GET /api/documents

# Delete document
DELETE /api/documents/{doc_id}

# System statistics
GET /api/stats

# Semantic search
POST /api/search
{
  "question": "search query"
}
```

### Response Format

```json
{
  "query": "Your question",
  "answer": "AI-generated answer with context",
  "sources": [
    {
      "doc_id": "document_identifier",
      "chunk": "relevant text excerpt",
      "score": 0.95
    }
  ],
  "confidence": 0.89,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🧪 Testing

### Backend Testing
```bash
cd studymate
python test_upload.py  # Test document upload
pytest tests/         # Run full test suite
```

### Frontend Testing
```bash
cd studymate-frontend
npm test              # Run React tests
npm run test:coverage # Coverage report
```

## 📁 Project Structure

```
StudyMate/
├── studymate/                    # Backend application
│   ├── services/
│   │   ├── advanced_rag.py     # Professional RAG engine
│   │   ├── simple_rag.py       # Lightweight fallback
│   │   └── watsonx_client.py   # IBM Watson integration
│   ├── professional_server.py  # Main FastAPI server
│   ├── requirements.txt        # Python dependencies
│   └── setup_professional.py   # Automated setup script
├── studymate-frontend/          # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard/
│   │   │   │   ├── QATabProfessional.tsx  # Professional Q&A interface
│   │   │   │   └── UploadTab.tsx          # Document upload
│   │   │   └── Layout/
│   │   └── contexts/           # React context providers
│   ├── package.json           # Node.js dependencies
│   └── public/               # Static assets
├── data/                     # Data storage
│   ├── uploads/             # Uploaded PDFs
│   └── advanced_indexes/    # FAISS indexes and metadata
└── README.md               # This file
```

## 🔒 Security Features

- **Input validation** on all API endpoints
- **File type restrictions** (PDF only)
- **Size limits** on uploads
- **CORS configuration** for secure frontend-backend communication
- **Environment variable** protection for sensitive credentials
- **Error handling** without exposing internal details

## 🚀 Performance Optimizations

- **FAISS vector indexing** for sub-millisecond similarity search
- **Efficient chunking** with overlap for context preservation
- **Normalized embeddings** for optimized cosine similarity
- **Background processing** for large document uploads
- **Persistent indexes** for fast application restarts
- **Connection pooling** for database operations

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **FAISS installation issues**
   ```bash
   pip install faiss-cpu --no-cache-dir
   ```

3. **Watson authentication errors**
   - Verify API key and project ID in .env
   - Check IBM Cloud service status
   - Ensure proper permissions on Watson ML service

4. **Frontend build errors**
   ```bash
   cd studymate-frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python professional_server.py

# Frontend development mode
cd studymate-frontend
npm run dev
```

## 📈 Roadmap

- [ ] **Multi-language support** (Spanish, French, German)
- [ ] **Advanced analytics** dashboard
- [ ] **Document summarization** features
- [ ] **Export capabilities** (PDF, DOCX)
- [ ] **Collaborative features** (shared documents)
- [ ] **Mobile app** (React Native)
- [ ] **Cloud deployment** (Docker, Kubernetes)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Watson** for enterprise-grade LLM capabilities
- **Hugging Face** for SentenceTransformers
- **Facebook Research** for FAISS vector search
- **FastAPI team** for the excellent web framework
- **React team** for the frontend framework
- **Material-UI** for professional UI components

## 📞 Support

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/abi6374/StudyMate/issues)
- **Documentation**: Check the `/docs` endpoint for API documentation
- **Community**: Join our discussions in GitHub Discussions

---

**Made with ❤️ for students worldwide**

Transform your study experience with AI-powered document interaction!
