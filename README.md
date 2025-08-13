# StudyMate - AI-Powered PDF Q&A System

StudyMate is an intelligent academic assistant that uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from your PDF documents. Built with FastAPI backend and React frontend, it features semantic search, document indexing, and multilingual support.

## 🚀 Features

### Core Functionality
- **PDF Upload & Processing**: Upload and automatically extract text from PDF documents
- **Intelligent Q&A**: Ask questions and get contextual answers based on your documents
- **RAG Implementation**: TF-IDF based semantic search with document chunking
- **Real-time Processing**: Fast document indexing and query responses
- **Citation System**: Answers include source references and confidence scores

### Advanced Features
- **Multilingual Support**: 5+ languages (English, Spanish, French, German, Italian)
- **Response Styles**: Comprehensive, concise, or detailed answer formats
- **Document Management**: Upload, index, and remove documents dynamically
- **Confidence Scoring**: AI confidence levels for answer reliability
- **Semantic Search**: TF-IDF vectorization with cosine similarity matching

### User Interface
- **Modern React Frontend**: Material-UI components with responsive design
- **Authentication System**: Login/register with session management
- **Interactive Dashboard**: Multiple tabs for different functionalities
- **File Management**: Drag-and-drop upload with progress indicators
- **Real-time Feedback**: Toast notifications and loading states

## 🏗️ Architecture

### Backend (FastAPI)
- **RAG Engine**: `services/simple_rag.py` - TF-IDF based semantic search
- **API Server**: `simple_server.py` - RESTful endpoints with CORS support
- **PDF Processing**: PyMuPDF for text extraction
- **Document Storage**: Local file system with metadata tracking

### Frontend (React + TypeScript)
- **Modern Stack**: React 18, TypeScript, Material-UI
- **State Management**: Context API for auth, theme, and language
- **Routing**: React Router for navigation
- **HTTP Client**: Fetch API for backend communication

### Data Flow
```
PDF Upload → Text Extraction → Document Chunking → TF-IDF Vectorization → Index Storage
Query Input → Semantic Search → Context Retrieval → Answer Generation → Response
```

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**

### Backend Setup
```bash
cd studymate
pip install -r requirements.txt
python simple_server.py
```

### Frontend Setup
```bash
cd studymate-frontend
npm install
npm start
```

## 🔧 Configuration

### Backend Configuration
- **Port**: 8000 (configurable in `simple_server.py`)
- **Upload Directory**: `data/uploads/`
- **Index Storage**: `data/indexes/`
- **Chunk Size**: 512 tokens (configurable)
- **Chunk Overlap**: 50 tokens (configurable)

### Frontend Configuration
- **API Base URL**: `http://localhost:8000`
- **Development Port**: 3000
- **CORS**: Configured for cross-origin requests

## 📚 API Endpoints

### Document Management
- `POST /api/documents/upload` - Upload and index PDF
- `GET /api/documents` - List uploaded documents
- `DELETE /api/documents/{filename}` - Remove document

### Q&A System
- `POST /api/qa/ask` - Ask questions about documents
- `GET /api/health` - Health check endpoint

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration

## 🧠 RAG Implementation Details

### Document Processing
1. **Text Extraction**: PyMuPDF extracts text from PDFs
2. **Preprocessing**: Text cleaning and normalization
3. **Chunking**: Smart splitting with configurable overlap
4. **Vectorization**: TF-IDF vector creation for semantic search

### Query Processing
1. **Query Vectorization**: Convert user questions to TF-IDF vectors
2. **Similarity Search**: Cosine similarity to find relevant chunks
3. **Context Assembly**: Combine top-k relevant passages
4. **Answer Generation**: Contextual response with citations

### Performance Optimizations
- **Lightweight Architecture**: TF-IDF instead of heavy transformer models
- **Efficient Storage**: JSON-based index persistence
- **Fast Retrieval**: Optimized cosine similarity computation
- **Memory Management**: Chunked processing for large documents

## 🌍 Multilingual Support

Supported languages with localized responses:
- **English** (en)
- **Spanish** (es)
- **French** (fr)
- **German** (de)
- **Italian** (it)

## 🎯 Usage Examples

### Basic Q&A Flow
1. **Upload PDF**: Drag and drop your academic PDF
2. **Wait for Processing**: Document gets indexed automatically
3. **Ask Questions**: Type your question in natural language
4. **Get Answers**: Receive contextual answers with citations

### Advanced Features
- **Multiple Documents**: Upload multiple PDFs for cross-document Q&A
- **Response Customization**: Choose comprehensive, concise, or detailed answers
- **Language Selection**: Get responses in your preferred language
- **Source Verification**: Check citations and confidence scores

## 🧪 Testing

### Backend Testing
```bash
cd studymate
python test_upload.py  # Test upload functionality
```

### Manual Testing
1. Upload a PDF through the web interface
2. Ask questions about the document content
3. Verify answers include proper citations
4. Test document removal functionality

## 🚀 Deployment

### Local Development
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- API Docs: `http://localhost:8000/docs`

### Production Considerations
- Configure environment variables
- Set up proper CORS policies
- Implement production database
- Add authentication middleware
- Configure file storage (cloud)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- **Advanced RAG**: Implement dense vector embeddings
- **Database Integration**: PostgreSQL with vector extensions
- **Cloud Storage**: AWS S3 or Google Cloud Storage
- **User Management**: Enhanced authentication and authorization
- **Analytics**: Usage statistics and performance metrics
- **Mobile App**: React Native companion app

## 🐛 Troubleshooting

### Common Issues
1. **Upload Failures**: Check CORS configuration and file size limits
2. **Backend Connection**: Verify both servers are running
3. **PDF Processing**: Ensure PDFs contain extractable text
4. **Memory Issues**: Adjust chunk size for large documents

### Debug Mode
Enable detailed logging by setting debug flags in the configuration files.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review API documentation at `/docs`

---

**StudyMate** - Making academic research smarter with AI-powered document understanding.
