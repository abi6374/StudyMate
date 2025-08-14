# 🎓 StudyMate - Dual AI Model Implementation Guide

StudyMate now supports **TWO AI MODEL OPTIONS** for enhanced document Q&A:

## 🤖 AI Model Options

### 1. **Local TF-IDF Model** (Default)
- ✅ **Privacy-focused**: All processing happens locally
- ✅ **Fast response times**: No API calls required
- ✅ **Works offline**: No internet required for Q&A
- ✅ **No API costs**: Completely free
- ⚡ **Perfect for**: Quick questions, basic document analysis

### 2. **Google Gemini AI** (Enhanced)
- 🧠 **Advanced AI**: Google's Gemini 1.5 Flash model
- 🎯 **Superior understanding**: Better context comprehension
- 📝 **Natural responses**: More human-like answers
- 🌐 **Requires**: Internet connection and API key
- 💰 **Cost**: Pay-per-use (very affordable for students)
- 🚀 **Perfect for**: Complex questions, detailed analysis, research

### 3. **Auto Mode** (Recommended)
- 🔄 **Intelligent fallback**: Uses Gemini if available, otherwise local
- 🎯 **Best of both worlds**: Combines speed and quality
- ⚙️ **Default setting**: Automatically optimizes for your setup

## 🚀 Quick Setup Guide

### Step 1: Install Dependencies
```bash
cd studymate
pip install -r requirements.txt
```

### Step 2: Configure AI Models

#### Option A: Local Model Only (No setup needed)
- Just run the application - local TF-IDF model works out of the box!

#### Option B: Enable Gemini AI (Recommended)
1. **Get Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a free account
   - Generate your API key

2. **Configure Environment**:
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Edit .env file and add your key:
   GEMINI_API_KEY=your_api_key_here
   ```

### Step 3: Start the Application
```bash
# Backend (Terminal 1)
cd studymate
python simple_server.py

# Frontend (Terminal 2)
cd studymate-frontend
npm start
```

## 🎯 How to Use

### 1. **Access the Application**
- Open your browser to `http://localhost:3000`
- Navigate to the **Q&A Tab**

### 2. **Choose Your AI Model**
- **Auto**: Let the system choose the best available model
- **Google Gemini AI**: Force use of Gemini (requires API key)
- **Local TF-IDF**: Force use of local model

### 3. **Upload Documents**
- Use the **Upload Tab** to add PDFs
- Documents are automatically indexed for both models

### 4. **Ask Questions**
- Type your question in the Q&A interface
- The system will show which AI model was used
- Compare responses between different models!

## 📊 Model Comparison

| Feature | Local TF-IDF | Gemini AI | Auto Mode |
|---------|--------------|-----------|-----------|
| **Setup** | None | API Key | API Key (optional) |
| **Cost** | Free | ~$0.001/query | Mixed |
| **Speed** | Very Fast | Fast | Optimized |
| **Quality** | Good | Excellent | Best Available |
| **Privacy** | 100% Local | Cloud-based | Mixed |
| **Offline** | ✅ Yes | ❌ No | Partial |

## 🔧 Advanced Configuration

### Gemini API Usage Tips
- **Free Tier**: 15 requests per minute, 1500 per day
- **Cost**: Very affordable for students (~$0.001 per question)
- **Quality**: Significantly better for complex questions

### Local Model Optimization
- **Chunk Size**: Adjust in `simple_server.py` (default: 512)
- **Document Types**: Works best with text-heavy PDFs
- **Speed**: Processes 1000+ documents efficiently

## 🌟 Example Workflows

### Research Papers
1. Upload multiple research papers
2. Use **Gemini AI** for complex analysis
3. Ask: "Compare the methodologies across these papers"

### Study Materials
1. Upload lecture notes and textbooks
2. Use **Auto Mode** for balanced performance
3. Ask: "Summarize the key concepts in Chapter 5"

### Quick Reference
1. Upload reference documents
2. Use **Local Model** for fast lookups
3. Ask: "What is the definition of X?"

## 🚨 Troubleshooting

### Gemini Not Working?
```bash
# Check your API key
echo $GEMINI_API_KEY

# Test connection
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  "https://generativelanguage.googleapis.com/v1/models"
```

### Local Model Issues?
- Ensure documents are uploaded
- Check console for indexing messages
- Restart server if index is corrupted

### Performance Optimization?
- **For Speed**: Use Local Model
- **For Quality**: Use Gemini AI
- **For Balance**: Use Auto Mode

## 📝 API Documentation

Access the interactive API docs at: `http://localhost:8000/docs`

### New Parameters:
- `ai_model`: "auto" | "local" | "gemini"
- `response_style`: "comprehensive" | "concise" | "detailed"

## 🎉 Success! You're Ready to Go!

Your StudyMate application now has dual AI model support! The interface will show which model was used for each response, allowing you to compare and choose the best option for your needs.
