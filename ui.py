import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any
import hashlib
import re
from googletrans import Translator
import requests

# Configure page
st.set_page_config(
    page_title="StudyMate - AI-Powered Learning Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .study-companion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .learning-path-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(240, 147, 251, 0.3);
    }
    
    .quiz-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(79, 172, 254, 0.3);
    }
    
    .achievement-badge {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        color: #8B4513;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
        box-shadow: 0 2px 8px rgba(252, 182, 159, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class StudyMateSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.translator = Translator()
        self.index = None
        self.chunks = []
        self.original_chunks = []  # Store original text before translation
        self.current_language = 'en'  # Default language
        self.translated_content = {}  # Cache for translated content
        self.study_sessions = []
        self.user_progress = {
            'questions_asked': 0,
            'concepts_learned': [],
            'study_time': 0,
            'achievements': [],
            'learning_streak': 0
        }
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish (Español)',
            'fr': 'French (Français)', 
            'de': 'German (Deutsch)',
            'it': 'Italian (Italiano)',
            'pt': 'Portuguese (Português)',
            'ru': 'Russian (Русский)',
            'ja': 'Japanese (日本語)',
            'ko': 'Korean (한국어)',
            'zh': 'Chinese (中文)',
            'ar': 'Arabic (العربية)',
            'hi': 'Hindi (हिन्दी)',
            'ta': 'Tamil (தமிழ்)',
            'te': 'Telugu (తెలుగు)',
            'bn': 'Bengali (বাংলা)',
            'ur': 'Urdu (اردو)',
            'th': 'Thai (ไทย)',
            'vi': 'Vietnamese (Tiếng Việt)',
            'nl': 'Dutch (Nederlands)',
            'sv': 'Swedish (Svenska)',
            'da': 'Danish (Dansk)',
            'no': 'Norwegian (Norsk)',
            'fi': 'Finnish (Suomi)',
            'pl': 'Polish (Polski)',
            'tr': 'Turkish (Türkçe)',
            'he': 'Hebrew (עברית)',
            'fa': 'Persian (فارسی)',
            'sw': 'Swahili (Kiswahili)',
            'af': 'Afrikaans'
        }
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return ""
    
    def translate_text(self, text, target_language='en', source_language='auto'):
        """Translate text using Google Translate"""
        try:
            if target_language == 'en' or target_language == source_language:
                return text
            
            # Check cache first
            cache_key = f"{hash(text)}_{target_language}"
            if cache_key in self.translated_content:
                return self.translated_content[cache_key]
            
            # Translate text in chunks to handle large content
            if len(text) > 4500:  # Google Translate has a limit
                chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
                translated_chunks = []
                
                for chunk in chunks:
                    if chunk.strip():
                        try:
                            result = self.translator.translate(
                                chunk, 
                                src=source_language, 
                                dest=target_language
                            )
                            translated_chunks.append(result.text)
                            time.sleep(0.1)  # Rate limiting
                        except Exception as e:
                            st.warning(f"Translation chunk failed: {str(e)}")
                            translated_chunks.append(chunk)  # Fallback to original
                
                translated_text = " ".join(translated_chunks)
            else:
                result = self.translator.translate(
                    text, 
                    src=source_language, 
                    dest=target_language
                )
                translated_text = result.text
            
            # Cache the translation
            self.translated_content[cache_key] = translated_text
            return translated_text
            
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            return text  # Return original text if translation fails
    
    def translate_document_content(self, target_language='en'):
        """Translate all document chunks to target language"""
        if not self.original_chunks:
            self.original_chunks = self.chunks.copy()  # Store original
        
        if target_language == 'en':
            self.chunks = self.original_chunks.copy()
            self.current_language = 'en'
            return
        
        if target_language == self.current_language:
            return  # Already in target language
        
        with st.spinner(f"Translating content to {self.supported_languages[target_language]}..."):
            translated_chunks = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(self.original_chunks):
                translated_chunk = self.translate_text(chunk, target_language)
                translated_chunks.append(translated_chunk)
                progress_bar.progress((i + 1) / len(self.original_chunks))
            
            self.chunks = translated_chunks
            self.current_language = target_language
            
            # Rebuild FAISS index with translated content
            if translated_chunks:
                self.build_faiss_index(translated_chunks)
            
            progress_bar.empty()
            st.success(f"✅ Content translated to {self.supported_languages[target_language]}!")
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks for better context"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only add substantial chunks
                chunks.append(chunk)
        return chunks
    
    def build_faiss_index(self, text_chunks):
        """Build FAISS index for semantic search"""
        embeddings = self.model.encode(text_chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks = text_chunks
    
    def semantic_search(self, query, top_k=3):
        """Perform semantic search using FAISS"""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx],
                    'score': float(score)
                })
        return results
    
    def generate_answer_with_llm(self, query, context_chunks, target_language='en'):
        """Generate answer using LLM with retrieved context"""
        # Simulated LLM response (replace with actual API call)
        context = "\n".join([chunk['text'] for chunk in context_chunks])
        
        # This would be replaced with actual OpenAI API call
        base_answer = f"""Based on the document content, here's what I found regarding your question: "{query}"

{context[:1000]}...

This information directly addresses your query by providing relevant context from the uploaded document. The semantic search identified the most relevant sections to help answer your question comprehensively."""
        
        # Translate the answer if needed
        if target_language != 'en':
            translated_answer = self.translate_text(base_answer, target_language)
            return translated_answer
        
        return base_answer
    
    def generate_study_companion_response(self, personality, query, context, target_language='en'):
        """Generate response from AI study companion with different personalities"""
        personalities = {
            "Encouraging Mentor": "I'm here to support your learning journey! Let's break this down together.",
            "Socratic Questioner": "Great question! But let me ask you - what do you think this means?",
            "Practical Teacher": "Let's focus on the key concepts and how you can apply them.",
            "Creative Explainer": "Imagine this concept as a story - let me paint you a picture!"
        }
        
        intro = personalities.get(personality, "Let me help you understand this better.")
        response = f"{intro}\n\n{context}\n\nWould you like me to explain any specific part in more detail?"
        
        # Translate response if needed
        if target_language != 'en':
            translated_response = self.translate_text(response, target_language)
            return translated_response
        
        return response
    
    def generate_adaptive_questions(self, topic, difficulty="medium"):
        """Generate adaptive quiz questions based on topic and difficulty"""
        questions = {
            "easy": [
                f"What is the main concept discussed about {topic}?",
                f"Can you identify key terms related to {topic}?",
                f"What are the basic components of {topic}?"
            ],
            "medium": [
                f"How does {topic} relate to other concepts in the document?",
                f"What are the implications of {topic} in practical applications?",
                f"Can you analyze the relationship between different aspects of {topic}?"
            ],
            "hard": [
                f"Critically evaluate the approach to {topic} presented in the document.",
                f"How would you synthesize {topic} with other theories or frameworks?",
                f"What are the potential limitations or criticisms of {topic}?"
            ]
        }
        return questions.get(difficulty, questions["medium"])

# Initialize session state
if 'study_mate' not in st.session_state:
    st.session_state.study_mate = StudyMateSystem()
if 'current_companion' not in st.session_state:
    st.session_state.current_companion = "Encouraging Mentor"
if 'study_session_active' not in st.session_state:
    st.session_state.study_session_active = False
if 'session_start_time' not in st.session_state:
    st.session_start_time = None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🧠 StudyMate - AI-Powered Learning Companion</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("📚 Study Dashboard")
        
        # Study session controls
        if not st.session_state.study_session_active:
            if st.button("🚀 Start Study Session", type="primary"):
                st.session_state.study_session_active = True
                st.session_state.session_start_time = time.time()
                st.success("Study session started!")
        else:
            if st.button("⏹️ End Study Session"):
                if st.session_state.session_start_time:
                    session_duration = time.time() - st.session_state.session_start_time
                    st.session_state.study_mate.user_progress['study_time'] += session_duration
                st.session_state.study_session_active = False
                st.success("Study session ended!")
        
        # Progress tracking
        st.subheader("📈 Your Progress")
        progress = st.session_state.study_mate.user_progress
        st.metric("Questions Asked", progress['questions_asked'])
        st.metric("Study Time (minutes)", f"{progress['study_time']/60:.1f}")
        st.metric("Learning Streak", progress['learning_streak'])
        
        # Achievements
        if progress['achievements']:
            st.subheader("🏆 Achievements")
            for achievement in progress['achievements']:
                st.markdown(f'<div class="achievement-badge">{achievement}</div>', unsafe_allow_html=True)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Document Upload", "❓ Q&A Chat", "🤖 AI Companions", "📊 Learning Analytics", "🧩 Adaptive Quiz"])
    
    with tab1:
        st.header("📄 Upload Your Study Materials")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload your academic PDFs to create a searchable knowledge base"
        )
        
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                all_text = ""
                for file in uploaded_files:
                    text = st.session_state.study_mate.extract_text_from_pdf(file)
                    all_text += f"\n\n--- Document: {file.name} ---\n{text}"
                
                if all_text.strip():
                    chunks = st.session_state.study_mate.chunk_text(all_text)
                    st.session_state.study_mate.build_faiss_index(chunks)
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} documents with {len(chunks)} text chunks!")
                    
                    # Show document preview
                    with st.expander("📖 Document Preview"):
                        st.write(f"**Total characters processed:** {len(all_text):,}")
                        st.write(f"**Text chunks created:** {len(chunks)}")
                        st.text_area("Sample content:", all_text[:500] + "...", height=200)
    
    with tab2:
        st.header("💬 Interactive Q&A Chat")
        
        if st.session_state.study_mate.index is None:
            st.warning("⚠️ Please upload and process documents first!")
        else:
            # Chat interface
            st.subheader("Ask questions about your documents:")
            
            # Chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.write(f"**🙋 You:** {question}")
                    st.write(f"**🤖 StudyMate:** {answer}")
                    st.divider()
            
            # New question input
            question = st.text_input("Your question:", placeholder="e.g., What are the main concepts in chapter 2?")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("Ask Question", type="primary")
            
            if ask_button and question:
                with st.spinner("Searching and generating answer..."):
                    # Semantic search
                    results = st.session_state.study_mate.semantic_search(question, top_k=3)
                    
                    if results:
                        # Generate answer
                        answer = st.session_state.study_mate.generate_answer_with_llm(question, results)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                        # Update progress
                        st.session_state.study_mate.user_progress['questions_asked'] += 1
                        
                        st.rerun()
                    else:
                        st.error("No relevant information found. Try rephrasing your question.")
    
    with tab3:
        st.header("🤖 AI Study Companions")
        
        st.markdown("""
        Choose your AI study companion! Each has a unique personality and teaching style:
        """)
        
        # Companion selection
        companions = {
            "Encouraging Mentor": "🌟 Supportive and motivating, helps build confidence",
            "Socratic Questioner": "🤔 Asks thought-provoking questions to deepen understanding",
            "Practical Teacher": "🎯 Focuses on real-world applications and key concepts",
            "Creative Explainer": "🎨 Uses analogies, stories, and creative examples"
        }
        
        selected_companion = st.selectbox(
            "Choose your study companion:",
            list(companions.keys()),
            index=list(companions.keys()).index(st.session_state.current_companion)
        )
        
        if selected_companion != st.session_state.current_companion:
            st.session_state.current_companion = selected_companion
            st.success(f"✨ {selected_companion} is now your study companion!")
        
        # Display companion info
        st.markdown(f'<div class="study-companion-card"><h3>{selected_companion}</h3><p>{companions[selected_companion]}</p></div>', unsafe_allow_html=True)
        
        # Companion chat
        if st.session_state.study_mate.index is not None:
            st.subheader(f"💬 Chat with {selected_companion}")
            
            companion_question = st.text_input("Ask your companion:", key="companion_chat")
            
            if st.button("Chat with Companion") and companion_question:
                results = st.session_state.study_mate.semantic_search(companion_question, top_k=2)
                if results:
                    context = results[0]['text'][:500] + "..."
                    response = st.session_state.study_mate.generate_study_companion_response(
                        selected_companion, companion_question, context
                    )
                    st.markdown(f"**{selected_companion}:** {response}")
    
    with tab4:
        st.header("📊 Learning Analytics Dashboard")
        
        # Create sample analytics data
        progress = st.session_state.study_mate.user_progress
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Questions", progress['questions_asked'], delta=1 if progress['questions_asked'] > 0 else 0)
        
        with col2:
            st.metric("Study Time", f"{progress['study_time']/60:.1f} min", delta="5.2 min")
        
        with col3:
            st.metric("Concepts Mastered", len(progress['concepts_learned']), delta=1)
        
        # Learning progress chart
        if progress['questions_asked'] > 0:
            # Sample data for demonstration
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            activity_data = np.random.poisson(3, 30)  # Random activity data
            
            df = pd.DataFrame({
                'Date': dates,
                'Questions_Asked': activity_data,
                'Study_Minutes': np.random.normal(25, 10, 30)
            })
            
            fig = px.line(df, x='Date', y=['Questions_Asked', 'Study_Minutes'], 
                         title="Learning Activity Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Learning path visualization
            st.subheader("🛤️ Your Learning Journey")
            learning_paths = [
                "Basic Concepts Understanding ✅",
                "Advanced Applications 🔄",
                "Critical Analysis 📋",
                "Synthesis & Integration ⏳"
            ]
            
            for i, path in enumerate(learning_paths):
                progress_val = min(100, (progress['questions_asked'] * 25) + (i * 10))
                st.markdown(f'<div class="learning-path-card">{path}</div>', unsafe_allow_html=True)
                st.progress(progress_val / 100)
    
    with tab5:
        st.header("🧩 Adaptive Quiz System")
        
        if st.session_state.study_mate.index is None:
            st.warning("⚠️ Please upload documents first to generate personalized quizzes!")
        else:
            st.subheader("📝 Personalized Quiz Generation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                topic = st.text_input("Topic for quiz:", placeholder="e.g., machine learning, photosynthesis")
            
            with col2:
                difficulty = st.selectbox("Difficulty Level:", ["easy", "medium", "hard"])
            
            if st.button("Generate Quiz Questions", type="primary"):
                if topic:
                    questions = st.session_state.study_mate.generate_adaptive_questions(topic, difficulty)
                    
                    st.markdown(f'<div class="quiz-card"><h4>🎯 {difficulty.title()} Level Quiz: {topic}</h4></div>', unsafe_allow_html=True)
                    
                    for i, question in enumerate(questions, 1):
                        st.write(f"**Question {i}:** {question}")
                        answer = st.text_area(f"Your answer for Q{i}:", key=f"quiz_answer_{i}")
                    
                    if st.button("Submit Quiz"):
                        st.success("Quiz submitted! Great job on completing the adaptive quiz.")
                        # Update achievements
                        new_achievement = f"Quiz Master - {topic} ({difficulty})"
                        if new_achievement not in st.session_state.study_mate.user_progress['achievements']:
                            st.session_state.study_mate.user_progress['achievements'].append(new_achievement)
                            st.balloons()
            
            # Quiz history and performance
            st.subheader("📈 Quiz Performance Analytics")
            
            # Sample performance data
            quiz_data = pd.DataFrame({
                'Quiz Topic': ['Machine Learning', 'Data Structures', 'Algorithms', 'Statistics'],
                'Score': [85, 92, 78, 88],
                'Difficulty': ['medium', 'easy', 'hard', 'medium'],
                'Date': ['2024-01-15', '2024-01-18', '2024-01-22', '2024-01-25']
            })
            
            fig = px.bar(quiz_data, x='Quiz Topic', y='Score', color='Difficulty',
                        title="Quiz Performance by Topic")
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **StudyMate Features:** Semantic Search | AI Companions | Adaptive Learning | Progress Tracking | Interactive Quizzes")

if __name__ == "__main__":
    main()