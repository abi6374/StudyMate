# StudyMate

AI-Powered PDF-Based Academic Q&A & Learning Assistant.

## Core Features Implemented (Initial Skeleton)
- PDF upload (Streamlit UI) -> extraction via PyMuPDF
- Text chunking & preprocessing
- Embedding generation (Sentence Transformers) & FAISS index build
- Semantic retrieval
- LLM answer generation stub (IBM Watsonx / fallback local model)
- Multi-PDF management scaffolding

## Run (Dev)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Environment Variables
Create `.env`:
```
WATSONX_API_KEY=your_key
WATSONX_PROJECT_ID=your_project
WATSONX_URL=https://us-south.ml.cloud.ibm.com
MODEL_ID=mixtral-8x7b-instruct
```

## Roadmap
See `docs/ROADMAP.md`.
