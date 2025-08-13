import streamlit as st
from typing import Dict, Any

def init_session_state():
    defaults: Dict[str, Any] = {
        "documents": {},  # doc_id -> metadata
        "index": None,
        "embeddings_model": None,
        "retriever": None,
        "chat_history": [],
        "flashcards": [],
        "notes": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
