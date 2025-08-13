import streamlit as st
from app.ui import pages
from app.core.state import init_session_state

st.set_page_config(page_title="StudyMate", layout="wide")
init_session_state()

PAGES = {
    "Upload & Index": pages.page_upload,
    "Ask Questions": pages.page_qa,
    "Summaries": pages.page_summary,
    "Flashcards": pages.page_flashcards,
    "Quizzes": pages.page_quiz,
    "Notes": pages.page_notes,
}

st.sidebar.title("StudyMate")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[choice]()
