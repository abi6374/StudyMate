import streamlit as st
from pathlib import Path
from app.services.pdf_loader import extract_text_chunks
from app.services.embeddings import embed_texts
from app.services.vector_store import FaissVectorStore, StoredChunk
from app.services.watsonx_client import build_prompt, generate_answer
import numpy as np

UPLOAD_DIR = Path("data/uploads")
INDEX_PATH = Path("data/indexes/main_index")


def ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def page_upload():
    ensure_dirs()
    st.header("Upload & Index PDFs")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        any_added = False
        for uf in uploaded_files:
            file_path = UPLOAD_DIR / uf.name
            with open(file_path, 'wb') as f:
                f.write(uf.read())
            st.success(f"Saved {uf.name}")
            # extract -> embed -> add to index
            doc_id = uf.name.replace('.pdf', '')
            chunks = extract_text_chunks(file_path, doc_id)
            texts = [c.text for c in chunks]
            if not texts:
                st.warning(f"No text extracted for {uf.name}")
                continue
            embeddings = embed_texts(texts)
            dim = embeddings.shape[1]
            if st.session_state.index is None:
                st.session_state.index = FaissVectorStore(dim=dim, store_path=INDEX_PATH)
            stored_chunks = [StoredChunk(chunk_id=c.chunk_id, text=c.text, metadata={"doc_id": c.doc_id, "page": c.page_number, **c.meta}) for c in chunks]
            st.session_state.index.add(embeddings, stored_chunks)
            any_added = True
        if any_added and st.session_state.index is not None:
            st.session_state.index.persist()
            st.success("Index updated & persisted.")
        elif not any_added:
            st.info("Uploads processed but no extractable text found.")


def page_qa():
    st.header("Ask Questions")
    if st.session_state.index is None:
        st.info("Please upload and index PDFs first.")
        return
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        # embed question using same model
        q_emb = embed_texts([question])
        results = st.session_state.index.search(q_emb[0], k=5)
        contexts = []
        citations = []
        for chunk, score in results:
            contexts.append(chunk.text)
            citations.append(f"{chunk.metadata.get('doc_id')} p.{chunk.metadata.get('page')}")
        prompt = build_prompt(question, contexts)
        answer = generate_answer(prompt)
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("**Citations:** " + "; ".join(citations))
        st.session_state.chat_history.append({"q": question, "a": answer, "citations": citations})
    if st.session_state.chat_history:
        with st.expander("Conversation History"):
            for turn in reversed(st.session_state.chat_history):
                st.markdown(f"**Q:** {turn['q']}")
                st.markdown(f"**A:** {turn['a']}")
                st.caption("; ".join(turn['citations']))


def page_summary():
    st.header("Summaries (Coming Soon)")
    st.info("Feature stub.")


def page_flashcards():
    st.header("Flashcards (Coming Soon)")
    st.info("Feature stub.")


def page_quiz():
    st.header("Quiz Generation (Coming Soon)")
    st.info("Feature stub.")


def page_notes():
    st.header("Notes (Coming Soon)")
    st.info("Feature stub.")
