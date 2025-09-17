import torch  # Initialize PyTorch early to prevent torch.classes errors
import streamlit as st
from user_management import UserManager
from datetime import datetime
from dotenv import load_dotenv
import os
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
import time
from background_processor import get_background_processor
from datetime import timedelta
# from typing import Tuple  # Uncomment if needed for Python <3.9

# Load custom CSS
def load_css():
    try:
        with open("styles/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

def delete_file_chunks(qdrant_client, source_file: str):  # Remove typing hint 'tuple[bool, str]' if <3.9
    """Delete all chunks associated with a specific source file"""
    try:
        # First, get all points with this source file to count them
        scroll_result = qdrant_client.scroll(
            collection_name="legal_regulations",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file)
                    )
                ]
            ),
            limit=10000,  # Large limit to get all matching points
            with_payload=False,
            with_vectors=False
        )
        points = scroll_result[0]
        count = len(points)

        if count == 0:
            return False, "No chunks found for this file"

        # Delete all points with this source file
        qdrant_client.delete(
            collection_name="legal_regulations",
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file)
                    )
                ]
            )
        )
        return True, f"Deleted {count} chunks"
    except Exception as e:
        return False, str(e)

# Load environment variables
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin-secure-2024")
st.set_page_config(page_title="Admin Panel", page_icon="👨‍💼", layout="wide")
# Load CSS immediately after page config
load_css()

def init_admin_systems():
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    collection_name = "legal_regulations"
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    # Always ensure payload index for source_file field exists
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source_file",
            field_schema="keyword"
        )
    except Exception as e:
        # Index might already exist, which is fine
        if "already exists" not in str(e).lower():
            print(f"Warning: Could not create payload index: {e}")
    collection = client
    return user_manager, chunker, collection, embedding_model

def admin_login():
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    if not st.session_state.admin_authenticated:
        st.markdown("""
            <div class="login-card">
                <div class="brand-logo">👨‍💼</div>
                <h1 style="margin: 0; text-align: center;">Admin Control Panel</h1>
                <p style="text-align: center; color: var(--text-muted); margin-bottom: 2rem;">
                    Administrative Access Required
                </p>
            </div>
        """, unsafe_allow_html=True)
        with st.form("admin_login_form"):
            pwd = st.text_input(
                "Admin Password", 
                type="password",
                placeholder="Enter admin password"
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button("🔐 Access Admin Panel", use_container_width=True)
            if submit_button and pwd:
                if pwd == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    # Create admin session
                    if 'admin_session_id' not in st.session_state:
                        st.session_state.admin_session_id = str(uuid.uuid4())
                    # Create session in database
                    user_manager, _, _, _ = init_admin_systems()
                    user_manager.create_session("ADMIN", st.session_state.admin_session_id)
                    st.rerun()
                else:
                    st.error("Invalid password")
            elif submit_button:
                st.error("Please enter the admin password")
        return False
    return True

def process_uploaded_file(uploaded_file, chunker, qdrant_client, embedding_model):
    try:
        t = uploaded_file.type
        if t == "application/pdf":
            text = extract_pdf_text(uploaded_file)
        elif t == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_docx_text(uploaded_file)
        elif t == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        else:
            return False, f"Unsupported file type: {t}"
        if text.startswith("Error"):
            return False, text
        st.write(f"📊 File size: {len(text)} characters")
        st.write("🔍 First 500 characters:")
        st.text(text[:500])
        st.write(
            f"🏷️ XML detection: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}"
        )
        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        st.write(f"📦 Chunks created: {len(chunks)}")
        if not chunks:
            return False, "No chunks were created - check file format"
        chunk_texts = [ch['text'] for ch in chunks]
        st.write("🧠 Generating embeddings locally...")
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
        points = []
        now = datetime.now().isoformat()
        for i, (ch, embedding) in enumerate(zip(chunks, embeddings)):
            vector = embedding.tolist()
            payload = {
                'text': ch['text'],
                **ch['metadata'],
                'source_file': uploaded_file.name,
                'upload_date': now,
                'processed_by': 'admin'
            }
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)
        st.write("📤 Uploading vectors to Qdrant...")
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            qdrant_client.upsert(
                collection_name="legal_regulations",
                points=batch
            )
        return True, f"Processed {len(chunks)} chunks from {uploaded_file.name}"
    except Exception as e:
        return False, f"Error processing file: {e}"

def main():
    if not admin_login():
        st.stop()

    st.markdown("""
        <div class="dashboard-header">
            <h1 style="margin: 0; color: white;">👨‍💼 Admin Control Panel</h1>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                System Administration & Knowledge Base Management
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Check if we have an admin session ID
    if 'admin_session_id' not in st.session_state:
        st.session_state.admin_session_id = str(uuid.uuid4())

    # Validate admin session with extended timeout (48 hours)
    user_manager, _, _, _ = init_admin_systems()
    if not user_manager.is_session_valid(st.session_state.admin_session_id, hours_timeout=48):
        st.error("Your admin session has expired. Please log in again.")
        st.session_state.admin_authenticated = False
        st.rerun()

    um, chunker, coll, embedding_model = init_admin_systems()
    tab1, tab2 = st.tabs(["👥 Users", "📚 Knowledge Base"])
    # ... [rest of your code remains unchanged] ...

if __name__ == "__main__":
    main()
