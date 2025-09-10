__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from dotenv import load_dotenv
import os
import uuid
import time
from datetime import datetime, timedelta
from user_management import UserManager
import chromadb
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def init_systems():
    client = openai
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    # Use new default client; will create a fresh database automatically
    vector_client = chromadb.Client()

    collection = vector_client.get_or_create_collection(
        name="legal_regulations",
        metadata={"description": "Multi-state employment law regulations"}
    )
    return client, user_manager, chunker, collection

# Page config
st.set_page_config(
    page_title="Legal Compliance Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Hide Streamlit branding
st.markdown("""
<style>
    .stAppDeployButton {display:none;}
    .stDecoration {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def check_authentication():
    """Check if user is authenticated"""
    client, user_manager, chunker, collection = init_systems()
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        session_id = get_session_id()
        if user_manager.is_session_valid(session_id, hours_timeout=24):
            user_manager.update_session_activity(session_id)
            return True, client, user_manager, chunker, collection
        else:
            st.session_state.authenticated = False
            st.error("🕐 Your session has expired. Please log in again.")
            time.sleep(2)
            st.rerun()

    st.markdown("# 🔐 Legal Compliance Assistant")
    st.markdown("**Professional AI-Powered Legal Analysis**")
    st.markdown("---")
    with st.form("login_form"):
        st.markdown("### Enter Your Access Code")
        access_code = st.text_input(
            "Access Code",
            type="password",
            placeholder="Enter your access code...",
            help="Contact your administrator for access"
        )
        submit_button = st.form_submit_button("🚀 Access Assistant")
        if submit_button and access_code:
            if user_manager.validate_access_code(access_code):
                session_id = get_session_id()
                user_manager.create_session(access_code, session_id)
                user_manager.update_last_login(access_code)
                st.session_state.authenticated = True
                st.session_state.access_code = access_code
                st.session_state.login_time = datetime.now()
                st.success("✅ Access granted! Loading assistant...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid or expired access code")
                time.sleep(2)

    with st.expander("ℹ️ Session Information"):
        st.write("• Sessions expire after 24 hours of inactivity")
        st.write("• Your session renews with each interaction")
        st.write("• Access can be revoked by administrator")

    return False, None, None, None, None

def search_knowledge_base(collection, query, n_results=5):
    """Search the legal knowledge base"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        dists = results.get('distances', [[]])[0]
        return list(zip(docs, metas, dists))
    except Exception:
        return []

def process_uploaded_file(uploaded_file, chunker, collection):
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

        # Debug info
        st.write(f"📊 File size: {len(text)} characters")
        st.write("🔍 First 500 characters:")
        st.text(text[:500])
        st.write(f"🏷️ XML detection: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}")

        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        st.write(f"📦 Chunks created: {len(chunks)}")
        if not chunks:
            return False, "No chunks were created
