__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import uuid
import time
from datetime import datetime, timedelta

import streamlit as st
import openai
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings

from user_management import UserManager
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Hide Streamlit header/footer and branding
st.markdown("""
<style>
  .stAppViewerButton, .stAppViewerIcon, .stAppViewerLink,
  .css-1avcm0n.e1fqkh3o2,
  .stAppDeployButton, .stDecoration, #MainMenu,
  footer, header {
    visibility: hidden !important;
    height: 0 !important;
    margin: 0;
    padding: 0;
  }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_systems():
    """
    Initialize OpenAI client, user manager, chunker, and Chroma collection
    using the new Chroma Settings API (no legacy config).
    """
    client = openai
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))

    # New Chroma client configuration (no legacy settings)
    settings = Settings()
    vector_client = chromadb.Client(settings=settings)
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

def get_session_id():
    """
    Generate or retrieve a unique session ID stored in Streamlit session_state.
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def check_authentication():
    """
    Display login form or verify existing session. Returns
    (authenticated_flag, client, user_manager, chunker, collection).
    """
    client, user_manager, chunker, collection = init_systems()

    if st.session_state.get('authenticated'):
        session_id = get_session_id()
        if user_manager.is_session_valid(session_id, hours_timeout=24):
            user_manager.update_session_activity(session_id)
            return True, client, user_manager, chunker, collection
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
        if st.form_submit_button("🚀 Access Assistant") and access_code:
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
    """
    Query the Chroma collection for relevant legal text chunks.
    """
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
    """
    Extract text from PDF, DOCX, or TXT, chunk it, and add to the Chroma collection.
    """
    try:
        t = uploaded_file.type
        if t == "application/pdf":
            text = extract_pdf_text(uploaded_file)
        elif t.endswith("wordprocessingml.document"):
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
        xml_flag = text.strip().startswith("<?xml") or "<code type=" in text
        st.write(f"🏷️ XML detection: {'XML' if xml_flag else 'Plain text'}")

        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        st.write(f"📦 Chunks created: {len(chunks)}")
        if not chunks:
            return False, "No chunks were created - check file format"

        docs, metas, ids = [], [], []
        now = datetime.now().isoformat()
        for ch in chunks:
            docs.append(ch['text'])
            metas.append({
                **ch['metadata'],
                'source_file': uploaded_file.name,
                'upload_date': now,
                'processed_by': 'admin'
            })
            ids.append(f"{uploaded_file.name}_{ch['metadata']['chunk_id']}")

        batch_size = 5000
        for i in range(0, len(docs), batch_size):
            collection.add(
                documents=docs[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

        return True, f"Processed {len(chunks)} chunks from {uploaded_file.name}"

    except Exception as e:
        return False, f"Error processing file: {e}"

def main_app():
    """
    Main application logic: authentication, chat interface, and file uploads.
    """
    authenticated, client, user_manager, chunker, collection = check_authentication()
    if not authenticated:
        st.stop()

    # Header and logout button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("# ⚖️ Elite Legal Compliance Assistant")
        st.markdown("*Powered by GPT-5 with maximum quality analysis*")
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.access_code = None
            st.success("👋 Logged out successfully")
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your legal compliance assistant. I specialize in NY, NJ, and CT employment law. Ask me a question!"
        }]

    # Sidebar: session and knowledge-base info
    with st.sidebar:
        st.markdown("### 👤 Session Info")
        if 'login_time' in st.session_state:
            duration = datetime.now() - st.session_state.login_time
            hrs, rem = divmod(int(duration.total_seconds()), 3600)
            mins, _ = divmod(rem, 60)
            st.write(f"**Active:** {hrs}h {mins}m")
            left = timedelta(hours=24) - duration
            if left.total_seconds() > 0:
                lh, lr = divmod(int(left.total_seconds()), 3600)
                lm, _ = divmod(lr, 60)
                st.write(f"**Auto-logout:** {lh}h {lm}m")

        st.markdown("### 🏆 Quality Settings")
        st.write("**Analysis**: Maximum")
        st.write("**Reasoning**: High")
        st.write("**Depth**: Comprehensive")

        st.markdown("### 📚 Knowledge Base")
        try:
            cnt = collection.count()
            st.write(f"**Legal Provisions:** {cnt}")
            st.write("**Jurisdictions:** NY, NJ, CT")
        except:
            st.write("**Status:** Initializing...")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user prompt
    if prompt := st.chat_input("Ask me about legal compliance requirements..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            prog = st.empty()
            bar = st.progress(0)

            prog.text("🔍 Searching legal knowledge base...")
            bar.progress(20)
            results = search_knowledge_base(collection, prompt, n_results=8)

            prog.text("🧠 Applying high-effort reasoning...")
            bar.progress(50)
            context = "\n\n".join(f"Legal Text: {doc}" for doc, _, _ in results) or "No relevant legal text found."
            system_prompt = f"""{LEGAL_COMPLIANCE_SYSTEM_PROMPT}
Available Legal Context:
{context}
User Question: {prompt}"""

            prog.text("⚖️ Generating structured response...")
            bar.progress(75)
            resp = client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"system","content":system_prompt}],
                temperature=0.7,
                max_tokens=1500
            )

            bar.progress(100)
            prog.text("✅ Analysis complete!")

            ai_response = resp.choices[0].message.content
            st.markdown(ai_response)
            st.session_state.messages.append({"role":"assistant","content":ai_response})

            # Show sources consulted
            if results:
                st.markdown("### 📚 Sources Consulted")
                for doc, meta, dist in results:
                    label = f"{meta.get('source_file','Unknown')} – Chunk {meta.get('chunk_id','N/A')}"
                    with st.expander(label, expanded=False):
                        st.code(doc, language="text")
                        st.write(meta)
                        st.write(f"Relevance: {dist:.3f}")

if __name__ == "__main__":
    main_app()
