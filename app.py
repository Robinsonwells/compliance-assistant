import streamlit as st
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import uuid
import time
from datetime import datetime, timedelta
from user_management import UserManager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldType, FieldIndex
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Load environment variables
load_dotenv()

@st.cache_resource
def init_systems():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    
    # Initialize local embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
    
    vector_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    collection_name = "legal_regulations"
    
    # Create collection if it doesn't exist
    try:
        vector_client.get_collection(collection_name)
    except Exception:
        vector_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            field_indexes=[
                FieldIndex(field_name="source_file", field_type=FieldType.KEYWORD)
            ]
        )
    
    collection = vector_client
    return client, user_manager, chunker, collection, embedding_model

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
    client, user_manager, chunker, collection, embedding_model = init_systems()
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        session_id = get_session_id()
        if user_manager.is_session_valid(session_id, hours_timeout=24):
            user_manager.update_session_activity(session_id)
            return True, client, user_manager, chunker, collection, embedding_model
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
    return False, None, None, None, None, None

def search_knowledge_base(qdrant_client, embedding_model, query, n_results=5):
    """Search the legal knowledge base"""
    try:
        # Generate embedding for the query using local model
        query_vector = embedding_model.encode([query])[0].tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_vector,
            limit=n_results,
            with_payload=True
        )
        
        # Convert results to match original format
        results = []
        for result in search_results:
            doc = result.payload.get('text', '')
            meta = {k: v for k, v in result.payload.items() if k != 'text'}
            dist = 1 - result.score  # Convert similarity to distance
            results.append((doc, meta, dist))
        
        return results
    except Exception:
        return []

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
        # Debug info
        st.write(f"📊 File size: {len(text)} characters")
        st.write("🔍 First 500 characters:")
        st.text(text[:500])
        st.write(f"🏷️ XML detection: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}")
        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        st.write(f"📦 Chunks created: {len(chunks)}")
        if not chunks:
            return False, "No chunks were created - check file format"
        
        # Collect all chunk texts for batch embedding generation
        chunk_texts = [ch['text'] for ch in chunks]
        
        # Generate embeddings locally in batch
        st.write("🧠 Generating embeddings locally...")
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
        
        # Prepare points for Qdrant
        points = []
        now = datetime.now().isoformat()
        
        for i, (ch, embedding) in enumerate(zip(chunks, embeddings)):
            vector = embedding.tolist()
            
            # Prepare payload
            payload = {
                'text': ch['text'],
                **ch['metadata'],
                'source_file': uploaded_file.name,
                'upload_date': now,
                'processed_by': 'admin'
            }
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upload points to Qdrant in batches
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

def main_app():
    authenticated, client, user_manager, chunker, collection, embedding_model = check_authentication()
    if not authenticated:
        st.stop()
    # Header and logout
    col1, col2 = st.columns([6,1])
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
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your legal compliance assistant. I specialize in NY, NJ, and CT employment law. Ask me a question!"
        }]
    with st.sidebar:
        st.markdown("### 👤 Session Info")
        if 'login_time' in st.session_state:
            duration = datetime.now() - st.session_state.login_time
            hrs, rem = divmod(int(duration.total_seconds()), 3600)
            mins, _ = divmod(rem, 60)
            st.write(f"**Active:** {hrs}h {mins}m")
            left = timedelta(hours=24) - duration
            if left.total_seconds()>0:
                lh, lr = divmod(int(left.total_seconds()),3600)
                lm,_=divmod(lr,60)
                st.write(f"**Auto-logout:** {lh}h {lm}m")
        st.markdown("### 🏆 Quality Settings")
        st.write("**Analysis**: Maximum")
        st.write("**Reasoning**: High")
        st.write("**Depth**: Comprehensive")
        st.markdown("### 📚 Knowledge Base")
        try:
            collection_info = collection.get_collection("legal_regulations")
            cnt = collection_info.points_count
            st.write(f"**Legal Provisions:** {cnt}")
            st.write("**Jurisdictions:** NY, NJ, CT")
        except:
            st.write("**Status:** Initializing...")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask me about legal compliance requirements..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            prog = st.empty(); bar = st.progress(0)
            prog.text("🔍 Searching legal knowledge base..."); bar.progress(20)
            results = search_knowledge_base(collection, embedding_model, prompt, n_results=8)
            prog.text("🧠 Applying high-effort reasoning..."); bar.progress(50)
            context = "\n\n".join(f"Legal Text: {doc}" for doc,_,_ in results) or "No relevant legal text found."
            system_prompt = f"""{LEGAL_COMPLIANCE_SYSTEM_PROMPT}
Available Legal Context:
{context}
User Question: {prompt}"""
            prog.text("⚖️ Generating structured response..."); bar.progress(75)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            bar.progress(100); prog.text("✅ Analysis complete!")
            ai_response = response.choices[0].message.content
            st.markdown(ai_response)
            st.session_state.messages.append({"role":"assistant","content":ai_response})
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
