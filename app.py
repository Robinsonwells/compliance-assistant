import streamlit as st
import os
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from user_management import UserManager
import chromadb
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Load environment variables
load_dotenv()

@st.cache_resource
def init_systems():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    vector_client = chromadb.PersistentClient(path="./legal_compliance_db")
    collection = vector_client.get_or_create_collection(
        name="legal_regulations",
        metadata={"description": "Multi-state employment law regulations"}
    )
    return client, user_manager, chunker, collection

# Streamlit page configuration and hide branding
st.set_page_config(page_title="Legal Compliance Assistant", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
    .stAppDeployButton, .stDecoration, #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def check_authentication():
    client, user_manager, chunker, collection = init_systems()
    if st.session_state.get('authenticated'):
        sid = get_session_id()
        if user_manager.is_session_valid(sid, hours_timeout=24):
            user_manager.update_session_activity(sid)
            return True, client, user_manager, chunker, collection
        st.session_state.authenticated = False
        st.error("🕐 Session expired. Please log in again.")
        time.sleep(2)
        st.rerun()

    st.markdown("# 🔐 Legal Compliance Assistant")
    st.markdown("**Professional AI-Powered Legal Analysis**")
    st.markdown("---")
    with st.form("login_form"):
        access_code = st.text_input("Access Code", type="password", placeholder="Enter access code")
        submit = st.form_submit_button("🚀 Access Assistant")
        if submit and access_code:
            if user_manager.validate_access_code(access_code):
                sid = get_session_id()
                user_manager.create_session(access_code, sid)
                user_manager.update_last_login(access_code)
                st.session_state.authenticated = True
                st.session_state.login_time = datetime.now()
                st.success("✅ Access granted!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid or expired access code")
                time.sleep(2)

    with st.expander("ℹ️ Session Information"):
        st.write("• Sessions expire after 24h inactivity")
        st.write("• Session renews each interaction")

    return False, None, None, None, None

def search_knowledge_base(collection, query, n_results=5):
    try:
        r = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        docs, metas, dists = r['documents'][0], r['metadatas'][0], r['distances'][0]
        return list(zip(docs, metas, dists))
    except:
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

        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        now = datetime.now().isoformat()
        docs, metas, ids = [], [], []
        for ch in chunks:
            docs.append(ch['text'])
            metas.append({
                **ch['metadata'],
                'source_file': uploaded_file.name,
                'upload_date': now,
                'processed_by': 'admin'
            })
            ids.append(f"{uploaded_file.name}_{ch['metadata']['chunk_id']}")
        for i in range(0, len(docs), 5000):
            collection.add(
                documents=docs[i:i+5000],
                metadatas=metas[i:i+5000],
                ids=ids[i:i+5000]
            )
        return True, f"Processed {len(chunks)} chunks from {uploaded_file.name}"
    except Exception as e:
        return False, f"Error processing file: {e}"

def main_app():
    authenticated, client, user_manager, chunker, collection = check_authentication()
    if not authenticated:
        st.stop()

    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown("# ⚖️ Elite Legal Compliance Assistant")
        st.markdown("*Powered by GPT-5 with maximal depth and verbosity*")
    with c2:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

    st.markdown("---")
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I specialize in NY, NJ, and CT employment law. Ask me anything."
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about legal compliance..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            prog = st.empty()
            bar = st.progress(0)
            prog.text("🔍 Searching legal knowledge base..."); bar.progress(20)
            results = search_knowledge_base(collection, prompt, n_results=8)
            prog.text("🧠 Preparing context..."); bar.progress(50)
            context = "\n\n".join(f"{doc}" for doc,_,_ in results) or "No context found."
            system_prompt = (
                f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\n\n"
                f"Context:\n{context}\n\nUser: {prompt}"
            )
            prog.text("⚖️ Generating response with GPT-5..."); bar.progress(75)

            response = client.responses.create(
                model="gpt-5",
                input=system_prompt,
                reasoning={"effort":"maximal"},
                text={"verbosity":"maximal"},
                sampling={"temperature":0.0, "top_p":1.0, "frequency_penalty":0.0, "presence_penalty":0.0},
                max_tokens=128000,
                stream=False
            )

            bar.progress(100)
            prog.text("✅ Done")
            reply = response.output_text
            st.markdown(reply)
            st.session_state.messages.append({"role":"assistant","content":reply})

            if results:
                st.markdown("### 📚 Sources Consulted")
                for doc, meta, dist in results:
                    label = f"{meta.get('source_file','Unknown')} – Chunk {meta.get('chunk_id','N/A')}"
                    with st.expander(label):
                        st.code(doc)
                        st.write(meta)
                        st.write(f"Relevance: {dist:.3f}")

if __name__ == "__main__":
    main_app()
