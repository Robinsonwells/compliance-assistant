import pysqlite3 as sqlite3  # ensure sqlite compatibility
import sys
sys.modules['sqlite3'] = sqlite3

import os
import uuid
import time
from datetime import datetime, timedelta

import streamlit as st
import openai
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from user_management import UserManager
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page config & CSS tweaks
st.set_page_config(page_title="Legal Compliance Assistant", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
  .stAppViewerButton, .stAppViewerIcon, .stAppViewerLink, .css-1avcm0n.e1fqkh3o2,
  .stAppDeployButton, .stDecoration, #MainMenu, footer, header { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_systems():
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    vector_client = chromadb.Client(Settings(persist_directory="./legal_compliance_db"))
    collection = vector_client.get_or_create_collection(
        name="legal_regulations",
        metadata={"description": "Multi-state employment law regulations"}
    )
    return user_manager, chunker, collection

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def check_authentication():
    user_manager, chunker, collection = init_systems()
    if st.session_state.get('authenticated'):
        sid = get_session_id()
        if user_manager.is_session_valid(sid, hours_timeout=24):
            user_manager.update_session_activity(sid)
            return True, user_manager, chunker, collection
        st.session_state.authenticated = False
        st.error("🕐 Session expired. Please log in again.")
        time.sleep(2)
        st.rerun()
    st.markdown("# 🔐 Legal Compliance Assistant")
    st.markdown("**Professional AI-Powered Legal Analysis**")
    st.markdown("---")
    with st.form("login"):
        code = st.text_input("Access Code", type="password", placeholder="Enter access code")
        if st.form_submit_button("🚀 Access Assistant") and code:
            if user_manager.validate_access_code(code):
                sid = get_session_id()
                user_manager.create_session(code, sid)
                user_manager.update_last_login(code)
                st.session_state.authenticated = True
                st.session_state.login_time = datetime.now()
                st.success("✅ Access granted!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid or expired code")
                time.sleep(2)
    with st.expander("ℹ️ Session Info"):
        st.write("• Sessions expire after 24h inactivity")
        st.write("• Session renews each interaction")
    return False, None, None, None

def search_knowledge_base(collection, query, n_results=5):
    try:
        res = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents','metadatas','distances']
        )
        docs = res['documents'][0]
        metas = res['metadatas'][0]
        dists = res['distances'][0]
        return list(zip(docs, metas, dists))
    except:
        return []

def process_uploaded_file(uploaded, chunker, collection):
    try:
        t = uploaded.type
        if t=="application/pdf":
            text = extract_pdf_text(uploaded)
        elif t=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_docx_text(uploaded)
        elif t=="text/plain":
            text = uploaded.read().decode("utf-8")
        else:
            return False, f"Unsupported type: {t}"
        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        now = datetime.now().isoformat()
        docs, metas, ids = [], [], []
        for ch in chunks:
            docs.append(ch['text'])
            metas.append({**ch['metadata'], 'file':uploaded.name,'date':now})
            ids.append(f"{uploaded.name}_{ch['metadata']['chunk_id']}")
        for i in range(0, len(docs), 5000):
            collection.add(documents=docs[i:i+5000], metadatas=metas[i:i+5000], ids=ids[i:i+5000])
        return True, f"Processed {len(chunks)} chunks"
    except Exception as e:
        return False, f"Error: {e}"

def main_app():
    auth, user_manager, chunker, collection = check_authentication()
    if not auth:
        st.stop()
    c1,c2=st.columns([6,1])
    with c1:
        st.markdown("# ⚖️ Elite Legal Compliance Assistant")
        st.markdown("*GPT-5-powered structured analysis*")
    with c2:
        if st.button("🚪 Logout"):
            st.session_state.authenticated=False
            st.rerun()
    st.markdown("---")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Hello! Ask about NY/NJ/CT employment law."}]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if prompt := st.chat_input("Ask legal compliance..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            prog=st.empty(); bar=st.progress(0)
            prog.text("🔍 Searching knowledge base"); bar.progress(20)
            results=search_knowledge_base(collection,prompt,8)
            prog.text("🧠 Building context"); bar.progress(50)
            context = "\n\n".join(f"{doc}" for doc,_,_ in results) or "No context found."
            system_prompt = f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\nContext:\n{context}\nUser: {prompt}"
            prog.text("⚖️ Generating response"); bar.progress(75)
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"system","content":system_prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            bar.progress(100); prog.text("✅ Done")
            reply = resp.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role":"assistant","content":reply})
            if results:
                st.markdown("### 📚 Sources")
                for doc,meta,dist in results:
                    with st.expander(f"{meta.get('file')} – chunk {meta.get('chunk_id')}"):
                        st.code(doc)
                        st.write(f"Relevance: {dist:.3f}")

if __name__=="__main__":
    main_app()
