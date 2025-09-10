import streamlit as st
from user_management import UserManager
from datetime import datetime
from dotenv import load_dotenv
import os
import chromadb
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
import time

def delete_file_chunks(collection, source_file: str) -> tuple[bool, str]:
    """Delete all chunks associated with a specific source file"""
    try:
        result = collection.get(where={"source_file": source_file}, include=['metadatas'])
        metas = result.get('metadatas', [])
        count = len(metas)
        if count == 0:
            return False, "No chunks found for this file"
        collection.delete(where={"source_file": source_file})
        return True, f"Deleted {count} chunks"
    except Exception as e:
        return False, str(e)

# Load environment variables
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin-secure-2024")

st.set_page_config(page_title="Admin Panel", page_icon="👨‍💼", layout="wide")

def init_admin_systems():
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    client = chromadb.PersistentClient(path="./legal_compliance_db")
    collection = client.get_or_create_collection(
        name="legal_regulations",
        metadata={"description": "Multi-state employment law regulations"}
    )
    return user_manager, chunker, collection

def admin_login():
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    if not st.session_state.admin_authenticated:
        st.title("🔑 Admin Login")
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password")
        return False
    return True

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

        st.write(f"📊 File size: {len(text)} characters")
        st.write("🔍 First 500 characters:")
        st.text(text[:500])
        st.write(f"🏷️ XML detection: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}")

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

        batch = 5000
        for i in range(0, len(docs), batch):
            collection.add(
                documents=docs[i:i+batch],
                metadatas=metas[i:i+batch],
                ids=ids[i:i+batch]
            )
        return True, f"Processed {len(chunks)} chunks from {uploaded_file.name}"
    except Exception as e:
        return False, f"Error processing file: {e}"

def main():
    if not admin_login():
        st.stop()

    st.title("👨‍💼 Admin Control Panel")
    _, logout_col = st.columns([6, 1])
    with logout_col:
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

    um, chunker, coll = init_admin_systems()
    tab1, tab2 = st.tabs(["👥 Users", "📚 Knowledge Base"])

    with tab1:
        st.header("User Management")
        left, right = st.columns(2)
        with left:
            st.subheader("➕ New Access Code")
            with st.form("add_u"):
                name = st.text_input("Client/Company *")
                email = st.text_input("Email (optional)")
                days = st.number_input("Days valid", value=365, min_value=1)
                if st.form_submit_button("Generate"):
                    if name.strip():
                        code = um.add_user(name.strip(), email.strip() or None, "basic", days)
                        st.success("Access code created")
                        st.code(code)
                        st.warning("Save this code now!")
                    else:
                        st.error("Name required")
        with right:
            st.subheader("👥 Manage Users")
            users = um.get_all_users()
            if users:
                for u in users:
                    code, name, email, created, last, active, exp, tier = u
                    with st.container():
                        info_col, action_col = st.columns([4, 1])
                        with info_col:
                            status = "🟢 Active" if active else "🔴 Inactive"
                            st.write(f"**{name}** ({status})")
                            st.write("**Access Code:**")
                            st.code(code)
                            # Fixed: move columns outside any conditional button logic
                            dates_col, email_col = st.columns(2)
                            with dates_col:
                                st.caption(f"Created: {created[:10]}")
                                st.caption(f"Last login: {last[:16] if last else 'Never'}")
                            with email_col:
                                if email:
                                    st.caption(f"Email: {email}")
                                st.caption(f"Expires: {exp[:10] if exp else 'No expiry'}")
                        with action_col:
                            if active and st.button("🚫 Revoke", key=f"r_{code}"):
                                um.deactivate_user(code)
                                st.success(f"Revoked access for {name}")
                                st.rerun()
                        st.divider()
            else:
                st.info("No users created yet")

    with tab2:
        st.header("Knowledge Base")
        try:
            total = coll.count()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Chunks", total)
            m2.metric("Jurisdictions", "NY,NJ,CT")
            m3.metric("Status", "Active" if total > 0 else "Empty")
        except:
            st.warning("DB init error")

        st.markdown("---")
        st.subheader("📄 Upload Documents")
        uploads = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
        if uploads and st.button("Process"):
            for f in uploads:
                st.write(f"Processing {f.name}")
                ok, msg = process_uploaded_file(f, chunker, coll)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("---")
        st.subheader("🗂️ Browse Files & Chunks")
        try:
            all_metas = coll.get(include=['metadatas']).get('metadatas', [])
            files = {}
            for md in all_metas:
                sf = md.get('source_file', 'Unknown')
                files[sf] = files.get(sf, 0) + 1

            if files:
                st.write(f"📊 **Total Files:** {len(files)} | **Total Chunks:** {sum(files.values())}")
                for fn, cnt in files.items():
                    with st.expander(f"📄 {fn} ({cnt} chunks)", expanded=False):
                        chunks_to_show = st.selectbox(
                            "Chunks to display:", [10, 25, 50, 100],
                            index=1, key=f"chunks_{fn}"
                        )
                        if st.button("🔍 Browse Chunks", key=f"browse_{fn}"):
                            try:
                                fr = coll.get(
                                    where={"source_file": fn},
                                    limit=chunks_to_show,
                                    include=['documents', 'metadatas']
                                )
                                docs = fr.get('documents', [])
                                metas = fr.get('metadatas', [])
                                if docs:
                                    st.success(f"Displaying {len(docs)} chunks from {fn}")
                                    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                                        # Use container instead of nested expander
                                        with st.container():
                                            st.markdown(f"**Chunk {i}: {meta.get('chunk_id','N/A')}**")
                                            st.text_area(
                                                "Content", doc,
                                                height=150, key=f"chunk_txt_{fn}_{i}"
                                            )
                                            st.json(meta, expanded=False)
                                else:
                                    st.warning("No chunks to display")
                            except Exception as e:
                                st.error(f"Error browsing chunks for {fn}: {e}")

                        if st.button(f"🗑️ Delete {fn}", key=f"del_{fn}"):
                            st.session_state[f"confirm_del_{fn}"] = True
                        if st.session_state.get(f"confirm_del_{fn}"):
                            if st.button("✅ Confirm Delete", key=f"confirm_{fn}"):
                                ok, msg = delete_file_chunks(coll, fn)
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(msg)
                                st.session_state[f"confirm_del_{fn}"] = False
                                time.sleep(1)
                                st.rerun()
                            if st.button("❌ Cancel", key=f"cancel_{fn}"):
                                st.session_state[f"confirm_del_{fn}"] = False
                                st.rerun()
            else:
                st.info("No files uploaded yet")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
