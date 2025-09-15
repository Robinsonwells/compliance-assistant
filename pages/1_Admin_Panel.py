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

def delete_file_chunks(qdrant_client, source_file: str) -> tuple[bool, str]:
    """Delete all chunks associated with a specific source file"""
    try:
        # First, get all points with this source file to count them
        scroll_result = client.scroll(
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
        client.delete(
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

def init_admin_systems():
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    
    # Initialize local embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Qdrant client
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
        
        # Create payload index for source_file field
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source_file",
            field_schema="keyword"
        )
    
    collection = client
    return user_manager, chunker, collection, embedding_model

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

def main():
    if not admin_login():
        st.stop()

    st.title("👨‍💼 Admin Control Panel")
    _, logout_col = st.columns([6, 1])
    with logout_col:
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

    um, chunker, coll, embedding_model = init_admin_systems()
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
                    # Create layout columns unconditionally
                    info_col, action_col = st.columns([4, 1])
                    dates_col, email_col = st.columns(2)
                    with info_col:
                        status = "🟢 Active" if active else "🔴 Inactive"
                        st.write(f"**{name}** ({status})")
                        st.write("**Access Code:**")
                        st.code(code)
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
            collection_info = coll.get_collection("legal_regulations")
            total = collection_info.points_count
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Chunks", total)
            m2.metric("Jurisdictions", "NY,NJ,CT")
            m3.metric("Status", "Active" if total > 0 else "Empty")
        except:
            st.warning("DB init error")

        st.markdown("---")
        st.subheader("📄 Upload Documents")
        uploads = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf','docx','txt'])
        if uploads and st.button("Process"):
            for f in uploads:
                st.write(f"Processing {f.name}")
                ok,msg = process_uploaded_file(f,chunker,coll,embedding_model)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("---")
        st.subheader("🗂️ Browse Files & Chunks")
        try:
            # Get ALL points using pagination to extract accurate file information
            all_points = []
            next_page_offset = None
            
            while True:
                scroll_result = coll.scroll(
                    collection_name="legal_regulations",
                    limit=1000,  # Reasonable batch size
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_page_offset = scroll_result
                all_points.extend(points)
                
                # Break if no more pages
                if next_page_offset is None:
                    break
            
            files = {}
            for point in all_points:
                sf = point.payload.get('source_file', 'Unknown')
                files[sf] = files.get(sf, 0) + 1

            if files:
                st.write(f"📊 **Total Files:** {len(files)} | **Total Chunks:** {sum(files.values())}")
                for fn, cnt in files.items():
                    with st.expander(f"📄 {fn} ({cnt} chunks)", expanded=False):
                        chunks_to_show = st.selectbox(
                            "Chunks to display:",
                            [10,25,50,100],
                            index=1,
                            key=f"chunks_{fn}"
                        )
                        if st.button("🔍 Browse Chunks", key=f"browse_{fn}"):
                            try:
                                scroll_result = coll.scroll(
                                    collection_name="legal_regulations",
                                    scroll_filter=Filter(
                                        must=[
                                            FieldCondition(
                                                key="source_file",
                                                match=MatchValue(value=fn)
                                            )
                                        ]
                                    ),
                                    limit=chunks_to_show,
                                    with_payload=True,
                                    with_vectors=False
                                )
                                
                                points = scroll_result[0]
                                if points:
                                    st.success(f"Displaying {len(points)} chunks")
                                    for i, point in enumerate(points, start=1):
                                        doc = point.payload.get('text', '')
                                        meta = {k: v for k, v in point.payload.items() if k != 'text'}
                                        with st.container():
                                            st.markdown(f"**Chunk {i}: {meta.get('chunk_id', 'N/A')}**")
                                            st.text_area("Content", doc, height=150, key=f"chunk_txt_{fn}_{i}")
                                            st.json(meta, expanded=False)
                                else:
                                    st.warning("No chunks to display")
                            except Exception as e:
                                st.error(f"Error browsing chunks for {fn}: {e}")

                        if st.button(f"🗑️ Delete {fn}", key=f"del_{fn}"):
                            st.session_state[f"confirm_del_{fn}"]=True
                        if st.session_state.get(f"confirm_del_{fn}"):
                            if st.button("✅ Confirm Delete", key=f"confirm_{fn}"):
                                ok,msg=delete_file_chunks(coll, fn)
                                if ok: st.success(msg)
                                else: st.error(msg)
                                st.session_state[f"confirm_del_{fn}"]=False
                                time.sleep(1); st.rerun()
                            if st.button("❌ Cancel", key=f"cancel_{fn}"):
                                st.session_state[f"confirm_del_{fn}"]=False
                                st.rerun()
            else:
                st.info("No files uploaded yet")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
