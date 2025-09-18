import torch  # Initialize PyTorch early to prevent torch.classes errors
import streamlit as st
from user_management import UserManager
from datetime import datetime
from dotenv import load_dotenv
import os
import re
import uuid
import hashlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
import time

# Load custom CSS
def load_css():
    try:
        with open("styles/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

def delete_file_chunks(qdrant_client, source_file: str) -> tuple[bool, str]:
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

def natural_sort_key(s):
    """Convert a string into a list of strings and integers for natural sorting"""
    if not s:
        return [0]
    # Split the string into parts, converting numeric parts to integers
    parts = re.split(r'(\d+)', str(s))
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return result

def parse_hierarchical_number(num_str: str) -> tuple:
    """Parse hierarchical numbers like '8 AAC 05.030' or '1.0.1' into comparable tuples"""
    if not num_str or not isinstance(num_str, str):
        return (0,)
    
    # Extract the numeric hierarchical part from complex section numbers
    # Handle formats like "8 AAC 05.030", "12 NYCRR 142-2.1", etc.
    hierarchical_match = re.search(r'(\d+(?:\.\d+)*)', num_str)
    if hierarchical_match:
        hierarchical_part = hierarchical_match.group(1)
    else:
        # If no hierarchical pattern found, try to extract any numbers
        numbers = re.findall(r'\d+', num_str)
        if numbers:
            hierarchical_part = '.'.join(numbers)
        else:
            return (0,)
    
    # Split by dots and convert each part to integer
    parts = []
    for part in hierarchical_part.split('.'):
        try:
            parts.append(int(part.strip()))
        except ValueError:
            # Non-numeric parts get converted to 0
            parts.append(0)
    
    return tuple(parts) if parts else (0,)

def sort_chunks_by_document_order(chunks):
    """Sort chunks by their document order using section_number and subsection_index"""
    def chunk_sort_key(chunk):
        section_number = chunk.payload.get('section_number', '0')
        subsection_index = chunk.payload.get('subsection_index', '0')
        
        # Parse hierarchical numbers for proper sorting
        section_key = parse_hierarchical_number(section_number)
        subsection_key = parse_hierarchical_number(subsection_index)
        
        return (section_key, subsection_key)
    
    return sorted(chunks, key=chunk_sort_key)

# Load environment variables
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin-secure-2024")

st.set_page_config(page_title="Admin Panel", page_icon="👨‍💼", layout="wide")

# Load CSS immediately after page config
load_css()

@st.cache_resource
def init_admin_systems():
    """Initialize admin systems with caching to improve performance"""
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
    
    # Ensure payload index for content_hash field exists
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="content_hash",
            field_schema="keyword"
        )
    except Exception as e:
        # Index might already exist, which is fine
        if "already exists" not in str(e).lower():
            print(f"Warning: Could not create content_hash payload index: {e}")
    
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
                    st.rerun()
                else:
                    st.error("Invalid password")
            elif submit_button:
                st.error("Please enter the admin password")
        
        return False
    return True

def process_uploaded_file(uploaded_file, chunker, qdrant_client, embedding_model, progress_bar=None, status_text=None):
    """Process uploaded file with progress tracking and idempotency"""
    try:
        # Read file content once and calculate hash for idempotency
        file_content = uploaded_file.read()
        content_hash = hashlib.md5(file_content).hexdigest()
        
        # Reset file pointer for text extraction
        uploaded_file.seek(0)
        
        if status_text:
            status_text.info(f"🔍 Checking if {uploaded_file.name} has already been processed...")
        
        # Check if this exact file content has already been processed
        try:
            existing_check = qdrant_client.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=uploaded_file.name)
                        ),
                        FieldCondition(
                            key="content_hash",
                            match=MatchValue(value=content_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            if existing_check[0]:  # If any points found
                if status_text:
                    status_text.warning(f"⏭️ Skipping {uploaded_file.name} - identical content already processed")
                return True, f"Skipped {uploaded_file.name} - already processed with same content", content_hash
        except Exception as e:
            # If check fails, continue with processing
            if status_text:
                status_text.warning(f"Could not check for duplicates: {e}")
        
        if progress_bar:
            progress_bar.progress(10)
        if status_text:
            status_text.info(f"📄 Extracting text from {uploaded_file.name}...")
        
        t = uploaded_file.type
        if t == "application/pdf":
            text = extract_pdf_text(uploaded_file)
        elif t == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_docx_text(uploaded_file)
        elif t == "text/plain":
            text = file_content.decode("utf-8")
        else:
            return False, f"Unsupported file type: {t}", content_hash

        if text.startswith("Error"):
            return False, text, content_hash

        if progress_bar:
            progress_bar.progress(25)
        if status_text:
            status_text.info(f"📊 Extracted {len(text)} characters from {uploaded_file.name}")
            status_text.info(f"🏷️ Format: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}")

        if progress_bar:
            progress_bar.progress(40)
        if status_text:
            status_text.info(f"🔍 Creating semantic chunks for {uploaded_file.name}...")
        
        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        if not chunks:
            return False, "No chunks were created - check file format", content_hash
        
        if status_text:
            status_text.info(f"📦 Created {len(chunks)} chunks from {uploaded_file.name}")

        # Collect all chunk texts for batch embedding generation
        chunk_texts = [ch['text'] for ch in chunks]
        
        if progress_bar:
            progress_bar.progress(60)
        if status_text:
            status_text.info(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings locally in batch
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
        
        if progress_bar:
            progress_bar.progress(80)
        if status_text:
            status_text.info(f"📤 Uploading {len(chunks)} chunks to vector database...")
        
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
                'content_hash': content_hash,
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
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            qdrant_client.upsert(
                collection_name="legal_regulations",
                points=batch
            )
            
            # Update progress during upload
            if progress_bar:
                upload_progress = 80 + (20 * (i + len(batch)) / len(points))
                progress_bar.progress(int(upload_progress))
        
        if progress_bar:
            progress_bar.progress(100)
        if status_text:
            status_text.success(f"✅ Successfully processed {len(chunks)} chunks from {uploaded_file.name}")
        
        return True, f"Processed {len(chunks)} chunks from {uploaded_file.name}", content_hash

    except Exception as e:
        if status_text:
            status_text.error(f"❌ Error processing {uploaded_file.name}: {e}")
        return False, f"Error processing file: {e}", None

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
    
    _, logout_col = st.columns([5, 1])
    with logout_col:
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

    um, chunker, coll, embedding_model = init_admin_systems()
    tab1, tab2 = st.tabs(["👥 Users", "📚 Knowledge Base"])

    with tab1:
        st.markdown("### 👥 User Management")
        left, right = st.columns(2)
        with left:
            st.markdown("#### ➕ Create New Access Code")
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
            st.markdown("#### 👥 Active Users")
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
        st.markdown("### 📚 Knowledge Base Management")
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
        st.markdown("#### 📄 Document Upload")
        uploads = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf','docx','txt'])
        
        if uploads and st.button("🚀 Process All Documents"):
            with st.status("🔄 Processing documents...", expanded=True) as status:
                total_files = len(uploads)
                processed_count = 0
                skipped_count = 0
                error_count = 0
                
                st.write(f"📋 **Processing Queue:** {total_files} files")
                st.write("---")
                
                for i, uploaded_file in enumerate(uploads, 1):
                    st.write(f"**File {i}/{total_files}: {uploaded_file.name}**")
                    
                    # Create progress bar and status text for this file
                    file_progress = st.progress(0)
                    file_status = st.empty()
                    
                    try:
                        # Process the file with progress tracking
                        success, message, content_hash = process_uploaded_file(
                            uploaded_file, chunker, coll, embedding_model, 
                            progress_bar=file_progress, status_text=file_status
                        )
                        
                        if success:
                            if "Skipped" in message:
                                skipped_count += 1
                                file_status.warning(f"⏭️ {message}")
                            else:
                                processed_count += 1
                                file_status.success(f"✅ {message}")
                        else:
                            error_count += 1
                            file_status.error(f"❌ {message}")
                    
                    except Exception as e:
                        error_count += 1
                        file_status.error(f"❌ Unexpected error processing {uploaded_file.name}: {e}")
                    
                    # Clear the progress bar after processing
                    file_progress.empty()
                    
                    # Add separator between files (except for the last one)
                    if i < total_files:
                        st.write("---")
                
                # Update final status
                if error_count == 0:
                    if skipped_count == total_files:
                        status.update(label="⏭️ All documents were already processed", state="complete")
                    elif skipped_count > 0:
                        status.update(label=f"✅ Processing complete! {processed_count} processed, {skipped_count} skipped", state="complete")
                    else:
                        status.update(label=f"✅ All {processed_count} documents processed successfully!", state="complete")
                else:
                    status.update(label=f"⚠️ Processing complete with issues: {processed_count} processed, {skipped_count} skipped, {error_count} errors", state="error")
                
                st.write("🎉 **Batch processing finished!**")
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", total_files)
                with col2:
                    st.metric("Processed", processed_count, delta=processed_count if processed_count > 0 else None)
                with col3:
                    st.metric("Skipped", skipped_count, delta=skipped_count if skipped_count > 0 else None)
                with col4:
                    st.metric("Errors", error_count, delta=error_count if error_count > 0 else None, delta_color="inverse")
            
            # Refresh the page to clear the file uploader and update the file list
            time.sleep(2)  # Give user time to see the final status
            st.rerun()

        st.markdown("---")
        st.markdown("#### 🗂️ Browse Files & Chunks")
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
                            [10, 25, 50, 100],
                            index=1,
                            key=f"chunks_{fn}"
                        )
                        if st.button("🔍 Browse Chunks", key=f"browse_{fn}"):
                            try:
                                # Fetch ALL chunks for this file first
                                all_chunks = []
                                next_page_offset = None
                                
                                while True:
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
                                        limit=1000,  # Large batch size for efficiency
                                        offset=next_page_offset,
                                        with_payload=True,
                                        with_vectors=False
                                    )
                                    
                                    points, next_page_offset = scroll_result
                                    all_chunks.extend(points)
                                    
                                    # Break if no more pages
                                    if next_page_offset is None:
                                        break
                                
                                # Sort chunks by document order
                                sorted_chunks = sort_chunks_by_document_order(all_chunks)
                                
                                # Apply the display limit to sorted chunks
                                points = sorted_chunks[:chunks_to_show]
                                
                                if points:
                                    st.success(f"Displaying {len(points)} chunks (sorted chronologically, showing chunks 1-{len(points)} of {len(all_chunks)} total)")
                                    for i, point in enumerate(points, start=1):
                                        doc = point.payload.get('text', '')
                                        meta = {k: v for k, v in point.payload.items() if k != 'text'}
                                        with st.container():
                                            st.markdown(f"**Chunk {i}:** Section {meta.get('section_number', 'N/A')}.{meta.get('subsection_index', '0')} - {meta.get('section_title', 'N/A')}")
                                            st.caption(f"Chunk ID: {meta.get('chunk_id', 'N/A')} | Semantic Type: {meta.get('semantic_type', 'N/A')}")
                                            st.text_area("Content", doc, height=150, key=f"chunk_txt_{fn}_{i}")
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