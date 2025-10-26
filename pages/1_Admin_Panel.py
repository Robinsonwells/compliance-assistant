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

def audit_knowledge_base(qdrant_client):
    """Audit knowledge base for data quality issues"""
    try:
        # Get all points to analyze
        all_points = []
        next_page_offset = None
        
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name="legal_regulations",
                limit=1000,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            all_points.extend(points)
            
            if next_page_offset is None:
                break
        
        # Analyze document status distribution
        status_counts = {}
        year_counts = {}
        files_by_status = {}
        problematic_chunks = []
        
        for point in all_points:
            payload = point.payload
            status = payload.get('document_status', 'unknown')
            year = payload.get('year', 'unknown')
            source_file = payload.get('source_file', 'unknown')
            
            # Count by status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by year
            year_counts[year] = year_counts.get(year, 0) + 1
            
            # Group files by status
            if status not in files_by_status:
                files_by_status[status] = set()
            files_by_status[status].add(source_file)
            
            # Flag problematic chunks
            if status in ['proposed_withdrawn', 'repealed', 'unknown']:
                problematic_chunks.append({
                    'id': point.id,
                    'source_file': source_file,
                    'status': status,
                    'section': payload.get('section_number', 'N/A'),
                    'citation': payload.get('citation', 'N/A'),
                    'text_preview': payload.get('text', '')[:200] + '...'
                })
        
        return {
            'total_chunks': len(all_points),
            'status_counts': status_counts,
            'year_counts': year_counts,
            'files_by_status': files_by_status,
            'problematic_chunks': problematic_chunks
        }
        
    except Exception as e:
        return {'error': str(e)}

def clean_problematic_chunks(qdrant_client, chunk_ids):
    """Delete problematic chunks by their IDs"""
    try:
        qdrant_client.delete(
            collection_name="legal_regulations",
            points_selector=chunk_ids
        )
        return True, f"Deleted {len(chunk_ids)} problematic chunks"
    except Exception as e:
        return False, str(e)

def update_chunk_status(qdrant_client, chunk_id, new_status):
    """Update the document status of a specific chunk"""
    try:
        # Get the current point
        point = qdrant_client.retrieve(
            collection_name="legal_regulations",
            ids=[chunk_id],
            with_payload=True,
            with_vectors=True
        )[0]
        
        # Update the payload
        updated_payload = point.payload.copy()
        updated_payload['document_status'] = new_status
        
        # Upsert the updated point
        qdrant_client.upsert(
            collection_name="legal_regulations",
            points=[PointStruct(
                id=chunk_id,
                vector=point.vector,
                payload=updated_payload
            )]
        )
        
        return True, f"Updated chunk {chunk_id} status to {new_status}"
    except Exception as e:
        return False, str(e)

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


def sort_chunks_by_document_order(chunks):
    """Sort chunks by their document order using section_number and subsection_index"""
    def chunk_sort_key(chunk):
        section_number = chunk.payload.get('section_number', '0')
        subsection_index = chunk.payload.get('subsection_index', '0')
        
        # Use natural sorting for proper alphanumeric ordering
        section_key = natural_sort_key(section_number)
        subsection_key = natural_sort_key(subsection_index)
        
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
    try:
        # Initialize user management
        user_manager = UserManager()
        
        # Initialize chunker
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OPENAI_API_KEY environment variable is not set")
            st.stop()
        chunker = LegalSemanticChunker(openai_api_key)
        
        # Initialize local embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            st.error("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
            st.stop()
        
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
        
        # Ensure payload index for document_status field exists
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="document_status",
                field_schema="keyword"
            )
        except Exception as e:
            # Index might already exist, which is fine
            if "already exists" not in str(e).lower():
                print(f"Warning: Could not create document_status payload index: {e}")
        
        # Ensure payload index for year field exists
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="year",
                field_schema="integer"
            )
        except Exception as e:
            # Index might already exist, which is fine
            if "already exists" not in str(e).lower():
                print(f"Warning: Could not create year payload index: {e}")
        
        return user_manager, chunker, client, embedding_model
        
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        st.error("Please check your environment variables and try refreshing the page.")
        st.info("If the problem persists, contact your system administrator.")
        st.stop()

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
        
        # Check for existing chunks with this exact content hash
        try:
            existing_result = qdrant_client.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="content_hash",
                            match=MatchValue(value=content_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            existing_points, _ = existing_result
            
            if existing_points:
                # Found chunks with identical content hash
                existing_filename = existing_points[0].payload.get('source_file', 'unknown')
                if existing_filename == uploaded_file.name:
                    # Same file, same content - true duplicate
                    if status_text:
                        status_text.warning(f"⏭️ Skipping {uploaded_file.name} - identical content already processed")
                    return True, f"Skipped {uploaded_file.name} - already processed with same content", content_hash
                else:
                    # Different filename but same content - still a duplicate
                    if status_text:
                        status_text.warning(f"⏭️ Skipping {uploaded_file.name} - identical content exists as {existing_filename}")
                    return True, f"Skipped {uploaded_file.name} - identical content exists as {existing_filename}", content_hash
            
            # Check if there are chunks with same filename but different content
            filename_result = qdrant_client.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=uploaded_file.name)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            filename_points, _ = filename_result
            
            if filename_points:
                # Same filename but different content - replace the old version
                if status_text:
                    status_text.info(f"🔄 Found existing chunks for {uploaded_file.name} with different content - replacing...")
                
                # Delete existing chunks for this filename
                qdrant_client.delete(
                    collection_name="legal_regulations",
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="source_file",
                                match=MatchValue(value=uploaded_file.name)
                            )
                        ]
                    )
                )
                
                if status_text:
                    status_text.success(f"✅ Deleted old chunks for {uploaded_file.name}")
                    
        except Exception as e:
            # If duplicate check fails, log warning but continue with processing
            if status_text:
                status_text.warning(f"⚠️ Could not check for duplicates: {e}")
            print(f"Duplicate check error for {uploaded_file.name}: {e}")
            # Continue with processing - don't skip the file
        
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
        
        try:
            chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        except Exception as chunking_error:
            if status_text:
                status_text.error(f"❌ Error during chunking: {chunking_error}")
            return False, f"Chunking failed: {chunking_error}", content_hash
            
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
        try:
            embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
        except Exception as embedding_error:
            if status_text:
                status_text.error(f"❌ Error generating embeddings: {embedding_error}")
            return False, f"Embedding generation failed: {embedding_error}", content_hash
        
        if progress_bar:
            progress_bar.progress(80)
        if status_text:
            status_text.info(f"📤 Uploading {len(chunks)} chunks to vector database...")
        
        # Prepare points for Qdrant
        points = []
        now = datetime.now().isoformat()
        
        try:
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
        except Exception as point_creation_error:
            if status_text:
                status_text.error(f"❌ Error creating vector points: {point_creation_error}")
            return False, f"Point creation failed: {point_creation_error}", content_hash

        # Upload points to Qdrant in batches
        try:
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
        except Exception as upload_error:
            if status_text:
                status_text.error(f"❌ Error uploading to vector database: {upload_error}")
            return False, f"Database upload failed: {upload_error}", content_hash
        
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
    
    # Initialize theme state
    if "admin_theme" not in st.session_state:
        st.session_state.admin_theme = "dark"
    
    # Add theme data attribute
    st.markdown(f'<script>document.documentElement.setAttribute("data-theme", "{st.session_state.admin_theme}");</script>', unsafe_allow_html=True)
    
    try:
        st.markdown("""
            <div class="dashboard-header">
                <h1 style="margin: 0; color: white;">👨‍💼 Admin Control Panel</h1>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                    System Administration & Knowledge Base Management
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Theme toggle for admin panel
        def toggle_admin_theme():
            st.session_state.admin_theme = "light" if st.session_state.admin_theme == "dark" else "dark"
        
        theme_col, _, logout_col = st.columns([1, 4, 1])
        with theme_col:
            if st.button("🎨 Theme"):
                toggle_admin_theme()
                st.rerun()
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
                        try:
                            if name.strip():
                                code = um.add_user(name.strip(), email.strip() or None, "basic", days)
                                st.success("Access code created")
                                st.code(code)
                                st.warning("Save this code now!")
                            else:
                                st.error("Name required")
                        except Exception as e:
                            st.error(f"❌ Error creating user: {str(e)}")
                            st.info("Please try again or contact your system administrator.")
            with right:
                st.markdown("#### 👥 Active Users")
                try:
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
                                    try:
                                        um.deactivate_user(code)
                                        st.success(f"Revoked access for {name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error revoking user: {str(e)}")
                            st.divider()
                    else:
                        st.info("No users created yet")
                except Exception as e:
                    st.error(f"❌ Error loading users: {str(e)}")
                    st.info("Please refresh the page or contact your system administrator.")

        with tab2:
            st.markdown("### 📚 Knowledge Base Management")
            try:
                collection_info = coll.get_collection("legal_regulations")
                total = collection_info.points_count
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Chunks", total)
                m2.metric("Jurisdictions", "NY,NJ,CT")
                m3.metric("Status", "Active" if total > 0 else "Empty")
            except Exception as e:
                st.warning(f"⚠️ Could not load database info: {str(e)}")
                st.info("Database connection may be temporarily unavailable.")

            # Data Quality Audit Section
            st.markdown("---")
            st.markdown("#### 🔍 Data Quality Audit")
            
            if st.button("🔍 Run Knowledge Base Audit"):
                with st.spinner("Auditing knowledge base..."):
                    audit_results = audit_knowledge_base(coll)
                    
                    if 'error' in audit_results:
                        st.error(f"❌ Audit failed: {audit_results['error']}")
                    else:
                        st.success(f"✅ Audit complete! Analyzed {audit_results['total_chunks']} chunks")
                        
                        # Display status distribution
                        st.markdown("##### 📊 Document Status Distribution")
                        status_cols = st.columns(len(audit_results['status_counts']))
                        for i, (status, count) in enumerate(audit_results['status_counts'].items()):
                            with status_cols[i]:
                                color = "normal" if status == "current" else "inverse"
                                st.metric(status.replace('_', ' ').title(), count, delta_color=color)
                        
                        # Display year distribution
                        st.markdown("##### 📅 Year Distribution")
                        year_data = audit_results['year_counts']
                        if year_data:
                            st.bar_chart(year_data)
                        
                        # Show files by status
                        st.markdown("##### 📁 Files by Status")
                        for status, files in audit_results['files_by_status'].items():
                            if status != 'current':
                                with st.expander(f"⚠️ {status.replace('_', ' ').title()} Files ({len(files)})", expanded=False):
                                    for file in sorted(files):
                                        st.write(f"• {file}")
                        
                        # Show problematic chunks
                        if audit_results['problematic_chunks']:
                            st.markdown("##### 🚨 Problematic Chunks")
                            st.warning(f"Found {len(audit_results['problematic_chunks'])} chunks that may need attention")
                            
                            # Group by file for easier management
                            chunks_by_file = {}
                            for chunk in audit_results['problematic_chunks']:
                                file = chunk['source_file']
                                if file not in chunks_by_file:
                                    chunks_by_file[file] = []
                                chunks_by_file[file].append(chunk)
                            
                            for file, chunks in chunks_by_file.items():
                                with st.expander(f"🗂️ {file} ({len(chunks)} problematic chunks)", expanded=False):
                                    for chunk in chunks[:5]:  # Show first 5 chunks
                                        st.write(f"**Status:** {chunk['status']}")
                                        st.write(f"**Section:** {chunk['section']}")
                                        st.write(f"**Citation:** {chunk['citation']}")
                                        st.write(f"**Preview:** {chunk['text_preview']}")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            if st.button("✅ Mark Current", key=f"current_{chunk['id']}"):
                                                success, msg = update_chunk_status(coll, chunk['id'], 'current')
                                                if success:
                                                    st.success(msg)
                                                else:
                                                    st.error(msg)
                                        with col2:
                                            if st.button("🗑️ Delete", key=f"delete_{chunk['id']}"):
                                                success, msg = clean_problematic_chunks(coll, [chunk['id']])
                                                if success:
                                                    st.success(msg)
                                                else:
                                                    st.error(msg)
                                        with col3:
                                            if st.button("📝 Review", key=f"review_{chunk['id']}"):
                                                st.info("Marked for manual review")
                                        
                                        st.divider()
                                    
                                    if len(chunks) > 5:
                                        st.info(f"... and {len(chunks) - 5} more chunks in this file")
                                    
                                    # Bulk actions for file
                                    st.markdown("**Bulk Actions for this file:**")
                                    bulk_col1, bulk_col2 = st.columns(2)
                                    with bulk_col1:
                                        if st.button(f"🗑️ Delete All {len(chunks)} Chunks", key=f"bulk_delete_{file}"):
                                            chunk_ids = [c['id'] for c in chunks]
                                            success, msg = clean_problematic_chunks(coll, chunk_ids)
                                            if success:
                                                st.success(msg)
                                                st.rerun()
                                            else:
                                                st.error(msg)
                                    with bulk_col2:
                                        if st.button(f"✅ Mark All {len(chunks)} as Current", key=f"bulk_current_{file}"):
                                            success_count = 0
                                            for chunk in chunks:
                                                success, _ = update_chunk_status(coll, chunk['id'], 'current')
                                                if success:
                                                    success_count += 1
                                            st.success(f"Updated {success_count}/{len(chunks)} chunks")
                                            if success_count == len(chunks):
                                                st.rerun()
                        else:
                            st.success("✅ No problematic chunks found! All documents appear to be current.")
                        
                        # Store audit results in session state for reference
                        st.session_state['last_audit'] = audit_results
            st.markdown("---")
            st.markdown("#### 📄 Document Upload")
            uploads = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf','docx','txt'])
            
            if uploads and st.button("🚀 Process All Documents"):
                try:
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
                                st.error(f"🚨 Critical error during file processing: {str(e)}")
                                st.info("Your session remains active. You can continue with other files.")
                            
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
                    
                    # Recommend audit after upload
                    if processed_count > 0:
                        st.info("💡 **Recommendation:** Run a Data Quality Audit above to verify the uploaded documents have correct status metadata.")
                    
                    # Refresh the page to clear the file uploader and update the file list
                    time.sleep(2)  # Give user time to see the final status
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"🚨 Critical error during document processing: {str(e)}")
                    st.error("Your session remains active. Please try again or contact your system administrator.")
                    st.info("💡 Tip: Try uploading fewer files at once if you continue to experience issues.")

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
                    
                    # Show audit summary if available
                    if 'last_audit' in st.session_state:
                        audit = st.session_state['last_audit']
                        status_counts = audit.get('status_counts', {})
                        if status_counts:
                            current_count = status_counts.get('current', 0)
                            total_count = sum(status_counts.values())
                            problematic_count = total_count - current_count
                            
                            if problematic_count > 0:
                                st.warning(f"⚠️ **Data Quality Alert:** {problematic_count} chunks may need attention (not marked as 'current')")
                            else:
                                st.success("✅ **Data Quality:** All chunks are marked as current")
                    
                    for fn, cnt in files.items():
                        with st.expander(f"📄 {fn} ({cnt} chunks)", expanded=False):
                            # Show file-specific status info if available
                            if 'last_audit' in st.session_state:
                                audit = st.session_state['last_audit']
                                problematic_chunks = [c for c in audit.get('problematic_chunks', []) if c['source_file'] == fn]
                                if problematic_chunks:
                                    st.warning(f"⚠️ This file has {len(problematic_chunks)} chunks that may need attention")
                                else:
                                    st.success("✅ All chunks from this file appear to be current")
                            
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
                                            
                                            # Highlight document status
                                            doc_status = meta.get('document_status', 'unknown')
                                            status_color = "🟢" if doc_status == "current" else "🟡" if doc_status == "unknown" else "🔴"
                                            
                                            with st.container():
                                                # Phase 3: Enhanced chunk display with validation info
                                                version_num = meta.get('version_number', 1)
                                                validation_warnings = meta.get('validation_warnings', 0)
                                                citation_issues = meta.get('citation_issues', 0)
                                                
                                                version_indicator = f" (v{version_num})" if version_num > 1 else ""
                                                warning_indicator = f" ⚠️{validation_warnings}" if validation_warnings > 0 else ""
                                                citation_indicator = f" 🔍{citation_issues}" if citation_issues > 0 else ""
                                                
                                                st.markdown(f"**Chunk {i}:** Section {meta.get('section_number', 'N/A')}.{meta.get('subsection_index', '0')} - {meta.get('section_title', 'N/A')} {status_color}{version_indicator}{warning_indicator}{citation_indicator}")
                                                st.caption(f"Status: {doc_status} | Year: {meta.get('year', 'N/A')} | Citation: {meta.get('citation', 'N/A')}")
                                                st.caption(f"Chunk ID: {meta.get('chunk_id', 'N/A')} | Semantic Type: {meta.get('semantic_type', 'N/A')} | Fingerprint: {meta.get('document_fingerprint', 'N/A')[:12]}...")
                                                st.text_area("Content", doc, height=150, key=f"chunk_txt_{fn}_{i}")
                                                st.json(meta, expanded=False)
                                    else:
                                        st.warning("No chunks to display")
                                except Exception as e:
                                    st.error(f"❌ Error browsing chunks for {fn}: {e}")
                                    st.info("Please try again or contact your system administrator.")

                            if st.button(f"🗑️ Delete {fn}", key=f"del_{fn}"):
                                st.session_state[f"confirm_del_{fn}"] = True
                            if st.session_state.get(f"confirm_del_{fn}"):
                                if st.button("✅ Confirm Delete", key=f"confirm_{fn}"):
                                    try:
                                        ok, msg = delete_file_chunks(coll, fn)
                                        if ok:
                                            st.success(msg)
                                        else:
                                            st.error(msg)
                                        st.session_state[f"confirm_del_{fn}"] = False
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error deleting file: {str(e)}")
                                        st.info("Please try again or contact your system administrator.")
                                if st.button("❌ Cancel", key=f"cancel_{fn}"):
                                    st.session_state[f"confirm_del_{fn}"] = False
                                    st.rerun()
                else:
                    st.info("No files uploaded yet")
            except Exception as e:
                st.error(f"❌ Error loading file information: {str(e)}")
                st.info("Database connection may be temporarily unavailable. Please refresh the page.")
                
    except Exception as e:
        st.error(f"🚨 Critical application error: {str(e)}")
        st.error("Your session remains active, but some features may be unavailable.")
        st.info("Please refresh the page or contact your system administrator.")
        st.info("💡 If this error persists, try logging out and logging back in.")
        try:
            # Provide emergency logout option
            if st.button("🚪 Emergency Logout", key="emergency_logout"):
                st.session_state.admin_authenticated = False
                st.rerun()
        except:
            pass  # Don't let logout button errors crash the error handler

if __name__ == "__main__":
    main()