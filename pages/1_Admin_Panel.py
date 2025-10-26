import streamlit as st
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import hashlib
import io
from typing import List, Dict, Any
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import custom modules
from user_management import UserManager
from advanced_chunking import (
    LegalSemanticChunker, 
    extract_pdf_text, 
    extract_docx_text,
    strip_xml_tags,
    detect_document_format
)
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Initialize components
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
    from sentence_transformers import SentenceTransformer
    import openai
    
    # Initialize clients
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Initialize legal chunker
    legal_chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY", ""))
    
except Exception as e:
    st.error(f"Failed to initialize components: {e}")
    st.stop()

def init_admin_systems():
    """Initialize admin systems with payload index verification"""
    collection_name = "legal_regulations"
    
    try:
        # Ensure collection exists
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            st.info(f"✅ Collection exists with {collection_info.points_count} points")
        except Exception:
            # Create collection if it doesn't exist
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            st.info("✅ Created new collection")
        
        # Create payload indexes
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD
            )
            st.info("✅ Created/verified source_file payload index")
        except Exception as e:
            if "already exists" not in str(e).lower():
                st.warning(f"⚠️ Could not create source_file index: {e}")
        
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="content_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
            st.info("✅ Created/verified content_hash payload index")
        except Exception as e:
            if "already exists" not in str(e).lower():
                st.warning(f"⚠️ Could not create content_hash index: {e}")
        
        # Verify payload indexes are working
        try:
            # Test the content_hash index
            test_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="content_hash", match=MatchValue(value="test_hash_verify"))
                ]),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            logger.info("✅ content_hash index is working")
        except Exception as e:
            logger.error(f"❌ content_hash index NOT working: {e}")
            
        try:
            # Test the source_file index
            test_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_file", match=MatchValue(value="test_file_verify"))
                ]),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            logger.info("✅ source_file index is working")
        except Exception as e:
            logger.error(f"❌ source_file index NOT working: {e}")
        
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize admin systems: {e}")
        return False

def process_uploaded_file(uploaded_file, chunker, qdrant_client, embedding_model, progress_bar=None, status_text=None):
    """Process uploaded file with enhanced logging and proper error handling"""
    try:
        # Read file content and calculate hash
        file_content = uploaded_file.read()
        content_hash = hashlib.md5(file_content).hexdigest()
        
        # Enhanced logging at start
        logger.info(f"📋 Processing file: {uploaded_file.name}")
        logger.info(f"📊 Content hash: {content_hash}")
        logger.info(f"📏 File size: {len(file_content)} bytes")
        
        # Check for duplicates
        try:
            # Check for existing content hash
            existing_result = qdrant_client.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(must=[
                    FieldCondition(key="content_hash", match=MatchValue(value=content_hash))
                ]),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            existing_points, _ = existing_result
            logger.info(f"🔍 Found {len(existing_points)} existing chunks with content hash: {content_hash}")
            
            if existing_points:
                existing_filename = existing_points[0].payload.get("source_file", "unknown")
                logger.info(f"📁 Existing file: {existing_filename}, Current file: {uploaded_file.name}")
                
                if existing_filename == uploaded_file.name:
                    logger.info("⏭️ SKIPPING - Same file, same content (true duplicate)")
                    if status_text:
                        status_text.warning(f"⏭️ Skipping {uploaded_file.name} - identical content already processed")
                    return True, f"Skipped {uploaded_file.name} - already processed with same content", content_hash
                else:
                    logger.info(f"⏭️ SKIPPING - Different filename but identical content (duplicate of {existing_filename})")
                    if status_text:
                        status_text.warning(f"⏭️ Skipping {uploaded_file.name} - identical content exists as {existing_filename}")
                    return True, f"Skipped {uploaded_file.name} - identical content exists as {existing_filename}", content_hash
            
            # Check for same filename (for replacement)
            filename_result = qdrant_client.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_file", match=MatchValue(value=uploaded_file.name))
                ]),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            filename_points, _ = filename_result
            
            if filename_points:
                logger.info(f"🔄 Found {len(filename_points)} existing chunks with filename: {uploaded_file.name} - will replace")
                # Delete existing chunks for this filename
                delete_document_from_qdrant(uploaded_file.name)
            else:
                logger.info(f"✅ No existing chunks found - proceeding with fresh upload")
                
        except Exception as e:
            logger.error(f"Duplicate check failed for {uploaded_file.name}: {e}", exc_info=True)
            if status_text:
                status_text.error(f"❌ Duplicate check failed: {e}")
            # CRITICAL: Return False to stop processing if duplicate check fails
            return False, f"Duplicate check failed: {e}", None
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text_content = extract_pdf_text(io.BytesIO(file_content))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_content = extract_docx_text(io.BytesIO(file_content))
        else:
            # Text or XML file
            text_content = file_content.decode('utf-8')
        
        if not text_content or len(text_content.strip()) < 50:
            return False, f"No valid text content extracted from {uploaded_file.name}", None
        
        # Process with legal chunker
        chunks = chunker.legal_aware_chunking(text_content, max_chunk_size=1200)
        
        if not chunks:
            return False, f"No valid chunks extracted from {uploaded_file.name}", None
        
        # Generate embeddings and upload
        points = []
        upload_date = datetime.now().isoformat()
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = embedding_model.encode(chunk['text']).tolist()
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        **chunk['metadata'],
                        'text': chunk['text'],
                        'source_file': uploaded_file.name,
                        'chunk_index': i,
                        'upload_date': upload_date,
                        'content_hash': content_hash
                    }
                )
                points.append(point)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i} from {uploaded_file.name}: {e}")
                continue
        
        if not points:
            return False, f"No valid points created for {uploaded_file.name}", None
        
        # Upload to Qdrant in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                qdrant_client.upsert(
                    collection_name="legal_regulations",
                    points=batch
                )
                total_uploaded += len(batch)
                
                if progress_bar:
                    progress_bar.progress((i + len(batch)) / len(points))
                    
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        if total_uploaded > 0:
            return True, f"Successfully uploaded {uploaded_file.name} ({total_uploaded} chunks)", content_hash
        else:
            return False, f"Failed to upload any chunks from {uploaded_file.name}", None
            
    except Exception as e:
        logger.error(f"Error processing {uploaded_file.name}: {e}", exc_info=True)
        return False, f"Error processing {uploaded_file.name}: {e}", None

def delete_document_from_qdrant(filename: str) -> bool:
    """Delete all chunks of a document from Qdrant with proper pagination"""
    try:
        # Get all points for this document using proper pagination
        all_point_ids = []
        next_page_offset = None
        
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=filename))]
                ),
                limit=1000,
                offset=next_page_offset,
                with_payload=False,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            
            if not points:
                break
            
            # Extract point IDs from this batch
            batch_point_ids = [point.id for point in points]
            all_point_ids.extend(batch_point_ids)
            
            # If no next page offset, we've processed all points
            if next_page_offset is None:
                break
        
        if not all_point_ids:
            return True  # No points to delete
        
        # Delete points in batches
        batch_size = 100
        deleted_count = 0
        
        for i in range(0, len(all_point_ids), batch_size):
            batch_ids = all_point_ids[i:i + batch_size]
            qdrant_client.delete(
                collection_name="legal_regulations",
                points_selector=batch_ids
            )
            deleted_count += len(batch_ids)
        
        # Verify deletion was successful
        remaining_count = qdrant_client.count(
            collection_name="legal_regulations",
            count_filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=filename))]
            )
        ).count
        
        if remaining_count > 0:
            st.warning(f"⚠️ Warning: {remaining_count} chunks of {filename} may still remain in database")
            return False
        
        logger.info(f"🗑️ Successfully deleted {deleted_count} chunks from {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting document from Qdrant: {e}", exc_info=True)
        return False

def get_uploaded_documents() -> List[Dict[str, Any]]:
    """Get list of uploaded documents from Qdrant"""
    try:
        # Get all unique documents by scrolling through all points
        all_documents = {}
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
            
            for point in points:
                payload = point.payload
                filename = payload.get('source_file', 'Unknown')
                
                if filename not in all_documents:
                    all_documents[filename] = {
                        'filename': filename,
                        'upload_date': payload.get('upload_date', 'Unknown'),
                        'jurisdiction': payload.get('jurisdiction', 'Unknown'),
                        'chunk_count': 0,
                        'content_hash': payload.get('content_hash', 'Unknown')
                    }
                
                all_documents[filename]['chunk_count'] += 1
            
            if next_page_offset is None:
                break
        
        return list(all_documents.values())
        
    except Exception as e:
        st.error(f"Error getting uploaded documents: {e}")
        return []

# Check authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.error("🔒 Access denied. Please log in through the main application.")
    st.stop()

# Initialize admin systems
if not init_admin_systems():
    st.error("Failed to initialize admin systems")
    st.stop()

# Admin Panel UI
st.title("🔧 Admin Panel")
st.markdown("---")

# User Management Section
st.header("👥 User Management")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Add New User")
    with st.form("add_user_form"):
        client_name = st.text_input("Client Name", placeholder="Enter client name")
        email = st.text_input("Email (Optional)", placeholder="client@example.com")
        
        col_tier, col_days = st.columns(2)
        with col_tier:
            subscription_tier = st.selectbox("Subscription Tier", ["basic", "premium", "enterprise"])
        with col_days:
            days_valid = st.number_input("Days Valid", min_value=1, max_value=3650, value=365)
        
        submitted = st.form_submit_button("Create User", use_container_width=True)
        
        if submitted and client_name:
            try:
                access_code = user_manager.add_user(
                    client_name=client_name,
                    email=email if email else None,
                    subscription_tier=subscription_tier,
                    days_valid=days_valid
                )
                st.success(f"✅ User created successfully!")
                st.code(f"Access Code: {access_code}", language="text")
            except Exception as e:
                st.error(f"❌ Error creating user: {e}")

with col2:
    st.subheader("Quick Stats")
    try:
        users = user_manager.get_all_users()
        active_users = len([u for u in users if u[5]])  # is_active column
        st.metric("Total Users", len(users))
        st.metric("Active Users", active_users)
    except Exception as e:
        st.error(f"Error loading stats: {e}")

# User List
st.subheader("📋 User List")
try:
    users = user_manager.get_all_users()
    
    if users:
        for user in users:
            access_code, client_name, email, created_at, last_login, is_active, expires_at, subscription_tier = user
            
            with st.expander(f"👤 {client_name} ({access_code})"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Email:** {email or 'Not provided'}")
                    st.write(f"**Created:** {created_at}")
                    st.write(f"**Last Login:** {last_login or 'Never'}")
                    st.write(f"**Expires:** {expires_at}")
                    st.write(f"**Tier:** {subscription_tier}")
                
                with col2:
                    status = "🟢 Active" if is_active else "🔴 Inactive"
                    st.write(f"**Status:** {status}")
                
                with col3:
                    if is_active:
                        if st.button(f"🚫 Deactivate", key=f"deactivate_{access_code}"):
                            if user_manager.deactivate_user(access_code):
                                st.success("User deactivated")
                                st.rerun()
                            else:
                                st.error("Failed to deactivate user")
    else:
        st.info("No users found")
        
except Exception as e:
    st.error(f"Error loading users: {e}")

st.markdown("---")

# Document Management Section
st.header("📚 Document Management")

# Upload section
st.subheader("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload legal documents",
    type=['pdf', 'docx', 'txt', 'xml'],
    accept_multiple_files=True,
    help="Upload employment law documents for NY, NJ, and CT"
)

if uploaded_files:
    if st.button("🚀 Process Documents", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_uploads = 0
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")
            
            success, message, content_hash = process_uploaded_file(
                uploaded_file, 
                legal_chunker, 
                qdrant_client, 
                embedding_model,
                progress_bar,
                status_text
            )
            
            if success:
                successful_uploads += 1
                status_text.success(message)
            else:
                status_text.error(message)
            
            progress_bar.progress((i + 1) / total_files)
        
        status_text.text(f"✅ Completed! {successful_uploads}/{total_files} files processed successfully.")
        
        if successful_uploads > 0:
            st.success(f"🎉 Successfully processed {successful_uploads} documents!")
            st.rerun()

# Document list
st.subheader("📋 Uploaded Documents")

documents = get_uploaded_documents()

if documents:
    for doc in documents:
        with st.expander(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Upload Date:** {doc['upload_date']}")
                st.write(f"**Jurisdiction:** {doc['jurisdiction']}")
                st.write(f"**Chunks:** {doc['chunk_count']}")
                st.write(f"**Content Hash:** {doc['content_hash'][:16]}...")
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{doc['filename']}"):
                    if delete_document_from_qdrant(doc['filename']):
                        st.success(f"✅ Deleted {doc['filename']}")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to delete {doc['filename']}")
else:
    st.info("📭 No documents uploaded yet.")