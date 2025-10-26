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
    """Initialize admin systems with proper payload indexing"""
    collection_name = "legal_regulations"
    
    try:
        # Ensure collection exists
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            logger.info(f"✅ Collection '{collection_name}' exists with {collection_info.points_count} points")
        except Exception:
            # Collection doesn't exist, create it
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info(f"✅ Created collection '{collection_name}'")
        
        # Create payload indexes for efficient querying
        try:
            # Create index for content_hash field
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="content_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info("✅ Created/verified content_hash payload index")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"⚠️ Could not create content_hash index: {e}")
        
        try:
            # Create index for source_file field
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info("✅ Created/verified source_file payload index")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"⚠️ Could not create source_file index: {e}")
        
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
        logger.error(f"❌ Failed to initialize admin systems: {e}")
        return False

def process_uploaded_file(uploadedfile, chunker, qdrantclient, embeddingmodel, progressbar=None, statustext=None):
    """Process uploaded file with enhanced duplicate detection and logging"""
    try:
        # Read file content and calculate hash
        filecontent = uploadedfile.read()
        contenthash = hashlib.md5(filecontent).hexdigest()
        
        # Enhanced logging at start
        logger.info(f"📋 Processing file: {uploadedfile.name}")
        logger.info(f"📊 Content hash: {contenthash}")
        logger.info(f"📏 File size: {len(filecontent)} bytes")
        
        if statustext:
            statustext.info(f"🔍 Checking for duplicates: {uploadedfile.name}")
        
        try:
            # Check for existing content with same hash
            existingresult = qdrantclient.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(must=[
                    FieldCondition(key="content_hash", match=MatchValue(value=contenthash))
                ]),
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            existingpoints, _ = existingresult
            logger.info(f"🔍 Found {len(existingpoints)} existing chunks with content hash: {contenthash}")
            
            if existingpoints:
                existing_filename = existingpoints[0].payload.get("source_file", "unknown")
                logger.info(f"📁 Existing file: {existing_filename}, Current file: {uploadedfile.name}")
                
                # Check if it's the exact same file
                if existing_filename == uploadedfile.name:
                    logger.info("⏭️ SKIPPING - Same file, same content (true duplicate)")
                    if statustext:
                        statustext.warning(f"⏭️ Skipping {uploadedfile.name} - identical content already processed")
                    return True, f"Skipped {uploadedfile.name} - already processed with same content", contenthash
                else:
                    logger.info(f"⏭️ SKIPPING - Different filename but identical content (duplicate of {existing_filename})")
                    if statustext:
                        statustext.warning(f"⏭️ Skipping {uploadedfile.name} - identical content exists as {existing_filename}")
                    return True, f"Skipped {uploadedfile.name} - identical content exists as {existing_filename}", contenthash
            
            # Check for existing file with same name (for replacement)
            filenameresult = qdrantclient.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(must=[
                    FieldCondition(key="source_file", match=MatchValue(value=uploadedfile.name))
                ]),
                limit=10,
                with_payload=True,
                with_vectors=False
            )
            
            filenamepoints, _ = filenameresult
            
            if filenamepoints:
                logger.info(f"🔄 Found {len(filenamepoints)} existing chunks with filename: {uploadedfile.name} - will replace")
                if statustext:
                    statustext.info(f"🔄 Replacing existing version of {uploadedfile.name}")
                
                # Delete existing chunks for this filename
                delete_result = delete_document_from_qdrant(uploadedfile.name, qdrantclient)
                if not delete_result:
                    logger.error(f"❌ Failed to delete existing chunks for {uploadedfile.name}")
                    if statustext:
                        statustext.error(f"❌ Failed to delete existing version of {uploadedfile.name}")
                    return False, f"Failed to delete existing version of {uploadedfile.name}", None
            else:
                logger.info(f"✅ No existing chunks found - proceeding with fresh upload")
                if statustext:
                    statustext.info(f"✅ Processing new file: {uploadedfile.name}")
        
        except Exception as e:
            logger.error(f"Duplicate check failed for {uploadedfile.name}: {e}", exc_info=True)
            if statustext:
                statustext.error(f"❌ Duplicate check failed: {e}")
            # CRITICAL: Return False to stop processing if duplicate check fails
            return False, f"Duplicate check failed: {e}", None
        
        # Process the document
        if statustext:
            statustext.info(f"🔄 Processing document content...")
        
        # Reset file pointer and extract text based on file type
        uploadedfile.seek(0)
        
        if uploadedfile.type == "application/pdf":
            text_content = extract_pdf_text(uploadedfile)
        elif uploadedfile.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_content = extract_docx_text(uploadedfile)
        else:
            # Text or XML file
            text_content = filecontent.decode('utf-8')
        
        if not text_content or len(text_content.strip()) < 100:
            logger.error(f"❌ Insufficient content extracted from {uploadedfile.name}")
            if statustext:
                statustext.error(f"❌ Could not extract sufficient content from {uploadedfile.name}")
            return False, f"Could not extract sufficient content from {uploadedfile.name}", None
        
        logger.info(f"📝 Extracted {len(text_content)} characters from {uploadedfile.name}")
        
        # Chunk the document
        if statustext:
            statustext.info(f"🧠 Creating semantic chunks...")
        
        chunks = chunker.legal_aware_chunking(text_content, max_chunk_size=1200)
        
        if not chunks:
            logger.error(f"❌ No valid chunks extracted from {uploadedfile.name}")
            if statustext:
                statustext.error(f"❌ No valid chunks could be created from {uploadedfile.name}")
            return False, f"No valid chunks could be created from {uploadedfile.name}", None
        
        logger.info(f"📝 Created {len(chunks)} chunks from {uploadedfile.name}")
        
        # Generate embeddings and upload
        if statustext:
            statustext.info(f"🚀 Uploading {len(chunks)} chunks to database...")
        
        points = []
        upload_date = datetime.now().isoformat()
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = embeddingmodel.encode(chunk['text']).tolist()
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        **chunk['metadata'],
                        'text': chunk['text'],
                        'source_file': uploadedfile.name,
                        'chunk_index': i,
                        'upload_date': upload_date,
                        'content_hash': contenthash
                    }
                )
                points.append(point)
                
                if progressbar and i % 10 == 0:
                    progressbar.progress((i + 1) / len(chunks))
                
            except Exception as e:
                logger.error(f"❌ Error processing chunk {i} from {uploadedfile.name}: {e}")
                continue
        
        if not points:
            logger.error(f"❌ No valid points created for {uploadedfile.name}")
            if statustext:
                statustext.error(f"❌ Failed to create valid data points from {uploadedfile.name}")
            return False, f"Failed to create valid data points from {uploadedfile.name}", None
        
        # Upload to Qdrant in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                qdrantclient.upsert(
                    collection_name="legal_regulations",
                    points=batch
                )
                total_uploaded += len(batch)
                logger.info(f"📤 Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
                
                if progressbar:
                    progressbar.progress(min(1.0, (i + len(batch)) / len(points)))
                
            except Exception as e:
                logger.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        if total_uploaded > 0:
            logger.info(f"✅ Successfully uploaded {uploadedfile.name} ({total_uploaded} chunks)")
            if statustext:
                statustext.success(f"✅ Successfully uploaded {uploadedfile.name} ({total_uploaded} chunks)")
            
            # Verify upload
            verify_count = qdrantclient.count(
                collection_name="legal_regulations",
                count_filter=Filter(
                    must=[
                        FieldCondition(key="source_file", match=MatchValue(value=uploadedfile.name)),
                        FieldCondition(key="content_hash", match=MatchValue(value=contenthash))
                    ]
                )
            ).count
            
            logger.info(f"✅ Verification: {verify_count} chunks confirmed in database")
            
            return True, f"Successfully uploaded {uploadedfile.name} ({total_uploaded} chunks)", contenthash
        else:
            logger.error(f"❌ Failed to upload any chunks from {uploadedfile.name}")
            if statustext:
                statustext.error(f"❌ Failed to upload {uploadedfile.name}")
            return False, f"Failed to upload {uploadedfile.name}", None
            
    except Exception as e:
        logger.error(f"❌ Error processing {uploadedfile.name}: {e}", exc_info=True)
        if statustext:
            statustext.error(f"❌ Error processing {uploadedfile.name}: {e}")
        return False, f"Error processing {uploadedfile.name}: {e}", None

def delete_document_from_qdrant(filename: str, qdrantclient) -> bool:
    """Delete all chunks of a document from Qdrant with proper pagination"""
    try:
        logger.info(f"🗑️ Starting deletion of {filename}")
        
        # Get all points for this document using proper pagination
        all_point_ids = []
        next_page_offset = None
        
        while True:
            scroll_result = qdrantclient.scroll(
                collection_name="legal_regulations",
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_file", match=MatchValue(value=filename))]
                ),
                limit=1000,  # Process in batches for better memory management
                offset=next_page_offset,
                with_payload=False,
                with_vectors=False
            )
            
            points, next_page_offset = scroll_result
            
            if not points:
                break  # No more points to process
            
            # Extract point IDs from this batch
            batch_point_ids = [point.id for point in points]
            all_point_ids.extend(batch_point_ids)
            
            logger.info(f"🔍 Found {len(batch_point_ids)} more chunks to delete (total: {len(all_point_ids)})")
            
            # If no next page offset, we've processed all points
            if next_page_offset is None:
                break
        
        if not all_point_ids:
            logger.info(f"✅ No chunks found for {filename} - nothing to delete")
            return True  # No points to delete
        
        logger.info(f"🗑️ Deleting {len(all_point_ids)} chunks for {filename}")
        
        # Delete points in batches
        batch_size = 100
        deleted_count = 0
        
        for i in range(0, len(all_point_ids), batch_size):
            batch_ids = all_point_ids[i:i + batch_size]
            try:
                qdrantclient.delete(
                    collection_name="legal_regulations",
                    points_selector=batch_ids
                )
                deleted_count += len(batch_ids)
                logger.info(f"🗑️ Deleted batch {i//batch_size + 1}/{(len(all_point_ids)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"❌ Error deleting batch {i//batch_size + 1}: {e}")
                continue
        
        # Verify deletion was successful
        remaining_count = qdrantclient.count(
            collection_name="legal_regulations",
            count_filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=filename))]
            )
        ).count
        
        if remaining_count > 0:
            logger.warning(f"⚠️ Warning: {remaining_count} chunks of {filename} may still remain in database")
            return False
        
        logger.info(f"✅ Successfully deleted {deleted_count} chunks from {filename}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error deleting document from Qdrant: {e}", exc_info=True)
        return False

def get_uploaded_documents(qdrantclient) -> List[Dict[str, Any]]:
    """Get list of uploaded documents from Qdrant"""
    try:
        # Get all unique documents by scrolling through all points
        all_documents = {}
        next_page_offset = None
        
        while True:
            scroll_result = qdrantclient.scroll(
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
        logger.error(f"❌ Error getting uploaded documents: {e}")
        return []

def show_admin_panel():
    """Display admin panel interface"""
    st.title("🔧 Admin Panel")
    
    # Initialize admin systems
    if not init_admin_systems():
        st.error("❌ Failed to initialize admin systems")
        return
    
    # File upload section
    st.header("📤 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload legal documents",
        type=['pdf', 'docx', 'txt', 'xml'],
        accept_multiple_files=True,
        help="Upload employment law documents for processing"
    )
    
    if uploaded_files:
        if st.button("🚀 Process All Files", use_container_width=True):
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
                    progressbar=progress_bar,
                    statustext=status_text
                )
                
                if success:
                    successful_uploads += 1
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text(f"✅ Completed! {successful_uploads}/{total_files} files processed successfully.")
            
            if successful_uploads > 0:
                st.success(f"🎉 Successfully processed {successful_uploads} documents!")
                st.rerun()
    
    # Document management section
    st.header("📋 Document Management")
    
    documents = get_uploaded_documents(qdrant_client)
    
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
                        if delete_document_from_qdrant(doc['filename'], qdrant_client):
                            st.success(f"✅ Deleted {doc['filename']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete {doc['filename']}")
    else:
        st.info("📭 No documents uploaded yet.")

# Main execution
if __name__ == "__main__":
    show_admin_panel()