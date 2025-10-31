import streamlit as st
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import hashlib
import io
from typing import List, Dict, Any
import traceback

# Load environment variables
load_dotenv()

# Import custom modules
from user_management import UserManager
from processing_manager import ProcessingSessionManager
from advanced_chunking import (
    LegalSemanticChunker, 
    extract_pdf_text, 
    extract_docx_text,
    iterative_pdf_extraction,
    iterative_docx_extraction,
    iterative_text_extraction,
    strip_xml_tags,
    detect_document_format
)
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Initialize components
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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
    
    # Initialize processing session manager
    processing_manager = ProcessingSessionManager()
    
    # Initialize legal chunker
    legal_chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY", ""))
    
except Exception as e:
    st.error(f"Failed to initialize components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PEO Compliance Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = os.path.join("styles", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def ensure_collection_exists():
    """Ensure the Qdrant collection exists with proper configuration"""
    collection_name = "legal_regulations"
    
    try:
        # Try to get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        
        # Ensure payload indexes exist for efficient querying
        try:
            from qdrant_client.models import PayloadSchemaType
            
            # Create index for source_file field
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Create index for content_hash field
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="content_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
        except Exception as index_error:
            # Indexes might already exist, which is fine
            if "already exists" not in str(index_error).lower():
                pass  # Silently handle index creation issues
        
        return True
    except Exception:
        # Collection doesn't exist, create it
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
            # Create payload indexes for the new collection
            try:
                from qdrant_client.models import PayloadSchemaType
                
                # Create index for source_file field
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="source_file",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                # Create index for content_hash field
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content_hash",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
            except Exception as index_error:
                pass  # Silently handle index creation issues
            
            return True
        except Exception as e:
            return False

def calculate_content_hash(text: str) -> str:
    """Calculate SHA-256 hash of content for tracking purposes"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def delete_document_from_qdrant(filename: str) -> bool:
    """Delete all chunks of a document from Qdrant"""
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
                limit=1000,  # Process in smaller batches for better memory management
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
        
        st.info(f"🗑️ Successfully deleted {deleted_count} chunks from {filename}")
        return True
        
    except Exception as e:
        st.error(f"Error deleting document from Qdrant: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False
def process_and_upload_document_iterative(uploaded_file, filename: str, progress_bar, status_text) -> bool:
    """Process document iteratively with Supabase tracking to handle large files"""
    session_id = None
    
    try:
        # Calculate file hash and size
        if hasattr(uploaded_file, 'read'):
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            file_content_bytes = uploaded_file.read()
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)  # Reset for later use
        else:
            file_content_bytes = uploaded_file
        
        file_size = len(file_content_bytes)
        file_hash = hashlib.sha256(file_content_bytes).hexdigest()
        
        # Create processing session
        session_id = processing_manager.create_session(filename, file_size, file_hash)
        status_text.text(f"📄 Created processing session for {filename}")
        progress_bar.progress(0.05)
        
        # Determine file type and set up iterative extraction
        file_type = uploaded_file.type if hasattr(uploaded_file, 'type') else 'text/plain'
        
        total_chunks_uploaded = 0
        batch_number = 0
        upload_date = datetime.now().isoformat()
        
        # Update session to extraction phase
        processing_manager.update_session_progress(
            session_id, 0, 'extracting_content', 
            metadata={'file_type': file_type, 'file_size_mb': file_size / (1024 * 1024)}
        )
        
        # Choose appropriate iterative extraction method
        if file_type == "application/pdf":
            status_text.text(f"📖 Processing PDF: {filename} (iterative extraction)")
            content_generator = iterative_pdf_extraction(uploaded_file, page_batch_size=3)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            status_text.text(f"📝 Processing DOCX: {filename} (iterative extraction)")
            content_generator = iterative_docx_extraction(uploaded_file, paragraph_batch_size=30)
        else:
            # Text or XML file
            if isinstance(file_content_bytes, bytes):
                file_content_str = file_content_bytes.decode('utf-8')
            else:
                file_content_str = str(file_content_bytes)
            status_text.text(f"📄 Processing text file: {filename} (iterative extraction)")
            content_generator = iterative_text_extraction(file_content_str, chunk_size=30000)
        
        # Process content in batches
        for content_batch in content_generator:
            batch_number += 1
            batch_content = content_batch['content']
            batch_info = content_batch['batch_info']
            extraction_progress = content_batch['progress']
            
            status_text.text(f"🔄 Processing {batch_info}...")
            
            # Update progress (extraction phase takes 20% of total progress)
            current_progress = 0.05 + (extraction_progress * 0.2)
            progress_bar.progress(current_progress)
            
            try:
                # Chunk the content using legal chunker
                status_text.text(f"⚖️ Legal chunking for {batch_info}...")
                chunks = legal_chunker.legal_aware_chunking(batch_content, max_chunk_size=1200)
                
                if not chunks:
                    status_text.text(f"⚠️ No valid chunks from {batch_info}, skipping...")
                    continue
                
                # Update session with chunking progress
                processing_manager.update_session_progress(
                    session_id, total_chunks_uploaded, 'chunking_content',
                    metadata={
                        'current_batch': batch_number,
                        'batch_info': batch_info,
                        'chunks_in_batch': len(chunks)
                    }
                )
                
                # Process chunks in smaller sub-batches for memory efficiency
                sub_batch_size = 50  # Process 50 chunks at a time
                batch_chunks_uploaded = 0
                
                for sub_batch_start in range(0, len(chunks), sub_batch_size):
                    sub_batch_end = min(sub_batch_start + sub_batch_size, len(chunks))
                    sub_batch_chunks = chunks[sub_batch_start:sub_batch_end]
                    
                    status_text.text(f"🧠 Generating embeddings for {batch_info} (sub-batch {sub_batch_start//sub_batch_size + 1})...")
                    
                    # Extract texts for embedding
                    sub_batch_texts = [ch['text'] for ch in sub_batch_chunks]
                    
                    # Generate embeddings
                    try:
                        sub_batch_embeddings = embedding_model.encode(sub_batch_texts, show_progress_bar=False)
                    except Exception as embedding_error:
                        st.error(f"❌ Embedding generation failed for {batch_info}: {embedding_error}")
                        continue
                    
                    # Create points for Qdrant
                    sub_batch_points = []
                    for i, (ch, embedding) in enumerate(zip(sub_batch_chunks, sub_batch_embeddings)):
                        try:
                            vector = embedding.tolist()
                            
                            point = PointStruct(
                                id=str(uuid.uuid4()),
                                vector=vector,
                                payload={
                                    **ch['metadata'],
                                    'text': ch['text'],
                                    'source_file': filename,
                                    'chunk_index': total_chunks_uploaded + batch_chunks_uploaded + i,
                                    'upload_date': upload_date,
                                    'content_hash': file_hash,
                                    'batch_info': batch_info,
                                    'processing_session_id': session_id
                                }
                            )
                            sub_batch_points.append(point)
                        except Exception as point_error:
                            st.error(f"❌ Point creation failed: {point_error}")
                            continue
                    
                    if not sub_batch_points:
                        continue
                    
                    # Upload to Qdrant
                    status_text.text(f"☁️ Uploading {len(sub_batch_points)} chunks to database...")
                    try:
                        qdrant_client.upsert(
                            collection_name="legal_regulations",
                            points=sub_batch_points
                        )
                        
                        batch_chunks_uploaded += len(sub_batch_points)
                        total_chunks_uploaded += len(sub_batch_points)
                        
                        # Update session progress
                        processing_manager.update_session_progress(
                            session_id, total_chunks_uploaded, 'uploading_chunks',
                            metadata={
                                'current_batch': batch_number,
                                'batch_info': batch_info,
                                'total_batches_processed': batch_number
                            }
                        )
                        
                        # Create checkpoint
                        processing_manager.create_checkpoint(
                            session_id, 'batch_completed', total_chunks_uploaded, batch_number,
                            {'batch_info': batch_info, 'chunks_in_batch': batch_chunks_uploaded}
                        )
                        
                        status_text.text(f"✅ Uploaded batch from {batch_info} ({total_chunks_uploaded} total chunks)")
                        
                        # Update progress (remaining 75% for processing and uploading)
                        processing_progress = 0.25 + (extraction_progress * 0.75)
                        progress_bar.progress(processing_progress)
                        
                    except Exception as upload_error:
                        st.error(f"❌ Upload failed for {batch_info}: {upload_error}")
                        continue
                    
                    # Clear memory
                    del sub_batch_texts, sub_batch_embeddings, sub_batch_points
                
                # Clear batch memory
                del chunks
                
            except Exception as batch_error:
                st.error(f"❌ Error processing {batch_info}: {batch_error}")
                continue
        
        # Complete the session
        if total_chunks_uploaded > 0:
            processing_manager.complete_session(session_id, total_chunks_uploaded)
            progress_bar.progress(1.0)
            status_text.text(f"✅ Successfully processed {filename} ({total_chunks_uploaded} chunks)")
            st.success(f"🎉 Successfully processed {filename} ({total_chunks_uploaded} chunks)")
            return True
        else:
            processing_manager.fail_session(session_id, "No chunks were successfully uploaded")
            st.error(f"❌ No chunks were uploaded from {filename}")
            return False
            
    except Exception as e:
        error_msg = f"Critical error processing {filename}: {e}"
        st.error(f"❌ {error_msg}")
        st.error(f"🔧 Error details: {traceback.format_exc()}")
        
        if session_id:
            processing_manager.fail_session(session_id, error_msg)
        
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

def process_and_upload_document(file_content: str, filename: str) -> bool:
    """Process document and upload to Qdrant"""
    try:
        # Calculate content hash for tracking purposes
        content_hash = calculate_content_hash(file_content)

        st.info(f"📄 Processing {filename}...")
        
        # Process the document using legal chunker
        st.info(f"🔄 Processing {filename} into semantic chunks...")
        chunks = legal_chunker.legal_aware_chunking(file_content, max_chunk_size=1200)
        
        if not chunks:
            st.error(f"❌ No valid chunks extracted from {filename} - this is unusual")
            st.info(f"📝 File content length: {len(file_content)} characters")
            st.info(f"📝 File content preview: {file_content[:200]}...")
            st.error(f"❌ No valid chunks extracted from {filename}")
            return False
        
        st.info(f"📝 Extracted {len(chunks)} chunks from {filename}")
        
        # Process chunks in batches to reduce memory usage for large documents
        batch_size = 100  # Process 100 chunks at a time
        total_chunks = len(chunks)
        total_uploaded = 0
        upload_date = datetime.now().isoformat()
        
        st.info(f"🧠 Processing {total_chunks} chunks in batches of {batch_size}...")
        
        # Process chunks in batches to manage memory usage
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            batch_number = (batch_start // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            st.info(f"🔄 Processing batch {batch_number}/{total_batches} ({len(batch_chunks)} chunks)")
            
            try:
                # Extract texts for this batch only
                batch_texts = [ch['text'] for ch in batch_chunks]
                
                # Generate embeddings for this batch
                batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
                
                # Create points for this batch
                batch_points = []
                for i, (ch, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    vector = embedding.tolist()
                    
                    # Create point
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            **ch['metadata'],
                            'text': ch['text'],
                            'source_file': filename,
                            'chunk_index': batch_start + i,
                            'upload_date': upload_date,
                            'content_hash': content_hash  # Store the document hash
                        }
                    )
                    batch_points.append(point)
                
                # Upload this batch to Qdrant
                qdrant_client.upsert(
                    collection_name="legal_regulations",
                    points=batch_points
                )
                
                total_uploaded += len(batch_points)
                st.info(f"✅ Uploaded batch {batch_number}/{total_batches} ({total_uploaded}/{total_chunks} chunks total)")
                
                # Clear batch variables to free memory
                del batch_texts, batch_embeddings, batch_points
                
            except Exception as batch_error:
                st.error(f"❌ Error processing batch {batch_number}: {batch_error}")
                st.error(f"🔧 Batch size: {len(batch_chunks)}")
                continue
        
        if total_uploaded > 0:
            st.success(f"✅ Successfully uploaded {filename} ({total_uploaded} chunks)")
            return True
        else:
            st.error(f"❌ Failed to upload any chunks from {filename} - total_uploaded = {total_uploaded}")
            st.error(f"❌ Failed to upload any chunks from {filename}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error processing {filename}: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"🔧 File content length: {len(file_content) if file_content else 'None'}")
        st.error(f"🔧 Content hash: {content_hash if 'content_hash' in locals() else 'Not calculated'}")
        return False

def process_and_upload_document_with_progress(file_content: str, filename: str, progress_bar, status_text) -> bool:
    """Process document and upload to Qdrant with enhanced progress tracking"""
    try:
        # Calculate content hash for tracking purposes
        content_hash = calculate_content_hash(file_content)
        
        status_text.text(f"📄 Analyzing {filename}...")
        progress_bar.progress(0.1)
        
        # Process the document using legal chunker
        status_text.text(f"🔄 Extracting semantic chunks from {filename}...")
        chunks = legal_chunker.legal_aware_chunking(file_content, max_chunk_size=1200)
        
        if not chunks:
            st.error(f"❌ No valid chunks extracted from {filename}")
            st.info(f"📝 File content length: {len(file_content)} characters")
            st.info(f"📝 File content preview: {file_content[:200]}...")
            return False
        
        progress_bar.progress(0.2)
        status_text.text(f"📝 Extracted {len(chunks)} chunks from {filename}")
        
        # Process chunks in batches to reduce memory usage for large documents
        batch_size = 100  # Process 100 chunks at a time
        total_chunks = len(chunks)
        total_uploaded = 0
        upload_date = datetime.now().isoformat()
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        status_text.text(f"🧠 Processing {total_chunks} chunks in {total_batches} batches...")
        
        # Process chunks in batches to manage memory usage
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            batch_number = (batch_start // batch_size) + 1
            
            # Update progress based on batch completion
            batch_progress = 0.2 + (0.8 * (batch_number - 1) / total_batches)
            progress_bar.progress(batch_progress)
            status_text.text(f"🔄 Processing batch {batch_number}/{total_batches} ({len(batch_chunks)} chunks)")
            
            try:
                # Extract texts for this batch only
                status_text.text(f"📝 Extracting text from batch {batch_number}/{total_batches}...")
                batch_texts = [ch['text'] for ch in batch_chunks]
                
                # Generate embeddings for this batch
                status_text.text(f"🧠 Generating embeddings for batch {batch_number}/{total_batches}...")
                try:
                    batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
                except Exception as embedding_error:
                    st.error(f"❌ Embedding generation failed for batch {batch_number}: {embedding_error}")
                    st.error(f"🔧 Batch texts length: {len(batch_texts)}")
                    st.error(f"🔧 Sample text length: {len(batch_texts[0]) if batch_texts else 'No texts'}")
                    continue
                
                # Create points for this batch
                status_text.text(f"📦 Creating data points for batch {batch_number}/{total_batches}...")
                batch_points = []
                for i, (ch, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    try:
                        vector = embedding.tolist()
                        
                        # Create point
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                **ch['metadata'],
                                'text': ch['text'],
                                'source_file': filename,
                                'chunk_index': batch_start + i,
                                'upload_date': upload_date,
                                'content_hash': content_hash
                            }
                        )
                        batch_points.append(point)
                    except Exception as point_error:
                        st.error(f"❌ Point creation failed for chunk {i} in batch {batch_number}: {point_error}")
                        continue
                
                if not batch_points:
                    st.error(f"❌ No valid points created for batch {batch_number}")
                    continue
                
                # Upload this batch to Qdrant
                status_text.text(f"☁️ Uploading batch {batch_number}/{total_batches} to database...")
                try:
                    qdrant_client.upsert(
                        collection_name="legal_regulations",
                        points=batch_points
                    )
                    
                    batch_uploaded = len(batch_points)
                    total_uploaded += batch_uploaded
                    
                    # Update progress
                    batch_progress = 0.2 + (0.8 * batch_number / total_batches)
                    progress_bar.progress(batch_progress)
                    status_text.text(f"✅ Uploaded batch {batch_number}/{total_batches} ({total_uploaded}/{total_chunks} chunks total)")
                    
                except Exception as upload_error:
                    st.error(f"❌ Qdrant upload failed for batch {batch_number}: {upload_error}")
                    st.error(f"🔧 Batch points count: {len(batch_points)}")
                    st.error(f"🔧 Upload error details: {traceback.format_exc()}")
                    continue
                
                # Clear batch variables to free memory
                del batch_texts, batch_embeddings, batch_points
                
            except Exception as batch_error:
                st.error(f"❌ Critical error in batch {batch_number}: {batch_error}")
                st.error(f"🔧 Batch size: {len(batch_chunks)}")
                st.error(f"🔧 Batch error details: {traceback.format_exc()}")
                
                # Try to continue with next batch
                continue
        
        # Final progress update
        progress_bar.progress(1.0)
        
        if total_uploaded > 0:
            status_text.text(f"✅ Successfully uploaded {filename} ({total_uploaded}/{total_chunks} chunks)")
            st.success(f"✅ Successfully uploaded {filename} ({total_uploaded} chunks)")
            return True
        else:
            status_text.text(f"❌ Failed to upload any chunks from {filename}")
            st.error(f"❌ Failed to upload any chunks from {filename} - total_uploaded = {total_uploaded}")
            return False
            
    except Exception as e:
        st.error(f"❌ Critical error processing {filename}: {e}")
        st.error(f"🔧 Error details: {traceback.format_exc()}")
        st.error(f"🔧 File content length: {len(file_content) if file_content else 'None'}")
        st.error(f"🔧 Content hash: {content_hash if 'content_hash' in locals() else 'Not calculated'}")
        return False

def search_legal_database(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search the legal database using semantic similarity"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search Qdrant
        search_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        results = []
        for result in search_results:
            results.append({
                'text': result.payload.get('text', ''),
                'citation': result.payload.get('citation', 'Unknown'),
                'jurisdiction': result.payload.get('jurisdiction', 'Unknown'),
                'section_number': result.payload.get('section_number', 'Unknown'),
                'score': result.score,
                'source_file': result.payload.get('source_file', 'Unknown')
            })
        
        return results
        
    except Exception as e:
        st.error(f"Error searching database: {e}")
        return []

def generate_legal_response(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate response using OpenAI with legal context"""
    try:
        # Prepare context from search results
        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"\n--- Source {i} ---\n"
            context += f"Citation: {result['citation']}\n"
            context += f"Jurisdiction: {result['jurisdiction']}\n"
            context += f"Content: {result['text']}\n"
        
        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": LEGAL_COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Legal Sources:\n{context}"}
        ]
        
        # Generate response
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

# Authentication functions
def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    return st.session_state.authenticated

def authenticate_user(access_code: str) -> bool:
    """Authenticate user with access code"""
    if user_manager.validate_access_code(access_code):
        if user_manager.create_session(access_code, st.session_state.session_id):
            user_manager.update_last_login(access_code)
            st.session_state.authenticated = True
            st.session_state.access_code = access_code
            return True
    return False

def logout_user():
    """Logout current user"""
    st.session_state.authenticated = False
    if 'access_code' in st.session_state:
        del st.session_state.access_code
    if 'session_id' in st.session_state:
        del st.session_state.session_id
    st.rerun()

# Main application
def main():
    # Ensure collection exists
    if not ensure_collection_exists():
        st.error("Failed to initialize database. Please check your configuration.")
        st.stop()
    
    # Check authentication
    if not check_authentication():
        show_login_page()
        return
    
    # Validate session
    if not user_manager.is_session_valid(st.session_state.session_id):
        st.error("Your session has expired. Please log in again.")
        logout_user()
        return
    
    # Update session activity
    user_manager.update_session_activity(st.session_state.session_id)
    
    # Show main application
    show_main_application()

def show_login_page():
    """Display login page"""
    st.title("🔐 PEO Compliance Assistant")
    st.markdown("### Access Required")
    
    # Create a simple login form
    with st.form("login_form"):
        st.write("Please enter your access code to continue:")
        access_code = st.text_input("Access Code", type="password")
        submit_button = st.form_submit_button("Access System")
        
        if submit_button:
            if access_code:
                if authenticate_user(access_code):
                    st.success("✅ Access granted! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid access code. Please try again.")
            else:
                st.error("❌ Please enter an access code.")
    
    st.info("💡 Contact your administrator if you need an access code.")

def show_main_application():
    """Display main application interface"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚖️ PEO Compliance Assistant")
        st.markdown("*Employment law guidance for New York, New Jersey, and Connecticut*")
    
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
    
    # Show legal assistant content directly
    show_legal_assistant_content()
    
    # Handle chat input outside of tabs
    if prompt := st.chat_input("Ask about employment law in NY, NJ, or CT..."):
        handle_chat_input(prompt)

def show_legal_assistant_content():
    """Display legal assistant chat interface content (without chat input)"""
    st.markdown("### 💬 Ask Your Legal Question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_chat_input(prompt):
    """Handle chat input and generate response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching legal database..."):
            # Search for relevant legal information
            search_results = search_legal_database(prompt, limit=5)
            
            # Generate response
            response = generate_legal_response(prompt, search_results)
            
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the display
    st.rerun()
def show_knowledge_base():
    """Display knowledge base management interface"""
    st.markdown("### 📚 Knowledge Base Management")

    # Initialize session state for tracking processed files and processing status
    if "processed_file_ids" not in st.session_state:
        st.session_state.processed_file_ids = set()
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    # Upload section
    st.markdown("#### 📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload legal documents (PDF, DOCX, TXT, XML)",
        type=['pdf', 'docx', 'txt', 'xml'],
        accept_multiple_files=True,
        help="Upload employment law documents for NY, NJ, and CT",
        key="doc_uploader"
    )

    # Check if files are new (not already processed in this session)
    new_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.processed_file_ids:
                new_files.append(uploaded_file)

    # Show processing complete message and clear button
    if st.session_state.processing_complete:
        st.success("✅ Processing complete! Upload new files or clear to continue.")
        if st.button("🔄 Clear and Upload More", use_container_width=True):
            st.session_state.processed_file_ids.clear()
            st.session_state.processing_complete = False
            st.rerun()

    # New iterative processing button for large files
    if new_files and not st.session_state.is_processing and not st.session_state.processing_complete:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Process Documents (Standard)", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                successful_uploads = 0
                total_files = len(new_files)

                for i, uploaded_file in enumerate(new_files):
                    status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")

                    try:
                        # Extract text based on file type
                        if uploaded_file.type == "application/pdf":
                            file_content = extract_pdf_text(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            file_content = extract_docx_text(uploaded_file)
                        else:
                            # Text or XML file
                            file_content = str(uploaded_file.read(), "utf-8")

                        # Process and upload
                        if process_and_upload_document(file_content, uploaded_file.name):
                            successful_uploads += 1
                            # Mark this file as processed IMMEDIATELY
                            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                            st.session_state.processed_file_ids.add(file_id)

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")

                    progress_bar.progress((i + 1) / total_files)

                status_text.text(f"✅ Completed! {successful_uploads}/{total_files} files processed successfully.")

                if successful_uploads > 0:
                    st.success(f"🎉 Successfully processed {successful_uploads} documents!")
                    # Mark processing as complete - this prevents the button from showing again
                    st.session_state.processing_complete = True
        
        with col2:
            if st.button("🔄 Process Documents (Iterative - Large Files)", use_container_width=True):
                st.session_state.is_processing = True
                progress_bar = st.progress(0)
                status_text = st.empty()

                successful_uploads = 0
                total_files = len(new_files)

                for i, uploaded_file in enumerate(new_files):
                    status_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name} (iterative mode)")

                    try:
                        # Use iterative processing
                        if process_and_upload_document_iterative(uploaded_file, uploaded_file.name, progress_bar, status_text):
                            successful_uploads += 1
                            # Mark this file as processed IMMEDIATELY
                            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                            st.session_state.processed_file_ids.add(file_id)

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")

                    # Update overall progress
                    overall_progress = (i + 1) / total_files
                    progress_bar.progress(overall_progress)

                status_text.text(f"✅ Completed! {successful_uploads}/{total_files} files processed successfully.")

                if successful_uploads > 0:
                    st.success(f"🎉 Successfully processed {successful_uploads} documents!")
                    # Mark processing as complete - this prevents the button from showing again
                    st.session_state.processing_complete = True
                
                st.session_state.is_processing = False

    elif uploaded_files and not new_files and not st.session_state.processing_complete:
        st.info("ℹ️ These files have already been processed in this session. Clear to upload more.")
    
    # Document management section
    st.markdown("#### 📋 Uploaded Documents")
    
    # Show processing sessions
    processing_sessions = processing_manager.get_all_processing_sessions()
    
    if processing_sessions:
        st.markdown("##### 🔄 Processing Sessions")
        for session in processing_sessions:
            status_emoji = {
                'processing': '🔄',
                'completed': '✅', 
                'failed': '❌'
            }.get(session['status'], '❓')
            
            with st.expander(f"{status_emoji} {session['file_name']} - {session['status'].title()} ({session['chunks_uploaded']} chunks)"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Status:** {session['status'].title()}")
                    st.write(f"**Current Phase:** {session['current_phase']}")
                    st.write(f"**File Size:** {session['file_size'] / (1024*1024):.1f} MB")
                    st.write(f"**Chunks Uploaded:** {session['chunks_uploaded']}")
                    if session['total_chunks_expected']:
                        progress_pct = (session['chunks_uploaded'] / session['total_chunks_expected']) * 100
                        st.write(f"**Progress:** {progress_pct:.1f}%")
                    st.write(f"**Started:** {session['start_time']}")
                    if session['end_time']:
                        st.write(f"**Completed:** {session['end_time']}")
                    if session['error_message']:
                        st.error(f"**Error:** {session['error_message']}")
                
                with col2:
                    if st.button(f"🗑️ Delete Session", key=f"delete_session_{session['id']}"):
                        if processing_manager.delete_session(session['id']):
                            st.success(f"✅ Deleted session for {session['file_name']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete session")
    
    documents = get_uploaded_documents()
    
    if documents:
        st.markdown("##### 📚 Uploaded Documents")
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
        st.info("📭 No documents uploaded yet. Upload some legal documents to get started!")

if __name__ == "__main__":
    main()