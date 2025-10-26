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
            st.info("✅ Created/verified source_file payload index")
            
            # Create index for content_hash field
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="content_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
            st.info("✅ Created/verified content_hash payload index")
            
        except Exception as index_error:
            # Indexes might already exist, which is fine
            if "already exists" not in str(index_error).lower():
                st.warning(f"⚠️ Could not create payload indexes: {index_error}")
        
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
                st.info("✅ Created source_file payload index for new collection")
                
                # Create index for content_hash field
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content_hash",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                st.info("✅ Created content_hash payload index for new collection")
                
            except Exception as index_error:
                st.warning(f"⚠️ Could not create payload indexes for new collection: {index_error}")
            
            return True
        except Exception as e:
            st.error(f"Failed to create collection: {e}")
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
        
        # Generate embeddings and upload to Qdrant
        points = []
        upload_date = datetime.now().isoformat()
        
        st.info(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        
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
                        'source_file': filename,
                        'chunk_index': i,
                        'upload_date': upload_date,
                        'content_hash': content_hash  # Store the document hash
                    }
                )
                points.append(point)
                
            except Exception as e:
                st.error(f"❌ Error processing chunk {i} from {filename}: {e}")
                st.error(f"🔧 Chunk content preview: {chunk.get('text', 'No text')[:100]}...")
                st.error(f"Error processing chunk {i} from {filename}: {e}")
                continue
        
        if not points:
            st.error(f"❌ No valid points created for {filename} - this indicates a serious issue")
            st.info(f"📊 Original chunks count: {len(chunks)}")
            st.error(f"❌ No valid points created for {filename}")
            return False
        
        st.info(f"📤 Uploading {len(points)} points to Qdrant...")
        
        # Upload to Qdrant in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                st.info(f"📤 Uploading batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
                qdrant_client.upsert(
                    collection_name="legal_regulations",
                    points=batch
                )
                total_uploaded += len(batch)
                st.info(f"📊 Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            except Exception as e:
                st.error(f"❌ Error uploading batch {i//batch_size + 1}: {e}")
                st.error(f"🔧 Batch size: {len(batch)}, Batch preview: {[p.id for p in batch[:3]]}")
                st.error(f"Error uploading batch {i//batch_size + 1}: {e}")
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
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="login-card">
        <h1>⚖️ PEO Compliance Assistant</h1>
        <p>Access employment law guidance for NY, NJ, and CT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form"):
        st.markdown("### 🔐 Access Required")
        access_code = st.text_input(
            "Enter your access code:",
            type="password",
            placeholder="Enter your unique access code"
        )
        
        submitted = st.form_submit_button("Access System", use_container_width=True)
        
        if submitted:
            if access_code:
                if authenticate_user(access_code):
                    st.success("✅ Access granted! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid access code. Please check your code and try again.")
            else:
                st.warning("⚠️ Please enter your access code.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_main_application():
    """Display main application interface"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="dashboard-header">
            <h1>⚖️ PEO Compliance Assistant</h1>
            <p>Employment law guidance for New York, New Jersey, and Connecticut</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
    
    # Main tabs
    tab1, tab2 = st.tabs(["💬 Legal Assistant", "📚 Knowledge Base"])
    
    with tab1:
        show_legal_assistant()
    
    with tab2:
        show_knowledge_base()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_legal_assistant():
    """Display legal assistant chat interface"""
    st.markdown("### 💬 Ask Your Legal Question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about employment law in NY, NJ, or CT..."):
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

    # Only show process button if there are new files and not currently processing
    elif new_files and not st.session_state.is_processing:
        if st.button("🚀 Process Documents", use_container_width=True):
            st.session_state.is_processing = True
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
                        # Mark this file as processed
                        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                        st.session_state.processed_file_ids.add(file_id)

                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

                progress_bar.progress((i + 1) / total_files)

            status_text.text(f"✅ Completed! {successful_uploads}/{total_files} files processed successfully.")

            if successful_uploads > 0:
                st.success(f"🎉 Successfully processed {successful_uploads} documents!")

            # Mark processing as complete and reset processing flag
            st.session_state.is_processing = False
            st.session_state.processing_complete = True
            st.rerun()

    elif uploaded_files and not new_files and not st.session_state.processing_complete:
        st.info("ℹ️ These files have already been processed in this session. Clear to upload more.")
    
    # Document management section
    st.markdown("#### 📋 Uploaded Documents")
    
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
        st.info("📭 No documents uploaded yet. Upload some legal documents to get started!")

if __name__ == "__main__":
    main()