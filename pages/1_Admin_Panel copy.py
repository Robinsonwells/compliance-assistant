import streamlit as st
from user_management import UserManager
from datetime import datetime, timedelta
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
        chunks_list = metas if metas else []
        count = len(chunks_list)
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

        # DEBUG INFO - Added to diagnose file processing issues
        st.write(f"📊 File size: {len(text)} characters")
        st.write(f"🔍 First 500 characters:")
        st.text(text[:500])
        st.write(f"🏷️ XML detection: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}")

        chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
        st.write(f"📦 Chunks created: {len(chunks)}")
        
        # Only proceed if chunks were created
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
    c1, c2 = st.columns([6,1])
    with c2:
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
                    
                    # Create container for each user
                    with st.container():
                        # User info row
                        col_info, col_actions = st.columns([4, 1])
                        
                        with col_info:
                            # Status indicator
                            status = "🟢 Active" if active else "🔴 Inactive"
                            st.write(f"**{name}** ({status})")
                            
                            # Access code with copy functionality
                            st.write("**Access Code:**")
                            st.code(code, language="text")
                            
                            # Additional info
                            col_dates, col_email = st.columns(2)
                            with col_dates:
                                st.caption(f"Created: {created[:10]}")
                                st.caption(f"Last login: {last[:16] if last else 'Never'}")
                            with col_email:
                                if email:
                                    st.caption(f"Email: {email}")
                                st.caption(f"Expires: {exp[:10] if exp else 'No expiry'}")
                        
                        with col_actions:
                            if active:
                                if st.button("🚫 Revoke", key=f"r_{code}"):
                                    um.deactivate_user(code)
                                    st.success(f"Revoked access for {name}")
                                    st.rerun()
                            else:
                                st.write("*Revoked*")
                        
                        # Add divider between users
                        st.divider()
            else:
                st.info("No users created yet")

    with tab2:
        st.header("Knowledge Base")
        try:
            total = coll.count()
            m1,m2,m3 = st.columns(3)
            m1.metric("Total Chunks", total)
            m2.metric("Jurisdictions","NY,NJ,CT")
            m3.metric("Status","Active" if total>0 else "Empty")
        except:
            st.warning("DB init error")

        st.markdown("---")
        st.subheader("📄 Upload Documents")
        ups = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf','docx','txt'])
        if ups and st.button("Process"):
            for f in ups:
                st.write(f"Processing {f.name}")
                ok,msg = process_uploaded_file(f,chunker,coll)
                st.success(msg) if ok else st.error(msg)

        st.markdown("---")
        st.subheader("🗂️ Browse Files & Chunks")
        try:
            result = coll.get(include=['metadatas'])
            metas = result.get('metadatas', [])
            flat = metas if metas else []
            
            files = {}
            for md in flat:
                sf = md.get('source_file','Unknown')
                files.setdefault(sf,0)
                files[sf]+=1
            
            if files:
                st.write(f"📊 **Total Files:** {len(files)} | **Total Chunks:** {sum(files.values())}")
                
                # Create expandable sections for each file
                for fn, cnt in files.items():
                    with st.expander(f"📄 {fn} ({cnt} chunks)", expanded=False):
                        
                        # File-specific controls
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # File-specific chunk browsing
                            chunks_to_show = st.selectbox(
                                f"Chunks to display for {fn}:", 
                                [10, 25, 50, 100], 
                                index=1,  # Default to 25
                                key=f"chunks_{fn}"
                            )
                        
                        with col2:
                            if st.button(f"🔍 Browse Chunks", key=f"browse_{fn}"):
                                try:
                                    # Get chunks for this specific file
                                    file_result = coll.get(
                                        where={"source_file": fn},
                                        limit=chunks_to_show, 
                                        include=['documents', 'metadatas']
                                    )
                                    docs = file_result.get('documents', [])
                                    file_metas = file_result.get('metadatas', [])
                                    
                                    if docs and file_metas:
                                        st.success(f"📋 Displaying {len(docs)} chunks from {fn}")
                                        
                                        for i, (doc, metadata) in enumerate(zip(docs, file_metas)):
                                            source_file = metadata.get('source_file', 'Unknown File')
                                            section = metadata.get('section', 'Unknown Section')
                                            
                                            with st.expander(f"Chunk {i+1}: {section}", expanded=False):
                                                # Show chunk content
                                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                                                
                                                # Show metadata
                                                st.json({
                                                    "Section": section,
                                                    "Citation": metadata.get('citation', 'N/A'),
                                                    "Upload Date": metadata.get('upload_date', '')[:10] if metadata.get('upload_date') else 'N/A',
                                                    "Chunk ID": metadata.get('chunk_id', 'N/A'),
                                                    "Section Type": metadata.get('section_type', 'N/A'),
                                                    "Semantic Quality": metadata.get('semantic_density', 'N/A')
                                                })
                                    else:
                                        st.warning(f"No chunks found for {fn}")
                                        
                                except Exception as e:
                                    st.error(f"Error browsing chunks for {fn}: {e}")
                        
                        # File-specific search
                        st.markdown(f"### 🔍 Search within {fn}")
                        search_query = st.text_input(
                            f"Search {fn}:", 
                            placeholder=f"Enter keywords to search within {fn}...",
                            key=f"search_{fn}"
                        )
                        
                        if search_query and st.button(f"Search", key=f"search_btn_{fn}"):
                            try:
                                # Search only within this specific file
                                results = coll.query(
                                    query_texts=[search_query],
                                    where={"source_file": fn},  # Filter by file
                                    n_results=10,  # Reasonable number for file-specific search
                                    include=['documents', 'metadatas', 'distances']
                                )
                                
                                if results['documents'] and results['documents'][0]:
                                    st.success(f"🎯 Found {len(results['documents'][0])} matches in {fn}")
                                    
                                    for i, (doc, metadata, distance) in enumerate(zip(
                                        results['documents'][0], 
                                        results['metadatas'][0], 
                                        results.get('distances', [[0]*len(results['documents'][0])])[0]
                                    )):
                                        section = metadata.get('section', 'Unknown')
                                        
                                        with st.expander(f"Match {i+1}: {section} (Relevance: {distance:.3f})", expanded=False):
                                            st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                                            st.json({
                                                "Section": section,
                                                "Citation": metadata.get('citation', 'N/A'),
                                                "Relevance Score": f"{distance:.3f}",
                                                "Section Type": metadata.get('section_type', 'N/A')
                                            })
                                else:
                                    st.warning(f"No matches found for '{search_query}' in {fn}")
                                    
                            except Exception as e:
                                st.error(f"Error searching {fn}: {e}")
                        
                        # File management (delete)
                        st.markdown("---")
                        st.markdown("**File Management**")
                        col_del1, col_del2 = st.columns([1, 1])
                        
                        with col_del1:
                            if st.button(f"🗑️ Delete {fn}", key=f"delete_{fn}", type="secondary"):
                                st.session_state[f"del_{fn}"] = True
                        
                        with col_del2:
                            if st.session_state.get(f"del_{fn}"):
                                if st.button(f"✅ Confirm Delete", key=f"confirm_{fn}", type="primary"):
                                    ok, msg = delete_file_chunks(coll, fn)
                                    st.success(msg) if ok else st.error(msg)
                                    st.session_state[f"del_{fn}"] = False
                                    time.sleep(1)
                                    st.rerun()
                                if st.button(f"❌ Cancel", key=f"cancel_{fn}"):
                                    st.session_state[f"del_{fn}"] = False
                                    st.rerun()
            else:
                st.info("No files uploaded yet")
                
        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()
