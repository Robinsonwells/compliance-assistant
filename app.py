import streamlit as st
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import uuid
import time
from datetime import datetime, timedelta
from user_management import UserManager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT
from typing import Dict, List, Optional
import json

# Load environment variables
load_dotenv()

class GPT5Handler:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # ✅ CORRECT - Keep this as a dictionary, don't overwrite it
        self.models = {
            "gpt-5": "Full reasoning model for complex tasks",
            "gpt-5-mini": "Balanced performance and cost", 
            "gpt-5-nano": "Ultra-fast responses",
            "gpt-5-chat": "Advanced conversational model"
        }
        
        # Reasoning effort levels
        self.reasoning_efforts = {
            "minimal": {"description": "Fastest responses", "cost": "lowest", "speed": "very fast", "use_case": "Simple definitions"},
            "low": {"description": "Light reasoning", "cost": "low", "speed": "fast", "use_case": "Basic compliance questions"},
            "medium": {"description": "Balanced performance", "cost": "medium", "speed": "moderate", "use_case": "Standard legal analysis"},
            "high": {"description": "Maximum accuracy", "cost": "highest", "speed": "slow", "use_case": "Complex multi-jurisdictional issues"}
        }

    def create_legal_response(
        self,
        system_prompt: str,
        user_query: str,
        legal_context: str,
        model: str = "gpt-5",
        reasoning_effort: str = "high",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        conversation_id: Optional[str] = None
    ) -> Dict:
        """Create GPT-5 legal response using Responses API with fallback to Chat Completions"""
        
        # Prepare the full prompt with legal context
        full_prompt = f"""{system_prompt}

Available Legal Context:
{legal_context}

User Question: {user_query}"""
        
        # Try Responses API first (GPT-5 preferred method)
        try:
            request_params = {
                "model": model,
                "input": [{"role": "user", "content": full_prompt}],
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add conversation continuity if available
            if conversation_id:
                request_params["previous_response_id"] = conversation_id
            
            # Make Responses API request
            response = self.client.responses.create(**request_params)
            
            return {
                "success": True,
                "content": self._extract_content(response),
                "response_id": getattr(response, 'id', None),
                "model_used": getattr(response, 'model', model),
                "reasoning_tokens": self._get_reasoning_tokens(response),
                "total_tokens": getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                "reasoning_effort": reasoning_effort,
                "fallback_used": False
            }
            
        except Exception as responses_error:
            # Fallback to Chat Completions API
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Legal Context:\n{legal_context}\n\nQuestion: {user_query}"}
                ]
                
                request_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Add reasoning parameter for GPT-5 models
                if model.startswith("gpt-5"):
                    request_params["reasoning_effort"] = reasoning_effort
                
                response = self.client.chat.completions.create(**request_params)
                
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model_used": response.model,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "reasoning_tokens": 0,
                    "reasoning_effort": reasoning_effort,
                    "fallback_used": True
                }
                
            except Exception as chat_error:
                return {
                    "success": False,
                    "error": f"Both APIs failed. Responses API: {str(responses_error)}, Chat API: {str(chat_error)}",
                    "content": None
                }

    def _extract_content(self, response) -> str:
        """Extract text content from Responses API response"""
        try:
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text
            return "No content extracted from response"
        except Exception:
            return str(response)

    def _get_reasoning_tokens(self, response) -> int:
        """Extract reasoning token count"""
        try:
            if (hasattr(response, 'usage') and 
                hasattr(response.usage, 'output_tokens_details') and
                hasattr(response.usage.output_tokens_details, 'reasoning_tokens')):
                return response.usage.output_tokens_details.reasoning_tokens
            return 0
        except Exception:
            return 0

@st.cache_resource
def init_systems():
    gpt5_handler = GPT5Handler()
    user_manager = UserManager()
    chunker = LegalSemanticChunker(os.getenv("OPENAI_API_KEY"))
    
    # Initialize local embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
    
    vector_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    collection_name = "legal_regulations"
    
    # Create collection if it doesn't exist
    try:
        vector_client.get_collection(collection_name)
    except Exception:
        vector_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        # Create payload index for source_file field
        vector_client.create_payload_index(
            collection_name=collection_name,
            field_name="source_file",
            field_schema="keyword"
        )
    
    collection = vector_client
    return gpt5_handler, user_manager, chunker, collection, embedding_model

# Page config
st.set_page_config(
    page_title="Legal Compliance Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Hide Streamlit branding
st.markdown("""
<style>
    .stAppDeployButton {display:none;}
    .stDecoration {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def check_authentication():
    """Check if user is authenticated"""
    gpt5_handler, user_manager, chunker, collection, embedding_model = init_systems()
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        session_id = get_session_id()
        if user_manager.is_session_valid(session_id, hours_timeout=24):
            user_manager.update_session_activity(session_id)
            return True, gpt5_handler, user_manager, chunker, collection, embedding_model
        else:
            st.session_state.authenticated = False
            st.error("🕐 Your session has expired. Please log in again.")
            time.sleep(2)
            st.rerun()
    st.markdown("# 🔐 Legal Compliance Assistant")
    st.markdown("**Professional AI-Powered Legal Analysis**")
    st.markdown("---")
    with st.form("login_form"):
        st.markdown("### Enter Your Access Code")
        access_code = st.text_input(
            "Access Code",
            type="password",
            placeholder="Enter your access code...",
            help="Contact your administrator for access"
        )
        submit_button = st.form_submit_button("🚀 Access Assistant")
        if submit_button and access_code:
            if user_manager.validate_access_code(access_code):
                session_id = get_session_id()
                user_manager.create_session(access_code, session_id)
                user_manager.update_last_login(access_code)
                st.session_state.authenticated = True
                st.session_state.access_code = access_code
                st.session_state.login_time = datetime.now()
                st.success("✅ Access granted! Loading assistant...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid or expired access code")
                time.sleep(2)
    with st.expander("ℹ️ Session Information"):
        st.write("• Sessions expire after 24 hours of inactivity")
        st.write("• Your session renews with each interaction")
        st.write("• Access can be revoked by administrator")
    return False, None, None, None, None, None

def search_knowledge_base(qdrant_client, embedding_model, query, n_results=5):
    """Search the legal knowledge base"""
    try:
        # Generate embedding for the query using local model
        query_vector = embedding_model.encode([query])[0].tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_vector,
            limit=n_results,
            with_payload=True
        )
        
        # Convert results to match original format
        results = []
        for result in search_results:
            doc = result.payload.get('text', '')
            meta = {k: v for k, v in result.payload.items() if k != 'text'}
            dist = 1 - result.score  # Convert similarity to distance
            results.append((doc, meta, dist))
        
        return results
    except Exception:
        return []

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
        # Debug info
        st.write(f"📊 File size: {len(text)} characters")
        st.write("🔍 First 500 characters:")
        st.text(text[:500])
        st.write(f"🏷️ XML detection: {'XML' if text.strip().startswith('<?xml') or '<code type=' in text else 'Plain text'}")
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

def main_app():
    authenticated, gpt5_handler, user_manager, chunker, collection, embedding_model = check_authentication()
    if not authenticated:
        st.stop()
    
    # Header and logout
    col1, col2 = st.columns([6,1])
    with col1:
        st.markdown("# ⚖️ Elite Legal Compliance Assistant")
        st.markdown("*Powered by GPT-5 with maximum quality analysis*")
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.access_code = None
            st.success("👋 Logged out successfully")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    # Initialize conversation tracking for GPT-5
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your legal compliance assistant. I specialize in NY, NJ, and CT employment law. Ask me a question!"
        }]
    
    with st.sidebar:
        st.markdown("### 👤 Session Info")
        if 'login_time' in st.session_state:
            duration = datetime.now() - st.session_state.login_time
            hrs, rem = divmod(int(duration.total_seconds()), 3600)
            mins, _ = divmod(rem, 60)
            st.write(f"**Active:** {hrs}h {mins}m")
            left = timedelta(hours=24) - duration
            if left.total_seconds()>0:
                lh, lr = divmod(int(left.total_seconds()),3600)
                lm,_=divmod(lr,60)
                st.write(f"**Auto-logout:** {lh}h {lm}m")
        
        st.markdown("### 🧠 GPT-5 Configuration")
        
        # Model Selection
        selected_model = st.selectbox(
            "GPT-5 Model:",
            options=list(gpt5_handler.models.keys()),
            index=0,  # Default to full gpt-5
            help="Choose based on complexity needs"
        )
        
        # Reasoning Effort
        reasoning_effort = st.selectbox(
            "Reasoning Effort:",
            options=list(gpt5_handler.reasoning_efforts.keys()),
            index=3,  # Default to "high" for legal work
            help="Higher effort = better accuracy but slower/more expensive"
        )
        
        # Show current settings
        effort_info = gpt5_handler.reasoning_efforts[reasoning_effort]
        st.write(f"**Speed**: {effort_info['speed'].title()}")
        st.write(f"**Cost**: {effort_info['cost'].title()}")
        st.write(f"**Best for**: {effort_info['use_case']}")
        
        st.markdown("### 📚 Knowledge Base")
        try:
            collection_info = collection.get_collection("legal_regulations")
            cnt = collection_info.points_count
            st.write(f"**Legal Provisions:** {cnt}")
            st.write("**Jurisdictions:** NY, NJ, CT")
        except:
            st.write("**Status:** Initializing...")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show GPT-5 metadata for assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                with st.expander("🔍 Response Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", msg["metadata"].get("model_used", "N/A"))
                    with col2:
                        st.metric("Reasoning Effort", msg["metadata"].get("reasoning_effort", "N/A"))
                    with col3:
                        st.metric("Total Tokens", msg["metadata"].get("total_tokens", 0))
                    
                    if msg["metadata"].get("reasoning_tokens", 0) > 0:
                        st.metric("Reasoning Tokens", msg["metadata"]["reasoning_tokens"])
                    
                    if msg["metadata"].get("fallback_used"):
                        st.warning("⚠️ Used Chat Completions API fallback")
    
    if prompt := st.chat_input("Ask me about legal compliance requirements..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            prog = st.empty(); bar = st.progress(0)
            prog.text("🔍 Searching legal knowledge base..."); bar.progress(20)
            results = search_knowledge_base(collection, embedding_model, prompt, n_results=8)
            prog.text(f"🧠 GPT-5 applying {reasoning_effort} reasoning..."); bar.progress(50)
            context = "\n\n".join(f"Legal Text: {doc}" for doc,_,_ in results) or "No relevant legal text found."
            
            prog.text("⚖️ Generating structured legal response..."); bar.progress(75)
            
            # Use GPT-5 Handler with Responses API
            response_result = gpt5_handler.create_legal_response(
                system_prompt=LEGAL_COMPLIANCE_SYSTEM_PROMPT,
                user_query=prompt,
                legal_context=context,
                model=selected_model,
                reasoning_effort=reasoning_effort,
                max_tokens=2000,
                temperature=0.1,
                conversation_id=st.session_state.conversation_id
            )
            
            bar.progress(100); prog.text("✅ Analysis complete!")
            
            if response_result["success"]:
                ai_response = response_result["content"]
                
                # Update conversation ID for stateful conversations
                if "response_id" in response_result:
                    st.session_state.conversation_id = response_result["response_id"]
                
                st.markdown(ai_response)
                
                # Add response with metadata to session state
                message_data = {
                    "role": "assistant",
                    "content": ai_response,
                    "metadata": {
                        "model_used": response_result.get("model_used", selected_model),
                        "reasoning_effort": response_result.get("reasoning_effort", reasoning_effort),
                        "total_tokens": response_result.get("total_tokens", 0),
                        "reasoning_tokens": response_result.get("reasoning_tokens", 0),
                        "fallback_used": response_result.get("fallback_used", False)
                    }
                }
                st.session_state.messages.append(message_data)
            else:
                error_message = f"I apologize, but I encountered an error: {response_result.get('error', 'Unknown error')}"
                st.error(error_message)
                st.session_state.messages.append({"role":"assistant","content":error_message})
            
            if results:
                st.markdown("### 📚 Sources Consulted")
                for doc, meta, dist in results:
                    label = f"{meta.get('source_file','Unknown')} – Chunk {meta.get('chunk_id','N/A')}"
                    with st.expander(label, expanded=False):
                        st.code(doc, language="text")
                        st.write(meta)
                        st.write(f"Relevance: {dist:.3f}")

if __name__ == "__main__":
    main_app()