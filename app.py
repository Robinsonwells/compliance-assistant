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
from user_management import UserManager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT
from typing import Dict, List, Optional

class GPT5Handler:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # GPT-5 Model Configuration
        self.models = {
            "gpt-5": "Full reasoning model for complex tasks",
            "gpt-5-mini": "Balanced performance and cost", 
            "gpt-5-nano": "Ultra-fast responses",
            "gpt-4o": "Fallback model (supports all parameters)"
        }
        
        # GPT-5 specific parameters
        self.reasoning_efforts = {
            "minimal": {"description": "Fastest responses, minimal thinking"},
            "low": {"description": "Light reasoning"},
            "medium": {"description": "Balanced performance (default)"},
            "high": {"description": "Maximum accuracy, deep thinking"}
        }
        
        self.verbosity_levels = {
            "low": "Concise responses",
            "medium": "Balanced detail (default)",
            "high": "Comprehensive explanations"
        }

    def get_available_models(self) -> List[str]:
        """Return list of available model names"""
        return list(self.models.keys())

    def get_model_description(self, model_name: str) -> str:
        """Get description for a specific model"""
        return self.models.get(model_name, "Unknown model")

    def is_gpt5_model(self, model_name: str) -> bool:
        """Check if model is GPT-5 series"""
        return model_name.startswith("gpt-5")

    def create_chat_completion(
        self,
        messages: List[Dict],
        model: str = "gpt-5",
        reasoning_effort: str = "medium",
        verbosity: str = "medium",
        max_tokens: int = 4000,
        temperature: float = 0.7
    ) -> Dict:
        """Create chat completion with proper parameter handling for GPT-5"""
        try:
            # Base parameters that work for all models
            request_params = {
                "model": model,
                "messages": messages
            }
            
            # Handle GPT-5 specific parameters
            if self.is_gpt5_model(model):
                # ✅ GPT-5 uses max_completion_tokens instead of max_tokens
                request_params["max_completion_tokens"] = max_tokens
                
                # ✅ GPT-5 uses reasoning_effort parameter
                request_params["reasoning_effort"] = reasoning_effort
                
                # ❌ GPT-5 does NOT support temperature - it's fixed at 1.0
                # ❌ Do not include temperature parameter for GPT-5
                
            else:
                # Legacy models (GPT-4o, etc.) use old parameters
                request_params["max_tokens"] = max_tokens
                request_params["temperature"] = temperature
                # Legacy models don't support reasoning_effort or verbosity
            
            # Make API request
            response = self.client.chat.completions.create(**request_params)
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model_used": response.model,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "reasoning_effort": reasoning_effort if self.is_gpt5_model(model) else "N/A",
                "verbosity": verbosity if self.is_gpt5_model(model) else "N/A",
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }

    def create_responses_api(
        self,
        input_text: str,
        model: str = "gpt-5",
        reasoning_effort: str = "medium",
        verbosity: str = "medium",
        max_tokens: int = 4000
    ) -> Dict:
        """Create response using the newer Responses API (recommended for GPT-5)"""
        try:
            # Responses API has different parameter names
            request_params = {
                "model": model,
                "input": [{"role": "user", "content": input_text}]
            }
            
            if self.is_gpt5_model(model):
                # ✅ Responses API uses max_output_tokens
                request_params["max_output_tokens"] = max_tokens
                
                # ✅ Reasoning and verbosity parameters
                request_params["reasoning"] = {"effort": reasoning_effort}
                request_params["verbosity"] = verbosity
                
                # ❌ Still no temperature support
            else:
                # Legacy model fallback
                request_params["max_output_tokens"] = max_tokens
            
            # Use Responses API
            response = self.client.responses.create(**request_params)
            
            # Extract content from Responses API format
            content = self._extract_responses_content(response)
            
            return {
                "success": True,
                "content": content,
                "model_used": response.model if hasattr(response, 'model') else model,
                "response_id": response.id if hasattr(response, 'id') else None,
                "reasoning_effort": reasoning_effort,
                "verbosity": verbosity
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }

    def _extract_responses_content(self, response) -> str:
        """Extract content from Responses API response"""
        try:
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text
            return "No content extracted"
        except Exception:
            return str(response)
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
    if 'gpt5_handler' not in st.session_state:
        st.session_state.gpt5_handler = gpt5_handler
    
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
            "Model:",
            options=gpt5_handler.get_available_models(),
            index=0,
            help="GPT-5 models have different parameter support"
        )
        
        # Show model info
        st.info(gpt5_handler.get_model_description(selected_model))
        
        # Show parameter compatibility
        if gpt5_handler.is_gpt5_model(selected_model):
            st.success("✅ GPT-5 Model - Uses new parameters")
            
            # GPT-5 specific controls
            reasoning_effort = st.selectbox(
                "Reasoning Effort:",
                options=list(gpt5_handler.reasoning_efforts.keys()),
                index=2,
                help="Controls how much the model 'thinks'"
            )
            
            verbosity = st.selectbox(
                "Verbosity:",
                options=list(gpt5_handler.verbosity_levels.keys()),
                index=1,
                help="Controls response length and detail"
            )
            
            # API Choice
            api_choice = st.radio(
                "API Type:",
                ["Responses API (Recommended)", "Chat Completions API"],
                help="Responses API is optimized for GPT-5"
            )
            
            st.warning("⚠️ GPT-5 does not support temperature (fixed at 1.0)")
            
        else:
            st.info("ℹ️ Legacy Model - Uses traditional parameters")
            reasoning_effort = "medium"
            verbosity = "medium"
            api_choice = "Chat Completions API"
            
            # Legacy model controls
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        
        # Common settings
        with st.expander("Advanced Settings"):
            max_tokens = st.slider("Max Tokens", 100, 8000, 4000)
        
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
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", msg["metadata"].get("model_used", "N/A"))
                    with col2:
                        st.metric("Reasoning Effort", msg["metadata"].get("reasoning_effort", "N/A"))
                    with col3:
                        st.metric("Verbosity", msg["metadata"].get("verbosity", "N/A"))
                    with col4:
                        st.metric("Tokens", msg["metadata"].get("total_tokens", 0))
                    
    
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
            
            # Set defaults for legacy models
            temp = temperature if not gpt5_handler.is_gpt5_model(selected_model) else 0.7
            
            # Prepare messages with system prompt and legal context
            messages = [
                {"role": "system", "content": f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\n\nAvailable Legal Context:\n{context}"}
            ]
            
            # Add conversation history
            for msg in st.session_state.messages:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Choose API based on model and user preference
            if api_choice == "Responses API (Recommended)" and gpt5_handler.is_gpt5_model(selected_model):
                # Use Responses API for GPT-5
                full_prompt = f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\n\nAvailable Legal Context:\n{context}\n\nUser Question: {prompt}"
                response_result = gpt5_handler.create_responses_api(
                    input_text=full_prompt,
                    model=selected_model,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    max_tokens=max_tokens
                )
            else:
                # Use Chat Completions API
                response_result = gpt5_handler.create_chat_completion(
                    messages=messages,
                    model=selected_model,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    max_tokens=max_tokens,
                    temperature=temp
                )
            
            bar.progress(100); prog.text("✅ Analysis complete!")
            
            if response_result["success"]:
                ai_response = response_result["content"]
                
                st.markdown(ai_response)
                
                # Add response with metadata to session state
                message_data = {
                    "role": "assistant",
                    "content": ai_response,
                    "metadata": {
                        "model_used": response_result.get("model_used", selected_model),
                        "reasoning_effort": response_result.get("reasoning_effort", reasoning_effort),
                        "verbosity": response_result.get("verbosity", verbosity),
                        "total_tokens": response_result.get("total_tokens", 0),
                        "finish_reason": response_result.get("finish_reason", "N/A")
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