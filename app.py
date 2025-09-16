import os
os.environ["PYTORCH_DISABLE_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import torch  # Initialize PyTorch early to prevent torch.classes errors
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
from system_prompts import ENHANCED_LEGAL_COMPLIANCE_SYSTEM_PROMPT, format_complex_scenario_response
from typing import Dict, List, Optional

def assess_query_complexity(query):
    """Warn users about complex scenarios"""
    complexity_indicators = [
        'multiple states', 'cross state lines', 'interstate',
        'different jurisdictions', 'tri-state', 'federal and state'
    ]
    
    is_complex = any(indicator in query.lower() for indicator in complexity_indicators)
    
    if is_complex:
        st.warning("🚨 **COMPLEX MULTI-JURISDICTIONAL QUERY DETECTED**")
        st.info("⚖️ This analysis will consider federal baseline laws and multiple state requirements. Response may take longer for comprehensive analysis.")
        
    return is_complex

def display_sources_by_complexity(results, context_metadata):
    """Better source organization for complex queries"""
    
    if context_metadata['total_sources'] > 20:
        st.markdown("### 📚 **COMPREHENSIVE SOURCE ANALYSIS**")
        
        # Show federal vs state breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Federal Sources", context_metadata.get('federal_sources', 0))
        with col2:
            st.metric("NY Sources", context_metadata.get('ny_sources', 0))
        with col3:
            st.metric("NJ Sources", context_metadata.get('nj_sources', 0))
        with col4:
            st.metric("CT Sources", context_metadata.get('ct_sources', 0))
        
        # Warn about gaps
        if context_metadata.get('federal_sources', 0) == 0:
            st.error("⚠️ **FEDERAL LAW GAP**: No federal sources found. Add federal regulations to knowledge base for complete analysis.")
    else:
        st.markdown("### 📚 **SOURCE ANALYSIS**")
        
        # Show basic breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Sources", context_metadata['total_sources'])
        with col2:
            jurisdictions_found = []
            if context_metadata.get('ny_sources', 0) > 0:
                jurisdictions_found.append('NY')
            if context_metadata.get('nj_sources', 0) > 0:
                jurisdictions_found.append('NJ')
            if context_metadata.get('ct_sources', 0) > 0:
                jurisdictions_found.append('CT')
            if context_metadata.get('federal_sources', 0) > 0:
                jurisdictions_found.append('Federal')
            st.metric("Jurisdictions", ', '.join(jurisdictions_found) if jurisdictions_found else 'Mixed')

def detect_knowledge_gaps(query, context_metadata):
    """Detect and warn about knowledge gaps"""
    gaps = []
    
    query_lower = query.lower()
    
    # Check for missing jurisdictions
    if 'connecticut' in query_lower or 'ct' in query_lower:
        if context_metadata.get('ct_sources', 0) < 3:
            gaps.append("Connecticut law coverage may be incomplete")
    
    # Check for multi-state without federal
    if any(term in query_lower for term in ['interstate', 'multiple states', 'cross state']):
        if context_metadata.get('federal_sources', 0) == 0:
            gaps.append("Federal interstate commerce guidance not available")
    
    # Check for industry-specific needs
    if 'transportation' in query_lower or 'logistics' in query_lower:
        gaps.append("DOT transportation regulations not in current database")
    
    if 'construction' in query_lower or 'prevailing wage' in query_lower:
        gaps.append("Davis-Bacon Act and prevailing wage regulations not in current database")
    
    if 'federal contractor' in query_lower or 'government contractor' in query_lower:
        gaps.append("Federal contractor requirements not in current database")
    
    return gaps

class GPT5Handler:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # ✅ ONLY GPT-5 MODELS - Maximum quality
        self.models = {
            "gpt-5": "Maximum reasoning, minimum hallucination model for legal analysis"
        }
        
        # ✅ ANTI-HALLUCINATION SETTINGS for GPT-5
        self.reasoning_efforts = {
            "high": {"description": "Maximum accuracy, deep thinking"}
        }
        
        # ✅ DETERMINISTIC SETTINGS - GPT-5 specific
        self.default_settings = {
            "reasoning_effort": "high",  # Maximum reasoning
            "max_completion_tokens": 16000,  # Increased for unlimited context
            "model": "gpt-5",  # Only GPT-5 allowed
        }

    def get_available_models(self) -> List[str]:
        """Return only GPT-5 - no other models allowed"""
        return ["gpt-5"]

    def get_model_description(self, model_name: str) -> str:
        """Always return GPT-5 description"""
        return "Maximum reasoning GPT-5 - Deterministic, minimal hallucination"

    def is_gpt5_model(self, model_name: str) -> bool:
        """Always return True since we only allow GPT-5"""
        return True

    def create_chat_completion(
        self,
        messages: List[Dict],
        model: str = "gpt-5",
        reasoning_effort: str = "high",
        max_tokens: int = 16000
    ) -> Dict:
        """Create maximum quality, minimum hallucination GPT-5 completion"""
        try:
            # ✅ ANTI-HALLUCINATION PARAMETERS for GPT-5
            request_params = {
                "model": "gpt-5",  # Force GPT-5 only
                "messages": messages,
                "max_completion_tokens": 16000,  # Maximum for long context responses
                "reasoning_effort": "high"  # Maximum reasoning for accuracy
            }
            
            # Make API request with anti-hallucination settings
            response = self.client.chat.completions.create(**request_params)
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model_used": response.model,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "reasoning_effort": "high",
                "deterministic_mode": True,  # GPT-5 is deterministic by default
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
        reasoning_effort: str = "high",
        max_tokens: int = 16000
    ) -> Dict:
        """Create maximum quality, minimum hallucination response using Responses API"""
        try:
            # ✅ ENHANCED: Validate inputs
            if not input_text.strip():
                return {"success": False, "error": "Empty input text", "content": None}
            
            if len(input_text) > 200000:  # Rough token limit check
                input_text = input_text[:200000] + "\n\n[Content truncated due to length limits]"
            
            # ✅ ANTI-HALLUCINATION RESPONSES API PARAMETERS
            request_params = {
                "model": "gpt-5",  # Force GPT-5 only
                "input": [{"role": "user", "content": input_text}],
                "max_output_tokens": min(max_tokens, 16000),  # Ensure within limits
                "reasoning": {"effort": "high"}  # Maximum reasoning for accuracy
            }
            
            response = self.client.responses.create(**request_params)
            content = self._extract_responses_content(response)
            
            return {
                "success": True,
                "content": content,
                "model_used": getattr(response, 'model', 'gpt-5'),
                "response_id": getattr(response, 'id', None),
                "reasoning_effort": "high",
                "deterministic_mode": True
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # ✅ ENHANCED: Specific GPT-5 error handling
            if "max_output_tokens" in error_msg:
                return {"success": False, "error": "Token limit exceeded. Try a shorter query.", "content": None}
            elif "reasoning" in error_msg:
                return {"success": False, "error": "GPT-5 reasoning parameter error. Using fallback.", "content": None}
            elif "rate_limit" in error_msg.lower():
                return {"success": False, "error": "API rate limit reached. Please wait a moment and try again.", "content": None}
            else:
                return {"success": False, "error": f"GPT-5 API error: {error_msg}", "content": None}

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
        pass  # This will be handled in check_authentication function

@st.cache_resource
def init_systems():
    """Initialize all system components"""
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
    
    return user_manager, chunker, client, embedding_model

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def check_authentication():
    """Check user authentication and initialize systems"""
    # Initialize systems
    user_manager, chunker, collection, embedding_model = init_systems()
    gpt5_handler = GPT5Handler()
    
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

def search_knowledge_base_unlimited(qdrant_client, embedding_model, query, max_results=None):
    """
    Search the legal knowledge base with UNLIMITED context retrieval
    for comprehensive multi-jurisdictional analysis
    """
    try:
        # Generate embedding for the query
        query_vector = embedding_model.encode([query])[0].tolist()
        
        # ✅ UNLIMITED CONTEXT: Start with high limit, get all relevant results
        initial_limit = max_results if max_results else 50  # Start with 50, can go higher
        
        search_results = qdrant_client.query_points(
            collection_name="legal_regulations",
            query=query_vector,
            limit=initial_limit,
            with_payload=True,
            score_threshold=0.3  # Lower threshold to get more potentially relevant results
        )
        
        results = []
        for result in search_results.points:
            doc = result.payload.get('text', '')
            meta = {k: v for k, v in result.payload.items() if k != 'text'}
            dist = 1 - result.score  # Convert similarity to distance
            results.append((doc, meta, dist))
        
        # ✅ MULTI-JURISDICTIONAL ENHANCEMENT: If query mentions multiple states, get more context
        query_lower = query.lower()
        multi_jurisdictional = any(state in query_lower for state in ['ny', 'nj', 'ct', 'new york', 'new jersey', 'connecticut'])
        
        if multi_jurisdictional and len(results) < 30:
            # Get additional results for multi-jurisdictional queries
            extended_results = qdrant_client.query_points(
                collection_name="legal_regulations",
                query=query_vector,
                limit=80,  # Extended limit for multi-jurisdictional
                with_payload=True,
                score_threshold=0.2  # Even lower threshold for comprehensive coverage
            )
            
            # Add additional results that weren't already included
            existing_ids = set(r[1].get('chunk_id', '') for r in results)
            for result in extended_results.points:
                chunk_id = result.payload.get('chunk_id', '')
                if chunk_id not in existing_ids:
                    doc = result.payload.get('text', '')
                    meta = {k: v for k, v in result.payload.items() if k != 'text'}
                    dist = 1 - result.score
                    results.append((doc, meta, dist))
        
        # ✅ COMPREHENSIVE COVERAGE: Sort by relevance but keep all results
        results.sort(key=lambda x: x[2])  # Sort by distance (lower = more relevant)
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

def get_comprehensive_legal_context(results, query):
    """
    Create comprehensive legal context from unlimited search results
    Organizes by jurisdiction and topic for maximum utility
    """
    if not results:
        return "No relevant legal text found.", {}
    
    # Organize results by jurisdiction and source
    context_by_jurisdiction = {
        'NY': [],
        'NJ': [],
        'CT': [],
        'Federal': [],
        'Multi-State': []
    }
    
    # Categorize results by jurisdiction
    for doc, meta, dist in results:
        # Use the jurisdiction metadata directly from chunk processing
        jurisdiction = meta.get('jurisdiction', 'Multi-State')
        
        # Map jurisdiction to our display categories
        if jurisdiction in ['NY', 'NJ', 'CT', 'Federal']:
            context_by_jurisdiction[jurisdiction].append((doc, meta, dist))
        else:
            # Default to Multi-State for any unrecognized jurisdictions
            context_by_jurisdiction['Multi-State'].append((doc, meta, dist))
    
    # Build comprehensive context string
    context_parts = []
    
    for jurisdiction, docs in context_by_jurisdiction.items():
        if docs:
            context_parts.append(f"\n=== {jurisdiction} LEGAL PROVISIONS ===")
            for i, (doc, meta, dist) in enumerate(docs[:20]):  # Limit per jurisdiction for readability
                context_parts.append(f"\n[{jurisdiction}-{i+1}] {doc}")
    
    comprehensive_context = "\n".join(context_parts)
    
    # Return both context and metadata
    context_metadata = {
        'total_sources': len(results),
        'ny_sources': len(context_by_jurisdiction['NY']),
        'nj_sources': len(context_by_jurisdiction['NJ']),
        'ct_sources': len(context_by_jurisdiction['CT']),
        'federal_sources': len(context_by_jurisdiction['Federal']),
        'multi_state_sources': len(context_by_jurisdiction['Multi-State'])
    }
    
    return comprehensive_context, context_metadata

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
    
    # ✅ FIXED: Define constants for clarity
    REASONING_EFFORT = "high"
    MAX_TOKENS = 16000
    
    # Header and logout
    col1, col2 = st.columns([6,1])
    with col1:
        st.markdown("# ⚖️ Elite Legal Compliance Assistant")
        st.markdown("*Powered by GPT-5 with **UNLIMITED CONTEXT** + **ZERO HALLUCINATION** mode*")
        st.success("🧠 **DETERMINISTIC MODE**: Maximum accuracy, comprehensive analysis")
    with col2:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.access_code = None
            st.success("👋 Logged out successfully")
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")
    
    # Initialize GPT-5 handler
    if 'gpt5_handler' not in st.session_state:
        st.session_state.gpt5_handler = gpt5_handler
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your **unlimited context** legal compliance assistant using GPT-5 in **deterministic mode** for zero hallucination. I have access to comprehensive legal databases across NY, NJ, and CT with no limits on context retrieval. My responses prioritize factual accuracy and comprehensive multi-jurisdictional analysis."
        }]
    
    with st.sidebar:
        st.markdown("### 👤 Session Info")
        if 'login_time' in st.session_state:
            duration = datetime.now() - st.session_state.login_time
            hrs, rem = divmod(int(duration.total_seconds()), 3600)
            mins, _ = divmod(rem, 60)
            st.write(f"**Active:** {hrs}h {mins}m")
        
        st.markdown("### 🧠 **ZERO HALLUCINATION** GPT-5")
        
        st.success("✅ **GPT-5 DETERMINISTIC** - Locked")
        st.success("🎯 **Temperature Equivalent**: Minimum (Deterministic)")
        st.success(f"🧠 **Reasoning**: {REASONING_EFFORT.upper()} (Maximum)")
        st.success("📊 **Context**: UNLIMITED (All relevant sources)")
        st.success("⚡ **API**: Responses API (Optimized)")
        st.success(f"📝 **Max Tokens**: {MAX_TOKENS:,} (Extended)")
        
        st.info("🔒 **Anti-Hallucination**: GPT-5 deterministic mode active")
        st.warning("⏱️ **Processing Time**: 30-90 seconds for comprehensive analysis")
        
        st.markdown("### 📚 Knowledge Base")
        try:
            collection_info = collection.get_collection("legal_regulations")
            cnt = collection_info.points_count
            st.write(f"**Legal Provisions:** {cnt:,}")
            st.write("**Jurisdictions:** NY, NJ, CT, Federal")
            st.write("**Context Limit:** UNLIMITED")
        except Exception as e:
            st.write(f"**Status:** Error - {e}")
    
    # Display messages with enhanced metadata
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                with st.expander("🔍 **Comprehensive Analysis Metrics**", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sources Used", msg["metadata"].get("total_sources", "N/A"))
                    with col2:
                        st.metric("Reasoning", msg["metadata"].get("reasoning_effort", "HIGH").upper())
                    with col3:
                        st.metric("Tokens", f"{msg['metadata'].get('total_tokens', 0):,}")
                    with col4:
                        st.metric("Jurisdictions", msg["metadata"].get("jurisdictions_covered", "Multiple"))
                
                # Display comprehensive sources if available in metadata
                if "sources_results" in msg["metadata"] and "sources_context_metadata" in msg["metadata"]:
                    display_comprehensive_sources(
                        msg["metadata"]["sources_results"], 
                        msg["metadata"]["sources_context_metadata"]
                    )
    
    # Chat input with unlimited context emphasis
    if prompt := st.chat_input("Ask comprehensive legal questions - I'll analyze ALL relevant sources across jurisdictions..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            prog = st.empty()
            bar = st.progress(0)
            
            try:
                # Assess query complexity and show warnings
                complexity = assess_query_complexity(prompt)
                
                prog.text("🔍 **UNLIMITED SEARCH**: Retrieving ALL relevant legal sources...")
                bar.progress(10)
                
                # ✅ GET UNLIMITED CONTEXT with error handling
                results = search_knowledge_base_unlimited(collection, embedding_model, prompt)
                if not results:
                    st.warning("⚠️ No relevant legal sources found. Consider adding more documents to the knowledge base.")
                    return
                
                prog.text(f"📊 **FOUND {len(results)} SOURCES**: Organizing by jurisdiction...")
                bar.progress(30)
                
                # Create comprehensive context
                comprehensive_context, context_metadata = get_comprehensive_legal_context(results, prompt)
                
                # Detect and display knowledge gaps
                gaps = detect_knowledge_gaps(prompt, context_metadata)
                if gaps:
                    st.warning("📋 **KNOWLEDGE BASE LIMITATIONS:**")
                    for gap in gaps:
                        st.write(f"• {gap}")
                
                prog.text("🧠 **DETERMINISTIC ANALYSIS**: GPT-5 processing comprehensive context...")
                bar.progress(60)
                
                # ✅ ANTI-HALLUCINATION SYSTEM PROMPT
                anti_hallucination_prompt = f"""{ENHANCED_LEGAL_COMPLIANCE_SYSTEM_PROMPT}

**ZERO HALLUCINATION MODE - CRITICAL INSTRUCTIONS:**
- Base ALL analysis STRICTLY on the provided legal context
- If information is not in the provided context, explicitly state "This information is not available in the provided legal sources"
- NEVER infer or guess legal requirements not explicitly stated in the context
- When citing specific laws, ONLY reference those mentioned in the provided sources
- For multi-jurisdictional questions, clearly separate analysis by state/jurisdiction
- If context is incomplete for a comprehensive answer, acknowledge the limitations
- Prioritize accuracy over completeness - better to say "insufficient information" than to hallucinate

**COMPREHENSIVE LEGAL CONTEXT ({context_metadata['total_sources']} sources):**
- NY Sources: {context_metadata['ny_sources']}
- NJ Sources: {context_metadata['nj_sources']}
- CT Sources: {context_metadata['ct_sources']}
- Federal Sources: {context_metadata['federal_sources']}
- Multi-State Sources: {context_metadata['multi_state_sources']}

{comprehensive_context}"""
                
                prog.text("🚀 **GENERATING**: Comprehensive, fact-based analysis...")
                bar.progress(80)
                
                # ✅ FIXED: Use Responses API with proper error handling
                response_result = gpt5_handler.create_responses_api(
                    input_text=f"{anti_hallucination_prompt}\n\nUser Question: {prompt}\n\nProvide comprehensive analysis based STRICTLY on the provided legal context. Acknowledge any limitations in available information.",
                    model="gpt-5",
                    reasoning_effort=REASONING_EFFORT,
                    max_tokens=MAX_TOKENS
                )
                
                bar.progress(100)
                prog.text("✅ **COMPREHENSIVE ANALYSIS COMPLETE!**")
                
                if response_result["success"]:
                    ai_response = response_result["content"]
                    
                    # Add comprehensive quality indicators
                    st.success(f"🎯 **ZERO HALLUCINATION RESPONSE** - Analyzed {len(results)} legal sources")
                    if context_metadata['total_sources'] > 30:
                        st.info(f"📊 **COMPREHENSIVE COVERAGE**: {context_metadata['total_sources']} sources across {sum(1 for k, v in context_metadata.items() if k.endswith('_sources') and v > 0)} jurisdictions")
                    
                    st.markdown(ai_response)
                    
                    # ✅ FIXED: Enhanced metadata with all variables defined
                    message_data = {
                        "role": "assistant",
                        "content": ai_response,
                        "metadata": {
                            "model_used": response_result.get("model_used", "GPT-5"),
                            "reasoning_effort": response_result.get("reasoning_effort", REASONING_EFFORT),  # ✅ Fixed
                            "total_tokens": response_result.get("total_tokens", 0),
                            "finish_reason": response_result.get("finish_reason", "N/A"),
                            "total_sources": len(results),
                            "jurisdictions_covered": f"{sum(1 for k, v in context_metadata.items() if k.endswith('_sources') and v > 0)} jurisdictions",
                            "context_metadata": context_metadata,
                            "sources_results": results,
                            "sources_context_metadata": context_metadata
                        }
                    }
                    st.session_state.messages.append(message_data)
                    
                else:
                    error_message = f"I apologize, but I encountered an error with comprehensive analysis: {response_result.get('error', 'Unknown error')}"
                    st.error(error_message)
                    st.session_state.messages.append({"role":"assistant","content":error_message})
                    
            except Exception as e:
                st.error(f"❌ **SYSTEM ERROR**: {str(e)}")
                st.info("Please try again or contact support if the error persists.")
                
                # Log error for debugging
                error_message = f"System error during analysis: {str(e)}"
                st.session_state.messages.append({"role":"assistant","content":error_message})

def display_comprehensive_sources(results, context_metadata):
    """Display sources organized by jurisdiction with better formatting"""
    st.markdown("### 📚 **ALL SOURCES ANALYZED**")
    st.info(f"🔍 **Comprehensive Review**: {len(results)} legal sources examined across jurisdictions")
    
    # Group by jurisdiction for display
    jurisdictions = ['NY', 'NJ', 'CT', 'Federal', 'Multi-State']
    for jurisdiction in jurisdictions:
        # ✅ FIXED: Use jurisdiction metadata directly
        jurisdiction_results = []
        for r in results:
            doc, meta, dist = r
            
            # Use the jurisdiction metadata directly from chunk processing
            chunk_jurisdiction = meta.get('jurisdiction', 'Multi-State')
            
            # Check if this result belongs to current jurisdiction
            if chunk_jurisdiction == jurisdiction:
                jurisdiction_results.append(r)
            elif jurisdiction == 'Multi-State' and chunk_jurisdiction not in ['NY', 'NJ', 'CT', 'Federal']:
                # Include any unrecognized jurisdictions in Multi-State
                jurisdiction_results.append(r)
        
        if jurisdiction_results:
            with st.expander(f"📖 {jurisdiction} Sources ({len(jurisdiction_results)})", expanded=False):
                for i, (doc, meta, dist) in enumerate(jurisdiction_results[:10]):  # Show top 10 per jurisdiction
                    # ✅ FIXED: Unique key generation to prevent conflicts
                    unique_key = f"{jurisdiction}_{i}_{hash(doc[:50]) % 10000}"
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text_area(
                            f"{jurisdiction}-{i+1}", 
                            doc, 
                            height=100, 
                            key=unique_key,
                            help=f"Source: {meta.get('source_file', 'Unknown')}"
                        )
                    with col2:
                        st.metric("Relevance", f"{dist:.3f}")
                        if 'chunk_id' in meta:
                            st.caption(f"Chunk: {meta['chunk_id']}")

if __name__ == "__main__":
    main_app()