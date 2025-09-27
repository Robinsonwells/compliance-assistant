import torch  # Initialize PyTorch early to prevent torch.classes errors
import streamlit as st
import os
from dotenv import load_dotenv
from user_management import UserManager
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from system_prompts import ENHANCED_LEGAL_COMPLIANCE_SYSTEM_PROMPT
import uuid
from datetime import datetime
import json
import re
from typing import List, Dict, Any, Tuple
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Compliance Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("styles/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css()
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Initialize systems
@st.cache_resource
def init_systems():
    """Initialize all required systems"""
    try:
        # User management
        user_manager = UserManager()
        
        # Embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
        
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        openai_client = OpenAI(api_key=openai_api_key)
        
        return user_manager, embedding_model, qdrant_client, openai_client
        
    except Exception as e:
        st.error(f"System initialization error: {e}")
        st.stop()

def store_systems_in_session():
    """Store initialized systems in session state for callback access"""
    if 'systems_initialized' not in st.session_state:
        user_manager, embedding_model, qdrant_client, openai_client = init_systems()
        st.session_state.user_manager = user_manager
        st.session_state.embedding_model = embedding_model
        st.session_state.qdrant_client = qdrant_client
        st.session_state.openai_client = openai_client
        st.session_state.systems_initialized = True

def _init_session_state_systems():
    """Initialize systems in session state if not already done"""
    if 'systems_initialized' not in st.session_state:
        user_manager, embedding_model, qdrant_client, openai_client = init_systems()
        st.session_state.user_manager = user_manager
        st.session_state.embedding_model = embedding_model
        st.session_state.qdrant_client = qdrant_client
        st.session_state.openai_client = openai_client
        st.session_state.systems_initialized = True

def _submit_question_callback():
    """Callback function to handle question submission"""
    if st.session_state.user_legal_query and st.session_state.user_legal_query.strip():
        # Store the question to process
        st.session_state.query_to_process = st.session_state.user_legal_query.strip()
        # Clear the input immediately
        st.session_state.user_legal_query = ""

def authenticate_user():
    """Handle user authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if not st.session_state.authenticated:
        st.markdown("""
            <div class="login-card">
                <div class="brand-logo">⚖️</div>
                <h1 style="margin: 0; text-align: center;">AI Compliance Assistant</h1>
                <p style="text-align: center; color: var(--text-muted); margin-bottom: 2rem;">
                    Professional Employment Organization Legal Research Tool
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            access_code = st.text_input(
                "Access Code",
                placeholder="Enter your access code",
                help="Contact your administrator for an access code"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button("🔐 Access System", use_container_width=True)
            
            if submit_button and access_code:
                _init_session_state_systems()
                user_manager = st.session_state.user_manager
                
                if user_manager.validate_access_code(access_code.strip().upper()):
                    # Create session
                    if user_manager.create_session(access_code.strip().upper(), st.session_state.session_id):
                        st.session_state.authenticated = True
                        st.session_state.access_code = access_code.strip().upper()
                        user_manager.update_last_login(access_code.strip().upper())
                        st.rerun()
                    else:
                        st.error("Session creation failed. Please try again.")
                else:
                    st.error("Invalid or expired access code")
            elif submit_button:
                st.error("Please enter an access code")
        
        return False
    
    return True

def adaptive_relevance_search(query: str, qdrant_client, embedding_model, openai_client):
    """Progressive search that stops when sufficient relevant content is found"""
    
    print("Starting adaptive relevance search...")
    start_time = time.time()
    
    # Progressive search with expanding thresholds
    search_phases = [
        {"limit": 50, "threshold": 0.8, "phase": "high_precision"},
        {"limit": 150, "threshold": 0.6, "phase": "medium_precision"}, 
        {"limit": 400, "threshold": 0.4, "phase": "broad_coverage"},
        {"limit": 800, "threshold": 0.25, "phase": "comprehensive_sweep"}
    ]
    
    all_candidates = []
    query_embedding = embedding_model.encode([query])[0].tolist()
    
    for phase in search_phases:
        print(f"Phase: {phase['phase']} (limit: {phase['limit']}, threshold: {phase['threshold']})")
        
        # Search this phase
        phase_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_embedding,
            limit=phase['limit'],
            score_threshold=phase['threshold'],
            with_payload=True,
            with_vectors=False
        )
        
        # Add new results (avoid duplicates)
        existing_ids = {getattr(c, 'id', hash(str(c))) for c in all_candidates}
        new_results = [r for r in phase_results if getattr(r, 'id', hash(str(r))) not in existing_ids]
        all_candidates.extend(new_results)
        
        print(f"Found {len(new_results)} new chunks (total: {len(all_candidates)})")
        
        # AI decides if we have enough relevant content
        if len(all_candidates) >= 30:  # Minimum threshold before asking AI
            sufficiency_check = assess_content_sufficiency(query, all_candidates, openai_client)
            
            if sufficiency_check['sufficient']:
                print(f"AI determined sufficient content found: {sufficiency_check['reason']}")
                break
            else:
                print(f"AI wants more content: {sufficiency_check['reason']}")
    
    print(f"Discovery phase complete: {len(all_candidates)} candidates in {time.time() - start_time:.1f}s")
    return all_candidates

def assess_content_sufficiency(query: str, candidates: List, openai_client) -> Dict:
    """AI determines if we have sufficient relevant content"""
    try:
        # Create overview of current candidates
        coverage_summary = create_coverage_summary(candidates)
        
        sufficiency_prompt = f"""Legal Query: "{query}"

Current Search Results Summary:
- Total Sources: {len(candidates)}
- Jurisdictions: {', '.join(coverage_summary['jurisdictions'])}
- Topics Covered: {', '.join(coverage_summary['topics'])}
- Relevance Range: {coverage_summary['score_range']}

Sample High-Relevance Sources:
{coverage_summary['top_sources']}

ASSESSMENT TASK:
Do we have sufficient legal sources to comprehensively answer this query?

Consider:
1. Does this cover the main legal requirements?
2. Are key jurisdictions represented?
3. Are we missing critical legal aspects?

Respond in JSON format:
{{
    "sufficient": true/false,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation of why sufficient/insufficient",
    "missing_aspects": ["list", "of", "gaps"] or [],
    "continue_search": true/false
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": sufficiency_prompt}],
            max_completion_tokens=200,
            temperature=0.1
        )
        
        assessment = json.loads(response.choices[0].message.content)
        return assessment
        
    except Exception as e:
        print(f"Sufficiency assessment failed: {e}")
        # Conservative fallback: continue searching unless we have a lot
        return {
            "sufficient": len(candidates) > 200,
            "confidence": 0.5,
            "reason": f"Assessment failed, fallback decision based on count: {len(candidates)}",
            "missing_aspects": [],
            "continue_search": len(candidates) <= 200
        }

def create_coverage_summary(candidates: List) -> Dict:
    """Create a summary of what we've found so far"""
    jurisdictions = set()
    topics = set()
    scores = []
    top_sources = []
    
    for candidate in candidates[:50]:  # Analyze top 50 for speed
        payload = candidate.payload
        jurisdictions.add(payload.get('jurisdiction', 'Unknown'))
        
        # Extract topics from section titles and text
        section_title = payload.get('section_title', '').lower()
        text_preview = payload.get('text', '')[:200].lower()
        
        # Common legal topics
        legal_topics = ['wage', 'overtime', 'break', 'leave', 'safety', 'discrimination', 'harassment']
        for topic in legal_topics:
            if topic in section_title or topic in text_preview:
                topics.add(topic)
        
        scores.append(candidate.score)
        
        if len(top_sources) < 5:
            top_sources.append(f"- {payload.get('citation', 'N/A')}: {payload.get('text', '')[:150]}...")
    
    return {
        'jurisdictions': list(jurisdictions),
        'topics': list(topics),
        'score_range': f"{min(scores):.3f} - {max(scores):.3f}" if scores else "N/A",
        'top_sources': '\n'.join(top_sources)
    }

def ai_driven_relevance_filter(query: str, candidates: List, openai_client) -> List:
    """AI determines exact set of relevant chunks needed"""
    try:
        print(f"AI-driven filtering of {len(candidates)} candidates...")
        
        # Process in batches for accuracy while maintaining flexibility
        relevant_chunks = []
        batch_size = 30  # Smaller batches for better AI accuracy
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            
            # FIX #1: Proper indentation for the relevance_prompt
            relevance_prompt = f"""Legal Query: "{query}"

TASK: Determine which sources are relevant for answering this query. Be inclusive - if a source might be helpful, include it.

RELEVANCE CRITERIA:
- ESSENTIAL: Directly answers the query or provides critical legal requirements
- USEFUL: Provides any context, related requirements, or background information that could be helpful
- SKIP: Only if completely unrelated to the legal topic

For each source, respond with: ESSENTIAL, USEFUL, or SKIP

Sources to evaluate:
"""
            
            for j, chunk in enumerate(batch):
                payload = chunk.payload
                relevance_prompt += f"""
{j+1}. Citation: {payload.get('citation', 'N/A')}
   Jurisdiction: {payload.get('jurisdiction', 'N/A')}
   Section: {payload.get('section_number', 'N/A')} - {payload.get('section_title', 'N/A')}
   Text: {payload.get('text', '')[:300]}...
   Score: {chunk.score:.3f}
"""
            
            # Get AI relevance decisions
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a legal research expert. Be selective - only mark sources as ESSENTIAL or USEFUL if they truly contribute to answering the query."
                }, {
                    "role": "user", 
                    "content": relevance_prompt
                }],
                max_completion_tokens=batch_size * 10,
                temperature=0.1
            )
            
            # Parse decisions
            decisions = response.choices[0].message.content.strip().split('\n')
            
            for j, decision in enumerate(decisions):
                if j < len(batch):
                    decision_clean = decision.strip().upper()
                    if 'ESSENTIAL' in decision_clean or 'USEFUL' in decision_clean:
                        chunk_with_relevance = batch[j]
                        # Store relevance level
                        relevance_level = 'ESSENTIAL' if 'ESSENTIAL' in decision_clean else 'USEFUL'
                        chunk_with_relevance.payload['ai_relevance'] = relevance_level
                        # Debug: Print what's being assigned
                        print(f"Assigning relevance '{relevance_level}' to chunk: {chunk_with_relevance.payload.get('citation', 'N/A')}")
                        relevant_chunks.append(chunk_with_relevance)
        
        # FIX #2: Corrected reference to payload for relevance check
        essential_count = len([c for c in relevant_chunks if c.payload.get('ai_relevance', '') == 'ESSENTIAL'])
        useful_count = len([c for c in relevant_chunks if c.payload.get('ai_relevance', '') == 'USEFUL'])
        
        print(f"AI selected {len(relevant_chunks)} relevant chunks ({essential_count} essential, {useful_count} useful)")
        
        # FIX #3: Add fallback mechanism when no chunks are selected
        if len(relevant_chunks) == 0:
            print("AI filtered out all chunks - using fallback selection")
            top_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)[:20]
            for i, chunk in enumerate(top_candidates):
                # Assign relevance based on score
                if chunk.score > 0.6:
                    chunk.payload['ai_relevance'] = 'ESSENTIAL'
                else:
                    chunk.payload['ai_relevance'] = 'USEFUL'
                relevant_chunks.append(chunk)
            
            print(f"Fallback selected {len(relevant_chunks)} chunks based on relevance scores")
        
        return relevant_chunks
        
    except Exception as e:
        print(f"AI relevance filtering failed: {e}")
        # Fallback: take top candidates by score
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:min(100, len(candidates))]

def search_legal_database_adaptive(query: str, qdrant_client, embedding_model, openai_client):
    """Fully adaptive search - uses exactly what's needed, no more, no less"""
    start_time = time.time()
    
    try:
        print("=== ADAPTIVE LEGAL SEARCH ===")
        
        # Stage 1: Progressive discovery until sufficient
        candidates = adaptive_relevance_search(query, qdrant_client, embedding_model, openai_client)
        
        if not candidates:
            return {"error": "No relevant legal sources found"}
        
        # Stage 2: AI determines exact relevant set
        relevant_chunks = ai_driven_relevance_filter(query, candidates, openai_client)
        
        if not relevant_chunks:
            return {"error": "No sources deemed relevant by AI"}
        
        # Build context from exactly the chunks needed
        context_parts = []
        # FIX #4: Corrected reference to payload for relevance check
        essential_chunks = [c for c in relevant_chunks if c.payload.get('ai_relevance', '') == 'ESSENTIAL']
        useful_chunks = [c for c in relevant_chunks if c.payload.get('ai_relevance', '') == 'USEFUL']
        
        # Essential sources first (full detail)
        if essential_chunks:
            context_parts.append("=== ESSENTIAL SOURCES ===")
            for i, chunk in enumerate(essential_chunks, 1):
                # Ensure relevance is stored in payload
                if 'ai_relevance' not in chunk.payload:
                    chunk.payload['ai_relevance'] = 'ESSENTIAL'
                
                context_parts.append(f"""Essential Source {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Jurisdiction: {chunk.payload.get('jurisdiction', 'N/A')}  
Section: {chunk.payload.get('section_number', 'N/A')} - {chunk.payload.get('section_title', 'N/A')}
Legal Text: {chunk.payload.get('text', '')}
""")
        
        # Useful sources second (structured)
        if useful_chunks:
            context_parts.append("\n=== SUPPORTING SOURCES ===")
            for i, chunk in enumerate(useful_chunks, 1):
                # Ensure relevance is stored in payload
                if 'ai_relevance' not in chunk.payload:
                    chunk.payload['ai_relevance'] = 'USEFUL'
                    
                context_parts.append(f"""Supporting Source {i}:
Citation: {chunk.payload.get('citation', 'N/A')} ({chunk.payload.get('jurisdiction', 'N/A')})
Text: {chunk.payload.get('text', '')}
""")
        
        total_time = time.time() - start_time
        
        return {
            'relevant_chunks': relevant_chunks,
            'context': '\n'.join(context_parts),
            'search_stats': {
                'total_candidates_evaluated': len(candidates),
                'essential_sources': len(essential_chunks),
                'supporting_sources': len(useful_chunks),
                'total_relevant': len(relevant_chunks),
                'processing_time_seconds': round(total_time, 1)
            }
        }
        
    except Exception as e:
        print(f"Adaptive search failed: {e}")
        return {"error": str(e)}

def generate_legal_response_gpt5(query: str, search_data, openai_client):
    """Generate final legal response using GPT-5 with adaptive context"""
    try:
        print("Generating legal response with GPT-5...")
        
        # Extract context from adaptive search results
        if isinstance(search_data, dict) and 'context' in search_data:
            context = search_data['context']
            stats = search_data['search_stats']
            
            print(f"GPT-5 context: {stats['essential_sources']} essential + {stats['supporting_sources']} supporting sources")
        else:
            # Fallback for legacy format
            context = str(search_data)
            stats = {"total_relevant": "unknown"}
        
        # Enhanced legal system prompt for GPT-5
        legal_system_prompt = """You are an expert legal research assistant specializing in employment law compliance. 
        
Your task is to provide comprehensive legal analysis based on the provided legal sources. 

RESPONSE REQUIREMENTS:
1. QUOTE relevant legal provisions verbatim with exact citations
2. ANALYZE the legal requirements in detail
3. EXPLAIN compliance obligations clearly
4. IDENTIFY potential risks or penalties
5. PROVIDE actionable guidance where appropriate

Always cite sources using the exact citations provided in the context."""

        # Construct GPT-5 prompt
        gpt5_prompt = f"""{legal_system_prompt}

LEGAL QUESTION: {query}

LEGAL SOURCES AND CONTEXT:
{context}

Please provide a comprehensive legal analysis addressing this question."""

        print(f"Making GPT-5 Responses API call (context length: {len(gpt5_prompt)} characters)")
        
        # Correct GPT-5 Responses API call
        response = openai_client.responses.create(
            model="gpt-5",
            input=gpt5_prompt,  # ✅ Correct format (string, not list)
            reasoning={
                "effort": "high"  # Use high reasoning for complex legal analysis
            },
            text={
                "verbosity": "high"  # Detailed legal explanations
            },
            max_output_tokens=16384  # Allow for comprehensive responses
        )
        
        # Extract response using correct attribute
        response_text = response.output_text
        
        print(f"GPT-5 response length: {len(response_text) if response_text else 0} characters")
        
        if response_text and response_text.strip():
            return {
                "success": True,
                "content": response_text,
                "model": "gpt-5",
                "reasoning_effort": "high",
                "response_id": getattr(response, 'id', None)
            }
        else:
            return {
                "success": False,
                "error": "Empty response from GPT-5",
                "content": "GPT-5 returned an empty response. Please try rephrasing your question."
            }
            
    except Exception as e:
        print(f"GPT-5 error: {e}")
        return {
            "success": False,
            "error": str(e),
            "content": f"Error generating GPT-5 response: {str(e)}"
        }

def display_sources_expander(search_data):
    """Display all sources used in a collapsible expander"""
    if isinstance(search_data, dict) and 'relevant_chunks' in search_data:
        relevant_chunks = search_data['relevant_chunks']
        stats = search_data['search_stats']
        
        with st.expander(f"📋 Sources Referenced ({len(relevant_chunks)} sources)", expanded=False):
            st.markdown(f"""
            **Search Statistics:**
            - Total Candidates Evaluated: {stats['total_candidates_evaluated']}
            - Essential Sources: {stats['essential_sources']}
            - Supporting Sources: {stats['supporting_sources']}
            - Processing Time: {stats['processing_time_seconds']}s
            """)
            
            # FIX #5: Corrected relevance checking logic
            essential_chunks = []
            useful_chunks = []

            for chunk in relevant_chunks:
                relevance = chunk.payload.get('ai_relevance', None)
                if relevance == 'ESSENTIAL':
                    essential_chunks.append(chunk)
                elif relevance == 'USEFUL':
                    useful_chunks.append(chunk)
                else:
                    # If no relevance assigned, check score to categorize
                    if chunk.score > 0.7:
                        chunk.payload['ai_relevance'] = 'ESSENTIAL'
                        essential_chunks.append(chunk)
                    elif chunk.score > 0.4:
                        chunk.payload['ai_relevance'] = 'USEFUL'
                        useful_chunks.append(chunk)
            
            # Display Essential Sources
            if essential_chunks:
                st.markdown("### 🔴 Essential Sources")
                st.markdown("*These sources directly answer your legal question:*")
                
                for i, chunk in enumerate(essential_chunks, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Essential Source {i}**")
                            st.markdown(f"**Citation:** {chunk.payload.get('citation', 'N/A')}")
                            st.markdown(f"**Section:** {chunk.payload.get('section_number', 'N/A')} - {chunk.payload.get('section_title', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Jurisdiction:** {chunk.payload.get('jurisdiction', 'N/A')}")
                            st.markdown(f"**Relevance Score:** {chunk.score:.3f}")
                        
                        # Use HTML details/summary instead of st.expander to avoid nesting
                        citation = chunk.payload.get('citation', 'N/A')
                        section_num = chunk.payload.get('section_number', 'N/A')
                        section_title = chunk.payload.get('section_title', 'N/A')
                        jurisdiction = chunk.payload.get('jurisdiction', 'N/A')
                        legal_text = chunk.payload.get('text', '')
                        
                        st.markdown(f"""
                        <details style="margin-bottom: 1rem; border: 1px solid var(--border-light); border-radius: 8px; background: var(--bg-secondary);">
                            <summary style="padding: 1rem; cursor: pointer; font-weight: 600; color: var(--text-primary);">
                                📄 Essential Source {i}: {citation}
                            </summary>
                            <div style="padding: 0 1rem 1rem 1rem; border-top: 1px solid var(--border-light); margin-top: 0.5rem;">
                                <p><strong>Citation:</strong> {citation}</p>
                                <p><strong>Section:</strong> {section_num} - {section_title}</p>
                                <p><strong>Jurisdiction:</strong> {jurisdiction}</p>
                                <p><strong>Legal Text:</strong></p>
                                <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: 6px; max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.9em; line-height: 1.4; color: var(--text-primary);">
                                    {legal_text}
                                </div>
                            </div>
                        </details>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
            
            # Display Supporting Sources
            if useful_chunks:
                st.markdown("### 🟡 Supporting Sources")
                st.markdown("*These sources provide important context and related information:*")
                
                for i, chunk in enumerate(useful_chunks, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Supporting Source {i}**")
                            st.markdown(f"**Citation:** {chunk.payload.get('citation', 'N/A')}")
                            st.markdown(f"**Section:** {chunk.payload.get('section_number', 'N/A')} - {chunk.payload.get('section_title', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Jurisdiction:** {chunk.payload.get('jurisdiction', 'N/A')}")
                            st.markdown(f"**Relevance Score:** {chunk.score:.3f}")
                        
                        # Use HTML details/summary instead of st.expander to avoid nesting
                        citation = chunk.payload.get('citation', 'N/A')
                        section_num = chunk.payload.get('section_number', 'N/A')
                        section_title = chunk.payload.get('section_title', 'N/A')
                        jurisdiction = chunk.payload.get('jurisdiction', 'N/A')
                        legal_text = chunk.payload.get('text', '')
                        
                        st.markdown(f"""
                        <details style="margin-bottom: 1rem; border: 1px solid var(--border-light); border-radius: 8px; background: var(--bg-secondary);">
                            <summary style="padding: 1rem; cursor: pointer; font-weight: 600; color: var(--text-primary);">
                                📄 Supporting Source {i}: {citation}
                            </summary>
                            <div style="padding: 0 1rem 1rem 1rem; border-top: 1px solid var(--border-light); margin-top: 0.5rem;">
                                <p><strong>Citation:</strong> {citation}</p>
                                <p><strong>Section:</strong> {section_num} - {section_title}</p>
                                <p><strong>Jurisdiction:</strong> {jurisdiction}</p>
                                <p><strong>Legal Text:</strong></p>
                                <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: 6px; max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 0.9em; line-height: 1.4; color: var(--text-primary);">
                                    {legal_text}
                                </div>
                            </div>
                        </details>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
    
    elif isinstance(search_data, list):
        # Legacy format fallback
        with st.expander(f"📋 Sources Referenced ({len(search_data)} sources)", expanded=False):
            for i, source in enumerate(search_data, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Source {i}**")
                        st.markdown(f"**Citation:** {source.get('citation', 'N/A')}")
                        st.markdown(f"**Section:** {source.get('section_number', 'N/A')} - {source.get('section_title', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Jurisdiction:** {source.get('jurisdiction', 'N/A')}")
                        st.markdown(f"**Relevance Score:** {source.get('score', 0):.3f}")
                    
                    # Full legal text in expandable section
                    with st.expander(f"View Full Legal Text - Source {i}", expanded=False):
                        st.markdown(f"``````")
                    
                    st.divider()

def extract_legal_entities(query: str) -> Dict[str, List[str]]:
    """Extract legal entities from query for enhanced search"""
    entities = {
        'jurisdictions': [],
        'topics': [],
        'legal_concepts': []
    }
    
    query_lower = query.lower()
    
    # Extract jurisdictions
    jurisdiction_patterns = {
        'NY': ['new york', 'ny', 'nycrr'],
        'NJ': ['new jersey', 'nj', 'njac'],
        'CT': ['connecticut', 'ct', 'conn'],
        'Federal': ['federal', 'flsa', 'fmla', 'usc', 'cfr']
    }
    
    for jurisdiction, patterns in jurisdiction_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            entities['jurisdictions'].append(jurisdiction)
    
    # Extract legal topics
    topic_patterns = [
        'minimum wage', 'overtime', 'break', 'meal period', 'sick leave',
        'family leave', 'vacation', 'holiday', 'wage', 'hour', 'employment',
        'discrimination', 'harassment', 'safety', 'workers compensation'
    ]
    
    for topic in topic_patterns:
        if topic in query_lower:
            entities['topics'].append(topic)
    
    # Extract legal concepts
    concept_patterns = [
        'requirement', 'violation', 'penalty', 'compliance', 'exemption',
        'definition', 'procedure', 'application', 'enforcement'
    ]
    
    for concept in concept_patterns:
        if concept in query_lower:
            entities['legal_concepts'].append(concept)
    
    return entities

def analyze_query_complexity(query: str, openai_client) -> Dict[str, Any]:
    """Analyze query complexity to determine search strategy"""
    try:
        complexity_prompt = f"""
        Analyze this legal compliance query for complexity and scope:
        Query: "{query}"
        
        Respond with JSON only:
        {{
            "complexity": "simple|moderate|complex|comprehensive",
            "jurisdictions": ["list of jurisdictions mentioned"],
            "topics": ["list of legal topics"],
            "requires_comprehensive_search": true|false,
            "estimated_sources_needed": 25,
            "search_strategy": "standard|multi_jurisdictional|comprehensive",
            "explanation": "brief explanation of complexity assessment"
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use faster model for analysis
            messages=[{
                "role": "system",
                "content": "You are a legal research expert. Analyze queries precisely and respond only with valid JSON."
            }, {
                "role": "user", 
                "content": complexity_prompt
            }],
            max_completion_tokens=300,
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Query complexity analysis failed: {e}")
        # Fallback to simple analysis
        return {
            "complexity": "moderate",
            "jurisdictions": [],
            "topics": [],
            "requires_comprehensive_search": True,
            "estimated_sources_needed": 100,
            "search_strategy": "comprehensive",
            "explanation": "Fallback analysis due to parsing error"
        }

def intelligent_wide_retrieval(query: str, qdrant_client, embedding_model, complexity_analysis: Dict) -> List:
    """Cast wide net using multiple search strategies based on complexity"""
    try:
        all_candidates = {}  # Use dict to avoid duplicates
        entities = extract_legal_entities(query)
        
        # Determine search parameters based on complexity
        if complexity_analysis["complexity"] == "simple":
            base_limit = 100
            score_threshold = 0.4
        elif complexity_analysis["complexity"] == "moderate":
            base_limit = 300
            score_threshold = 0.35
        elif complexity_analysis["complexity"] == "complex":
            base_limit = 600
            score_threshold = 0.3
        else:  # comprehensive
            base_limit = 1000
            score_threshold = 0.25
        
        print(f"Wide retrieval: {complexity_analysis['complexity']} query, searching {base_limit} candidates")
        
        # Strategy 1: Direct semantic search
        query_embedding = embedding_model.encode([query])[0].tolist()
        semantic_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_embedding,
            limit=base_limit,
            score_threshold=score_threshold
        )
        
        for result in semantic_results:
            all_candidates[result.id] = result
        
        # Strategy 2: Entity-based expansion for complex queries
        if complexity_analysis["complexity"] in ["complex", "comprehensive"]:
            # Search by jurisdictions
            for jurisdiction in entities['jurisdictions']:
                jurisdiction_embedding = embedding_model.encode([f"{jurisdiction} legal requirements"])[0].tolist()
                jurisdiction_results = qdrant_client.search(
                    collection_name="legal_regulations",
                    query_vector=jurisdiction_embedding,
                    limit=200,
                    score_threshold=0.3
                )
                for result in jurisdiction_results:
                    all_candidates[result.id] = result
            
            # Search by topics
            for topic in entities['topics']:
                topic_query = f"{topic} legal requirements"
                topic_embedding = embedding_model.encode([topic_query])[0].tolist()
                topic_results = qdrant_client.search(
                    collection_name="legal_regulations",
                    query_vector=topic_embedding,
                    limit=150,
                    score_threshold=0.3
                )
                for result in topic_results:
                    all_candidates[result.id] = result
        
        # Strategy 3: Related provision discovery for comprehensive queries
        if complexity_analysis["complexity"] == "comprehensive" and len(semantic_results) > 0:
            for result in semantic_results[:20]:  # Top 20 semantic matches
                citation = result.payload.get('citation', '')
                jurisdiction = result.payload.get('jurisdiction', '')
                
                if jurisdiction:
                    # Find other provisions from same jurisdiction
                    related_results = qdrant_client.scroll(
                        collection_name="legal_regulations",
                        scroll_filter=Filter(
                            must=[FieldCondition(
                                key="jurisdiction",
                                match=MatchValue(value=jurisdiction)
                            )]
                        ),
                        limit=100
                    )
                    
                    for related in related_results[0]:
                        if related.payload.get('text', '').lower().find(query.lower().split()[0]) != -1:
                            all_candidates[related.id] = related
        
        final_candidates = list(all_candidates.values())
        print(f"Wide retrieval found {len(final_candidates)} candidate chunks")
        return final_candidates
        
    except Exception as e:
        print(f"Wide retrieval failed: {e}")
        # Fallback to basic search
        query_embedding = embedding_model.encode([query])[0].tolist()
        return qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_embedding,
            limit=200,
            score_threshold=0.3
        )

def ai_relevance_filter(query: str, candidates: List, openai_client) -> List:
    """Let GPT-5 determine what's actually relevant"""
    try:
        relevant_chunks = []
        batch_size = 50  # Process in smaller batches for better accuracy
        
        print(f"AI relevance filtering {len(candidates)} candidates in batches of {batch_size}")
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            
            # Create relevance assessment prompt
            relevance_prompt = f"""
LEGAL QUERY: {query}

TASK: For each source below, determine relevance level. Be precise and conservative.

RELEVANCE LEVELS:
- ESSENTIAL: Directly answers the query or provides critical legal requirements
- IMPORTANT: Provides valuable related information or context
- SUPPLEMENTARY: Tangentially related but potentially useful
- IRRELEVANT: Not related to the query

Respond with ONLY the relevance level for each source (one per line):

"""
            
            # Add sources to prompt
            for j, chunk in enumerate(batch):
                text_preview = chunk.payload.get('text', '')[:400]
                relevance_prompt += f"""
SOURCE {j+1}:
Citation: {chunk.payload.get('citation', 'N/A')}
Jurisdiction: {chunk.payload.get('jurisdiction', 'N/A')}
Section: {chunk.payload.get('section_number', 'N/A')} - {chunk.payload.get('section_title', 'N/A')}
Text Preview: {text_preview}...

"""
            
            # Get AI relevance decisions
            relevance_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for filtering
                messages=[{
                    "role": "system",
                    "content": "You are a legal research expert. Assess relevance precisely. Respond only with relevance levels, one per line."
                }, {
                    "role": "user",
                    "content": relevance_prompt
                }],
                max_completion_tokens=len(batch) * 15,
                temperature=0.1
            )
            
            # Parse relevance decisions
            decisions = relevance_response.choices[0].message.content.strip().split('\n')
            
            # Include ESSENTIAL and IMPORTANT chunks
            for j, decision in enumerate(decisions):
                if j < len(batch):  # Safety check
                    decision_clean = decision.strip().upper()
                    if any(level in decision_clean for level in ['ESSENTIAL', 'IMPORTANT']):
                        chunk_with_relevance = batch[j]
                        # Add relevance metadata
                        if hasattr(chunk_with_relevance, 'payload'):
                            chunk_with_relevance.payload['ai_relevance'] = decision_clean
                        else:
                            chunk_with_relevance.ai_relevance = decision_clean
                        relevant_chunks.append(chunk_with_relevance)
        
        essential_count = len([c for c in relevant_chunks if 'ESSENTIAL' in str(getattr(c, 'ai_relevance', getattr(c.payload, 'ai_relevance', '')))])
        important_count = len([c for c in relevant_chunks if 'IMPORTANT' in str(getattr(c, 'ai_relevance', getattr(c.payload, 'ai_relevance', '')))])
        
        print(f"AI filtered to {len(relevant_chunks)} relevant chunks ({essential_count} essential, {important_count} important)")
        return relevant_chunks
        
    except Exception as e:
        print(f"AI relevance filtering failed: {e}")
        # Fallback: return top candidates by score
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:100]

def build_adaptive_context(query: str, relevant_chunks: List) -> str:
    """Build context that adapts to the number of relevant sources found"""
    try:
        # Categorize chunks by AI-determined relevance
        essential_chunks = []
        important_chunks = []
        
        for chunk in relevant_chunks:
            relevance = getattr(chunk, 'ai_relevance', getattr(chunk.payload, 'ai_relevance', ''))
            if 'ESSENTIAL' in str(relevance):
                essential_chunks.append(chunk)
            elif 'IMPORTANT' in str(relevance):
                important_chunks.append(chunk)
        
        print(f"Context building: {len(essential_chunks)} essential, {len(important_chunks)} important chunks")
        
        context_parts = []
        
        # Always include all essential chunks (full text)
        if essential_chunks:
            context_parts.append("=== ESSENTIAL LEGAL PROVISIONS ===")
            context_parts.append(f"The following {len(essential_chunks)} sources directly address your query and MUST be quoted and analyzed:")
            context_parts.append("")
            
            for i, chunk in enumerate(essential_chunks, 1):
                context_parts
            context_parts.append(f"""
ESSENTIAL SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Jurisdiction: {chunk.payload.get('jurisdiction', 'N/A')}
Section: {chunk.payload.get('section_number', 'N/A')} - {chunk.payload.get('section_title', 'N/A')}
Legal Text: {chunk.payload.get('text', '')}
""")
            context_parts.append("")

        # Include important chunks (summary format if many)
        if important_chunks:
            context_parts.append("=== IMPORTANT SUPPORTING PROVISIONS ===")
            context_parts.append(f"The following {len(important_chunks)} sources provide important context:")
            context_parts.append("")

            for i, chunk in enumerate(important_chunks, 1):
                context_parts.append(f"""
IMPORTANT SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')} ({chunk.payload.get('jurisdiction', 'N/A')})
Text: {chunk.payload.get('text', '')[:500]}...
""")
                context_parts.append("")

        return "\n".join(context_parts)

    except Exception as e:
        print(f"Context building failed: {e}")
        # Fallback to simple concatenation
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:50], 1):  # Limit to 50 chunks
            context_parts.append(f"""
SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Jurisdiction: {chunk.payload.get('jurisdiction', 'N/A')}
Text: {chunk.payload.get('text', '')}
""")
        return "\n".join(context_parts)

def process_legal_query():
    """Main function to process legal queries"""
    if not authenticate_user():
        return

    # Store systems in session for callbacks
    store_systems_in_session()

    # Get systems from session state
    user_manager = st.session_state.user_manager
    embedding_model = st.session_state.embedding_model
    qdrant_client = st.session_state.qdrant_client
    openai_client = st.session_state.openai_client

    # Page header
    st.markdown("""
        <div class="page-header">
            <h1>⚖️ AI Compliance Assistant</h1>
            <p>Professional Employment Organization Legal Research</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for user info and logout
    with st.sidebar:
        st.markdown("### 👤 Session Info")
        st.markdown(f"**Access Code:** {st.session_state.access_code}")
        st.markdown(f"**Session ID:** {st.session_state.session_id[:8]}...")

        if st.button("🚪 Logout", use_container_width=True):
            user_manager.end_session(st.session_state.session_id)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Query input
    with st.form("legal_query_form", clear_on_submit=True):
        query = st.text_area(
            "Legal Question",
            placeholder="Enter your compliance question...",
            height=100,
            key="user_legal_query"
        )

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            submitted = st.form_submit_button("🔍 Research", use_container_width=True)

    # Process query
    if submitted and query.strip():
        st.session_state.query_to_process = query.strip()

    if hasattr(st.session_state, 'query_to_process') and st.session_state.query_to_process:
        query_to_process = st.session_state.query_to_process

        with st.spinner("🔍 Analyzing legal requirements..."):
            try:
                # Step 1: Analyze query complexity
                complexity_analysis = analyze_query_complexity(query_to_process, openai_client)

                # Step 2: Wide retrieval based on complexity
                candidates = intelligent_wide_retrieval(query_to_process, qdrant_client, embedding_model, complexity_analysis)

                if not candidates:
                    st.error("No relevant legal sources found for your query.")
                    return

                # Step 3: AI relevance filtering
                relevant_chunks = ai_relevance_filter(query_to_process, candidates, openai_client)

                if not relevant_chunks:
                    st.error("No sources were deemed relevant by AI analysis.")
                    return

                # Step 4: Build adaptive context
                context = build_adaptive_context(query_to_process, relevant_chunks)

                # Step 5: Generate response with GPT-5
                response_data = generate_legal_response_gpt5(query_to_process, {
                    'context': context,
                    'relevant_chunks': relevant_chunks,
                    'search_stats': {
                        'total_candidates_evaluated': len(candidates),
                        'essential_sources': len([c for c in relevant_chunks if 'ESSENTIAL' in str(getattr(c, 'ai_relevance', getattr(c.payload, 'ai_relevance', '')))]),
                        'supporting_sources': len([c for c in relevant_chunks if 'IMPORTANT' in str(getattr(c, 'ai_relevance', getattr(c.payload, 'ai_relevance', '')))]),
                        'total_relevant': len(relevant_chunks),
                        'processing_time_seconds': 0
                    }
                }, openai_client)

                # Display results
                if response_data["success"]:
                    st.markdown("### ⚖️ Legal Analysis")
                    st.markdown(response_data["content"])

                    # Display sources
                    display_sources_expander({
                        'relevant_chunks': relevant_chunks,
                        'search_stats': {
                            'total_candidates_evaluated': len(candidates),
                            'essential_sources': len([c for c in relevant_chunks if 'ESSENTIAL' in str(getattr(c, 'ai_relevance', getattr(c.payload, 'ai_relevance', '')))]),
                            'supporting_sources': len([c for c in relevant_chunks if 'IMPORTANT' in str(getattr(c, 'ai_relevance', getattr(c.payload, 'ai_relevance', '')))]),
                            'total_relevant': len(relevant_chunks),
                            'processing_time_seconds': 0
                        }
                    })
                else:
                    st.error(f"Failed to generate response: {response_data.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

        # Clear the processed query
        del st.session_state.query_to_process

if __name__ == "__main__":
    process_legal_query()
