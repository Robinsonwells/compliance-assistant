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
            
            relevance_prompt = f"""Legal Query: "{query}"

TASK: Determine which sources are relevant for answering this query. Be selective - only include sources that add value.

RELEVANCE CRITERIA:
- ESSENTIAL: Directly answers the query or provides critical legal requirements
- USEFUL: Provides valuable context or related requirements  
- SKIP: Not directly related or redundant with already selected content

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
                        chunk_with_relevance.ai_relevance = 'ESSENTIAL' if 'ESSENTIAL' in decision_clean else 'USEFUL'
                        relevant_chunks.append(chunk_with_relevance)
        
        essential_count = len([c for c in relevant_chunks if getattr(c, 'ai_relevance', '') == 'ESSENTIAL'])
        useful_count = len([c for c in relevant_chunks if getattr(c, 'ai_relevance', '') == 'USEFUL'])
        
        print(f"AI selected {len(relevant_chunks)} relevant chunks ({essential_count} essential, {useful_count} useful)")
        
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
        essential_chunks = [c for c in relevant_chunks if getattr(c, 'ai_relevance', '') == 'ESSENTIAL']
        useful_chunks = [c for c in relevant_chunks if getattr(c, 'ai_relevance', '') == 'USEFUL']
        
        # Essential sources first (full detail)
        if essential_chunks:
            context_parts.append("=== ESSENTIAL SOURCES ===")
            for i, chunk in enumerate(essential_chunks, 1):
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
                context_parts.append(f"""ESSENTIAL SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Jurisdiction: {chunk.payload.get('jurisdiction', 'N/A')}
Section: {chunk.payload.get('section_number', 'N/A')} - {chunk.payload.get('section_title', 'N/A')}
Full Legal Text: "{chunk.payload.get('text', '')}"
""")
        
        # Include important chunks (full text if manageable, otherwise structured)
        if important_chunks:
            context_parts.append("\n=== IMPORTANT RELATED PROVISIONS ===")
            context_parts.append(f"The following {len(important_chunks)} sources provide important context and should be referenced where relevant:")
            context_parts.append("")
            
            if len(important_chunks) <= 30:
                # Include full text for manageable number
                for i, chunk in enumerate(important_chunks, 1):
                    context_parts.append(f"""IMPORTANT SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Jurisdiction: {chunk.payload.get('jurisdiction', 'N/A')}
Legal Text: "{chunk.payload.get('text', '')}"
""")
            else:
                # Provide structured summary for large number
                context_parts.append(f"[{len(important_chunks)} Important Sources - Structured Summary]")
                context_parts.append("")
                
                # Group by jurisdiction
                by_jurisdiction = {}
                for chunk in important_chunks:
                    jurisdiction = chunk.payload.get('jurisdiction', 'Unknown')
                    if jurisdiction not in by_jurisdiction:
                        by_jurisdiction[jurisdiction] = []
                    by_jurisdiction[jurisdiction].append(chunk)
                
                for jurisdiction, chunks in by_jurisdiction.items():
                    context_parts.append(f"**{jurisdiction} Provisions ({len(chunks)} sources):**")
                    for chunk in chunks[:10]:  # Limit per jurisdiction
                        context_parts.append(f"- {chunk.payload.get('citation', 'N/A')}: {chunk.payload.get('text', '')[:200]}...")
                    if len(chunks) > 10:
                        context_parts.append(f"- ... and {len(chunks) - 10} additional {jurisdiction} sources")
                    context_parts.append("")
        
        final_context = "\n".join(context_parts)
        
        # Check context length and provide summary if too large
        if len(final_context) > 800000:  # 800K character limit
            print("Context too large, creating executive summary")
            context_parts = []
            
            # Essential sources (always include)
            if essential_chunks:
                context_parts.append("=== ESSENTIAL LEGAL PROVISIONS (FULL TEXT) ===")
                for i, chunk in enumerate(essential_chunks[:20], 1):  # Limit to top 20 essential
                    context_parts.append(f"""ESSENTIAL SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Text: "{chunk.payload.get('text', '')}"
""")
            
            # Important sources (summary only)
            if important_chunks:
                context_parts.append("\n=== IMPORTANT SOURCES (SUMMARY) ===")
                context_parts.append("Key Citations and Provisions:")
                for chunk in important_chunks[:50]:  # Top 50 important
                    context_parts.append(f"- {chunk.payload.get('citation', 'N/A')} ({chunk.payload.get('jurisdiction', 'N/A')}): {chunk.payload.get('text', '')[:150]}...")
            
            final_context = "\n".join(context_parts)
        
        return final_context
        
    except Exception as e:
        print(f"Context building failed: {e}")
        # Fallback to simple concatenation
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:50], 1):
            context_parts.append(f"""SOURCE {i}:
Citation: {chunk.payload.get('citation', 'N/A')}
Text: "{chunk.payload.get('text', '')}"
""")
        return "\n\n".join(context_parts)

def search_legal_database(query: str, qdrant_client, embedding_model, openai_client, adaptive: bool = True):
    """Enhanced search with AI-driven relevance and adaptive context"""
    try:
        if adaptive:
            # Use new adaptive search
            return search_legal_database_adaptive(query, qdrant_client, embedding_model, openai_client)
        else:
            # Fallback to original method
            query_embedding = embedding_model.encode([query])[0].tolist()
            search_results = qdrant_client.search(
                collection_name="legal_regulations",
                query_vector=query_embedding,
                limit=50,
                with_payload=True
            )
            
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'text': result.payload.get('text', ''),
                    'citation': result.payload.get('citation', ''),
                    'section_number': result.payload.get('section_number', ''),
                    'section_title': result.payload.get('section_title', ''),
                    'jurisdiction': result.payload.get('jurisdiction', ''),
                    'score': result.score
                })
            
            return formatted_results
            
    except Exception as e:
        print(f"Enhanced search failed: {e}")
        # Ultimate fallback
        query_embedding = embedding_model.encode([query])[0].tolist()
        search_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_embedding,
            limit=50,
            with_payload=True
        )
        
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'text': result.payload.get('text', ''),
                'citation': result.payload.get('citation', ''),
                'section_number': result.payload.get('section_number', ''),
                'section_title': result.payload.get('section_title', ''),
                'jurisdiction': result.payload.get('jurisdiction', ''),
                'score': result.score
            })
        
        return formatted_results

def generate_legal_response(query: str, search_data):
    """Generate legal response using GPT-5 with adaptive context"""
    return generate_legal_response_gpt5(query, search_data, st.session_state.openai_client)

def _process_legal_question_logic(prompt: str):
    """Complete adaptive processing with GPT-5 final response"""
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.status("🔍 Analyzing your legal question...", expanded=True) as status:
        try:
            # Get systems from session state
            qdrant_client = st.session_state.qdrant_client
            embedding_model = st.session_state.embedding_model
            openai_client = st.session_state.openai_client
            
            # Step 1: Adaptive search with GPT-4o-mini intelligence
            status.write("📚 Adaptive legal search with AI filtering...")
            search_data = search_legal_database(prompt, qdrant_client, embedding_model, openai_client, adaptive=True)
            
            if 'error' in search_data:
                error_response = f"Search error: {search_data['error']}"
                st.session_state.messages.append({"role": "assistant", "content": error_response})
                status.update(label="❌ Search failed", state="error")
                return
            
            # Step 2: Show adaptive results
            stats = search_data['search_stats']
            status.write(f"✅ Found {stats['total_relevant']} relevant sources ({stats['essential_sources']} essential)")
            
            # Step 3: Generate GPT-5 response
            status.write("🧠 Generating comprehensive legal analysis with GPT-5...")
            response = generate_legal_response(prompt, search_data)
            
            if response["success"]:
                st.session_state.messages.append({"role": "assistant", "content": response["content"]})
                status.update(label="✅ GPT-5 legal analysis complete!", state="complete")
            else:
                error_msg = f"GPT-5 analysis failed: {response.get('error', 'Unknown error')}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                status.update(label="❌ GPT-5 error", state="error")
                
        except Exception as e:
            error_response = f"Processing error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_response})
            status.update(label="❌ Processing failed", state="error")

def main():
    """Main application logic"""
    # Initialize systems in session state
    _init_session_state_systems()
    user_manager = st.session_state.user_manager
    
    # Authenticate user
    if not authenticate_user():
        return
    
    # Validate session
    if not user_manager.is_session_valid(st.session_state.session_id):
        st.error("Your session has expired. Please log in again.")
        st.session_state.authenticated = False
        st.rerun()
    
    # Update session activity
    user_manager.update_session_activity(st.session_state.session_id)
    
    # Initialize UI state and processing variables
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False
    if "query_to_process" not in st.session_state:
        st.session_state.query_to_process = None
    if "user_legal_query" not in st.session_state:
        st.session_state.user_legal_query = ""
    
    # Main application interface
    st.markdown("""
        <div class="dashboard-header">
            <h1 style="margin: 0; color: white;">AI Legal Compliance Assistant</h1>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                Professional Employment Organization Legal Research Tool
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    st.markdown("### 💬 Legal Research Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Enhanced message formatting
            st.markdown(f'<div class="message-content">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Typing indicator
    if st.session_state.is_typing:
        st.markdown("""
            <div class="typing-indicator visible">
                <span>AI is analyzing legal sources</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Enhanced input area for legal questions
    st.markdown("---")
    st.markdown("### 🔍 Ask Your Legal Compliance Question")
    
    with st.container():
        prompt_input = st.text_area(
            "Enter your detailed legal question:",
            height=120,
            placeholder="Example: What are the meal break requirements for employees working 8+ hours in Connecticut? Include any exceptions for different industries and penalty requirements for violations.",
            help="Provide detailed questions for more comprehensive legal analysis. Include specific jurisdictions, industries, or circumstances for better results.",
            key="user_legal_query",
            value=st.session_state.user_legal_query
        )
        
        # Submit button with better positioning
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.button(
                "🔍 Research Legal Requirements", 
                use_container_width=True,
                type="primary",
                disabled=not prompt_input.strip(),
                on_click=_submit_question_callback
            )
    
    # Process queued question after the input area is rendered
    if st.session_state.query_to_process:
        query = st.session_state.query_to_process
        st.session_state.query_to_process = None  # Clear to prevent re-processing
        _process_legal_question_logic(query)
        st.rerun()  # Refresh to show new messages

if __name__ == "__main__":
    main()
