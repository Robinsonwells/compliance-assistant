import streamlit as st
import os
from dotenv import load_dotenv
import uuid
import requests
import re
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Get API keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# US States and abbreviations for complexity scoring
US_STATES = {
    'alabama', 'al', 'alaska', 'ak', 'arizona', 'az', 'arkansas', 'ar', 'california', 'ca',
    'colorado', 'co', 'connecticut', 'ct', 'delaware', 'de', 'florida', 'fl', 'georgia', 'ga',
    'hawaii', 'hi', 'idaho', 'id', 'illinois', 'il', 'indiana', 'in', 'iowa', 'ia',
    'kansas', 'ks', 'kentucky', 'ky', 'louisiana', 'la', 'maine', 'me', 'maryland', 'md',
    'massachusetts', 'ma', 'michigan', 'mi', 'minnesota', 'mn', 'mississippi', 'ms',
    'missouri', 'mo', 'montana', 'mt', 'nebraska', 'ne', 'nevada', 'nv', 'new hampshire', 'nh',
    'new jersey', 'nj', 'new mexico', 'nm', 'new york', 'ny', 'north carolina', 'nc',
    'north dakota', 'nd', 'ohio', 'oh', 'oklahoma', 'ok', 'oregon', 'or', 'pennsylvania', 'pa',
    'rhode island', 'ri', 'south carolina', 'sc', 'south dakota', 'sd', 'tennessee', 'tn',
    'texas', 'tx', 'utah', 'ut', 'vermont', 'vt', 'virginia', 'va', 'washington', 'wa',
    'west virginia', 'wv', 'wisconsin', 'wi', 'wyoming', 'wy'
}

# Cost per 15k tokens (research-backed estimates)
COST_PER_15K_TOKENS = {
    "low": 0.07,      # $0.06-0.08 range
    "medium": 0.13,   # $0.13 baseline
    "high": 0.28      # $0.25-0.30 range
}

# Cost per token (calculated from 15k baseline)
COST_PER_TOKEN = {
    "low": COST_PER_15K_TOKENS["low"] / 15000,
    "medium": COST_PER_15K_TOKENS["medium"] / 15000,
    "high": COST_PER_15K_TOKENS["high"] / 15000
}

# Import custom modules
from user_management import UserManager
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT

# Initialize components
try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    import torch
    
    # Initialize clients
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    # Initialize embedding model with proper device handling
    # Use GPU if available, fallback to CPU
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embedding_model = embedding_model.to(device)

    # Log which device is being used
    print(f"Using device: {device} for embeddings")
    
    # Initialize OpenAI client with extended timeout for GPT-5
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=600.0  # 10 minute timeout for GPT-5 reasoning
    )
    
    # Initialize user manager
    user_manager = UserManager()
    
except Exception as e:
    st.error(f"Failed to initialize components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PEO Compliance Assistant",
    page_icon="",
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

def search_legal_database(query: str, limit: int = 20) -> List[Dict[str, Any]]:
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

def calculate_complexity_score(query: str) -> tuple[int, dict]:
    """Calculate complexity score based on 10 factors (0-30 points max)"""
    query_lower = query.lower()
    words = query.split()
    word_count = len(words)
    
    score_breakdown = {}
    total_score = 0
    
    # 1. State count (1-3 pts)
    state_mentions = sum(1 for word in query_lower.split() if word in US_STATES)
    if state_mentions == 1:
        state_score = 1
    elif state_mentions == 2:
        state_score = 2
    elif state_mentions >= 3:
        state_score = 3
    else:
        state_score = 0
    score_breakdown["State count"] = f"{state_mentions} states = {state_score}pts"
    total_score += state_score
    
    # 2. Query length (1-3 pts)
    if word_count < 10:
        length_score = 1
    elif word_count <= 20:
        length_score = 2
    else:
        length_score = 3
    score_breakdown["Query length"] = f"{word_count} words = {length_score}pts"
    total_score += length_score
    
    # 3. Comparison keywords (1-3 pts)
    comparison_keywords = ['vs', 'versus', 'compare', 'contrast', 'difference', 'different']
    comparison_count = sum(1 for keyword in comparison_keywords if keyword in query_lower)
    comparison_score = min(3, comparison_count)
    score_breakdown["Comparison keywords"] = f"{comparison_count} found = {comparison_score}pts"
    total_score += comparison_score
    
    # 4. Exception keywords (1-3 pts)
    exception_keywords = ['except', 'unless', 'however', 'exemption', 'but', 'although']
    exception_count = sum(1 for keyword in exception_keywords if keyword in query_lower)
    exception_score = min(3, exception_count)
    score_breakdown["Exception keywords"] = f"{exception_count} found = {exception_score}pts"
    total_score += exception_score
    
    # 5. Hypothetical keywords (1-3 pts)
    hypothetical_keywords = ['if', 'suppose', 'scenario', 'what if', 'hypothetical', 'assume']
    hypothetical_count = sum(1 for keyword in hypothetical_keywords if keyword in query_lower)
    hypothetical_score = min(3, hypothetical_count)
    score_breakdown["Hypothetical keywords"] = f"{hypothetical_count} found = {hypothetical_score}pts"
    total_score += hypothetical_score
    
    # 6. Analysis keywords (1-3 pts)
    analysis_keywords = ['analyze', 'evaluate', 'implications', 'impact', 'consequences', 'assess']
    analysis_count = sum(1 for keyword in analysis_keywords if keyword in query_lower)
    analysis_score = min(3, analysis_count)
    score_breakdown["Analysis keywords"] = f"{analysis_count} found = {analysis_score}pts"
    total_score += analysis_score
    
    # 7. Conflict keywords (1-3 pts)
    conflict_keywords = ['conflicting', 'preemption', 'override', 'supersede', 'conflict', 'clash']
    conflict_count = sum(1 for keyword in conflict_keywords if keyword in query_lower)
    conflict_score = min(3, conflict_count)
    score_breakdown["Conflict keywords"] = f"{conflict_count} found = {conflict_score}pts"
    total_score += conflict_score
    
    # 8. Multi-step indicators (1-3 pts)
    multistep_keywords = ['first', 'then', 'also', 'additionally', 'furthermore', 'next', 'step']
    multistep_count = sum(1 for keyword in multistep_keywords if keyword in query_lower)
    multistep_score = min(3, multistep_count)
    score_breakdown["Multi-step indicators"] = f"{multistep_count} found = {multistep_score}pts"
    total_score += multistep_score
    
    # 9. Complexity phrases (1-3 pts)
    complexity_keywords = ['edge case', 'grey area', 'gray area', 'unclear', 'ambiguous', 'complex']
    complexity_count = sum(1 for keyword in complexity_keywords if keyword in query_lower)
    complexity_score = min(3, complexity_count)
    score_breakdown["Complexity phrases"] = f"{complexity_count} found = {complexity_score}pts"
    total_score += complexity_score
    
    # 10. Question type (1-3 pts)
    if any(word in query_lower for word in ['what is', 'define', 'definition']):
        question_score = 1  # Simple fact lookup
        question_type = "Fact lookup"
    elif any(word in query_lower for word in ['how', 'why', 'when', 'where']):
        question_score = 2  # Interpretation needed
        question_type = "Interpretation"
    elif any(word in query_lower for word in ['should', 'recommend', 'best', 'strategy']):
        question_score = 3  # Synthesis required
        question_type = "Synthesis"
    else:
        question_score = 2  # Default to interpretation
        question_type = "Interpretation"
    score_breakdown["Question type"] = f"{question_type} = {question_score}pts"
    total_score += question_score
    
    return total_score, score_breakdown

def get_reasoning_effort(complexity_score: int) -> str:
    """Map complexity score to reasoning effort level"""
    # Low is no longer an option, minimum is medium
    if complexity_score >= 23:
        return "high"
    else:
        return "medium"

def classify_reasoning_effort_with_gpt4o_mini(query: str) -> str:
    """Use GPT-4o-mini to classify reasoning effort required for the query"""
    try:
        # Import the classifier prompt
        from system_prompts import GPT4O_MINI_CLASSIFIER_SYSTEM_PROMPT
        
        # Check for explicit user override phrases first
        query_lower = query.lower()
        
        # High effort overrides
        high_effort_phrases = [
            "use high effort", "high reasoning", "detailed analysis", 
            "thorough", "comprehensive analysis"
        ]
        if any(phrase in query_lower for phrase in high_effort_phrases):
            return "high"
        
        # Low effort overrides changed to medium
        low_effort_phrases = [
            "use low effort", "low reasoning", "quick answer",
            "simple answer", "brief"
        ]
        if any(phrase in query_lower for phrase in low_effort_phrases):
            return "medium"
        
        # Medium effort overrides
        medium_effort_phrases = [
            "use medium effort", "medium reasoning"
        ]
        if any(phrase in query_lower for phrase in medium_effort_phrases):
            return "medium"
        
        # No override detected, use GPT-4o-mini to classify
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GPT4O_MINI_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            max_tokens=10,
            temperature=0,
            timeout=10
        )
        
        # Parse and validate response
        effort_level = response.choices[0].message.content.strip().lower()
        
        # Validate response is one of the expected values
        # Map low to medium since low is no longer an option
        if effort_level == "low":
            return "medium"
        elif effort_level in ["medium", "high"]:
            return effort_level
        else:
            # Invalid response, default to medium
            print(f"Invalid classifier response: {effort_level}, defaulting to medium")
            return "medium"
            
    except Exception as e:
        # Any error, default to medium (safe middle ground)
        print(f"Error in GPT-4o-mini classifier: {e}, defaulting to medium")
        return "medium"

def calculate_estimated_cost(total_tokens: int, reasoning_effort: str) -> float:
    """Calculate estimated cost based on token usage and reasoning effort"""
    return total_tokens * COST_PER_TOKEN[reasoning_effort]

def call_perplexity_auditor(original_query: str, main_answer: str) -> Dict[str, Any]:
    """Step 3: Selective fact-checker - only flags material issues and adds context"""
    try:
        if not PERPLEXITY_API_KEY:
            return {
                "report": "✅ **VERIFICATION SKIPPED:** API key not configured.",
                "citations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "finish_reason": "error"
            }

        prompt = f"""You are an independent legal auditor with web search access. Your job is NOT to verify every claim, but rather to catch material issues and add important context.

CRITICAL: SEARCH ONLY FOR LABOR AND EMPLOYMENT LAW
This query is about: {original_query}
Only search for and cite sources about labor law, employment regulations, wage/hour law, worker classification, or workplace compliance. DO NOT search for or cite property law, tax law, criminal law, or any non-employment topics.

ONLY flag issues if you find:
1. MORE RECENT, CONTRADICTORY information (not just missing details)
2. MISSING CONTEXT or important legal nuance that affects interpretation
3. UPCOMING LAWS or recent legislative changes relevant to the answer
4. OUTDATED INFORMATION where laws have been substantively changed

DO NOT flag:
- Lack of detail in summaries
- Missing citations (if the law stated is correct)
- Pedantic formatting issues
- Hypothetical "what if" scenarios
- Laws that "could" change or might be proposed

IMPORTANT CONTEXT:
Many citations in the knowledge base are from when laws were established, not their last update.
This is normal and acceptable. Only flag if you find a MORE RECENT source that contradicts the core claim.

PRIORITIZE THESE OFFICIAL SOURCES FOR CURRENT INFORMATION:
- State legislature websites (.gov domains) - most current
- Cornell Legal Information Institute (law.cornell.edu) - authoritative summaries
- Federal regulations (ecfr.gov, regulations.gov) - official versions
- State Department of Labor websites - current enforcement info
- Government agency websites (DOL, EEOC, NLRB) - interpretations
- Official case law databases - recent rulings

IGNORE (and do not cite):
- Blog posts about legal topics
- Commercial legal advice websites
- Legal services marketing content
- Non-authoritative sources
- Wikipedia or user-generated content
- Secondary sources that aren't citing to primary law

ORIGINAL QUESTION:
{original_query}

ANSWER TO VERIFY:
{main_answer}

YOUR TASK - FOCUS ONLY ON MATERIAL ISSUES:

1. SEARCH FOR CONTRADICTIONS
   - If the answer states "law X applies," search for current status of law X
   - If the answer cites a specific statute, search for RECENT amendments
   - If the answer gives a dollar amount, verify it's current as of 2025
   - If the answer states a requirement, confirm it hasn't been repealed

2. SEARCH FOR MISSING CONTEXT
   - Are there recent state law changes that provide important nuance?
   - Is there a recent federal law that interacts with state law differently?
   - Are there upcoming bills or pending amendments (within next 12 months)?
   - Are there enforcement priorities or recent guidance from DOL/EEOC/NLRB?

3. SEARCH FOR STATUTE UPDATES
   - Extract each statute cited: (e.g., "Cal. Lab. Code § 510")
   - Search: "[State] [statute code] [statute number] current"
   - Check: Last amendment date, current version, recent changes
   - Only flag if: More recent version contradicts the answer

CRITICAL GUIDELINES:
- Do NOT criticize summaries for lacking detail
- Do NOT flag citations based on their age (old citations can be correct)
- Do NOT flag laws based on what they "could" do or "might" do
- Do NOT introduce irrelevant law (e.g., federal law if query is state-specific)
- Do NOT assume information is wrong just because you can't verify it
- Do NOT require citations for every fact (as long as it's stated correctly)

REPORT FORMAT:

🚨 **MATERIAL ISSUES FOUND** (only if actual contradictions or critical context)
- [Issue]: [What the answer stated] vs [What current law shows]
- [Citation to current source]
- [Why this matters to the user]

OR if no material issues:

✅ **VERIFICATION COMPLETE**
- Core claims verified against current sources
- No contradictions found between answer and current law
- Any updates or context worth noting: [list if relevant]

⚠️ **ADDED CONTEXT** (only if materially important)
- [Context]: [Why user should know this]
- [Relevant source or upcoming change]

📊 **SUMMARY**
- Accuracy: High/Medium/Low (based on material issues only)
- Main message is [correct/needs clarification]: [brief statement]
- Recommend: [specific action, if any]

STRICT RULE:
If you find NO material contradictions, NO missing critical context, NO upcoming changes that affect the answer:
REPORT: ✅ VERIFICATION COMPLETE

Don't add lengthy analysis if there are no issues to report. Brevity is good. Don't pontificate or speculate."""

        # Make request to Perplexity API
        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": "sonar-reasoning-pro",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "web_search_options": {
                "max_search_results": 15
            }
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_data = response.json()
        
        # Extract the message content
        message_content = response_data["choices"][0]["message"]["content"]
        
        # Remove <think>...</think> section using regex
        clean_content = re.sub(r'<think>.*?</think>\s*', '', message_content, flags=re.DOTALL)
        
        # Extract citations from search_results
        citations = []
        search_results = response_data.get("search_results", [])
        for result in search_results:
            citations.append({
                "title": result.get("title", "Source"),
                "url": result.get("url", "#"),
                "snippet": result.get("snippet", "")  # Include snippet for potential future use
            })
        
        # Extract usage information
        usage = response_data.get("usage", {})
        
        return {
            "report": clean_content.strip(),
            "citations": citations,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "finish_reason": response_data["choices"][0].get("finish_reason", "stop")
        }
        
    except Exception as e:
        print(f"Error in independent auditor: {e}")
        return {
            "report": f"⚠️ **AUDITOR ERROR:** {str(e)}\n\nPlease verify the information independently using official government sources.",
            "citations": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "finish_reason": "error"
        }

def format_audit_with_citations(audit_content: str, citations: List[Dict]) -> str:
    """Convert Perplexity citations to clickable links"""
    if not citations:
        return audit_content
    
    try:
        output = audit_content
        
        # Build citation links
        citation_links = []
        for i, cite in enumerate(citations, 1):
            title = cite.get('title', 'Source')
            url = cite.get('url', '#')
            
            # Create HTML link
            link = f'<a href="{url}" target="_blank" title="{title}" style="color: #0969da; text-decoration: none;">[{i}]</a>'
            
            # Replace citation marker in text
            citation_marker = f"[{i}]"
            output = output.replace(citation_marker, link)
            
            # Store for reference list
            citation_links.append(f"[{i}] {title}\n    {url}")
        
        # Add reference list at the end
        if citation_links:
            output += "\n\n**Sources:**\n" + "\n".join(citation_links)
        
        return output
        
    except Exception as e:
        print(f"Error formatting citations: {e}")
        return audit_content

def generate_legal_response(query: str, search_results: List[Dict[str, Any]], reasoning_effort: str = None) -> str:
    """Generate response using OpenAI with legal context"""
    try:
        # Use provided reasoning effort or classify if not provided
        if reasoning_effort is None:
            reasoning_effort = classify_reasoning_effort_with_gpt4o_mini(query)
        
        # Still calculate complexity score for display purposes
        complexity_score, score_breakdown = calculate_complexity_score(query)
        
        # Prepare context from search results
        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"\n--- Source {i} ---\n"
            context += f"Citation: {result['citation']}\n"
            context += f"Jurisdiction: {result['jurisdiction']}\n"
            context += f"Content: {result['text']}\n"
        
        # Prepare input for GPT-5 Responses API
        input_text = f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\n\nQuery: {query}\n\nRelevant Legal Sources:\n{context}"
        
        # Generate response using GPT-5 Responses API
        response = openai_client.responses.create(
            model="gpt-5",
            input=input_text,
            max_output_tokens=None,  # No token limit - track usage
            reasoning={"effort": reasoning_effort},
            text={"verbosity": "high"}
        )

        # Extract token usage information
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        reasoning_tokens = getattr(getattr(usage, 'output_tokens_details', None), 'reasoning_tokens', 0)
        total_tokens = usage.total_tokens
        
        # Log token usage to console
        print(f"Query: {query[:100]}...")
        print(f"Reasoning Effort: {reasoning_effort}")
        print(f"Input Tokens: {input_tokens:,}")
        print(f"Output Tokens: {output_tokens:,}")
        print(f"Reasoning Tokens: {reasoning_tokens:,}")
        print(f"Total Tokens: {total_tokens:,}")
        
        # Get the AI response text
        ai_response = response.output_text
        
        # Calculate estimated cost
        estimated_cost = calculate_estimated_cost(total_tokens, reasoning_effort)
        
        # Create detailed breakdown for transparency
        complexity_details = "\n".join([f"• {factor}: {score}" for factor, score in score_breakdown.items()])
        
        # Append token usage information to the response
        token_info = f"""

**Reasoning Effort:** {reasoning_effort.upper()} (Classified by GPT-4o-mini)

**Tokens Used:** {total_tokens:,}
**Token Breakdown:**
• Input: {input_tokens:,}
• Output: {output_tokens:,}
• Reasoning: {reasoning_tokens:,}

"""
        
        return ai_response + token_info
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"Error generating response: {e}")
        
        if "timeout" in error_msg:
            return "The query is taking longer than expected to process. Please try simplifying your question or try again later."
        else:
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
    st.title("PEO Compliance Assistant")
    st.markdown("### Access Required")
    
    # Create a simple login form
    with st.form("login_form"):
        st.write("Please enter your access code to continue:")
        access_code = st.text_input("Access Code", type="password")
        submit_button = st.form_submit_button("Access System")
        
        if submit_button:
            if access_code:
                if authenticate_user(access_code):
                    st.success("Access granted! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid access code. Please try again.")
            else:
                st.error("Please enter an access code.")
    
    st.info("Contact your administrator if you need an access code.")

def show_main_application():
    """Display main application interface"""

    # Optimized header for mobile-first design
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("<h1 style='margin-bottom: 4px;'>PEO Compliance Assistant</h1>", unsafe_allow_html=True)
        st.caption("Comprehensive employment law guidance for all 50 U.S. states and federal law")

    with header_col2:
        if st.button("Logout", key="logout_btn", help="Logout", use_container_width=True):
            logout_user()

    # Add visual separator
    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid var(--border-light);'>", unsafe_allow_html=True)

    # Show legal assistant content directly
    show_legal_assistant_content()

    # Handle chat input outside of tabs
    if prompt := st.chat_input("Ask about employment law in any U.S. state or federal law..."):
        handle_chat_input(prompt)

def show_legal_assistant_content():
    """Display legal assistant chat interface content (without chat input)"""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show welcome message if no messages
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; color: var(--text-muted);'>
            <h3>Welcome to PEO Compliance Assistant</h3>
            <p>Ask me about employment law in any U.S. state or federal law</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat messages with avatars
        for message in st.session_state.messages:
            avatar = "USER" if message["role"] == "user" else "AI"
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
        # Create status placeholder for real-time updates
        status_placeholder = st.empty()
        
        with st.spinner("Processing your legal query..."):
            # Step 1: Classify reasoning effort
            status_placeholder.info("🤔 Choosing reasoning effort...")
            reasoning_effort = classify_reasoning_effort_with_gpt4o_mini(prompt)
            
            # Show determined effort level
            effort_emoji = {"medium": "🧠", "high": "🔬"}
            status_placeholder.success(f"{effort_emoji.get(reasoning_effort, '🧠')} Reasoning effort: **{reasoning_effort.upper()}**")
            
            # Brief pause to let user see the effort level
            import time
            time.sleep(0.5)
            
            # Step 2: Search legal database
            status_placeholder.info("🔍 Searching legal database...")
            search_results = search_legal_database(prompt)
            
            # Show search results count
            if search_results:
                status_placeholder.success(f"📚 Found {len(search_results)} relevant legal sources")
            else:
                status_placeholder.warning("📚 No specific legal sources found - using general knowledge")
            
            time.sleep(0.5)
            
            # Step 3: Generate response with dynamic status based on effort
            if reasoning_effort == "medium":
                status_placeholder.info("🧠 Analyzing with medium reasoning effort...")
            else:
                status_placeholder.info("🔬 Performing deep analysis with high reasoning effort... This may take 3-10 minutes.")
            
            response = generate_legal_response(prompt, search_results, reasoning_effort)
            
            # Clear status placeholder before showing final response
            status_placeholder.empty()
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Step 4: Independent Fact-Checking (separate message)
    with st.chat_message("assistant"):
        st.markdown("### 🕵️ Independent Fact-Check Report")
        st.markdown("*Audit performed by Sonar Reasoning Pro for objectivity*")
        
        # Create status placeholder for auditor
        audit_status = st.empty()
        
        try:
            with st.spinner("Fact-checking with web search..."):
                audit_status.info("🌐 Sonar Reasoning Pro searching current legal sources...")
                
                # Call independent auditor
                audit_result = call_perplexity_auditor(prompt, response)
                
                audit_status.info("📋 Generating audit report by Sonar Reasoning Pro...")
                time.sleep(0.5)
                
                # Format audit report with citations
                formatted_audit = format_audit_with_citations(
                    audit_result["report"], 
                    audit_result["citations"]
                )
                
                # Extract usage information
                usage = audit_result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Add token usage information
                if total_tokens > 0:
                    token_info = f"""

**Tokens Used:** {total_tokens:,}
**Token Breakdown:**
• Input: {prompt_tokens:,}
• Output: {completion_tokens:,}
"""
                    formatted_audit += token_info
                
                # Clear status and show results
                audit_status.empty()
                
                # Display formatted audit report
                st.markdown(formatted_audit, unsafe_allow_html=True)
                
                # Show audit metadata
                if audit_result["citations"]:
                    st.caption(f"✅ Verified by Sonar Reasoning Pro against {len(audit_result['citations'])} authoritative sources")
                else:
                    st.caption("⚠️ Sonar Reasoning Pro: No web sources found - manual verification recommended")
        
        except Exception as e:
            audit_status.empty()
            st.error(f"⚠️ **Fact-checking unavailable:** {str(e)}")
            st.info("Please verify the information independently using official government sources.")
    
    # Add audit report to chat history as separate message
    audit_content = f"### 🕵️ Independent Fact-Check Report\n\n{audit_result.get('report', 'Fact-checking unavailable')}"
    st.session_state.messages.append({"role": "assistant", "content": audit_content})

if __name__ == "__main__":
    main()
