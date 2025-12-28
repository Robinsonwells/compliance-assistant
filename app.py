import streamlit as st
import os
from dotenv import load_dotenv
import uuid
import requests
import re
from typing import List, Dict, Any, Generator, Tuple, Optional
import time

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


def get_error_suggestions(error_type: str, error_message: str) -> str:
    """Return troubleshooting suggestions based on error type."""
    suggestions = {
        "AuthenticationError": "Check that your OPENAI_API_KEY is valid and has not expired.",
        "InvalidRequestError": "The model name or parameters may be incorrect. Verify 'gpt-5' is available on your account.",
        "BadRequestError": "The API rejected the request. This may indicate incompatible parameters (e.g., background + stream).",
        "RateLimitError": "Too many requests. Wait a moment and try again.",
        "APIConnectionError": "Network issue connecting to OpenAI. Check your internet connection.",
        "Timeout": "The request took too long. The model may be overloaded.",
        "APIError": "OpenAI API returned an error. This may be temporary.",
    }

    for key, suggestion in suggestions.items():
        if key.lower() in error_type.lower():
            return suggestion

    if "timeout" in error_message.lower():
        return suggestions["Timeout"]
    if "rate" in error_message.lower() and "limit" in error_message.lower():
        return suggestions["RateLimitError"]

    return "An unexpected error occurred. Check the error details below."


def display_streaming_error(
    exception: Exception,
    elapsed_time: float,
    placeholder
) -> dict:
    """Display a detailed error card in the Streamlit UI for debugging streaming failures."""
    error_type = type(exception).__name__
    error_message = str(exception)
    error_repr = repr(exception)

    status_code = getattr(exception, 'status_code', None)
    response_body = None
    if hasattr(exception, 'response'):
        try:
            response_body = str(exception.response)
        except:
            pass

    suggestion = get_error_suggestions(error_type, error_message)

    error_details = {
        "type": error_type,
        "message": error_message,
        "elapsed": elapsed_time,
        "status_code": status_code,
        "suggestion": suggestion
    }

    with placeholder.container():
        st.error(f"Streaming Failed: {error_type}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Error Type:** `{error_type}`")
            if status_code:
                st.markdown(f"**HTTP Status:** `{status_code}`")
        with col2:
            st.markdown(f"**Elapsed Time:** `{elapsed_time:.1f}s`")

        st.markdown(f"**Suggestion:** {suggestion}")

        with st.expander("Full Error Details (click to expand)", expanded=True):
            st.markdown("**Error Message:**")
            st.code(error_message, language=None)

            if response_body:
                st.markdown("**API Response:**")
                st.code(response_body, language=None)

            st.markdown("**Copy-Paste Debug Info:**")
            debug_text = f"""STREAMING ERROR REPORT
Type: {error_type}
Message: {error_message}
Status Code: {status_code}
Elapsed: {elapsed_time:.1f}s
Full Repr: {error_repr}"""
            st.code(debug_text, language=None)

        st.info("Attempting fallback to non-streaming mode...")

    return error_details


# Import custom modules
from user_management import UserManager
from settings_manager import SettingsManager
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT
from chat_logger import ChatLogger
from background_response_manager import BackgroundResponseManager

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
        timeout=60.0,  # 60 second timeout to prevent read timeouts
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

    # Initialize chat logger
    chat_logger = ChatLogger()

    # Initialize background response manager
    background_manager = BackgroundResponseManager()

    # Initialize settings manager with production-grade fallback
    try:
        settings_manager = SettingsManager()
        settings_manager.initialize_default_settings()
        print("✅ Settings manager initialized successfully")

        # Verify settings are accessible
        test_value = settings_manager.get_setting('show_rag_chunks', 'false')
        print(f"✅ Settings verified: show_rag_chunks = {test_value}")

    except Exception as e:
        print(f"❌ Settings manager initialization failed: {e}")
        print("⚠️ Using fallback settings manager - all settings will use safe defaults")

        # Production-grade fallback with logging
        class FallbackSettingsManager:
            def __init__(self):
                self.defaults = {
                    'show_rag_chunks': 'false'
                }
                self.warned_keys = set()

            def get_setting(self, key: str, default: str = 'true') -> str:
                """Always returns safe default, logs once per key"""
                value = self.defaults.get(key, default)

                if key not in self.warned_keys:
                    print(f"⚠️ Fallback mode: {key} = {value}")
                    self.warned_keys.add(key)

                return value

            def clear_cache(self):
                pass  # No-op in fallback mode

            def initialize_default_settings(self):
                pass  # No-op in fallback mode

        settings_manager = FallbackSettingsManager()

except Exception as e:
    st.error(f"Failed to initialize components: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Legal Assistant",
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
            limit=50,
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

1. MORE RECENT, CONTRADICTORY information (not just missing details)

DO NOT flag:
- Hypothetical "what if" scenarios

IMPORTANT CONTEXT:
This is normal and acceptable. Only flag if you find a MORE RECENT source that contradicts the core claim.

PRIORITIZE THESE OFFICIAL SOURCES FOR CURRENT INFORMATION:
- State legislature websites (.gov domains) - most current
- Cornell Legal Information Institute (law.cornell.edu) - authoritative summaries
- Federal regulations (ecfr.gov, regulations.gov) - official versions
- State Department of Labor websites - current enforcement info
- Official case law databases - recent rulings
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
- Do NOT introduce irrelevant law (e.g., federal law if query is state-specific)
- Do NOT assume information is wrong just because you can't verify it
- Do NOT require citations for every fact (as long as it's stated correctly)
🚨 **MATERIAL ISSUES FOUND** (only if actual contradictions or critical context)
- [Issue]: [What the answer stated] vs [What current law shows]
OR if no material issues:

✅ **VERIFICATION COMPLETE**
- Core claims verified against current sources
⚠️ **ADDED CONTEXT** (only if materially important)
- [Context]: [Why user should know this]

- Recommend: [specific action, if any]
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

def generate_legal_response(query: str, search_results: List[Dict[str, Any]], selected_model: str, reasoning_effort: str) -> str:
    """Generate response using OpenAI with legal context"""
    try:
        # Prepare context from search results
        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"\n--- Source {i} ---\n"
            context += f"Citation: {result['citation']}\n"
            context += f"Jurisdiction: {result['jurisdiction']}\n"
            context += f"Content: {result['text']}\n"
        
        if selected_model == "GPT-5":
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
            
            # Get the AI response text
            ai_response = response.output_text
            
        else:  # Sonar Reasoning Pro (RAG Only)
            # Prepare messages for Perplexity API
            messages = [
                {"role": "system", "content": LEGAL_COMPLIANCE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nRelevant Legal Sources:\n{context}"}
            ]
            
            # Make request to Perplexity API with search disabled
            url = "https://api.perplexity.ai/chat/completions"
            payload = {
                "model": "sonar-reasoning-pro",
                "messages": messages,
                "disable_search": True,  # Disable web search - use only provided context
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract the message content
            ai_response = response_data["choices"][0]["message"]["content"]
            
            # Extract usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            reasoning_tokens = 0  # Sonar Reasoning Pro doesn't separate reasoning tokens
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        # Log token usage to console
        print(f"Model: {selected_model}")
        print(f"Query: {query[:100]}...")
        print(f"Reasoning Effort: {reasoning_effort}")
        print(f"Input Tokens: {input_tokens:,}")
        print(f"Output Tokens: {output_tokens:,}")
        print(f"Reasoning Tokens: {reasoning_tokens:,}")
        print(f"Total Tokens: {total_tokens:,}")
        
        # Calculate estimated cost (using GPT-5 pricing as baseline)
        estimated_cost = calculate_estimated_cost(total_tokens, reasoning_effort)
        
        return ai_response, total_tokens, estimated_cost, input_tokens, output_tokens, reasoning_tokens
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"Error generating response with {selected_model}: {e}")
        
        if "timeout" in error_msg:
            return "The query is taking longer than expected to process. Please try simplifying your question or try again later.", 0, 0.0, 0, 0, 0
        else:
            st.error(f"Error generating response with {selected_model}: {e}")


def generate_legal_response_streaming(
    query: str,
    search_results: List[Dict[str, Any]],
    reasoning_effort: str,
    access_code: str,
    session_id: str,
    status_callback=None,
    text_callback=None
) -> Tuple[str, int, float, int, int, int, Optional[str]]:
    """
    Generate GPT-5 response using background mode with streaming for progressive display.
    Returns: (response_text, total_tokens, cost, input_tokens, output_tokens, reasoning_tokens, response_id)
    """
    context = ""
    for i, result in enumerate(search_results, 1):
        context += f"\n--- Source {i} ---\n"
        context += f"Citation: {result['citation']}\n"
        context += f"Jurisdiction: {result['jurisdiction']}\n"
        context += f"Content: {result['text']}\n"

    input_text = f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\n\nQuery: {query}\n\nRelevant Legal Sources:\n{context}"

    start_time = time.time()
    first_token_time = None
    accumulated_text = ""
    sequence_number = 0
    response_id = None

    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0

    try:
        stream = openai_client.responses.create(
            model="gpt-5",
            input=input_text,
            max_output_tokens=None,
            reasoning={"effort": reasoning_effort},
            text={"verbosity": "high"},
            background=True,
            stream=True,
            store=True
        )

        last_event_time = time.time()
        last_save_time = time.time()
        chunk_count = 0
        pending_response_created = False

        for event in stream:
            current_time = time.time()
            elapsed = current_time - start_time
            event_type = event.type

            if event_type == "response.created":
                response_id = event.response.id
                print(f"[STREAMING] Started background response: {response_id}")

                background_manager.create_pending_response(
                    response_id=response_id,
                    access_code=access_code,
                    session_id=session_id,
                    user_query=query,
                    search_results=search_results,
                    reasoning_effort=reasoning_effort
                )
                background_manager.update_status(response_id, "in_progress")
                pending_response_created = True

            elif event_type == "response.output_text.delta":
                if first_token_time is None:
                    first_token_time = elapsed
                    ttft = first_token_time
                    background_manager.set_time_to_first_token(response_id, ttft)
                    print(f"[STREAMING] First token received at {ttft:.2f}s")

                chunk = event.delta
                accumulated_text += chunk
                sequence_number += 1
                chunk_count += 1
                last_event_time = current_time

                if text_callback:
                    text_callback(accumulated_text)

                if current_time - last_save_time > 10:
                    background_manager.update_partial_response(
                        response_id, accumulated_text, sequence_number
                    )
                    last_save_time = current_time

            elif event_type == "response.output_text.done":
                print(f"[STREAMING] Text output complete: {len(accumulated_text)} chars")

            elif event_type == "response.completed":
                usage = event.response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                total_tokens = usage.total_tokens

                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)

                elapsed = time.time() - start_time
                print(f"[STREAMING] Complete in {elapsed:.1f}s")
                print(f"[STREAMING] Tokens - Input: {input_tokens}, Output: {output_tokens}, Reasoning: {reasoning_tokens}, Total: {total_tokens}")

                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens
                }
                background_manager.mark_completed(response_id, accumulated_text, token_usage)

            elif event_type == "response.incomplete":
                reason = getattr(event.response.incomplete_details, 'reason', 'unknown')
                print(f"[STREAMING] Response incomplete: {reason}")
                if status_callback:
                    status_callback(f"Response incomplete: {reason}")

            elif event_type == "response.failed":
                error_msg = getattr(event.response.error, 'message', 'Unknown error')
                print(f"[STREAMING] Response failed: {error_msg}")
                background_manager.mark_failed(response_id, error_msg)
                raise Exception(f"OpenAI response failed: {error_msg}")

            time_since_last = current_time - last_event_time
            if first_token_time is None and time_since_last > 30:
                if status_callback:
                    if time_since_last > 90:
                        status_callback("Extended reasoning in progress... Your response is saved and safe to refresh.")
                    elif time_since_last > 60:
                        status_callback("Deep reasoning in progress... This query requires extensive analysis.")
                    else:
                        status_callback("GPT-5 is thinking deeply...")

        estimated_cost = calculate_estimated_cost(total_tokens, reasoning_effort)
        return accumulated_text, total_tokens, estimated_cost, input_tokens, output_tokens, reasoning_tokens, response_id

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[STREAMING] Error after {elapsed:.1f}s: {error_msg}")

        if response_id:
            background_manager.mark_failed(response_id, error_msg)

        if "timeout" in error_msg.lower() or elapsed > 90:
            if response_id and accumulated_text:
                background_manager.update_partial_response(response_id, accumulated_text, sequence_number)
                return accumulated_text + "\n\n[Response may be incomplete due to timeout]", 0, 0.0, 0, 0, 0, response_id

        raise


def get_polling_status_message(elapsed_seconds: float) -> str:
    """Return context-aware status message based on elapsed time."""
    if elapsed_seconds < 5:
        return "Processing your query..."
    elif elapsed_seconds < 30:
        return f"Analyzing... ({int(elapsed_seconds)}s)"
    elif elapsed_seconds < 60:
        return f"Researching legal sources... ({int(elapsed_seconds)}s)"
    elif elapsed_seconds < 180:
        return f"Deep reasoning in progress... ({int(elapsed_seconds)}s)\nYour response is being prepared."
    else:
        return f"Processing extended analysis... ({int(elapsed_seconds)}s)\nComplex reasoning still in progress."


def generate_legal_response_polling(
    query: str,
    search_results: List[Dict[str, Any]],
    reasoning_effort: str,
    access_code: str,
    session_id: str,
    status_callback=None,
    text_callback=None
) -> Tuple[str, int, float, int, int, int, Optional[str]]:
    """
    Generate GPT-5 response using polling for complex/long-running queries.
    More reliable than streaming for extended reasoning tasks.
    Returns: (response_text, total_tokens, cost, input_tokens, output_tokens, reasoning_tokens, response_id)
    """
    context = ""
    for i, result in enumerate(search_results, 1):
        context += f"\n--- Source {i} ---\n"
        context += f"Citation: {result['citation']}\n"
        context += f"Jurisdiction: {result['jurisdiction']}\n"
        context += f"Content: {result['text']}\n"

    input_text = f"{LEGAL_COMPLIANCE_SYSTEM_PROMPT}\n\nQuery: {query}\n\nRelevant Legal Sources:\n{context}"

    start_time = time.time()
    response_id = None
    polling_interval = 2.0

    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0

    try:
        response = openai_client.responses.create(
            model="gpt-5",
            input=input_text,
            max_output_tokens=None,
            reasoning={"effort": reasoning_effort},
            text={"verbosity": "high"},
            background=True,
            stream=False,
            store=True
        )

        response_id = response.id
        print(f"[POLLING] Started background response: {response_id}")

        background_manager.create_pending_response(
            response_id=response_id,
            access_code=access_code,
            session_id=session_id,
            user_query=query,
            search_results=search_results,
            reasoning_effort=reasoning_effort
        )
        background_manager.update_status(response_id, "in_progress")

        while True:
            elapsed = time.time() - start_time

            if status_callback:
                status_msg = get_polling_status_message(elapsed)
                status_callback(status_msg)

            result = openai_client.responses.retrieve(response_id)

            if result.status == "completed":
                usage = result.usage
                input_tokens = usage.input_tokens if usage else 0
                output_tokens = usage.output_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0

                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)

                response_text = result.output_text
                elapsed = time.time() - start_time
                print(f"[POLLING] Complete in {elapsed:.1f}s")
                print(f"[POLLING] Tokens - Input: {input_tokens}, Output: {output_tokens}, Reasoning: {reasoning_tokens}, Total: {total_tokens}")

                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens
                }
                background_manager.mark_completed(response_id, response_text, token_usage)

                if text_callback:
                    text_callback(response_text)

                estimated_cost = calculate_estimated_cost(total_tokens, reasoning_effort)
                return response_text, total_tokens, estimated_cost, input_tokens, output_tokens, reasoning_tokens, response_id

            elif result.status == "failed":
                error_msg = getattr(result.error, 'message', 'Unknown error') if result.error else 'Unknown error'
                print(f"[POLLING] Response failed: {error_msg}")
                background_manager.mark_failed(response_id, error_msg)
                raise Exception(f"OpenAI response failed: {error_msg}")

            elif result.status in ["queued", "in_progress"]:
                time.sleep(polling_interval)

            else:
                print(f"[POLLING] Unknown status: {result.status}")
                time.sleep(polling_interval)

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"[POLLING] Error after {elapsed:.1f}s: {error_msg}")

        if response_id:
            background_manager.mark_failed(response_id, error_msg)

        raise


def generate_legal_response_smart(
    query: str,
    search_results: List[Dict[str, Any]],
    reasoning_effort: str,
    access_code: str,
    session_id: str,
    status_callback=None,
    text_callback=None
) -> Tuple[str, int, float, int, int, int, Optional[str]]:
    """
    Smart router that chooses between streaming and polling based on query complexity.
    - Simple queries (complexity < 15): Try streaming first, fall back to polling
    - Complex queries (complexity >= 15): Use polling directly for reliability
    """
    complexity_score, _ = calculate_complexity_score(query)
    complexity_threshold = 15

    print(f"[SMART] Query complexity: {complexity_score} (threshold: {complexity_threshold})")

    if complexity_score < complexity_threshold:
        print(f"[SMART] Using streaming for simple query")
        try:
            return generate_legal_response_streaming(
                query=query,
                search_results=search_results,
                reasoning_effort=reasoning_effort,
                access_code=access_code,
                session_id=session_id,
                status_callback=status_callback,
                text_callback=text_callback
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "stream" in error_msg:
                print(f"[SMART] Streaming failed, falling back to polling: {e}")
                if status_callback:
                    status_callback("Switching to background processing...")
                return generate_legal_response_polling(
                    query=query,
                    search_results=search_results,
                    reasoning_effort=reasoning_effort,
                    access_code=access_code,
                    session_id=session_id,
                    status_callback=status_callback,
                    text_callback=text_callback
                )
            raise
    else:
        print(f"[SMART] Using polling for complex query")
        if status_callback:
            status_callback("Complex query detected - using reliable background processing...")
        return generate_legal_response_polling(
            query=query,
            search_results=search_results,
            reasoning_effort=reasoning_effort,
            access_code=access_code,
            session_id=session_id,
            status_callback=status_callback,
            text_callback=text_callback
        )


def check_and_recover_pending_response(session_id: str) -> Optional[Dict[str, Any]]:
    """Check for any pending or recently completed responses for this session"""
    active = background_manager.get_active_response_for_session(session_id)
    if active:
        return active

    recent = background_manager.get_recently_completed_response(session_id, minutes=5)
    if recent:
        return recent

    return None


def retrieve_background_response(response_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a background response by its ID"""
    try:
        response = openai_client.responses.retrieve(response_id)

        if response.status == "completed":
            usage = response.usage
            return {
                "status": "completed",
                "text": response.output_text,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "reasoning_tokens": getattr(getattr(usage, 'output_tokens_details', None), 'reasoning_tokens', 0) if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
        elif response.status == "failed":
            return {
                "status": "failed",
                "error": getattr(response.error, 'message', 'Unknown error') if response.error else 'Unknown error'
            }
        else:
            return {
                "status": response.status,
                "text": None
            }
    except Exception as e:
        print(f"Error retrieving response {response_id}: {e}")
        return None


# Authentication functions
def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # If not authenticated but we have session data, try to restore authentication
    if (not st.session_state.authenticated and 
        'access_code' in st.session_state and 
        'session_id' in st.session_state):
        
        # Check if the existing session is still valid
        if user_manager.is_session_valid(st.session_state.session_id):
            # Restore authentication
            st.session_state.authenticated = True
            # Update session activity for tracking purposes
            user_manager.update_session_activity(st.session_state.session_id)
        else:
            # Session/access code is no longer valid, clear everything
            logout_user()
    
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
        st.error("Your access code has expired or is no longer valid. Please log in again.")
        logout_user()
        return
    
    # Update session activity
    user_manager.update_session_activity(st.session_state.session_id)

    # Initialize settings cache in session state (once per session only)
    if 'settings_cache' not in st.session_state:
        st.session_state.settings_cache = {
            'show_rag_chunks_enabled': False,
            'default_reasoning_effort': 'automatic',
            'initialized': False,
            'load_timestamp': None,
            'error_count': 0
        }

    # Load settings from database ONCE per session (not on every interaction)
    if not st.session_state.settings_cache['initialized']:
        try:
            raw_value = settings_manager.get_setting('show_rag_chunks', 'false')
            st.session_state.settings_cache['show_rag_chunks_enabled'] = (raw_value == 'true')

            effort_value = settings_manager.get_enum_setting(
                'default_reasoning_effort',
                'automatic',
                ['automatic', 'medium', 'high']
            )
            st.session_state.settings_cache['default_reasoning_effort'] = effort_value

            st.session_state.settings_cache['initialized'] = True
            st.session_state.settings_cache['load_timestamp'] = None
            print(f"Settings loaded: show_rag_chunks = {raw_value}, default_reasoning_effort = {effort_value}")
        except Exception as e:
            st.session_state.settings_cache['error_count'] += 1
            print(f"Error loading settings (attempt {st.session_state.settings_cache['error_count']}): {e}")
            if st.session_state.settings_cache['error_count'] >= 3:
                st.session_state.settings_cache['initialized'] = True
                print("Max retries reached, using defaults")

    # Show main application
    show_main_application()

def show_login_page():
    """Display login page"""
    # Initialize contact info modal state
    if 'show_contact_info_modal' not in st.session_state:
        st.session_state.show_contact_info_modal = False
    
    # Initialize contact info modal state
    if 'show_contact_info_modal' not in st.session_state:
        st.session_state.show_contact_info_modal = False
    
    st.title("Legal Assistant")
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
    
    # Contact info section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Need an access code?**")
    with col2:
        if st.button("Contact Info", key="contact_info_btn", use_container_width=True):
            st.session_state.show_contact_info_modal = True
    
    # Contact info modal
    if st.session_state.show_contact_info_modal:
        st.markdown("---")
        with st.container():
            st.markdown("### 📞 Contact Information")
            st.markdown("**Robinson Wells**")
            st.markdown("📱 Phone: [(208) 631-4918](tel:+12086314918)")
            st.markdown("💼 LinkedIn: [Robinson Wells](https://www.linkedin.com/in/robinson-wells-b22634237/)")
            st.markdown("📧 Email: [robinson.wells@instantlegalai.org](mailto:robinson.wells@instantlegalai.org)")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Close", key="close_contact_info"):
                    st.session_state.show_contact_info_modal = False
                    st.rerun()

def show_main_application():
    """Display main application interface"""

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "GPT-5"

    if 'sonar_reasoning_effort_override' not in st.session_state:
        st.session_state.sonar_reasoning_effort_override = "Auto"

    if 'recovery_checked' not in st.session_state:
        st.session_state.recovery_checked = False

    if not st.session_state.recovery_checked and 'session_id' in st.session_state:
        st.session_state.recovery_checked = True
        try:
            pending = check_and_recover_pending_response(st.session_state.session_id)
            if pending:
                if pending.get('status') in ['queued', 'in_progress', 'streaming']:
                    st.info(f"📡 **Previous query still processing:** \"{pending.get('user_query', 'Unknown')[:100]}...\"")

                    response_id = pending.get('response_id')
                    if response_id:
                        result = retrieve_background_response(response_id)
                        if result and result.get('status') == 'completed':
                            st.success("Your previous query has completed!")
                            with st.expander("View recovered response", expanded=True):
                                st.markdown(result.get('text', 'No content available'))
                                st.caption(f"Tokens: {result.get('total_tokens', 0):,}")

                            if "messages" not in st.session_state:
                                st.session_state.messages = []
                            st.session_state.messages.append({
                                "role": "user",
                                "content": pending.get('user_query', '')
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result.get('text', '')
                            })
                        elif result and result.get('status') == 'failed':
                            st.error(f"Previous query failed: {result.get('error', 'Unknown error')}")
                        else:
                            st.warning("Query is still processing. Check back in a moment.")

                elif pending.get('status') == 'completed':
                    final_response = pending.get('final_response')
                    if final_response and final_response not in str(st.session_state.get('messages', [])):
                        st.success("Found a recently completed response!")
                        with st.expander("View recovered response", expanded=False):
                            st.markdown(final_response)

        except Exception as e:
            print(f"Recovery check failed: {e}")

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("<h1 style='margin-bottom: 4px;'>Compliance Assistant</h1>", unsafe_allow_html=True)

        # Set GPT-5 as the only model
        st.session_state.selected_model = "GPT-5"

    with header_col2:
        if st.button("Logout", key="logout_btn", help="Logout", use_container_width=True):
            logout_user()

    # Add visual separator
    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid var(--border-light);'>", unsafe_allow_html=True)

    # Reasoning Effort Selector - visible on main chat UI
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**AI Reasoning Effort:**")
    with col2:
        # Initialize user effort preference in session state
        if 'user_effort_preference' not in st.session_state:
            st.session_state.user_effort_preference = 'automatic'

        # Get admin default for display
        admin_default = st.session_state.settings_cache.get('default_reasoning_effort', 'automatic')

        # Create selectbox with clear labels
        effort_options = {
            'automatic': f'Automatic (Admin: {admin_default.title()})',
            'medium': 'Medium',
            'high': 'High'
        }

        selected_effort = st.selectbox(
            "Select Effort",
            options=list(effort_options.keys()),
            format_func=lambda x: effort_options[x],
            key='effort_selector',
            label_visibility='collapsed'
        )

        # Update session state
        st.session_state.user_effort_preference = selected_effort

    # Add brief explanation
    st.caption("🧠 **Medium:** Balanced reasoning • 🔬 **High:** Deep analysis (slower, more thorough)")

    # Add visual separator
    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid var(--border-light);'>", unsafe_allow_html=True)

    # Debug sidebar for production troubleshooting
    if st.sidebar.checkbox("🔧 Debug Mode", key="debug_mode_toggle"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Settings Debug")

        # Show cache state
        cache_state = st.session_state.get('settings_cache', {})
        st.sidebar.write("**Cache State:**")
        st.sidebar.json(cache_state)

        # Show cache age
        if cache_state.get('load_timestamp'):
            age = time.time() - cache_state['load_timestamp']
            st.sidebar.write(f"**Cache Age:** {age:.1f}s")

        # Show current effective value
        show_chunks = cache_state.get('show_rag_chunks_enabled', True)
        st.sidebar.write(f"**Effective Setting:** show_rag_chunks = {show_chunks}")

        # Manual refresh button
        if st.sidebar.button("🔄 Force Reload Settings"):
            st.session_state.settings_cache = {
                'show_rag_chunks_enabled': True,
                'initialized': False,
                'load_timestamp': None,
                'error_count': 0
            }
            st.sidebar.success("Cache cleared - will reload on next interaction")
            st.rerun()

        # Direct DB query button
        if st.sidebar.button("📊 Query Database"):
            try:
                db_value = settings_manager.get_setting('show_rag_chunks', 'false')
                st.sidebar.success(f"DB value: {db_value}")
            except Exception as e:
                st.sidebar.error(f"DB error: {e}")

    # Show legal assistant content directly
    show_legal_assistant_content()

    # Handle chat input outside of tabs
    if prompt := st.chat_input("Ask any compliance question"):
        handle_chat_input(prompt)

def show_legal_assistant_content():
    """Display legal assistant chat interface content (without chat input)"""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show welcome message if no messages
    if len(st.session_state.messages) == 0:
        pass
    else:
        # Display chat messages with avatars
        for message in st.session_state.messages:
            avatar = "USER" if message["role"] == "user" else "AI"
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Display retrieved chunks in expander for assistant responses
                # Check if showing chunks is enabled in settings (use cached setting from session state)
                show_chunks_enabled = st.session_state.settings_cache.get('show_rag_chunks_enabled', False)

                if show_chunks_enabled and message["role"] == "assistant" and message.get("chunks"):
                    chunks = message["chunks"]

                    # Calculate average relevance score
                    avg_score = sum(chunk.get('score', 0) for chunk in chunks) / len(chunks) if chunks else 0

                    with st.expander(f"📚 View {len(chunks)} Retrieved Legal Sources (Avg Relevance: {avg_score:.1%})", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            # Display chunk citation as header
                            st.markdown(f"### Source {i}: {chunk.get('citation', 'Unknown')}")

                            # Display relevance score with visual indicator
                            score = chunk.get('score', 0)
                            score_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🟠"
                            st.markdown(f"{score_color} **Relevance Score:** {score:.1%}")

                            # Display metadata
                            st.markdown(f"**Jurisdiction:** {chunk.get('jurisdiction', 'Unknown')} | **Section:** {chunk.get('section_number', 'Unknown')}")

                            if chunk.get('source_file'):
                                st.markdown(f"**Source File:** {chunk.get('source_file')}")

                            # Display chunk text content
                            st.markdown("**Content:**")
                            st.markdown(f"> {chunk.get('text', 'No content available')}")

                            # Add separator between chunks
                            if i < len(chunks):
                                st.divider()

def handle_chat_input(prompt):
    """Handle chat input and generate response with streaming support"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    selected_model = st.session_state.get('selected_model', 'GPT-5')
    access_code = st.session_state.get('access_code', 'unknown')

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        text_placeholder = st.empty()
        elapsed_placeholder = st.empty()

        user_effort_preference = st.session_state.get('user_effort_preference', 'automatic')
        reasoning_effort_source = "User"

        if user_effort_preference == 'automatic':
            reasoning_effort_source = "Auto"
            if selected_model == "GPT-5":
                status_placeholder.info("Choosing reasoning effort automatically...")
                reasoning_effort = classify_reasoning_effort_with_gpt4o_mini(prompt)
            else:
                status_placeholder.info("Calculating complexity score...")
                complexity_score, _ = calculate_complexity_score(prompt)
                reasoning_effort = get_reasoning_effort(complexity_score)
        else:
            reasoning_effort = user_effort_preference
            status_placeholder.info(f"Using your selected reasoning effort: {user_effort_preference}...")

        effort_emoji = {"medium": "🧠", "high": "🔬"}
        model_emoji = {"GPT-5": "🤖", "Sonar Reasoning Pro (RAG Only)": "🧠"}
        effort_display = f"{reasoning_effort.upper()} ({reasoning_effort_source})"
        status_placeholder.success(f"{model_emoji.get(selected_model, '🤖')} Model: **{selected_model}** | {effort_emoji.get(reasoning_effort, '🧠')} Effort: **{effort_display}**")

        time.sleep(0.5)

        status_placeholder.info("🔍 Searching legal database...")
        search_results = search_legal_database(prompt)

        if search_results:
            search_results = sorted(search_results, key=lambda x: x.get('score', 0), reverse=True)

        if search_results:
            status_placeholder.success(f"📚 Found {len(search_results)} relevant legal sources")
        else:
            status_placeholder.warning("📚 No specific legal sources found - using general knowledge")

        time.sleep(0.5)

        if selected_model == "GPT-5":
            complexity_score, _ = calculate_complexity_score(prompt)
            use_polling = complexity_score >= 15

            if use_polling:
                status_placeholder.info(f"Complex query detected (score: {complexity_score}) - using reliable background processing...")
            elif reasoning_effort == "medium":
                status_placeholder.info("GPT-5 analyzing with streaming (medium effort)...")
            else:
                status_placeholder.info("GPT-5 performing deep analysis (high effort)... Text will appear as it's generated.")

            start_time = time.time()
            streaming_text = [""]

            def update_text(text):
                streaming_text[0] = text
                text_placeholder.markdown(text + " ▌")
                elapsed = time.time() - start_time
                elapsed_placeholder.caption(f"Elapsed: {elapsed:.0f}s")

            def update_status(msg):
                status_placeholder.info(msg)

            try:
                result = generate_legal_response_smart(
                    query=prompt,
                    search_results=search_results,
                    reasoning_effort=reasoning_effort,
                    access_code=access_code,
                    session_id=st.session_state.session_id,
                    status_callback=update_status,
                    text_callback=update_text
                )

                ai_response_text, total_tokens, estimated_cost, input_tokens, output_tokens, reasoning_tokens, response_id = result

                status_placeholder.empty()
                elapsed_placeholder.empty()
                text_placeholder.markdown(ai_response_text)

            except Exception as e:
                status_placeholder.empty()
                elapsed_placeholder.empty()
                elapsed_time = time.time() - start_time

                error_placeholder = st.empty()
                display_streaming_error(e, elapsed_time, error_placeholder)

                if streaming_text[0]:
                    ai_response_text = streaming_text[0] + "\n\n[Response interrupted - partial content shown]"
                    text_placeholder.markdown(ai_response_text)
                    total_tokens = 0
                    estimated_cost = 0.0
                    input_tokens = 0
                    output_tokens = 0
                    reasoning_tokens = 0
                else:
                    ai_response_text, total_tokens, estimated_cost, input_tokens, output_tokens, reasoning_tokens = generate_legal_response(
                        prompt, search_results, selected_model, reasoning_effort
                    )
                    text_placeholder.markdown(ai_response_text)

        else:
            if reasoning_effort == "medium":
                status_placeholder.info(f"Sonar Reasoning Pro analyzing legal sources with medium effort ({reasoning_effort_source})...")
            else:
                status_placeholder.info(f"Sonar Reasoning Pro performing deep analysis with high effort ({reasoning_effort_source})...")

            ai_response_text, total_tokens, estimated_cost, input_tokens, output_tokens, reasoning_tokens = generate_legal_response(
                prompt, search_results, selected_model, reasoning_effort
            )

            status_placeholder.empty()
            text_placeholder.markdown(ai_response_text)

        model_short = "GPT-5" if selected_model == "GPT-5" else "Sonar-RP"
        effort_with_source = f"{reasoning_effort.upper()} ({reasoning_effort_source})"
        if selected_model == "Sonar Reasoning Pro (RAG Only)":
            st.caption(f"Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {total_tokens:,} | Model: {model_short} | Effort: {effort_with_source} | *Cost estimate based on GPT-5 pricing")
        else:
            st.caption(f"Input: {input_tokens:,} | Output: {output_tokens:,} | Reasoning: {reasoning_tokens:,} | Total: {total_tokens:,} | Model: {model_short} | Effort: {effort_with_source}")

        show_chunks_enabled = st.session_state.settings_cache.get('show_rag_chunks_enabled', False)

        if show_chunks_enabled and search_results:
            avg_score = sum(chunk.get('score', 0) for chunk in search_results) / len(search_results)

            with st.expander(f"📚 View {len(search_results)} Retrieved Legal Sources (Avg Relevance: {avg_score:.1%})", expanded=False):
                for i, chunk in enumerate(search_results, 1):
                    st.markdown(f"### Source {i}: {chunk.get('citation', 'Unknown')}")

                    score = chunk.get('score', 0)
                    score_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🟠"
                    st.markdown(f"{score_color} **Relevance Score:** {score:.1%}")

                    st.markdown(f"**Jurisdiction:** {chunk.get('jurisdiction', 'Unknown')} | **Section:** {chunk.get('section_number', 'Unknown')}")

                    if chunk.get('source_file'):
                        st.markdown(f"**Source File:** {chunk.get('source_file')}")

                    st.markdown("**Content:**")
                    st.markdown(f"> {chunk.get('text', 'No content available')}")

                    if i < len(search_results):
                        st.divider()
    
    # Add assistant response to chat history with chunks (only if setting is enabled)
    # Use cached setting from session state (loaded once per session)
    show_chunks_enabled = st.session_state.settings_cache.get('show_rag_chunks_enabled', False)
    message_data = {
        "role": "assistant",
        "content": ai_response_text
    }
    # Only include chunks in message history if the setting is enabled
    if show_chunks_enabled:
        message_data["chunks"] = search_results if search_results else []

    st.session_state.messages.append(message_data)
    
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
                audit_result = call_perplexity_auditor(prompt, ai_response_text)
                
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
                
                # Log the chat interaction with error handling
                try:
                    chat_logger.log_chat(
                        access_code=st.session_state.access_code,
                        user_query=prompt,
                        gpt5_response=ai_response_text,
                        perplexity_response=audit_result["report"],
                        session_id=st.session_state.session_id,
                        reasoning_effort=reasoning_effort,
                        tokens_used=total_tokens,
                        cost_estimate=estimated_cost,
                        main_model_used=selected_model
                    )
                except Exception as e:
                    # Don't fail the user experience if logging fails
                    print(f"Warning: Failed to log chat: {e}")
                
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
