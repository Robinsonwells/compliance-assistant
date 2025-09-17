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
                user_manager, _, _, _ = init_systems()
                
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

def search_legal_database(query: str, qdrant_client, embedding_model, top_k: int = 5):
    """Search the legal database using semantic similarity"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Search Qdrant
        search_results = qdrant_client.search(
            collection_name="legal_regulations",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Format results
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
        st.error(f"Database search error: {e}")
        return []

def generate_legal_response(query: str, search_results: list, openai_client):
    """Generate legal response using OpenAI GPT-5 Responses API with search results"""
    try:
        # Prepare context from search results
        context = "\n\n".join([
            f"LEGAL TEXT {i+1}:\n\"{result['text']}\"\n(Source: {result['citation']})"
            for i, result in enumerate(search_results[:3])  # Use top 3 results
        ])
        
        # Debug information
        print(f"DEBUG: Processing query with {len(search_results)} sources")
        print(f"DEBUG: Context length: {len(context)} characters")
        
        # Prepare input for GPT-5 Responses API
        input_content = f"""SYSTEM INSTRUCTIONS:
{ENHANCED_LEGAL_COMPLIANCE_SYSTEM_PROMPT}

USER QUERY: {query}

AVAILABLE LEGAL CONTEXT:
{context}"""
        
        # Validate context length
        total_content_length = len(input_content)
        if total_content_length > 100000:  # Conservative limit
            print(f"WARNING: Content length {total_content_length} may exceed model limits")
        
        print("DEBUG: Making OpenAI GPT-5 Responses API call...")
        
        # Call OpenAI GPT-5 Responses API
        response = openai_client.responses.create(
            model="gpt-5",
            input=input_content,
            reasoning={
                "effort": "high"  # Use medium reasoning for legal analysis
            },
            text={
                "verbosity": "high"  # Use high verbosity for detailed legal explanations
            }
        )
        
        # Extract and validate response
        if response and hasattr(response, 'output_text'):
            content = response.output_text
            print(f"DEBUG: Received response of length: {len(content) if content else 0}")
            
            if content and content.strip():
                return {
                    "success": True,
                    "content": content,
                    "error": None,
                    "response_id": getattr(response, 'id', None)  # Store response ID for multi-turn conversations
                }
            else:
                print("DEBUG: Empty response content from GPT-5")
                return {
                    "success": False,
                    "content": None,
                    "error": "Empty response from GPT-5 Responses API"
                }
        else:
            print("DEBUG: Invalid response structure from GPT-5")
            return {
                "success": False,
                "content": None,
                "error": "Invalid response structure from GPT-5 Responses API"
            }
    
    except Exception as e:
        error_msg = f"GPT-5 API error: {e}"
        print(f"DEBUG: Exception in generate_legal_response: {error_msg}")
        return {
            "success": False,
            "content": None,
            "error": error_msg
        }

def main():
    """Main application logic"""
    # Initialize systems
    user_manager, embedding_model, qdrant_client, openai_client = init_systems()
    
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
    
    # Main application interface
    st.markdown("""
        <div class="dashboard-header">
            <h1 style="margin: 0; color: white;">⚖️ AI Legal Compliance Assistant</h1>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                Professional Employment Organization Legal Research Tool
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Quick Reference")
        st.markdown("""
        **AI Model:** GPT-5 (Responses API)
        - **Reasoning:** Medium effort for thorough legal analysis
        - **Verbosity:** High for detailed explanations
        
        **Supported Jurisdictions:**
        - New York (NY)
        - New Jersey (NJ) 
        - Connecticut (CT)
        - Federal Regulations
        
        **Query Examples:**
        - "Minimum wage requirements in NY"
        - "Overtime rules for healthcare workers"
        - "Multi-state payroll compliance"
        - "On-call pay differences between CT, NY, and NJ"
        """)
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    # Chat interface
    st.markdown("### 💬 Legal Research Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a legal compliance question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching legal database..."):
                search_results = search_legal_database(prompt, qdrant_client, embedding_model)
                
                if search_results:
                    # Generate AI response
                    st.info(f"🔍 Found {len(search_results)} relevant sources - Processing with GPT-5...")
                    
                    response = generate_legal_response(prompt, search_results, openai_client)
                    
                    if response["success"]:
                        ai_response = response["content"]
                        if ai_response and ai_response.strip():
                            st.success("✅ GPT-5 legal analysis generated successfully")
                            st.markdown(ai_response)
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        else:
                            error_msg = "Empty response from GPT-5. Please try again."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        error_msg = response.get("error", "Unknown GPT-5 API error")
                        st.error(f"🚨 AI Response Generation Failed: {error_msg}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                        
                        # Show debugging information
                        with st.expander("🔧 Debug Information"):
                            st.write(f"**Sources found:** {len(search_results)}")
                            st.write(f"**Model:** gpt-5 (Responses API)")
                            st.write(f"**Reasoning effort:** medium")
                            st.write(f"**Verbosity:** high")
                            st.write(f"**Error details:** {error_msg}")
                            if response.get("response_id"):
                                st.write(f"**Response ID:** {response['response_id']}")
                    
                    # Show sources
                    with st.expander("📚 Sources Referenced"):
                        for i, result in enumerate(search_results[:3], 1):
                            st.markdown(f"**Source {i}:** {result['citation']}")
                            st.markdown(f"*Relevance Score: {result['score']:.3f}*")
                            st.text_area(
                                f"Legal Text {i}",
                                result['text'][:500] + "..." if len(result['text']) > 500 else result['text'],
                                height=100,
                                key=f"source_{i}_{len(st.session_state.messages)}"
                            )
                else:
                    response = "I couldn't find relevant legal information in the database for your query. Please try rephrasing your question with more specific terms or contact legal counsel for assistance."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()