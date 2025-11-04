import streamlit as st
import os
from dotenv import load_dotenv
import uuid
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

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
    device = 'cpu'  # Force CPU usage to avoid meta tensor issues
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Ensure model is properly loaded on CPU
    embedding_model = embedding_model.to(device)
    
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
        # Define simple queries that don't need high reasoning effort
        simple_queries = ['test', 'hello', 'hi', 'help', 'what', 'how are you']
        query_lower = query.lower().strip()
        
        # Determine reasoning effort based on query complexity
        if query_lower in simple_queries or len(query.split()) < 5:
            reasoning_effort = "low"
        else:
            reasoning_effort = "medium"  # Use medium instead of high for better performance
        
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
        reasoning_tokens = usage.output_tokens_details.get('reasoning_tokens', 0) if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details else 0
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
        
        # Append token usage information to the response
        token_info = f"""

---
**Token Usage:**
- **Input Tokens:** {input_tokens:,} (query + context + system prompt)
- **Output Tokens:** {output_tokens:,} (reasoning + visible response)
- **Reasoning Tokens:** {reasoning_tokens:,} (internal AI reasoning)
- **Total Tokens:** {total_tokens:,}
- **Reasoning Effort:** {reasoning_effort}
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
        if st.button("↗", key="logout_btn", help="Logout", use_container_width=True):
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
            <h3>👋 Welcome to PEO Compliance Assistant</h3>
            <p>Ask me about employment law in any U.S. state or federal law</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat messages with avatars
        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "⚖️"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

def handle_chat_input(prompt):
    """Handle chat input and generate response"""
    # Define simple queries for dynamic spinner text
    simple_queries = ['test', 'hello', 'hi', 'help', 'what', 'how are you']
    query_lower = prompt.lower().strip()
    
    # Set dynamic spinner text based on query complexity
    if query_lower in simple_queries or len(prompt.split()) < 5:
        spinner_text = "Searching legal database..."
    else:
        spinner_text = "Analyzing complex query... This may take 1-2 minutes for detailed legal analysis."
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner(spinner_text):
            # Search legal database
            search_results = search_legal_database(prompt)
            
            # Generate response
            response = generate_legal_response(prompt, search_results)
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
