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
    import openai
    
    # Initialize clients
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
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
        # Prepare context from search results
        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"\n--- Source {i} ---\n"
            context += f"Citation: {result['citation']}\n"
            context += f"Jurisdiction: {result['jurisdiction']}\n"
            context += f"Content: {result['text']}\n"
        
        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": LEGAL_COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Legal Sources:\n{context}"}
        ]
        
        # Generate response
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=messages,
            max_tokens=1500,
            temperature=0.1,
            reasoning_effort="high",
            verbosity="high"
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
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
/* STREAMLIT CHAT INPUT COMPLETE OVERRIDE */

/* Step 1: Remove default borders and outlines */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] [role="textbox"] {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  background: #21262d !important;
}

/* Step 2: Blue focus state when clicked */
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] input:focus,
[data-testid="stChatInput"] [role="textbox"]:focus {
  border: 2px solid #0969da !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15) !important;
  background: #21262d !important;
}

/* Step 3: CRITICAL - Find the actual button container and make it relative */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] form > div,
[data-testid="stChatInput"] [data-baseweb] {
  position: relative !important;
}

/* Step 4: NUCLEAR BUTTON POSITIONING - Target every possible button selector */
[data-testid="stChatInput"] button,
[data-testid="stChatInput"] [role="button"],
[data-testid="stChatInput"] input[type="submit"],
[data-testid="stChatInput"] [type="submit"],
[data-testid="stChatInput"] [kind="primary"],
[data-testid="stChatInput"] [data-testid*="button"],
[data-testid="stChatInput"] [class*="button"],
[data-testid="stChatInput"] [class*="Button"],
[data-testid="stChatInput"] [class*="submit"],
[data-testid="stChatInput"] [class*="Submit"] {
  position: absolute !important;
  right: 8px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  width: 40px !important;
  height: 40px !important;
  min-width: 40px !important;
  min-height: 40px !important;
  background: #0969da !important;
  border: none !important;
  border-radius: 50% !important;
  z-index: 9999 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  box-shadow: 0 2px 8px rgba(9, 105, 218, 0.3) !important;
}

/* Step 5: Button hover and active states */
[data-testid="stChatInput"] button:hover,
[data-testid="stChatInput"] [role="button"]:hover {
  background: #0860ca !important;
  transform: translateY(-50%) scale(1.05) !important;
}

[data-testid="stChatInput"] button:active,
[data-testid="stChatInput"] [role="button"]:active {
  transform: translateY(-50%) scale(0.95) !important;
}

/* Step 6: Button icon styling */
[data-testid="stChatInput"] button svg,
[data-testid="stChatInput"] [role="button"] svg {
  width: 20px !important;
  height: 20px !important;
  color: white !important;
}

/* Step 7: Add padding to textarea so text doesn't overlap button */
[data-testid="stChatInput"] textarea {
  padding-right: 56px !important;
  padding-left: 16px !important;
  padding-top: 12px !important;
  padding-bottom: 12px !important;
  border-radius: 24px !important;
  min-height: 48px !important;
}

/* Step 8: Mobile responsive */
@media (max-width: 767px) {
  [data-testid="stChatInput"] button,
  [data-testid="stChatInput"] [role="button"] {
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    right: 6px !important;
  }
  
  [data-testid="stChatInput"] button svg,
  [data-testid="stChatInput"] [role="button"] svg {
    width: 18px !important;
    height: 18px !important;
  }
  
  [data-testid="stChatInput"] textarea {
    padding-right: 48px !important;
    min-height: 44px !important;
  }
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Force CSS override after Streamlit loads - this runs every time the page renders
st.markdown("""
<style>
/* NUCLEAR OPTION - Override ALL possible Streamlit focus states */
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] [role="textbox"],
[data-testid="stChatInput"] [contenteditable],
div[data-baseweb="textarea"],
div[data-baseweb="input"] {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  background: #21262d !important;
}

/* Remove ALL borders and outlines in default state */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] [role="textbox"],
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="input"],
[data-testid="stChatInput"] [contenteditable] {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  background: #21262d !important;
}

/* Override Streamlit's BaseWeb component borders */
[data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
[data-testid="stChatInput"] div[data-baseweb="input"] > div,
[data-testid="stChatInput"] [class*="StyledTextAreaRoot"],
[data-testid="stChatInput"] [class*="StyledInputRoot"] {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Keep the blue focus state when clicked */
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible,
[data-testid="stChatInput"] *:focus-within,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] input:focus,
[data-testid="stChatInput"] [role="textbox"]:focus,
[data-testid="stChatInput"] [contenteditable]:focus,
div[data-baseweb="textarea"]:focus,
div[data-baseweb="input"]:focus,
[data-testid="stChatInput"] .focused,
[data-testid="stChatInput"] [class*="focus"],
[data-testid="stChatInput"] [class*="Focus"],
[data-testid="stChatInput"] [aria-expanded="true"] {
  border: 2px solid #0969da !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15) !important;
  background: #21262d !important;
}

/* Ensure bottom borders stay grey and don't turn red */
[data-testid="stChatInput"],
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] textarea {
  border-bottom: 1px solid #30363d !important;
  border-bottom-color: #30363d !important;
}

[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] form:focus-within,
[data-testid="stChatInput"] textarea:focus {
  border-bottom: 1px solid #30363d !important;
  border-bottom-color: #30363d !important;
}
</style>
""", unsafe_allow_html=True)