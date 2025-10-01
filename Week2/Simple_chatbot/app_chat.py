"""
AI Chat Application with Groq
File: app.py
Run: streamlit run app.py
"""

import time
import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import warnings
import urllib3

# Disable SSL warnings for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Sidebar text - more specific selectors */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: white !important;
    }
    
    /* Fix selectbox visibility */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1e3c72 !important;
        border-radius: 8px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox option {
        background-color: white !important;
        color: #1e3c72 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1e3c72 !important;
    }
    
    /* Fix slider styling */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background-color: white !important;
    }
    
    /* Fix checkbox styling */
    [data-testid="stSidebar"] .stCheckbox > label > div {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Headers */
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
        font-size: 2.5rem !important;
    }
    
    h2, h3 {
        color: white;
        font-weight: 600;
    }
    
    /* Buttons - Enhanced styling */
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton button:active {
        transform: translateY(0px);
    }
    
    /* Secondary buttons */
    .stButton button[kind="secondary"] {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Metrics - Enhanced */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.15);
        padding: 15px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border-left: 4px solid;
    }
    
    .stSuccess {
        background: rgba(72, 187, 120, 0.2) !important;
        border-left-color: #48bb78;
    }
    
    .stInfo {
        background: rgba(66, 153, 225, 0.2) !important;
        border-left-color: #4299e1;
    }
    
    .stWarning {
        background: rgba(237, 137, 54, 0.2) !important;
        border-left-color: #ed8936;
    }
    
    /* Input field */
    .stChatInput {
        border-radius: 25px;
    }
    
    .stChatInput > div > div {
        border-radius: 25px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%) !important;
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    /* Caption styling */
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.85rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
        border-right-color: transparent !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file!")
    st.info("""
    **Setup Instructions:**
    
    1. Create a `.env` file in the project folder
    2. Add your Groq API key:
    
    ```
    GROQ_API_KEY=gsk_your_key_here
    ```
    
    **Get FREE Groq API Key:**
    - Visit: https://console.groq.com/
    - Sign up (free, no credit card)
    - Go to "API Keys"
    - Create a new key
    - Copy and paste into .env file
    
    **Free Tier Benefits:**
    - ✅ 30 requests per minute
    - ✅ 14,400 requests per day
    - ✅ Super fast responses
    - ✅ No credit card required
    """)
    st.stop()

# Initialize Groq client with SSL verification disabled (for corporate networks)
try:
    import httpx
    # Create custom HTTP client that bypasses SSL verification
    http_client = httpx.Client(verify=False)
    client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
except Exception as e:
    st.error(f"❌ Failed to initialize Groq client: {e}")
    st.stop()

# -------------------------------
# Initialize Session State
# -------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if 'saved_chats' not in st.session_state:
    st.session_state.saved_chats = {}

if 'request_count' not in st.session_state:
    st.session_state.request_count = 0

if 'settings' not in st.session_state:
    st.session_state.settings = {
        'model': 'llama-3.3-70b-versatile',
        'temperature': 0.7,
        'max_tokens': 2048,
        'stream_response': True
    }

# -------------------------------
# Available Groq Models
# -------------------------------
GROQ_MODELS = {
    'llama-3.3-70b-versatile': '🦙 Llama 3.3 70B (Best Quality)',
    'llama-3.1-70b-versatile': '🦙 Llama 3.1 70B (Fast & Smart)',
    'llama-3.1-8b-instant': '⚡ Llama 3.1 8B (Ultra Fast)',
    'mixtral-8x7b-32768': '🎯 Mixtral 8x7B (Long Context)',
    'gemma2-9b-it': '💎 Gemma 2 9B (Balanced)'
}

# -------------------------------
# Helper Functions
# -------------------------------
def save_chat_history():
    """Save current chat to session state"""
    if st.session_state.chat_history:
        chat_name = st.session_state.chat_history[0]['content'][:30] + "..."
        st.session_state.saved_chats[st.session_state.current_chat_id] = {
            'name': chat_name,
            'messages': st.session_state.chat_history.copy(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }

def load_chat(chat_id):
    """Load a saved chat"""
    if chat_id in st.session_state.saved_chats:
        st.session_state.chat_history = st.session_state.saved_chats[chat_id]['messages'].copy()
        st.session_state.current_chat_id = chat_id

def new_chat():
    """Start a new chat"""
    save_chat_history()
    st.session_state.chat_history = []
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def delete_chat(chat_id):
    """Delete a saved chat"""
    if chat_id in st.session_state.saved_chats:
        del st.session_state.saved_chats[chat_id]

def export_chat():
    """Export chat as JSON"""
    return json.dumps({
        'chat_id': st.session_state.current_chat_id,
        'messages': st.session_state.chat_history,
        'exported_at': datetime.now().isoformat()
    }, indent=2)

def clear_all_chats():
    """Clear all saved chats"""
    st.session_state.saved_chats = {}
    st.session_state.chat_history = []

# -------------------------------
# AI Generation Function
# -------------------------------
def generate_with_groq(prompt, history, settings):
    """Generate response using Groq"""
    try:
        # Convert history to Groq format
        messages = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})
        
        # Call Groq API
        response = client.chat.completions.create(
            model=settings['model'],
            messages=messages,
            temperature=settings['temperature'],
            max_tokens=settings['max_tokens']
        )
        
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("⚡ Groq AI Chat")
    st.caption("Lightning fast AI responses")
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 Select Model")
    st.session_state.settings['model'] = st.selectbox(
        "Choose AI Model:",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        index=0,
        key="model_selector"
    )
    
    st.markdown("---")
    
    # Chat Management
    st.subheader("💬 Chat Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat", use_container_width=True):
            new_chat()
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            if st.session_state.chat_history:
                save_chat_history()
                st.success("Saved!")
            else:
                st.warning("No messages to save")
    
    # Clear all button
    if st.session_state.saved_chats:
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
            clear_all_chats()
            st.rerun()
    
    st.markdown("---")
    
    # Saved Chats
    if st.session_state.saved_chats:
        st.markdown("#### 📚 Saved Chats")
        
        for chat_id, chat_data in list(st.session_state.saved_chats.items())[:10]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(
                    f"💬 {chat_data['name'][:20]}",
                    key=f"load_{chat_id}",
                    use_container_width=True
                ):
                    load_chat(chat_id)
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{chat_id}", use_container_width=True):
                    delete_chat(chat_id)
                    st.rerun()
        
        st.caption(f"{len(st.session_state.saved_chats)} saved chats")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    
    st.session_state.settings['temperature'] = st.slider(
        "🌡️ Temperature",
        0.0, 2.0,
        st.session_state.settings['temperature'],
        0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    st.session_state.settings['max_tokens'] = st.slider(
        "📏 Max Response Length",
        256, 8000,
        st.session_state.settings['max_tokens'],
        256,
        help="Maximum tokens in response"
    )
    
    st.session_state.settings['stream_response'] = st.checkbox(
        "✨ Stream Response",
        st.session_state.settings['stream_response'],
        help="Show response word by word"
    )
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Messages", len(st.session_state.chat_history))
        st.metric("Saved", len(st.session_state.saved_chats))
    
    with col2:
        st.metric("Requests", st.session_state.request_count)
        user_msgs = sum(1 for m in st.session_state.chat_history if m['role'] == 'user')
        st.metric("Your Msgs", user_msgs)
    
    st.markdown("---")
    
    # Export
    if st.session_state.chat_history:
        st.subheader("📤 Export")
        chat_json = export_chat()
        st.download_button(
            label="💾 Download Chat",
            data=chat_json,
            file_name=f"chat_{st.session_state.current_chat_id}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # API Info
    st.subheader("ℹ️ API Info")
    st.success("✅ Groq Connected")
    st.caption("Free Tier Limits:")
    st.caption("• 30 requests/minute")
    st.caption("• 14,400 requests/day")
    
    st.markdown("---")
    st.caption("Powered by Groq ⚡")
    st.caption("Made with ❤️ using Streamlit")

# -------------------------------
# Main Chat Interface
# -------------------------------
st.title("💬 AI Chat Assistant")
st.caption(f"Using: {GROQ_MODELS[st.session_state.settings['model']]}")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(
        message["role"], 
        avatar="👤" if message["role"] == "user" else "🤖"
    ):
        st.markdown(message["content"])

# Welcome message
if not st.session_state.chat_history:
    st.info("""
    👋 **Welcome to AI Chat Assistant!**
    
    **Features:**
    - ⚡ Lightning fast responses with Groq
    - 🤖 Multiple AI models to choose from
    - 💾 Save and load conversations
    - ⚙️ Customizable settings (temperature, length)
    - 📊 Usage statistics
    - 📤 Export chats as JSON
    
    **Getting Started:**
    1. Type your message in the input box below
    2. Adjust model and settings in the sidebar
    3. Save important conversations
    
    Start chatting now! 🚀
    """)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            response_text, error = generate_with_groq(
                prompt, 
                st.session_state.chat_history[:-1], 
                st.session_state.settings
            )
        
        if error:
            st.error(f"❌ Error: {error}")
            
            if "429" in error or "rate" in error.lower():
                st.warning("""
                **Rate limit reached!**
                
                Free tier limits:
                - 30 requests per minute
                - 14,400 requests per day
                
                Please wait a moment and try again.
                """)
            elif "401" in error or "authentication" in error.lower():
                st.error("""
                **Authentication failed!**
                
                Please check your GROQ_API_KEY in the .env file.
                Get a new key at: https://console.groq.com/
                """)
        
        elif response_text:
            # Stream response if enabled
            if st.session_state.settings['stream_response']:
                displayed_text = ""
                words = response_text.split()
                
                for i, word in enumerate(words):
                    displayed_text += word + " "
                    message_placeholder.markdown(displayed_text + "▌")
                    
                    # Adjust sleep time based on response length
                    if len(words) > 100:
                        time.sleep(0.02)
                    else:
                        time.sleep(0.05)
                
                message_placeholder.markdown(response_text)
            else:
                message_placeholder.markdown(response_text)
            
            # Save assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            st.session_state.request_count += 1
            
            # Auto-save every 5 messages
            if len(st.session_state.chat_history) % 5 == 0:
                save_chat_history()