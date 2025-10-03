import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(
    page_title="AI Story Generator",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    .story-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 1rem;
        color: #333;
        line-height: 1.8;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 12px;
        font-size: 16px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Check API key
if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY not found! Please set it in your .env file.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Header
st.markdown("# 📚 AI Story Generator")
st.markdown('<p class="subtitle">✨ Turn your ideas into captivating short stories</p>', unsafe_allow_html=True)

# Create a container for the input section
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        # User input with placeholder
        prompt = st.text_input(
            "What should your story be about?",
            placeholder="e.g., A robot learning to feel emotions...",
            label_visibility="visible"
        )
        
        # Advanced options in expander
        with st.expander("⚙️ Advanced Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                max_tokens = st.slider("Story Length", 200, 1000, 500, 50)
            with col_b:
                temperature = st.slider("Creativity", 0.0, 1.0, 0.8, 0.1)
        
        # Generate button
        generate_btn = st.button("🎨 Generate Story", use_container_width=True)

# Story generation
if generate_btn:
    if not prompt.strip():
        st.warning("⚠️ Please enter a story prompt!")
    else:
        with st.spinner("✨ Crafting your story..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a creative storyteller. Write engaging, well-structured short stories with vivid descriptions and compelling narratives."
                        },
                        {
                            "role": "user",
                            "content": f"Write a short story about: {prompt}"
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                story = response.choices[0].message.content
                
                # Display story in a nice container
                st.markdown("---")
                st.markdown("### 📖 Your Story")
                
                # Story container with custom styling
                st.markdown(f"""
                <div class="story-container">
                    <p style="color: #333; white-space: pre-wrap;">{story}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns([2, 2, 2])
                with col1:
                    st.download_button(
                        label="📥 Download Story",
                        data=story,
                        file_name="my_story.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("🔄 Generate Another", use_container_width=True):
                        st.rerun()
                with col3:
                    st.button("📋 Copy to Clipboard", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Try a different prompt or check your API key.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: white; opacity: 0.7;'>Powered by Groq AI 🚀</p>",
    unsafe_allow_html=True
)