# app.py
# Streamlit UI for Joke Generator with Groq API

import streamlit as st
from joke_generator import joke_generator
from guardrails import guardrails

# Page config
st.set_page_config(
    page_title="AI Joke Generator",
    page_icon="😂",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #FF6B6B;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .joke-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">AI Joke Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Groq AI with Smart Guardrails</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    joke_type = st.radio(
        "Select Joke Type",
        ["Pun (Setup + Punchline)", "One-Liner"],
        help="Choose your preferred joke style"
    )
    
    num_jokes = st.slider(
        "Number of Jokes",
        min_value=1,
        max_value=5,
        value=3,
        help="How many jokes to generate at once"
    )
    
    st.divider()
    
    st.info("""
    **Smart Guardrails Active**
    
    AI-powered content filtering blocks:
    - Violence & terrorism
    - Death & tragedies
    - Religion & politics
    - Race & discrimination
    - Sexual content
    - Disabilities
    
    Topics checked BEFORE calling API
    """)
    
    st.divider()
    
    if st.button("Show Safe Topics"):
        safe = guardrails.get_safe_topics()
        st.success(f"**Try these:**\n\n{', '.join(safe)}")
    
    st.divider()
    
    # Stats
    if 'jokes_generated' not in st.session_state:
        st.session_state.jokes_generated = 0
    
    if 'jokes_blocked' not in st.session_state:
        st.session_state.jokes_blocked = 0
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Generated", st.session_state.jokes_generated)
    with col_stat2:
        st.metric("Blocked", st.session_state.jokes_blocked)

# Main content
st.markdown("### Enter Your Topic")

col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "Topic",
        placeholder="e.g., coffee, programming, cats, pizza, monday...",
        help="Enter any topic for joke generation",
        label_visibility="collapsed"
    )

with col2:
    generate_btn = st.button("Generate Jokes", type="primary", use_container_width=True)

# Generate jokes
if generate_btn:
    if not topic:
        st.warning("Please enter a topic first")
    else:
        with st.spinner("AI is crafting your jokes..."):
            
            # Map joke type
            j_type = "pun" if "Pun" in joke_type else "oneliner"
            
            # Check guardrails first
            first_result = joke_generator.generate_joke(topic, j_type)
            
            if not first_result['success']:
                # Topic blocked by guardrails
                st.error(first_result['error'])
                st.session_state.jokes_blocked += 1
                
                if 'safe_topics' in first_result:
                    st.markdown("---")
                    st.info("Try these safe topics instead:")
                    
                    # Display as clickable buttons
                    cols = st.columns(5)
                    for i, safe_topic in enumerate(first_result['safe_topics'][:10]):
                        with cols[i % 5]:
                            if st.button(safe_topic, key=f"safe_{i}"):
                                st.rerun()
            else:
                # Generate multiple jokes
                joke_word = "joke" if num_jokes == 1 else "jokes"
                st.success(f"Generated {num_jokes} {joke_word} successfully")
                st.markdown("---")
                
                # Display first joke
                jokes_to_display = [first_result]
                
                # Generate remaining jokes
                for i in range(num_jokes - 1):
                    result = joke_generator.generate_joke(topic, j_type)
                    if result['success']:
                        jokes_to_display.append(result)
                
                # Display all jokes
                for idx, result in enumerate(jokes_to_display, 1):
                    if result['type'] == 'pun':
                        st.markdown(f"**• Setup:** {result['setup']}")
                        st.markdown(f"**• Punchline:** {result['punchline']}")
                    else:  # oneliner
                        st.markdown(f"**•** {result['joke']}")
                    
                    if idx < len(jokes_to_display):
                        st.markdown("")
                    
                    st.session_state.jokes_generated += 1
                
                st.markdown("---")
                
                # Feedback buttons
                st.markdown("**Rate these jokes:**")
                col_a, col_b, col_c = st.columns([1, 1, 2])
                with col_a:
                    if st.button("Funny", key="like", use_container_width=True):
                        st.success("Glad you liked them")
                with col_b:
                    if st.button("Not funny", key="dislike", use_container_width=True):
                        st.info("Try another topic for new jokes")
                with col_c:
                    if st.button("Generate More", key="regenerate", use_container_width=True):
                        st.rerun()

# Display session stats
if st.session_state.jokes_generated > 0:
    with st.expander("Session Stats"):
        st.write(f"**Total Jokes Generated:** {st.session_state.jokes_generated}")
        st.write(f"**Topics Blocked by Guardrails:** {st.session_state.jokes_blocked}")
        if st.session_state.jokes_generated > 0:
            total_attempts = st.session_state.jokes_generated + st.session_state.jokes_blocked
            block_rate = (st.session_state.jokes_blocked / total_attempts * 100)
            st.write(f"**Safety Rate:** {block_rate:.1f}% of attempts blocked")

# Footer
st.markdown("---")
st.caption("""
Made with Streamlit + Groq API | 
AI-Based Guardrails Active | 
Powered by Llama 3.3 70B | 
GenAI Learning Project
""")