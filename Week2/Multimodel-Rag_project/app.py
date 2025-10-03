"""
Multimodal RAG Chatbot - Streamlit Application
"""
import streamlit as st
from datetime import datetime
import time

from config.settings import settings
from utils.logger import setup_logger
from utils.database import vector_db, memory_db
from utils.helpers import generate_session_id

logger = setup_logger("streamlit_app")

st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False

init_session_state()

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("🤖 LLM Provider")
    available_providers = settings.get_available_providers()
    
    if not available_providers:
        st.error("No LLM providers configured!")
        st.stop()
    
    selected_provider = st.selectbox("Provider", available_providers, key="provider")
    models = settings.SUPPORTED_LLM_PROVIDERS.get(selected_provider, [])
    selected_model = st.selectbox("Model", models, key="model")
    
    st.divider()
    
    st.subheader("🔍 RAG Type")
    rag_type = st.radio("Select RAG Variant", settings.RAG_VARIANTS, key="rag_type")
    
    st.divider()
    
    st.subheader("📊 Parameters")
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("Top K", 1, 10, 5, key="top_k")
    with col2:
        similarity = st.number_input("Similarity", 0.0, 1.0, 0.3, 0.1, key="similarity")
    
    if rag_type == "Hybrid RAG":
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.6, 0.1, key="semantic_weight")
    else:
        semantic_weight = 0.6
    
    st.divider()
    
    st.subheader("✨ Features")
    enable_memory = st.checkbox("💭 Memory", value=True, key="memory")
    enable_guardrails = st.checkbox("🛡️ Guardrails", value=True, key="guardrails")
    enable_web_search = st.checkbox("🌐 Web Search", value=False, key="web_search")
    
    st.divider()
    
    st.subheader("🎛️ Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temperature")
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100, key="max_tokens")
    
    st.divider()
    
    if st.button("🚀 Initialize RAG", type="primary"):
        with st.spinner("Initializing RAG system..."):
            try:
                from modules.rag_selector import RAGSelector
                from modules.memory_manager import RAGWithMemory
                from modules.guardrails import RAGWithGuardrails
                from modules.web_search import RAGWithWebSearch
                
                rag = RAGSelector(
                    rag_type=rag_type,
                    provider_name=selected_provider.lower(),
                    model_name=selected_model,
                    top_k=top_k,
                    similarity_threshold=similarity,
                    semantic_weight=semantic_weight
                )
                
                if enable_memory:
                    rag = RAGWithMemory(rag, session_id=st.session_state.session_id, max_messages=10)
                
                if enable_guardrails:
                    rag = RAGWithGuardrails(rag, toxicity_threshold=0.7, enable_guardrails=True)
                
                if enable_web_search:
                    rag = RAGWithWebSearch(rag, enable_web_search=True)
                
                st.session_state.rag_instance = rag
                st.session_state.rag_initialized = True
                st.success("✅ RAG initialized!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed: {e}")
                logger.error(f"Initialization failed: {e}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if hasattr(st.session_state.rag_instance, 'clear_memory'):
            st.session_state.rag_instance.clear_memory()
        st.rerun()
    
    st.divider()
    if st.session_state.rag_initialized:
        st.success("✅ RAG Active")
    else:
        st.warning("⚠️ Click Initialize RAG")

st.title("🤖 Multimodal RAG Chatbot")
st.caption("Upload documents and ask questions")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📊 Statistics"])

with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("metadata"):
                meta = message["metadata"]
                cols = st.columns(4)
                cols[0].caption(f"⏱️ {meta.get('time', 0):.2f}s")
                cols[1].caption(f"📄 {meta.get('docs', 0)} docs")
                if meta.get('blocked'):
                    cols[2].caption("🛡️ Blocked")
                if meta.get('web_search'):
                    cols[3].caption("🌐 Web")
            
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {src.get('filename', 'Unknown')}**")
                        st.caption(f"Similarity: {src.get('similarity', 0):.2f}")
    
    if prompt := st.chat_input("Ask a question..."):
        if not st.session_state.rag_initialized:
            st.error("❌ Please initialize RAG first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_instance.query(
                            prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            return_sources=True
                        )
                        
                        answer = response.get('answer', 'No answer')
                        st.markdown(answer)
                        
                        metadata = {
                            'time': response.get('total_time', 0),
                            'docs': response.get('retrieved_count', 0),
                            'blocked': response.get('blocked', False),
                            'web_search': response.get('used_web_search', False)
                        }
                        
                        cols = st.columns(4)
                        cols[0].caption(f"⏱️ {metadata['time']:.2f}s")
                        cols[1].caption(f"📄 {metadata['docs']} docs")
                        if metadata['blocked']:
                            cols[2].caption("🛡️ Blocked")
                        if metadata['web_search']:
                            cols[3].caption("🌐 Web")
                        
                        sources = response.get('sources', [])
                        if sources:
                            with st.expander("📚 Sources"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(f"**{i}. {src.get('filename')}**")
                                    st.caption(f"Similarity: {src.get('similarity', 0):.2f}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "metadata": metadata
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

with tab2:
    from modules.document_processor import document_processor
    
    st.subheader("📄 Document Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, or Images",
            type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📤 Process Documents", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status.text(f"Processing {file.name}...")
                try:
                    result = document_processor.process_uploaded_file(file, save_to_disk=True)
                    if result['success']:
                        st.success(f"✅ {file.name}: {result['chunks']} chunks")
                    else:
                        st.error(f"❌ {file.name}: {result.get('error')}")
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                
                progress.progress((i + 1) / len(uploaded_files))
            
            status.text("✅ Complete!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        st.write("### Stats")
        stats = vector_db.get_stats()
        st.metric("Vectors", stats['total_vectors'])
        st.metric("Documents", stats.get('total_documents', 0))
    
    st.divider()
    
    st.write("### Uploaded Documents")
    docs = document_processor.get_all_documents()
    
    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc['filename']}"):
                col1, col2, col3 = st.columns(3)
                col1.text(f"Type: {doc['file_type']}")
                col2.text(f"Size: {doc['file_size']}")
                col3.text(f"Chunks: {doc['chunk_count']}")
                st.caption(f"Uploaded: {doc['upload_time']}")
                
                if st.button("🗑️ Delete", key=f"del_{doc['document_id']}"):
                    document_processor.delete_document(doc['document_id'])
                    st.success("Deleted!")
                    st.rerun()
    else:
        st.info("No documents yet")

with tab3:
    st.subheader("📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Messages", len(st.session_state.messages))
    col1.metric("Session", st.session_state.session_id[:8] + "...")
    
    stats = vector_db.get_stats()
    col2.metric("Vectors", stats['total_vectors'])
    col2.metric("Documents", len(memory_db.get_all_documents()))
    
    api_status = settings.validate_api_keys()
    configured = sum(1 for v in api_status.values() if v)
    col3.metric("APIs", f"{configured}/6")
    col3.metric("RAG", "Active" if st.session_state.rag_initialized else "Inactive")
    
    st.divider()
    
    st.write("### API Status")
    for api, status in api_status.items():
        col1, col2 = st.columns([3, 1])
        col1.text(api)
        if status:
            col2.success("✅")
        else:
            col2.error("❌")
    
    if st.session_state.rag_initialized:
        st.divider()
        st.write("### Configuration")
        st.json({
            "rag_type": rag_type,
            "provider": selected_provider,
            "model": selected_model,
            "top_k": top_k,
            "similarity": similarity,
            "temperature": temperature,
            "max_tokens": max_tokens
        })

st.divider()
st.caption("Multimodal RAG Chatbot | Powered by Groq")