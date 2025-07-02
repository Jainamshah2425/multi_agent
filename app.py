import streamlit as st
import os
from datetime import datetime
from config import config
from document_processor import DocumentProcessor
from qa_engine import QAEngine
from typing import Optional

from utils import display_sources, truncate_text, logger

# Page configuration
st.set_page_config(
    page_title="Multi-Document QA Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
    background-color: #f8f9fa;
}
.source-box {
    background-color: #e9ecef;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_engine" not in st.session_state:
    st.session_state.qa_engine = QAEngine()
if "document_processor" not in st.session_state:
    st.session_state.document_processor = DocumentProcessor()

# Main header
st.markdown('<h1 class="main-header">🧠 Multi-Document QA Agent</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # LLM Selection
    llm_option = st.sidebar.radio("Choose LLM:", ["mistral", "google", "local"])
    st.session_state.llm_type = llm_option
    
    # Settings
    st.markdown("### 🔧 Settings")
    enable_web_search = st.checkbox("Enable Web Search", value=True)
    max_sources = st.slider("Max Sources", 1, 10, 5)
    
    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Statistics
    st.markdown("### 📊 Statistics")
    st.metric("Chat Messages", len(st.session_state.chat_history))
    
    if os.path.exists(config.INDEX_DIR):
        st.success("✅ Index Ready")
    else:
        st.warning("⚠️ No Index Found")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Document upload section
    st.markdown("### 📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload multiple documents to build your knowledge base"
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            success = st.session_state.document_processor.process_uploaded_files(uploaded_files)
            
        if success:
            st.success(f"✅ Successfully processed {len(uploaded_files)} documents!")
        else:
            st.error("❌ Error processing documents. Please check the logs.")

with col2:
    # Quick stats
    if uploaded_files:
        st.markdown("### 📈 Upload Summary")
        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.size} bytes)")

# Chat interface
st.markdown("### 💬 Chat Interface")

# Query input
query = st.text_input(
    "Ask a question about your documents:",
    placeholder="Enter your question here...",
    key="query_input"
)

# Submit button
if st.button("🚀 Ask Question", type="primary") and query:
    with st.spinner("Generating answer..."):
        result = st.session_state.qa_engine.answer_query(query, llm_option)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "answer": result['answer'],
            "sources": result['sources'],
            "llm_type": llm_option
        })
        
        # Keep only recent history
        if len(st.session_state.chat_history) > config.MAX_CHAT_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-config.MAX_CHAT_HISTORY:]
    
    # Clear input
    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 📜 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"❓ {truncate_text(chat['query'])} ({chat['timestamp'].strftime('%H:%M:%S')})"):
            st.markdown(f"**Question:** {chat['query']}")
            st.markdown(f"**Answer:** {chat['answer']}")
            st.markdown(f"**LLM Used:** {chat['llm_type']}")
            
            if chat['sources']:
                display_sources(chat['sources'])

