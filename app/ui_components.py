import streamlit as st
import os
from model_manager import ModelManager 
from vector_store_manager import VECTOR_STORE_PATH
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()

def create_sidebar():
    """Create and manage sidebar elements"""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/chat.png", width=100)
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV",
            key="file_uploader"
        )

        col1, col2 = st.columns(2)
        with col1:
            clear_chat = st.button("🗑️ Clear Chat")
        with col2:
            clear_docs = st.button("🗑️ Clear Docs")

        if clear_chat:
            st.session_state.messages = []
            st.success("Chat cleared!")
            
        if clear_docs:
            if os.path.exists(VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(VECTOR_STORE_PATH)
                st.success("Documents cleared!")
                st.session_state.pop("vector_store_ready", None)

        return uploaded_files

 # Custom CSS for better appearance
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f3e5f5;
        }
        </style>
    """, unsafe_allow_html=True)

