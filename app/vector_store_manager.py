import streamlit as st
from langchain_community.vectorstores import FAISS
import os

VECTOR_STORE_PATH = "faiss_index"

class VectorStoreManager:
    @staticmethod
    def check_vector_store_exists():
         return os.path.exists(VECTOR_STORE_PATH) and os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss")


    @staticmethod
    def safe_load_vector_store(embedding_model):
        try:
            if not VectorStoreManager.check_vector_store_exists():
                st.error("Please process documents first before asking questions.")
                return None
                
            return FAISS.load_local(
                VECTOR_STORE_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None


