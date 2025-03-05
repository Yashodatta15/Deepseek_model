import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from app.model_manager import ModelManager

MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", 
    "chunk_size": 300,
    "chunk_overlap": 60,
    "llm_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
}

VECTOR_STORE_PATH = "faiss_index"

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embedding_model"],
            model_kwargs={'device': 'cuda'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG["chunk_size"],
            chunk_overlap=MODEL_CONFIG["chunk_overlap"]
        )

    def process_documents(self, files):
        try:
            text = self.extract_text(files)
            if not text:
                st.error("No valid text content found in the uploaded documents.")
                return False

            chunks = self.text_splitter.split_text(text)
            if not chunks:
                st.error("No valid text chunks created from the documents.")
                return False

            vector_store = None
            batch_size = 100
            
            # Create a more interactive progress bar
            progress_text = "Processing documents... Please wait"
            my_bar = st.progress(0, text=progress_text)
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, self.embedding_model)
                else:
                    temp_store = FAISS.from_texts(batch, self.embedding_model)
                    vector_store.merge_from(temp_store)
                
                progress = min(1.0, (i + batch_size) / len(chunks))
                my_bar.progress(progress, text=f"{progress_text} ({int(progress * 100)}%)")

            vector_store.save_local(VECTOR_STORE_PATH)
            st.success("🎉 Documents processed successfully!")
            st.balloons()
            st.snow()
            return True

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False


    def extract_text(self, files):
        """Extract text from documents with error handling"""
        text_contents = []
        
        for file in files:
            try:
                text = None
                if file.name.endswith('.pdf'):
                    text = self._process_pdf(file)
                elif file.name.endswith('.txt'):
                    text = self._process_txt(file)
                elif file.name.endswith('.csv'):
                    text = self._process_csv(file)
                else:
                    st.warning(f"⚠️ Unsupported file type: {file.name}")
                    continue
                    
                if text and text.strip():
                    text_contents.append(text)
                    
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                
        return "\n\n".join(text_contents)

    def _process_pdf(self, file):
        try:
            reader = PdfReader(file)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return " ".join(text)
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def _process_txt(self, file):
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return ""

    def _process_csv(self, file):
        try:
            df = pd.read_csv(file)
            return df.to_string()
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return ""

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()



