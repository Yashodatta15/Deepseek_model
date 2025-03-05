import streamlit as st
from document_processor import DocumentProcessor 
from model_manager import ModelManager 
from vector_store_manager import VectorStoreManager 
from ui_components import initialize_session_state, create_sidebar 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def main():
    st.set_page_config(
        page_title="📚 Interactive Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()
    
    st.title("📚 Interactive Document Q&A")
    st.markdown("---")
    
    doc_processor = DocumentProcessor()
    uploaded_files = create_sidebar()

    if uploaded_files:
        if st.sidebar.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                if doc_processor.process_documents(uploaded_files):
                    st.session_state["vector_store_ready"] = True
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input("💭 Ask about your documents..."):
        if not VectorStoreManager.check_vector_store_exists():
            st.error("📂 Please upload and process documents first!")
            return
            
        llm = st.session_state.model_manager.initialize_model()
        if not llm:
            return

        vector_store = VectorStoreManager.safe_load_vector_store(doc_processor.embedding_model)
        if not vector_store:
            return
                
        qa_chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=PromptTemplate(
                template="""You are an AI document chatbot.
                Answer all questions based on the provided context. 
                Be precise and to the point. 
                If asked anything outside of the context, respond with: 
                "I don't have any knowledge."
                Context: {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )
        )
        
        docs = vector_store.similarity_search(query, k=3)
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = qa_chain.invoke(
                    {"input_documents": docs, "question": query},
                    return_only_outputs=True
                )
                st.markdown(response["output_text"])
            
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})

if __name__ == "__main__":
    main()
