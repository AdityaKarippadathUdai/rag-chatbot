# app.py
import streamlit as st
from rag_pipeline import process_uploaded_files, create_vector_store, generate_document_summary, create_rag_chain
from langchain.chains import RetrievalQA
import os
import json
from io import BytesIO

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None

# Function to save chat history
def save_chat_history():
    try:
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state.messages, f, indent=2)
    except Exception as e:
        st.warning(f"Error saving chat history: {str(e)}")

# Function to load chat history
def load_chat_history():
    if os.path.exists("chat_history.json"):
        try:
            with open("chat_history.json", "r") as f:
                st.session_state.messages = json.load(f)
        except Exception as e:
            st.warning(f"Error loading chat history: {str(e)}")

# Load existing chat history
load_chat_history()

# Streamlit app configuration
st.set_page_config(page_title="RAG Chatbot with Document Upload", page_icon="🤖", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Chatbot Settings")
    st.markdown("Upload documents and configure the model.")
    
    # Model selection
    model_options = {
        "flan-t5-small": "google/flan-t5-small",
        "flan-t5-base": "google/flan-t5-base"
    }
    selected_model = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        help="Smaller models (e.g., flan-t5-small) are faster on CPU but less accurate."
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, DOCX, MD)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True
    )
    
    # Process files when uploaded
    if uploaded_files:
        with st.spinner("Processing documents..."):
            try:
                embeddings, llm = create_rag_chain(model_options[selected_model])
                chunks, documents = process_uploaded_files(uploaded_files)
                if chunks:
                    st.session_state.vector_store = create_vector_store(chunks, embeddings)
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    st.session_state.document_summary = generate_document_summary(documents, llm)
                    st.success("Documents processed successfully!")
                else:
                    st.error("No valid documents to process.")
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")
                st.markdown("Try a smaller model (e.g., flan-t5-small), reduce document size, or check system memory.")
                st.stop()

    # Slider for retrieved chunks
    k = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=3)
    if st.session_state.qa_chain:
        st.session_state.qa_chain.retriever.search_kwargs["k"] = k
    
    # Download chat history
    if st.session_state.messages:
        chat_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

# Main chat interface
st.title("Conversational AI Chatbot with RAG")

# Display document summary
if st.session_state.document_summary:
    with st.expander("Document Summary"):
        st.markdown(st.session_state.document_summary)

# Display warning if no documents
if not st.session_state.qa_chain:
    st.warning("Please upload documents to start querying.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    response = result["result"]
                    sources = result["source_documents"]
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.stop()
            
            st.markdown(response)
            with st.expander("Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}** (from {doc.metadata.get('source', 'N/A')}, page {doc.metadata.get('page', 'N/A')}): {doc.page_content[:200]}...")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_history()