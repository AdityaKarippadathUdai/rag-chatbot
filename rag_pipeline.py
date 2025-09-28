# rag_pipeline.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile

def process_uploaded_files(uploaded_files, max_file_size=10 * 1024 * 1024):
    """
    Process uploaded files (PDF, TXT, DOCX, MD) and return document chunks and raw documents.
    """
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            if uploaded_file.size > max_file_size:
                print(f"Skipping {uploaded_file.name}: File size exceeds 10MB limit.")
                continue
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif uploaded_file.name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif uploaded_file.name.endswith(".md"):
                    try:
                        loader = UnstructuredMarkdownLoader(file_path)
                    except ImportError:
                        print(f"Skipping {uploaded_file.name}: UnstructuredMarkdownLoader not available.")
                        continue
                else:
                    print(f"Skipping {uploaded_file.name}: Unsupported file type.")
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks, documents

def create_vector_store(chunks, embeddings):
    """
    Create a Chroma vector store from document chunks.
    """
    if not chunks:
        return None
    persist_directory = "./chroma_db"
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )
    return vector_store

def generate_document_summary(documents, llm):
    """
    Generate a summary of the uploaded documents.
    """
    if not documents:
        return "No documents to summarize."
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following document in 2-3 sentences:\n\n{text}"
    )
    text = "\n".join([doc.page_content for doc in documents[:3]])  # Limit to first 3 docs
    try:
        chain = summary_prompt | llm
        summary = chain.invoke({"text": text})
        return summary if isinstance(summary, str) else summary.get("text", "Summary unavailable.")
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def create_rag_chain(model_id, k=3):
    """
    Create a RAG chain with the specified Hugging Face model.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text2text-generation",
            pipeline_kwargs={"max_length": 256, "truncation": True}
        )
        return embeddings, llm
    except Exception as e:
        raise Exception(f"Error initializing model: {str(e)}")
