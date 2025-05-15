import os
import shutil
from typing import List

# LangChain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "mistral:instruct"  

EMBEDDING_MODEL = "nomic-embed-text"
DATA_DIR = "knowledge_base"
VECTORSTORE_PATH = "faiss_index_store"

# --- RAG Pipeline Setup ---
def load_documents(data_dir: str) -> List:
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
    documents = loader.load()
    if not documents:
        print(f"No documents found in {data_dir}. Adding placeholder...")
        dummy_file_path = os.path.join(data_dir, "empty_placeholder.txt")
        if not os.path.exists(dummy_file_path):
            with open(dummy_file_path, "w") as f:
                f.write("This is a placeholder document because the knowledge base was empty.")
        documents = TextLoader(dummy_file_path).load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: List) -> List:
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def get_vector_store(chunks: List, embeddings_model: OllamaEmbeddings, persist_directory: str = None) -> FAISS:
    if persist_directory and os.path.exists(persist_directory):
        print(f"Loading existing vector store from {persist_directory}...")
        try:
            return FAISS.load_local(persist_directory, embeddings_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}. Rebuilding...")

    print("Creating new vector store...")
    from langchain_core.documents import Document
    if not chunks:
        chunks = [Document(page_content="Placeholder content for empty vector store.")]
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    if persist_directory:
        vector_store.save_local(persist_directory)
    return vector_store

def setup_rag_chain(llm_model: ChatOllama, vector_store: FAISS):
    print("Setting up RAG chain...")
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})

    template = """
    You are a helpful AI assistant. Use the following retrieved context to answer the question.
    If you don't know the answer from the context, just say that you don't know.
    Do not make up an answer. Keep your answers concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return rag_chain

# --- Main Application Logic ---
def main():
    print("Initializing Ollama models...")
    try:
        ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)
        ollama_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL, temperature=0.3)
        ollama_llm.invoke("Hello")  # Test
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and models are pulled.")
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        with open(os.path.join(DATA_DIR, "put_your_documents_here.txt"), "w") as f:
            f.write("Add your .txt files to this directory.")

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    vector_store = get_vector_store(chunks, ollama_embeddings, VECTORSTORE_PATH)
    rag_chain = setup_rag_chain(ollama_llm, vector_store)

    print("\nText-based RAG assistant is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        if user_input.lower() == "clear index":
            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
                print("Vector store cleared.")
                vector_store = get_vector_store(chunks, ollama_embeddings, VECTORSTORE_PATH)
                rag_chain = setup_rag_chain(ollama_llm, vector_store)
            else:
                print("No vector store found to clear.")
            continue
        print("Thinking...")
        try:
            response = rag_chain.invoke(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
