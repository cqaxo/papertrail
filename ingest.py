import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load documents from the specified directory and chunk them into
def load_documents(docs_dir="data"):
    """Load documents from the specified directory."""
    documents = []
    for filepath in pathlib.Path(docs_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(filepath))
        documents.extend(loader.load())

    return documents

# Chunk documents into smaller pieces for better processing by the language model
def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(docs)

    return chunked_documents

# Get the embedding model: used for both building and querying the store
def get_embeddings():
    """The embedding model: used for both building and querying the store."""

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build a vector store from the chunked documents
def build_vector_store(chunks, persist_directory="chroma_db"):
    """Build a vector store from the chunked documents."""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    return vector_store

# Load the vector store from the specified directory
def load_vector_store(persist_directory="chroma_db"):
    """Load the existing vector store from disk."""
    embeddings = get_embeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vector_store

def retrieve(query, k=4):
    """Retrieve the top-k most relevant chunks for a query."""
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)

    return results

# Main
if __name__ == "__main__":
    documents = load_documents()
    chunked_documents = chunk_documents(documents)
    vector_store = build_vector_store(chunked_documents)
    retrieved_results = retrieve("What does the heading of this document say?")
    print(f"Retrieved {len(retrieved_results)} relevant chunks.")
    print(retrieved_results[0].page_content)