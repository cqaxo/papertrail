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

# Build a vector store from the chunked documents
def build_vector_store(chunks, persist_directory="chroma_db"):
    """Build a vector store from the chunked documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    return vector_store

# Main
if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    print(documents[0].page_content[:200])
    print(documents[0].metadata)
    chunked_documents = chunk_documents(documents)
    print(f"Chunked into {len(chunked_documents)} pieces.")
    print(chunked_documents[0].page_content[:200])
    vector_store = build_vector_store(chunked_documents)
    vector_list =vector_store.similarity_search("What does the heading of this document say?")
    print(f"Found {len(vector_list)} similar chunks.")
    print(vector_list[0].page_content)
    