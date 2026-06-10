import re
import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from functools import lru_cache


# Repeated boilerplate this PDF stamps on nearly every page. Stripped before
# chunking so it does not dilute chunk embeddings.
_RUNNING_HEADER = "NIST AI 100-1 AI RMF 1.0"
_DOI_URL = "https://doi.org/10.6028/NIST.AI.100-1"
_FREE_OF_CHARGE = "This publication is available free of charge from:"
_PAGE_MARKER = re.compile(r"^Page\s+(\d+|[ivxlcdm]+)$", re.IGNORECASE)


# Strip boilerplate from the text, including the running header, page markers, and publication notice. This is done before chunking to prevent dilution of chunk embeddings.
def strip_boilerplate(text):
    """Remove the running header, page markers, and publication notice from page text."""
    # The header and URL are distinctive strings, so remove them wherever they appear (they sometimes sit glued to real content on the same line).
    text = text.replace(_RUNNING_HEADER, " ").replace(_DOI_URL, " ")
    # Page markers and the notice line are removed at the line level, so we never touch the words "Page" or "free of charge" where they show up mid-sentence.
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if _PAGE_MARKER.match(stripped) or stripped == _FREE_OF_CHARGE:
            continue
        kept.append(line)

    return "\n".join(kept)


# Load documents from the specified directory and chunk them into
def load_documents(docs_dir="data"):
    """Load documents from the specified directory."""
    documents = []
    for filepath in pathlib.Path(docs_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(filepath))
        documents.extend(loader.load())
        for doc in documents:
            doc.page_content = strip_boilerplate(doc.page_content)

    return documents


# Chunk documents into smaller pieces for better processing by the language model
def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(docs)

    return chunked_documents


# Get the embedding model: used for both building and querying the store
@lru_cache(maxsize=1)
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
@lru_cache(maxsize=1)
def load_vector_store(persist_directory="chroma_db"):
    """Load the existing vector store from disk."""
    embeddings = get_embeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vector_store


# Retrieve the top-k most relevant chunks for a query
def retrieve(query, k=6):
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