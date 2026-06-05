import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents from the specified directory and chunk them into
def load_documents(docs_dir="data"):
    """Load documents from the specified directory."""
    documents = []
    for filepath in pathlib.Path(docs_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(filepath))
        documents.extend(loader.load())

    return documents

# Chunk documents into smaller pieces for better processing by the language model.
def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(docs)

    return chunked_documents

# Main
if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    print(documents[0].page_content[:200])
    print(documents[0].metadata)
    chunked_documents = chunk_documents(documents)
    print(f"Chunked into {len(chunked_documents)} pieces.")
    print(chunked_documents[0].page_content[:200])