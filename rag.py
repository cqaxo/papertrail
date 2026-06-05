import os
from langchain_ollama import ChatOllama
from ingest import retrieve

# Initialize the Ollama language model from environment variable or default to "llama3.1:8b"
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

def answer(query, k=6):
    """Answer a query using the retrieved chunks."""
    retrieved_chunks = retrieve(query, k=k)
    context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
    prompt = f"""You are answering questions using only the provided context from a set of documents. Base your answer solely on the context below. If the context does not contain enough information to answer the question, say "I don't know based on the provided documents." Do not use outside knowledge and do not guess.

    Context:
    {context}

    Question: {query}"""
    response = llm.invoke(prompt)

    return response.content, retrieved_chunks

if __name__ == "__main__":
    response = answer("Does this document contain words written in latin?")
    response2 = answer("What is the meaning of life?")
    print(response)
    print(response2)