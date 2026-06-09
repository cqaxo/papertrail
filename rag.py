import os
from langchain_ollama import ChatOllama
from ingest import retrieve
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the language model (LLM) based on available environment variables
def get_llm():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=gemini_key)
    else:
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

# Initialize the LLM
llm = get_llm()

# Friendly label for the active provider, derived from the chosen model object so it can never disagree with what's actually answering.
if type(llm).__name__ == "ChatGoogleGenerativeAI":
    PROVIDER_LABEL = "Gemini (gemini-2.5-flash)"
else:
    PROVIDER_LABEL = f"local Llama via Ollama ({os.getenv('OLLAMA_MODEL', 'llama3.1:8b')})"

# Function to answer a query using retrieved chunks
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