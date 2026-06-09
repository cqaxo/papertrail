# PaperTrail

A local "chat with your documents" app. Ask questions about a set of PDFs and get answers grounded in the source text, with citations back to the exact chunks each answer came from, and an honest "I don't know" when the documents do not cover the question.

**Live demo:** https://papertrail-demo.streamlit.app/

![PaperTrail](docs/demo.png)

## Features

- **Grounded answers.** Responses are built only from your documents, not the model's general knowledge.
- **Citations.** Every answer shows the source chunks it was drawn from, so you can verify it.
- **Honest refusal.** When the answer is not in the documents, PaperTrail says so instead of guessing.
- **Local and private by default.** Runs entirely on your machine with Ollama. No data leaves your computer.
- **Deployable.** The same app runs a public demo on a free cloud API when no local GPU is available.
- **Configurable.** Swap the generation model without touching code.

## Demo

The hosted demo is preloaded with the NIST AI Risk Management Framework, a public-domain document. Try:

- "What are the four core functions of the AI RMF?"
- "What does the GOVERN function cover?"
- "What are the characteristics of trustworthy AI?"
- "What is the capital of France?" (to see the honest refusal when the answer is not in the documents)

> The hosted demo runs on a free API tier, so it may occasionally hit rate limits under heavy use, in which case it shows a brief notice and recovers shortly. Running locally uses Ollama instead, which has no rate limits and keeps everything on your machine.

## How it works

```mermaid
flowchart LR
    A[PDFs in data/] --> B[Load and chunk]
    B --> C[Embed chunks<br/>all-MiniLM-L6-v2]
    C --> D[(Chroma<br/>vector store)]
    E[User question] --> F[Embed query]
    F --> D
    D --> G[Top-k relevant chunks]
    G --> H[Grounding prompt + LLM<br/>Ollama or Gemini]
    H --> I[Answer + cited sources]
```

1. **Ingestion (occasional).** PDFs in `data/` are loaded, split into overlapping chunks, embedded with a local sentence-transformers model, and written to a persisted Chroma vector store.
2. **Query (every question).** The question is embedded, the most similar chunks are retrieved, and those chunks are passed to the language model inside a grounding prompt that instructs it to answer only from the provided context.
3. **Generation.** The model returns an answer (or an honest refusal), and the UI shows the source chunks behind it.

## Tech stack

- **Orchestration:** LangChain
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (local, via `langchain-huggingface`)
- **Vector store:** ChromaDB (persisted to disk)
- **Generation:** Ollama for local, private inference, or Google Gemini (`gemini-2.5-flash-lite` via `langchain-google-genai`) for the hosted demo
- **PDF parsing:** pypdf via LangChain's `PyPDFLoader`
- **UI:** Streamlit

## Run it locally

Prerequisites: Python 3.12, and either Ollama (for local generation) or a Gemini API key.

```bash
git clone https://github.com/cqaxo/papertrail.git
cd papertrail
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For local generation, install [Ollama](https://ollama.com) and pull the default model:

```bash
ollama pull llama3.1:8b
```

Then start the app:

```bash
streamlit run app.py
```

The repo ships with a prebuilt vector store (the NIST demo corpus), so you can ask questions right away.

### Use your own documents

```bash
# drop your PDFs into data/, then rebuild the store
rm -rf chroma_db
python ingest.py
streamlit run app.py
```

### Configuration

- `OLLAMA_MODEL`: the local model to use (default `llama3.1:8b`). For example: `export OLLAMA_MODEL=qwen2.5:14b`.
- `GEMINI_API_KEY`: if set, generation uses Google Gemini (`gemini-2.5-flash-lite`) instead of Ollama. This lets the app run without a local GPU and is what the hosted demo uses. The key is read from the environment and is never stored in the repo.

## Design decisions

**RAG over fine-tuning.** Retrieval keeps answers grounded in the current documents and makes the source of every answer inspectable, without the cost and staleness of training a model on the content.

**Local embeddings, configurable generation.** Embeddings run locally with a small, fast model, so indexing is free and private. Generation runs locally on Ollama by default and can switch to a cloud API. Nothing about the pipeline assumes a particular machine.

**Runtime provider switch.** Generation uses Google Gemini when `GEMINI_API_KEY` is set and falls back to local Ollama when it is not. Using the key's presence as the switch means local development needs no configuration and the deployed app needs no code change. Local runs stay free and fully private, while the hosted demo can run on a host with no GPU.

**One source of truth for embeddings.** The same `get_embeddings()` function is used to build and to query the store, so the two can never drift to different models (a mismatch would silently break similarity search). The embedding model and the store are cached so they load once per process rather than on every question.

**Grounding and honest refusal.** The prompt instructs the model to answer only from the retrieved context and to reply with a fixed phrase, "I don't know based on the provided documents," when the answer is not there. The fixed phrase also makes the refusal easy to detect for evaluation later.

## Limitations and roadmap

Retrieval quality is the main known limitation. The small embedding model matches largely on keyword overlap rather than deeper meaning, and naive page-based chunking can pull in boilerplate (titles, author blocks, page footers, reference lists) that crowds out the passage that actually answers a question. On clean, well-structured documents this works well; on messy ones, the right chunk is sometimes missed. This is documented openly because measuring and improving it is the point of Phase 2.

Phase 2 plans:

- An evaluation harness (for example RAGAS or DeepEval) to produce before-and-after numbers, so retrieval changes can be proven rather than asserted.
- Better retrieval: stripping boilerplate before indexing, a stronger embedding model, and cross-encoder reranking of the top results.
- Multi-tenant access control, so different users can be scoped to different documents.
- Possibly replacing the LangChain wrappers with a direct retrieve-and-generate loop for transparency.