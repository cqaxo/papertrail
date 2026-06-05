# PaperTrail

A "chat with your documents" RAG application that answers questions grounded in
source text, returns citations, and refuses to answer when the documents don't
support a response. Built incrementally; each item below maps to a single commit.

## Stack

- **Retrieval/orchestration:** LangChain (Phase 1) → direct retrieval loop (Phase 2)
- **Vector store:** Chroma
- **Embeddings:** sentence-transformers (local, no API cost)
- **Generation:** Claude
- **UI:** Streamlit
- **Evaluation:** RAGAS / DeepEval

## Phase 1 — MVP

Goal: a working, deployed, end-to-end RAG app by the end of weekend one.

- [x] `chore: project scaffold`
- [ ] `feat: document loading and chunking`
- [ ] `feat: embeddings and vector store`
- [ ] `feat: retrieval`
- [ ] `feat: grounded answer generation` — includes refusal behavior ("I don't
  know" when retrieved context is insufficient)
- [ ] `feat: streamlit chat UI`
- [ ] `feat: surface source citations in UI`
- [ ] `docs: architecture and design decisions in README` — written to explain
  *why*, not just *what*
- [ ] `chore: deploy to streamlit community cloud`

## Phase 2 — Differentiation

Goal: turn a working demo into a portfolio piece that gets a callback.

- [ ] `test: evaluation harness` — 50+ Q&A test cases; baseline faithfulness and
  hallucination metrics via RAGAS/DeepEval
- [ ] `refactor: replace LangChain core with direct retrieval loop` — own the
  retrieve → generate path end to end
- [ ] `perf: tune chunking and retriever, record before/after metrics` — capture
  the measured reliability improvement (e.g. hallucination rate before vs. after)
- [ ] `feat: multi-tenant access control via metadata filtering` — keep user A
  from retrieving user B's documents
- [ ] `docs: add eval results and architecture decisions`

## Design decisions (to document as they're made)

- **Why RAG over fine-tuning** — freshness and control over the source corpus
  without retraining
- **Chunk size and overlap** — rationale for the chosen values
- **top-k** — how many chunks are retrieved per query, and why
- **Local embeddings + API generation** — cost and deployment tradeoffs
- **Refusal behavior** — grounding over fluency
- **Before/after reliability metrics** — the measured impact of tuning
