# Persian RAG — Query Rewrite + Routing + Hybrid Retrieval + Parent-Child Chunking

Built for a single ~625-page Persian document. API-based stack:

- **Embeddings**: Cohere `embed-multilingual-v3.0`
- **Vector store**: Qdrant (hybrid dense + sparse, RRF fusion)
- **Sparse encoding**: BM25 over Hazm-normalized Persian tokens
- **Rerank**: Cohere `rerank-multilingual-v3.0`
- **LLM**: OpenAI GPT-4o-mini (rewrite/routing) + GPT-4o (generation)
- **Orchestration**: LangGraph

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

If you don't have a Qdrant Cloud instance, run one locally instead:
```bash
docker run -p 6333:6333 qdrant/qdrant
# then set QDRANT_URL=http://localhost:6333 and leave QDRANT_API_KEY empty
```

## Ingest your document

```bash
python -m persian_rag.ingest /path/to/document.pdf
```

This parses the PDF, builds parent chunks (by heading/section, ~1800 tokens)
and child chunks (~350 tokens, overlap 60), embeds the children, and upserts
everything into Qdrant. Parent chunk text is stored in `parents.sqlite`
(not indexed — only fetched by id after retrieval).

## Query

```bash
python -m persian_rag.rag "متن سوال شما اینجا"
```

or import `run_query` from `persian_rag.rag` in your own code.

## Pipeline

```
query -> rewrite + intent classify -> route
                                        |- direct_answer      (chitchat / out-of-scope)
                                        |- simple_retrieval    (single factual query)
                                        `- multi_hop           (decomposed sub-queries)
                                              |
                                    hybrid retrieve (dense+sparse, children collection)
                                              |
                                    dedupe -> fetch parent chunks (sqlite)
                                              |
                                    Cohere rerank
                                              |
                                    generate (GPT-4o, grounded in parent context, cites page range)
```

## Files

- `persian_rag/config.py` — settings, all tunable knobs (chunk sizes, top_k, model names)
- `persian_rag/persian_text.py` — Hazm-based normalization/tokenization for sparse search
- `persian_rag/chunking.py` — parent-child chunk builder from parsed PDF
- `persian_rag/ingest.py` — PDF parse -> chunk -> embed -> upsert (CLI entry point)
- `persian_rag/vectorstore.py` — Qdrant collection setup, hybrid upsert, hybrid query
- `persian_rag/parent_store.py` — sqlite key/value store for parent chunk text
- `persian_rag/query_processing.py` — LLM-based rewrite + intent classification (structured JSON output)
- `persian_rag/retrieval.py` — hybrid retrieve -> dedupe -> fetch parents -> rerank
- `persian_rag/generate.py` — final grounded generation call
- `persian_rag/graph.py` — LangGraph state graph wiring all of the above with conditional routing
- `persian_rag/rag.py` — CLI / library entry point (`run_query`)

## Notes / things to tune once you see real results

- **Chunk sizes**: 350-token children / 1800-token parents is a reasonable start for a dense technical/legal-style document. If your 625-page doc is narrative prose, children can go up to ~500 tokens.
- **RRF weighting**: `vectorstore.py` currently fuses dense/sparse 50/50 via Qdrant's built-in RRF. If Persian sparse search is noisy on your document (common with poorly-OCR'd PDFs), you can down-weight it.
- **Rerank top_k**: retrieves 20 children per (sub-)query, reranks down to top 5 parents sent to generation. Adjust in `config.py`.
- **Multi-hop**: sub-queries from `query_processing.py` are retrieved independently and merged before rerank — this is a simple decomposition strategy, not iterative/agentic multi-hop. Good enough for most multi-part questions; say if you want the agentic version (re-query based on partial answers).
