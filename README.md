# Enterprise Hybrid-RAG & Semantic Cache Gateway

A production-grade hybrid retrieval gateway that pairs **dense vector search** (FAISS) with **BM25 sparse lexical scoring**, fuses results using **Reciprocal Rank Fusion (RRF)**, reranks with a **cross-encoder**, and caches query embeddings with a **semantic similarity cache**. Responses stream via **Server-Sent Events (SSE)**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│            Semantic Cache (in-memory)        │
│  cosine_similarity(q_emb, stored) >= 0.96   │
│  → Cache HIT: stream cached response         │
└────────────────────┬────────────────────────┘
                     │ MISS
                     ▼
┌─────────────────────────────────────────────┐
│              Hybrid Retriever                │
│                                             │
│   ┌──────────────┐   ┌──────────────────┐   │
│   │  BM25 (rank- │   │  Dense (FAISS /  │   │
│   │  bm25)       │   │  numpy fallback) │   │
│   └──────┬───────┘   └────────┬─────────┘   │
│          │  top-N candidates  │              │
│          └─────────┬──────────┘              │
│                    ▼                         │
│        Reciprocal Rank Fusion (k=60)         │
└────────────────────┬────────────────────────┘
                     │ fused ranking
                     ▼
┌─────────────────────────────────────────────┐
│           Cross-Encoder Reranker             │
│  token overlap + keyword density +           │
│  positional score + title match bonus        │
└────────────────────┬────────────────────────┘
                     │ top-k documents
                     ▼
┌─────────────────────────────────────────────┐
│         Answer Synthesis + SSE Stream        │
│  FastAPI StreamingResponse → text/event-     │
│  stream → browser EventSource               │
└─────────────────────────────────────────────┘
                     │
                     ▼
           Store in Semantic Cache
```

### Key components

| Component | File | Description |
|-----------|------|-------------|
| **FastAPI app** | `backend/main.py` | SSE streaming endpoint, stats, health, cache flush |
| **Hybrid retriever** | `backend/retriever.py` | BM25 + FAISS dense search, RRF fusion |
| **Semantic cache** | `backend/cache.py` | Cosine similarity cache with TTL eviction |
| **Reranker** | `backend/reranker.py` | Cross-encoder scoring via token overlap + keyword features |
| **Corpus** | `backend/corpus.py` | 27 technical documents across 8 domains |
| **Frontend** | `frontend/index.html` | Full chat UI with pipeline trail, source pills, latency breakdown |

---

## Quickstart

### Local (no Docker)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start the API server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 4. Open the frontend
open frontend/index.html           # macOS
# or just open the file in your browser
```

### With Docker Compose

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.
Open `frontend/index.html` directly in your browser (no separate server needed).

---

## API Reference

### `POST /api/query`

Stream a hybrid RAG response.

**Request body:**
```json
{
  "query": "How does Reciprocal Rank Fusion work?",
  "top_k": 5
}
```

**Response:** `text/event-stream` with the following event sequence:

| Event | Payload | Description |
|-------|---------|-------------|
| `cache_hit` | `{similarity, latency_ms}` | Emitted instead of retrieval events when cache hits |
| `retrieval_start` | `{query}` | Hybrid retrieval beginning |
| `retrieval_done` | `{n_candidates, latency_ms, top_scores}` | Retrieval + RRF fusion complete |
| `rerank_done` | `{n_results, latency_ms, top_titles}` | Cross-encoder reranking complete |
| `token` | `{text}` | Streaming token chunk (many events) |
| `done` | `{cache_hit, answer, docs, latency_ms, ...}` | Full response with all latency breakdowns |
| `error` | `{message}` | On exception |

---

### `GET /api/stats`

Returns system-level statistics.

```json
{
  "total_queries": 42,
  "cache": {
    "hit_rate": 0.4286,
    "total_requests": 42,
    "hits": 18,
    "misses": 24,
    "cache_size": 24
  },
  "avg_latency_ms": 312.5,
  "avg_retrieval_latency_ms": 48.2,
  "avg_rerank_latency_ms": 12.1,
  "uptime_seconds": 3600.0
}
```

---

### `GET /api/health`

```json
{
  "status": "healthy",
  "components": {
    "retriever": true,
    "reranker": true,
    "cache": true
  },
  "uptime_seconds": 3600.0
}
```

---

### `DELETE /api/cache`

Flush all entries from the semantic cache.

```json
{
  "status": "cleared",
  "entries_removed": 18
}
```

---

## Configuration

Key parameters can be adjusted at initialization in `backend/main.py`:

| Parameter | Default | Where |
|-----------|---------|-------|
| Cache similarity threshold | `0.96` | `SemanticCache(threshold=...)` |
| Cache TTL | `3600s` | `SemanticCache(ttl_seconds=...)` |
| RRF k constant | `60` | `HybridRetriever(rrf_k=...)` |
| Dense retrieval candidates | `20` | `HybridRetriever(dense_candidates=...)` |
| Sparse retrieval candidates | `20` | `HybridRetriever(sparse_candidates=...)` |

---

## Running Tests

```bash
# From the project root
pytest tests/ -v

# With coverage
pytest tests/ -v --tb=short
```

Tests cover:

- RRF fusion formula correctness and edge cases
- Cache cosine similarity threshold boundary conditions
- Cache TTL expiry (via time mocking)
- Cache hit/miss statistics accounting
- Hybrid retrieval relevance and score ordering
- Deterministic embedding properties

---

## Extending to Production

**Replace the numpy embedding fallback** with a real model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(query, normalize_embeddings=True)
```

**Use real Redis** for the semantic cache:
```python
import redis
from redisvl.index import SearchIndex
# Enable the redis service in docker-compose.yml
```

**Replace the simulated cross-encoder** with a real model:
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc.text) for doc in docs])
```

**Swap the corpus** for your own documents by populating `backend/corpus.py`
with chunks from your knowledge base, or by implementing a loader that reads
from a vector database (Pinecone, Weaviate, Qdrant, pgvector).
