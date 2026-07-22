# Hybrid RAG Pipeline

A production-oriented Retrieval-Augmented Generation API for document question-answering. Upload PDFs, DOCX, or TXT files; ask questions; get answers grounded in those documents with verifiable source citations.

Built with FastAPI, LangChain, and PostgreSQL/pgvector. Deployed on AWS (ECS Fargate + RDS PostgreSQL); the deployment is teardown/redeploy-on-demand, so this repo is built to run locally with a single command.

<!-- Add a screenshot of the Swagger UI at /docs or the test console here -->
<!-- ![RAG API](docs/screenshot.png) -->

---

## What makes this more than a basic RAG demo

Most RAG examples are: embed chunks → cosine search → stuff into an LLM. This one implements the retrieval and production concerns a real system needs:

- **Hybrid retrieval** — dense vector search (pgvector) *and* BM25 keyword search, fused with Reciprocal Rank Fusion, so it handles both semantic paraphrase and exact-term matches (product names, IDs) that pure embeddings miss.
- **Cross-encoder reranking** — a two-stage funnel: cheap bi-encoder retrieval narrows to candidates, an expensive cross-encoder accurately ranks the final few.
- **Parent-child chunking** — retrieves on small precise chunks, generates on their larger parent context, decoupling retrieval accuracy from answer completeness.
- **Grounded citations** — the source metadata is injected into the LLM's context so citations are derived from real chunk data, not hallucinated, and the API returns a separate machine-verifiable sources array.
- **Production hardening** — per-IP rate limiting, layered prompt-injection defense, async background ingestion with job tracking, token-level response streaming, and per-query cost telemetry.

---

## Architecture

### Retrieval pipeline

```
Query
  ├─→ Semantic search (pgvector, cosine)     → top 50 candidates
  └─→ BM25 keyword search (in-memory index)  → top 50 candidates
              ↓
      Reciprocal Rank Fusion (RRF)           → merged, deduped, top 10
              ↓
      Parent resolution (child → parent text) → full-context windows
              ↓
      Cross-encoder rerank (ms-marco-MiniLM) → top 3
              ↓
      gpt-4o-mini + citation prompt          → grounded answer
```

### Ingestion pipeline

Uploads are processed asynchronously so the API responds immediately:

```
POST /upload → 202 Accepted + job_id   (returns instantly)
      ↓ (background task)
  parse (PDF/DOCX/TXT) → parent chunks (2000 chars) → stored in parent_documents
      ↓
  child chunks (500 chars) → embedded (text-embedding-3-small) → pgvector
      ↓
  BM25 index updated
      ↓
  processing_jobs status → complete    (poll via GET /status/{job_id})
```

### Components

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, endpoints, lifespan startup, LLM answer chain |
| `retrieval.py` | Hybrid retrieval — semantic + BM25 + RRF + parent resolution + rerank |
| `ingestion.py` | Document parsing, parent-child chunking, embedding, BM25 build/rebuild |
| `database.py` | Postgres/pgvector wiring, table creation, job & cost tracking |
| `security.py` | Layered prompt-injection defense (regex + LLM classifier) |
| `models.py` | Pydantic request/response schemas |
| `config.py` | Configuration and tunable constants |
| `test_console.html` | Browser-based manual endpoint tester |

---

## Tech stack

**API:** FastAPI, Uvicorn
**Retrieval:** LangChain, pgvector (dense), rank-bm25 (sparse), sentence-transformers (cross-encoder reranking)
**Models:** OpenAI `text-embedding-3-small` (embeddings), `gpt-4o-mini` (generation + injection classifier)
**Data:** PostgreSQL 16 with the pgvector extension
**Infra:** Docker, Docker Compose (local), AWS ECS Fargate + RDS PostgreSQL + ECR + Secrets Manager (cloud)

---

## Running locally

The whole stack — API plus a pgvector-enabled Postgres — runs with Docker Compose. You only need Docker and an OpenAI API key.

### 1. Clone

```bash
git clone https://github.com/prem-goswami/hybrid-rag-pipeline.git
cd hybrid-rag-pipeline
```

### 2. Provide your OpenAI key

Docker Compose reads `OPENAI_API_KEY` from a `.env` file next to `docker-compose.yml`. Create one:

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

(`.env` is gitignored — your key never gets committed.)

### 3. Start

```bash
docker compose up --build
```

First build takes a few minutes (it bakes in the CPU PyTorch build and the cross-encoder model). When you see `[Startup] Ready.`, the API is up. Compose also starts Postgres and enables pgvector automatically.

### 4. Use it

- **Interactive API docs (Swagger):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Manual test console:** open `test_console.html` in a browser and set the **Base URL** field to `http://localhost:8000` (it ships pointed at a cloud URL — change it to localhost).

A typical flow: `POST /upload` a document → poll `GET /status/{job_id}` until `complete` → `POST /query` with a question.

> **Note on Postgres data:** Compose stores Postgres data in a named volume (`pgdata`), so your ingested documents survive `docker compose down` and come back on the next `up`. The BM25 index is rebuilt from Postgres automatically on every startup.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF/DOCX/TXT. Returns `202` + `job_id`; processing runs in the background. |
| `GET` | `/status/{job_id}` | Poll ingestion status: `pending` → `processing` → `complete` / `failed`. |
| `POST` | `/query` | Ask a question. Returns a grounded answer + a sources array. Rate limited, injection-filtered. |
| `POST` | `/query/stream` | Same as `/query` but streams the answer token-by-token (SSE), with sources appended after a `__SOURCES__` delimiter. |
| `GET` | `/documents` | List ingested documents and their chunk counts. |
| `DELETE` | `/document/{document_id}` | Delete a document's chunks (by filename) and rebuild the BM25 index. |
| `GET` | `/costs` | Per-query token/cost telemetry for recent queries. |
| `GET` | `/health` | Liveness + database reachability. |

### Example query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

Returns an `answer` with inline `[Source: filename, Page: N]` citations and a `sources` array carrying chunk IDs, rerank scores, and matched snippets for verification.

---

## Design decisions worth calling out

**Why PostgreSQL/pgvector instead of a dedicated vector database.** The system needs vectors *and* parent chunk text *and* job state *and* cost telemetry. A dedicated vector store handles only the first; the rest need a relational database anyway. pgvector unifies all four in one instance with transactional consistency — one datastore to run instead of two to keep in sync. At this scale the performance tradeoff is irrelevant and the operational simplicity is real.

**Why hybrid retrieval.** Dense embeddings capture meaning but are weak on rare exact tokens (a specific product name may not surface for a direct question about it); BM25 nails exact matches but is blind to paraphrase. Running both and fusing with RRF gives recall neither achieves alone. RRF fuses on *rank* rather than *score*, which sidesteps the problem that cosine similarity and BM25 scores aren't on comparable scales.

**Why a cross-encoder only at the end.** A cross-encoder scores query-document pairs with full cross-attention — far more accurate than embedding similarity, but it can't be precomputed, so scoring the whole corpus is infeasible. Hence the funnel: cheap retrieval for 50 candidates, RRF to 10, expensive reranking only on those 10.

**Why parent-child chunking.** Small chunks embed precisely (one idea, clean vector); large chunks give the LLM enough context to answer well. These conflict, so the system embeds and searches on small children, then swaps in the larger parent text before generation — retrieve precisely, generate with context.

**Why the container is stateless.** The BM25 index is rebuilt from Postgres on every startup rather than relied on as a persisted file, because container filesystems are ephemeral. Postgres is the single source of truth; the index is a derived cache. This is what makes redeploy-on-demand clean — a fresh container comes up fully working against the existing database.

---

## Production hardening

- **Rate limiting** (`slowapi`) — per-IP limits on the expensive endpoints (`/query`, `/query/stream`, `/upload`), returning `429` when exceeded, so no single caller can drain the OpenAI budget on an unauthenticated API.
- **Prompt-injection defense** (`security.py`) — layered input validation on `/query`: a deterministic regex pre-filter catches known injection phrasings for free (whitespace-evasion-resistant), and an LLM classifier catches rephrased/semantic attempts the regex misses. Fails closed, with a distinct response for a genuine block versus a classifier outage, and a uniform user-facing message that doesn't leak which layer fired.
- **Async ingestion** — uploads return immediately with a job ID; heavy parsing/embedding runs in the background with status tracked in Postgres.
- **Cost telemetry** — every query logs token counts and USD cost by model to a `query_costs` table, surfaced via `/costs`.
- **Streaming** — `/query/stream` streams tokens as they're generated for a responsive UX.

---

## Deployment

The system runs on AWS: the container image is stored in **ECR**, run on **ECS Fargate** (via ECS Express Mode), against **RDS PostgreSQL 16** with the pgvector extension, with the OpenAI key injected from **Secrets Manager** and the database VPC-isolated (no public endpoint; reachable only from the application's security group).

Because the always-on AWS cost isn't justified for a portfolio project between demos, the live infrastructure is torn down and redeployed on demand. The stateless-container design (tables self-create on startup, BM25 rebuilds from Postgres) means a redeploy from the ECR image against the retained RDS instance comes up fully working with data intact.

---

## Roadmap / known limitations

Being explicit about what's *not* done, since knowing the gaps matters as much as the features:

- **Injection defense covers `/query` but not `/query/stream`** — the classifier guard should be applied to the streaming endpoint too.
- **BM25 index and rate-limit counters are per-container** (in-memory / local pickle). This is why the deployment runs a single task; scaling horizontally needs BM25 moved to Postgres full-text search and rate-limit state moved to Redis/ElastiCache.
- **Rate-limit key is the direct client IP** — behind a load balancer this may key on the LB's IP; production would read the real client IP from `X-Forwarded-For`.
- **Secrets partially wired** — the OpenAI key is injected from Secrets Manager, but the database credentials are still passed as plain environment variables (they're embedded in connection strings); these should be composed from individually-injected secret values.
- **No indirect-injection defense** — the query filter addresses direct injection, not malicious instructions hidden inside uploaded documents.
- **No retrieval evaluation set** — quality is verified by hand; recall@k / nDCG on a golden set would let reranking's contribution be measured rather than assumed.
- **CORS is fully open and there is no authentication** — appropriate for a local/demo project, not for public exposure.

---

## Project structure

```
.
├── main.py               # FastAPI app + endpoints
├── retrieval.py          # hybrid retrieval pipeline
├── ingestion.py          # parsing, chunking, embedding, BM25
├── database.py           # Postgres/pgvector + job/cost tracking
├── security.py           # prompt-injection defense
├── models.py             # Pydantic schemas
├── config.py             # configuration
├── test_console.html     # browser endpoint tester
├── Dockerfile
├── docker-compose.yml    # local stack: API + pgvector Postgres
├── init.sql              # enables pgvector on the local DB
└── requirements.txt
```
