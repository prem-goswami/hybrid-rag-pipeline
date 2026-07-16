import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

# Raw psycopg3 connection string for direct SQL (job state, document listing)
PG_DSN = os.getenv(
    "PG_DSN", "host=localhost port=5432 dbname=wikidb user=pguser password=pass"
)

# ── Vector Store ──────────────────────────────────────────
COLLECTION_NAME = "week3_rag_docs"
VECTOR_SIZE = 1536  # text-embedding-3-small output dimensions

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── Models ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Retrieval ─────────────────────────────────────────────
SEMANTIC_TOP_K = 50  # candidates pulled from pgvector before RRF
BM25_TOP_K = 50  # candidates pulled from BM25 before RRF
RERANK_TOP_K = 10  # candidates passed to cross-encoder
FINAL_TOP_K = 3  # chunks passed to LLM

# ── BM25 ──────────────────────────────────────────────────
BM25_INDEX_PATH = "bm25_index.pkl"  # persisted index location

# ── File Upload ───────────────────────────────────────────
UPLOAD_DIR = "uploads"  # temp PDF storage
ALLOWED_EXTENSION = {".pdf", ".txt", ".docx"}

# config.py additions
EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small
INPUT_COST_PER_1K = 0.00015  # gpt-4o-mini input
OUTPUT_COST_PER_1K = 0.0006  # gpt-4o-mini output
