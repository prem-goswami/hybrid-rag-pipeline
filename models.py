from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Upload ────────────────────────────────────────────────
class UploadResponse(BaseModel):
    job_id: str
    message: str
    filename: str


# ── Job Status ────────────────────────────────────────────
class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "processing" | "complete" | "failed"
    filename: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    chunks_created: Optional[int] = None


# ── Query ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(
        default=3, ge=1, le=10
    )  # how many final chunks to pass to LLM — default 3


class SourceChunk(BaseModel):
    chunk_id: str
    source: str  # filename the chunk came from
    page: int
    content_preview: str  # first 200 chars of chunk text
    matched_snippet: str
    rerank_score: float
    original_rank: int  # pgvector rank before reranking


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str


# ── Documents ─────────────────────────────────────────────
class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]
    total: int


# ── Delete ────────────────────────────────────────────────
class DeleteResponse(BaseModel):
    message: str
    document_id: str
    chunks_deleted: int
