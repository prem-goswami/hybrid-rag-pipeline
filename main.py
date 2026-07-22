import os
import shutil
import asyncio
import sys
import json
import tiktoken
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from config import COLLECTION_NAME

# Windows uses an event loop (ProactorEventLoop) which is compatible with psycopg3's low-level TCP database socket implementations.
# We just need to tell Python to use the older, universally compatible WindowsSelectorEventLoopPolicy right as the application starts.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from security import regex_prefilter, llm_classifier



from config import (
    UPLOAD_DIR,
    ALLOWED_EXTENSION,
    LLM_MODEL,
    OPENAI_API_KEY,
    EMBEDDING_COST_PER_1K,
    INPUT_COST_PER_1K,
    OUTPUT_COST_PER_1K,
)
from database import (
    init_vectorstore_table,
    init_job_table,
    init_parent_docs_table,
    init_costs_table,
    create_job,
    get_job,
    get_db_conn,
    log_query_cost,
)
from ingestion import ingest_file, rebuild_bm25_from_db, count_tokens
from retrieval import hybrid_retrieve
from models import (
    UploadResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    DocumentListResponse,
    DocumentRecord,
    DeleteResponse,
)


# ── Lifespan ──────────────────────────────────────────────
# Runs startup logic before the server accepts requests
# and teardown logic when the server shuts down
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("[Startup] Initializing database tables...")
    Path(UPLOAD_DIR).mkdir(exist_ok=True)  # create uploads/ dir if missing
    await init_vectorstore_table()  # create pgvector table if not exists
    await init_job_table()  # create processing_jobs table if not exists
    await init_parent_docs_table()  # creating parents table if not exsist
    await init_costs_table()  # creating cost table if not exsists
    await rebuild_bm25_from_db()  # pickle doesn't survive container restarts — rebuild from Postgres
    print("[Startup] Ready.")
    yield
    # SHUTDOWN (nothing to clean up for now)
    print("[Shutdown] Server stopping.")


app = FastAPI(
    title="hybrid-rag-pipeline",
    description="Production RAG backend with hybrid retrieval and reranking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── LLM + Prompt (module level — loaded once) ─────────────
llm = ChatOpenAI(
    model=LLM_MODEL, temperature=0, streaming=True, openai_api_key=OPENAI_API_KEY
)

CITATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise assistant. Answer the user's question 
    using ONLY the context provided below. 

    For every claim you make, cite the source using [Source: filename, Page: N].
    If the answer is not in the context, say "I don't have enough information to answer this."

    Context:
    {context}""",
        ),
        ("human", "{input}"),
    ]
)

# Without this, create_stuff_documents_chain only stuffs page_content into
# {context} — the LLM never sees filename/page, so it fabricates citations
# from whatever heading text happens to be in the chunk.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {source_filename}, Page: {page}]\n{page_content}"
)

stuff_chain = create_stuff_documents_chain(
    llm=llm, prompt=CITATION_PROMPT, document_prompt=DOCUMENT_PROMPT
)


# ── Helpers ───────────────────────────────────────────────
def validate_file_extension(file: UploadFile) -> None:
    """Raise HTTPException if file extension is not supported."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Supported types are: {', '.join(ALLOWED_EXTENSION)}",
        )


# ── Stream Response Query ─────────────────────────────────────────────
async def stream_query_response(
    question: str, documents: list[Document], sources: list[SourceChunk]
):
    """
    Async generator — yields tokens as the LLM produces them,
    then yields a final delimited block containing source citations.
    """

    # Reconstruct the full text sent to the LLM to calculate prompt tokens
    context_text = "\n\n".join([doc.page_content for doc in documents])
    full_prompt = f"Context:\n{context_text}\n\nQuestion:\n{question}"

    encoding = tiktoken.encoding_for_model(LLM_MODEL)
    prompt_tokens = len(encoding.encode(full_prompt))

    generated_answer = ""

    # Why astream() over invoke() — the actual mechanism: invoke() sends the request, blocks,
    # and returns only when OpenAI's API has finished generating every token. astream()
    # opens the same request but reads the response as a Server-Sent Events stream from OpenAI's API itself
    async for token in stuff_chain.astream({"input": question, "context": documents}):
        generated_answer += token
        yield token

    # Stream is over. Calculate the output tokens
    completion_tokens = len(encoding.encode(generated_answer))

    input_cost = (prompt_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (completion_tokens / 1000) * OUTPUT_COST_PER_1K

    # We use create_task so it doesn't block the final sources from sending!
    asyncio.create_task(
        log_query_cost(
            question=question,
            num_chunks=len(documents),
            context_tokens=len(encoding.encode(context_text)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding_cost=0.0,
            input_cost=input_cost,
            output_cost=output_cost,
        )
    )

    # Sources arrive AFTER the answer is fully streamed.
    # Client-side, split on this delimiter to separate answer text from citation data.
    sources_json = json.dumps([s.model_dump() for s in sources], default=str)
    yield f"\n\n__SOURCES__{sources_json}"


# ── Build Parent Context Preview ─────────────────────────────────────────────
def build_context_preview(parent_text: str, child_text: str, window: int = 250) -> str:
    """Slice of parent_text centered on where child_text sits, padded by `window` chars each side."""
    idx = parent_text.find(child_text[:50])  # match on a stable prefix
    if idx == -1:
        return parent_text[: window * 2]  # fallback if the exact substring can't be located
    start = max(0, idx - window)
    end = min(len(parent_text), idx + len(child_text) + window)
    return f"{'…' if start > 0 else ''}{parent_text[start:end]}{'…' if end < len(parent_text) else ''}"



# ── Endpoints ─────────────────────────────────────────────


# POST /upload
@app.post("/upload", response_model=UploadResponse, status_code=202)
@limiter.limit("10/minute")
async def upload_file(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept a file upload, save to disk, return job_id immediately.
    Processing happens in the background — poll /status/{job_id} for updates.
    202 Accepted = request received, processing not yet complete.
    """
    # Step 1 — validate file type before touching disk
    validate_file_extension(file)

    # Step 2 — save file to uploads/ directory
    upload_path = Path(UPLOAD_DIR) / file.filename
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Step 3 — create job record in DB, get job_id back
    job_id = await create_job(file.filename)

    # Step 4 — register background task and return immediately
    background_tasks.add_task(
        ingest_file, job_id=job_id, file_path=str(upload_path), filename=file.filename
    )

    return UploadResponse(
        job_id=job_id,
        message="file upload accepted. Processing in background.",
        filename=file.filename,
    )


# GET /status/{job_id}
@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Poll this endpoint to check ingestion progress."""
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(**job)


# POST /query
@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request:Request, payload: QueryRequest):
    """
    Run hybrid retrieval + reranking + LLM answer generation.
    Returns answer with cited sources.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    
    # ── Security: prompt-injection defense ──
    # Layer 1 — deterministic regex pre-filter (free, instant)
    if regex_prefilter(payload.question):
        raise HTTPException(
            status_code=400,
            detail="Query blocked: potential prompt injection detected.",
        )

    # Layer 2 — LLM classifier (fail-closed on error, but distinguish block from outage)
    try:
        if await asyncio.get_event_loop().run_in_executor(
            None, llm_classifier, payload.question
        ):
            raise HTTPException(
                status_code=400,
                detail="Query blocked: potential prompt injection detected.",
            )
    except HTTPException:
        raise  # re-raise the intentional 400 — don't let the block below swallow it
    except Exception:
        raise HTTPException(
            status_code=503, detail="Safety check unavailable, please resubmit."
        )
    

    # Step 1 — hybrid retrieval (sync — run in thread pool)
    try:
        chunks = await hybrid_retrieve(payload.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Upload files before querying.",
        )

    # Step 2 — convert retrieved dicts back to LangChain Document objects for stuff_chain
    documents = [
        Document(page_content=c["text"], metadata={"page": 0, **c["metadata"]})
        for c in chunks
    ]

    # Count context tokens BEFORE calling the LLM
    context_text = "\n\n".join(d.page_content for d in documents)
    context_tokens = count_tokens(context_text)
    question_tokens = count_tokens(payload.question)
    prompt_tokens = context_tokens + question_tokens

    # Step 3 — LLM answer generation (sync — run in thread pool)
    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stuff_chain.invoke(
                {"input": payload.question, "context": documents}
            ),
        )
        answer = answer.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    completion_tokens = count_tokens(answer)
    # Cost math
    embedding_cost = (question_tokens / 1000) * EMBEDDING_COST_PER_1K
    input_cost = (prompt_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (completion_tokens / 1000) * OUTPUT_COST_PER_1K

    await log_query_cost(
        question=payload.question,
        num_chunks=len(chunks),
        context_tokens=context_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_cost=embedding_cost,
        input_cost=input_cost,
        output_cost=output_cost,
    )

    # Step 4 — build source citations from chunk metadata
    sources = [
        SourceChunk(
            chunk_id=str(c["chunk_id"]),
            source=c["metadata"].get("source_filename", "unknown"),
            page=c["metadata"].get("page", 0),
            content_preview=build_context_preview(c["text"], c.get("child_text", c["text"])),
            matched_snippet=c.get("child_text", c["text"])[:200],
            rerank_score=c.get("rerank_score", 0.0),
            original_rank=c.get("original_rank", 0),
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources, question=payload.question)


# POST /query/stream
@app.post("/query/stream")
@limiter.limit("10/minute")
async def query_stream(request: Request, payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # ── Security: prompt-injection defense ──
    # Layer 1 — deterministic regex pre-filter (free, instant)
    if regex_prefilter(payload.question):
        raise HTTPException(
            status_code=400,
            detail="Query blocked: potential prompt injection detected.",
        )

    # Layer 2 — LLM classifier (fail-closed on error, but distinguish block from outage)
    try:
        if await asyncio.get_event_loop().run_in_executor(
            None, llm_classifier, payload.question
        ):
            raise HTTPException(
                status_code=400,
                detail="Query blocked: potential prompt injection detected.",
            )
    except HTTPException:
        raise  # re-raise the intentional 400 — don't let the block below swallow it
    except Exception:
        raise HTTPException(
            status_code=503, detail="Safety check unavailable, please resubmit."
        )

    # Retrieval is still synchronous (BM25, cross-encoder, pgvector) — still needs run_in_executor
    chunks = await hybrid_retrieve(payload.question)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    documents = [
        Document(page_content=c["text"], metadata={"page": 0, **c["metadata"]})
        for c in chunks
    ]
    sources = [
        SourceChunk(
            chunk_id=str(c["chunk_id"]),
            source=c["metadata"].get("source_filename", "unknown"),
            page=c["metadata"].get("page", 0),
            content_preview=build_context_preview(c["text"], c.get("child_text", c["text"])),
            matched_snippet=c.get("child_text", c["text"])[:200],
            rerank_score=c.get("rerank_score", 0.0),
            original_rank=c.get("original_rank", 0),
        )
        for c in chunks
    ]

    # No response_model here — StreamingResponse and Pydantic validation don't mix
    return StreamingResponse(
        stream_query_response(payload.question, documents, sources),
        media_type="text/event-stream",
    )


# GET /documents
@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all unique documents stored in pgvector.
    Groups chunks by source_filename and counts chunks per document.
    """
    try:
        async with get_db_conn() as conn:
            rows = await conn.execute(f"""
                SELECT
                    langchain_metadata->>'source_filename' AS filename,
                    COUNT(*)                               AS chunk_count
                FROM {COLLECTION_NAME}
                WHERE langchain_metadata->>'source_filename' IS NOT NULL
                GROUP BY langchain_metadata->>'source_filename'
                """)
            records = await rows.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    documents = [
        DocumentRecord(
            document_id=row[0],  # row[0] is filename
            filename=row[0],
            chunk_count=row[1],  # row[1] is chunk_count
            uploaded_at=datetime.now(timezone.utc),
        )
        for row in records
    ]

    return DocumentListResponse(documents=documents, total=len(documents))


# DELETE /document/{document_id}
@app.delete("/document/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """
    Delete all chunks belonging to a document by source_filename.
    Rebuilds BM25 index from remaining DB chunks after deletion.
    """
    try:
        async with get_db_conn() as conn:
            # Count chunks before deletion so we can report how many were removed
            count_row = await conn.execute(
                f"""
                SELECT COUNT(*) FROM {COLLECTION_NAME}
                WHERE langchain_metadata->>'source_filename' = %s
                """,
                (document_id,),
            )
            count = (await count_row.fetchone())[0]

            if count == 0:
                raise HTTPException(
                    status_code=404, detail=f"No document found with id '{document_id}'"
                )

            # Delete all chunks for this document
            await conn.execute(
                f"""
                DELETE FROM {COLLECTION_NAME}
                WHERE langchain_metadata->>'source_filename' = %s
                """,
                (document_id,),
            )

    except HTTPException:
        raise  # re-raise 404 without wrapping it in 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    # Rebuild BM25 from remaining DB chunks — deletion invalidates the incremental index
    await rebuild_bm25_from_db()

    return DeleteResponse(
        message=f"Document '{document_id}' deleted successfully.",
        document_id=document_id,
        chunks_deleted=count,
    )


# GET /health
@app.get("/health")
async def health():
    """Basic health check — returns degraded if DB is unreachable."""
    try:
        async with get_db_conn() as conn:
            await conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"degraded: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "vector_store": COLLECTION_NAME,
    }


# See Query Costs
@app.get("/costs")
async def get_costs():
    async with get_db_conn() as conn:
        recent_row = await conn.execute(
            "SELECT SUM(total_cost_usd), COUNT(*) FROM query_costs"
        )
        recent_cost, recent_count = await recent_row.fetchone()

        breakdown_rows = await conn.execute(
            "SELECT question, total_cost_usd, created_at FROM query_costs ORDER BY created_at DESC LIMIT 50"
        )
        breakdown = await breakdown_rows.fetchall()

    return {
        "recent_spend_usd": float(recent_cost or 0),
        "recent_queries_tracked": recent_count,
        "note": "Reflects only the last 10 queries — older cost records are pruned.",
        "recent_queries": [
            {"question": r[0], "cost_usd": float(r[1]), "timestamp": r[2].isoformat()}
            for r in breakdown
        ],
    }