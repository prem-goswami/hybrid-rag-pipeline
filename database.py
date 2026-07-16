import os
import uuid
import psycopg
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# LangChain's modern postgres database wrappers
from langchain_postgres import PGEngine, PGVectorStore
from langchain_openai import OpenAIEmbeddings

# Pulling configurations we locked down in config.py
from config import (
    DATABASE_URL,
    PG_DSN,
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

# ── PGEngine (LangChain Integration) ──────────────────────────────────
engine = PGEngine.from_connection_string(DATABASE_URL)


# def init_vectorstore_table():
#     engine.init_vectorstore_table(table_name=COLLECTION_NAME, vector_size=VECTOR_SIZE)


async def init_vectorstore_table():
    """Manually creates the vector table if it doesn't exist."""

    create_extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    
    # We use a multiline string (triple quotes) to hold the SQL
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS week3_rag_docs (
            langchain_id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(1536) NOT NULL,
            langchain_metadata JSONB
        );
    """

    # We create the index separately
    create_index_sql = """
        CREATE INDEX IF NOT EXISTS ix_week3_rag_docs_embedding 
        ON week3_rag_docs USING hnsw (embedding vector_cosine_ops);
    """

    # We use the connection to actually execute the string
    async with get_db_conn() as conn:
        await conn.execute(create_extension_sql)
        await conn.execute(create_table_sql)
        await conn.execute(create_index_sql)
        print("[Database] Verified table 'week3_rag_docs' exists.")


def get_vector_store() -> PGVectorStore:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    return PGVectorStore.create_sync(
        engine=engine, embedding_service=embeddings, table_name=COLLECTION_NAME
    )


# parents database
async def init_parent_docs_table():
    async with get_db_conn() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_documents (
                parent_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                parent_text    TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                page           INT
            )
        """)


# ── Raw psycopg3 Connection Layer ───────────────────────────
# createa a async manager that manages connections opening and closing with the db using conn using psycopg
@asynccontextmanager
async def get_db_conn():
    async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
        try:
            yield conn
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise


# ── Job State Tracking Table ───────────────────────────────────────
async def init_job_table():
    async with get_db_conn() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id         TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at   TIMESTAMPTZ,
                error          TEXT,
                chunks_created INT
            )
            """)


async def create_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    async with get_db_conn() as conn:
        await conn.execute(
            """
        INSERT INTO processing_jobs (job_id,filename,status,created_at)
        VALUES (%s, %s, 'pending', %s)
        """,
            (
                job_id,
                filename,
                datetime.now(timezone.utc),
            ),
        )
    return job_id


async def update_job(
    job_id: str, status: str, chunks_created: int = None, error: str = None
):
    async with get_db_conn() as conn:
        await conn.execute(
            """
            UPDATE processing_jobs 
            SET status = %s, completed_at = %s, chunks_created = %s, error = %s
            WHERE job_id = %s
            """,
            (
                status,
                datetime.now(timezone.utc),
                chunks_created,
                error,
                job_id,
            ),
        )


async def get_job(job_id: str):
    async with get_db_conn() as conn:
        row = await conn.execute(
            """
           SELECT * FROM processing_jobs 
           WHERE job_id = %s 
            """,
            (job_id,),
        )

        record = await row.fetchone()

    if record is None:
        return None
    cols = [
        "job_id",
        "filename",
        "status",
        "created_at",
        "completed_at",
        "error",
        "chunks_created",
    ]

    return dict(zip(cols, record))


# database.py additions


async def init_costs_table():
    async with get_db_conn() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_costs (
                id                   SERIAL PRIMARY KEY,
                query_id             UUID NOT NULL DEFAULT gen_random_uuid(),
                question             TEXT NOT NULL,
                created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                num_chunks_retrieved INT NOT NULL,
                context_tokens       INT NOT NULL,
                prompt_tokens        INT NOT NULL,
                completion_tokens    INT NOT NULL,
                embedding_cost_usd   NUMERIC(10,6) NOT NULL,
                input_cost_usd       NUMERIC(10,6) NOT NULL,
                output_cost_usd      NUMERIC(10,6) NOT NULL,
                total_cost_usd       NUMERIC(10,6) NOT NULL
            )
        """)


async def log_query_cost(
    question: str,
    num_chunks: int,
    context_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    embedding_cost: float,
    input_cost: float,
    output_cost: float,
):
    total_cost = embedding_cost + input_cost + output_cost
    async with get_db_conn() as conn:
        await conn.execute(
            """
            INSERT INTO query_costs
                (question, num_chunks_retrieved, context_tokens, prompt_tokens,
                 completion_tokens, embedding_cost_usd, input_cost_usd,
                 output_cost_usd, total_cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                question,
                num_chunks,
                context_tokens,
                prompt_tokens,
                completion_tokens,
                embedding_cost,
                input_cost,
                output_cost,
                total_cost,
            ),
        )

        # Prune the table to strictly 10 rows
        await conn.execute("""
            DELETE FROM query_costs 
            WHERE query_id NOT IN (
                SELECT query_id 
                FROM query_costs 
                ORDER BY created_at DESC 
                LIMIT 10
            );
            """)
