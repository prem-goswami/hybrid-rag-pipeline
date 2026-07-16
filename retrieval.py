import pickle
from pathlib import Path

# Specialized cross-encoder neural layers for deep query-document cross-attention
from sentence_transformers import CrossEncoder

# Pulling configurations we locked down in config.py
from config import (
    COLLECTION_NAME,
    RERANKER_MODEL,
    SEMANTIC_TOP_K,
    BM25_TOP_K,
    RERANK_TOP_K,
    FINAL_TOP_K,
)
from database import get_db_conn, get_vector_store
from ingestion import load_bm25

# ── Cross-Encoder Neural Loading Matrix ──────────────────────────────────
# This is loaded once at the absolute module level when the server boots up.
# This prevents reloading ~90MB of model weights on every single HTTP network request loop!
print(
    f"[Retrieval] Initializing Cross-Encoder deep attention weights: {RERANKER_MODEL}"
)
cross_encoder = CrossEncoder(RERANKER_MODEL)


def semantic_search(query: str, top_k: int = SEMANTIC_TOP_K) -> list[dict]:

    vector_store = get_vector_store()

    # Execute a vector search that extracts BOTH the Document objects AND their metrics scores
    results = vector_store.similarity_search_with_score(query, top_k)

    # Parse result arrays and transform native database types into our operational schema dicts
    parsed_candidates = []

    for rank_idx, (doc, distance_score) in enumerate(results):
        db_chunk_id = doc.metadata.get("id")
        if not db_chunk_id:
            db_chunk_id = f"dense_syn_{rank_idx}"

        normalized_similarity = round(1.0 - float(distance_score), 4)

        parsed_candidates.append(
            {
                "chunk_id": db_chunk_id,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": normalized_similarity,
                "engine_source": "semantic",
            }
        )
    print(
        f"[Dense Stage] Completed database scan. Retrieved {len(parsed_candidates)} matching candidate slices."
    )
    return parsed_candidates


def bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """
    Search the persisted BM25 index for top_k keyword matches.
    Returns empty list if no index exists yet.
    """
    bm25, chunks = load_bm25()

    if bm25 is None:
        print("[Retrieval] No BM25 index found — skipping keyword search")
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # scores is a numpy array — one score per chunk in corpus
    # pair each score with its chunk and sort descending
    scored_positions = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    return [
        {
            "chunk_id": str(chunks[idx]["metadata"].get("id", f"sparse_syn_{idx}")),
            "text": chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"],
            "score": round(float(score), 4),
            "engine_source": "bm25",
        }
        for idx, score in scored_positions
        if score > 0  # skip zero-score chunks — no keyword overlap at all
    ]


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    top_k: int = RERANK_TOP_K,
    k=60,
):
    rrf_scores = {}
    chunk_store = {}
    # 1. Process dense semantic results list (1-based ranking position allocation)
    for rank, chunk in enumerate(semantic_results, 1):
        uid = chunk["chunk_id"]

        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (k + rank))
        chunk_store[uid] = chunk

        if uid not in chunk_store:
            chunk_store[uid] = chunk
        # 2. Process sparse keyword BM25 results list and accumulate weights
    for rank, chunk in enumerate(bm25_results, 1):
        uid = chunk["chunk_id"]

        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (k + rank))

        if uid not in chunk_store:
            chunk_store[uid] = chunk

    # Sort unique tracking IDs descending by their aggregated multi-channel weights
    sorted_uids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused_candidates = []
    for uid in sorted_uids[:top_k]:
        base_payload = chunk_store[uid]
        fused_candidates.append(
            {**base_payload, "rrf_score": round(rrf_scores[uid], 6)}
        )

    print(
        f"[RRF Fusion] Deduplication complete. Forwarding Top-{len(fused_candidates)} consensus candidates."
    )
    return fused_candidates


def rerank(query: str, candidates: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
    if not candidates:
        return []
    print(
        f"[Rerank Stage] Applying deep cross-attention layers across {len(candidates)} fused candidates..."
    )
    # 1. Construct a flat list of text pairs: (query, document_chunk_text)
    pairs = [(query, c["text"]) for c in candidates]

    # Returns an array of raw floating-point logit relevance scores
    scores = cross_encoder.predict(pairs)

    # 3. Enrich our dictionary items with rank performance data for production metrics observability
    for i, candidate in enumerate(candidates):
        candidate["original_rank"] = i + 1  # Tracks where it sat after the RRF step
        candidate["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    final_output = reranked[:top_k]

    print(
        f"[Rerank Stage] Finished neural re-scoring. Sliced candidate pool down to top-{len(final_output)} chunks."
    )
    return final_output


async def resolve_parents(child_candidates: list[dict]) -> list[dict]:
    """
    Given child chunks (from semantic/BM25/RRF), fetch their parent's
    full text and swap it in before reranking. De-duplicates parent_ids
    so if 3 children map to the same parent, that parent's text appears once.
    """
    parent_ids = list(
        {
            c["metadata"].get("parent_id")
            for c in child_candidates
            if c["metadata"].get("parent_id")
        }
    )
    if not parent_ids:
        return child_candidates  # no parent linkage — fall back to child text as-is

    async with get_db_conn() as conn:
        rows = await conn.execute(
            "SELECT parent_id, parent_text FROM parent_documents WHERE parent_id = ANY(%s)",
            (parent_ids,),
        )
        records = await rows.fetchall()
    parent_lookup = {str(pid): text for pid, text in records}

    resolved = []
    seen_parents = set()
    for c in child_candidates:
        parent_id = c["metadata"].get("parent_id")
        if parent_id and parent_id in parent_lookup:
            if parent_id in seen_parents:
                continue  # this parent's full text already added — skip duplicate child
            seen_parents.add(parent_id)
            resolved.append(
                {
                    **c,
                    "child_text": c["text"],  # preserve the original matched snippet
                    "text": parent_lookup[parent_id],  # swap child text for parent text
                }
            )
        else:
            resolved.append(
                {**c, "child_text": c["text"]}
            )  # no parent found — keep child text

    return resolved


async def hybrid_retrieve(query: str) -> list[dict]:
    """
    Full pipeline:
    semantic(50) + bm25(50) → RRF → top 10 → cross-encoder → top 3
    """
    print(f"\n[Hybrid Pipeline] Beginning orchestration flow for question: '{query}'")

    semantic_results = semantic_search(query)
    bm25_results = bm25_search(query)

    print(
        f"[Hybrid Pipeline] Raw candidates extracted. Semantic: {len(semantic_results)} | BM25: {len(bm25_results)}"
    )

    fused = reciprocal_rank_fusion(semantic_results, bm25_results)

    fused = await resolve_parents(fused)
    print(
        f"[Hybrid Pipeline] After RRF deduplication: {len(fused)} candidate windows forwarded."
    )

    # resolve to parent text before reranking
    final = rerank(query, fused)
    print(
        f"[Hybrid Pipeline] Pipeline executed successfully. Returning top-{len(final)} grounded facts for generation.\n"
    )

    return final
