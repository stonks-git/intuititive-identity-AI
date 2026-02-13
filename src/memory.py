"""Memory Store — Postgres + pgvector + Google embeddings.

Handles all Layer 2 operations: embed, store, retrieve, search.
Used by the cognitive loop, memory gate, consolidation, and idle loop.
"""

import asyncio
import logging
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
from google import genai

from .llm import retry_llm_call
from .config import RetryConfig

logger = logging.getLogger("agent.memory")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768

# Semantic type prefixes baked into embeddings (ENGRAM, MIRIX papers)
MEMORY_TYPE_PREFIXES = {
    "episodic":    "Personal experience memory: ",
    "semantic":    "Factual knowledge: ",
    "procedural":  "How-to instruction: ",
    "preference":  "User preference: ",
    "reflection":  "Self-reflection insight: ",
    "correction":  "Past error correction: ",
    "narrative":   "Identity narrative: ",
    "tension":     "Internal contradiction: ",
}


class MemoryStore:
    """Async interface to agent memory backed by Postgres + pgvector."""

    def __init__(self, retry_config: RetryConfig | None = None):
        self.pool: asyncpg.Pool | None = None
        self.genai_client: genai.Client | None = None
        self.retry_config = retry_config or RetryConfig()
        self.safety: "SafetyMonitor | None" = None  # Set by cognitive loop

    async def connect(self):
        """Initialize DB pool and embedding client."""
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://agent:agent_secret@localhost:5432/agent_memory",
        )
        self.pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            logger.warning("GOOGLE_API_KEY not set — embeddings unavailable")

        logger.info("Memory store connected.")

    async def close(self):
        """Close DB pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Memory store closed.")

    # --- EMBEDDINGS ---

    async def embed(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: str | None = None,
    ) -> list[float]:
        """Embed text using Google gemini-embedding-001 with retry.

        Args:
            text: Content to embed.
            task_type: Gemini task type — RETRIEVAL_DOCUMENT (storage),
                       RETRIEVAL_QUERY (search), SEMANTIC_SIMILARITY (novelty),
                       CLUSTERING (consolidation).
            title: Optional title hint for the embedding model.
        """
        if not self.genai_client:
            raise RuntimeError("Embedding client not initialized (missing API key?)")

        async def _call():
            embed_config = genai.types.EmbedContentConfig(
                output_dimensionality=EMBED_DIMENSIONS,
                task_type=task_type,
            )
            if title:
                embed_config.title = title
            result = await self.genai_client.aio.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=embed_config,
            )
            return result.embeddings[0].values

        return await retry_llm_call(
            _call,
            config=self.retry_config,
            label="embed",
        )

    async def embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: str | None = None,
    ) -> list[list[float]]:
        """Batch embed up to 100 texts in a single API call."""
        if not self.genai_client:
            raise RuntimeError("Embedding client not initialized (missing API key?)")
        if not texts:
            return []

        results = []
        # Gemini supports up to 100 texts per call
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]

            async def _call(b=batch):
                embed_config = genai.types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIMENSIONS,
                    task_type=task_type,
                )
                if title:
                    embed_config.title = title
                result = await self.genai_client.aio.models.embed_content(
                    model=EMBED_MODEL,
                    contents=b,
                    config=embed_config,
                )
                return [e.values for e in result.embeddings]

            batch_results = await retry_llm_call(
                _call,
                config=self.retry_config,
                label="embed_batch",
            )
            results.extend(batch_results)

        return results

    def prefixed_content(self, content: str, memory_type: str) -> str:
        """Prepend semantic type prefix for embedding."""
        prefix = MEMORY_TYPE_PREFIXES.get(memory_type, "")
        return f"{prefix}{content}"

    # --- STORE ---

    async def store_memory(
        self,
        content: str,
        memory_type: str = "semantic",
        source: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        evidence_count: int = 0,
        metadata: dict | None = None,
        source_tag: str | None = None,
    ) -> str:
        """Embed and store a memory chunk. Returns the memory ID."""
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        prefixed = self.prefixed_content(content, memory_type)
        embedding = await self.embed(
            prefixed, task_type="RETRIEVAL_DOCUMENT", title=memory_type,
        )
        now = datetime.now(timezone.utc)

        await self.pool.execute(
            """
            INSERT INTO memories (id, content, type, embedding, created_at, updated_at,
                                  source, tags, confidence, importance, evidence_count, metadata,
                                  source_tag)
            VALUES ($1, $2, $3, $4::halfvec, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
            memory_id,
            content,
            memory_type,
            str(embedding),
            now,
            now,
            source,
            tags or [],
            confidence,
            importance,
            evidence_count,
            json.dumps(metadata or {}),
            source_tag,
        )

        logger.info(f"Stored memory {memory_id}: {content[:80]}...")
        return memory_id

    async def store_insight(
        self,
        content: str,
        source_memory_ids: list[str],
        importance: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a consolidated insight that supersedes source memories.

        - Creates a new high-importance insight memory
        - Links it to source memories via memory_supersedes
        - Lowers importance of source memories (they remain queryable)
        """
        # Store the insight
        insight_id = await self.store_memory(
            content=content,
            memory_type="semantic",
            source="consolidation",
            tags=tags,
            confidence=0.8,
            importance=importance,
            evidence_count=len(source_memory_ids),
            metadata=metadata,
        )

        # Link to source memories
        for source_id in source_memory_ids:
            await self.pool.execute(
                """
                INSERT INTO memory_supersedes (insight_id, source_id)
                VALUES ($1, $2) ON CONFLICT DO NOTHING
                """,
                insight_id,
                source_id,
            )

        # Lower importance of source memories (don't delete them)
        await self.pool.execute(
            """
            UPDATE memories SET importance = LEAST(importance, 0.3)
            WHERE id = ANY($1)
            """,
            source_memory_ids,
        )

        logger.info(
            f"Insight {insight_id} supersedes {len(source_memory_ids)} memories: "
            f"{content[:80]}..."
        )
        return insight_id

    # --- INTROSPECTION ---

    async def why_do_i_believe(self, memory_id: str) -> list[dict]:
        """Trace the evidence chain for a memory/insight.

        Returns the source memories that formed this belief,
        following the supersedes chain recursively.
        """
        rows = await self.pool.fetch(
            """
            WITH RECURSIVE evidence_chain AS (
                -- Start with direct sources
                SELECT s.source_id, 1 AS depth
                FROM memory_supersedes s
                WHERE s.insight_id = $1

                UNION ALL

                -- Follow chain deeper
                SELECT s.source_id, ec.depth + 1
                FROM memory_supersedes s
                JOIN evidence_chain ec ON s.insight_id = ec.source_id
                WHERE ec.depth < 5  -- max depth to prevent loops
            )
            SELECT DISTINCT m.id, m.content, m.type, m.confidence,
                   m.importance, m.created_at, m.source, m.tags,
                   ec.depth
            FROM evidence_chain ec
            JOIN memories m ON m.id = ec.source_id
            ORDER BY ec.depth, m.created_at
            """,
            memory_id,
        )
        return [dict(r) for r in rows]

    async def get_insights_for(self, source_memory_id: str) -> list[dict]:
        """Find insights that were built from a given source memory."""
        rows = await self.pool.fetch(
            """
            SELECT m.id, m.content, m.importance, m.evidence_count, m.created_at
            FROM memory_supersedes s
            JOIN memories m ON m.id = s.insight_id
            WHERE s.source_id = $1
            ORDER BY m.importance DESC
            """,
            source_memory_id,
        )
        return [dict(r) for r in rows]

    # --- RETRIEVAL-INDUCED MUTATION (§3.8) ---

    async def apply_retrieval_mutation(
        self,
        retrieved_ids: list[str],
        near_miss_ids: list[str] | None = None,
        vector_scores: dict[str, float] | None = None,
    ) -> None:
        """Retrieval-induced mutation — retrieval itself reshapes memory.

        Retrieved memories get reinforced (depth_weight alpha +0.1).
        Near-misses (rank 6-20) get mild suppression (depth_weight beta +0.05).
        Dormant memories recovered under strong cues (cosine>0.9) get double boost.
        Immutable memories are never suppressed.

        Safety ceilings (§3.9) applied: diminishing returns on gain,
        hard ceiling on weight center, rate limiting in consolidation cycles.

        Args:
            retrieved_ids: Top-ranked memory IDs to reinforce.
            near_miss_ids: Lower-ranked IDs to mildly suppress.
            vector_scores: Optional {id: cosine_similarity} for dormant recovery.
        """
        now = datetime.now(timezone.utc)

        if retrieved_ids:
            # Fetch current weights for safety checks
            rows = await self.pool.fetch(
                """
                SELECT id, depth_weight_alpha, depth_weight_beta, immutable
                FROM memories WHERE id = ANY($1)
                """,
                retrieved_ids,
            )
            mem_info = {r["id"]: r for r in rows}

            for mid in retrieved_ids:
                info = mem_info.get(mid)
                if not info:
                    continue

                alpha = float(info["depth_weight_alpha"])
                beta = float(info["depth_weight_beta"])
                is_dormant = vector_scores and vector_scores.get(mid, 0) > 0.9
                base_gain = 0.2 if is_dormant else 0.1

                # Apply safety ceilings
                gain = base_gain
                if self.safety:
                    allowed, adj_alpha, adj_beta, reasons = (
                        self.safety.check_weight_change(
                            memory_id=mid,
                            current_alpha=alpha,
                            current_beta=beta,
                            delta_alpha=base_gain,
                            is_immutable=bool(info["immutable"]),
                        )
                    )
                    if not allowed:
                        logger.debug(f"Safety blocked reinforcement for {mid}: {reasons}")
                        gain = 0.0
                    else:
                        gain = adj_alpha  # May be reduced by diminishing returns

                if gain > 0:
                    await self.pool.execute(
                        """
                        UPDATE memories
                        SET access_count = access_count + 1,
                            last_accessed = $1,
                            access_timestamps = array_append(
                                COALESCE(access_timestamps, ARRAY[]::timestamptz[]), $1
                            ),
                            depth_weight_alpha = depth_weight_alpha + $3,
                            updated_at = $1
                        WHERE id = $2
                        """,
                        now,
                        mid,
                        gain,
                    )
                else:
                    # Still update access metadata even if weight change blocked
                    await self.pool.execute(
                        """
                        UPDATE memories
                        SET access_count = access_count + 1,
                            last_accessed = $1,
                            access_timestamps = array_append(
                                COALESCE(access_timestamps, ARRAY[]::timestamptz[]), $1
                            ),
                            updated_at = $1
                        WHERE id = $2
                        """,
                        now,
                        mid,
                    )

                if is_dormant and gain > 0:
                    logger.info(f"Dormant recovery: {mid} (cosine>0.9, gain={gain:.3f})")

            logger.debug(f"Reinforced {len(retrieved_ids)} retrieved memories")

        if near_miss_ids:
            # Mild suppression: beta +0.05 (never suppress immutable)
            # Safety: check each near-miss through ceilings
            for mid in near_miss_ids:
                suppression = 0.05
                if self.safety:
                    allowed, _, adj_beta, reasons = self.safety.check_weight_change(
                        memory_id=mid,
                        current_alpha=1.0,  # Approx — full check not needed for suppression
                        current_beta=4.0,
                        delta_beta=suppression,
                    )
                    # Suppression is always mild; hard ceiling doesn't block beta increases.
                    # We still log it through safety for audit trail.

            await self.pool.execute(
                """
                UPDATE memories
                SET depth_weight_beta = depth_weight_beta + 0.05,
                    updated_at = $1
                WHERE id = ANY($2)
                  AND NOT immutable
                """,
                now,
                near_miss_ids,
            )
            logger.debug(f"Suppressed {len(near_miss_ids)} near-miss memories")

    # --- RETRIEVE ---

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """Embed query and find most similar memories via pgvector cosine search."""
        query_embedding = await self.embed(query, task_type="RETRIEVAL_QUERY")

        rows = await self.pool.fetch(
            """
            SELECT id, content, type, confidence, importance,
                   access_count, last_accessed, tags, source, created_at,
                   1 - (embedding <=> $1::halfvec) AS similarity
            FROM memories
            WHERE 1 - (embedding <=> $1::halfvec) > $2
            ORDER BY embedding <=> $1::halfvec
            LIMIT $3
            """,
            str(query_embedding),
            min_similarity,
            top_k,
        )

        # Update access counts
        if rows:
            now = datetime.now(timezone.utc)
            ids = [r["id"] for r in rows]
            await self.pool.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = $1
                WHERE id = ANY($2)
                """,
                now,
                ids,
            )

        return [dict(r) for r in rows]

    async def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        mutate: bool = True,
        reinforce_top_k: int = 5,
    ) -> list[dict]:
        """Hybrid search: dense (pgvector) + sparse (tsvector) fused with RRF.

        Returns top_k results with RRF-fused scores incorporating recency.
        Per-connection iterative scan enabled for better recall.

        When mutate=True (default), applies retrieval-induced mutation (§3.8):
        top reinforce_top_k results get reinforced, remaining get mild suppression.
        """
        query_embedding = await self.embed(query, task_type="RETRIEVAL_QUERY")

        async with self.pool.acquire() as conn:
            # Enable iterative scan for better recall on this connection
            await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")

            rows = await conn.fetch(
                """
                WITH dense AS (
                    SELECT id, content, type, confidence, importance,
                           depth_weight_alpha, depth_weight_beta,
                           access_count, last_accessed, tags, source,
                           created_at, compressed, immutable, source_tag,
                           1 - (embedding <=> $1::halfvec) AS vector_score,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::halfvec) AS dense_rank
                    FROM memories
                    ORDER BY embedding <=> $1::halfvec
                    LIMIT 50
                ),
                sparse AS (
                    SELECT id, content, type, confidence, importance,
                           depth_weight_alpha, depth_weight_beta,
                           access_count, last_accessed, tags, source,
                           created_at, compressed, immutable, source_tag,
                           ts_rank_cd(content_tsv, websearch_to_tsquery('english', $2)) AS text_score,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', $2)) DESC
                           ) AS sparse_rank
                    FROM memories
                    WHERE content_tsv @@ websearch_to_tsquery('english', $2)
                    ORDER BY text_score DESC
                    LIMIT 50
                ),
                combined AS (
                    SELECT
                        COALESCE(d.id, s.id) AS id,
                        COALESCE(d.content, s.content) AS content,
                        COALESCE(d.type, s.type) AS type,
                        COALESCE(d.confidence, s.confidence) AS confidence,
                        COALESCE(d.importance, s.importance) AS importance,
                        COALESCE(d.depth_weight_alpha, s.depth_weight_alpha) AS depth_weight_alpha,
                        COALESCE(d.depth_weight_beta, s.depth_weight_beta) AS depth_weight_beta,
                        COALESCE(d.access_count, s.access_count) AS access_count,
                        COALESCE(d.last_accessed, s.last_accessed) AS last_accessed,
                        COALESCE(d.tags, s.tags) AS tags,
                        COALESCE(d.source, s.source) AS source,
                        COALESCE(d.created_at, s.created_at) AS created_at,
                        COALESCE(d.compressed, s.compressed) AS compressed,
                        COALESCE(d.immutable, s.immutable) AS immutable,
                        COALESCE(d.source_tag, s.source_tag) AS source_tag,
                        COALESCE(d.vector_score, 0) AS vector_score,
                        COALESCE(s.text_score, 0) AS text_score,
                        -- RRF: 1/(k + rank) with k=60
                        COALESCE(1.0 / (60 + d.dense_rank), 0) AS rrf_dense,
                        COALESCE(1.0 / (60 + s.sparse_rank), 0) AS rrf_sparse,
                        -- Recency: 7-day half-life exponential decay
                        EXP(-0.693 * EXTRACT(EPOCH FROM (NOW() - COALESCE(d.created_at, s.created_at))) / 604800.0) AS recency_score
                    FROM dense d
                    FULL OUTER JOIN sparse s ON d.id = s.id
                )
                SELECT *,
                       (rrf_dense + rrf_sparse) AS rrf_score,
                       -- Weighted composite: RRF + recency + importance
                       0.5 * (rrf_dense + rrf_sparse)
                       + 0.3 * recency_score
                       + 0.2 * (depth_weight_alpha / (depth_weight_alpha + depth_weight_beta))
                       AS weighted_score
                FROM combined
                ORDER BY weighted_score DESC
                LIMIT $3
                """,
                str(query_embedding),
                query,
                top_k,
            )

        results = [dict(r) for r in rows]

        # Retrieval-induced mutation (§3.8)
        if mutate and results:
            retrieved_ids = [r["id"] for r in results[:reinforce_top_k]]
            near_miss_ids = [r["id"] for r in results[reinforce_top_k:]]
            vector_scores = {r["id"]: r.get("vector_score", 0) for r in results}
            await self.apply_retrieval_mutation(
                retrieved_ids, near_miss_ids, vector_scores,
            )
            # Record co-access for retrieved memories (Hebbian learning)
            from .relevance import update_co_access
            await update_co_access(self.pool, retrieved_ids)

        return results

    async def search_reranked(
        self,
        query: str,
        top_k: int = 5,
        hybrid_top_k: int = 20,
    ) -> list[dict]:
        """Hybrid search + FlashRank cross-encoder reranking.

        Returns top_k results after reranking hybrid_top_k candidates.
        Final score: 0.6 * rerank_score + 0.4 * weighted_score.

        Mutation (§3.8) is applied post-reranking so the actual top results
        get reinforced, not the pre-reranking top.
        """
        candidates = await self.search_hybrid(
            query, top_k=hybrid_top_k, mutate=False,
        )
        if not candidates:
            return []

        # Lazy-load FlashRank (34MB model, first call downloads it)
        if not hasattr(self, "_ranker"):
            from flashrank import Ranker
            self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        from flashrank import RerankRequest

        # Build rerank request
        passages = [
            {"id": c["id"], "text": c["content"], "meta": c}
            for c in candidates
        ]
        request = RerankRequest(query=query, passages=passages)

        # CPU-bound reranking in thread pool
        reranked = await asyncio.to_thread(self._ranker.rerank, request)

        # Merge rerank score with weighted_score from hybrid search
        results = []
        for item in reranked[:top_k]:
            meta = item["meta"]
            rerank_score = item["score"]
            weighted_score = meta.get("weighted_score", 0.5)
            meta["rerank_score"] = rerank_score
            meta["final_score"] = 0.6 * rerank_score + 0.4 * weighted_score
            results.append(meta)

        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Retrieval-induced mutation (§3.8) — applied post-reranking
        # Top results = reinforced, remaining candidates = near-misses
        if results:
            retrieved_ids = [r["id"] for r in results]
            near_miss_ids = [
                c["id"] for c in candidates if c["id"] not in set(retrieved_ids)
            ]
            vector_scores = {c["id"]: c.get("vector_score", 0) for c in candidates}
            await self.apply_retrieval_mutation(
                retrieved_ids, near_miss_ids, vector_scores,
            )
            from .relevance import update_co_access
            await update_co_access(self.pool, retrieved_ids)

        return results

    async def get_memory(self, memory_id: str) -> dict | None:
        """Fetch a single memory by ID."""
        row = await self.pool.fetchrow(
            "SELECT * FROM memories WHERE id = $1", memory_id
        )
        return dict(row) if row else None

    async def get_random_memory(self) -> dict | None:
        """Pull a random memory — used by the idle loop / DMN."""
        row = await self.pool.fetchrow(
            "SELECT id, content, type, confidence, importance, tags, created_at "
            "FROM memories ORDER BY RANDOM() LIMIT 1"
        )
        return dict(row) if row else None

    async def memory_count(self) -> int:
        """Total number of stored memories."""
        return await self.pool.fetchval("SELECT COUNT(*) FROM memories")

    # --- SCRATCH BUFFER ---

    async def buffer_scratch(
        self,
        content: str,
        source: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Write to scratch buffer with 24h TTL (entry gate temp storage)."""
        scratch_id = f"scratch_{uuid.uuid4().hex[:12]}"
        await self.pool.execute(
            """
            INSERT INTO scratch_buffer (id, content, source, tags, metadata, expires_at)
            VALUES ($1, $2, $3, $4, $5, NOW() + INTERVAL '24 hours')
            """,
            scratch_id,
            content,
            source,
            tags or [],
            json.dumps(metadata or {}),
        )
        return scratch_id

    async def flush_scratch(self, older_than_minutes: int = 0) -> list[dict]:
        """Retrieve and delete scratch entries. Default: all non-expired."""
        rows = await self.pool.fetch(
            """
            DELETE FROM scratch_buffer
            WHERE buffered_at < NOW() - INTERVAL '1 minute' * $1
              AND (expires_at IS NULL OR expires_at > NOW())
            RETURNING *
            """,
            older_than_minutes,
        )
        return [dict(r) for r in rows]

    async def cleanup_expired_scratch(self) -> int:
        """Delete expired scratch entries. Returns count deleted."""
        result = await self.pool.execute(
            "DELETE FROM scratch_buffer WHERE expires_at < NOW()"
        )
        count = int(result.split()[-1]) if result else 0
        if count:
            logger.info(f"Cleaned up {count} expired scratch entries")
        return count

    async def recover_crash_scratch(self, last_flush_time: datetime) -> list[dict]:
        """Crash recovery: find entries older than last known flush."""
        rows = await self.pool.fetch(
            """
            SELECT * FROM scratch_buffer
            WHERE buffered_at < $1
            ORDER BY buffered_at
            """,
            last_flush_time,
        )
        return [dict(r) for r in rows]

    # --- NOVELTY CHECK ---

    async def check_novelty(self, content: str, threshold: float = 0.85) -> tuple[bool, float]:
        """Check if content is already in memory (for gate novelty scoring).

        Returns (is_novel, max_similarity).
        """
        embedding = await self.embed(content, task_type="SEMANTIC_SIMILARITY")
        row = await self.pool.fetchrow(
            """
            SELECT 1 - (embedding <=> $1::halfvec) AS similarity
            FROM memories
            ORDER BY embedding <=> $1::halfvec
            LIMIT 1
            """,
            str(embedding),
        )

        if row is None:
            return True, 0.0

        max_sim = float(row["similarity"])
        return max_sim < threshold, max_sim

    # --- DECAY ---

    async def get_stale_memories(
        self, stale_days: int = 90, min_access_count: int = 3
    ) -> list[dict]:
        """Find memories eligible for decay."""
        rows = await self.pool.fetch(
            """
            SELECT id, content, importance, access_count, last_accessed
            FROM memories
            WHERE (last_accessed IS NULL OR last_accessed < NOW() - INTERVAL '1 day' * $1)
              AND access_count < $2
              AND importance > 0.05
            ORDER BY importance ASC
            """,
            stale_days,
            min_access_count,
        )
        return [dict(r) for r in rows]

    async def decay_memories(self, memory_ids: list[str], factor: float = 0.5):
        """Halve importance of stale memories (never delete)."""
        await self.pool.execute(
            """
            UPDATE memories
            SET importance = importance * $1, updated_at = NOW()
            WHERE id = ANY($2)
            """,
            factor,
            memory_ids,
        )
        logger.info(f"Decayed {len(memory_ids)} memories by factor {factor}")

    async def avg_depth_weight_center(self, where: str | None = None) -> float:
        """Average depth_weight center (alpha / (alpha + beta)) across memories.

        Args:
            where: Optional SQL WHERE clause filter (e.g. for high-weight only).
        """
        base = "SELECT AVG(depth_weight_alpha / (depth_weight_alpha + depth_weight_beta)) FROM memories"
        if where:
            base += f" WHERE {where}"
        result = await self.pool.fetchval(base)
        return float(result) if result is not None else 0.0

    async def search_corrections(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve top-k correction memories by similarity to attention focus."""
        rows = await self.pool.fetch(
            """
            SELECT id, content, compressed, confidence, importance,
                   depth_weight_alpha, depth_weight_beta,
                   1 - (embedding <=> $1::halfvec) AS similarity
            FROM memories
            WHERE memory_type = 'correction'
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::halfvec
            LIMIT $2
            """,
            str(query_embedding),
            top_k,
        )
        return [dict(r) for r in rows]

    async def store_correction(
        self,
        trigger: str,
        original_reasoning: str,
        correction: str,
        context: str | None = None,
        confidence: float = 0.8,
    ) -> str:
        """Store a System 2 correction in the reflection bank."""
        content = (
            f"Trigger: {trigger}\n"
            f"Original reasoning: {original_reasoning}\n"
            f"Correction: {correction}"
        )
        if context:
            content += f"\nContext: {context}"
        return await self.store_memory(
            content=content,
            memory_type="correction",
            source="system2_escalation",
            confidence=confidence,
            importance=confidence,
            metadata={
                "trigger": trigger[:200],
                "correction_type": "system2_override",
            },
        )
