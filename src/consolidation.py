"""Consolidation Engine — two-tier memory processing.

Tier 1 (Constant, §4.1): Always-running, rate-limited, cheap background processes.
  - Weight decay tick: nudge unused memories via contradict(0.01)
  - Contradiction scan: random memory pairs, check for conflicts
  - Pattern detection: cluster recent memories for emerging themes

Tier 2 (Deep, §4.2-4.6): Periodic "sleep cycle" with LLM-driven operations.
  - Merge + insight + narrative + re-compression (§4.2)
  - Promotion + safety checks (§4.3)
  - Decay + reconsolidation (§4.4)
  - Gate tuning + Dirichlet evolution (§4.5)
  - Contextual retrieval (§4.6)

Both tiers write to the same memory store. The agent doesn't "notice"
consolidation — priorities subtly shift. Next identity render picks up changes.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
from google import genai

from .llm import retry_llm_call
from .config import RetryConfig

logger = logging.getLogger("agent.consolidation")

# ── CONSTANTS ───────────────────────────────────────────────────────────────

# Tier 1: Constant consolidation intervals (seconds)
DECAY_TICK_INTERVAL = 300       # 5 min between decay ticks
CONTRADICTION_SCAN_INTERVAL = 600  # 10 min between contradiction scans
PATTERN_DETECT_INTERVAL = 900   # 15 min between pattern detection runs

# Tier 1: Decay parameters
DECAY_NUDGE_AMOUNT = 0.01      # Gentle beta increase per tick
DECAY_STALE_HOURS = 24         # Consider "unused" if not accessed in 24h

# Tier 2: Deep consolidation
DEEP_INTERVAL_SECONDS = 3600   # 1 hour default (overridden by config)
MERGE_SIMILARITY_THRESHOLD = 0.85
INSIGHT_QUESTION_COUNT = 3
INSIGHT_PER_QUESTION = 5

# Promotion thresholds
PROMOTE_GOAL_MIN_COUNT = 5
PROMOTE_GOAL_MIN_DAYS = 14
PROMOTE_GOAL_REINFORCE = 2.0
PROMOTE_IDENTITY_MIN_COUNT = 10
PROMOTE_IDENTITY_MIN_DAYS = 30
PROMOTE_IDENTITY_REINFORCE = 5.0

# Decay thresholds
DECAY_STALE_DAYS = 90
DECAY_MIN_ACCESS = 3
DECAY_CONTRADICT_AMOUNT = 1.0


# ── TIER 1: CONSTANT BACKGROUND CONSOLIDATION (§4.1) ───────────────────────


class ConstantConsolidation:
    """Always-running lightweight consolidation — the agent's metabolism.

    Rate-limited operations that run continuously in the background:
    - Weight decay nudges on stale memories
    - Contradiction scanning between recent memory pairs
    - Pattern detection via clustering
    """

    def __init__(self, memory, config=None, retry_config=None):
        self.memory = memory
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self._last_decay = 0.0
        self._last_contradiction = 0.0
        self._last_pattern = 0.0
        self._genai_client = None

    async def run(self, shutdown_event: asyncio.Event):
        """Main constant consolidation loop."""
        logger.info("Constant consolidation (Tier 1) started")

        # Init Gemini client for contradiction scan
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self._genai_client = genai.Client(api_key=api_key)

        tick_interval = 30  # Check every 30 seconds which operations are due

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=tick_interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            now = time.monotonic()

            # Run whichever operations are due (rate-limited)
            try:
                if now - self._last_decay >= DECAY_TICK_INTERVAL:
                    await self._decay_tick()
                    self._last_decay = now

                if now - self._last_contradiction >= CONTRADICTION_SCAN_INTERVAL:
                    await self._contradiction_scan()
                    self._last_contradiction = now

                if now - self._last_pattern >= PATTERN_DETECT_INTERVAL:
                    await self._pattern_detection()
                    self._last_pattern = now

            except Exception as e:
                logger.error(f"Constant consolidation error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Back off on error

        logger.info("Constant consolidation (Tier 1) stopped")

    async def _decay_tick(self):
        """Gently nudge unused memories toward decay.

        Memories not accessed in 24h get a tiny beta increase (0.01).
        This is NOT aggressive decay — just subtle drift for unused memories.
        """
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=DECAY_STALE_HOURS)

        result = await self.memory.pool.execute(
            """
            UPDATE memories
            SET depth_weight_beta = depth_weight_beta + $1,
                updated_at = NOW()
            WHERE (last_accessed IS NULL OR last_accessed < $2)
              AND NOT immutable
              AND depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.1
            """,
            DECAY_NUDGE_AMOUNT,
            stale_cutoff,
        )

        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.debug(f"Decay tick: nudged {count} stale memories (beta +{DECAY_NUDGE_AMOUNT})")

    async def _contradiction_scan(self):
        """Pick random memory pairs from recent activity, check for contradictions.

        Uses LLM to detect semantic opposition between recently accessed memories.
        """
        if not self._genai_client:
            return

        # Get 10 recently accessed memories
        rows = await self.memory.pool.fetch(
            """
            SELECT id, content, type, depth_weight_alpha, depth_weight_beta
            FROM memories
            WHERE last_accessed IS NOT NULL
              AND last_accessed > NOW() - INTERVAL '24 hours'
            ORDER BY last_accessed DESC
            LIMIT 10
            """,
        )
        if len(rows) < 2:
            return

        # Pick 2 random pairs to check
        import random
        memories = [dict(r) for r in rows]
        pairs_checked = 0
        for _ in range(min(2, len(memories) // 2)):
            pair = random.sample(memories, 2)
            contradiction = await self._check_contradiction_pair(pair[0], pair[1])
            pairs_checked += 1
            if contradiction:
                await self._store_tension(pair[0], pair[1], contradiction)

        logger.debug(f"Contradiction scan: checked {pairs_checked} pairs")

    async def _check_contradiction_pair(
        self, mem_a: dict, mem_b: dict,
    ) -> str | None:
        """Check if two memories contradict each other via LLM."""
        prompt = (
            "Do these two memories contradict each other? "
            "If yes, briefly describe the contradiction in one sentence. "
            "If no, reply exactly 'NO'.\n\n"
            f"Memory A: {mem_a['content'][:500]}\n\n"
            f"Memory B: {mem_b['content'][:500]}"
        )

        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=100,
                        temperature=0.1,
                    ),
                )
                return response.text.strip()

            result = await retry_llm_call(
                _call, config=self.retry_config, label="contradiction_scan",
            )

            if result and result.upper() != "NO" and len(result) > 5:
                return result
        except Exception as e:
            logger.warning(f"Contradiction check failed: {e}")

        return None

    async def _store_tension(self, mem_a: dict, mem_b: dict, description: str):
        """Store a detected contradiction as a tension memory."""
        content = (
            f"Tension between memories: {description}\n"
            f"Memory A ({mem_a['id']}): {mem_a['content'][:200]}\n"
            f"Memory B ({mem_b['id']}): {mem_b['content'][:200]}"
        )
        await self.memory.store_memory(
            content=content,
            memory_type="tension",
            source="constant_consolidation",
            metadata={
                "source_a": mem_a["id"],
                "source_b": mem_b["id"],
                "surface_count": 0,
                "resolved": False,
            },
        )
        logger.info(f"Tension detected: {description[:80]}")

    async def _pattern_detection(self):
        """Cluster recent memories, look for emerging themes.

        Simple approach: embed recent memories, find clusters via cosine
        similarity above threshold. Log detected patterns.
        """
        # Get embeddings of recent memories
        rows = await self.memory.pool.fetch(
            """
            SELECT id, content, type, embedding,
                   depth_weight_alpha, depth_weight_beta
            FROM memories
            WHERE created_at > NOW() - INTERVAL '7 days'
              AND embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 50
            """,
        )
        if len(rows) < 3:
            return

        # Parse embeddings and find clusters
        memories = []
        for r in rows:
            emb_str = r["embedding"]
            if emb_str:
                try:
                    # pgvector returns string like '[0.1,0.2,...]'
                    emb = np.array([float(x) for x in emb_str.strip("[]").split(",")])
                    memories.append({"id": r["id"], "content": r["content"],
                                    "type": r["type"], "embedding": emb})
                except (ValueError, AttributeError):
                    continue

        if len(memories) < 3:
            return

        # Simple greedy clustering: assign each memory to nearest cluster
        clusters = self._greedy_cluster(memories, threshold=MERGE_SIMILARITY_THRESHOLD)

        # Log significant clusters (3+ members)
        significant = [c for c in clusters if len(c) >= 3]
        if significant:
            logger.info(
                f"Pattern detection: {len(significant)} clusters found "
                f"({', '.join(str(len(c)) + ' members' for c in significant[:3])})"
            )

    @staticmethod
    def _greedy_cluster(
        memories: list[dict], threshold: float = 0.85,
    ) -> list[list[dict]]:
        """Simple greedy clustering by cosine similarity."""
        clusters: list[list[dict]] = []
        assigned = set()

        for i, mem_i in enumerate(memories):
            if i in assigned:
                continue
            cluster = [mem_i]
            assigned.add(i)
            emb_i = mem_i["embedding"]
            norm_i = np.linalg.norm(emb_i)
            if norm_i == 0:
                continue

            for j, mem_j in enumerate(memories):
                if j in assigned:
                    continue
                emb_j = mem_j["embedding"]
                norm_j = np.linalg.norm(emb_j)
                if norm_j == 0:
                    continue
                sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))
                if sim >= threshold:
                    cluster.append(mem_j)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters


# ── TIER 2: DEEP CONSOLIDATION (§4.2-4.6) ──────────────────────────────────


class DeepConsolidation:
    """Periodic deep consolidation — the agent's "sleep cycle".

    Triggered hourly or on cumulative importance threshold.
    Operations: merge, insight, narrative, promotion, decay, gate tuning.
    """

    def __init__(self, memory, layers, config=None, retry_config=None):
        self.memory = memory
        self.layers = layers
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self.cycle_count = 0
        self._cumulative_importance = 0.0
        self._importance_threshold = 50.0  # Trigger early if lots of important content
        self._genai_client = None

    async def run(self, shutdown_event: asyncio.Event):
        """Run deep consolidation on schedule."""
        interval = DEEP_INTERVAL_SECONDS
        if self.config:
            interval = self.config.raw.get("consolidation", {}).get(
                "base_interval_minutes", 60,
            ) * 60

        logger.info(f"Deep consolidation (Tier 2) started. Interval: {interval}s")

        # Init Gemini client
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self._genai_client = genai.Client(api_key=api_key)

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            await self._run_cycle()

        logger.info("Deep consolidation (Tier 2) stopped")

    async def _run_cycle(self):
        """Run one deep consolidation cycle."""
        self.cycle_count += 1
        cycle_id = f"deep_{self.cycle_count}_{int(time.time())}"
        logger.info(f"Deep consolidation cycle #{self.cycle_count} starting ({cycle_id})")
        start = time.monotonic()

        try:
            # Enable Phase B safety for this cycle
            if self.memory.safety:
                self.memory.safety.enable_phase_b()

            # §4.2: Merge + insight + narrative
            await self._merge_and_insight(cycle_id)

            # §4.3: Promotion
            await self._promote_patterns(cycle_id)

            # §4.4: Decay + reconsolidation
            await self._decay_and_reconsolidate(cycle_id)

            # §4.5: Gate tuning (placeholder — needs outcome data)
            await self._tune_parameters(cycle_id)

            # §4.6: Contextual retrieval
            await self._contextual_retrieval(cycle_id)

            # Clean up safety per-cycle state
            if self.memory.safety:
                self.memory.safety.end_consolidation_cycle(cycle_id)

        except Exception as e:
            logger.error(f"Deep consolidation cycle #{self.cycle_count} failed: {e}", exc_info=True)

        elapsed = time.monotonic() - start
        logger.info(f"Deep consolidation cycle #{self.cycle_count} complete ({elapsed:.1f}s)")

        # Reset cumulative importance
        self._cumulative_importance = 0.0

    # ── §4.2: MERGE + INSIGHT + NARRATIVE ───────────────────────────────────

    async def _merge_and_insight(self, cycle_id: str):
        """Stanford two-phase reflection: questions → insights. Plus narrative."""
        if not self._genai_client:
            logger.warning("No Gemini client — skipping merge/insight")
            return

        # Phase 1: Question generation from recent memories
        recent = await self.memory.pool.fetch(
            """
            SELECT id, content, type, importance,
                   depth_weight_alpha, depth_weight_beta
            FROM memories
            WHERE type != 'correction'
            ORDER BY created_at DESC
            LIMIT 100
            """,
        )
        if len(recent) < 5:
            logger.debug("Too few memories for insight generation")
            return

        memory_texts = [f"[{r['type']}] {r['content'][:200]}" for r in recent]
        memory_block = "\n".join(memory_texts[:50])  # Limit context size

        questions = await self._generate_questions(memory_block)
        if not questions:
            return

        # Phase 2: Insight extraction per question
        insights_created = 0
        for question in questions[:INSIGHT_QUESTION_COUNT]:
            insights = await self._extract_insights(question, memory_block)
            for insight_text in insights[:INSIGHT_PER_QUESTION]:
                # Find source memories for this insight
                source_ids = await self._find_source_memories(insight_text, recent)
                if source_ids:
                    await self.memory.store_insight(
                        content=insight_text,
                        source_memory_ids=source_ids,
                        tags=["consolidation", f"cycle_{self.cycle_count}"],
                        metadata={"question": question, "cycle_id": cycle_id},
                    )
                    insights_created += 1

        logger.info(f"Merge/insight: {insights_created} insights created from {len(questions)} questions")

        # Cluster similar memories and generate narratives
        await self._cluster_and_narrate(recent, cycle_id)

        # Behavioral contradiction detection
        await self._detect_behavioral_contradictions(cycle_id)

        # Re-compress memories with cluster context
        await self._recompress_memories(recent, cycle_id)

    async def _generate_questions(self, memory_block: str) -> list[str]:
        """Phase 1: Generate salient questions from recent memories."""
        prompt = (
            "Given these recent memories from an AI agent, what are the "
            f"{INSIGHT_QUESTION_COUNT} most salient high-level questions "
            "that emerge? Return ONLY the questions, one per line.\n\n"
            f"{memory_block}"
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=500,
                        temperature=0.3,
                    ),
                )
                return response.text.strip()

            result = await retry_llm_call(
                _call, config=self.retry_config, label="question_gen",
            )
            return [q.strip("- •0123456789.") .strip() for q in result.split("\n") if q.strip()]
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
            return []

    async def _extract_insights(
        self, question: str, memory_block: str,
    ) -> list[str]:
        """Phase 2: Extract insights for a question using memory context."""
        prompt = (
            f"Question: {question}\n\n"
            "Based on these memories, provide up to 5 high-level insights "
            "that answer this question. Be concise. One insight per line.\n\n"
            f"{memory_block}"
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=500,
                        temperature=0.3,
                    ),
                )
                return response.text.strip()

            result = await retry_llm_call(
                _call, config=self.retry_config, label="insight_extract",
            )
            return [i.strip("- •0123456789.").strip() for i in result.split("\n") if i.strip()]
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            return []

    async def _find_source_memories(
        self, insight_text: str, candidates: list,
    ) -> list[str]:
        """Find which memories support an insight via embedding similarity."""
        try:
            insight_emb = await self.memory.embed(
                insight_text, task_type="SEMANTIC_SIMILARITY",
            )
        except Exception:
            return []

        rows = await self.memory.pool.fetch(
            """
            SELECT id, 1 - (embedding <=> $1::halfvec) AS sim
            FROM memories
            WHERE id = ANY($2)
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::halfvec
            LIMIT 5
            """,
            str(insight_emb),
            [r["id"] for r in candidates],
        )
        return [r["id"] for r in rows if r["sim"] > 0.5]

    async def _cluster_and_narrate(self, recent: list, cycle_id: str):
        """Cluster similar memories and generate narratives for 3+ clusters."""
        if not self._genai_client:
            return

        # Get embeddings for recent memories
        memories_with_emb = []
        for r in recent:
            row = await self.memory.pool.fetchrow(
                "SELECT embedding FROM memories WHERE id = $1", r["id"],
            )
            if row and row["embedding"]:
                try:
                    emb = np.array([
                        float(x) for x in str(row["embedding"]).strip("[]").split(",")
                    ])
                    memories_with_emb.append({
                        "id": r["id"], "content": r["content"],
                        "type": r["type"], "embedding": emb,
                    })
                except (ValueError, AttributeError):
                    continue

        if len(memories_with_emb) < 3:
            return

        clusters = ConstantConsolidation._greedy_cluster(
            memories_with_emb, threshold=MERGE_SIMILARITY_THRESHOLD,
        )

        narratives_created = 0
        for cluster in clusters:
            if len(cluster) < 3:
                continue

            # Generate narrative for this cluster
            cluster_text = "\n".join(
                f"- {m['content'][:150]}" for m in cluster[:10]
            )
            narrative = await self._generate_narrative(cluster_text)
            if narrative:
                source_ids = [m["id"] for m in cluster]
                await self.memory.store_memory(
                    content=narrative,
                    memory_type="narrative",
                    source="deep_consolidation",
                    tags=["narrative", f"cycle_{self.cycle_count}"],
                    metadata={
                        "source_ids": source_ids[:10],
                        "cluster_size": len(cluster),
                        "cycle_id": cycle_id,
                    },
                )
                narratives_created += 1

        if narratives_created:
            logger.info(f"Narratives: {narratives_created} identity narratives generated")

    async def _generate_narrative(self, cluster_text: str) -> str | None:
        """Generate a causal narrative for a cluster of related memories."""
        prompt = (
            "These memories form a cluster of related experiences/beliefs:\n\n"
            f"{cluster_text}\n\n"
            "Write a brief causal narrative (1-2 sentences) in first person "
            "that explains WHY this pattern exists. Start with 'I came to...' "
            "or 'I value...' or similar. Be specific, not generic."
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=200,
                        temperature=0.4,
                    ),
                )
                return response.text.strip()

            return await retry_llm_call(
                _call, config=self.retry_config, label="narrative_gen",
            )
        except Exception as e:
            logger.warning(f"Narrative generation failed: {e}")
            return None

    async def _detect_behavioral_contradictions(self, cycle_id: str):
        """Find contradictions between high-weight values and recent behavior."""
        if not self._genai_client:
            return

        # Get high-weight value/goal memories
        values = await self.memory.pool.fetch(
            """
            SELECT id, content, type
            FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.7
              AND type IN ('semantic', 'preference', 'reflection', 'narrative')
            ORDER BY depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) DESC
            LIMIT 10
            """,
        )
        if not values:
            return

        # Get recent behavioral/episodic memories
        behaviors = await self.memory.pool.fetch(
            """
            SELECT id, content FROM memories
            WHERE type = 'episodic'
              AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 20
            """,
        )
        if not behaviors:
            return

        behavior_text = "\n".join(
            f"- {r['content'][:150]}" for r in behaviors[:10]
        )

        tensions_created = 0
        for value in values[:5]:
            contradiction = await self._check_value_behavior_contradiction(
                value["content"], behavior_text,
            )
            if contradiction:
                content = (
                    f"Tension: {contradiction}\n"
                    f"Value: {value['content'][:200]}\n"
                    f"Contradicting behaviors observed in recent memory."
                )
                await self.memory.store_memory(
                    content=content,
                    memory_type="tension",
                    source="deep_consolidation",
                    metadata={
                        "value_id": value["id"],
                        "surface_count": 0,
                        "resolved": False,
                        "cycle_id": cycle_id,
                    },
                )
                tensions_created += 1

        if tensions_created:
            logger.info(f"Behavioral contradictions: {tensions_created} tensions detected")

    async def _check_value_behavior_contradiction(
        self, value_content: str, behavior_text: str,
    ) -> str | None:
        """Check if behaviors contradict a value via LLM."""
        prompt = (
            f"The agent claims to value: {value_content[:300]}\n\n"
            f"Recent behavioral examples:\n{behavior_text}\n\n"
            "Do any behaviors contradict this value? "
            "If yes, describe the contradiction in one sentence. "
            "If no, reply exactly 'NO'."
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=100,
                        temperature=0.1,
                    ),
                )
                return response.text.strip()

            result = await retry_llm_call(
                _call, config=self.retry_config, label="contradiction_detect",
            )
            if result and result.upper() != "NO" and len(result) > 5:
                return result
        except Exception as e:
            logger.warning(f"Value-behavior contradiction check failed: {e}")
        return None

    async def _recompress_memories(self, recent: list, cycle_id: str):
        """Re-compress memories using cluster context for better generalizations."""
        if not self._genai_client:
            return

        # Find memories with single-context compressions that could benefit
        rows = await self.memory.pool.fetch(
            """
            SELECT id, content, compressed, type
            FROM memories
            WHERE compressed IS NOT NULL
              AND source != 'deep_consolidation'
              AND created_at > NOW() - INTERVAL '7 days'
            LIMIT 20
            """,
        )
        if not rows:
            return

        recompressed = 0
        for row in rows:
            # Get related memories for context
            related = await self.memory.pool.fetch(
                """
                SELECT content FROM memories
                WHERE id != $1
                  AND type = $2
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> (SELECT embedding FROM memories WHERE id = $1)
                LIMIT 5
                """,
                row["id"],
                row["type"],
            )
            if len(related) < 2:
                continue

            context = "\n".join(f"- {r['content'][:100]}" for r in related)
            new_compressed = await self._recompress_one(
                row["content"], row["compressed"], context,
            )
            if new_compressed and new_compressed != row["compressed"]:
                await self.memory.pool.execute(
                    "UPDATE memories SET compressed = $1, updated_at = NOW() WHERE id = $2",
                    new_compressed, row["id"],
                )
                recompressed += 1

        if recompressed:
            logger.info(f"Re-compressed {recompressed} memories with cluster context")

    async def _recompress_one(
        self, content: str, old_compressed: str, context: str,
    ) -> str | None:
        """Re-compress a single memory using broader cluster context."""
        prompt = (
            "Original memory:\n"
            f"{content[:300]}\n\n"
            "Current compression:\n"
            f"{old_compressed}\n\n"
            "Related memories:\n"
            f"{context}\n\n"
            "Generate a more general-purpose compression (1 sentence) that captures "
            "the essence considering related memories. Be concise."
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=100,
                        temperature=0.2,
                    ),
                )
                return response.text.strip()

            return await retry_llm_call(
                _call, config=self.retry_config, label="recompress",
            )
        except Exception as e:
            logger.warning(f"Re-compression failed: {e}")
            return None

    # ── §4.3: PROMOTION + SAFETY ────────────────────────────────────────────

    async def _promote_patterns(self, cycle_id: str):
        """Promote repeated patterns via depth_weight reinforcement.

        Pattern 5+ times over 14+ days → reinforce(2.0) → goal-range (~0.6-0.7)
        Pattern 10+ times over 30+ days → reinforce(5.0) → identity-range (~0.8+)
        """
        now = datetime.now(timezone.utc)

        # Find memories eligible for goal-range promotion
        goal_candidates = await self.memory.pool.fetch(
            """
            SELECT id, content, type, access_count, created_at,
                   depth_weight_alpha, depth_weight_beta, immutable,
                   evidence_count, confidence
            FROM memories
            WHERE access_count >= $1
              AND created_at < NOW() - INTERVAL '1 day' * $2
              AND depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) < 0.65
              AND NOT immutable
            ORDER BY access_count DESC
            LIMIT 20
            """,
            PROMOTE_GOAL_MIN_COUNT,
            PROMOTE_GOAL_MIN_DAYS,
        )

        promoted_goal = 0
        for mem in goal_candidates:
            gain = PROMOTE_GOAL_REINFORCE
            if self.memory.safety:
                allowed, adj_gain, _, reasons = self.memory.safety.check_weight_change(
                    memory_id=mem["id"],
                    current_alpha=float(mem["depth_weight_alpha"]),
                    current_beta=float(mem["depth_weight_beta"]),
                    delta_alpha=gain,
                    is_immutable=bool(mem["immutable"]),
                    evidence_count=mem["evidence_count"],
                    confidence=float(mem["confidence"]),
                    cycle_id=cycle_id,
                )
                if not allowed:
                    logger.debug(f"Promotion blocked for {mem['id']}: {reasons}")
                    continue
                gain = adj_gain

            if gain > 0:
                await self.memory.pool.execute(
                    """
                    UPDATE memories
                    SET depth_weight_alpha = depth_weight_alpha + $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    gain, mem["id"],
                )
                promoted_goal += 1

        # Find memories eligible for identity-range promotion
        identity_candidates = await self.memory.pool.fetch(
            """
            SELECT id, content, type, access_count, created_at,
                   depth_weight_alpha, depth_weight_beta, immutable,
                   evidence_count, confidence
            FROM memories
            WHERE access_count >= $1
              AND created_at < NOW() - INTERVAL '1 day' * $2
              AND depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) BETWEEN 0.65 AND 0.82
              AND NOT immutable
            ORDER BY access_count DESC
            LIMIT 10
            """,
            PROMOTE_IDENTITY_MIN_COUNT,
            PROMOTE_IDENTITY_MIN_DAYS,
        )

        promoted_identity = 0
        for mem in identity_candidates:
            gain = PROMOTE_IDENTITY_REINFORCE
            if self.memory.safety:
                allowed, adj_gain, _, reasons = self.memory.safety.check_weight_change(
                    memory_id=mem["id"],
                    current_alpha=float(mem["depth_weight_alpha"]),
                    current_beta=float(mem["depth_weight_beta"]),
                    delta_alpha=gain,
                    is_immutable=bool(mem["immutable"]),
                    evidence_count=mem["evidence_count"],
                    confidence=float(mem["confidence"]),
                    cycle_id=cycle_id,
                )
                if not allowed:
                    logger.debug(f"Identity promotion blocked for {mem['id']}: {reasons}")
                    continue
                gain = adj_gain

            if gain > 0:
                await self.memory.pool.execute(
                    """
                    UPDATE memories
                    SET depth_weight_alpha = depth_weight_alpha + $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    gain, mem["id"],
                )
                promoted_identity += 1

        if promoted_goal or promoted_identity:
            logger.info(
                f"Promotion: {promoted_goal} to goal-range, "
                f"{promoted_identity} to identity-range"
            )

    # ── §4.4: DECAY + RECONSOLIDATION ───────────────────────────────────────

    async def _decay_and_reconsolidate(self, cycle_id: str):
        """Decay stale memories, reconsolidate changed insights."""
        # Aggressive decay: 90+ days unused, access_count < 3
        stale = await self.memory.get_stale_memories(
            stale_days=DECAY_STALE_DAYS, min_access_count=DECAY_MIN_ACCESS,
        )
        if stale:
            stale_ids = [m["id"] for m in stale]

            # Apply via safety if available
            decayed = 0
            for mid in stale_ids:
                if self.memory.safety:
                    allowed, _, adj_beta, reasons = self.memory.safety.check_weight_change(
                        memory_id=mid,
                        current_alpha=1.0,  # Approximate
                        current_beta=4.0,
                        delta_beta=DECAY_CONTRADICT_AMOUNT,
                        cycle_id=cycle_id,
                    )
                    # Decay is always applied (safety just logs)

                await self.memory.pool.execute(
                    """
                    UPDATE memories
                    SET depth_weight_beta = depth_weight_beta + $1, updated_at = NOW()
                    WHERE id = $2 AND NOT immutable
                    """,
                    DECAY_CONTRADICT_AMOUNT, mid,
                )
                decayed += 1

            logger.info(f"Decay: {decayed} stale memories suppressed (beta +{DECAY_CONTRADICT_AMOUNT})")

        # Reconsolidation: re-evaluate insights whose sources changed recently
        await self._reconsolidate_changed_insights(cycle_id)

    async def _reconsolidate_changed_insights(self, cycle_id: str):
        """Re-evaluate insights when source memories have changed."""
        # Find insights whose source memories were updated recently
        rows = await self.memory.pool.fetch(
            """
            SELECT DISTINCT i.id AS insight_id, i.content AS insight_content,
                   i.evidence_count
            FROM memory_supersedes ms
            JOIN memories i ON i.id = ms.insight_id
            JOIN memories s ON s.id = ms.source_id
            WHERE s.updated_at > NOW() - INTERVAL '1 hour'
              AND i.updated_at < s.updated_at
            LIMIT 10
            """,
        )
        if not rows:
            return

        reconsolidated = 0
        for row in rows:
            # Get current source memories
            sources = await self.memory.why_do_i_believe(row["insight_id"])
            if not sources:
                continue

            # Check if insight still holds given updated sources
            source_text = "\n".join(
                f"- {s['content'][:150]}" for s in sources[:5]
            )

            if self._genai_client:
                updated = await self._revalidate_insight(
                    row["insight_content"], source_text,
                )
                if updated and updated != row["insight_content"]:
                    await self.memory.pool.execute(
                        """
                        UPDATE memories SET content = $1, updated_at = NOW()
                        WHERE id = $2
                        """,
                        updated, row["insight_id"],
                    )
                    reconsolidated += 1

        if reconsolidated:
            logger.info(f"Reconsolidation: {reconsolidated} insights updated")

    async def _revalidate_insight(
        self, insight: str, source_text: str,
    ) -> str | None:
        """Check if an insight still holds given updated source evidence."""
        prompt = (
            f"Original insight: {insight}\n\n"
            f"Current source evidence:\n{source_text}\n\n"
            "Does this insight still hold? If it needs updating based on the "
            "evidence, provide the updated insight. If it still holds as-is, "
            "reply exactly 'UNCHANGED'."
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=200,
                        temperature=0.2,
                    ),
                )
                return response.text.strip()

            result = await retry_llm_call(
                _call, config=self.retry_config, label="reconsolidate",
            )
            if result and result.upper() != "UNCHANGED":
                return result
        except Exception as e:
            logger.warning(f"Insight revalidation failed: {e}")
        return None

    # ── §4.5: GATE TUNING + DIRICHLET EVOLUTION ────────────────────────────

    async def _tune_parameters(self, cycle_id: str):
        """Adjust gate parameters and Dirichlet alphas based on outcomes.

        Gate analysis: dropped content needed later → too aggressive.
        Persisted but never retrieved → too permissive.
        """
        # Analyze gate effectiveness: check scratch buffer for patterns
        # This needs outcome tracking data which accumulates over time.
        # For now, log stats and prepare the infrastructure.

        total_memories = await self.memory.memory_count()
        avg_center = await self.memory.avg_depth_weight_center()

        # Check entropy of weight distribution
        if self.memory.safety and total_memories > 20:
            centers = await self.memory.pool.fetch(
                """
                SELECT depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) AS center
                FROM memories
                WHERE NOT immutable
                """,
            )
            weight_centers = [float(r["center"]) for r in centers]
            self.memory.safety.check_entropy(weight_centers)

        logger.debug(
            f"Tuning: {total_memories} memories, avg center={avg_center:.3f}"
        )

    # ── §4.6: CONTEXTUAL RETRIEVAL ──────────────────────────────────────────

    async def _contextual_retrieval(self, cycle_id: str):
        """Generate contextual preamble per memory for better embeddings.

        Anthropic: 67% retrieval failure reduction with full pipeline.
        """
        if not self._genai_client:
            return

        # Find memories lacking contextualized content
        rows = await self.memory.pool.fetch(
            """
            SELECT id, content, type, source, created_at
            FROM memories
            WHERE (metadata IS NULL OR metadata::text NOT LIKE '%content_contextualized%')
              AND embedding IS NOT NULL
              AND content IS NOT NULL
              AND length(content) > 20
            ORDER BY created_at DESC
            LIMIT 20
            """,
        )
        if not rows:
            return

        contextualized = 0
        for row in rows:
            preamble = await self._generate_context_preamble(row)
            if preamble:
                contextualized_content = f"{preamble} {row['content']}"

                # Re-embed with contextualized content
                try:
                    new_embedding = await self.memory.embed(
                        self.memory.prefixed_content(
                            contextualized_content, row["type"],
                        ),
                        task_type="RETRIEVAL_DOCUMENT",
                        title=row["type"],
                    )

                    await self.memory.pool.execute(
                        """
                        UPDATE memories
                        SET embedding = $1::halfvec,
                            metadata = jsonb_set(
                                COALESCE(metadata::jsonb, '{}'::jsonb),
                                '{content_contextualized}',
                                to_jsonb($2::text)
                            ),
                            updated_at = NOW()
                        WHERE id = $3
                        """,
                        str(new_embedding),
                        contextualized_content,
                        row["id"],
                    )
                    contextualized += 1
                except Exception as e:
                    logger.warning(f"Contextual re-embed failed for {row['id']}: {e}")

        if contextualized:
            logger.info(f"Contextual retrieval: {contextualized} memories re-embedded")

    async def _generate_context_preamble(self, memory_row) -> str | None:
        """Generate WHO/WHEN/WHY context preamble for a memory."""
        prompt = (
            f"Memory type: {memory_row['type']}\n"
            f"Source: {memory_row['source'] or 'unknown'}\n"
            f"Created: {memory_row['created_at']}\n"
            f"Content: {memory_row['content'][:300]}\n\n"
            "Give a short context preamble (WHO created this, WHEN, and WHY) "
            "in one sentence. This will be prepended to improve search retrieval."
        )
        try:
            async def _call():
                response = await self._genai_client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=100,
                        temperature=0.1,
                    ),
                )
                return response.text.strip()

            return await retry_llm_call(
                _call, config=self.retry_config, label="context_preamble",
            )
        except Exception as e:
            logger.warning(f"Context preamble generation failed: {e}")
            return None


# ── CONSOLIDATION ENGINE (COORDINATOR) ──────────────────────────────────────


class ConsolidationEngine:
    """Two-tier consolidation coordinator.

    Runs both constant (metabolism) and deep (sleep) consolidation concurrently.
    """

    def __init__(self, config, layers, memory, retry_config=None):
        self.constant = ConstantConsolidation(
            memory=memory, config=config, retry_config=retry_config,
        )
        self.deep = DeepConsolidation(
            memory=memory, layers=layers, config=config, retry_config=retry_config,
        )

    async def run(self, shutdown_event: asyncio.Event):
        """Run both consolidation tiers concurrently."""
        logger.info("Consolidation engine starting (Tier 1 + Tier 2)")
        await asyncio.gather(
            self.constant.run(shutdown_event),
            self.deep.run(shutdown_event),
        )
        logger.info("Consolidation engine stopped")
