"""Idle Loop — the Default Mode Network (§4.7).

Self-generated input feeding the main cognitive loop when attention is free.
NOT a separate processing pipeline — DMN generates inputs that queue for
the main cognitive loop via AttentionAllocator. Processing is identical
to user messages. Attention allocation determines whether DMN input wins.

Stochastic sampling biases:
  - High weight + low recent access (neglected important memories)
  - Memories conflicting with high-weight memories (tension detection)
  - Temporally distant memories (creative association)
  - High-weight self-referential memories (spontaneous introspection)

Three output channels:
  1. Memory + goal connection → self-prompt (purposeful)
  2. Disparate memory connection → insight for consolidation (creative)
  3. Memory + identity connection → evaluation signal (identity refinement)

Activity suppression: DMN items have low urgency, lose to user messages naturally.
2-hop spreading activation enabled during DMN cycles.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone

import numpy as np

from .attention import AttentionCandidate

logger = logging.getLogger("agent.idle")

# Default urgency for DMN candidates (low — suppressed during conversation)
DMN_URGENCY = 0.2

# Sampling bias weights
BIAS_NEGLECTED = 0.35      # High weight + low recent access
BIAS_TENSION = 0.20        # Conflicting with high-weight memories
BIAS_TEMPORAL = 0.20       # Temporally distant (creative association)
BIAS_INTROSPECTION = 0.25  # High-weight self-referential


class IdleLoop:
    """The agent's resting state — Default Mode Network.

    Generates DMN candidates and queues them for the cognitive loop.
    Heartbeat interval adapts to activity level (biological anticorrelation).
    """

    def __init__(self, config, layers, memory, dmn_queue: asyncio.Queue):
        self.config = config
        self.layers = layers
        self.memory = memory
        self.dmn_queue = dmn_queue
        self.last_activity = datetime.now(timezone.utc)
        self.heartbeat_count = 0
        self._recent_topics: list[str] = []  # Track for entropy guard

    def _get_interval(self) -> float:
        """Adaptive heartbeat interval based on idle time."""
        idle_seconds = (datetime.now(timezone.utc) - self.last_activity).total_seconds()
        idle_minutes = idle_seconds / 60

        intervals = self.config.raw.get("idle", {}).get("intervals", {})

        if idle_minutes < 10:
            return intervals.get("post_task_minutes", 1) * 60
        elif idle_minutes < 60:
            return intervals.get("idle_10min", 5) * 60
        elif idle_minutes < 240:
            return intervals.get("idle_1hour", 15) * 60
        else:
            return intervals.get("idle_4hours", 30) * 60

    async def run(self, shutdown_event: asyncio.Event):
        """Run the idle loop."""
        logger.info("Idle loop (DMN) started.")

        while not shutdown_event.is_set():
            interval = self._get_interval()

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=interval,
                )
                break
            except asyncio.TimeoutError:
                pass

            try:
                await self._heartbeat()
            except Exception as e:
                logger.error(f"DMN heartbeat error: {e}", exc_info=True)

        logger.info("Idle loop stopped.")

    async def _heartbeat(self):
        """One heartbeat of the default mode network.

        Sample a memory, generate a DMN thought, queue it for the cognitive loop.
        """
        self.heartbeat_count += 1

        idle_minutes = (datetime.now(timezone.utc) - self.last_activity).total_seconds() / 60
        logger.debug(
            f"DMN heartbeat #{self.heartbeat_count} "
            f"(idle: {idle_minutes:.0f}m, interval: {self._get_interval():.0f}s)"
        )

        # Sample a memory with stochastic biases
        sampled = await self._sample_memory()
        if not sampled:
            return

        memory_content = sampled["content"]
        memory_type = sampled.get("type", "unknown")
        memory_id = sampled["id"]

        # Determine output channel and generate thought
        channel, thought = await self._generate_thought(sampled)
        if not thought:
            return

        # Entropy guard: check if we keep surfacing the same topic
        if self._is_repetitive(thought):
            logger.debug("DMN entropy guard: suppressing repetitive topic")
            return

        self._recent_topics.append(thought[:50])
        if len(self._recent_topics) > 20:
            self._recent_topics.pop(0)

        # Embed the thought for attention allocation
        try:
            embedding = await self.memory.embed(thought, task_type="RETRIEVAL_QUERY")
            emb_array = np.array(embedding)
        except Exception:
            emb_array = None

        # Create attention candidate with DMN urgency
        candidate = AttentionCandidate(
            content=thought,
            source_tag="internal_dmn",
            embedding=emb_array,
            urgency=DMN_URGENCY,
            metadata={
                "channel": channel,
                "source_memory_id": memory_id,
                "source_memory_type": memory_type,
                "heartbeat": self.heartbeat_count,
            },
        )

        # Queue for cognitive loop (non-blocking)
        try:
            self.dmn_queue.put_nowait(candidate)
            logger.debug(f"DMN queued [{channel}]: {thought[:60]}")
        except asyncio.QueueFull:
            logger.debug("DMN queue full — dropping candidate")

    async def _sample_memory(self) -> dict | None:
        """Stochastic memory sampling with biases toward interesting content."""
        # Roll which bias to use
        roll = random.random()

        if roll < BIAS_NEGLECTED:
            return await self._sample_neglected()
        elif roll < BIAS_NEGLECTED + BIAS_TENSION:
            return await self._sample_tension_candidate()
        elif roll < BIAS_NEGLECTED + BIAS_TENSION + BIAS_TEMPORAL:
            return await self._sample_temporal()
        else:
            return await self._sample_introspective()

    async def _sample_neglected(self) -> dict | None:
        """High weight + low recent access — neglected important memories."""
        row = await self.memory.pool.fetchrow(
            """
            SELECT id, content, type, depth_weight_alpha, depth_weight_beta,
                   last_accessed, tags
            FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.5
              AND (last_accessed IS NULL OR last_accessed < NOW() - INTERVAL '7 days')
              AND embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
            """,
        )
        return dict(row) if row else await self.memory.get_random_memory()

    async def _sample_tension_candidate(self) -> dict | None:
        """Memory that might conflict with high-weight memories."""
        # Get a random high-weight memory, then find something potentially conflicting
        high = await self.memory.pool.fetchrow(
            """
            SELECT id, content, type, embedding
            FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.7
              AND embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
            """,
        )
        if not high:
            return await self.memory.get_random_memory()

        # Find a memory that's moderately similar but different type
        row = await self.memory.pool.fetchrow(
            """
            SELECT id, content, type, depth_weight_alpha, depth_weight_beta,
                   last_accessed, tags
            FROM memories
            WHERE id != $1
              AND type != $2
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> (SELECT embedding FROM memories WHERE id = $1)) BETWEEN 0.3 AND 0.7
            ORDER BY RANDOM()
            LIMIT 1
            """,
            high["id"],
            high["type"],
        )
        return dict(row) if row else await self.memory.get_random_memory()

    async def _sample_temporal(self) -> dict | None:
        """Temporally distant memories — creative association."""
        row = await self.memory.pool.fetchrow(
            """
            SELECT id, content, type, depth_weight_alpha, depth_weight_beta,
                   last_accessed, tags
            FROM memories
            WHERE created_at < NOW() - INTERVAL '30 days'
              AND embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
            """,
        )
        return dict(row) if row else await self.memory.get_random_memory()

    async def _sample_introspective(self) -> dict | None:
        """High-weight self-referential memories — spontaneous introspection."""
        row = await self.memory.pool.fetchrow(
            """
            SELECT id, content, type, depth_weight_alpha, depth_weight_beta,
                   last_accessed, tags
            FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.6
              AND type IN ('reflection', 'narrative', 'preference', 'tension')
              AND embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1
            """,
        )
        return dict(row) if row else await self.memory.get_random_memory()

    async def _generate_thought(self, memory: dict) -> tuple[str, str | None]:
        """Generate a DMN thought from a sampled memory.

        Returns (channel, thought_text) or (channel, None) if no thought generated.
        """
        content = memory.get("content", "")
        mem_type = memory.get("type", "unknown")

        # Check for goal connection (channel 1: purposeful)
        if self.layers.layer1:
            goals = self.layers.layer1.get("active_goals", [])
            for goal in goals:
                goal_desc = goal.get("description", "")
                if goal_desc and any(
                    word in content.lower()
                    for word in goal_desc.lower().split()[:5]
                ):
                    thought = (
                        f"[DMN/goal] I noticed a memory that connects to my goal "
                        f"'{goal_desc[:80]}': {content[:200]}"
                    )
                    return "goal_connection", thought

        # Check for identity connection (channel 3: identity refinement)
        if mem_type in ("reflection", "narrative", "tension"):
            thought = (
                f"[DMN/identity] Revisiting a {mem_type} memory: {content[:200]}. "
                f"Does this still feel true?"
            )
            return "identity_refinement", thought

        # Check for spreading activation (2-hop) to find creative connections
        mem_id = memory.get("id", "")
        if mem_id and self.memory.pool:
            try:
                from .relevance import spread_activation
                activated = await spread_activation(
                    self.memory.pool,
                    seed_ids=[mem_id],
                    hops=2,  # 2-hop during DMN
                    top_k_per_hop=3,
                )
                if activated:
                    # Get the most activated partner
                    top_id = max(activated, key=activated.get)
                    partner = await self.memory.get_memory(top_id)
                    if partner and partner["content"] != content:
                        thought = (
                            f"[DMN/creative] Connection between: "
                            f"'{content[:100]}' and '{partner['content'][:100]}'"
                        )
                        return "creative_insight", thought
            except Exception as e:
                logger.debug(f"Spreading activation failed: {e}")

        # Default: surface the memory as a general reflection
        thought = f"[DMN/reflect] Surfacing memory ({mem_type}): {content[:200]}"
        return "general_reflection", thought

    def _is_repetitive(self, thought: str) -> bool:
        """Entropy guard: check if DMN keeps surfacing the same topic."""
        if len(self._recent_topics) < 3:
            return False

        # Simple check: if the first 50 chars match 2+ recent topics
        prefix = thought[:50].lower()
        matches = sum(1 for t in self._recent_topics[-5:] if t.lower() == prefix)
        return matches >= 2

    def notify_activity(self):
        """Called when the agent processes user input — resets idle timer."""
        self.last_activity = datetime.now(timezone.utc)

    async def metacognitive_review(self) -> str | None:
        """Generate a metacognitive review prompt during idle cycles.

        "Last cycle I prioritized X over Y. Was that right?"
        These are regular DMN outputs that compete for attention.
        """
        # This will be triggered periodically during extended idle
        thought = (
            "[DMN/meta] Reflecting on recent attention choices: "
            "Were my priorities aligned with my goals? "
            "Was there anything I dismissed too quickly?"
        )
        return thought
