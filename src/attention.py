"""Attention Allocation — salience-based competition among input candidates.

Every cognitive cycle, before the main LLM call:
  1. Collect all pending input candidates
  2. Compute salience for each
  3. Highest salience wins attention
  4. Losers stay in queue with decaying urgency
  5. Generate cognitive state report for LLM context injection

The attention allocation is subconscious (computed in Python).
The cognitive state report makes it visible to conscious processing (the LLM).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .activation import cosine_similarity

logger = logging.getLogger("agent.attention")

# Default urgency values by source type
DEFAULT_URGENCY = {
    "external_user": 0.8,
    "internal_dmn": 0.2,
    "internal_consolidation": 0.3,
    "internal_gut": 0.4,
    "internal_scheduled": 0.5,
}

# Urgency decay per cycle for losing candidates
URGENCY_DECAY = 0.9


@dataclass
class AttentionCandidate:
    """A candidate competing for attention this cycle."""
    content: str
    source_tag: str
    embedding: np.ndarray | None = None
    urgency: float = 0.5
    salience: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.urgency == 0.5 and self.source_tag in DEFAULT_URGENCY:
            self.urgency = DEFAULT_URGENCY[self.source_tag]


class AttentionAllocator:
    """Salience-based attention competition."""

    def __init__(self):
        self._queue: list[AttentionCandidate] = []
        self._previous_embedding: np.ndarray | None = None
        self._recent_embeddings: list[np.ndarray] = []  # rolling window for centroid
        self._max_recent = 20

    def add_candidate(self, candidate: AttentionCandidate):
        """Add a new candidate to the attention queue."""
        self._queue.append(candidate)

    def select_winner(
        self,
        goal_embeddings: list[tuple[str, float, np.ndarray]] | None = None,
        gut_delta: float | None = None,
    ) -> tuple[AttentionCandidate | None, list[AttentionCandidate], str]:
        """Select the highest-salience candidate.

        Returns (winner, losers, cognitive_state_report).
        """
        if not self._queue:
            return None, [], "[COGNITIVE STATE]\nNo pending input candidates.\n"

        # Score all candidates
        for candidate in self._queue:
            candidate.salience = self._compute_salience(
                candidate, goal_embeddings, gut_delta,
            )

        # Sort by salience descending
        self._queue.sort(key=lambda c: c.salience, reverse=True)

        winner = self._queue[0]
        losers = self._queue[1:]

        # Decay urgency of losers, keep in queue
        for loser in losers:
            loser.urgency *= URGENCY_DECAY

        # Update attention history
        if winner.embedding is not None:
            self._previous_embedding = winner.embedding
            self._recent_embeddings.append(winner.embedding)
            if len(self._recent_embeddings) > self._max_recent:
                self._recent_embeddings.pop(0)

        # Generate cognitive state report
        report = self._generate_report(winner, losers)

        # Clear queue (losers re-added externally if still relevant)
        self._queue = list(losers)

        logger.info(
            f"Attention: winner={winner.source_tag} "
            f"(salience={winner.salience:.3f}), "
            f"{len(losers)} losers queued"
        )

        return winner, losers, report

    def _compute_salience(
        self,
        candidate: AttentionCandidate,
        goal_embeddings: list[tuple[str, float, np.ndarray]] | None,
        gut_delta: float | None,
    ) -> float:
        """Compute salience for a candidate."""
        novelty = 0.5  # default
        relevance = 0.0
        emotional_charge = 0.0

        if candidate.embedding is not None:
            # Novelty: how different from recent context
            if self._recent_embeddings:
                max_sim = max(
                    cosine_similarity(candidate.embedding, e)
                    for e in self._recent_embeddings[-5:]
                )
                novelty = 1.0 - max_sim
            else:
                novelty = 1.0  # everything novel at start

            # Relevance to active goals
            if goal_embeddings:
                weighted_sim = 0.0
                total_weight = 0.0
                for _text, weight, vec in goal_embeddings:
                    sim = cosine_similarity(candidate.embedding, vec)
                    weighted_sim += weight * sim
                    total_weight += weight
                if total_weight > 0:
                    relevance = weighted_sim / total_weight

        # Emotional charge from gut delta
        if gut_delta is not None:
            emotional_charge = abs(gut_delta)

        raw = (
            novelty * 0.3
            + relevance * 0.3
            + emotional_charge * 0.2
            + candidate.urgency * 0.2
        )
        return raw

    def _generate_report(
        self,
        winner: AttentionCandidate,
        losers: list[AttentionCandidate],
    ) -> str:
        """Generate cognitive state report for LLM context injection."""
        lines = ["[COGNITIVE STATE]", "Attention candidates this cycle:"]

        all_candidates = [winner] + losers
        for c in all_candidates[:5]:  # cap at 5 for token budget
            label = c.source_tag.replace("_", " ").title()
            preview = c.content[:60].replace("\n", " ")
            lines.append(f'  - {label}: "{preview}" (salience: {c.salience:.2f})')

        winner_label = winner.source_tag.replace("_", " ").title()
        lines.append(f"Winner: {winner_label}")

        if winner.source_tag != "external_user" and any(
            c.source_tag == "external_user" for c in losers
        ):
            lines.append("Note: User message was deprioritized this cycle.")

        return "\n".join(lines) + "\n"

    @property
    def previous_attention_embedding(self) -> np.ndarray | None:
        """Previous cycle's attention embedding for context inertia."""
        return self._previous_embedding

    @property
    def attention_centroid(self) -> np.ndarray | None:
        """Recency-weighted average of recent attention embeddings."""
        if not self._recent_embeddings:
            return None
        # Simple recency weighting: more recent = higher weight
        n = len(self._recent_embeddings)
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        centroid = np.zeros_like(self._recent_embeddings[0])
        for w, emb in zip(weights, self._recent_embeddings):
            centroid += w * emb
        return centroid

    @property
    def queue_size(self) -> int:
        return len(self._queue)
