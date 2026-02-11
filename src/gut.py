"""Gut Feeling — two-centroid delta model (§5.1).

Maps to Free Energy Principle. The "gut feeling" is the delta between
the subconscious centroid (who I am) and the attention centroid (what's
happening now). Magnitude = motivational intensity. Direction = valence.

Subconscious centroid:
  0.5 * weighted_avg(L0) + 0.25 * weighted_avg(L1) + 0.25 * weighted_avg(L2)

Attention centroid:
  Recency-weighted average of current context embeddings.

Delta = attention - subconscious (768-dim)
  - Feeds into hybrid relevance (§3.1) as emotional/valence component
  - Feeds into attention allocation (§3.10) as emotional_charge
  - Logged for PCA during deep consolidation
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("agent.gut")


@dataclass
class GutDelta:
    """A single gut feeling measurement."""
    delta: np.ndarray           # 768-dim difference vector
    magnitude: float            # L2 norm of delta
    direction: np.ndarray       # Unit vector (delta / magnitude)
    context: str                # What triggered this measurement
    timestamp: float = field(default_factory=time.time)
    outcome_id: str | None = None  # Forward-linkable for PCA (§5.3)


class GutFeeling:
    """Two-centroid gut feeling model.

    Maintains subconscious and attention centroids, computes delta,
    and provides emotional signals for the rest of the cognitive system.
    """

    # Layer weights for subconscious centroid (starting points — consolidation evolves)
    LAYER_WEIGHTS = {"L0": 0.5, "L1": 0.25, "L2": 0.25}

    # Recency half-life for attention centroid (in embeddings seen)
    ATTENTION_HALFLIFE = 10

    def __init__(self):
        self.subconscious_centroid: np.ndarray | None = None
        self.attention_centroid: np.ndarray | None = None
        self._attention_history: list[np.ndarray] = []
        self._delta_log: list[GutDelta] = []
        self._max_log = 500

    def update_subconscious(
        self,
        l0_embeddings: list[np.ndarray] | None = None,
        l1_embeddings: list[np.ndarray] | None = None,
        l2_embeddings: list[np.ndarray] | None = None,
        l2_weights: list[float] | None = None,
    ) -> np.ndarray | None:
        """Recompute subconscious centroid from layer embeddings.

        Called after deep consolidation or at session start.
        L2 = importance-weighted average of all memory embeddings.
        """
        components = {}

        if l0_embeddings:
            components["L0"] = np.mean(l0_embeddings, axis=0)

        if l1_embeddings:
            components["L1"] = np.mean(l1_embeddings, axis=0)

        if l2_embeddings:
            if l2_weights and len(l2_weights) == len(l2_embeddings):
                # Importance-weighted average
                weights = np.array(l2_weights)
                total = weights.sum()
                if total > 0:
                    weights = weights / total
                    components["L2"] = np.average(l2_embeddings, axis=0, weights=weights)
            else:
                components["L2"] = np.mean(l2_embeddings, axis=0)

        if not components:
            return None

        # Weighted combination
        centroid = np.zeros_like(next(iter(components.values())))
        total_weight = 0.0
        for layer, emb in components.items():
            w = self.LAYER_WEIGHTS.get(layer, 0.1)
            centroid += w * emb
            total_weight += w

        if total_weight > 0:
            centroid /= total_weight

        self.subconscious_centroid = centroid
        logger.debug(f"Subconscious centroid updated from {list(components.keys())}")
        return centroid

    def update_attention(self, embedding: np.ndarray) -> np.ndarray:
        """Update attention centroid with a new context embedding.

        Uses recency-weighted averaging with exponential decay.
        """
        self._attention_history.append(embedding)

        # Keep last N embeddings
        max_history = 50
        if len(self._attention_history) > max_history:
            self._attention_history = self._attention_history[-max_history:]

        # Recency-weighted average (exponential decay)
        n = len(self._attention_history)
        weights = np.array([
            np.exp(-0.693 * (n - 1 - i) / self.ATTENTION_HALFLIFE)
            for i in range(n)
        ])
        weights /= weights.sum()

        self.attention_centroid = np.average(
            self._attention_history, axis=0, weights=weights,
        )
        return self.attention_centroid

    def compute_delta(self, context: str = "") -> GutDelta | None:
        """Compute the current gut feeling delta.

        Returns None if either centroid is unavailable.
        """
        if self.subconscious_centroid is None or self.attention_centroid is None:
            return None

        delta = self.attention_centroid - self.subconscious_centroid
        magnitude = float(np.linalg.norm(delta))

        if magnitude > 0:
            direction = delta / magnitude
        else:
            direction = np.zeros_like(delta)

        gut = GutDelta(
            delta=delta,
            magnitude=magnitude,
            direction=direction,
            context=context[:200],
        )

        # Log for PCA
        self._delta_log.append(gut)
        if len(self._delta_log) > self._max_log:
            self._delta_log.pop(0)

        return gut

    @property
    def emotional_charge(self) -> float:
        """Current emotional charge for attention allocation.

        Normalized gut magnitude: 0.0 = calm, 1.0 = high intensity.
        Replaces the placeholder in AttentionAllocator.
        """
        if not self._delta_log:
            return 0.0

        latest = self._delta_log[-1]
        # Normalize: typical magnitude range is 0-2 for normalized embeddings
        return min(1.0, latest.magnitude / 2.0)

    @property
    def emotional_alignment(self) -> float:
        """Emotional alignment score for hybrid relevance (§3.1).

        Replaces the neutral 0.5 default in relevance.compute_emotional_alignment().
        Based on whether current attention aligns with or diverges from identity.
        """
        if not self._delta_log:
            return 0.5  # Neutral

        latest = self._delta_log[-1]
        # Low magnitude = aligned with self (high score)
        # High magnitude = diverging from self (low score)
        # Mapped: 0 magnitude → 1.0, 2+ magnitude → 0.0
        return max(0.0, 1.0 - latest.magnitude / 2.0)

    def gut_summary(self) -> str:
        """One-line summary for system prompt injection."""
        if not self._delta_log:
            return "Gut: neutral (no data)"

        latest = self._delta_log[-1]
        intensity = "high" if latest.magnitude > 1.0 else "moderate" if latest.magnitude > 0.5 else "low"
        alignment = "aligned" if latest.magnitude < 0.5 else "divergent" if latest.magnitude > 1.0 else "shifting"

        return f"Gut: {intensity} intensity, {alignment} with identity (mag={latest.magnitude:.2f})"

    def get_delta_log(self, last_n: int = 50) -> list[GutDelta]:
        """Get recent delta log for PCA analysis."""
        return self._delta_log[-last_n:]

    def link_outcome(self, outcome_id: str, last_n: int = 1) -> None:
        """Forward-link recent deltas to an outcome (§5.3)."""
        for delta in self._delta_log[-last_n:]:
            delta.outcome_id = outcome_id
