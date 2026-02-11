"""Memory Gate — Entry and Exit gates for the cognitive loop.

Entry gate: Stochastic filter on every message, buffers to scratch.
Exit gate: ACT-R activation → 3x3 decision matrix → action.

All skip/persist probabilities are stochastic and evolvable by consolidation.
ACT-R equation structure is kept; parameters are human-calibrated starting
points that evolve through experience.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .activation import compute_activation, spreading_activation, cosine_similarity

logger = logging.getLogger("agent.gate")


# ── ENTRY GATE ──────────────────────────────────────────────────────────────


@dataclass
class EntryGateConfig:
    """Stochastic entry gate parameters. All skip rates are evolvable."""
    min_content_length: int = 10
    short_content_skip_rate: float = 0.95   # P(skip | content < min_length)
    mechanical_skip_rate: float = 0.90      # P(skip | mechanical content)
    base_buffer_rate: float = 0.99          # P(buffer | normal content)
    mechanical_prefixes: list[str] = field(default_factory=lambda: [
        "/", "[tool:", "[system:", "[error:", "```",
    ])


class EntryGate:
    """Stochastic entry gate — fires on every message, buffers to scratch.

    Not deterministic. Every skip has a stochastic floor that occasionally
    lets content through, giving the consolidation worker data on whether
    the heuristics are wrong. Skip rates evolve over time.
    """

    def __init__(self, config: EntryGateConfig | None = None):
        self.config = config or EntryGateConfig()
        self._stats = {"evaluated": 0, "buffered": 0, "skipped": 0}

    def evaluate(
        self, content: str, source: str = "unknown", source_tag: str = "external_user",
    ) -> tuple[bool, dict]:
        """Evaluate content for scratch buffering.

        Returns (should_buffer, metadata).
        Metadata includes decision reasoning for consolidation learning.
        """
        self._stats["evaluated"] += 1
        content_stripped = content.strip()

        metadata = {
            "source": source,
            "source_tag": source_tag,
            "content_length": len(content_stripped),
            "gate_decision": None,
            "gate_reason": None,
            "skip_probability": None,
            "dice_roll": None,
        }

        # Short content — high skip probability, but stochastic
        if len(content_stripped) < self.config.min_content_length:
            return self._stochastic_decision(
                skip_rate=self.config.short_content_skip_rate,
                reason="short_content",
                metadata=metadata,
            )

        # Mechanical content — high skip probability, but stochastic
        if self._is_mechanical(content_stripped):
            return self._stochastic_decision(
                skip_rate=self.config.mechanical_skip_rate,
                reason="mechanical",
                metadata=metadata,
            )

        # Normal content — high buffer probability, but stochastic
        return self._stochastic_decision(
            skip_rate=1.0 - self.config.base_buffer_rate,
            reason="normal_content",
            metadata=metadata,
        )

    def _stochastic_decision(
        self, skip_rate: float, reason: str, metadata: dict
    ) -> tuple[bool, dict]:
        """Make a stochastic buffer/skip decision."""
        roll = random.random()
        metadata["skip_probability"] = skip_rate
        metadata["dice_roll"] = roll

        if roll < skip_rate:
            metadata["gate_decision"] = "skip"
            metadata["gate_reason"] = reason
            self._stats["skipped"] += 1
            return False, metadata
        else:
            metadata["gate_decision"] = "buffer"
            metadata["gate_reason"] = f"{reason}_stochastic_pass"
            self._stats["buffered"] += 1
            return True, metadata

    def _is_mechanical(self, content: str) -> bool:
        """Check if content looks like tool/system output."""
        for prefix in self.config.mechanical_prefixes:
            if content.startswith(prefix):
                return True
        return False

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ── EXIT GATE ───────────────────────────────────────────────────────────────

# 3x3 decision matrix cell names
PERSIST_HIGH = "persist_high"        # Core + Contradicting / Core + Novel
PERSIST_FLAG = "persist_flag"        # Core + Contradicting (max priority)
PERSIST = "persist"                  # Peripheral + Contradicting
REINFORCE = "reinforce"              # Core + Confirming
BUFFER = "buffer"                    # Peripheral + Novel
SKIP = "skip"                        # Peripheral + Confirming
DROP = "drop"                        # Irrelevant row


@dataclass
class ExitGateConfig:
    """ACT-R exit gate with 3x3 decision matrix.

    Relevance axis from spreading activation S_i.
    Novelty axis from check_novelty().
    """
    # Relevance thresholds (from S_i)
    core_threshold: float = 0.6
    peripheral_threshold: float = 0.3
    # Novelty thresholds
    confirming_sim: float = 0.85
    novel_sim: float = 0.6
    contradiction_sim: float = 0.7
    # Stochastic noise floor for "drop" cells
    drop_noise_floor: float = 0.02
    # v0.1 emotional charge: centroid distance placeholder
    emotional_charge_bonus: float = 0.15
    emotional_charge_threshold: float = 0.3


class ExitGate:
    """ACT-R → 3x3 decision matrix exit gate.

    Pipeline: content → ACT-R activation → classify into 3x3 matrix → action.

    3x3 Matrix (Relevance × Novelty):
                     Confirming          Novel              Contradicting
    Core           | Reinforce(mod)    | PERSIST(high)    | PERSIST+FLAG(max)
    Peripheral     | Skip(low)         | Buffer(mod)      | Persist(high)
    Irrelevant     | Drop              | Drop(noise)      | Drop(noise)

    Gate starts PERMISSIVE — over-persisting is recoverable, dropping is permanent.
    """

    def __init__(self, config: ExitGateConfig | None = None):
        self.config = config or ExitGateConfig()
        self._stats = {
            "evaluated": 0, "persisted": 0, "dropped": 0,
            "buffered": 0, "reinforced": 0, "flagged": 0,
        }

    async def evaluate(
        self,
        content: str,
        memory_store,
        layers,
        attention_embedding: np.ndarray | None = None,
        conversation_context: list[dict] | None = None,
    ) -> tuple[bool, float, dict]:
        """Score content through 3x3 matrix. Returns (should_persist, score, metadata)."""
        self._stats["evaluated"] += 1

        # 1. Embed content for similarity computations
        content_embedding = np.array(
            await memory_store.embed(content, task_type="SEMANTIC_SIMILARITY"),
            dtype=np.float32,
        )

        # 2. RELEVANCE AXIS — spreading activation
        layer_embeddings = layers.get_all_layer_embeddings()
        s_i = spreading_activation(
            content_embedding, attention_embedding, layer_embeddings,
        )

        # 3. NOVELTY AXIS — check against existing memories
        is_novel, max_sim = await memory_store.check_novelty(content)
        contradiction = self._detect_contradiction_heuristic(
            content, memory_store, max_sim
        )

        # 4. v0.1 EMOTIONAL CHARGE placeholder (centroid distance)
        # Will be replaced by gut.py (§5.1)
        emotional_charge = 0.0  # neutral until gut feeling implemented

        # 5. CLASSIFY into 3x3 matrix
        relevance_class = self._classify_relevance(s_i)
        novelty_class = self._classify_novelty(max_sim, contradiction)
        cell = self._matrix_decision(relevance_class, novelty_class)

        # 6. Compute gate score
        score = self._cell_score(cell, s_i, max_sim, contradiction, emotional_charge)

        # 7. Apply stochastic noise floor (even "drop" cells can surprise)
        if cell == DROP and random.random() < self.config.drop_noise_floor:
            cell = BUFFER
            score = max(score, 0.1)

        should_persist = cell in (PERSIST_HIGH, PERSIST_FLAG, PERSIST, REINFORCE)
        should_buffer = cell == BUFFER

        # Stats
        if cell == PERSIST_FLAG:
            self._stats["flagged"] += 1
            self._stats["persisted"] += 1
        elif should_persist:
            self._stats["persisted"] += 1
        elif should_buffer:
            self._stats["buffered"] += 1
        else:
            if cell == REINFORCE:
                self._stats["reinforced"] += 1
            else:
                self._stats["dropped"] += 1

        metadata = {
            "spreading_activation": s_i,
            "max_similarity": max_sim,
            "contradiction": contradiction,
            "emotional_charge": emotional_charge,
            "relevance_class": relevance_class,
            "novelty_class": novelty_class,
            "matrix_cell": cell,
            "final_score": score,
            "decision": cell,
        }

        logger.info(
            f"Exit gate: {cell} "
            f"(score={score:.3f}, S_i={s_i:.3f}, sim={max_sim:.3f}, "
            f"rel={relevance_class}, nov={novelty_class})"
        )
        return should_persist or should_buffer, score, metadata

    def _classify_relevance(self, s_i: float) -> str:
        if s_i > self.config.core_threshold:
            return "core"
        elif s_i > self.config.peripheral_threshold:
            return "peripheral"
        return "irrelevant"

    def _classify_novelty(self, max_sim: float, contradiction: float) -> str:
        if contradiction > 0.3:
            return "contradicting"
        if max_sim > self.config.confirming_sim:
            return "confirming"
        if max_sim < self.config.novel_sim:
            return "novel"
        return "novel"  # default to novel (permissive gate)

    def _matrix_decision(self, relevance: str, novelty: str) -> str:
        """3x3 matrix lookup."""
        matrix = {
            ("core", "confirming"):     REINFORCE,
            ("core", "novel"):          PERSIST_HIGH,
            ("core", "contradicting"):  PERSIST_FLAG,
            ("peripheral", "confirming"): SKIP,
            ("peripheral", "novel"):    BUFFER,
            ("peripheral", "contradicting"): PERSIST,
            ("irrelevant", "confirming"): DROP,
            ("irrelevant", "novel"):    DROP,
            ("irrelevant", "contradicting"): DROP,
        }
        return matrix.get((relevance, novelty), DROP)

    def _cell_score(
        self, cell: str, s_i: float, max_sim: float,
        contradiction: float, emotional_charge: float,
    ) -> float:
        """Compute a score for the cell decision (used for downstream ranking)."""
        base_scores = {
            PERSIST_FLAG: 0.95,
            PERSIST_HIGH: 0.85,
            PERSIST: 0.70,
            REINFORCE: 0.50,
            BUFFER: 0.40,
            SKIP: 0.15,
            DROP: 0.05,
        }
        score = base_scores.get(cell, 0.05)

        # Emotional charge bonus
        if emotional_charge > self.config.emotional_charge_threshold:
            score += self.config.emotional_charge_bonus

        # Modulate by actual spreading activation strength
        score = score * (0.5 + 0.5 * s_i)

        return min(1.0, max(0.0, score))

    def _detect_contradiction_heuristic(
        self, content: str, memory_store, max_sim: float,
    ) -> float:
        """Layer 1 contradiction detection: negation marker asymmetry.

        Cheap heuristic (~0ms). Layers 2-3 (embedding opposition, LLM micro-call)
        added in later tasks.
        """
        if max_sim < self.config.contradiction_sim:
            return 0.0
        # We don't have the existing content here yet — return 0 for now.
        # Full contradiction detection will use the retrieved memory content
        # from check_novelty when that method is enhanced to return content.
        return 0.0

    @staticmethod
    def detect_contradiction_negation(new_content: str, existing_content: str) -> float:
        """Negation marker asymmetry heuristic (~0ms)."""
        negation_markers = [
            "not", "dont", "doesnt", "isnt", "wasnt", "wont",
            "cant", "never", "no longer", "stopped", "changed",
            "actually", "instead", "wrong", "incorrect", "mistaken",
            "however", "but actually", "on the contrary", "opposite",
            "disagree", "unlike", "different from",
        ]
        new_lower = new_content.lower()
        existing_lower = existing_content.lower()
        asymmetry = sum(
            1 for m in negation_markers
            if (m in new_lower) != (m in existing_lower)
        )
        return min(1.0, asymmetry * 0.15)

    @property
    def stats(self) -> dict:
        return dict(self._stats)
