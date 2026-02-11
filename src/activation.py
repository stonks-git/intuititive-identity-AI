"""ACT-R Activation Equation — 4-component memory activation scoring.

Implements the classic ACT-R equation: A_i = B_i + S_i + P_i + ε_i

Components:
  B_i: Base-level learning (recency/frequency from access_timestamps)
  S_i: Spreading activation (cosine sim to context + L0/L1 embeddings)
  P_i: Partial matching penalty (metadata mismatches)
  ε_i: Logistic noise (s=0.4)

Parameters are human-calibrated starting points. Consolidation evolves them.
Decades-validated cognitive science: d=0.5, s=0.4, P=-1.0, tau=0.0.
"""

import math
import random
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger("agent.activation")

# ACT-R default parameters (human-calibrated starting points)
DEFAULT_DECAY_D = 0.5       # base-level decay rate
DEFAULT_NOISE_S = 0.4       # logistic noise spread
DEFAULT_MISMATCH_P = -1.0   # partial matching penalty scale
DEFAULT_THRESHOLD_TAU = 0.0  # persist threshold


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def base_level_activation(
    access_timestamps: list[datetime],
    now: datetime | None = None,
    d: float = DEFAULT_DECAY_D,
) -> float:
    """B_i = ln(sum(t_j^{-d})) where t_j is seconds since each access.

    Uses access_timestamps array (TSM paper: dialogue time).
    Falls back to 0.0 if no access history.
    """
    if not access_timestamps:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc)

    total = 0.0
    for ts in access_timestamps:
        # Ensure timezone-aware comparison
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_seconds = max(1.0, (now - ts).total_seconds())
        total += age_seconds ** (-d)

    if total <= 0:
        return 0.0
    return math.log(total)


def spreading_activation(
    memory_embedding: np.ndarray,
    attention_embedding: np.ndarray | None = None,
    layer_embeddings: list[tuple[str, float, np.ndarray]] | None = None,
    context_weight: float = 0.4,
    identity_weight: float = 0.6,
) -> float:
    """S_i = weighted cosine similarity to context and identity/goals.

    Args:
        memory_embedding: The memory's embedding vector.
        attention_embedding: Current attention focus embedding (from §3.10).
        layer_embeddings: (text, weight, vector) tuples from layers.get_all_layer_embeddings().
        context_weight: Weight for attention context similarity.
        identity_weight: Weight for L0/L1 identity/goal similarity.
    """
    score = 0.0

    # Context relevance (attention embedding)
    if attention_embedding is not None:
        ctx_sim = cosine_similarity(memory_embedding, attention_embedding)
        score += context_weight * ctx_sim

    # Identity/goal relevance (L0/L1 embeddings)
    if layer_embeddings:
        weighted_sim = 0.0
        total_weight = 0.0
        for _text, weight, vec in layer_embeddings:
            sim = cosine_similarity(memory_embedding, vec)
            weighted_sim += weight * sim
            total_weight += weight
        if total_weight > 0:
            score += identity_weight * (weighted_sim / total_weight)

    return min(score, 1.0)


def partial_matching_penalty(
    memory_metadata: dict[str, Any],
    query_metadata: dict[str, Any],
    p: float = DEFAULT_MISMATCH_P,
) -> float:
    """P_i = P * sum(mismatch_k) — penalize metadata mismatches.

    Checks type, source_tag, and tags overlap.
    """
    mismatches = 0.0

    # Type mismatch
    if query_metadata.get("type") and memory_metadata.get("type"):
        if query_metadata["type"] != memory_metadata["type"]:
            mismatches += 0.3

    # Source mismatch
    if query_metadata.get("source_tag") and memory_metadata.get("source_tag"):
        if query_metadata["source_tag"] != memory_metadata["source_tag"]:
            mismatches += 0.2

    # Tags overlap (less overlap = more penalty)
    q_tags = set(query_metadata.get("tags", []))
    m_tags = set(memory_metadata.get("tags", []))
    if q_tags and m_tags:
        overlap = len(q_tags & m_tags) / max(len(q_tags | m_tags), 1)
        mismatches += 0.5 * (1.0 - overlap)

    return p * mismatches


def logistic_noise(s: float = DEFAULT_NOISE_S) -> float:
    """ε_i — ACT-R standard logistic noise.

    Provides stochastic floor so the gate can surprise itself.
    """
    p = random.random()
    p = max(0.001, min(0.999, p))
    return s * math.log(p / (1.0 - p))


def compute_activation(
    memory_embedding: np.ndarray,
    access_timestamps: list[datetime],
    memory_metadata: dict[str, Any] | None = None,
    query_metadata: dict[str, Any] | None = None,
    attention_embedding: np.ndarray | None = None,
    layer_embeddings: list[tuple[str, float, np.ndarray]] | None = None,
    d: float = DEFAULT_DECAY_D,
    s: float = DEFAULT_NOISE_S,
    p: float = DEFAULT_MISMATCH_P,
    tau: float = DEFAULT_THRESHOLD_TAU,
) -> tuple[float, dict[str, float]]:
    """Compute full ACT-R activation: A_i = B_i + S_i + P_i + ε_i

    Returns (activation_score, component_breakdown).
    """
    b_i = base_level_activation(access_timestamps, d=d)
    s_i = spreading_activation(
        memory_embedding, attention_embedding, layer_embeddings,
    )
    p_i = partial_matching_penalty(
        memory_metadata or {}, query_metadata or {}, p=p,
    )
    eps_i = logistic_noise(s)

    a_i = b_i + s_i + p_i + eps_i

    components = {
        "base_level": b_i,
        "spreading": s_i,
        "partial_match": p_i,
        "noise": eps_i,
        "total": a_i,
        "threshold": tau,
        "above_threshold": a_i > tau,
    }

    return a_i, components
