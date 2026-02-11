"""Hybrid Relevance — 5-component scoring with Dirichlet-sampled blend weights.

Components (each 0-1):
  1. Semantic similarity: cosine(memory_embedding, attention_embedding)
  2. Co-access (Hebbian): max co-access score with currently-active memories
  3. Pure noise: uniform random 0-1 (creative exploration)
  4. Emotional/valence alignment: neutral (0.5) until gut feeling (§5.1)
  5. Temporal recency: exponential decay from last_accessed

Dirichlet stochastic blend: alpha params sampled each cycle.
Cold start: semantic-heavy. Mature: balanced.
Meta-learning: consolidation evolves alphas based on outcome quality.
"""

import logging
import random
from datetime import datetime, timezone

import numpy as np

from .activation import cosine_similarity

logger = logging.getLogger("agent.relevance")

# Cold-start Dirichlet alphas (semantic-heavy, minimal co-access/emotional)
COLD_START_ALPHA = {
    "semantic": 12.0,
    "coactivation": 1.0,
    "noise": 0.5,
    "emotional": 0.5,
    "recency": 3.0,
}

# Target Dirichlet alphas (mature system)
TARGET_ALPHA = {
    "semantic": 8.0,
    "coactivation": 5.0,
    "noise": 0.5,
    "emotional": 3.0,
    "recency": 2.0,
}

# Recency decay: 7-day half-life
RECENCY_HALF_LIFE_SECONDS = 7 * 24 * 3600


def compute_semantic_similarity(
    memory_embedding: np.ndarray,
    attention_embedding: np.ndarray | None,
) -> float:
    """Component 1: cosine similarity to current attention focus."""
    if attention_embedding is None:
        return 0.0
    return max(0.0, cosine_similarity(memory_embedding, attention_embedding))


def compute_coactivation(
    memory_id: str,
    active_memory_ids: list[str],
    co_access_scores: dict[tuple[str, str], float],
) -> float:
    """Component 2: max co-access score with currently-active memories."""
    if not active_memory_ids or not co_access_scores:
        return 0.0
    max_score = 0.0
    for active_id in active_memory_ids:
        key = tuple(sorted([memory_id, active_id]))
        score = co_access_scores.get(key, 0.0)
        max_score = max(max_score, score)
    return min(1.0, max_score)


def compute_noise() -> float:
    """Component 3: pure uniform random. Creative exploration."""
    return random.random()


def compute_emotional_alignment(gut_delta: float | None = None) -> float:
    """Component 4: mood-congruent recall. Neutral until gut feeling (§5.1)."""
    if gut_delta is None:
        return 0.5  # neutral
    return max(0.0, min(1.0, 0.5 + gut_delta))


def compute_recency(last_accessed: datetime | None) -> float:
    """Component 5: exponential decay from last_accessed. 7-day half-life."""
    if last_accessed is None:
        return 0.0
    now = datetime.now(timezone.utc)
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)
    age_seconds = max(0, (now - last_accessed).total_seconds())
    return float(np.exp(-0.693 * age_seconds / RECENCY_HALF_LIFE_SECONDS))


def sample_blend_weights(
    memory_count: int = 0,
    alpha_override: dict[str, float] | None = None,
) -> dict[str, float]:
    """Sample Dirichlet blend weights for this cycle.

    Cold start (< 100 memories): semantic-heavy.
    Mature (> 100): balanced target alphas.
    """
    if alpha_override:
        alpha = alpha_override
    elif memory_count < 100:
        alpha = COLD_START_ALPHA
    else:
        # Linear interpolation between cold-start and target
        t = min(1.0, (memory_count - 100) / 900)  # saturates at 1000
        alpha = {
            k: COLD_START_ALPHA[k] + t * (TARGET_ALPHA[k] - COLD_START_ALPHA[k])
            for k in COLD_START_ALPHA
        }

    keys = list(alpha.keys())
    weights = np.random.dirichlet([alpha[k] for k in keys])
    return dict(zip(keys, weights))


def compute_hybrid_relevance(
    memory_embedding: np.ndarray,
    memory_id: str,
    last_accessed: datetime | None,
    attention_embedding: np.ndarray | None = None,
    active_memory_ids: list[str] | None = None,
    co_access_scores: dict | None = None,
    gut_delta: float | None = None,
    blend_weights: dict[str, float] | None = None,
    memory_count: int = 0,
) -> tuple[float, dict]:
    """Compute the 5-component hybrid relevance score.

    Returns (score, component_breakdown).
    """
    if blend_weights is None:
        blend_weights = sample_blend_weights(memory_count)

    components = {
        "semantic": compute_semantic_similarity(memory_embedding, attention_embedding),
        "coactivation": compute_coactivation(
            memory_id, active_memory_ids or [], co_access_scores or {},
        ),
        "noise": compute_noise(),
        "emotional": compute_emotional_alignment(gut_delta),
        "recency": compute_recency(last_accessed),
    }

    score = sum(blend_weights[k] * components[k] for k in components)

    breakdown = {
        "components": components,
        "blend_weights": blend_weights,
        "final_score": score,
    }

    return score, breakdown


# ── SPREADING ACTIVATION (co-access network) ──────────────────────────────


async def spread_activation(
    pool,
    seed_ids: list[str],
    hops: int = 1,
    top_k_per_hop: int = 3,
) -> dict[str, float]:
    """Spread activation through co-access network.

    1-hop default, 2-hop during DMN/introspection.
    Decay per hop: [1.0, 0.3, 0.1]
    """
    if not seed_ids:
        return {}

    decay_per_hop = [1.0, 0.3, 0.1]
    activated: dict[str, float] = {}
    current_seeds = {id_: 1.0 for id_ in seed_ids}

    for hop in range(min(hops, len(decay_per_hop))):
        if not current_seeds:
            break

        seed_list = list(current_seeds.keys())
        rows = await pool.fetch(
            """
            SELECT memory_id_a, memory_id_b, co_access_count
            FROM memory_co_access
            WHERE memory_id_a = ANY($1) OR memory_id_b = ANY($1)
            ORDER BY co_access_count DESC
            """,
            seed_list,
        )

        next_seeds: dict[str, float] = {}
        # Group by seed, take top_k partners per seed
        partners_per_seed: dict[str, list[tuple[str, int]]] = {}
        for row in rows:
            a, b, count = row["memory_id_a"], row["memory_id_b"], row["co_access_count"]
            for seed in seed_list:
                if a == seed:
                    partners_per_seed.setdefault(seed, []).append((b, count))
                elif b == seed:
                    partners_per_seed.setdefault(seed, []).append((a, count))

        for seed, partners in partners_per_seed.items():
            partners.sort(key=lambda x: x[1], reverse=True)
            for partner_id, count in partners[:top_k_per_hop]:
                # Normalize count to 0-1 (cap at 20 co-accesses)
                normalized = min(1.0, count / 20.0)
                spread_score = current_seeds[seed] * normalized * decay_per_hop[hop]
                activated[partner_id] = max(
                    activated.get(partner_id, 0), spread_score,
                )
                next_seeds[partner_id] = spread_score

        current_seeds = next_seeds

    return activated


async def update_co_access(pool, memory_ids: list[str]):
    """Record co-access for Hebbian learning. Called when memories are co-retrieved."""
    if len(memory_ids) < 2:
        return

    pairs = []
    for i in range(len(memory_ids)):
        for j in range(i + 1, min(len(memory_ids), i + 5)):  # limit pairs
            a, b = sorted([memory_ids[i], memory_ids[j]])
            pairs.append((a, b))

    for a, b in pairs:
        await pool.execute(
            """
            INSERT INTO memory_co_access (memory_id_a, memory_id_b, co_access_count, last_co_accessed)
            VALUES ($1, $2, 1, NOW())
            ON CONFLICT (memory_id_a, memory_id_b)
            DO UPDATE SET co_access_count = memory_co_access.co_access_count + 1,
                         last_co_accessed = NOW()
            """,
            a, b,
        )
