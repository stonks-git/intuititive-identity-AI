"""Context Assembly — dynamic injection + stochastic identity injection.

Two parallel injection tracks:
  Track 1 — Situational: memories compete on injection_score = observed_weight * relevance
  Track 2 — Stochastic identity: top-N memories by depth_weight, roll observe(), inject fully or skip

Identity is a rendered view, not stored artifact. What surfaces depends on stochastic roll AND situation.
Immutable safety memories always injected.
"""

import logging
from datetime import datetime, timezone

import numpy as np

from .stochastic import StochasticWeight
from .tokens import count_tokens
from .activation import cosine_similarity

logger = logging.getLogger("agent.context")

# Token budget allocation
BUDGET_IMMUTABLE_SAFETY = 100
BUDGET_IDENTITY_AVG = 1500
BUDGET_IDENTITY_MAX = 3000
BUDGET_SITUATIONAL = 2000
BUDGET_COGNITIVE_STATE = 200
BUDGET_OUTPUT_BUFFER = 4000
# Remainder goes to conversation FIFO

# Identity injection threshold
IDENTITY_THRESHOLD = 0.6

# Top-N identity memories to consider
IDENTITY_TOP_N = 20


async def assemble_context(
    memory_store,
    layers,
    attention_embedding: np.ndarray | None,
    previous_attention_embedding: np.ndarray | None,
    cognitive_state_report: str,
    conversation: list[dict],
    total_budget: int = 131072,
    attention_text: str = "",
) -> dict:
    """Assemble the full context for an LLM call.

    Returns dict with keys: system_parts, identity_parts, situational_parts,
    cognitive_state, conversation_budget, conversation.
    """
    used_tokens = 0
    parts = {
        "immutable": [],
        "identity": [],
        "situational": [],
        "cognitive_state": cognitive_state_report,
        "conversation": conversation,
    }

    # ── Track 0: Immutable safety (always injected) ──────────────────
    immutable_memories = await _get_immutable_memories(memory_store)
    for mem in immutable_memories:
        parts["immutable"].append(mem["content"])
        used_tokens += count_tokens(mem["content"])

    # ── Track 2: Stochastic identity injection ───────────────────────
    identity_tokens = 0
    identity_memories = await _get_top_identity_memories(memory_store, IDENTITY_TOP_N)

    for mem in identity_memories:
        if identity_tokens >= BUDGET_IDENTITY_MAX:
            break
        weight = StochasticWeight(
            alpha=mem.get("depth_weight_alpha", 1.0),
            beta=mem.get("depth_weight_beta", 4.0),
        )
        observed = weight.observe()

        if observed > IDENTITY_THRESHOLD or mem.get("immutable", False):
            content = mem["content"]
            tokens = count_tokens(content)
            if identity_tokens + tokens <= BUDGET_IDENTITY_MAX:
                parts["identity"].append(content)
                identity_tokens += tokens

    used_tokens += identity_tokens

    # ── Cognitive state report ───────────────────────────────────────
    used_tokens += count_tokens(cognitive_state_report)

    # ── Track 1: Situational injection (competition-based) ───────────
    situational_budget = min(BUDGET_SITUATIONAL, total_budget - used_tokens - BUDGET_OUTPUT_BUFFER)
    if situational_budget > 0 and attention_embedding is not None:
        situational = await _get_situational_memories(
            memory_store, attention_embedding, situational_budget,
            query_text=attention_text,
        )
        for mem in situational:
            parts["situational"].append(
                mem.get("compressed") or mem["content"]
            )
    used_tokens += sum(count_tokens(s) for s in parts["situational"])

    # ── Context inertia ──────────────────────────────────────────────
    context_shift = 1.0
    if (attention_embedding is not None and previous_attention_embedding is not None):
        context_shift = 1.0 - cosine_similarity(attention_embedding, previous_attention_embedding)
    # Big shift flushes old context; small shift retains
    inertia = 0.05 if context_shift > 0.7 else 0.3

    # ── Conversation budget ──────────────────────────────────────────
    conversation_budget = total_budget - used_tokens - BUDGET_OUTPUT_BUFFER

    return {
        "parts": parts,
        "used_tokens": used_tokens,
        "conversation_budget": max(0, conversation_budget),
        "identity_token_count": identity_tokens,
        "context_shift": context_shift,
        "inertia": inertia,
    }


def render_system_prompt(context: dict) -> str:
    """Render assembled context into a system prompt string."""
    sections = []

    # Immutable safety
    if context["parts"]["immutable"]:
        sections.append("[SAFETY BOUNDARIES]")
        sections.extend(context["parts"]["immutable"])
        sections.append("")

    # Identity
    if context["parts"]["identity"]:
        sections.append("[IDENTITY — this cycle's active beliefs/values]")
        sections.extend(context["parts"]["identity"])
        sections.append("")

    # Situational memories
    if context["parts"]["situational"]:
        sections.append("[RELEVANT MEMORIES]")
        sections.extend(context["parts"]["situational"])
        sections.append("")

    # Cognitive state
    sections.append(context["parts"]["cognitive_state"])

    return "\n".join(sections)


def adaptive_fifo_prune(
    conversation: list[dict],
    budget: int,
    intensity: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Intensity-adaptive context pruning.

    Returns (kept_messages, pruned_messages).
    Pruned messages should be sent through exit gate before discard.

    intensity > 0.7 → keep ~90% of max (deep focus)
    0.3-0.7 → normal
    < 0.3 → keep ~30-40% (relaxed, cheap)
    """
    if not conversation:
        return [], []

    from .tokens import count_messages_tokens

    # Adaptive sizing based on intensity
    if intensity > 0.7:
        effective_budget = int(budget * 0.9)
    elif intensity < 0.3:
        effective_budget = int(budget * 0.35)
    else:
        effective_budget = budget

    total = count_messages_tokens(conversation)
    if total <= effective_budget:
        return conversation, []

    # Prune from the front (oldest first), preserving system messages
    kept = []
    pruned = []
    running_tokens = 0

    # Work backwards from most recent to keep recent context
    for msg in reversed(conversation):
        msg_tokens = count_tokens(msg.get("content", "")) + 4
        if running_tokens + msg_tokens <= effective_budget:
            kept.insert(0, msg)
            running_tokens += msg_tokens
        else:
            pruned.insert(0, msg)

    return kept, pruned


async def _get_immutable_memories(memory_store) -> list[dict]:
    """Fetch immutable=true memories."""
    rows = await memory_store.pool.fetch(
        "SELECT id, content FROM memories WHERE immutable = true"
    )
    return [dict(r) for r in rows]


async def _get_top_identity_memories(memory_store, top_n: int) -> list[dict]:
    """Fetch top-N memories by depth_weight center (identity candidates)."""
    rows = await memory_store.pool.fetch(
        """
        SELECT id, content, depth_weight_alpha, depth_weight_beta, immutable
        FROM memories
        WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.3
        ORDER BY depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) DESC
        LIMIT $1
        """,
        top_n,
    )
    return [dict(r) for r in rows]


async def _get_situational_memories(
    memory_store, attention_embedding: np.ndarray, budget: int,
    query_text: str = "",
) -> list[dict]:
    """Retrieve situationally relevant memories within token budget."""
    if not query_text:
        return []  # Can't search without query text
    candidates = await memory_store.search_hybrid(
        query=query_text,
        top_k=10,
    )

    # Fill within budget
    result = []
    used = 0
    for mem in candidates:
        content = mem.get("compressed") or mem.get("content", "")
        tokens = count_tokens(content)
        if used + tokens > budget:
            break
        result.append(mem)
        used += tokens

    return result
