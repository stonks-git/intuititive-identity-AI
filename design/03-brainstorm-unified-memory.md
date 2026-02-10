# Brainstorm Session 7: Unified Memory & Stochastic Cognition

**Date:** 2026-02-09, Session 7
**Status:** Brainstorm output — to be integrated into plan v3
**Supersedes:** The discrete three-layer injection model from DOCUMENTATION.md and new_plan_v2.md
**Does NOT supersede:** The layer promotion/consolidation mechanics, safety ceilings, or dual-process reasoning

---

## 1. Core Architectural Shift: From Layers to Unified Weighted Memory

### What changed

The original design had three discrete layers with fixed injection rules:
- **Layer 0 (Identity):** Always in context, every call
- **Layer 1 (Goals):** Always in context, every call
- **Layer 2 (Data):** Retrieved via RAG on demand

This is replaced by a **unified memory store** where every memory has a continuous `depth_weight` (0.0-1.0). There are no categorical layers for injection purposes. What gets injected into context is determined dynamically by a combination of weight and situational relevance.

### What stays the same

- The **promotion pathway** still exists: experiences crystallize into goals, goals crystallize into identity. The `depth_weight` reflects this — a deeply reinforced value will have weight ~0.9, a fresh episodic memory ~0.2.
- **Safety boundaries** remain as `immutable: true` flagged memories. These are the ONLY categorical distinction. Everything else is a continuous spectrum.
- **Consolidation** still merges, promotes, decays, and tunes. It now operates on weights rather than layer assignments.
- The **memory schema** (Postgres + pgvector) is unchanged. `depth_weight` replaces the old layer-based injection logic, not the storage.

### Why the shift

Fixed injection by layer is rigid. A human doesn't walk around with their full identity loaded at all times. You contextually activate the relevant slice of self based on the situation. A work meeting primes "professional you." A friend calling primes "casual you." Sometimes the wrong context gets primed and weirdness ensues — that's a feature, not a bug. The architecture should model this.

---

## 2. Stochastic Weights: The Quantum Model

### Principle

No weight is a fixed number. A `depth_weight` of 0.7 is a probability distribution centered at 0.7 that **collapses to a specific value each time it's observed**. This prevents deterministic, rigid behavior and ensures permanent exploration.

### Mechanism

```python
class StochasticWeight:
    def __init__(self, center: float, reinforcement_count: int):
        self.center = center
        self.reinforcement_count = reinforcement_count

    @property
    def sigma(self) -> float:
        """Uncertainty decreases with reinforcement -- memory crystallizes."""
        BASE_NOISE = 0.08
        NOISE_FLOOR = 0.01  # never fully deterministic
        return max(BASE_NOISE / sqrt(self.reinforcement_count + 1), NOISE_FLOOR)

    def observe(self) -> float:
        """Collapse the wave function -- sample a concrete value."""
        raw = random.gauss(self.center, self.sigma)
        return max(0.0, min(1.0, raw))  # clamp to [0, 1]
```

### Behavior

| Reinforcement count | Center | Sigma | Sample range (typical) |
|---------------------|--------|-------|------------------------|
| 1 | 0.7 | 0.057 | 0.59 - 0.81 |
| 5 | 0.7 | 0.033 | 0.64 - 0.76 |
| 25 | 0.7 | 0.016 | 0.67 - 0.73 |
| 100 | 0.7 | 0.01 | 0.68 - 0.72 |

**New memories are volatile** — their effective weight fluctuates widely. **Deeply reinforced memories are stable** but never perfectly rigid (the noise floor of 0.01 ensures permanent slight variability).

### Consequences

- A new insight at weight 0.5 might momentarily observe at 0.6 and surface above an established memory — **creative disruption**.
- A deeply held belief at 0.9 might observe at 0.88 while a challenger at 0.85 observes at 0.87 — **occasional perspective shift even on "settled" beliefs**.
- The system can never be fully stuck because noise ensures exploration continues at all levels.

---

## 3. Dynamic Context Injection: No Fixed Tiers

### The old model

```
weight >= 0.85 -> ALWAYS in context
weight 0.6-0.85 -> ALWAYS but compressed
weight 0.3-0.6 -> RAG-eligible
weight < 0.3 -> dormant
```

### The new model

**No fixed tiers. No "always in context." The token budget is the only hard constraint.**

At context assembly time:

1. Determine current **attention focus** (user message, DMN self-prompt, task output, gut signal — any input)
2. For every memory in the store, compute an **injection score**:
   ```
   injection_score = observed_weight * relevance(memory, current_attention)
   ```
   Both factors are stochastic (weight via quantum observation, relevance via the hybrid function described in Section 4).
3. Sort all memories by injection_score descending.
4. Fill context top-down until token budget is exhausted.
5. Higher-scoring entries get full text; lower-scoring entries get compressed form.

### The only exception: immutable safety boundaries

Memories flagged `immutable: true` (bootstrap safety constraints like "never give financial advice") are ALWAYS injected regardless of relevance score. This is a small set (~5-10 items, ~100 tokens). Everything else competes for context space on merit.

### Identity as emergent view, not stored artifact

There is no stored "I am" block. Identity is **rendered at context assembly time** from whatever high-weight memories won the competition for context space. Different situation = different memories win = different personality surfaces.

This means identity is always up-to-date. When a weight changes, the next render reflects it instantly. No stale cached self-concept.

### Pre-computed compressed summaries

For the compressed form (lower-scoring memories that still make it into context), each memory stores a pre-computed one-line summary alongside its full text. This summary is generated at gate time (piggybacking on the existing LLM call) and refreshed only when the memory mutates. No runtime LLM cost for compression.

```python
memory = {
    "full": "On 2024-03-15, user explained they hate verbose responses...",
    "compressed": "User strongly values brevity -- answer concisely",
    "depth_weight": StochasticWeight(center=0.72, reinforcement_count=8)
}
```

### Context inertia and context switch detection

Previous context bleeds into current activation — this creates continuity but can also cause inappropriate lingering (the "weirdness" of failing to reload appropriate personality when the situation changes).

```python
context_shift = 1.0 - embedding_similarity(current_attention, previous_attention)
if context_shift > 0.7:  # big topic/situation change
    inertia_coefficient = 0.05  # almost flush old context
else:
    inertia_coefficient = 0.3   # normal bleed-through
```

The inertia factor means a component of the previous context biases what memories are currently activated. Abrupt context switches dampen this inertia, allowing faster personality reloading.

---

## 4. The Hybrid Relevance Function

### Why embedding similarity alone is insufficient

Cosine similarity between embedding vectors measures **topical overlap**. It's good at: same-topic detection, synonym handling, domain clustering. It's bad at: emotional/tonal connections, negation (loves/hates X both score high), and deep psychological associations ("father was strict" connecting to "I flinch at authority").

The deepest memory connections are exactly the ones embeddings miss. The system needs a hybrid approach.

### Five components

Each component scores 0.0-1.0:

#### Component 1: Semantic Similarity (topical relevance)

```python
def semantic_relevance(memory_embedding, attention_embedding) -> float:
    return cosine_similarity(memory_embedding, attention_embedding)
```

Cosine similarity between the memory's embedding vector and the current attention focus embedding. Standard dense retrieval signal.

#### Component 2: Co-Access Score (Hebbian learned association)

```python
def coactivation_relevance(memory, attention_context, co_access_matrix) -> float:
    """How often has this memory activated alongside
    other memories that are currently active?"""
    active_memory_ids = attention_context.recently_activated_ids
    if not active_memory_ids:
        return 0.0
    scores = [co_access_matrix.get(frozenset({memory.id, aid}), 0.0)
              for aid in active_memory_ids]
    return max(scores)  # strongest association wins
```

This is the mechanism that learns non-obvious connections. "Father was strict" connects to authority situations not because embeddings see it, but because they've been co-activated enough times that the association is recorded. **Connections learned from experience, not computed from content.**

Backed by a `memory_co_access` join table in Postgres. Updated every time memories are co-retrieved (already planned in new_plan_v2.md Section 17e, item 8).

#### Component 3: Pure Noise (creative exploration)

```python
def noise_relevance() -> float:
    return random.random()  # uniform 0-1
```

Random value. Most of the time it surfaces garbage. Occasionally it surfaces a memory that creates a novel connection nobody would have predicted. The blend weight on this component (via Dirichlet, see below) controls "creativity temperature."

#### Component 4: Emotional/Valence Alignment (mood-congruent recall)

```python
def emotional_relevance(memory, current_emotional_state) -> float:
    """Mood-congruent recall: memories matching current emotional
    tone surface more easily."""
    if not memory.valence or not current_emotional_state:
        return 0.5  # neutral
    return 1.0 - abs(memory.valence - current_emotional_state.valence)
```

When the agent is in a cautious state, cautious memories surface preferentially. This creates realistic mood dynamics and feedback loops (anxiety -> anxious memories -> more anxiety) that the metacognitive layer should monitor.

#### Component 5: Temporal Recency (priming effect)

```python
def recency_relevance(memory, current_time) -> float:
    hours_since = (current_time - memory.last_accessed).total_seconds() / 3600
    return exp(-hours_since / RECENCY_HALF_LIFE)
```

Slight boost for recently accessed memories. Exponential decay. Models the priming effect — recent activations leave a trace.

### Stochastic blend via Dirichlet distribution

The five components are NOT blended with fixed ratios. Each iteration, the blend weights are drawn from a **Dirichlet distribution**:

```python
class HybridRelevance:
    def __init__(self):
        # Dirichlet concentration parameters -- learned over time
        self.alpha = {
            'semantic':      8.0,   # dominant, stable
            'coactivation':  5.0,   # important, somewhat stable
            'noise':         0.5,   # small but high-variance
            'emotional':     3.0,   # moderate
            'recency':       2.0,   # moderate
        }

    def compute(self, memory, attention_context) -> float:
        scores = {
            'semantic':     semantic_relevance(memory.embedding, attention_context.embedding),
            'coactivation': coactivation_relevance(memory, attention_context, co_access_matrix),
            'noise':        noise_relevance(),
            'emotional':    emotional_relevance(memory, attention_context.emotional_state),
            'recency':      recency_relevance(memory, now()),
        }

        # Stochastic blend weights from Dirichlet
        alpha_values = [self.alpha[k] for k in scores.keys()]
        blend_weights = np.random.dirichlet(alpha_values)

        return sum(w * s for w, s in zip(blend_weights, list(scores.values())))
```

#### Dirichlet behavior

The Dirichlet distribution produces a vector of weights that sum to 1.0, with randomness controlled by concentration parameters:

```
With alpha = [8, 5, 0.5, 3, 2]:
Typical draw:     [0.42, 0.27, 0.03, 0.16, 0.12]  -- semantic dominates, noise tiny
Occasional draw:  [0.35, 0.20, 0.18, 0.15, 0.12]  -- noise surges to 18%
Rare draw:        [0.25, 0.15, 0.31, 0.17, 0.12]  -- noise DOMINATES (pure exploration)
```

The noise component has alpha=0.5 (low concentration) so its contribution is highly variable. Most iterations near-zero. Occasionally it spikes and dominates — that's when creative leaps happen.

#### Meta-learning: the Dirichlet parameters evolve

```python
def update_blend_weights(self, outcome_quality, blend_used, lr=0.01):
    for component, weight_used in blend_used.items():
        if outcome_quality > 0.7:
            self.alpha[component] += lr * weight_used
        elif outcome_quality < 0.3:
            self.alpha[component] -= lr * weight_used * 0.5
```

The system learns how to blend its own relevance components. If noise-driven retrievals produce good outcomes, noise alpha increases (system becomes more exploratory). If they produce garbage, it decreases (system becomes more conservative). This is meta-learning — the system learning how to optimally retrieve.

### Convergence with permanent exploration

Over many iterations, the alpha parameters converge toward the optimal blend for this specific agent. But they never fully converge — the stochastic sampling from Dirichlet ensures permanent variability. The system approaches an optimum but keeps probing around it.

---

## 5. Full Retrieval Pipeline

### Assembly per cognitive cycle

```python
def retrieve_for_context(all_memories, attention_context, token_budget):
    hybrid = HybridRelevance()

    scored = []
    for memory in all_memories:
        # 1. Stochastic weight observation (quantum collapse)
        observed_weight = memory.depth_weight.observe()

        # 2. Stochastic relevance (hybrid blend with noise)
        relevance = hybrid.compute(memory, attention_context)

        # 3. Final injection score -- both factors are noisy
        injection_score = observed_weight * relevance
        scored.append((memory, injection_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 4. Fill context within token budget
    context_memories = []
    tokens_used = 0
    for memory, score in scored:
        form = memory.compressed if score < 0.6 else memory.full
        token_cost = count_tokens(form)
        if tokens_used + token_cost > token_budget:
            break
        context_memories.append((memory, form))
        tokens_used += token_cost

    # 5. Record co-access for Hebbian learning
    update_co_access_matrix([m for m, _ in context_memories])

    return context_memories
```

### Every call is different

Even for identical inputs, the stochastic elements (weight observation, Dirichlet blend, noise component) produce slightly different contexts:

- Run 1: surfaces memories A, B, C, D, E
- Run 2: surfaces memories A, B, C, F, D (F snuck in via noise)
- Run 3: surfaces memories A, B, G, C, D (G was a total surprise)

Run 3's inclusion of G might produce an insight that would NEVER happen in a deterministic system. If that insight is valuable, G's co-access score with the other memories increases, making it more likely to surface in similar contexts in the future. **Creative accidents get reinforced into stable associations.**

---

## 6. Self-Discussion Architecture

### The problem with user-centric design

The original architecture assumed the primary input is always a user message. But the agent might be:

- Talking to itself during DMN idle
- Processing its own memories during consolidation
- Reflecting on a gut feeling with no external input
- Running a self-prompt from an idle heartbeat

### The solution: attention-agnostic processing

All input sources feed the same cognitive loop. The query for relevance computation is the **current attention focus**, which can come from any source:

| Input source | Example attention focus |
|---|---|
| User message | "Tell me about Hetzner pricing" |
| DMN self-prompt | "I just remembered X, this connects to Y" |
| Consolidation insight | "I notice pattern Z forming across 12 memories" |
| Gut signal | "Something about current state feels uneasy" |
| Scheduled task | "It's time to check my cost expenditure" |

All five feed into the SAME cognitive loop with the SAME hybrid relevance function. The architecture doesn't distinguish "talking to user" vs "talking to self" in its processing pipeline. Only the **output routing** varies (reply to user vs log insight vs trigger action).

### Correction retrieval is attention-keyed, not user-keyed

```python
# Old (assumes user)
corrections = search(query=user_message, type="correction")

# New (attention-agnostic)
corrections = search(query=current_attention_focus, type="correction")
```

The reflection bank (System 2 corrections stored for future retrieval) works identically whether the agent is responding to a user or reflecting on its own thoughts.

### The DMN as self-generated input

The DMN/idle loop is not a separate processing pipeline. It generates inputs that feed into the main cognitive loop. The only difference from user conversation is:
1. The input is self-generated (surfaced memory + goal/value connection)
2. The output routing is internal (log insight, store reflection, or trigger action)
3. The processing is identical

### Strange loop via natural recursion

When the agent processes a DMN-surfaced memory, the processing itself generates new memories (the reflection). Those memories go through the gate. If they're weighted high enough, they influence future surfacing. The loop:

```
memory surfaces -> agent reflects -> reflection stored ->
reflection influences future surfacing -> loop
```

No special mechanism needed. The architecture naturally supports this because all inputs (including self-generated ones) go through the same pipeline.

---

## 7. Metacognitive Subsystem: Separate Context Windows

### The problem

Contradiction checks, confidence evaluations, boundary checks, and FOK lookups are metacognitive operations. If their prompts/responses go into the main context window, they pollute the agent's working memory with internal chatter.

### The solution: isolated meta context

```
Main context window:  The agent's "conscious" working memory
                      User messages, agent responses, RAG results, identity rendering

Meta context window:  The agent's "nervous system" -- signals, not thoughts
                      Contradiction checks
                      Confidence evaluations
                      Boundary checks
                      FOK lookups

                      These produce SIGNALS (bool/float) that influence
                      the main context, but their prompts/responses
                      never enter the main context.
```

### Implementation

Separate API calls with minimal context. Example contradiction check:

```python
async def check_contradiction(memory_a: str, memory_b: str) -> bool:
    """Isolated micro-call -- separate context, discarded after."""
    response = await llm_call(
        model="gemini-flash-lite",
        messages=[{
            "role": "user",
            "content": f"Does A contradict B? Answer YES or NO only.\nA: {memory_a}\nB: {memory_b}"
        }],
        max_tokens=3
    )
    return "YES" in response.upper()
```

No history, no identity injection, no RAG. Pure signal extraction. Cost: ~$0.0001 per check. The meta subsystem **reads from** the same memory store but has its own ephemeral context that's discarded after producing its signal.

### Contradiction detection: three-layer mechanism (confirmed)

1. **Negation heuristic** (~0ms): regex/keyword check for obvious negation patterns. Catches "X is good" vs "X is bad."
2. **Embedding opposition** (~5ms): high cosine similarity + semantic opposition detection. Catches subtle contradictions.
3. **LLM micro-call** (~100ms, separate context): Only fires when layers 1 and 2 are inconclusive. The most expensive check, used sparingly.

---

## 8. What This Changes in the Implementation Plan

### Affected components from new_plan_v2.md

| Component | Change |
|---|---|
| **2.3 Embed L0/L1** | Still needed. L0/L1 embeddings are used by the hybrid relevance function (semantic similarity to identity/goal vectors is one of many signals). |
| **2.6 ACT-R Activation** | Partially subsumed. The base-level learning (access decay) feeds into `depth_weight`. Spreading activation is now one component of the hybrid relevance function. Noise is handled by the stochastic weight model. Partial matching may be dropped or folded in. |
| **2.7 Exit Gate** | The 3x3 matrix still determines what ACTION to take (persist/reinforce/drop/flag). But the scoring that determines relevance and novelty now uses the hybrid relevance function. |
| **2.9 Wire MemoryStore** | Context assembly now uses the full retrieval pipeline from Section 5 instead of simple `search_hybrid()` with fixed budget allocation per layer. |
| **3.1 Adaptive FIFO** | Identity re-injection after pruning is replaced by: re-run the retrieval pipeline, which will naturally surface high-weight identity-relevant memories if the current attention warrants it. |
| **4.6 DMN/Idle Loop** | Uses the same cognitive loop with self-generated input (Section 6). No separate processing pipeline. |
| **5.1 Two-Centroid Gut** | Unchanged. The gut feeling (delta between subconscious and attention centroids) feeds into the emotional component of the hybrid relevance function and into the gate. |
| **Section 10 (Identity Injection)** | The two-tier injection strategy (hash always, full on triggers) is replaced by dynamic competition. Immutable safety memories are the only guaranteed injection. |

### New components needed

| Component | Description |
|---|---|
| `StochasticWeight` class | Gaussian sampling with reinforcement-dependent sigma. Used everywhere a weight is read. |
| `HybridRelevance` class | Five-component relevance function with Dirichlet blend. |
| `memory_co_access` table | Hebbian co-activation tracking. Schema: `(memory_id_a, memory_id_b, co_access_count, last_co_accessed)`. |
| Compressed summary generation | Piggyback on gate LLM call to generate one-line summaries stored alongside full text. |
| Context inertia tracking | Track previous attention embedding for inertia calculation and context switch detection. |
| Meta context isolation | Separate LLM call wrapper for metacognitive checks (no history, no identity, discard after signal). |

---

## 9. Open Questions Remaining

1. **Performance of scoring all memories per cycle.** If there are 100k memories, computing hybrid relevance for each is expensive. Need a pre-filter (e.g., pgvector top-500 by embedding similarity first, then full hybrid scoring on that subset).

2. **Dirichlet alpha initialization.** Starting values [8, 5, 0.5, 3, 2] are educated guesses. Should these be evolved from scratch (all equal) or seeded with these hints?

3. **Co-access matrix scaling.** With N memories, the matrix is O(N^2) in the worst case. Need pruning strategy (e.g., only track co-access above threshold, decay old associations).

4. **Emotional state bootstrapping.** The emotional/valence component requires a current emotional state. Where does this come from before the gut feeling system (5.1) is implemented? Neutral default (0.5) until then?

5. **Outcome quality signal for meta-learning.** The blend weight updater needs an `outcome_quality` signal. What constitutes a "good outcome" for a retrieval? User engagement? Memory being accessed again? Agent self-rating?

---

## 10. Summary of Design Decisions

| Decision | Chosen | Rationale |
|---|---|---|
| Discrete layers vs unified weights | Unified weights | More organic, mirrors human contextual personality activation |
| Fixed injection tiers vs dynamic competition | Dynamic competition | Identity shouldn't always reload; situational relevance matters |
| Deterministic weights vs stochastic | Stochastic (Gaussian with reinforcement-dependent sigma) | Prevents rigid behavior, enables creative disruption, models uncertainty |
| Fixed relevance blend vs stochastic | Stochastic (Dirichlet distribution) | Allows exploration/exploitation balance to emerge and evolve |
| "I am" as stored data vs rendered view | Rendered view | Always up-to-date, no stale self-concept, identity IS the weight distribution |
| User-centric vs attention-agnostic processing | Attention-agnostic | Agent must think to itself as naturally as it talks to users |
| Metacognitive checks in main context vs isolated | Isolated (separate context windows) | Prevents internal chatter from polluting working memory |
| Pure embedding similarity vs hybrid relevance | Hybrid (5 components) | Embeddings miss deep associations; Hebbian co-access and noise fill the gaps |
