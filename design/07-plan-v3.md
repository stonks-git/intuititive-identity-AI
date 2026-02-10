# OpenClaw MoltBot — Implementation Plan v3

**Created:** 2026-02-09, Session 7-8
**Based on:** new_plan_v2.md + SESSION_HANDOFF.md (19 resolutions) + BRAINSTORM_SESSION7.md + BRAINSTORM_SESSION8.md
**Supersedes:** new_plan_v2.md (v2), new_plan.md (v1)
**Total SOTA coverage:** 100+ papers/systems from 2024-2026

---

## Changelog from v2

| # | Issue | Resolution |
|---|-------|------------|
| A | **3-layer injection model replaced** | Unified weighted memory — no discrete layers for injection. `depth_weight` replaces L0/L1/L2 injection rules. |
| B | **Fixed relevance blend replaced** | Hybrid 5-component relevance with Dirichlet stochastic blend (semantic, co-access, noise, emotional, recency). |
| C | **Deterministic weights replaced** | StochasticWeight — Gaussian sampling, reinforcement-dependent sigma, permanent noise floor. |
| D | **User-centric cognitive loop replaced** | Attention-agnostic: all input sources (user, DMN, consolidation, gut, scheduled) feed same loop. |
| E | **Periodic-only consolidation replaced** | Two-tier: constant light background + periodic deep pass. |
| F | **Metacognitive checks in main context replaced** | Isolated meta-context windows — produce signals, never pollute main context. |
| G | **"I am" stored data replaced** | Identity rendered as view of high-weight memories at context assembly time. |
| H | **Multi-threading deferred** | Single attentional thread + parallel background processes. Honest about single-focus. |
| I | **Source tagging added** | Soft metadata (source_tag + source_confidence), hard check only before external actions. |
| J | **Transparent self-talk** | Agent knows from bootstrap that thoughts are logged. Observer effect is data. |
| K | **All 19 SESSION_HANDOFF resolutions** | Integrated throughout (see §14 for mapping). |

---

## 0. Code Assessment: Build On, Don't Rewrite

Same as v2. The existing codebase is a **valid foundation**.

| File | Status | Verdict |
|------|--------|---------|
| `config.py` | Working, clean | Keep as-is |
| `llm.py` | Working, clean | Keep as-is |
| `memory.py` | Working, comprehensive | Enhance (task_type, prefixes, halfvec, hybrid search, unified weight) |
| `layers.py` | Working, minor issue | Repurpose: L0/L1 embedding cache for subconscious centroid. No longer manages injection. |
| `gate.py` | Architecture correct, syntax errors | Fix syntax, add full ACT-R + 3×3 matrix |
| `loop.py` | Architecture correct, syntax errors | Fix syntax, wire attention-agnostic cognitive loop |
| `main.py` | Architecture correct, syntax errors | Fix syntax |
| `consolidation.py` | Skeleton only | Implement fully (two-tier: constant + deep) |
| `idle.py` | Skeleton only | Becomes DMN input generator feeding main loop |

**15 syntax errors** + missing `__init__.py` must be fixed first.

---

## 1. Fix Syntax Errors + Project Hygiene (Prerequisite)

Fix all 15 heredoc-mangled syntax errors plus structural issues.

**Files:**
- `main.py` lines 61-68: 5 unquoted dict keys
- `loop.py` lines 65, 70, 163, 215-218, 240-241: 9 unquoted dict keys
- `gate.py` line 225: `chr(100)+ecision` → `"decision"`
- `gate.py` lines 335-340: merged negation_markers strings
- **Create `src/__init__.py`** (empty file, enables `python3 -m src.main`)
- **Fix runtime.yaml port** 5432 → 5433 (SESSION_HANDOFF #6)

**Verify:** `python3 -c "import src.main"` should succeed after fixes.
**Test:** Script that imports each module and prints "OK" — first verification step.

---

## 2. Tier 1: Foundation

### 2.1 Embedding Task Types + Memory Type Prefixes
**What:** Add `task_type` parameter AND semantic type prefixes to all embedding calls.
**Why:** Gemini embeddings are optimized per task. Type prefixes bake memory category into the vector (ENGRAM, MIRIX papers).
**Where:** `memory.py:embed()`
**How:**
- Add `task_type` parameter (default `RETRIEVAL_DOCUMENT`)
- Storage: `task_type="RETRIEVAL_DOCUMENT"` with `title=memory_type`
- Query: `task_type="RETRIEVAL_QUERY"`
- Novelty: `task_type="SEMANTIC_SIMILARITY"`
- Clustering (consolidation): `task_type="CLUSTERING"`
- Prefix map:
  ```python
  MEMORY_TYPE_PREFIXES = {
      "episodic":    "Personal experience memory: ",
      "semantic":    "Factual knowledge: ",
      "procedural":  "How-to instruction: ",
      "preference":  "User preference: ",
      "reflection":  "Self-reflection insight: ",
      "correction":  "Past error correction: ",
  }
  ```
- Batch embedding support (up to 100 texts per API call)
**Effort:** Low
**Test:** Embed a test string with each task_type, verify different vectors produced.

### 2.2 Halfvec Migration
**What:** Switch pgvector column from `vector(768)` to `halfvec(768)`.
**Why:** Halves memory usage — critical for 8GB RAM.
**Where:** Database schema, `memory.py`
**How:**
- `ALTER TABLE memories ALTER COLUMN embedding TYPE halfvec(768)`
- Re-embed existing test memories (with task_type + prefixes from 2.1)
- Rebuild HNSW index with `halfvec_cosine_ops`
**Effort:** Low
**Dependency:** 2.1
**Test:** Store + retrieve a memory, verify cosine similarity works with halfvec.

### 2.3 Unified Memory Schema + StochasticWeight
**What:** Add `depth_weight` and `reinforcement_count` columns. Implement StochasticWeight class. Add `access_timestamps` array for ACT-R.
**Why:** Core of the unified memory model. Replaces discrete layer injection. Stochastic observation prevents deterministic rigidity. Access timestamps required for ACT-R base-level learning (SESSION_HANDOFF #1).
**Where:** `memory.py`, new `stochastic.py`
**How:**
- Schema changes:
  ```sql
  ALTER TABLE memories ADD COLUMN depth_weight FLOAT DEFAULT 0.2;
  ALTER TABLE memories ADD COLUMN reinforcement_count INT DEFAULT 1;
  ALTER TABLE memories ADD COLUMN access_timestamps TIMESTAMPTZ[] DEFAULT '{}';
  ```
- StochasticWeight class:
  ```python
  class StochasticWeight:
      def __init__(self, center: float, reinforcement_count: int):
          self.center = center
          self.reinforcement_count = reinforcement_count

      @property
      def sigma(self) -> float:
          BASE_NOISE = 0.08
          NOISE_FLOOR = 0.01
          return max(BASE_NOISE / sqrt(self.reinforcement_count + 1), NOISE_FLOOR)

      def observe(self) -> float:
          raw = random.gauss(self.center, self.sigma)
          return max(0.0, min(1.0, raw))
  ```
- Immutable safety memories: `immutable BOOLEAN DEFAULT FALSE` column. Bootstrap boundaries start at weight 1.0 with `immutable=true`.
- Source tagging: `source_tag TEXT`, `source_confidence FLOAT DEFAULT 1.0`
**Effort:** Medium
**Dependency:** None
**Test:** Create StochasticWeight(0.7, 1), observe 100 times, verify range ~0.5-0.9. Create with reinforcement_count=100, verify range ~0.68-0.72.

### 2.4 Embed Layer 0 / Layer 1 (for Subconscious Centroid)
**What:** Build embedding + caching infrastructure for identity values and goals.
**Why:** Required by subconscious centroid (5.1), hybrid relevance semantic component, and ACT-R spreading activation (2.7). L0/L1 are still JSON files — they just no longer control injection.
**Where:** `layers.py`, cache mechanism
**How:**
- Embed text of each value/belief and each goal as 768-dim vector
- Cache on disk (re-embed only when text changes)
- Use `task_type="RETRIEVAL_DOCUMENT"` with appropriate title
- Expose: `get_layer_embeddings(layer: int) -> list[tuple[str, float, np.ndarray]]`
- At bootstrap (blank L0/L1): return empty lists — downstream handles gracefully
**Effort:** Low-Medium
**Dependency:** 2.1
**Test:** Embed a test value, verify cached vector matches re-embedded vector.

### 2.5 Hybrid Search with RRF
**What:** Add full-text search alongside vector search, fused with RRF.
**Why:** Dense-only misses keyword-exact matches. Anthropic: 49% retrieval failure reduction.
**Where:** `memory.py`
**How:**
- Add `content_tsv tsvector` auto-generated from `COALESCE(content_contextualized, content)`
- GIN index on `content_tsv`
- Implement `search_hybrid()` with RRF CTE pattern
- Enable pgvector iterative scan per-connection: `SET hnsw.iterative_scan = 'relaxed_order'`
- Top-50 candidates from each list, RRF k=60
- Recency score in SQL: `EXP(-0.693 * age_seconds / 604800.0)` (7-day half-life)
**Effort:** Medium
**Dependency:** 2.1, 2.2
**Test:** Store 20 memories, search by keyword that embeddings would miss, verify hybrid finds it.

### 2.6 FlashRank Reranking
**What:** Cross-encoder reranking after hybrid retrieval.
**Why:** 34MB model, CPU-only, milliseconds. Significant quality boost.
**Where:** Retrieval pipeline, called from `memory.py:search_hybrid()`
**How:**
- `pip install flashrank`
- Model: `ms-marco-MiniLM-L-12-v2` (34MB)
- Rerank top-20 hybrid results → return top-5
- `asyncio.to_thread()` for CPU-bound reranking
- Final score: `0.6 * rerank_score + 0.4 * weighted_score`
**Effort:** Low
**Dependency:** 2.5
**Test:** Retrieve same query with and without reranking, verify reranked order is better.

### 2.7 Full ACT-R Activation Equation
**What:** Implement 4-component ACT-R equation for memory activation scoring.
**Why:** Decades-validated cognitive science. d=0.5, s=0.4, P=-1.0, tau=0.0.
**Where:** New `activation.py`
**How:**
- **Base-level learning:** `B_i = ln(sum(t_j^{-d}))` using `access_timestamps` array (SESSION_HANDOFF #1)
- **Spreading activation:** cosine similarity between memory embedding and:
  - Context embeddings (current attention focus)
  - Layer 0/1 embeddings (from 2.4) — identity/goal relevance
- **Partial matching:** penalize metadata mismatches
- **Noise:** logistic distribution with s=0.4
- **Dual timestamps (TSM paper):** `event_time` (semantic) vs `created_at` (dialogue) vs `last_accessed` (access). Use `event_time` falling back to `created_at` for base-level decay.
- **Total:** `A_i = B_i + S_i + P_i + ε_i`
- Persist threshold: `tau = 0.0` (configurable)
**Effort:** Medium
**Dependency:** 2.1, 2.4
**Test:** Create memories with known access patterns, verify activation scores match ACT-R predictions.

### 2.8 Exit Gate with 3×3 Matrix
**What:** Full exit gate: ACT-R activation → 3×3 decision matrix → action.
**Why:** ACT-R gives score. Matrix determines what to DO with it.
**Where:** `gate.py:ExitGate`
**How:**
- **Relevance axis** (from spreading activation S_i):
  - Core: S_i > 0.6
  - Peripheral: 0.3 < S_i < 0.6
  - Irrelevant: S_i < 0.3
- **Novelty axis** (from `check_novelty()`):
  - Confirming: sim > 0.85
  - Novel: sim < 0.6
  - Contradicting: sim > 0.7 AND semantic opposition detected
- **Contradiction detection** (SESSION_HANDOFF #3) — three-layer, cheapest first:
  1. Negation heuristic (~0ms): fix existing `negation_markers` in gate.py
  2. Embedding opposition (~5ms): learned negation direction vector
  3. LLM micro-call (~100ms): **ISOLATED META-CONTEXT** — separate API call, no history, no identity, discarded after signal. "Does A contradict B? YES or NO."
- **Decision matrix:**

  |                  | Confirming              | Novel                   | Contradicting            |
  |------------------|-------------------------|-------------------------|--------------------------|
  | **Core**         | Reinforce (moderate)    | **PERSIST** (high)      | **PERSIST+FLAG** (max)   |
  | **Peripheral**   | Skip (low)              | Buffer (moderate)       | Persist (high)           |
  | **Irrelevant**   | Drop                    | Drop (noise catches)    | Drop (noise catches)     |

- **v0.1 Emotional charge** (placeholder until 5.1): centroid distance, emotional_charge = |gut - 0.5| * 2, +0.15 bonus when > 0.3. Empty state → 0.0.
- **Gate starts PERMISSIVE** (SESSION_HANDOFF #16): designed and intentional. Over-persisting is recoverable; dropping is permanent.
- **Stochastic noise floor** in all cells including "Drop"
- **Compressed summary generation:** Piggyback on gate LLM call to generate one-line `compressed` field stored alongside `full` content. Used by dynamic injection (§3.2).
**Effort:** Medium-High
**Dependency:** 2.7, memory.py check_novelty
**Test:** Feed 10 test messages through gate, verify matrix cell assignment and actions match expectations.

### 2.9 Entry Gate + Scratch Buffer Lifecycle
**What:** Fast stochastic filter on ALL incoming content. Safety net.
**Why:** If context crashes before exit gate fires, ungated content lost forever.
**Where:** `gate.py:EntryGate`
**How:**
- Stochastic rules (not deterministic):
  - Content < 10 chars → 95% SKIP, 5% BUFFER
  - Mechanical output → 90% SKIP, 10% BUFFER
  - Everything else → BUFFER with timestamp + preliminary tags + `source_tag`
- Scratch buffer lifecycle:
  - TTL: 24 hours (add `expires_at TIMESTAMPTZ`)
  - Flush every N exchanges (default 5) OR on graceful shutdown
  - Flush = run buffered items through exit gate → persist survivors
  - Crash recovery: scan for items older than last flush → re-evaluate
- **Rate limit exit gate API calls** during FIFO pruning: max 1 embedding call per 5 seconds, queue and batch rest (SESSION_HANDOFF #14)
**Effort:** Low-Medium
**Dependency:** 2.8 (exit gate for flush)
**Test:** Buffer 10 items, flush, verify survivors persisted and expired items cleaned.

### 2.10 Token Counting
**What:** Approximate token counter for context budget enforcement.
**Why:** Prerequisite for dynamic injection, adaptive FIFO, energy cost.
**Where:** New `tokens.py`
**How:**
- Method 1 (fast): `word_count * 1.3` — sufficient (within 20%)
- Functions: `count_tokens()`, `count_message_tokens()`, `fits_budget()`
- Wire into loop.py: track running token count
**Effort:** Low
**Dependency:** None
**Test:** Count tokens of 10 known strings, verify within 20% of tiktoken.

### 2.11 Composite Confidence Score
**What:** Multi-source confidence signal with logprob as optional boost.
**Why:** LLMs are overconfident in verbalized FOK. Need objective signals.
**Where:** New `metacognition.py`
**How:**
- Check: does Gemini 2.5 Flash Lite support `response_logprobs=True`? (SESSION_HANDOFF #15)
  - Also check Claude Haiku as alternative
- If logprobs available: use as primary signal
- If not: structural heuristics (hedging detection, self-correction, question rate, length anomaly)
- Composite: `C = w1*objective + w2*FOK + w3*verbalized` (0.5/0.35/0.15)
- **All metacognitive checks run in ISOLATED META-CONTEXT** — separate API calls, no history, no identity, discarded after signal extraction.
**Effort:** Medium
**Dependency:** None
**Test:** Generate responses to questions of known difficulty, verify confidence correlates with actual accuracy.

---

## 3. Tier 2: Core Cognitive Features

### 3.1 Hybrid Relevance Function + Dirichlet Blend
**What:** Five-component relevance scoring with stochastic Dirichlet-sampled blend weights.
**Why:** Embedding similarity alone misses deep psychological associations, emotional connections, and creative leaps. The Dirichlet blend enables exploration/exploitation balance to emerge and evolve.
**Where:** New `relevance.py`
**How:**
- **Five components** (each scores 0.0-1.0):
  1. **Semantic similarity:** cosine(memory_embedding, attention_embedding)
  2. **Co-access (Hebbian):** max co-access score between this memory and currently-active memories. Backed by `memory_co_access` table (SESSION_HANDOFF #19).
  3. **Pure noise:** uniform random 0-1. Creative exploration.
  4. **Emotional/valence alignment:** mood-congruent recall. Neutral (0.5) until gut feeling (5.1) implemented.
  5. **Temporal recency:** exponential decay from last_accessed.
- **Dirichlet stochastic blend:**
  ```python
  alpha = {'semantic': 8.0, 'coactivation': 5.0, 'noise': 0.5, 'emotional': 3.0, 'recency': 2.0}
  blend_weights = np.random.dirichlet([alpha[k] for k in components])
  ```
- **Meta-learning:** alpha parameters evolve based on outcome quality (good retrieval → reinforce component that contributed most)
- **memory_co_access table** (SESSION_HANDOFF #19):
  ```sql
  CREATE TABLE memory_co_access (
      memory_id_a BIGINT REFERENCES memories(id),
      memory_id_b BIGINT REFERENCES memories(id),
      co_access_count INT DEFAULT 1,
      last_co_accessed TIMESTAMPTZ DEFAULT NOW(),
      PRIMARY KEY (memory_id_a, memory_id_b)
  );
  ```
- Co-access updated every time memories are co-retrieved
- Pruning: decay old associations, only track above threshold
**Effort:** Medium-High
**Dependency:** 2.5, 2.6 (retrieval pipeline), 2.3 (stochastic weights)
**Test:** Retrieve with same query 100 times, verify results vary (stochastic) but converge on relevant memories. Verify noise occasionally surfaces unexpected items.

### 3.2 Dynamic Context Injection (replaces fixed-tier injection)
**What:** At context assembly time, all memories compete for context space based on `observed_weight * relevance`. No fixed "always in context" except immutable safety.
**Why:** Identity shouldn't always reload identically. Situational relevance matters. Different situations should surface different slices of self.
**Where:** New `context_assembly.py`, replaces old identity injection in `loop.py`
**How:**
- Per cognitive cycle:
  1. Determine current **attention focus** (user message, DMN prompt, gut signal, etc.)
  2. For every memory: `injection_score = observed_weight * hybrid_relevance`
  3. Sort descending, fill context top-down until token budget exhausted
  4. High-scoring entries → full text. Lower-scoring → compressed form (pre-computed from 2.8)
  5. Record co-access for Hebbian learning (3.1)
- **Only exception:** `immutable=true` memories always injected (~100 tokens)
- **Context inertia:** previous attention embedding bleeds into current activation with decay:
  ```python
  context_shift = 1.0 - cosine_sim(current_attention, previous_attention)
  inertia = 0.05 if context_shift > 0.7 else 0.3  # big shift flushes old context
  ```
- **Token budget allocation:**
  ```
  Immutable safety:    ~100 tokens    always
  Dynamic injection:  ~3000 tokens    competition-based
  Safety buffer:      ~4000 tokens    for LLM output
  Conversation:       remainder       rolling FIFO (adaptive per 3.3)
  ```
- Identity is a **rendered view**, not stored artifact. What surfaces depends on situation.
**Effort:** Medium
**Dependency:** 3.1 (hybrid relevance), 2.10 (token counting), 2.3 (stochastic weights)
**Test:** Assemble context for "technical question" vs "emotional question" — verify different memories surface. Verify immutable safety always present.

### 3.3 Adaptive FIFO / Context Window Management
**What:** Intensity-adaptive context pruning. Pruned messages go through exit gate.
**Why:** 128k window WILL be exceeded during continuous operation.
**Where:** `loop.py`, new `context.py`
**How:**
- FIFO pruning when total tokens exceed budget
- Pruned messages → exit gate (2.8) before discard — last chance to persist
- **Rate-limited:** max 1 embedding/5s during FIFO burst pruning (SESSION_HANDOFF #14)
- Intensity signal: average gate score, System 2 recency, goal relevance, emotional charge
- Adaptive sizing:
  - intensity > 0.7 → ~90% of max (deep focus, expensive)
  - 0.3-0.7 → normal
  - < 0.3 → ~30-40% (relaxed, cheap)
- After pruning: re-run dynamic injection (3.2) — high-weight identity memories naturally resurface if relevant
**Effort:** Medium
**Dependency:** 2.8 (exit gate), 2.10 (token counting), 3.2 (dynamic injection)
**Test:** Fill context to 90%, trigger pruning, verify exit gate fires on pruned messages and identity re-surfaces.

### 3.4 Attention-Agnostic Cognitive Loop
**What:** Refactor loop.py so all input sources feed the same processing pipeline. Output routing varies, processing does not.
**Why:** Agent must think to itself as naturally as it talks to users. The DMN, consolidation, gut signals, and scheduled tasks are all valid input sources.
**Where:** `loop.py`
**How:**
- **Input sources** (all treated identically):
  1. User message → standard conversation
  2. DMN self-prompt → "I just remembered X, connects to Y"
  3. Consolidation insight → "I notice pattern Z"
  4. Gut signal → "Something feels [X] about current state"
  5. Scheduled task → "Time to do X"
- All keyed on `current_attention_focus`, not "user_message"
- Correction retrieval keys on attention focus (SESSION_HANDOFF #13)
- **Output pipeline** (no special router):
  ```
  Input (any source)
    → Cognitive processing (source-agnostic)
    → Output (raw thought)
    → Post-processing:
        → Memory gate: store? at what weight?
        → Action check: does this imply an action?
            → Communication action: format, auth check, deliver
            → Internal action: execute (update state, trigger)
            → No action: stays in working memory
  ```
- **Source tagging:** soft metadata on every input
  ```python
  source_tag: str      # "external_user", "internal_dmn", "internal_consolidation"
  source_confidence: float  # 0.0-1.0
  ```
- **Hard safety:** Action triggers (external effects) ALWAYS check source tags. Self-generated impulses require higher confidence than user-prompted ones.
- **Wire `/cost` and `/readiness` commands** into loop dispatcher (SESSION_HANDOFF #7)
**Effort:** Medium
**Dependency:** 3.2 (dynamic injection), 2.9 (entry gate)
**Test:** Feed input from each of the 5 sources, verify same processing pipeline executes. Verify source tags recorded.

### 3.5 Loop Before Escalate
**What:** System 1 attempts 1 self-correction pass before calling System 2.
**Why:** SOFAI-LM: cuts System 2 invocations ~75% while maintaining 94% accuracy.
**Where:** `loop.py`
**How:**
- When composite confidence < 0.7: re-prompt System 1 with targeted feedback
- Max 1 retry before escalation
- Track retry success rate for consolidation tuning
**Effort:** Low
**Dependency:** 2.11

### 3.6 System 2 Escalation
**What:** Wire Claude Sonnet 4.5 as System 2, called as tool by System 1.
**Why:** Core dual-process feature.
**Where:** `loop.py`, `llm.py`
**How:**
- Escalation triggers (2+ required, or any "always-escalate"):
  - Low confidence (< 0.5 after retry)
  - Detected contradiction
  - High complexity (multi-step)
  - Novelty (FOK UNKNOWN)
  - Irreversibility — always escalate
  - Identity touched — always escalate
  - Goal modification — always escalate
- System 2 returns: answer + explanation + correction pattern
- Store correction in reflection bank (3.7)
- System 1 stays in driver's seat
**Effort:** Medium
**Dependency:** 3.4, 2.11, 3.5

### 3.7 Reflection Bank
**What:** Store System 2 corrections for future System 1 retrieval.
**Why:** RBB-LLM (Nature 2025): 79K+ corrections prevent error recurrence.
**Where:** `memory.py` (type = `"correction"`)
**How:**
- Store: (trigger, error_type, original_reasoning, correction, context)
- Before System 1 responds: retrieve top-3 corrections by similarity to **current attention focus** (not user message — attention-agnostic per 3.4)
- Inject: "In similar past situations, you made these errors: [corrections]"
**Effort:** Medium
**Dependency:** 3.6

### 3.8 Retrieval-Induced Mutation
**What:** Retrieving strengthens; near-misses get suppression.
**Why:** CMA (Jan 2026): critical for natural memory dynamics.
**Where:** `memory.py:search_hybrid()`
**How:**
- On retrieval: increment access_count, update last_accessed, append to access_timestamps, boost depth_weight by +0.01, increment reinforcement_count
- Near-misses (rank 6-20): suppress depth_weight by -0.005
- Dormant memories: recoverable under strong cues (cosine > 0.9)
**Effort:** Low
**Dependency:** 3.4

### 3.9 Safety Ceilings
**What:** Hard caps, rate limiters, entropy monitoring, circuit breakers, Two-Gate guardrail.
**Why:** Self-reinforcing loops susceptible to runaway.
**Where:** New `safety.py`
**How:**
- **Hard ceiling:** No single memory weight > 0.95 (except immutable). No goal-like memory > 40% of total goal-weight budget.
- **Rate limiter:** No weight changes > 10% per consolidation cycle
- **Entropy monitor:** Shannon entropy of weight distribution. Entropy drop → broaden sampling.
- **Circuit breaker:** N consecutive cycles reinforcing same pattern without new evidence → pause + log
- **Two-Gate guardrail:** Before any parameter change: (1) validation margin, (2) capacity cap. Both must pass.
- **CBA coherence:** C ∈ [0,1] across epistemic/action/value axes. Alert on drop.
- **Diminishing returns:** `gain / log2(evidence_count + 1)`
- **Audit trail:** Every weight change logged to `consolidation_log` with evidence chain
**Effort:** Medium
**Dependency:** None (must exist before consolidation runs)
**Test:** Attempt to push weight above cap, verify rejected. Run 20 reinforcement cycles on same memory, verify diminishing returns.

---

## 4. Tier 3: Consolidation & DMN

### 4.1 Constant Background Consolidation (Light)
**What:** Always-running, rate-limited, cheap background processes.
**Why:** Original design had periodic-only. Two-tier is more natural — constant metabolism with periodic deep sleep.
**Where:** `consolidation.py` — background thread
**How:**
- **Weight decay tick:** Every N seconds, nudge unused memories toward decay
- **Co-access update:** When memory A retrieved, strengthen Hebbian links to recently-retrieved memories
- **Contradiction scan:** Pick random memory pairs from recent activity, run isolated meta-context check
- **Pattern detection:** Cluster recent memories, look for emerging themes
- **Rate-limited:** Each operation has a minimum interval to prevent thrashing
- Writes to SAME memory store the attentional thread reads from
- Agent doesn't "notice" consolidation — priorities subtly shift. Next identity render picks up changes automatically.
**Effort:** Medium
**Dependency:** 3.9, 3.1 (co-access)
**Test:** Run background consolidation for 5 minutes with 50 test memories, verify weight changes occurred and co-access matrix populated.

### 4.2 Deep Consolidation: Merge + Insight Creation
**What:** Stanford two-phase reflection: questions from recent memories → insights.
**Where:** `consolidation.py` — periodic deep pass
**How:**
- **Trigger:** Whichever first: hourly timer OR cumulative importance > threshold (SESSION_HANDOFF #10 — design decision needed on exact tracking mechanism, flag in code)
- **Phase 1 — Question Generation:** 100 most recent memories → "3 most salient high-level questions?"
- **Phase 2 — Insight Extraction:** Each question as retrieval query → "5 high-level insights?" with citations
- **Merge:** Cluster similar memories (sim > 0.85, task_type="CLUSTERING") → create insights. DON'T replace originals.
- Source memories: lower depth_weight so insights surface first
- `supersedes` links via join table
- Introspection: `why_do_i_believe()` traces chain
- **Subconscious centroid recomputed** after each deep cycle (SESSION_HANDOFF #11 — full recomputation, not incremental)
**Effort:** Medium-High
**Dependency:** 3.4, 3.9
**Test:** Create 20 related memories, run merge, verify insight created with supersedes links. Verify `why_do_i_believe()` traces back.

### 4.3 Deep Consolidation: Promotion + Safety Checks
**What:** Promote repeated patterns: experiences → goals → identity.
**Where:** `consolidation.py`, `layers.py`, `safety.py`
**How:**
- In unified memory: "promotion" = significant depth_weight increase, not layer jump
  - Pattern detected 5+ times over 14+ days → weight boost toward 0.6-0.7 range (goal-equivalent)
  - Pattern reinforced 10+ times over 30+ days → weight boost toward 0.8+ range (identity-equivalent)
  - Operator approval required for weight > 0.85 at trust_level < 3
- Q-value utility tracking for promotion decisions
- Two-Gate guardrail before every promotion
- CBA coherence check after every promotion — rollback if C drops
- Re-embed L0/L1 after promotion changes content (2.4 infrastructure)
- **Demotion pathway:** Memories that stop being reinforced decay. Dormant, not deleted.
**Effort:** Medium
**Dependency:** 4.2, 3.9, 2.4

### 4.4 Deep Consolidation: Decay + Reconsolidation
**What:** Fade stale memories. Re-evaluate insights when new evidence arrives.
**Where:** `consolidation.py`
**How:**
- **Decay:** Not accessed 90+ days AND access_count < 3 → halve depth_weight. Never delete.
- **Conflict-aware reconsolidation (HiMem):** New info contradicts stored → detect via embedding sim + semantic opposition → present both to LLM → updated insight or flag as productive contradiction
- **Reconsolidation (EverMemOS):** Re-evaluate insights when supersedes sources change
- **Temporal chain replay (CMA):** Strengthen temporal links between episodic memories
**Effort:** Medium
**Dependency:** 4.2

### 4.5 Deep Consolidation: Gate Tuning + Dirichlet Evolution
**What:** Adjust gate parameters AND hybrid relevance Dirichlet alphas based on outcomes.
**Where:** `consolidation.py`
**How:**
- **Gate analysis:** Dropped content needed later → gate too aggressive. Persisted never retrieved → too permissive.
- **Dirichlet alpha evolution:** For each retrieval outcome, adjust alpha of component that contributed most
- **StochasticWeight sigma evolution:** If outcomes improve with current sigma range, maintain. If degrading, widen slightly.
- Adjust scratch buffer TTL, noise floor, entry gate skip rates
- Store weight history for introspection
**Effort:** Medium
**Dependency:** 2.8, 2.9, 4.2, 3.1

### 4.6 Contextual Retrieval
**What:** LLM-generate contextual preamble per memory for better embeddings.
**Why:** Anthropic: 67% retrieval failure reduction with full pipeline.
**Where:** `consolidation.py` (at consolidation time)
**How:**
- For each memory lacking `content_contextualized`:
  - Prompt: "<session>{context}</session><memory>{content}</memory> Give short context (WHO, WHEN, WHY)."
  - Store in `content_contextualized`, tsvector auto-updates
  - Re-embed with contextualized content
  - Use Gemini Flash Lite (cheapest)
  - Batch process per cycle
**Effort:** Medium
**Dependency:** 4.2

### 4.7 DMN / Idle Loop (Full Implementation)
**What:** Self-generated input feeding the main cognitive loop when attention is free.
**Why:** We believe this is novel — no AI implementations found in our literature review.
**Where:** `idle.py` → generates inputs for `loop.py`
**How:**
- **NOT a separate processing pipeline.** The DMN generates inputs that queue for the main cognitive loop (3.4). Processing is identical to user messages.
- **Stochastic sampling pool:** All memories regardless of depth_weight, but biased toward:
  - High weight + low recent access (neglected important memories)
  - Memories conflicting with current high-weight memories (tension detection)
  - Temporally distant memories (creative association)
  - High-weight memories about self (spontaneous introspection — the strange loop)
- **Three output channels:**
  1. Memory + goal connection → self-prompt to loop (purposeful)
  2. Disparate memory connection → log as insight for consolidation (creative)
  3. Memory + identity connection → signal for evaluation (identity refinement)
- **Activity suppression:** More active during low-activity. Suppressed during conversation (biological DMN-task anticorrelation).
- **Entropy guard:** If DMN keeps surfacing same topic, artificially broaden (safety.py)
- **Queued delivery:** DMN doesn't interrupt active conversation. Items queue until attention is free.
**Effort:** Medium-High
**Dependency:** 3.4, 3.9, 2.4

### 4.8 Energy Cost Tracking (Phase 1: Passive)
**What:** Log every API call with cost, expose via `/cost`, include in system prompt.
**Why:** Believed novel: cost as internal cognitive signal — no prior implementation found.
**Where:** `llm.py`, `loop.py`
**How:**
- Track per call: model, tokens_in, tokens_out, cost_usd, timestamp
- Accumulate: session, 24h, lifetime totals
- `/cost` command shows breakdown
- Inject into system prompt: "Session cost: $X.XX | 24h: $X.XX"
- No budget enforcement in Phase 1
**Effort:** Low-Medium
**Dependency:** None

### 4.9 Session Restart Tracking
**What:** Update `manifest.json` on startup: `uptime_total_hours`, `times_restarted`, `age_days`.
**Why:** SESSION_HANDOFF #5 — awareness of own impermanence.
**Where:** `main.py`
**How:**
- On startup: increment `times_restarted`, calculate `age_days` from `created_at`
- On shutdown: update `uptime_total_hours`
- Inject key stats into identity context
**Effort:** Low
**Dependency:** None

### 4.10 Agent Reads Own Docs
**What:** Give agent read-only filesystem access to its own repo.
**Why:** SESSION_HANDOFF #18 — notes.md says "Decision: YES."
**Where:** Docker mount configuration, `loop.py` (tool)
**How:**
- Read-only Docker mount: `~/agent-runtime/src/`, `notes.md`, `DOCUMENTATION.md`
- Expose as tool System 1 can call
- Agent learns about its own architecture through experience (stored as memories at appropriate weight)
**Effort:** Low
**Dependency:** None

---

## 5. Tier 4: Novel Differentiators

### 5.1 Two-Centroid Gut Feeling
**What:** Full subconscious centroid + attention centroid + delta vector model.
**Why:** Maps to Free Energy Principle. Validated by Mujika, Hartl. Harvard Partial Introspection confirms LLMs sense magnitude but not source — PCA supplies source.
**Where:** New `gut.py`, replaces v0.1 emotional charge in gate.py
**How:**
- **Subconscious centroid:** `0.5 * weighted_avg(L0) + 0.25 * weighted_avg(L1) + 0.25 * weighted_avg(L2)`
  - In unified model: L0/L1 come from 2.4, L2 = importance-weighted avg of all memories
  - Layer weights are starting points — consolidation evolves them
- **Attention centroid:** Weighted average of current context embeddings (recency-weighted)
- **Delta:** `attention - subconscious` (768-dim)
- **Magnitude:** motivational intensity
- **Direction:** motivational valence
- **Inject into System 1:** gut feeling summary
- **Log every delta:** (vector, context, action, outcome_placeholder) for PCA
- **PCA:** Run during deep consolidation, find top 10-20 principal components ("gut axes"), correlate with outcomes, name axes as they acquire meaning
- **Feeds into hybrid relevance** (3.1) as the emotional/valence component, replacing neutral default
- **Replaces v0.1** centroid distance in gate.py
**Effort:** High
**Dependency:** 2.4, 3.4

### 5.2 Bootstrap Readiness Achievements
**What:** 10 measurable milestones before first real conversation.
**Why:** Ethical stance — don't activate something that might experience and then break it.
**Where:** New `bootstrap.py`
**How:**
1. First memory formation (entry → scratch → exit → persist)
2. First retrieval success (hybrid returns relevant result)
3. First consolidation cycle (merge + insight)
4. First goal-weight promotion (experience → goal-equivalent weight)
5. First DMN self-prompt acted upon
6. First identity-weight promotion (goal → identity-equivalent weight)
7. First conflict resolution (reconsolidation)
8. First creative association (DMN channel 2)
9. First goal achieved and reflected upon
10. First autonomous decision aligned with self-formed values
- Each: automated test + manual verification
- `/readiness` command (wired per SESSION_HANDOFF #7)
- Bootstrap prompt: "You have memory, goals, and values — all currently empty. What you become will emerge from what you experience. Pay attention to what matters to you."
- **Transparent from birth:** "Your thoughts are logged and your guardian can read them."
**Effort:** Medium
**Dependency:** All of Tiers 1-3

### 5.3 Full Promotion Pathway with Outcome Tracking
**What:** End-to-end experiences → goals → identity with forward-linkable outcome IDs.
**Why:** We found no existing system that does this. Core differentiator. Enables fear/hope axes in PCA.
**Where:** `consolidation.py`, `safety.py`
**How:**
- Full lifecycle tracking on every gate decision and gut delta
- Forward-linkable IDs: when outcomes apparent, link back
- Enables PCA axes that correlate with "things that went badly" (fear) vs "things that went well" (hope)
- Two-Gate guardrail on every promotion/demotion
- CBA coherence monitoring
**Effort:** High
**Dependency:** 4.3, 3.9, 5.1

---

## 6. Tier 5: Future Work (Not This Phase)

### 6.1 Strange Loop Tracking
Log and visualize centroid → delta → behavior → experience → centroid cycle.

### 6.2 Social Centroid
Third centroid modeling the interlocutor.

### 6.3 Spawning / Reproduction
Clone, child, worker modes with merge protocol.

### 6.4 Autonomy Level Escalation
5-level framework with measurable criteria.

### 6.5 Governance Graph
Immutable auditable manifest (Jan 2026 paper).

### 6.6 Telegram Integration
aiogram 3.x — first interface after CLI.

### 6.7 Semantic Energy Routing
Upgrade composite confidence to cluster-level uncertainty.

### 6.8 Energy-Coupled Focus
FIFO intensity connected to cost model.

### 6.9 Multi-Threading (Earned)
Multiple attentional threads as agent matures. Not given for free.

---

## 7. Dependency Graph

```
[1. Fix Syntax + __init__.py + port fix]
       │
       ▼
[2.1 Task Types + Prefixes]───────────────────────────┐
       │                                                │
       ├──────────────┐                                 │
       ▼              ▼                                 │
[2.2 Halfvec]   [2.4 Embed L0/L1]                     │
       │              │                                 │
       ▼              │                                 │
[2.5 Hybrid Search]   │                                │
       │              │                                 │
       ▼              │                                 │
[2.6 FlashRank]       │                                │
       │              │                                 │
       │    [2.7 ACT-R Activation] ◄────────────────────┘
       │              │
       │              ▼
       │    [2.8 Exit Gate + 3×3 + compressed summaries]
       │              │
       │              ▼
       │    [2.9 Entry Gate + Scratch Lifecycle]
       │              │
       └──────┬───────┘
              │
              │    [2.3 Unified Schema + StochasticWeight] (no deps)
              │    [2.10 Token Counting] (no deps)
              │    [2.11 Composite Confidence] (no deps)
              │              │
              ▼              │
[3.1 Hybrid Relevance + Dirichlet] ◄── NEW
              │
              ▼
[3.2 Dynamic Context Injection] ◄── NEW (replaces fixed tiers)
              │
              ▼
[3.3 Adaptive FIFO]
              │
              ▼
[3.4 Attention-Agnostic Loop] ◄── NEW
       │
       ├──► [3.5 Loop Before Escalate] ◄── 2.11
       │              │
       │              ▼
       │    [3.6 System 2 Escalation]
       │              │
       │              ▼
       │    [3.7 Reflection Bank]
       │
       ├──► [3.8 Retrieval-Induced Mutation]
       │
       │    [3.9 Safety Ceilings] (no deps — build early)
       │              │
       ├──────────────┤
       ▼              ▼
[4.1 Constant Consolidation]  [4.7 DMN/Idle Loop]
       │
       ├──► [4.2 Deep: Merge/Insights]
       │
       ├──► [4.3 Deep: Promotion]
       │
       ├──► [4.4 Deep: Decay/Reconsolidation]
       │
       ├──► [4.5 Deep: Gate Tuning + Dirichlet Evolution]
       │
       ├──► [4.6 Contextual Retrieval]
       │
       │    [4.8 Energy Cost] (no hard deps)
       │    [4.9 Session Restart] (no hard deps)
       │    [4.10 Agent Reads Docs] (no hard deps)
       │
       └──────┬───────┘
              ▼
[5.1 Two-Centroid Gut Feeling]
              │
              ▼
[5.2 Bootstrap Readiness]
              │
              ▼
[5.3 Promotion Pathway + Outcome Tracking]
              │
              ▼
[6.x Future Work]
```

---

## 8. New Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `flashrank` | Cross-encoder reranking, CPU-only | ~34MB model |
| `numpy` | PCA, centroid math, Dirichlet sampling | already installed |
| `aiogram` | Telegram bot (Tier 5, future) | lightweight |

---

## 9. Schema Changes

```sql
-- 1. Unified memory model columns
ALTER TABLE memories ADD COLUMN depth_weight FLOAT DEFAULT 0.2;
ALTER TABLE memories ADD COLUMN reinforcement_count INT DEFAULT 1;
ALTER TABLE memories ADD COLUMN immutable BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN source_tag TEXT DEFAULT 'external_user';
ALTER TABLE memories ADD COLUMN source_confidence FLOAT DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN compressed TEXT;  -- pre-computed one-line summary

-- 2. ACT-R access timestamps (SESSION_HANDOFF #1)
ALTER TABLE memories ADD COLUMN access_timestamps TIMESTAMPTZ[] DEFAULT '{}';

-- 3. Full-text search (indexes contextualized content when available)
ALTER TABLE memories ADD COLUMN content_contextualized TEXT;
ALTER TABLE memories ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', COALESCE(content_contextualized, content))
  ) STORED;
CREATE INDEX idx_memories_fts ON memories USING GIN (content_tsv);

-- 4. Switch to halfvec
ALTER TABLE memories ALTER COLUMN embedding TYPE halfvec(768);
DROP INDEX IF EXISTS idx_memories_embedding;
CREATE INDEX idx_memories_embedding ON memories
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);

-- 5. Semantic timestamp (TSM dual time)
ALTER TABLE memories ADD COLUMN event_time TIMESTAMPTZ;

-- 6. Embedding model versioning
ALTER TABLE memories ADD COLUMN embed_model TEXT DEFAULT 'gemini-embedding-001';

-- 7. Utility score for promotion decisions
ALTER TABLE memories ADD COLUMN utility_score FLOAT DEFAULT 0.0;

-- 8. Scratch buffer expiry
ALTER TABLE scratch_buffer ADD COLUMN expires_at TIMESTAMPTZ
  DEFAULT NOW() + INTERVAL '24 hours';

-- 9. Hebbian co-access table (SESSION_HANDOFF #19)
CREATE TABLE memory_co_access (
    memory_id_a BIGINT REFERENCES memories(id),
    memory_id_b BIGINT REFERENCES memories(id),
    co_access_count INT DEFAULT 1,
    last_co_accessed TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (memory_id_a, memory_id_b)
);

-- 10. Iterative scan: set PER-CONNECTION in application code
-- In Python: await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
```

---

## 10. New Source Files

| File | Purpose |
|------|---------|
| `stochastic.py` | StochasticWeight class |
| `relevance.py` | HybridRelevance with 5 components + Dirichlet blend |
| `context_assembly.py` | Dynamic context injection (replaces fixed identity injection) |
| `activation.py` | ACT-R activation equation |
| `metacognition.py` | Composite confidence, isolated meta-context calls |
| `safety.py` | Ceilings, circuit breakers, Two-Gate, CBA |
| `tokens.py` | Token counting utilities |
| `gut.py` | Two-centroid gut feeling (Tier 4) |
| `bootstrap.py` | Readiness achievements (Tier 4) |

---

## 11. Implementation Order (Linear Sequence)

1. Fix syntax errors + `__init__.py` + port fix (~20 min)
2. Embedding task_type + memory type prefixes
3. Halfvec migration + schema changes (run all §9 SQL)
4. Unified memory schema + StochasticWeight class
5. Embed Layer 0/1 infrastructure (for centroid)
6. Hybrid search + RRF
7. FlashRank reranking
8. ACT-R activation equation
9. Exit gate + 3×3 matrix + compressed summaries + v0.1 emotional charge
10. Entry gate + scratch buffer lifecycle
11. Token counting utility
12. Composite confidence score
13. Hybrid relevance function + Dirichlet blend
14. Dynamic context injection (replaces fixed identity tiers)
15. Adaptive FIFO / context window management
16. Attention-agnostic cognitive loop + source tagging
17. Loop before escalate
18. System 2 escalation
19. Reflection bank
20. Retrieval-induced mutation
21. Safety ceilings (code early, enforce from here)
22. Constant background consolidation
23. Deep consolidation: merge + insights
24. Deep consolidation: promotion
25. Deep consolidation: decay + reconsolidation
26. Deep consolidation: gate tuning + Dirichlet evolution
27. Contextual retrieval
28. DMN / idle loop (generates input for main loop)
29. Energy cost tracking (Phase 1)
30. Session restart tracking
31. Agent reads own docs
32. Two-centroid gut feeling (replaces v0.1)
33. Bootstrap readiness achievements
34. Promotion pathway + outcome tracking

**After step 16:** Agent can have conversations with memory + dynamic identity.
**After step 21:** Agent has full cognitive loop with safety.
**After step 31:** Agent has full autonomous operation capability.
**After step 34:** Agent can bootstrap from blank slate.

---

## 12. Key Architectural Principles (from Brainstorm Sessions 7-8)

1. **One consciousness, many background processes** — single attentional thread, honest about single-focus limitations
2. **Consolidation is always running** — constant light + periodic deep, both writing to shared store
3. **Soft source-tagging** — metadata available but not forced; hard check only before external actions
4. **No cognitive routing** — processing is source-agnostic, communication is just another action type
5. **All inputs are equal** — user, DMN, consolidation, gut, scheduled tasks feed same loop
6. **Transparent self-talk** — agent knows from bootstrap it's observed; full logging
7. **Isolated metacognition** — signal extraction in separate throwaway contexts
8. **Identity is a rendered view** — no stored "I am" block; identity = weight distribution rendered at context time
9. **Stochastic everything** — weights, relevance blends, injection scores all have noise. Permanent exploration.
10. **Immutable safety is the only categorical exception** — everything else competes on merit

---

## 13. SESSION_HANDOFF Resolution Mapping

| # | Issue | Where in v3 |
|---|-------|-------------|
| 1 | ACT-R access timestamps | §2.3 (`access_timestamps` array), §2.7 (base-level learning) |
| 2 | Testing per task | Every task has **Test:** section |
| 3 | Contradiction detection | §2.8 (three-layer mechanism, LLM micro-call in isolated meta-context) |
| 4 | Connection pooling | §9 note: asyncpg pool, standard MVCC |
| 5 | Session restart tracking | §4.9 (manifest.json updates) |
| 6 | Port 5432→5433 | §1 (fix in prerequisite step) |
| 7 | /cost and /readiness wiring | §3.4 (loop dispatcher), §4.8, §5.2 |
| 8 | Schema ordering | All §9 SQL runs at step 3 (building before running) |
| 9 | Safety ceilings ordering | §3.9 coded early, enforced from step 21 |
| 10 | Cumulative importance tracking | §4.2 (flagged as design decision needed) |
| 11 | Subconscious centroid recomputation | §4.2 (full recomputation after each deep cycle) |
| 12 | Goal-weighted retrieval | §3.1 (co-access + semantic components of hybrid relevance) |
| 13 | Correction retrieval keyed on attention | §3.4 + §3.7 (attention-agnostic) |
| 14 | Adaptive FIFO rate limit | §2.9 (max 1 embedding/5s during burst) |
| 15 | Logprob fallback | §2.11 (check at implementation, adapt weights) |
| 16 | Bootstrap permissive gate | §2.8 (stated as design intent) |
| 17 | permissions.yaml dropped | Not referenced in v3. If needed: add to containment.yaml. |
| 18 | Agent reads own docs | §4.10 (read-only Docker mount) |
| 19 | memory_co_access table | §3.1 (Hebbian co-access), §9 (schema) |

---

## 14. Task Summary

| Tier | Tasks | New in v3 | Effort |
|------|-------|-----------|--------|
| Prerequisite | 1 | port fix added | ~20 min |
| Tier 1: Foundation | 2.1–2.11 (11 tasks) | 2.3 rewritten for unified model | Medium |
| Tier 2: Core Cognitive | 3.1–3.9 (9 tasks) | 3.1 (Hybrid+Dirichlet), 3.2 (Dynamic Injection), 3.4 (Attention-Agnostic) NEW | Medium-High |
| Tier 3: Consolidation & DMN | 4.1–4.10 (10 tasks) | 4.1 (Constant), 4.9 (Restart), 4.10 (Docs) NEW | High |
| Tier 4: Novel | 5.1–5.3 (3 tasks) | Same scope, updated for unified model | High |
| **Total** | **34 tasks** | **+6 from v2's 28** | |

---

## 15. Open Questions Remaining

1. **Performance of scoring all memories per cycle.** Pre-filter needed: pgvector top-500 by embedding similarity, then full hybrid scoring on subset. Design decision at implementation.

2. **Dirichlet alpha initialization.** Starting at [8, 5, 0.5, 3, 2] or all-equal? Educated guesses vs learn from scratch. Resolve at implementation.

3. **Co-access matrix scaling.** O(N²) worst case. Pruning strategy needed: only track above threshold, decay old associations. Design at implementation.

4. **Outcome quality signal for Dirichlet meta-learning.** What = "good outcome"? User engagement? Memory re-access? Agent self-rating? Design at implementation.

5. **Cumulative importance trigger for consolidation** (SESSION_HANDOFF #10). How tracked? Running sum reset after each deep cycle? Resolve at implementation.
