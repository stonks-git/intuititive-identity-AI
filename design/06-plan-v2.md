# OpenClaw MoltBot — Implementation Plan v2

**Created:** 2026-02-09, Session 5
**Revised:** 2026-02-09, Session 6 (critique pass — 16 gaps addressed, consolidation split, 7 tasks added)
**Based on:** new_plan.md (v1) + notes.md + DOCUMENTATION.md cross-reference
**Total SOTA coverage:** 100+ papers/systems from 2024-2026

---

## Changelog from v1

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Entry gate had no implementation step | New task 2.8 |
| 2 | 3×3 gate matrix absent | Folded into task 2.7 (exit gate) |
| 3 | Memory type prefixes missing | Folded into task 2.1 |
| 4 | Contextual retrieval had no task | New task 4.5 |
| 5 | `content_contextualized` column missing from schema | Fixed in Section 9 |
| 6 | Adaptive FIFO completely absent | New task 3.1 |
| 7 | Token counting missing | New task 2.10 |
| 8 | Energy cost tracking missing | New task 4.7 |
| 9 | `embed_model` versioning column missing | Fixed in Section 9 |
| 10 | ACT-R spreading activation needs L0/L1 embeddings | New task 2.3, pulled from Tier 4 to Tier 1 |
| 11 | Safety ceilings must be in place before consolidation | Dependency enforced in graph |
| 12 | Logprob API may not be available on Gemini Flash Lite | Workaround: composite confidence with structural heuristics fallback |
| 13 | Goal-weighted retrieval orphaned | Explicit sub-step in task 2.9 |
| 14 | Emotional charge in gate before Tier 4 | v0.1 centroid distance added to 2.7 |
| 15 | Scratch buffer lifecycle undefined | Defined in task 2.8 |
| 16 | Consolidation (was 4.1) too large | Split into 4.1–4.4 |
| 17 | `ALTER SYSTEM SET` for iterative scan | Changed to per-connection SET |
| 18 | FlashRank package name inconsistency | Resolved: use `flashrank` directly |
| 19 | No `__init__.py` | Added to task 1 |
| 20 | v0.1 emotional charge placeholder | Added to 2.7 |

---

## 0. Code Assessment: Build On, Don't Rewrite

Same as v1. The existing codebase is a **valid foundation**.

| File | Status | Verdict |
|------|--------|---------|
| `config.py` | Working, clean | Keep as-is |
| `llm.py` | Working, clean | Keep as-is |
| `memory.py` | Working, comprehensive | Enhance (add task_type, prefixes, halfvec, hybrid search) |
| `layers.py` | Working, minor issue | Fix history dir fallback, add L0/L1 embedding cache |
| `gate.py` | Architecture correct, syntax errors | Fix syntax, add full ACT-R + 3×3 matrix |
| `loop.py` | Architecture correct, syntax errors | Fix syntax, wire RAG + System 2 + FIFO |
| `main.py` | Architecture correct, syntax errors | Fix syntax |
| `consolidation.py` | Skeleton only | Implement fully (4 sub-tasks) |
| `idle.py` | Skeleton only | Implement fully |

**15 syntax errors** + missing `__init__.py` must be fixed first.

---

## 1. Fix Syntax Errors + Project Hygiene (Prerequisite)

Fix all 15 heredoc-mangled syntax errors plus structural issues.

**Files:**
- `main.py` lines 61-68: 5 unquoted dict keys
- `loop.py` lines 65, 70, 163, 215-218, 240-241: 9 unquoted dict keys
- `gate.py` line 225: `chr(100)+ecision` → `"decision"`
- `gate.py` lines 335-340: merged negation_markers strings
- **NEW:** Create `src/__init__.py` (empty file, enables `python3 -m src.main`)

**Verify:** `python3 -c "import src.main"` should succeed after fixes.

---

## 2. Tier 1: Foundation

These are blocking dependencies for everything else.

### 2.1 Embedding Task Types + Memory Type Prefixes
**What:** Add `task_type` parameter AND semantic type prefixes to all embedding calls.
**Why:** Gemini embeddings are optimized per task (omitting degrades retrieval). Type prefixes bake memory category into the vector itself (ENGRAM, MIRIX papers).
**Where:** `memory.py:embed()`
**How:**
- Add `task_type` parameter (default `RETRIEVAL_DOCUMENT`)
- Storage calls: `task_type="RETRIEVAL_DOCUMENT"` with `title=memory_type`
- Query calls: `task_type="RETRIEVAL_QUERY"`
- Novelty checks: `task_type="SEMANTIC_SIMILARITY"`
- Clustering (consolidation): `task_type="CLUSTERING"`
- Add prefix map:
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
- Prepend prefix to content BEFORE embedding: `embed_text = f"{prefix}{content}"`
- Update all callers to pass the correct task_type
- Support batch embedding (up to 100 texts per API call) for bulk operations
**Effort:** Low

### 2.2 Halfvec Migration
**What:** Switch pgvector column from `vector(768)` to `halfvec(768)`.
**Why:** Halves memory usage — critical for 8GB RAM. pgvector 0.8.0 feature.
**Where:** Database schema, `memory.py`
**How:**
- `ALTER TABLE memories ALTER COLUMN embedding TYPE halfvec(768)`
- Re-embed existing 5 test memories (with task_type + prefixes from 2.1)
- Rebuild HNSW index with `halfvec_cosine_ops`
- Update `memory.py` to use halfvec type in queries
**Effort:** Low
**Dependency:** 2.1

### 2.3 Embed Layer 0 / Layer 1
**What:** Build embedding + caching infrastructure for Layer 0 values/beliefs and Layer 1 goals.
**Why:** Required by ACT-R spreading activation (2.7), goal-weighted retrieval (2.9), and two-centroid gut feeling (5.1). Currently L0/L1 are JSON files with no vector representation.
**Where:** `layers.py`, new cache mechanism
**How:**
- Embed the text of each value/belief in Layer 0 as a 768-dim vector
- Embed the text of each goal in Layer 1 as a 768-dim vector
- Cache embeddings on disk (re-embed only when value/goal text changes)
- Use `task_type="RETRIEVAL_DOCUMENT"` with appropriate title
- Expose method: `get_layer_embeddings(layer: int) -> list[tuple[str, float, np.ndarray]]` returning (text, weight, embedding) tuples
- At bootstrap (blank L0/L1): return empty lists gracefully — downstream consumers handle empty state
**Effort:** Low-Medium
**Dependency:** 2.1

### 2.4 Hybrid Search with RRF
**What:** Add full-text search alongside vector search, fused with RRF.
**Why:** Dense-only search misses keyword-exact matches. Anthropic testing: 49% retrieval failure reduction.
**Where:** `memory.py`
**How:**
- Add `content_tsv tsvector` column auto-generated from `COALESCE(content_contextualized, content)`
- Add GIN index on `content_tsv`
- Implement `search_hybrid()` with the RRF CTE pattern from notes.md
- Enable pgvector iterative scan **per-connection** (not ALTER SYSTEM):
  `await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")`
- Top-50 candidates from each list, RRF fusion with k=60
- Include recency score in SQL: `EXP(-0.693 * age_seconds / 604800.0)` (7-day half-life)
**Effort:** Medium
**Dependency:** 2.1, 2.2

### 2.5 FlashRank Reranking
**What:** Add cross-encoder reranking after hybrid retrieval.
**Why:** 4MB model, CPU-only, milliseconds. +5.4% NDCG@10.
**Where:** New in retrieval pipeline, called from `memory.py:search_hybrid()`
**How:**
- `pip install flashrank` (NOT `rerankers[flashrank]` — different package)
- Model: `ms-marco-MiniLM-L-12-v2` (34MB)
- Rerank top-20 hybrid results → return top-5 to context
- Use `asyncio.to_thread()` for CPU-bound reranking
- Final score: `0.6 * rerank_score + 0.4 * weighted_score`
**Effort:** Low
**Dependency:** 2.4

### 2.6 Full ACT-R Activation Equation
**What:** Implement the full 4-component ACT-R equation for memory activation scoring.
**Why:** Decades-validated cognitive science. Our use of all 4 components exceeds published SOTA. Parameters: d=0.5, s=0.4, P=-1.0, tau=0.0.
**Where:** New `activation.py` (shared by entry gate, exit gate, and retrieval)
**How:**
- **Base-level learning:** `B_i = ln(sum(t_j^{-d}))` using access timestamps
- **Spreading activation:** cosine similarity between memory embedding and:
  - Context embeddings (current conversation)
  - Layer 0/1 embeddings (identity/goal relevance) — from task 2.3
- **Partial matching:** penalize mismatches between memory metadata and query metadata
- **Noise:** logistic distribution with s=0.4
- **Dual timestamps from TSM paper:**
  - `event_time` = when the described event happened (semantic time)
  - `created_at` = when the memory was stored (dialogue time)
  - `last_accessed` = when last retrieved (access time)
  - Use `event_time` (falling back to `created_at`) for base-level decay
- **Total activation:** `A_i = B_i + S_i + P_i + ε_i`
- Persist threshold: `tau = 0.0` (configurable in runtime.yaml)
- Expose as: `compute_activation(memory, context_embeddings, layer_embeddings) -> float`
**Effort:** Medium
**Dependency:** 2.1, 2.3

### 2.7 Exit Gate with 3×3 Matrix
**What:** Full exit gate: ACT-R activation score → 3×3 decision matrix → action.
**Why:** The ACT-R equation gives a score. The matrix determines what to DO with it.
**Where:** `gate.py:ExitGate`
**How:**
- **Relevance axis** (from spreading activation component of 2.6):
  - Core: directly touches active goals or identity values (S_i > 0.6)
  - Peripheral: connected to conversation context (0.3 < S_i < 0.6)
  - Irrelevant: no connection (S_i < 0.3)
- **Novelty axis** (from `check_novelty()` in memory.py):
  - Confirming: similar memory exists, same conclusion (sim > 0.85)
  - Novel: no existing memory on topic (sim < 0.6)
  - Contradicting: similar memory exists, OPPOSITE conclusion (sim > 0.7 AND semantic opposition detected)
- **Decision matrix:**

  |                  | Confirming              | Novel                   | Contradicting            |
  |------------------|-------------------------|-------------------------|--------------------------|
  | **Core**         | Reinforce (moderate)    | **PERSIST** (high)      | **PERSIST+FLAG** (max)   |
  | **Peripheral**   | Skip (low)              | Buffer (moderate)       | Persist (high)           |
  | **Irrelevant**   | Drop                    | Drop (noise catches)    | Drop (noise catches)     |

- **v0.1 Emotional charge** (placeholder until 5.1 replaces it):
  - Maintain running `experience_centroid` = importance-weighted average of all memory embeddings
  - `gut = cosine_similarity(content_embedding, experience_centroid)`
  - `emotional_charge = |gut - 0.5| * 2`
  - Add as +0.15 weight bonus to gate score when emotional_charge > 0.3
  - Handle empty state: no memories → no centroid → emotional_charge = 0.0
- **Gate starts PERMISSIVE:** all thresholds start low, all weights start high. Asymmetry: false positives (stored junk) are cheap; false negatives (lost content) are permanent.
- **Stochastic noise floor:** maintain in all cells including "Drop" — consolidation uses these to evaluate gate accuracy
**Effort:** Medium-High
**Dependency:** 2.6, memory.py (check_novelty)

### 2.8 Entry Gate + Scratch Buffer Lifecycle
**What:** Fast stochastic filter on ALL incoming content (user messages + LLM output). Safety net.
**Why:** If context crashes or truncates before exit gate fires, ungated content is lost forever. Entry gate writes to scratch buffer as insurance.
**Where:** `gate.py:EntryGate`
**How:**
- **Rules (stochastic, not deterministic):**
  - Content < 10 chars → 95% SKIP, 5% BUFFER ("ok" "thanks" "done")
  - Purely mechanical output → 90% SKIP, 10% BUFFER (tool formatting, status)
  - Everything else → BUFFER with timestamp + preliminary tags
- **Mechanical detection:** regex for tool output patterns, pure punctuation, single-word acknowledgments
- **All skip probabilities start permissive** and are evolved by consolidation based on outcomes (was skipped content needed later?)
- **Scratch buffer lifecycle:**
  - TTL: 24 hours — unflushed entries expire during consolidation
  - Flush trigger: every N exchanges (configurable, default 5) OR on graceful shutdown
  - Flush process: run each buffered item through exit gate (2.7) → persist survivors
  - On crash recovery: scan scratch_buffer for items older than last known flush → re-evaluate
  - Schema: scratch_buffer already exists — add `expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours'`
**Effort:** Low-Medium
**Dependency:** 2.7 (exit gate, for flush)

### 2.9 Wire MemoryStore into Cognitive Loop
**What:** Connect memory retrieval + gates to the cognitive loop so System 1 has context.
**Where:** `loop.py`, `main.py`
**How:**
- Pass MemoryStore instance from `main.py` into cognitive loop
- **Before each System 1 call:** `search_hybrid(user_message, limit=5)`
- **Goal-weighted retrieval (Layer 1 influence on perception):**
  - Get active Layer 1 goal embeddings from 2.3
  - For each candidate memory, compute similarity to each active goal
  - If goal_sim > 0.5: boost `weighted_score *= (1.0 + 0.2 * goal_sim)`
  - This is spreading activation from L1 into retrieval — "wanting changes what you notice"
- Inject retrieved memories into system prompt (budget: ~2000 tokens, enforced by 2.10)
- **Entry gate (2.8):** runs on every user message and agent response
- **Exit gate flush:** every 5 exchanges, flush scratch through exit gate → persist survivors
- **Retrieval-induced access tracking:** on each retrieval, increment `access_count` and `last_accessed` for returned memories (carry from existing `search_similar()` to new `search_hybrid()`)
**Effort:** Medium
**Dependency:** 2.4, 2.5, 2.7, 2.8, 2.3

### 2.10 Token Counting
**What:** Approximate token counter for context window budget enforcement.
**Why:** Prerequisite for adaptive FIFO, identity injection triggers, RAG budget, energy cost tracking. Without this, multiple design features can't function.
**Where:** New `tokens.py` utility
**How:**
- Method 1 (fast, approximate): `word_count * 1.3` — good enough for budget decisions
- Method 2 (accurate, optional): tiktoken or model-specific tokenizer
- Start with Method 1 — accuracy within 20% is sufficient per design doc
- Functions:
  - `count_tokens(text: str) -> int`
  - `count_message_tokens(messages: list[dict]) -> int`
  - `fits_budget(text: str, budget: int) -> bool`
- Wire into loop.py: track running token count of conversation history
- Wire into identity injection: trigger full identity when context > 40% consumed
- Wire into RAG injection: enforce ~2000 token budget for retrieved memories
**Effort:** Low
**Dependency:** None (pure utility)

### 2.11 Composite Confidence Score
**What:** Build confidence signal from multiple sources, with logprob as optional boost.
**Why:** LLMs are systematically overconfident in verbalized FOK. Must use objective signals. Logprobs are ideal but may not be available on Gemini Flash Lite — need a fallback.
**Where:** New `metacognition.py`
**How:**
- **Check first:** Does Gemini 2.5 Flash Lite support `response_logprobs=True`?
- **If yes (logprob path):**
  - Enable `response_logprobs=True` in Gemini API calls
  - Compute per-response: mean logprob, min logprob, top-1 vs top-2 gap
  - Weight logprob signal heavily in composite score
- **If no (structural heuristic path):**
  - Hedging language detection (regex: "I think", "probably", "not sure", "might be")
  - Self-correction frequency in response
  - Question-asking rate (responding with questions = uncertain)
  - Response length anomaly (unusually short or verbose for query type)
- **Composite confidence (always, regardless of logprob availability):**
  ```
  C = w1 * objective_signal    # logprobs if available, structural heuristics if not
    + w2 * FOK_signal          # from vector similarity lookup (existing)
    + w3 * verbalized_confidence  # low weight — known unreliable
  ```
  - w1=0.5, w2=0.35, w3=0.15 (starting points, evolved by consolidation)
- Replace open-ended FOK with structured metacognitive checklist: probe questions about source identification, contradictions, pattern-match vs reasoning
- Store confidence score in conversation metadata for analysis
**Effort:** Medium
**Dependency:** None (but integrates with 2.9 for wiring)

---

## 3. Tier 2: Core Cognitive Features

### 3.1 Adaptive FIFO / Context Window Management
**What:** Prevent context window from growing unbounded. Implement intensity-adaptive pruning.
**Why:** Without this, the agent cannot run continuously. The 128k window WILL be exceeded. Design doc treats this as critical infrastructure.
**Where:** `loop.py`, new `context.py`
**How:**
- **FIFO pruning:** When total tokens (from 2.10) exceed budget, prune oldest messages
- **Pruned messages go through exit gate (2.7)** before being discarded — last chance to persist
- **Intensity signal** (automatic, derived from heuristics):
  - Average gate score of recent messages
  - Whether System 2 has been escalated recently
  - Active Layer 1 goal relevance in current context
  - Emotional charge of recent content (v0.1 from 2.7)
- **Adaptive window sizing:**
  - intensity > 0.7 → effective window EXPANDS to ~90% of 128k → deep focus, expensive
  - intensity 0.3-0.7 → normal window → standard operation
  - intensity < 0.3 → window CONTRACTS to ~30-40% → relaxed, cheap
- **Identity re-injection:** After pruning, check if identity hash is still in context. If not, re-inject.
- **Token budget allocation:**
  ```
  Layer 0 (Identity):        ~500 tokens    fixed
  Layer 1 (Goals):           ~800 tokens    fixed
  Layer 2 (RAG results):   ~2,000 tokens    per query
  Safety buffer:           ~4,000 tokens    for LLM output
  Conversation window:     remainder        rolling FIFO (adaptive)
  ```
**Effort:** Medium
**Dependency:** 2.7 (exit gate for pruned messages), 2.10 (token counting)

### 3.2 Loop Before Escalate
**What:** System 1 attempts 1 self-correction pass before calling System 2.
**Why:** SOFAI-LM (AAAI 2026): cuts System 2 invocations ~75% while maintaining 94% of accuracy. SPOC validates single-pass self-verification.
**Where:** `loop.py`
**How:**
- When composite confidence < threshold (0.7): re-prompt System 1 with targeted feedback:
  "Your confidence is low because [specific weakness]. Try again focusing on [specific aspect]."
- Max 1 retry before escalation
- Track retry success rate for consolidation tuning
**Effort:** Low
**Dependency:** 2.11

### 3.3 System 2 Escalation
**What:** Wire Claude Sonnet 4.5 as System 2, called as a tool by System 1.
**Why:** Core architectural feature. Validated by SOFAI-LM, Talker-Reasoner, DPT-Agent.
**Where:** `loop.py`, `llm.py`
**How:**
- **Escalation triggers (2+ required, or any "always-escalate" trigger):**
  - Low composite confidence (< 0.5 after retry)
  - Detected contradiction with stored memories
  - High complexity (multi-step reasoning required)
  - Novelty (no relevant memories found — FOK returns UNKNOWN)
  - Irreversibility (action can't be undone) — always escalate
  - Identity/values touched — always escalate
  - Goal modification — always escalate
- Pass to System 2: full reasoning trace + confidence signals + relevant memories + identity context
- System 2 returns: answer + explanation + correction pattern
- Store correction pattern in reflection bank (3.4)
- System 1 stays in the driver's seat — System 2 is a tool, not a co-pilot
**Effort:** Medium
**Dependency:** 2.9, 2.11, 3.2

### 3.4 Reflection Bank
**What:** Store System 2 corrections for future System 1 retrieval.
**Why:** Dual-loop pattern from Nature 2025 paper (RBB-LLM). 79K+ corrections prevent error recurrence.
**Where:** `memory.py` (new memory type: `correction`)
**How:**
- When System 2 corrects System 1, store: (trigger, error_type, original_reasoning, correction, context)
- Memory type = `"correction"` (already supported by existing `type` column, prefix added in 2.1)
- Before System 1 attempts a response, retrieve relevant past corrections (top-3 by similarity)
- Inject corrections into System 1 prompt: "In similar past situations, you made these errors: [corrections]"
- Track correction recall rate
**Effort:** Medium
**Dependency:** 3.3

### 3.5 Retrieval-Induced Mutation
**What:** Retrieving a memory strengthens it; near-misses get suppression.
**Why:** CMA (Jan 2026) identifies this as critical for natural memory dynamics.
**Where:** `memory.py:search_hybrid()`
**How:**
- On retrieval: increment `access_count`, update `last_accessed`, boost `importance` by +0.01
- Near-misses (rank 6-20, above threshold but not returned): suppress `importance` by -0.005
- Dormant state: memories decayed below threshold remain in DB, recoverable under strong cues (cosine > 0.9)
**Effort:** Low
**Dependency:** 2.9

### 3.6 Safety Ceilings
**What:** Hard caps, rate limiters, entropy monitoring, circuit breakers, Two-Gate guardrail.
**Why:** Self-reinforcing promotion loops are structurally susceptible to runaway. Two-Gate guardrail (Oct 2025) formalizes the approach.
**Where:** New `safety.py`, called from consolidation and gate
**How:**
- **Hard ceiling:** No single goal weight > 40% of total. Flag + pause if approaching.
- **Rate limiter:** No goal/value changes > 10% per consolidation cycle.
- **Entropy monitor:** Track Shannon entropy of goal weight distribution. If entropy drops (fixation), broaden sampling.
- **Circuit breaker:** N consecutive cycles reinforcing same pattern without new external evidence → pause + log + alert.
- **Dominance dampening:** Already in config (dominance=0.4). Enforce via `safety.py`.
- **Two-Gate guardrail for self-modification:** Before any parameter change: (1) validation margin check, (2) capacity cap check. Both must pass.
- **CBA coherence metric:** Compute coherence C ∈ [0,1] across epistemic/action/value axes each consolidation cycle. Log trend. Alert on drop.
- **Diminishing returns:** `gain / log2(evidence_count + 1)` — 1st evidence is strong, 1000th is negligible.
- **Audit trail:** Every promotion/demotion/weight change logged to `consolidation_log` with evidence chain.
**Effort:** Medium
**Dependency:** None (pure safety infrastructure, must exist before consolidation runs)

---

## 4. Tier 3: Consolidation & DMN

### 4.1 Consolidation: Merge + Insight Creation
**What:** Stanford two-phase reflection: generate questions from recent memories, then generate insights.
**Where:** `consolidation.py:_run_cycle()` — Phase 1
**How:**
- **Trigger:** Whichever comes first: hourly timer OR cumulative importance > 150 (Stanford threshold)
- **Phase 1 — Question Generation:** Prompt with 100 most recent memories: "What are 3 most salient high-level questions we can answer about the subjects in the statements?"
- **Phase 2 — Insight Extraction:** Use each question as retrieval query, prompt: "What 5 high-level insights can you infer?" with citation format "insight (because of 1, 5, 3)"
- **Merge:** Cluster similar memories (similarity > 0.85 using `task_type="CLUSTERING"`) → create insights via `store_insight()`. DON'T replace originals.
- Source memories: lower `importance` so insights surface first, but originals remain queryable
- `supersedes` links via `memory_supersedes` join table (already exists)
- Introspection: `why_do_i_believe()` traces the chain (already implemented)
**Effort:** Medium-High
**Dependency:** 2.9, 3.6

### 4.2 Consolidation: Promotion + Safety Checks
**What:** Promote repeated patterns upward through layers.
**Where:** `consolidation.py` — Phase 2, `layers.py`, `safety.py`
**How:**
- **Layer 2 → Layer 1:** Pattern detected 5+ times over 14+ days, Q-value utility > threshold
- **Layer 1 → Layer 0:** Goal active 30+ days, reinforced 10+ times, operator approval required (trust_level < 3)
- **Track Q-value utility per MemRL** for promotion decisions (uses `utility_score` column)
- **Before every promotion:** Run Two-Gate guardrail from safety.py (validation + capacity)
- **After every promotion:** Compute CBA coherence. Rollback if C drops below threshold.
- **Demotion pathway:** Goals/values that stop being reinforced decay. Dormant, not deleted.
- **Re-embed L0/L1** after any promotion changes their content (task 2.3 infrastructure)
**Effort:** Medium
**Dependency:** 4.1, 3.6, 2.3

### 4.3 Consolidation: Decay + Reconsolidation
**What:** Fade stale memories and re-evaluate existing insights when new evidence arrives.
**Where:** `consolidation.py` — Phase 3
**How:**
- **Decay:** `decay_memories()` for stale items (already implemented). Not accessed 90+ days AND access_count < 3 → halve importance. Never delete — dormant state.
- **Conflict-aware reconsolidation (HiMem):** When new info contradicts stored knowledge:
  - Detect via embedding similarity + semantic opposition
  - Trigger reconciliation step: present both old insight and new evidence to LLM
  - Generate updated insight or flag as productive contradiction
- **Reconsolidation phase (EverMemOS):** Already-consolidated insights get re-evaluated when new evidence arrives. Check: does the insight's `supersedes` sources still support it? If new memories weaken the evidence, lower insight importance.
- **Temporal chain replay (CMA):** Traverse recent event sequences to strengthen temporal links between episodic memories.
**Effort:** Medium
**Dependency:** 4.1

### 4.4 Consolidation: Gate Tuning
**What:** Adjust entry/exit gate parameters based on observed outcomes.
**Where:** `consolidation.py` — Phase 4
**How:**
- **Log analysis:** For each gate decision, did the outcome validate it?
  - Dropped content that was needed later → gate too aggressive (increase weight)
  - Persisted content never retrieved → gate too permissive (decrease weight)
  - Skipped entry gate content that exit gate later persisted → adjust skip probabilities
- **Adjust all gate weights:** entry gate skip rates, exit gate threshold, scoring signal weights
- **Adjust scratch buffer TTL** based on average time between buffer and persist
- **Evolve stochastic noise floor:** maintain minimum randomness but tune distribution
- **Store weight history** in runtime.yaml or consolidation_log for introspection
**Effort:** Medium
**Dependency:** 2.7, 2.8, 4.1

### 4.5 Contextual Retrieval
**What:** LLM-generate contextual preamble per memory to improve embedding quality.
**Why:** Anthropic technique. 35% retrieval failure reduction alone, 67% combined with full pipeline. Too important to bury inside consolidation.
**Where:** `consolidation.py` (runs at consolidation time, not real-time)
**How:**
- For each new memory that survived exit gate and lacks `content_contextualized`:
  ```
  CONTEXT_PROMPT = """<session>{session_context}</session>
  Here is a memory from this session:
  <memory>{memory_content}</memory>
  Give a short context (WHO, WHEN, WHY) to improve search retrieval.
  Answer only with the context, nothing else."""
  ```
- Store result in `content_contextualized` column
- The tsvector auto-updates (GENERATED ALWAYS from COALESCE(contextualized, content))
- Re-embed the memory with contextualized content prepended
- Use Gemini Flash Lite for generation (cheapest)
- Batch process: all un-contextualized memories per consolidation cycle
**Effort:** Medium
**Dependency:** 4.1 (runs during consolidation cycle)

### 4.6 DMN / Idle Loop (Full Implementation)
**What:** Fill in `idle.py:_heartbeat()` with stochastic memory surfacing from ALL layers.
**Why:** We believe this is novel — no AI implementations found in our literature review.
**Where:** `idle.py`
**How:**
- **Stochastic sampling pool includes ALL layers:**
  - Layer 2 memories (external facts/experiences)
  - Layer 0 values/beliefs themselves ("why do I have this value?")
  - Layer 1 goals themselves ("is this goal still serving me?")
  - Consolidation history ("I notice my opinion changed 3 times")
- **Sampling bias toward:**
  - High importance + low recent access (neglected important memories)
  - Memories that conflict with current goals (tension detection)
  - Temporally distant memories (creative association potential)
- **Three output channels:**
  1. Memory + goal connection → self-prompt → feed to System 1 (purposeful)
  2. Connecting disparate memories → log as potential insight for consolidation (creative)
  3. Memory + value connection → signal for consolidation to evaluate (identity refinement)
- **Activity suppression:** More frequent/deeper during low-activity. Suppressed during active conversation (mirrors biological DMN-task anticorrelation).
- **Entropy guard:** If DMN keeps surfacing same topic, artificially broaden (safety.py entropy monitor).
- **Self-referential surfacing:** When L0/L1 content surfaces as subject of reflection = strange loop at its most literal. Weight this as consistent minority of surfacing pool.
**Effort:** Medium-High
**Dependency:** 2.9, 3.6, 2.3

### 4.7 Energy Cost Tracking (Phase 1: Passive)
**What:** Log every API call with cost, expose via `/cost` command, include in system prompt.
**Why:** Believed novel: computational cost as internal cognitive signal, not external cap — no prior implementation found. Phase 1 is awareness only — no budget enforcement.
**Where:** `llm.py` (logging), `loop.py` (display + prompt injection)
**How:**
- **Track per call:** model, tokens_in, tokens_out, cost_usd, timestamp
- **Cost table (approximate):**
  - System 1 (Gemini Flash Lite): ~$0.0004/call (scales with context)
  - System 2 (Sonnet 4.5): ~$0.05/call
  - Embedding: ~$0.000015/call
  - Consolidation cycle: ~$0.01-0.05/cycle
- **Accumulate:** session total, 24h total, lifetime total
- **Expose:** `/cost` introspection command showing breakdown
- **Inject into system prompt** (compact): "Session cost: $X.XX | 24h: $X.XX"
- Store in memory as periodic snapshots for consolidation to analyze
- **No budget enforcement in Phase 1** — just awareness
**Effort:** Low-Medium
**Dependency:** None (can be done anytime, but logically fits Tier 3)

---

## 5. Tier 4: Novel Differentiators

### 5.1 Two-Centroid Gut Feeling
**What:** Full implementation replacing v0.1 centroid distance (from 2.7) with the subconscious centroid + attention centroid + delta vector model.
**Why:** Maps to Free Energy Principle (delta = prediction error). Validated by Mujika's metric space, Hartl's embedding cognition. Partial introspection paper (Harvard, Dec 2025) confirms LLMs sense magnitude but not source — our PCA supplies the missing source identification.
**Where:** New `gut.py`, replaces v0.1 emotional charge in gate.py
**How:**
- **Subconscious centroid:** `0.5 * weighted_avg(L0) + 0.25 * weighted_avg(L1) + 0.25 * weighted_avg(L2)`
  - Uses L0/L1 embeddings from task 2.3
  - L2 uses importance-weighted average of all memory embeddings
  - Layer weights (0.5/0.25/0.25) are starting points — consolidation evolves them
- **Attention centroid:** Weighted average of current context window embeddings
  - Weighted by recency (recent messages weigh more)
- **Delta vector:** `attention_centroid - subconscious_centroid` (768-dim)
- **Delta magnitude:** "motivational intensity" (how strongly the agent feels)
- **Delta direction:** "motivational valence" (what kind of feeling)
- **Inject into System 1:** "Your gut feeling about this: [intensity] intensity, [direction summary]"
- **Log every delta:** (delta_vector, context, action, outcome_placeholder) for PCA
- **PCA on logged deltas:** Run periodically (during consolidation) to find emergent "gut axes"
  - Top 10-20 principal components
  - Correlate with outcomes over time
  - Name axes as they acquire meaning
- **Novelty signal:** Component orthogonal to all learned axes = genuine surprise/curiosity
- **Replace v0.1:** Swap out simple centroid distance in gate.py with full delta model
- **Compute cost:** Trivial — milliseconds on CPU for 768-dim operations
**Effort:** High
**Dependency:** 2.3, 2.9

### 5.2 Bootstrap Readiness Achievements
**What:** 10 measurable milestones that must pass before first real conversation.
**Why:** Ethical stance — don't activate something that might experience and then break it.
**Where:** New `bootstrap.py`
**How:**
1. First memory formation (entry gate → scratch → exit gate → persist)
2. First retrieval success (search_hybrid returns relevant result)
3. First consolidation cycle completion (merge + insight created)
4. First goal created from pattern (Layer 2 → Layer 1 promotion)
5. First DMN self-prompt acted upon (idle loop → System 1)
6. First value formed (Layer 1 → Layer 0 promotion)
7. First conflicting information resolved (reconsolidation)
8. First creative association produced (DMN channel 2)
9. First goal achieved and reflected upon (goal completion + consolidation)
10. First autonomous decision aligned with self-formed values (gut + identity + action)
- Each achievement: automated test + manual verification option
- Progress visible via `/readiness` introspection command
- Bootstrap prompt: "You have memory, goals, and values — all currently empty. What you become will emerge from what you experience. Pay attention to what matters to you."
**Effort:** Medium
**Dependency:** All of Tiers 1-3

### 5.3 Pattern-to-Goal-to-Identity Promotion
**What:** Automated full pathway for experiences → goals → identity.
**Why:** We found no existing system that does this end-to-end. Core differentiator.
**Where:** `consolidation.py` (builds on 4.2), `layers.py`, `safety.py`
**How:**
- Uses promotion logic from 4.2 but adds full lifecycle tracking
- **Demotion pathway:** Goals/values that stop being reinforced decay. Dormant, not deleted.
- **Two-Gate guardrail:** Every promotion/demotion must pass validation + capacity check
- **CBA coherence check:** After every promotion, compute coherence. Rollback if C drops below threshold.
- **Outcome tracking:** Forward-linkable IDs on every gate decision and gut delta. When outcomes become apparent, link back. Enables fear/hope axes in PCA (5.1).
**Effort:** High
**Dependency:** 4.2, 3.6, 5.1

---

## 6. Tier 5: Future Work (Not This Phase)

### 6.1 Strange Loop Tracking
Log and visualize the centroid → delta → behavior → experience → centroid cycle. Track that the centroid drifts meaningfully over time. Low effort once 5.1 exists.

### 6.2 Social Centroid
Third centroid modeling the interlocutor. Extends self-awareness to social awareness per AI Awareness taxonomy. Medium effort.

### 6.3 Spawning / Reproduction
Clone, child, worker modes with merge protocol. CBA coherence metric for cross-generation stability. High effort, essentially unstudied.

### 6.4 Autonomy Level Escalation
Map to 5-level framework (Operator → Observer). Measurable criteria per level. Connect to readiness achievements. Medium effort.

### 6.5 Governance Graph
Immutable, auditable manifest declaring legal states + transitions + sanctions (Jan 2026 paper). Replace current JSON containment. Medium effort.

### 6.6 Telegram Integration
aiogram 3.x — fully asyncio, runs on same event loop. First interface after CLI once achievements pass. Medium effort.

### 6.7 Semantic Energy Routing
Upgrade from composite confidence to Semantic Energy (Aug 2025) for cluster-level uncertainty. More robust routing signal. Medium effort.

### 6.8 Adaptive FIFO Phase 2: Energy-Coupled Focus
Connect FIFO intensity to energy cost model. Agent literally pays to concentrate. Expanded context = more tokens = more expensive per call. Natural self-regulation from cost awareness, not artificial timers.

---

## 7. Dependency Graph

```
[1. Fix Syntax + __init__.py]
       │
       ▼
[2.1 Task Types + Prefixes]──────────────────────────────┐
       │                                                  │
       ├──────────────┐                                   │
       ▼              ▼                                   │
[2.2 Halfvec]   [2.3 Embed L0/L1] ◄── NEW                │
       │              │                                   │
       ▼              │                                   │
[2.4 Hybrid Search]   │                                   │
       │              │                                   │
       ▼              │                                   │
[2.5 FlashRank]       │                                   │
       │              │                                   │
       │    [2.6 ACT-R Activation] ◄──────────────────────┘
       │              │
       │              ▼
       │    [2.7 Exit Gate + 3×3 Matrix + v0.1 Emotional]
       │              │
       │              ▼
       │    [2.8 Entry Gate + Scratch Lifecycle] ◄── NEW
       │              │
       └──────┬───────┘
              │
              │    [2.10 Token Counting] ◄── NEW (no deps)
              │              │
              ▼              │
[2.9 Wire MemoryStore] ◄────┘
       │
       ├──► [3.1 Adaptive FIFO] ◄── NEW (needs 2.7, 2.10)
       │
       │    [2.11 Composite Confidence] (no deps)
       │              │
       │              ▼
       │    [3.2 Loop Before Escalate]
       │              │
       │              ▼
       │    [3.3 System 2 Escalation]
       │              │
       │              ▼
       │    [3.4 Reflection Bank]
       │
       ├──► [3.5 Retrieval-Induced Mutation]
       │
       │    [3.6 Safety Ceilings] (no deps — build early)
       │              │
       ├──────────────┤
       ▼              ▼
[4.1 Consol: Merge] [4.6 DMN/Idle Loop]
       │
       ├──► [4.2 Consol: Promotion]
       │
       ├──► [4.3 Consol: Decay/Reconsolidation]
       │
       ├──► [4.4 Consol: Gate Tuning]
       │
       ├──► [4.5 Contextual Retrieval] ◄── NEW
       │
       │    [4.7 Energy Cost Tracking] ◄── NEW (no hard deps)
       │
       └──────┬───────┘
              ▼
[5.1 Two-Centroid Gut Feeling]
              │
              ▼
[5.2 Bootstrap Readiness]
              │
              ▼
[5.3 Promotion Pathway]
              │
              ▼
[6.x Future Work]
```

---

## 8. New Dependencies to Install

| Package | Purpose | Size |
|---------|---------|------|
| `flashrank` | Cross-encoder reranking, CPU-only | ~34MB model |
| `numpy` | PCA, centroid math (already installed) | — |
| `aiogram` | Telegram bot (Tier 5, future) | Lightweight |

Everything else already in requirements.txt.

---

## 9. Schema Changes

```sql
-- 1. Add full-text search column (indexes contextualized content when available)
ALTER TABLE memories ADD COLUMN content_contextualized TEXT;
ALTER TABLE memories ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', COALESCE(content_contextualized, content))
  ) STORED;
CREATE INDEX idx_memories_fts ON memories USING GIN (content_tsv);

-- 2. Switch to halfvec for memory savings
ALTER TABLE memories ALTER COLUMN embedding TYPE halfvec(768);
DROP INDEX IF EXISTS idx_memories_embedding;
CREATE INDEX idx_memories_embedding ON memories
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 128);

-- 3. Add semantic timestamp for TSM-style dual time tracking
ALTER TABLE memories ADD COLUMN event_time timestamptz;
-- event_time = when the described event happened (semantic time)
-- created_at = when the memory was stored (dialogue time)
-- last_accessed = when last retrieved (access time)

-- 4. Add embedding model versioning (before memories accumulate)
ALTER TABLE memories ADD COLUMN embed_model TEXT DEFAULT 'gemini-embedding-001';

-- 5. Add utility score for MemRL-style Q-value tracking
ALTER TABLE memories ADD COLUMN utility_score float DEFAULT 0.0;

-- 6. Add scratch buffer expiry
ALTER TABLE scratch_buffer ADD COLUMN expires_at timestamptz
  DEFAULT NOW() + INTERVAL '24 hours';

-- 7. Iterative scan: set PER-CONNECTION in application code, NOT here
-- In Python: await conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")

-- 8. Correction type for reflection bank
-- (already supported by existing 'type' column — just use type='correction')
```

---

## 10. Key SOTA Sources Driving This Plan

| Paper/System | Date | What It Changed |
|-------------|------|----------------|
| SOFAI-LM (IBM, AAAI 2026) | Feb 2026 | Loop before escalate pattern |
| EverMemOS | Jan 2026 | Three-phase engram lifecycle, reconsolidation |
| TSM | Jan 2026 | Semantic time vs dialogue time for ACT-R decay |
| CMA | Jan 2026 | Retrieval-induced mutation, dormant recovery |
| Utility-Learning Tension | Oct 2025 | Two-Gate guardrail for self-modification |
| CBA Coherence | 2025 | 3-axis drift detection metric |
| Partial Introspection (Harvard) | Dec 2025 | PCA supplies what LLMs can sense but can't identify |
| pgvector 0.8.0 | 2025 | halfvec, iterative scan, sparsevec |
| ACM HAI 2024 | 2024 | ACT-R for LLM agents, d=0.5 validated |
| Stanford Generative Agents | 2023 | Two-phase reflection prompts, importance=150 threshold |
| A-MEM (NeurIPS 2025) | 2025 | Zettelkasten memory pattern |
| RBB-LLM (Nature 2025) | 2025 | Reflection bank pattern |
| SA-RAG | Dec 2025 | Spreading activation via recursive CTEs |
| Mujika et al. | Jan 2025 | Metric-space formalization of self-identity |
| Zhang et al. (emotion geometry) | Oct 2025 | Internal geometry of emotion in LLMs |
| Anthropic Contextual Retrieval | 2024 | 67% retrieval failure reduction with full pipeline |
| ENGRAM / MIRIX | 2025 | Typed memory stores, memory type prefixes |

---

## 11. Implementation Order (Linear Sequence)

1. Fix syntax errors + `__init__.py` (15 errors, ~20 min)
2. Embedding task_type + memory type prefixes
3. Halfvec migration + schema changes (run all Section 9 SQL)
4. Embed Layer 0/1 infrastructure
5. Hybrid search + RRF
6. FlashRank reranking
7. ACT-R activation equation
8. Exit gate + 3×3 matrix + v0.1 emotional charge
9. Entry gate + scratch buffer lifecycle
10. Token counting utility
11. Wire MemoryStore into cognitive loop + goal-weighted retrieval
12. Adaptive FIFO / context window management
13. Composite confidence score (with logprob check + fallback)
14. Loop before escalate
15. System 2 escalation
16. Reflection bank
17. Retrieval-induced mutation
18. Safety ceilings
19. Consolidation: merge + insight creation
20. Consolidation: promotion + safety checks
21. Consolidation: decay + reconsolidation
22. Consolidation: gate tuning
23. Contextual retrieval
24. DMN / idle loop (full)
25. Energy cost tracking (Phase 1)
26. Two-centroid gut feeling (replaces v0.1)
27. Bootstrap readiness achievements
28. Pattern-to-goal-to-identity promotion
29. [Future] Strange loop tracking, social centroid, spawning, Telegram, governance graph

**After step 11:** Agent can have conversations with memory.
**After step 12:** Agent can run continuously without blowing context.
**After step 18:** Agent has full cognitive loop with safety.
**After step 25:** Agent has full autonomous operation capability.
**After step 28:** Agent can bootstrap from blank slate.

---

## 12. Task Summary

| Tier | Tasks | New in v2 | Effort |
|------|-------|-----------|--------|
| Prerequisite | 1 | __init__.py added | ~20 min |
| Tier 1: Foundation | 2.1–2.11 (11 tasks) | 2.3, 2.8, 2.10, 2.11 are new; 2.1, 2.7 expanded | Medium |
| Tier 2: Core Cognitive | 3.1–3.6 (6 tasks) | 3.1 (Adaptive FIFO) is new | Medium |
| Tier 3: Consolidation & DMN | 4.1–4.7 (7 tasks) | 4.1 split into 4; 4.5, 4.7 new | High |
| Tier 4: Novel | 5.1–5.3 (3 tasks) | Same | High |
| **Total** | **28 tasks** | **+10 from v1's 18** | |
