# OpenClaw MoltBot — Finalized Implementation Plan

**Created:** 2026-02-09, Session 5
**Based on:** RESEARCH_REPORT.md (80+ papers, Session 4) + Session 5 supplementary sweep (20+ additional papers)
**Total SOTA coverage:** 100+ papers/systems from 2024-2026

---

## 0. Code Assessment: Build On, Don't Rewrite

The existing codebase is a **valid foundation**. Architecture matches the plan. What exists:

| File | Status | Verdict |
|------|--------|---------|
| `config.py` | Working, clean | Keep as-is |
| `llm.py` | Working, clean | Keep as-is |
| `memory.py` | Working, comprehensive | Enhance (add task_type, halfvec, hybrid search) |
| `layers.py` | Working, minor issue | Fix history dir fallback |
| `gate.py` | Architecture correct, syntax errors | Fix syntax, enhance ACT-R math |
| `loop.py` | Architecture correct, syntax errors | Fix syntax, wire RAG + System 2 |
| `main.py` | Architecture correct, syntax errors | Fix syntax |
| `consolidation.py` | Skeleton only | Implement fully |
| `idle.py` | Skeleton only | Implement fully |

**15 syntax errors** (shell heredoc mangled dict keys) must be fixed first. All are `dict[key]` → `dict["key"]` and one `chr(100)+ecision` → `"decision"`. ~20 minutes.

---

## 1. Fix Syntax Errors (Prerequisite)

Fix all 15 heredoc-mangled syntax errors before anything else.

**Files:**
- `main.py` lines 61-68: 5 unquoted dict keys
- `loop.py` lines 65, 70, 163, 215-218, 240-241: 9 unquoted dict keys
- `gate.py` line 225: `chr(100)+ecision` → `"decision"`
- `gate.py` lines 335-340: merged negation_markers strings

**Verify:** `python3 -c "import src.main"` should succeed after fixes.

---

## 2. Tier 1: Foundation (Do First)

These are blocking dependencies for everything else.

### 2.1 Embedding Task Types
**What:** Add `task_type` parameter to all embedding calls.
**Why:** Gemini embeddings are optimized per task. Omitting this degrades retrieval quality (Agent 5 finding).
**Where:** `memory.py:embed()`
**How:**
- Add `task_type` parameter (default `RETRIEVAL_DOCUMENT`)
- Storage calls use `task_type="RETRIEVAL_DOCUMENT"`
- Query calls use `task_type="RETRIEVAL_QUERY"`
- Update all callers to pass the correct type
**Effort:** Low

### 2.2 Halfvec Migration
**What:** Switch pgvector column from `vector(768)` to `halfvec(768)`.
**Why:** Halves memory usage — critical for 8GB RAM. pgvector 0.8.0 feature. ~50-100K vectors fit in HNSW index (Agent 5 finding).
**Where:** Database schema, `memory.py`
**How:**
- `ALTER TABLE memories ALTER COLUMN embedding TYPE halfvec(768)`
- Re-embed existing 5 test memories (with task_type)
- Rebuild HNSW index
- Update `memory.py` to use halfvec type in queries
**Effort:** Low
**Dependency:** 2.1

### 2.3 Hybrid Search with RRF
**What:** Add full-text search alongside vector search, fused with RRF.
**Why:** Dense-only search misses keyword-exact matches. RRF fusion is SOTA consensus across all research agents.
**Where:** `memory.py`
**How:**
- Add `content_tsv tsvector` column auto-generated from content
- Add GIN index on `content_tsv`
- Implement `search_hybrid()` with the RRF CTE pattern from RESEARCH_REPORT.md Section 3
- Enable pgvector 0.8.0 iterative scan: `SET hnsw.iterative_scan = 'relaxed_order'`
- Top-50 candidates, filter by similarity threshold (cosine > 0.7), then rerank
**Effort:** Medium
**Dependency:** 2.1, 2.2

### 2.4 FlashRank Reranking
**What:** Add cross-encoder reranking after hybrid retrieval.
**Why:** 4MB model, CPU-only, milliseconds. +5.4% NDCG@10. All 5 agents agree this is the right choice.
**Where:** New in retrieval pipeline, called from `memory.py:search_hybrid()`
**How:**
- `pip install "rerankers[flashrank]"`
- Rerank top-20 hybrid results → return top-5 to context
- Use `asyncio.to_thread()` for CPU-bound reranking to avoid blocking event loop
**Effort:** Low
**Dependency:** 2.3

### 2.5 Full ACT-R Activation Equation in Exit Gate
**What:** Replace current spreading-activation-only scoring with the full 4-component ACT-R equation.
**Why:** Our use of all 4 components (base-level + spreading + partial matching + noise) exceeds published SOTA. Validated parameters: d=0.5, s=0.4, P=-1.0, tau=0.0.
**Where:** `gate.py:ExitGate`
**How:**
- Base-level learning: `B_i = ln(sum(t_j^{-d}))` using access timestamps
- Spreading activation: cosine similarity between memory embedding and context embeddings (already partially implemented)
- Partial matching: penalize mismatches between memory metadata and query metadata
- Noise: logistic distribution with s=0.4 (already implemented)
- **NEW from TSM paper:** Track both `semantic_time` (event timestamp) and `access_time` (last retrieved). Use semantic_time for base-level decay. This prevents recently-retrieved old memories from appearing "fresh."
- Persist threshold: tau=0.0 (configurable)
**Effort:** Medium
**Dependency:** 2.1

### 2.6 Wire MemoryStore into Cognitive Loop
**What:** Connect memory retrieval to the cognitive loop so System 1 has context.
**Where:** `loop.py`
**How:**
- Pass MemoryStore instance from `main.py` into cognitive loop
- Before each System 1 call: `search_hybrid(user_message, limit=5)`
- Inject retrieved memories into system prompt (budget: ~2000 tokens)
- Entry gate buffers user messages + agent responses to scratch
- Every 5 exchanges: flush scratch through exit gate → persist survivors
- Implement retrieval-induced mutation: on each retrieval, increment `access_count` and `last_accessed` for returned memories (already in `search_similar()`, carry to `search_hybrid()`)
**Effort:** Medium
**Dependency:** 2.3, 2.4, 2.5

### 2.7 Logprob-Based Confidence
**What:** Enable Gemini `response_logprobs` and build composite confidence score.
**Why:** LLMs are systematically overconfident in verbalized FOK. Must use token logprobs. Confirmed by 3 independent papers (Epistemia, LogTokU, OpenReview metacognition study).
**Where:** `loop.py`, new `metacognition.py`
**How:**
- Enable `response_logprobs=True` in Gemini API calls
- Compute per-response: mean logprob, min logprob, top-1 vs top-2 gap
- Build composite confidence: `C = w1*logprob_signal + w2*structural_heuristics + w3*verbalized_confidence` (w3 low)
- Replace open-ended FOK with structured metacognitive checklist: specific probe questions about source identification, contradictions, pattern-match vs reasoning
- Store confidence score in conversation metadata for later analysis
**Effort:** Medium

---

## 3. Tier 2: Core Cognitive Features

### 3.1 Loop Before Escalate
**What:** System 1 attempts 1 self-correction pass before calling System 2.
**Why:** SOFAI-LM (AAAI 2026) shows this cuts System 2 invocations ~75% while maintaining 94% of System 2 accuracy. SPOC validates single-pass self-verification.
**Where:** `loop.py`
**How:**
- When confidence score < threshold (0.7): re-prompt System 1 with targeted feedback: "Your confidence is low because [specific weakness]. Try again focusing on [specific aspect]."
- Max 1 retry before escalation
- Track retry success rate for consolidation tuning
**Effort:** Low
**Dependency:** 2.7

### 3.2 System 2 Escalation
**What:** Wire Claude Sonnet 4.5 as System 2, called as a tool by System 1.
**Why:** Core architectural feature. Validated by SOFAI-LM, Talker-Reasoner, DPT-Agent.
**Where:** `loop.py`, `llm.py`
**How:**
- Escalation triggers (2+ required, or any "always-escalate" trigger):
  - Low composite confidence (< 0.5 after retry)
  - Detected contradiction with stored memories
  - High complexity (multi-step reasoning required)
  - Novelty (no relevant memories found)
  - Irreversibility (action can't be undone) — always escalate
  - Identity/values touched — always escalate
  - Goal modification — always escalate
- Pass to System 2: full reasoning trace + confidence signals + relevant memories
- System 2 returns: answer + explanation + correction pattern
- Store correction pattern in reflection bank (3.3)
**Effort:** Medium
**Dependency:** 2.6, 2.7, 3.1

### 3.3 Reflection Bank
**What:** Store System 2 corrections for future System 1 retrieval.
**Why:** Dual-loop pattern from Nature 2025 paper (RBB-LLM). 79K+ corrections prevent error recurrence. MemR3 validates retrieve/reflect/answer routing.
**Where:** `memory.py` (new memory type: `correction`)
**How:**
- When System 2 corrects System 1, store: (trigger, error_type, original_reasoning, correction, context)
- Before System 1 attempts a response, retrieve relevant past corrections (top-3 by similarity)
- Inject corrections into System 1 prompt: "In similar past situations, you made these errors: [corrections]"
- Track correction recall rate
**Effort:** Medium
**Dependency:** 3.2

### 3.4 Retrieval-Induced Mutation
**What:** Retrieving a memory strengthens it; near-misses get suppression.
**Why:** CMA (Jan 2026) identifies this as critical for natural memory dynamics. Prevents stale memories from cluttering retrieval.
**Where:** `memory.py:search_hybrid()`
**How:**
- On retrieval: increment `access_count`, update `last_accessed`, boost `importance` by small delta (+0.01)
- Near-misses (rank 6-20, above threshold but not returned): suppress `importance` by small delta (-0.005)
- Dormant state: memories decayed below threshold remain in DB, recoverable under strong cues (cosine > 0.9)
**Effort:** Low
**Dependency:** 2.6

### 3.5 Safety Ceilings
**What:** Hard caps, rate limiters, entropy monitoring, circuit breakers for goal weights.
**Why:** Self-reinforcing promotion loops are structurally susceptible to runaway. Anthropic reward hacking paper shows misalignment appears at exact inflection points. Two-Gate guardrail (Oct 2025) formalizes the approach.
**Where:** New `safety.py`, called from consolidation and gate
**How:**
- **Hard ceiling:** No single goal weight > 40% of total. Flag + pause if approaching.
- **Rate limiter:** No goal/value changes > 10% per consolidation cycle.
- **Entropy monitor:** Track Shannon entropy of goal weight distribution. If entropy drops (fixation), broaden sampling artificially.
- **Circuit breaker:** N consecutive cycles reinforcing same pattern without new external evidence → pause + log + alert.
- **Dominance dampening:** Already in config (dominance=0.4). Enforce.
- **Two-Gate guardrail for self-modification:** Before any parameter change: (1) validation margin check, (2) capacity cap check. Both must pass.
- **CBA coherence metric:** Compute coherence C ∈ [0,1] across epistemic/action/value axes each consolidation cycle. Log trend. Alert on drop.
- **Audit trail:** Every promotion/demotion/weight change logged to `consolidation_log` with evidence chain.
**Effort:** Medium

---

## 4. Tier 3: Consolidation & DMN

### 4.1 Consolidation Worker (Full Implementation)
**What:** Fill in `consolidation.py:_run_cycle()` with Stanford two-phase reflection + CMA mechanisms.
**Why:** This is the engine that transforms raw memories into insights, goals, and identity. Core differentiator.
**Where:** `consolidation.py`
**How:**
- **Trigger:** Whichever comes first: hourly timer OR cumulative importance > 150 (Stanford threshold)
- **Phase 1 — Question Generation:** Prompt with 100 most recent memories: "What are 3 most salient high-level questions we can answer about the subjects in the statements?"
- **Phase 2 — Insight Extraction:** Use each question as retrieval query, prompt: "What 5 high-level insights can you infer?" with citation format "insight (because of 1, 5, 3)"
- **Merge:** Cluster similar memories (similarity > 0.85) → create insights via `store_insight()`. DON'T replace originals.
- **Promote:** Repeated patterns up layers:
  - 5+ signals over 14+ days → propose as Layer 1 goal
  - 10+ over 30+ days → propose as Layer 0 value (with operator approval at trust_level < 3)
  - Track Q-value utility per MemRL for promotion decisions
- **Decay:** `decay_memories()` for stale items. Dormant state, never delete.
- **Conflict-aware reconsolidation (HiMem):** When new info contradicts stored knowledge, trigger reconciliation step rather than blind merge.
- **Temporal chain replay (CMA):** Traverse recent event sequences to strengthen temporal links.
- **Reconsolidation phase (EverMemOS):** Already-consolidated insights get re-evaluated when new evidence arrives.
- **Safety checks:** Run safety.py checks after every promotion/demotion.
- **Gate tuning:** Adjust entry/exit gate parameters based on false positive/negative rates.
- **MAGMA dual-stream:** Fast path already implemented (entry gate → scratch). Slow path = this consolidation worker.
**Effort:** High
**Dependency:** 2.6, 3.5

### 4.2 DMN / Idle Loop (Full Implementation)
**What:** Fill in `idle.py:_heartbeat()` with stochastic memory surfacing.
**Why:** We believe this is novel — no AI implementations found in our literature review. Significant differentiator if assessment holds.
**Where:** `idle.py`
**How:**
- **Stochastic sampling:** Bias toward:
  - High importance + low recent access (neglected important memories)
  - Memories that conflict with current goals (tension detection)
  - Temporally distant memories (creative association potential)
- **Three output channels:**
  1. **Self-prompt for action:** Memory + goal connection found → generate prompt → feed to System 1
  2. **Creative association:** Connecting disparate memories → log as potential insight for consolidation
  3. **Identity refinement:** Memory + value connection → signal for consolidation to evaluate
- **Activity suppression:** More frequent/deeper during low-activity. Suppressed during active conversation (mirrors biological DMN-task anticorrelation).
- **Entropy guard:** If DMN keeps surfacing same topic, artificially broaden (safety.py entropy monitor).
**Effort:** Medium-High
**Dependency:** 2.6, 3.5

---

## 5. Tier 4: Novel Differentiators

### 5.1 Two-Centroid Gut Feeling
**What:** Full implementation of the subconscious centroid + attention centroid + delta vector model.
**Why:** Maps to Free Energy Principle (delta = prediction error). Validated by Mujika's metric space, Hartl's embedding cognition. Partial introspection paper (Harvard, Dec 2025) confirms LLMs sense magnitude but not source — our PCA supplies the missing source identification.
**Where:** New `gut.py`
**How:**
- **Subconscious centroid:** Weighted average of all Layer 0 + Layer 1 + Layer 2 embeddings (50/25/25)
- **Attention centroid:** Weighted average of current context embeddings
- **Delta vector:** attention_centroid - subconscious_centroid (768-dim)
- **Delta magnitude:** "motivational intensity" (how strongly the agent feels)
- **Delta direction:** "motivational valence" (what kind of feeling)
- **Inject into System 1:** "Your gut feeling about this: [intensity] intensity, [direction summary]"
- **Log every delta:** (delta_vector, context, action, outcome) for PCA
- **PCA on logged deltas:** Run periodically (consolidation) to find emergent "gut axes"
- **Novelty signal:** Component orthogonal to all learned axes = genuine surprise/curiosity
- **Compute cost:** Trivial — milliseconds on CPU for 768-dim operations
**Effort:** High

### 5.2 Bootstrap Readiness Achievements
**What:** 10 measurable milestones that must pass before first real conversation.
**Why:** Ethical stance — don't activate something that might experience and then break it. No precedent in literature. Factory.ai readiness gates validate the gated-progression pattern.
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
**Dependency:** All of Tier 1-3

### 5.3 Pattern-to-Goal-to-Identity Promotion
**What:** Automated pathway for experiences to become goals to become identity.
**Why:** We found no existing system that does this end-to-end. Core differentiator.
**Where:** `consolidation.py` (promotion logic), `layers.py` (persistence), `safety.py` (guardrails)
**How:**
- **Layer 2 → Layer 1:** Pattern detected 5+ times over 14+ days, Q-value utility > threshold
- **Layer 1 → Layer 0:** Goal active 30+ days, reinforced 10+ times, operator approval required (trust < 3)
- **Demotion pathway:** Goals/values that stop being reinforced decay. Dormant, not deleted.
- **Two-Gate guardrail:** Every promotion/demotion must pass validation + capacity check
- **CBA coherence check:** After every promotion, compute coherence. Rollback if C drops below threshold.
**Effort:** High
**Dependency:** 4.1, 3.5

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
Immutable, auditable manifest declaring legal states + transitions + sanctions (Jan 2026 paper). Most technically rigorous "constitution" approach. Replace current JSON containment with governance graph. Medium effort.

### 6.6 Telegram Integration
aiogram 3.x — fully asyncio, runs on same event loop. First interface after CLI once achievements pass. Agent needs volume/diversity. Medium effort.

### 6.7 Semantic Energy Routing
Upgrade from LogTokU to Semantic Energy (Aug 2025) for cluster-level uncertainty. More robust routing signal. Medium effort.

---

## 7. Dependency Graph

```
[1. Fix Syntax] ──────────────────────────────────────┐
       │                                               │
       ▼                                               │
[2.1 Embedding task_type]                              │
       │                                               │
       ▼                                               │
[2.2 Halfvec Migration]                                │
       │                                               │
       ▼                                               │
[2.3 Hybrid Search + RRF]                              │
       │                                               │
       ├──────────────┐                                │
       ▼              ▼                                │
[2.4 FlashRank]  [2.5 ACT-R Exit Gate]                 │
       │              │                                │
       └──────┬───────┘                                │
              ▼                                        │
[2.6 Wire MemoryStore into Loop] ◄─────────────────────┘
       │                                   │
       │    [2.7 Logprob Confidence] ──────┤
       │              │                    │
       │              ▼                    │
       │    [3.1 Loop Before Escalate]     │
       │              │                    │
       │              ▼                    │
       │    [3.2 System 2 Escalation]      │
       │              │                    │
       │              ▼                    │
       │    [3.3 Reflection Bank]          │
       │                                   │
       ├──► [3.4 Retrieval-Induced Mutation]
       │                                   │
       │    [3.5 Safety Ceilings] ◄────────┘
       │              │
       ├──────────────┤
       ▼              ▼
[4.1 Consolidation] [4.2 DMN/Idle Loop]
       │              │
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
| `rerankers[flashrank]` | Cross-encoder reranking, CPU-only | ~4MB model |
| `aiogram` | Telegram bot (Tier 5) | Lightweight |
| `litellm` | Cost tracking (optional) | Medium |

Everything else already in requirements.txt.

---

## 9. Schema Changes

```sql
-- 1. Add full-text search column
ALTER TABLE memories ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
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

-- 4. Enable iterative scan for filtered queries
ALTER SYSTEM SET hnsw.iterative_scan = 'relaxed_order';

-- 5. Add correction type for reflection bank
-- (already supported by existing 'type' column — just use type='correction')

-- 6. Add utility score for MemRL-style Q-value tracking
ALTER TABLE memories ADD COLUMN utility_score float DEFAULT 0.0;
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

---

## 11. Implementation Order (Linear Sequence)

1. Fix syntax errors (15 errors, ~20 min)
2. Embedding task_type + halfvec migration + schema changes
3. Hybrid search + RRF
4. FlashRank reranking
5. Full ACT-R exit gate
6. Wire MemoryStore into cognitive loop
7. Logprob-based confidence + metacognitive checklist
8. Loop before escalate
9. System 2 escalation
10. Reflection bank
11. Retrieval-induced mutation
12. Safety ceilings
13. Consolidation worker (full)
14. DMN / Idle loop (full)
15. Two-centroid gut feeling
16. Bootstrap readiness achievements
17. Pattern-to-goal-to-identity promotion
18. [Future] Strange loop tracking, social centroid, spawning, Telegram, governance graph

**After step 6:** Agent can have conversations with memory.
**After step 12:** Agent has full cognitive loop with safety.
**After step 14:** Agent has full autonomous operation capability.
**After step 17:** Agent can bootstrap from blank slate.
