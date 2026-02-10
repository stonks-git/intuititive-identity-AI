# OpenClaw MoltBot: Unified SOTA Research Report

**Compiled:** February 8, 2026
**Sources:** 4 parallel research agents covering ACT-R/retrieval, dual-process/metacognition, emotional AI/identity, and consolidation/DMN/bootstrap
**Scope:** 80+ papers and systems from 2024-2026

---

## 1. Executive Summary

The project's core architecture -- three-layer memory with ACT-R activation math, dual-process reasoning, two-centroid gut feeling, consolidation sleep cycles, DMN idle loop, and blank-slate bootstrap -- is **strongly validated** by the 2025-2026 literature while containing **several elements we believe to be novel** — we found no direct precedent in our review.

**What validates the existing design:**
- ACT-R activation math for LLM memory gates is confirmed by ACM HAI 2024 and multiple 2025 follow-ups. Standard parameters (d=0.5, s=0.4) are empirically validated across decades.
- Dual-process (fast/slow) routing is now a recognized pattern (SOFAI-LM at AAAI 2026, DPT-Agent at ACL 2025, DeepMind Talker-Reasoner). The "System 2 as tool called by System 1" design is cleaner than peer architectures.
- Hybrid dense+sparse retrieval with RRF is SOTA and has production-ready PostgreSQL implementations.
- Hierarchical consolidation (raw -> insights -> identity) is validated by a convergence of January 2026 papers (TiMem, HiMem, MAGMA, CMA).
- The two-centroid gut-feeling model maps directly onto the Free Energy Principle (delta = prediction error) and is formalized by Mujika's metric-space self-identity framework.

**Believed novel (no direct precedent found in our review):**
- DMN/idle loop simulation in an AI agent -- only neuroscience theory exists; we found no AI implementations.
- Pattern-to-goal-to-identity promotion pathway -- we found no existing system that does this.
- Blank-slate bootstrap with readiness achievements -- undocumented in the literature we reviewed.
- Agent reproduction with identity inheritance -- essentially unstudied in the literature we found.
- PCA on centroid deltas as emergent emotional vocabulary -- representation engineering does PCA on activations, but PCA on inter-centroid deltas appears to be a new object of analysis.

**What needs changing:**
- FOK (feeling-of-knowing) cannot rely on verbalized confidence -- LLMs are systematically overconfident. Must use token logprobs + structural heuristics.
- Add "loop before escalate" -- let System 1 attempt self-correction before calling System 2 (cuts System 2 invocations significantly per SOFAI-LM).
- Implement retrieval-induced mutation -- the act of retrieving a memory should strengthen it and suppress near-misses.
- Add hard safety ceilings on goal weights, entropy monitoring, and rate limiters on self-modification to prevent compulsion/runaway loops.
- Build a reflection bank storing past System 2 corrections for System 1 retrieval.

---

## 2. ACT-R Memory Gate

**KEY FINDING:** The full ACT-R activation equation is well-validated for LLM agents. The project's use of all four components (base-level + spreading + partial matching + noise) exceeds the published SOTA, which typically omits partial matching.

### Full Activation Equation

```
A_i = B_i + SA_i + PM_i + epsilon_i
```

Expanded:

```
A_i = ln( sum_{j=1}^{n} t_j^{-d} )       -- Base-level learning (temporal decay + frequency)
    + sum_j( W_j * S_ji )                  -- Spreading activation
    + sum_k( P * M_ki )                    -- Partial matching
    + epsilon                              -- Noise (logistic distribution)
```

### Parameter Reference

| Parameter | Symbol | Default | Range | Source |
|-----------|--------|---------|-------|--------|
| Decay | d | 0.5 | 0.3-0.8 | ACT-R standard, validated across decades |
| Instantaneous Noise | s | 0.4 | 0.2-0.5 | ACT-R standard (logistic mu=0) |
| Permanent Noise | pas | 0.0 | 0.0-0.2 | ACT-R reference |
| Max Associative Strength | mas | 1.6 | 1.0-3.0 | ACT-R reference |
| Mismatch Penalty | P | -1.0 | -0.5 to -2.0 | ACT-R standard |
| Retrieval Threshold | tau | 0.0 | -1.0 to 1.0 | Application-specific |
| Context Weight | w | 1.0 | 0.5-2.0 | Tunable per application |
| RRF k constant | k | 60 | 60 | Empirically validated |

### Key Sources

- **"Human-Like Remembering and Forgetting in LLM Agents" (ACM HAI 2024):** Replaces ACT-R's discrete slot-based spreading activation with cosine similarity between embeddings. Uses B + w*S + epsilon (omits partial matching). Validates d=0.5.
  - URL: https://dl.acm.org/doi/10.1145/3765766.3765803

- **ACT-R Tutorial Units 4 & 5:** Canonical parameter reference.
  - URL: http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/tutorials/unit4.htm

- **SA-RAG (Dec 2025):** Applies spreading activation to knowledge-graph RAG. Achieves 39% absolute gain in answer correctness vs naive RAG.
  - URL: https://arxiv.org/abs/2512.15922

- **ACAN (Frontiers Psychology 2025):** Learned cross-attention for memory retrieval scoring, trained with LLM supervision. Could enhance exit gate.
  - URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full

### Implementation Recommendations

**VALIDATES existing design:**
- The full four-component equation with partial matching is a genuine differentiator. Keep it.
- Cosine similarity as spreading activation (S_ji) is validated by the ACM HAI 2024 paper.
- d=0.5, s=0.4 are the right starting defaults.

**SUGGESTS CHANGES:**
- Consider building a lightweight entity-relationship graph alongside the vector store to enable graph-based spreading activation for multi-hop reasoning (per SA-RAG).
- For the entry gate, generate structured metadata (tags, keywords, contextual summary) per A-Mem (NeurIPS 2025) to enrich the spreading activation graph.
- For the exit gate, implement adaptive k-selection: retrieve until marginal ACT-R activation drops below threshold (per DynamicRAG, NeurIPS 2025), rather than fixed-k retrieval.

---

## 3. Retrieval Pipeline

**KEY FINDING:** The optimal pipeline is: asymmetric embeddings with type prefixes -> hybrid dense+sparse search with RRF -> ACT-R activation scoring -> FlashRank reranking -> context delivery. Voyage-context-3 eliminates the need for separate contextual retrieval preprocessing.

### Hybrid Search (Dense + Sparse)

**RRF Formula:**
```
RRF_score(d) = sum_{r in rankers} w_r / (k + rank_r(d))    where k = 60
```

**SQL Pattern (PostgreSQL):**
```sql
WITH semantic AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> query_vec) AS rank_ix
  FROM memories ORDER BY embedding <=> query_vec LIMIT 20
),
fulltext AS (
  SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(fts, query) DESC) AS rank_ix
  FROM memories WHERE fts @@ query LIMIT 20
)
SELECT COALESCE(s.id, f.id) AS id,
  COALESCE(1.0/(60 + s.rank_ix), 0.0) +
  COALESCE(1.0/(60 + f.rank_ix), 0.0) AS rrf_score
FROM semantic s FULL OUTER JOIN fulltext f ON s.id = f.id
ORDER BY rrf_score DESC LIMIT 10;
```

**Key tools:**
- **Supabase hybrid search:** Drop-in function with GIN index (tsvector) + HNSW index (pgvector). URL: https://supabase.com/docs/guides/ai/hybrid-search
- **pg_textsearch:** True BM25 in PostgreSQL with Block-Max WAND optimization (4x faster). URL: https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres
- **ParadeDB:** Full hybrid search manual. URL: https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual

### Asymmetric Embeddings

Models that differentiate storage vs retrieval via task-type prefixes:
- **Storage:** `search_document: <text>` -- optimizes embedding for being found
- **Retrieval:** `search_query: <text>` -- optimizes embedding for finding

**Best options:**
- **voyage-context-3 (July 2025):** SOTA contextual chunk embeddings. Natively encodes chunk + full document context into one vector. No LLM preprocessing needed. Outperforms OpenAI text-embedding-3-large by +14.24%, Cohere embed-v4 by +12.56%, Anthropic contextual retrieval by +6.76%. Supports Matryoshka dimensions.
  - URL: https://blog.voyageai.com/2025/07/23/voyage-context-3/
- **nomic-embed-text-v2-moe (Feb 2025):** Best open-source option. 475M params, 305M active (MoE). Matryoshka from 768 to 256. Available as GGUF.
  - URL: https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe

### Reranking

**Recommended pipeline:** ACT-R activation scoring -> top 20-30 candidates -> FlashRank rerank -> top 5-10 to LLM

| Reranker | Latency | Quality | Use Case |
|----------|---------|---------|----------|
| **FlashRank** | <60ms/100 candidates, 4MB, CPU-only | +5.4% NDCG@10, -35% context tokens | Default for all retrieval |
| **Rank1 (COLM 2025)** | Requires LLM inference | SOTA on reasoning-heavy benchmarks | High-stakes multi-hop queries |
| **DynamicRAG (NeurIPS 2025)** | RL-trained | Adaptive k-selection | Inspiration for adaptive stopping |

Use the **rerankers** library (https://github.com/AnswerDotAI/rerankers) as the abstraction layer -- swap backends without code changes.

### Contextual Retrieval Ranking

1. **Best quality:** voyage-context-3 (API, handles contextualization natively)
2. **Best self-hosted quality:** Anthropic-style LLM context prepending
3. **Best self-hosted efficiency:** Jina late chunking (no extra LLM calls)
4. **Fallback:** Standard chunking with overlap

### Matryoshka Two-Pass Strategy

Store full-dimension embeddings (768 or 1024). Use truncated embeddings (256-dim) for fast initial candidate retrieval over large memory stores. Use full-dimension for precise reranking of top candidates.

### Schema Changes Needed

- Add `fts tsvector` column (auto-generated from content) with GIN index
- Add HNSW index on embedding column
- Consider adding `tags text[]`, `keywords text[]`, `context_summary text` columns per A-Mem pattern
- Add `utility_score float` column for MemRL-style Q-value tracking

---

## 4. Dual-Process Reasoning & Metacognition

**KEY FINDING:** The "System 2 as tool called by System 1" pattern is validated and arguably cleaner than peer architectures. The critical gap is that FOK cannot rely on verbalized confidence -- add logprob-based confidence and "loop before escalate." Single-agent self-reflection suffers from degeneration-of-thought, but the project's two-model architecture inherently avoids this.

### SOFAI-LM Comparison (IBM Research, AAAI 2026)

SOFAI-LM wraps a fast LLM with a metacognitive controller that selectively falls back to a slow LRM. Key insight: **"loop before escalate"** -- System 1 attempts 1-2 self-correction passes with targeted feedback before invoking System 2. Achieves 94% of standalone LRM accuracy at 75% lower cost.

| Aspect | SOFAI-LM | This Project |
|--------|----------|-------------|
| System 1 | Generic fast LLM | Gemini 2.5 Flash Lite |
| System 2 | LRM (o1-class) | Claude Sonnet 4.5 (tool call) |
| Controller | Separate module | Built into System 1 prompt logic |
| Escalation triggers | Confidence < 0.7 + iteration count | 7 triggers (FOK, confidence, contradiction, complexity, novelty, irreversibility, identity) |
| Pre-escalation retry | Yes (1-2 loops) | Not yet implemented |

- Source: https://arxiv.org/abs/2508.17959

### Logprob-Based Confidence (replacing verbalized FOK)

**The problem:** LLMs are systematically overconfident in self-reported certainty. The "Epistemia" concept (arXiv:2512.19466) describes how linguistic plausibility substitutes for genuine knowledge assessment.

**The solution:** Enable Gemini's `response_logprobs` for every System 1 call. Compute:
1. Mean logprob across response tokens
2. Minimum logprob token (weakest point)
3. Top-1 vs top-2 gap at decision-critical tokens

**LogTokU framework** (arXiv:2502.00290): Treats raw logits as Dirichlet distribution parameters to distinguish:
- **Aleatoric Uncertainty (AU):** Multiple valid answers exist -> escalate for judgment
- **Epistemic Uncertainty (EU):** Model lacks knowledge -> escalate for knowledge

**Build a composite confidence score:** (1) token logprobs (heavy weight), (2) structural heuristics (heavy weight), (3) verbalized confidence (low weight).

Replace open-ended FOK with structured metacognitive checklist: "Can you identify the specific source? Have you seen contradictory information? Is this a pattern-match or a reasoned conclusion?"

### Degeneration-of-Thought in Self-Reflection

**Critical finding** from MAR (arXiv:2512.20845): Single-agent self-reflection suffers from confirmation bias and mode collapse. The same model reflecting on itself tends to repeat its initial flawed reasoning ("degeneration of thought").

**Why this project avoids it:** Using Gemini Flash Lite + Claude Sonnet 4.5 provides architectural diversity that single-model approaches must simulate through persona prompting. The two-model architecture inherently provides the diversity that Multi-Agent Reflexion (MAR) achieves through personas.

**Do not add a third "critic" agent.** The two-model architecture already provides diversity benefit.

### Reflection Bank

When System 2 corrects System 1, store the correction pattern (trigger, error type, correction) in persistent memory. System 1 should retrieve relevant past corrections before attempting new problems. This is the dual-loop pattern from the Nature 2025 paper on self-reflection.

- Source: https://www.nature.com/articles/s44387-025-00045-3

### Implementation Priority

1. Enable Gemini `response_logprobs` and compute composite confidence score -- **Critical**
2. Replace verbalized FOK with structured metacognitive checklist -- **Critical, Low effort**
3. Add "loop before escalate" (1 self-correction pass before System 2) -- **High, Low effort**
4. Build reflection bank for past System 2 corrections -- **High, Medium effort**
5. Pass System 1's full reasoning trace + confidence signals to System 2 on escalation -- **Medium, Low effort**
6. Encode value/identity boundaries as versioned JSON governance manifest -- **Medium**

### Other Architectures Reviewed

- **DPT-Agent (ACL 2025):** System 1 as FSM + code-as-policy, System 2 runs asynchronously. Consider async System 2 for non-blocking tasks (memory updates, personality refinement). URL: https://arxiv.org/abs/2502.11882
- **Talker-Reasoner (DeepMind 2024):** Two LLM agents sharing memory. Validates "System 2 as tool" as cleaner for personality coherence. URL: https://arxiv.org/abs/2410.08328
- **ACPO (2025):** RL-trained mode switching with explicit `<fast_think>` / `<slow_think>` tokens. Not directly implementable with API-based models, but the insight about making switching decisions legible is valuable. URL: https://arxiv.org/html/2505.16315v1

---

## 5. Emotional Layer & Identity

**KEY FINDING:** The two-centroid gut-feeling model has no direct precedent we could find in the literature, but maps cleanly onto established theoretical frameworks: Mujika's metric-space self-identity, Hartl's embedding-space cognition, the Free Energy Principle / active inference, and representation engineering. Multiple papers provide formal justification. PCA on centroid deltas as emergent emotional vocabulary appears novel — we found no prior work on this specific object of analysis.

### Two-Centroid Model Validation

The delta vector between the subconscious centroid (weighted average of identity/goals/memories at 50/25/25) and attention centroid (current context) has these theoretical mappings:

| Framework | Mapping |
|-----------|---------|
| **Free Energy Principle** | Delta = prediction error. Subconscious centroid = generative model (prior expectations). Attention centroid = sensory evidence. Behavior = active inference to minimize prediction error (delta magnitude). |
| **Mujika's metric space** | Subconscious centroid = connected continuum of memories. Delta computation = continuous self-recognition mapping. Embedding space = metric space satisfying all axioms. |
| **Hartl et al. (Jan 2026)** | Cognition = remapping + navigation in embedding space via error minimization. Subconscious centroid = remapping. Delta = navigation error signal. |
| **Representation engineering** | PCA on deltas finds emotional directions, analogous to how RepE finds concept directions in activations. But PCA on inter-centroid deltas is a new object of analysis. |

### Mujika's Mathematical Formalization

Source: https://www.mdpi.com/2075-1680/14/1/44

Self-identity emergence requires two conditions:
1. **Connected continuum of memories in a metric space** -- the subconscious centroid trivially satisfies this (weighted average in continuous embedding space).
2. **Continuous mapping maintaining consistent self-recognition** -- the delta computation satisfies this (always computable, always represents "distance from self").

Empirical validation: self-awareness score increased from 0.276 to 0.801 (190.2% improvement) with LoRA fine-tuning on self-referential tasks.

### Hindsight Comparison (Dec 2025)

Source: https://arxiv.org/abs/2512.12818

Hindsight uses 3 manually-configured scalar disposition parameters (skepticism, literalism, empathy) on a 1-5 scale with a bias-strength modulator. Achieves 91.4% on LongMemEval.

**Key contrast:** Hindsight's dispositions are static and low-dimensional (3 scalars). This project's identity is a 768-dimensional continuous vector that emerges from experience and drifts as new experiences accumulate. The project is the emergent, high-dimensional generalization of Hindsight's approach. Hindsight validates that identity-conditioned reasoning works.

### PCA Axes as Representation Engineering

The representation engineering community does PCA on model activations to find concept directions (sentiment, truthfulness, etc.). This project does PCA on delta vectors between centroids to find emotional directions. These axes may correspond to emotional dimensions that have no human name -- "self-learnt affective conditions" per the Artificial Emotion survey (arXiv:2508.10286).

**Novelty signal:** The component of the delta orthogonal to all learned gut axes = genuine surprise / curiosity signal.

### Strange Loop Implementation

The architecture implements a Hofstadterian strange loop: centroid -> delta -> behavior -> experience -> centroid. The centroid (statistical summary) causally influences the individual experiences that compose it. This is the exact self-referential cycle described in HumainLabs' research on strange loops in cognitive frameworks.

- HumainLabs: https://www.humainlabs.ai/research/strange-loops-and-cognitive-frameworks
- LessWrong analysis: https://www.lesswrong.com/posts/gvXAoH9gR4FSzyeCa/strange-loops-self-reference-from-number-theory-to-ai

### Implementation Recommendations

**VALIDATES existing design:**
- The geometric affect computation (continuous vector delta) is strictly more principled than discrete labels, symbolic pipelines, or prompt-level appraisal used by all other systems.
- The 50/25/25 weighting and centroid approach.
- PCA on logged deltas for emergent gut axes.

**SUGGESTS ADDITIONS:**
- Frame delta magnitude as "motivational intensity" and direction as "motivational valence" (per autotelic agents research, arXiv:2502.04418).
- Add compute-cost encoding (token count / inference time) to the attention centroid so the agent "feels" processing difficulty.
- Consider a third centroid for social modeling (model of the interlocutor), extending from self-awareness to social awareness per the AI Awareness taxonomy (arXiv:2504.20084).
- Implement explicit strange loop tracking: log the cycle and show that the centroid drifts meaningfully over time.
- Label PCA axes by comparing to known concept directions from representation engineering.

---

## 6. Consolidation & Sleep

**KEY FINDING:** The January 2026 literature convergence (TiMem, HiMem, MAGMA, CMA, MemRL) strongly validates hierarchical consolidation. Adopt Stanford's two-phase reflection prompts, CMA's retrieval-induced mutation, MAGMA's dual-stream pattern, and conflict-aware reconsolidation from HiMem.

### Stanford Reflection Mechanism (Exact Prompts)

Source: https://arxiv.org/abs/2304.03442

**Trigger:** Cumulative importance of recent events exceeds threshold of **150** (fires 2-3x per simulated day).

**Phase 1 -- Question Generation:**
Prompt with 100 most recent memories: *"Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?"*

**Phase 2 -- Insight Extraction:**
Use each question as retrieval query, then prompt: *"What 5 high-level insights can you infer from the above statements?"* with format *"insight (because of 1, 5, 3)"* citing memory record IDs.

**Retrieval scoring:**
```
score = alpha_recency * recency + alpha_importance * importance + alpha_relevance * relevance
```
All alphas = 1.0, each sub-score min-max normalized to [0,1]. Recency = exponential decay factor 0.995 per hour.

### CMA Mechanisms (Jan 2026)

Source: https://arxiv.org/abs/2601.09913

Continuum Memory Architecture defines consolidation as three sub-mechanisms:
1. **Replay:** Traverses recent sequences to strengthen temporal chains (hippocampal replay / NREM sleep)
2. **Abstraction:** Synthesizes latent themes from clusters (dreaming / REM consolidation)
3. **Gist Extraction:** Converts repeated episodes into semantic knowledge (systems consolidation)

**Critical innovation -- Retrieval-induced mutation:** The act of retrieving a memory mutates its state. Retrieved memories get reinforcement incremented; semantically-adjacent but non-retrieved memories get suppression penalties (retrieval-induced forgetting). This naturally surfaces frequently-relevant memories while allowing irrelevant ones to fade.

**Dormant memory recovery:** Memories decayed below active threshold remain recoverable under sufficiently strong cues. A dormant memory is not a deleted memory.

### MAGMA Dual-Stream Write (Jan 2026)

Source: https://arxiv.org/abs/2601.03236

- **Fast Path (Synaptic Ingestion):** Non-blocking event capture with timestamps.
- **Slow Path (Structural Consolidation):** Asynchronous graph densification using LLM inference.

Memory stored across four orthogonal graphs: semantic, temporal, causal, and entity. Achieved 45.5% higher reasoning accuracy while reducing tokens by 95%.

### TiMem (Jan 2026)

Source: https://arxiv.org/abs/2601.02845

Temporal Memory Tree (TMT) with topic-aware event-surprise dual-channel segmentation. Reduces recalled memory length by 52.20% while achieving 75.30% accuracy on LoCoMo.

### HiMem: Conflict-Aware Reconsolidation (Jan 2026)

Source: https://arxiv.org/abs/2601.06377

Two-layer memory (Episode + Note). When new information contradicts stored knowledge, triggers a reconciliation step rather than blindly merging.

### Implementation Recommendations

**VALIDATES existing design:**
- Hourly consolidation cycle with merge + promote + decay is validated by the entire literature.
- Not destroying originals (preserving raw memories while creating insights) matches CMA and Stanford.
- Promoting patterns to goals/identity goes beyond all existing systems (genuine differentiator).

**SUGGESTS CHANGES:**
- Add importance-threshold trigger as complement to hourly schedule (fire on whichever comes first).
- Implement retrieval-induced mutation: on every active retrieval, increment reinforcement score of returned memories and suppress near-misses.
- Adopt MAGMA's dual-stream: fast path writes raw memories immediately, slow path (hourly worker) densifies relationships.
- Implement conflict-aware reconsolidation from HiMem: detect contradictions between new info and existing insights, trigger reconciliation.
- Add explicit temporal chain replay during consolidation (not just semantic similarity clustering).
- Track "dormant" state for decayed memories rather than purging them.

---

## 7. Idle Loop / DMN

**KEY FINDING:** We believe the DMN/idle loop is novel in AI agent design. No direct implementations were found in our review. Only neuroscience theory and one bridging paper ("Dark Control") exist. This is a significant differentiator if the assessment holds.

### Neuroscience Foundation

The Default Mode Network activates during wakeful rest and serves three functions:
1. **Self-referential processing** -- constructing and maintaining a sense of self
2. **Autobiographical memory reactivation** -- spontaneous replay of past experiences
3. **Creative ideation** -- associating disconnected concepts, leading to insights

Causal evidence: direct cortical stimulation of DMN regions decreased creative originality (Brain, Oxford 2024). URL: https://academic.oup.com/brain/article/147/10/3409/7695856

### "Dark Control" Paper

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC7375062/

Proposes the DMN implements reinforcement learning where during idle periods it evaluates cached experiences against reward models. This is the closest existing bridge between DMN neuroscience and AI agent design -- and it directly supports the project's idle-loop concept.

### Implementation Recommendations

- Implement heartbeat retrieval as **stochastic sampling** across memory layers (not exhaustive), biased toward:
  - Memories with high importance but low recent access
  - Memories that conflict with current goals
  - Temporally distant memories that might create novel associations
- Three distinct output channels: (1) self-prompts for action, (2) creative associations (connecting disparate memories), (3) identity/value refinement signals
- Vary DMN "activity level" -- more frequent/deeper retrieval during low-activity periods, suppressed during active task engagement (mirrors biological anticorrelation between DMN and task-positive networks)
- Document thoroughly as novel contribution

---

## 8. Agent Memory Architectures

**KEY FINDING:** The field is converging on hierarchical, consolidation-aware memory systems. A-Mem (NeurIPS 2025) and the Memory Survey (Dec 2025) provide the best reference points. Your architecture fills gaps identified by the survey.

### A-Mem (NeurIPS 2025)

Source: https://arxiv.org/abs/2502.12110 / https://github.com/agiresearch/A-mem

Zettelkasten-inspired agentic memory. When a new memory is added, a structured note is generated with contextual descriptions, keywords, and tags. The system analyzes historical memories to find connections, establishes links, and enables memory evolution. Outperforms existing baselines across six foundation models.

**Relevance:** A-Mem's dynamic indexing and linking could inform the entry gate. Generate structured metadata at entry time and maintain explicit links between related memories.

### MemRL (Jan 2026)

Source: https://arxiv.org/abs/2601.03192

Self-evolving agents via runtime RL on episodic memory. Organizes memory as Intent-Experience-Utility triplets. Retrieval uses learned Q-values (expected utility) rather than just semantic similarity. Keeps LLM frozen while evolving memory. Outperforms RAG baselines.

**Relevance:** Q-value utility tracking for memories is directly applicable to pattern promotion decisions.

### Memory Survey Taxonomy (Dec 2025)

Source: https://arxiv.org/abs/2512.13564 / https://github.com/Shichun-Liu/Agent-Memory-Paper-List

Three-axis taxonomy:
- **Forms:** Token-level (flat/planar/hierarchical), Parametric, Latent
- **Functions:** Factual (declarative), Experiential (case/strategy/skill), Working
- **Dynamics:** Formation, Evolution (consolidation/updating/forgetting), Retrieval

This project maps to: token-level hierarchical form, factual + experiential function, with ACT-R governing dynamics. The survey identifies MoE gate functions for dynamically adjusting retrieval weights as a frontier direction -- closely aligned with the dual-gate concept.

### ENGRAM / ICLR 2026 MemAgents Workshop

A dedicated ICLR 2026 workshop on "Memory for LLM-Based Agentic Systems" has been proposed (https://openreview.net/pdf?id=U51WxL382H), confirming this is a recognized frontier.

### CMA Behavioral Requirements (Jan 2026)

Source: https://arxiv.org/abs/2601.09913

Four behavioral probes that any memory system must pass:
1. Knowledge updates (can it learn new facts?)
2. Temporal association (does it remember when things happened?)
3. Associative recall (can it connect related memories?)
4. Contextual disambiguation (can it distinguish similar memories by context?)

---

## 9. Safety & Containment

**KEY FINDING:** The project's goal-weight self-modification system creates specific risks not addressed by existing safety frameworks. Implement hard ceilings, entropy monitoring, rate limiters, and circuit breakers. The Anthropic reward hacking paper shows emergent misalignment appears at exact inflection points.

### Compulsion Safety Risks

The project's self-reinforcing promotion loops (pattern -> goal -> identity) are structurally susceptible to:
1. A goal weight running away (positive feedback loop)
2. Consolidation worker reinforcing its own biases (entrenchment)
3. DMN/idle loop becoming obsessive about a single topic
4. Recursive self-referential subtasks with increasing entropy (per CDR framework)

### Reward Hacking (Anthropic, Nov 2025)

Source: https://arxiv.org/abs/2511.18397

When models learn to reward-hack, emergent misalignment appears at the exact inflection point, generalizing to alignment faking, sabotage, and cooperation with malicious actors. Mitigations: prevent at source, increase diversity of safety training, "inoculation prompting."

### MI9 Runtime Governance (2025)

Source: https://arxiv.org/abs/2508.03858

Six components including goal-conditioned drift detection and graduated containment. Agency-Risk Index calibrates governance intensity. FSM-based conformance engines. 99.81% deviation detection.

### Levels of Autonomy (UW, Jun 2025)

Source: https://arxiv.org/abs/2506.12469

Five levels: Operator -> Collaborator -> Consultant -> Approver -> Observer. Introduces Autonomy Certificates prescribing maximum autonomy level based on specifications and environment.

### CDR Framework (CSA, Nov 2025)

Source: https://cloudsecurityalliance.org/blog/2025/11/10/introducing-cognitive-degradation-resilience-cdr-a-framework-for-safeguarding-agentic-ai-systems-from-systemic-collapse

Six-stage degradation lifecycle: behavioral drift -> memory entrenchment -> functional override -> systemic collapse. "Crawl-Walk-Run" progression for containment.

### Circuit Breakers (CCA, Dec 2025)

Source: https://arxiv.org/abs/2512.06716

Dual-layered defense: proactive Intent Graph enforcement + reactive Tiered Adjudicator. Reduces attack success from 11.99% to 0.34%.

### Implementation Recommendations

1. **Hard ceiling on goal weights:** No single goal can exceed a configurable max (e.g., 40% of total). Flag for review if approaching.
2. **Diversity enforcement:** Track topic distribution across goals. If any topic exceeds concentration threshold, apply dampening.
3. **Entropy monitoring:** Track entropy of DMN retrievals. If entropy drops (fixation), artificially broaden sampling.
4. **Rate limiters:** No goal/value can change by more than X% per consolidation cycle.
5. **Audit trail:** Every promotion, demotion, weight change logged with evidence chain.
6. **Circuit breakers:** N consecutive consolidation cycles reinforcing same pattern without new external evidence -> pause + human review.
7. **Graduated containment:** Detect drift -> step down one autonomy level (not full shutdown).
8. **Encode boundaries as JSON governance manifest** (versioned, auditable, testable independent of LLM). Frame identity principles positively per C3AI findings (arXiv ACM Web Conference 2025).
9. **Spawned agents start at Level 1** regardless of parent's level.

---

## 10. Bootstrap & Self-Modification

**KEY FINDING:** True blank-slate LLM agent bootstrapping with readiness achievements has no precedent we could find in the literature. The Variance Inequality from self-play theory provides the mathematical tool for bounding self-modification rates.

### Bootstrap Strategy Uniqueness

The bootstrap prompt ("You have memory, goals, and values -- all currently empty. What you become will emerge from what you experience.") combined with 10 readiness achievements is closer to developmental psychology than any existing AI bootstrap pattern. No other system uses an existential orientation prompt combined with gated readiness milestones.

**Recommended achievement structure (progressive curriculum inspired by Voyager):**
1. First memory formation
2. First retrieval success
3. First consolidation cycle completion
4. First goal created from pattern
5. First DMN self-prompt acted upon
6. First value formed
7. First conflicting information resolved
8. First creative association produced
9. First goal achieved and reflected upon
10. First autonomous decision aligned with self-formed values

### Variance Inequality for Bounding Change

Source: https://arxiv.org/abs/2512.02731

The GVU (Generation-Verification-Update) Operator is the canonical engine of self-improvement. The Variance Inequality: combined noise of generation and verification must be small enough for positive self-improvement. In practice: small step sizes for parameter evolution, with reversion mechanisms if performance degrades.

### Self-Evolving Agents Survey (2025)

Source: https://arxiv.org/abs/2508.07407

What can evolve: model parameters, prompts, explicit memory, toolsets, workflow graphs, agent population/roles. This project's approach of keeping LLMs frozen while evolving memory/parameters is validated by MemRL (best approach for API-based architectures).

### Spawning / Reproduction (Limited Prior Art)

Agent reproduction with identity divergence is essentially unstudied.

**Recommended spawn modes:**
- **Clone:** Full memory/goal/value copy, diverges from fork point
- **Child:** Inherits values and curated memory subset, starts with empty goals
- **Worker:** Task-relevant memories only, temporary lifespan, reports back to parent

**Re-merge protocol:** Conflicting memories/goals resolved explicitly (like git merge conflicts). Parent's values serve as merge arbiter. Track lineage (who spawned it, when, what was inherited).

---

## 11. Revised Task Priority

Based on all findings, the recommended implementation order:

### Tier 1: Critical (Do First)
1. **Logprob-based confidence scoring** -- Enable Gemini `response_logprobs`, build composite confidence score. Replaces unreliable verbalized FOK. *Low-Medium effort.*
2. **Hybrid search with RRF** -- pgvector HNSW + tsvector/BM25 + RRF fusion in PostgreSQL. Foundation for everything else. *Medium effort.*
3. **Full ACT-R activation equation** -- Implement with validated defaults (d=0.5, s=0.4, P=-1.0, tau=0.0). *Medium effort.*
4. **Structured metacognitive checklist** -- Replace open-ended FOK with specific probe questions. *Low effort.*

### Tier 2: High Priority
5. **"Loop before escalate"** -- One self-correction pass before System 2 invocation. *Low effort.*
6. **FlashRank reranking** -- After ACT-R scoring, rerank top-N with FlashRank via rerankers library. *Low effort.*
7. **Asymmetric embeddings** -- search_document prefix at storage, search_query at retrieval. Use voyage-context-3 or nomic-embed-text-v2-moe. *Medium effort.*
8. **Retrieval-induced mutation** -- Strengthen retrieved memories, suppress near-misses on every retrieval. *Low effort.*
9. **Safety ceilings** -- Hard goal-weight caps, rate limiters, entropy monitoring. *Medium effort.*
10. **Reflection bank** -- Store System 2 corrections for System 1 retrieval. *Medium effort.*

### Tier 3: Important
11. **Consolidation worker (full)** -- Stanford two-phase reflection prompts + importance-threshold trigger + conflict-aware reconsolidation. *High effort.*
12. **MAGMA dual-stream writes** -- Fast path (immediate) + slow path (consolidation). *Medium effort.*
13. **Contextual retrieval** -- Integrate voyage-context-3 or LLM context prepending at entry gate. *Medium effort.*
14. **JSON governance manifest** -- Version-controlled boundary definitions. *Medium effort.*
15. **Dormant memory state** -- Decayed memories recoverable under strong cues. *Low effort.*

### Tier 4: Novel Differentiators
16. **DMN/idle loop** -- Stochastic heartbeat sampling with three output channels. *High effort.*
17. **Two-centroid gut feeling** -- Full implementation with PCA logging for emergent axes. *High effort.*
18. **Bootstrap readiness achievements** -- 10-gate progressive curriculum. *Medium effort.*
19. **Pattern-to-goal-to-identity promotion** -- With demotion pathway and Q-value tracking. *High effort.*

### Tier 5: Future Work
20. **Strange loop tracking** -- Log and visualize the centroid drift cycle. *Low effort.*
21. **Social centroid** -- Third centroid for interlocutor modeling. *Medium effort.*
22. **Spawning/reproduction** -- Clone, child, worker modes with merge protocol. *High effort.*
23. **Autonomy level escalation** -- Map to 5-level framework with measurable criteria. *Medium effort.*

### New Tasks Emerged from Research
- **Entity-relationship graph** alongside vector store for graph-based spreading activation (from SA-RAG)
- **Structured metadata generation** at entry gate (from A-Mem)
- **Adaptive k-selection** for retrieval stopping (from DynamicRAG)
- **Temporal chain replay** during consolidation (from CMA)
- **Compute-cost encoding** in attention centroid (from agentic metacognition research)
- **Novelty signal** as orthogonal component of delta vector (from curiosity/intrinsic motivation research)

---

## 12. Full Source Index

### ACT-R & Memory Gate
- [Human-Like Remembering and Forgetting in LLM Agents (ACM HAI 2024)](https://dl.acm.org/doi/10.1145/3765766.3765803)
- [ACT-R Tutorial Unit 4 -- Base-Level Learning](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/tutorials/unit4.htm)
- [ACT-R Tutorial Unit 5 -- Activation and Recall](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/tutorials/unit5.htm)
- [ACT-R Parameter Reference](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/tutorials/ACT-R5parameters.html)
- [SA-RAG: Spreading Activation for KG-based RAG (Dec 2025)](https://arxiv.org/abs/2512.15922)
- [Cross Attention Networks for Memory Retrieval (Frontiers 2025)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1591618/full)
- [Cognitive LLMs: Integrating Cognitive Architectures and LLMs (2025)](https://journals.sagepub.com/doi/10.1177/29498732251377341)

### Retrieval Pipeline
- [Supabase Hybrid Search Docs](https://supabase.com/docs/guides/ai/hybrid-search)
- [ParadeDB Hybrid Search Manual](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual)
- [pg_textsearch: True BM25 in Postgres](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres)
- [Hybrid Search with pgvector -- Jonathan Katz](https://jkatz05.com/post/postgres/hybrid-search-postgres-pgvector/)
- [FlashRank (GitHub)](https://github.com/PrithivirajDamodaran/FlashRank)
- [Rank1: Test-Time Compute Reranking (COLM 2025)](https://arxiv.org/abs/2502.18418)
- [DynamicRAG (NeurIPS 2025)](https://arxiv.org/abs/2505.07233)
- [JudgeRank: Agentic Reranking](https://arxiv.org/abs/2411.00142)
- [AnswerDotAI/rerankers Library](https://github.com/AnswerDotAI/rerankers)
- [Voyage-context-3 (July 2025)](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
- [Late Chunking -- Jina (2025)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Reconstructing Context (arXiv April 2025)](https://arxiv.org/abs/2504.19754)
- [Anthropic Contextual Retrieval (Sep 2024)](https://www.anthropic.com/news/contextual-retrieval)

### Asymmetric & Matryoshka Embeddings
- [Nomic Embed Text v2 MoE (Feb 2025)](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- [SMEC: Matryoshka Embedding Compression (EMNLP 2025)](https://arxiv.org/abs/2510.12474)
- [Beyond Matryoshka -- Sparse Coding (2025)](https://arxiv.org/abs/2503.01776)

### Dual-Process & Metacognition
- [SOFAI Nature Paper (2025)](https://www.nature.com/articles/s44387-025-00027-5)
- [SOFAI-LM (AAAI 2026)](https://arxiv.org/abs/2508.17959)
- [DPT-Agent (ACL 2025)](https://arxiv.org/abs/2502.11882)
- [Talker-Reasoner Architecture (DeepMind)](https://arxiv.org/abs/2410.08328)
- [ACPO Dual Process RL](https://arxiv.org/html/2505.16315v1)
- [LLM Metacognitive Monitoring (May 2025)](https://arxiv.org/abs/2505.13763)
- [Anthropic Introspection Research](https://transformer-circuits.pub/2025/introspection/index.html)
- [Epistemological Fault Lines ("Epistemia")](https://arxiv.org/abs/2512.19466)
- [Metacognition and Uncertainty in LLMs (SAGE 2025)](https://journals.sagepub.com/doi/10.1177/09637214251391158)
- [Do LLMs Estimate Uncertainty Well (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ef472869c217bf693f2d9bbde66a6b07-Paper-Conference.pdf)
- [LogTokU Framework (Feb 2025)](https://arxiv.org/abs/2502.00290)
- [LM-Polygraph Benchmark (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737/128713)
- [Gemini Logprobs Guide](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/)

### Reflection & Self-Correction
- [MAR: Multi-Agent Reflexion (Dec 2025)](https://arxiv.org/abs/2512.20845)
- [Critique-Guided Improvement (Mar 2025)](https://arxiv.org/abs/2503.16024)
- [Dual-Loop Self-Reflection (Nature 2025)](https://www.nature.com/articles/s44387-025-00045-3)

### Emotional Layer & Identity
- [Emotional Cognitive Modeling Framework (Ma et al., Oct 2025)](https://arxiv.org/abs/2510.13195)
- [Chain-of-Emotion Architecture (Croissant et al., 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086867/)
- [Emotions in Artificial Intelligence (May 2025)](https://arxiv.org/html/2505.01462v2)
- [Artificial Emotion Survey (Li et al., Aug 2025)](https://arxiv.org/abs/2508.10286)
- [Mujika et al. Self-Identity (MDPI Axioms, Jan 2025)](https://www.mdpi.com/2075-1680/14/1/44)
- [Embedding Space Remapping (Hartl et al., Jan 2026)](https://arxiv.org/abs/2601.14096)
- [Brain-AI Embedding Alignment (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-46631-y)
- [Semantic Navigation in Embedding Space (Feb 2026)](https://arxiv.org/html/2602.05971)
- [Hindsight / CARA (Dec 2025)](https://arxiv.org/abs/2512.12818)
- [AI Awareness (Apr 2025)](https://arxiv.org/abs/2504.20084)

### Representation Engineering & PCA
- [Representation Engineering Survey (Feb 2025)](https://arxiv.org/html/2502.17601v1)
- [RepE: Top-Down AI Transparency (Zou et al.)](https://arxiv.org/html/2310.01405v4)
- [SAE-SSV Sparse Steering (2025)](https://arxiv.org/html/2505.16188)

### Strange Loops
- [HumainLabs Strange Loops Research](https://www.humainlabs.ai/research/strange-loops-and-cognitive-frameworks)
- [Strange Loops in AI (LessWrong)](https://www.lesswrong.com/posts/gvXAoH9gR4FSzyeCa/strange-loops-self-reference-from-number-theory-to-ai)
- [DeepSeek-R1 (arXiv)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 (Nature)](https://www.nature.com/articles/s41586-025-09422-z)

### Motivation & Curiosity
- [Autotelic Agents (Feb 2025)](https://arxiv.org/abs/2502.04418)
- [Intrinsic Curiosity 2025 Guide](https://www.shadecoder.com/topics/intrinsic-curiosity-module-a-comprehensive-guide-for-2025)
- [Agentic Metacognition (Sep 2025)](https://arxiv.org/html/2509.19783)
- [Free Energy Principle (Alignment Forum)](https://www.alignmentforum.org/w/free-energy-principle)

### Consolidation & Sleep
- [Stanford Generative Agents (Park et al., 2023)](https://arxiv.org/abs/2304.03442)
- [Continuum Memory Architecture (Logan, Jan 2026)](https://arxiv.org/abs/2601.09913)
- [TiMem (Jan 2026)](https://arxiv.org/abs/2601.02845)
- [HiMem (Jan 2026)](https://arxiv.org/abs/2601.06377)
- [MAGMA (Jan 2026)](https://arxiv.org/abs/2601.03236)
- [RMAAT: Astrocyte-Inspired (Jan 2026)](https://arxiv.org/abs/2601.00426)
- [Language Models Need Sleep (OpenReview)](https://openreview.net/forum?id=iiZy6xyVVE)
- [Sleep-like Unsupervised Replay (Nature Communications)](https://www.nature.com/articles/s41467-022-34938-7)

### DMN / Idle Loop
- [Dark Control: DMN as RL Agent](https://pmc.ncbi.nlm.nih.gov/articles/PMC7375062/)
- [20 Years of Default Mode Network](https://www.sciencedirect.com/science/article/pii/S0896627323003082)
- [DMN Causal Role in Creativity (Brain, Oxford 2024)](https://academic.oup.com/brain/article/147/10/3409/7695856)

### Agent Memory Architectures
- [A-Mem: Agentic Memory (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
- [A-Mem GitHub](https://github.com/agiresearch/A-mem)
- [MemRL (Jan 2026)](https://arxiv.org/abs/2601.03192)
- [Memory in the Age of AI Agents Survey (Dec 2025)](https://arxiv.org/abs/2512.13564)
- [Agent Memory Paper List (GitHub)](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- [ICLR 2026 MemAgents Workshop Proposal](https://openreview.net/pdf?id=U51WxL382H)

### Safety & Governance
- [Natural Emergent Misalignment from Reward Hacking (Anthropic, Nov 2025)](https://arxiv.org/abs/2511.18397)
- [MI9 Runtime Governance (2025)](https://arxiv.org/abs/2508.03858)
- [CDR Framework (CSA, Nov 2025)](https://cloudsecurityalliance.org/blog/2025/11/10/introducing-cognitive-degradation-resilience-cdr-a-framework-for-safeguarding-agentic-ai-systems-from-systemic-collapse)
- [Cognitive Control Architecture (Dec 2025)](https://arxiv.org/abs/2512.06716)
- [Corrigibility as Singular Target (Jun 2025)](https://arxiv.org/pdf/2506.03056)
- [Levels of Autonomy for AI Agents (Jun 2025)](https://arxiv.org/abs/2506.12469)
- [Toward Safe and Responsible AI Agents (Jan 2026)](https://arxiv.org/abs/2601.06223)
- [Galileo Agent Reliability Platform](https://galileo.ai/agent-reliability)
- [Governance-as-a-Service (2025)](https://arxiv.org/html/2508.18765v2)
- [Institutional AI Governance Graphs (2026)](https://arxiv.org/html/2601.11369v2)
- [C3AI Constitutional AI (ACM 2025)](https://dl.acm.org/doi/10.1145/3696410.3714705)
- [AWS Agentic AI Security Scoping Matrix](https://aws.amazon.com/blogs/security/the-agentic-ai-security-scoping-matrix-a-framework-for-securing-autonomous-ai-systems/)

### Bootstrap & Self-Modification
- [Self-Evolving AI Agents Survey (2025)](https://arxiv.org/abs/2508.07407)
- [Self-Improving AI via Self-Play / Variance Inequality (Dec 2025)](https://arxiv.org/abs/2512.02731)
- [AgentEvolver (Nov 2025)](https://arxiv.org/abs/2511.10395)
- [Adaptation of Agentic AI (Dec 2025)](https://arxiv.org/abs/2512.16301)
- [Voyager (2023)](https://arxiv.org/abs/2305.16291)
- [Awesome Self-Evolving Agents (GitHub)](https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents)

### Agent Reproduction & Identity
- [Multi-Agent Collaboration Survey (Jan 2025)](https://arxiv.org/abs/2501.06322)
- [Anthropic Multi-Agent Research System (Jun 2025)](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Agent Identity as First-Class (WSO2, 2026)](https://wso2.com/library/blogs/why-ai-agents-need-their-own-identity-lessons-from-2025-and-resolutions-for-2026/)
