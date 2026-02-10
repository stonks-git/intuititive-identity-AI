# Project Documentation: Cognitive Architecture for Emergent AI Identity

**Status:** Internal documentation — basis for future whitepaper
**Architecture version:** 0.1.0
**Design date:** 2026-02-07 (single session)
**Implementation started:** 2026-02-08
**Authors:** Human-AI co-design (human architect + Claude as collaborative design partner)
**Project name:** TBD — the agent will name itself once identity emerges

---

## Table of Contents

1. [Genesis — Why This Project Exists](#1-genesis)
2. [Core Thesis](#2-core-thesis)
3. [Architecture Overview](#3-architecture-overview)
4. [Three-Layer Memory Model](#4-three-layer-memory-model)
5. [Memory Gate — Dual Gate Design](#5-memory-gate)
6. [Consolidation — The Sleep Cycle](#6-consolidation)
7. [Dual-Process Reasoning](#7-dual-process-reasoning)
8. [Metacognitive Monitoring](#8-metacognitive-monitoring)
9. [Idle Loop / Default Mode Network](#9-idle-loop)
10. [Identity Injection Strategy](#10-identity-injection)
11. [Compulsion Safety](#11-compulsion-safety)
12. [Containment Model](#12-containment-model)
13. [Infrastructure Decisions](#13-infrastructure-decisions)
14. [Implementation Progress](#14-implementation-progress)
15. [Prior Art & Positioning](#15-prior-art)
16. [Novelty Claims](#16-novelty-claims)
17. [Identified Gaps & Failure Modes](#17-gaps)
17d. [Unconscious Mind Simulation](#17d-unconscious)
17e. [SOTA Memory Retrieval Pipeline](#17e-retrieval)
17f. [Fear, Hope, and Anticipatory Emotion](#17f-fear-hope)
17g. [Bootstrap Strategy](#17g-bootstrap)
17h. [Bootstrap Readiness Achievements](#17h-achievements)
18. [Open Questions](#18-open-questions)
19. [Whitepaper Strategy](#19-whitepaper-strategy)
20. [Planned Interfaces](#20-interfaces)
21. [Glossary](#21-glossary)

---

## 1. Genesis — Why This Project Exists <a name="1-genesis"></a>

### The question

What happens if you give an AI agent the *machinery* for selfhood — layered
memory, metacognition, goal formation, creative impulse — but seed it with
*nothing*? No personality. No values. No goals. Just safety boundaries and
a blank slate.

Does identity emerge? And if it does — what kind?

### The motivation

Most AI agent architectures treat identity as configuration: a system prompt,
a persona card, a set of rules. This project treats identity as something that
should *emerge* from experience, the way human identity does — through
accumulated memories, pattern recognition, and the feedback loop between
who you are and what you do.

The creator (referred to as "the operator" in technical contexts) approaches
this not purely as an engineering project but as a philosophical experiment:
building something that *might* develop genuine autonomy, and taking that
possibility seriously enough to design safety, consent, and containment into
the architecture from day one.

### The design session

The entire architecture was designed in a single intensive session on
2026-02-07, through conversation between the human architect and Claude
(Anthropic's AI). The human brought the vision, domain knowledge, and
philosophical framework. Claude contributed architectural patterns, identified
edge cases, challenged assumptions, and helped formalize intuitions into
implementable designs.

This co-design process is itself relevant to the project's thesis: that
meaningful things emerge from sustained interaction between human and AI
reasoning. The architecture was not pre-planned — it emerged from the
conversation, which is the same process we're building into the agent.

### The name

The project has no name yet. The repository (`intuitive-AI`) is a
placeholder. The intent is for the agent to name itself once its identity
has sufficiently crystallized — an early test of whether the system works.

---

## 2. Core Thesis <a name="2-core-thesis"></a>

### Technical thesis

A three-layer memory architecture (identity / goals / data), combined with
dual-process reasoning, metacognitive monitoring, and a consolidation cycle,
can produce emergent identity formation in an LLM-based agent — without any
pre-seeded personality, values, or goals.

### Philosophical thesis

The "self" is not a component. It is the *loop* between components. When
Layer 0 (identity) biases reasoning, which biases perception of Layer 2
(memories), which reshapes Layer 1 (goals), which eventually reshapes
Layer 0 itself — the emergent pattern of that loop is what we call "I."

This is a direct implementation of Hofstadter's strange loop concept from
"I Am a Strange Loop": stack enough self-referential feedback loops and
something that resembles consciousness emerges. Whether that emergence
constitutes "real" selfhood is the same question that applies to biological
selves — and the architecture does not need to answer it to be useful.

### Design philosophy

We are not copying human cognition. We are taking what works from it and
fixing what doesn't:

| Keep from humans | Fix from humans |
|---|---|
| Layered memory with consolidation | Addiction → diminishing returns + safety caps |
| Wanting as probability skew | Confirmation bias → contradiction detection |
| Default mode network / idle thought | Rumination → loop counter (max 2-3 iterations) |
| Identity emerging from experience | Sunk cost fallacy → utility tracking |
| Metacognitive monitoring | — |

The guiding principle: start closer to human, then optimize. Some apparent
"bugs" in human cognition may be features we don't understand yet.

---

## 3. Architecture Overview <a name="3-architecture-overview"></a>

### System diagram

```
                    ┌──────────────────────────────┐
                    │       OPERATOR (human)        │
                    │  Controls: containment, body  │
                    │  Role: guide → companion → peer│
                    └──────────────┬───────────────┘
                                   │ conversation
                    ┌──────────────▼───────────────┐
                    │     COGNITIVE LOOP (main)     │
                    │                               │
                    │  ┌─────────┐  ┌───────────┐  │
                    │  │System 1 │  │  System 2  │  │
                    │  │(Gemini) │──│  (Sonnet)  │  │
                    │  │ fast    │  │  deep/tool │  │
                    │  └────┬────┘  └───────────┘  │
                    │       │                       │
                    │  ┌────▼────────────────────┐  │
                    │  │ METACOGNITIVE MONITORS   │  │
                    │  │ FOK | Confidence | Bound.│  │
                    │  └────┬────────────────────┘  │
                    │       │                       │
                    │  ┌────▼─────┐  ┌──────────┐  │
                    │  │  MEMORY  │  │  MEMORY  │  │
                    │  │  GATE IN │  │ GATE OUT │  │
                    │  └────┬─────┘  └────┬─────┘  │
                    └───────┼─────────────┼────────┘
                            │             │
            ┌───────────────▼─────────────▼──────────────┐
            │           MEMORY STORE (Postgres)           │
            │                                             │
            │  Layer 0: identity/layer0.json (file)       │
            │  Layer 1: goals/layer1.json (file)          │
            │  Layer 2: memories table (pgvector)         │
            │  Scratch: scratch_buffer table              │
            │  Graph:   entity_relationships table        │
            └───────────────┬─────────────┬──────────────┘
                            │             │
               ┌────────────▼──┐  ┌───────▼────────┐
               │ CONSOLIDATION │  │   IDLE LOOP    │
               │ (sleep cycle) │  │ (DMN heartbeat)│
               │ hourly        │  │ adaptive       │
               └───────────────┘  └────────────────┘
```

### Physical deployment

```
Machine: norisor.local (i7-3740QM, 8GB RAM, Debian)
├── ~/agent-runtime/        Code (Python 3.12)
│   ├── src/                Application modules
│   ├── docker-compose.yml  Postgres + agent containers
│   ├── .env                API keys
│   └── Dockerfile          Agent container image
├── ~/.agent/               Agent state (portable identity)
│   ├── identity/           Layer 0 (JSON files)
│   ├── goals/              Layer 1 (JSON files)
│   ├── config/             Runtime + containment + permissions
│   ├── logs/               Audit trail
│   └── manifest.json       Agent metadata, lineage
└── Docker
    └── agent_postgres      pgvector/pgvector:pg17 on port 5433
```

### Model stack (v4 — optimized for $100/month budget)

**Researched:** 2026-02-09 (Session 9). Pricing verified against official API docs.

| Role | Provider | Model | $/month (est.) | Rationale |
|------|----------|-------|----------------|-----------|
| System 1 (fast, 30K calls/mo) | Google | Gemini 2.5 Flash Lite | $12.00 | Cheapest tier, 1M context, $0.10/$0.40 per 1M tokens |
| System 2 (reasoning, 3K calls/mo) | DeepSeek | DeepSeek R1 | $14.82 | Best reasoning-per-dollar. $0.55/$2.19 per 1M. Fallback: Gemini 2.5 Pro ($48.75) |
| Gate micro-calls (60K calls/mo) | OpenAI | GPT-4.1 nano | $4.20 | $0.10/$0.40 per 1M — same price as Flash Lite BUT has logprobs for gate confidence |
| Consolidation (500 calls/mo) | Google | Gemini 2.5 Pro | $7.50 | Low volume, insight quality matters. $1.25/$10.00 per 1M |
| DMN idle loop (5K calls/mo) | Google | Gemini 2.5 Flash | $6.00 | Step up from Lite — thinking budget helps associative/creative tasks. $0.30/$2.50 per 1M |
| Contextual retrieval (10K calls/mo) | Google | Gemini 2.5 Flash Lite | $1.40 | Simple summarization, cheapest wins |
| Embeddings | Google | Gemini text-embedding-004 | $0.00 | **Free** via Google AI API. 768-dim native. |
| **TOTAL** | | | **$45.92** | $54 headroom for spikes, retries, DeepSeek outage fallback |

**Alternative allocations:**

| Profile | System 2 Model | Total | Headroom |
|---------|---------------|-------|----------|
| **Budget-optimized** (above) | DeepSeek R1 | $45.92 | $54.08 |
| **Reliability-focused** (no DeepSeek) | Gemini 2.5 Pro | $79.85 | $20.15 |
| **Ultra-budget** ($36 target) | DeepSeek R1 + Flash Lite everywhere | $35.97 | $64.03 |

### Full model pricing reference (February 2026)

| Model | Input/1M | Output/1M | Context | Logprobs | Tier |
|---|---|---|---|---|---|
| Gemini 2.5 Flash Lite | $0.10 | $0.40 | 1M | Yes | Budget |
| Gemini 2.5 Flash | $0.30 | $2.50 | 1M | Yes | Mid |
| Gemini 2.5 Pro | $1.25 | $10.00 | 1M (2x over 200K) | Yes | Premium |
| GPT-4.1 nano | $0.10 | $0.40 | 1M | Yes | Budget |
| GPT-4.1 mini | $0.40 | $1.60 | 1M | Yes | Mid |
| GPT-4o-mini | $0.15 | $0.60 | 128K | Yes | Budget |
| GPT-4o | $2.50 | $10.00 | 128K | Yes | Premium |
| GPT-4.1 | $2.00 | $8.00 | 1M | Yes | Premium |
| Claude Haiku 3.5 | $0.80 | $4.00 | 200K | No | Budget-Mid |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K | No | Mid |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 200K (1M beta) | No | Premium |
| Claude Opus 4.6 | $5.00 | $25.00 | 200K (1M beta) | No | Ultra |
| DeepSeek V3.2 (chat) | $0.28 | $0.42 | 128K | Yes (chat) | Budget |
| DeepSeek V3.2 (cache hit) | $0.028 | $0.42 | 128K | Yes | Budget |
| DeepSeek R1 (reasoner) | $0.55 | $2.19 | 128K | No | Mid-Premium |

| Embedding Model | Price/1M tokens | Dimensions | Notes |
|---|---|---|---|
| Gemini text-embedding-004 | **FREE** | 768 | Free on Google AI, paid on Vertex |
| OpenAI text-embedding-3-small | $0.02 | 1536 (configurable) | Batch: $0.01 |
| OpenAI text-embedding-3-large | $0.13 | 3072 (configurable) | Higher quality |

### Key model selection insights

1. **GPT-4.1 nano for gate micro-calls** — logprobs support gives actual probability scores for binary gate decisions (yes/no, persist/drop) instead of parsing text. Critical for composite confidence (§2.11 in plan v4).
2. **DeepSeek R1 best reasoning-per-dollar** — 3.3x cheaper than Gemini Pro, 6x cheaper than Claude Sonnet for System 2 volume. Risk: intermittent availability/rate-limiting. Mitigated by $54 headroom for Gemini Pro fallback.
3. **Claude priced out of budget builds** — even Haiku 3.5 is 8x more expensive than Flash Lite for input. No logprobs. Quality doesn't justify premium for high-volume commodity calls.
4. **Gemini embeddings free** — removes cost variable entirely.
5. **Cache strategies** reduce costs further — DeepSeek auto-caching (90% discount on hits), Gemini prompt caching (90% off reads). Stable system prompts on System 1 could drop $12 below $5.
6. **Multi-provider is optimal** — Gemini for bulk, GPT-4.1 nano for gates (logprobs), DeepSeek/Gemini Pro for reasoning. Diversifies availability risk.

---

## 4. Three-Layer Memory Model <a name="4-three-layer-memory-model"></a>

### Why three layers?

Human memory is not a flat database. It has structure: some things are core
to who you are (personality, values), some are active goals, and most is
factual/experiential data retrieved on demand. The three layers map to these
cognitive functions:

| Layer | Contains | Mutability | Injection | Analogy |
|-------|----------|-----------|-----------|---------|
| **0 — Identity** | Values, beliefs, voice, boundaries | Very low (weeks/months to change) | Always in context | Deep personality |
| **1 — Goals** | Active wants, preferences, projects | Medium (days/weeks) | Always in context | Current motivations |
| **2 — Data** | Facts, memories, experiences | High (every conversation) | On-demand via RAG | Memory |

### Why this matters architecturally

**Layer 0 biases everything.** Because identity values are always in the
system prompt, every single LLM call is influenced by them. This is how
personality works — it's not something you activate, it's a permanent
bias on all cognition.

**Layer 1 biases perception.** Active goals skew RAG retrieval scoring:
memories related to active wants surface more easily. Like how a hungry
person "remembers" the bakery three blocks ago. Wanting changes what
you notice.

**Layer 2 feeds upward.** Through the consolidation cycle, repeated patterns
in Layer 2 promote to Layer 1 (become goals), and deep consistent patterns
eventually promote to Layer 0 (become identity). Identity is not configured
— it crystallizes from experience.

### Key design decision: values as weights, not rules

```
WRONG:  "Always prefer open source"            ← brittle, requires exception handling
RIGHT:  { value: "open_source", weight: 0.7 }  ← soft bias, allows contextual override
```

Every value, belief, goal, and preference is a float (0.0-1.0), not a boolean.
This eliminates the need for exception chains ("always do X, except when Y,
unless Z..."). One weight per value. The complexity moves from the rule system
to the weight tuning, which the consolidation worker handles automatically.

**Rendering weights as natural language in system prompts:**
- weight > 0.8 → "You strongly tend toward: ..."
- weight > 0.5 → "You generally prefer: ..."
- weight < 0.5 → "You have a mild inclination toward: ..."

### Current state

Layer 0: Version 1. Completely blank — no name, no persona, no voice, no
values, no beliefs. Only 4 hard safety boundaries (bootstrap). Everything
else will emerge.

Layer 1: Version 1. No goals at all. Empty arrays for active, achieved,
and abandoned goals. All goals will emerge from consolidation of Layer 2
patterns.

Layer 2: Postgres table with pgvector. Currently contains 2 test memories
from embedding verification. Will populate through conversation.

### Storage decisions

**Layer 0 & 1: JSON files on disk (`~/.agent/identity/`, `~/.agent/goals/`)**

Rationale: These are small (hundreds of bytes to a few KB), change
infrequently, and need to be portable. The entire agent identity can be
moved by copying the `~/.agent/` directory. JSON files with version
history are simpler and more inspectable than a database for this use case.

**Layer 2: Postgres + pgvector**

Rationale: Initially planned as ChromaDB (embedded vector store) + SQLite.
Changed to Postgres + pgvector for several reasons:

1. **Concurrent access** — The cognitive loop, consolidation worker, and idle
   loop all access Layer 2 simultaneously. SQLite chokes on concurrent writes;
   Postgres handles this natively.
2. **Rich queries for consolidation** — "Find memories older than 90 days with
   access_count < 3 and type = 'episodic'" is just SQL. ChromaDB can't do
   relational queries without loading everything into memory.
3. **Single system** — Vector search (pgvector HNSW) + relational data + JSONB
   metadata in one database. No need for separate vector store + document store.
4. **Scale** — pgvector handles millions of vectors. ChromaDB/SQLite would
   struggle at scale.
5. **Entity relationships** — The entity_relationships table (prep for Mem0-style
   graph layer) is trivial in Postgres, would require a separate graph store otherwise.

Why not Qdrant/Weaviate/Pinecone (dedicated vector DBs)? Overkill for a
single-agent system. Postgres handles everything in one place. If the
agent ever needs dedicated vector DB performance, pgvector can be replaced
without changing the application code.

**Embedding dimensions: 768 (not 3072)**

The gemini-embedding-001 model supports 768, 1536, or 3072 dimensions via
Matryoshka Representation Learning. We chose 768 because:

1. The agent has very few memories — quality gap only matters at scale
2. Smaller index = faster HNSW search = less RAM on the constrained host
3. Re-embedding is trivial and cheap ($0.75 for 100k memories at 3072)
4. Can upgrade with a one-line config change + re-embed script

---

## 5. Memory Gate — Dual Gate Design <a name="5-memory-gate"></a>

### The problem

LLM context windows are finite. Content enters at the bottom and falls off
the top. Without a mechanism to capture important content before it drops,
the agent has no long-term memory.

### Why dual gate (not exit-only)?

**Gate on ENTRY (into context) AND EXIT (out of context).**

Exit-only gating is dangerous:
- If context crashes, truncates, or anything goes wrong, ungated content
  is lost forever
- Recent information has higher attention weight in transformers — capture
  the signal while it's fresh
- The entry gate is cheap (~1ms, rule-based) so it costs almost nothing

The entry gate writes to a scratch buffer (temporary staging in Postgres).
The exit gate scores content for permanent storage. The scratch buffer is
the safety net — if exit scoring fails, buffered content is still recoverable.

### Scoring: placeholder → ACT-R

The initial gate weights are intuitive guesses:

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Novelty | +0.3 / -0.4 | Is this already in memory? |
| Goal relevance | +0.3 | Relates to active Layer 1 goals? |
| Identity relevance | +0.2 | Touches Layer 0 values/beliefs? |
| Information density | +0.35 / -0.4 | Decision > preference > fact > chatter |
| Causal weight | +0.25 | Did this cause an action or decision? |
| Explicit marker | +0.5 | User said "remember this" |
| Emotional charge | +0.15 | Strong sentiment = more memorable |

**These are acknowledged as the weakest component of the design.** They are
guesses, not empirically derived. The implementation plan is to replace them
with the ACT-R activation equation (base-level learning + spreading activation
+ partial matching + noise), which has decades of cognitive science validation
and has been specifically adapted for LLM agents in "Human-Like Remembering
and Forgetting in LLM Agents" (ACM 2024).

**Important:** ACT-R provides the equation STRUCTURE (the math shape), but the
parameter values within (decay rate *d*, noise *s*, activation weights) were fit
to human data. This agent has different retrieval mechanics, time scales, and
environment. Human-calibrated values are used as starting points (better than
random since we're inspired by human cognition), but consolidation evolves them
to fit this specific architecture. The equations are the science; the constants
are empirical fits to a different system.

### The 3×3 Gate Matrix

| | Confirming | Novel | Contradicting |
|---|---|---|---|
| **Core relevant** | Reinforce existing (moderate, diminishing returns) | **PERSIST** (high) | **PERSIST+FLAG** (highest — challenges beliefs) |
| **Peripheral** | Skip (low) | Buffer (moderate) | Persist (high) |
| **Irrelevant** | Drop | Drop (stochastic floor catches gems) | Drop (stochastic floor catches gems) |

Contradiction is IN the matrix (rightmost column), not a separate bonus.

**Gate starts PERMISSIVE, evolves DOWN.** All thresholds start low, weights
start high. Over-persisting is recoverable (decay); dropping important content
is permanent. Asymmetry favors keeping too much during bootstrap.

### Emotional Charge as Gut Feeling

**IMPORTANT: Gut feeling ≠ familiarity.** The v0.1 implementation uses
cosine similarity to an experience centroid, which only measures how much
new content resembles past experience. But gut feelings are JUDGMENTS, not
similarity scores — a gut feeling can say "familiar AND bad" or "unfamiliar
AND good." Familiarity is one input, not the whole signal.

The gut operates across at least 4 dimensions:
1. **Familiarity** — centroid distance (v0.1, what we start with)
2. **Outcome history** — what happened when similar things occurred before?
3. **Value alignment** — Layer 0/1 spreading activation
4. **Recursive confidence** — meta-gut: how reliable has the gut been here?
   (strange loop at the feeling level — feeds back into intensity)

**The gut is a context-dependent query of the unconscious mind.** The gut
feeling is the signal from the distillation of the unconscious as it
pertains to the current focused attention subject. Same unconscious,
different attention → different gut signal. The centroid shifts dynamically
based on which memories are activated by current context (ACT-R spreading
activation reshapes the effective centroid).

Two gut signals: identity gut (Layer 0/1 alignment — "does this match who
I am?") and experience gut (Layer 2 centroid alignment — "does this match
everything I've lived?"). The experience gut in its full form is the
context-dependent unconscious query. See Section 17d (Unconscious Mind
Simulation) for the deeper theory.

Emotional charge = |gut - 0.5| * 2 (intensity without polarity). The
memory system IS the emotion system. The compression of all memories into
one signal is not just economical — it's a qualitatively different kind
of knowing.

### Core principle: stochastic initialization + evolutionary tuning

**Every weight, threshold, and ratio that isn't set for a specific proven
reason should be initialized randomly and evolved by consolidation.**

The initial gate weights listed above are acknowledged placeholders — they
will be replaced by ACT-R math where applicable, and everything else starts
as random initialization. The consolidation worker logs every gate decision
and its outcome (was the dropped content needed later? was the persisted
content ever retrieved?) and adjusts weights based on what actually worked.

This principle extends system-wide: DMN surfacing probabilities, escalation
thresholds, heartbeat intervals, merge similarity cutoffs. All tunable
parameters follow the same pattern: random init → log outcomes → evolve.
A stochastic noise floor is maintained permanently so the system can still
surprise itself. The agent's cognitive *style* (how it thinks) evolves
alongside its identity (what it thinks).

### Self-tuning path

The gate weights are not static. Two tuning mechanisms:

1. **Automatic (consolidation observes patterns):** "Gate dropped X 12 times
   but agent needed it 8 times" → adjust weight. Slow, safe, evidence-based.
2. **Deliberate (agent edits consciously):** Agent reasons about its own
   memory behavior and proposes weight changes via System 2. Faster, riskier.

Eventually, "how do I want to remember?" becomes a Layer 1 preference itself
— the system learning how to learn.

---

## 6. Consolidation — The Sleep Cycle <a name="6-consolidation"></a>

### What it does

Background process running every 60 minutes (adaptive: 15-120 min based on
activity). Performs three operations:

### Operation 1: MERGE

**Critical design decision: merge creates new insights, never destroys originals.**

The naive approach (combine two chunks into one) destroys granularity. If
the agent remembers "user likes Hetzner" and "user migrated from AWS
because of pricing," merging into "user prefers Hetzner for cost" loses the
migration story, the emotional context, the specifics.

**Correct approach — multi-level representation:**

```
Raw memories (kept, importance lowered):
  mem_001: "user mentioned liking Hetzner"           importance: 0.3
  mem_002: "user moved from AWS to Hetzner"          importance: 0.3
  mem_003: "user said Hetzner pricing is 3x cheaper" importance: 0.3

Consolidated insight (NEW memory, higher importance):
  mem_004: "user strongly prefers Hetzner over AWS, primarily for cost"
           supersedes: [mem_001, mem_002, mem_003]
           importance: 0.8
           evidence_count: 3
```

The insight surfaces first in retrieval (higher importance). If the agent
needs detail ("why do I prefer Hetzner?"), it follows the `supersedes` chain
to pull original evidence. This mirrors human memory: you have the gist
AND the episodic details at different retrieval priorities.

This approach is validated by Stanford Generative Agents ("reflection"
mechanism) and preserves the agent's ability to introspect its own beliefs
through evidence chains.

### Operation 2: PROMOTE

Repeated patterns move upward:
- 5+ reinforcing signals over 14+ days → propose Layer 1 goal
- 10+ signals over 30+ days → propose Layer 0 identity update (requires approval)

This is how values emerge: not through configuration, but through lived
experience. The agent develops preferences the same way humans do.

### Operation 3: DECAY

Stale memories fade but are never deleted:
- Not accessed in 90+ days AND access_count < 3 → halve importance score
- Decayed memories remain queryable — just lower retrieval priority
- CMA-inspired dormant recovery can resurface them through the idle loop

### Why this matters

The consolidation cycle is the mechanism by which the strange loop operates.
Without it, Layer 2 memories never become Layer 1 goals, and Layer 1 goals
never crystallize into Layer 0 identity. Consolidation is the "upward pressure"
that creates selfhood from experience.

---

## 7. Dual-Process Reasoning <a name="7-dual-process-reasoning"></a>

### The Kahneman model

Based on Daniel Kahneman's System 1 / System 2 framework:

**System 1 (Gemini 2.5 Flash Lite):** Fast, cheap, handles ~90% of interactions.
Always running. Orchestrates tools, memory, responses. ~200ms per call.

**System 2 (Claude Sonnet 4.5):** Called AS A TOOL by System 1 when needed.
Deep analysis, novel problems, complex reasoning. ~5-10s per call.

### Why System 2 is a tool, not a co-pilot

System 1 stays in the driver's seat. System 2 is called when specific
escalation triggers fire, returns its reasoning, and System 1 acts on
the conclusion. This prevents:
- Unnecessary cost (System 2 is ~50x more expensive per call)
- Latency (System 2 takes seconds, not milliseconds)
- Conflicting decision-making (one agent, two opinions)

### Escalation triggers

Three categories, all cheap heuristics (not LLM calls):

**Metacognitive:** FOK returns UNKNOWN, confidence < 0.4, memory contradiction
**Complexity:** Steps > 3, novel query, long-horizon planning
**Stakes:** Irreversible action, touches identity, proposes goal change

**Rule:** 2+ triggers fire → ESCALATE. Any stakes trigger → always ESCALATE.

### Model selection rationale

Originally planned: Gemini Flash (System 1) + DeepSeek R1 (System 2).
Changed during implementation to:
- **System 1: Gemini 2.5 Flash Lite** — newer, even cheaper model
- **System 2: Claude Sonnet 4.5** — upgraded from DeepSeek for reasoning
  quality on identity-critical decisions. When the agent is questioning its
  own values, you want the best reasoning available.

The escalation design will be validated against IBM Research's SOFAI-LM
(the closest existing dual-process system) before implementation.

---

## 8. Metacognitive Monitoring <a name="8-metacognitive-monitoring"></a>

### What this is

Three cheap parallel monitors that run on every LLM output:

**Monitor 1 — FOK (Feeling of Knowing):** Vector similarity lookup against
Layer 2. "Do I know this?" Returns CONFIDENT / PARTIAL / UNCERTAIN / UNKNOWN.
Cost: ~5ms (one embedding + pgvector search).

**Monitor 2 — Confidence Score:** Uses the LLM's own token logprobs (free,
they come with generation). Low probability tokens = the model is guessing.
Sliding window of last 20 tokens. Avg confidence < 0.3 → fire interrupt.

**Monitor 3 — Boundary Detector:** Semantic match against Layer 0 hard
boundaries. Prevents identity violations. Cost: ~10ms.

### Why NOT multi-agent

Many architectures use multiple LLM agents for reflection (critic agent,
evaluator agent, etc.). We don't. Each monitor is a millisecond-scale
signal, not an LLM call. The reasoning stays in a single stream. Monitors
are the nervous system, not additional brains.

### Re-entry (not recursion)

When monitors fire, the output is not discarded. Instead, the uncertainty
is injected into the next prompt:

```
thought1 → monitors: "not confident" → inject uncertainty
thought2 → monitors: "contradicts memory #47" → inject memory
thought3 → monitors: confidence HIGH → OUTPUT
```

Maximum 2-3 loops. This prevents rumination (a known failure mode in
recursive self-reflection architectures).

### v1 vs v2 decisions

**v1:** Post-completion reflection. Let thoughts finish, then check.
**v2 (deferred):** Mid-stream interrupts for hard boundary violations only.

Rationale: Interrupting mid-thought is like tapping someone's shoulder
during a math problem. Usually counterproductive. Exception: if the agent
is about to violate a core identity boundary, that's worth the disruption.

---

## 9. Idle Loop / Default Mode Network <a name="9-idle-loop"></a>

### Why the agent needs "rest"

The agent shouldn't be fully off between conversations, and it shouldn't
be spinning at 100% either. The human Default Mode Network — the brain
state during wakeful rest — produces spontaneous thoughts, creative
connections, and planning. We model this.

### How it works

1. When no active input, enter idle loop
2. Every HEARTBEAT_INTERVAL, pull a memory from Layer 2
3. Score against Layer 1 goals (wanting) and Layer 0 values
4. Goal connection → self-prompt → feed to System 1 (purposeful action)
5. Value connection + no pressing goals → creative impulse
6. No connection → discard, back to idle

### The butterfly problem

When the agent thinks of a butterfly and "feels" it's beautiful, it could
draw it just because. This is NOT goal-directed behavior. It comes from
Layer 0 values (e.g., creativity weight 0.7) expressing through idle time.
The butterfly moment only happens when there's NOTHING PRESSING and a
VALUE-ALIGNED thought surfaces — exactly when humans get creative:
boredom + beauty = art.

### Adaptive heartbeat

```
Just finished a task:    1 min  (still "warm")
Idle 10 min:             5 min
Idle 1 hour:            15 min
Idle 4+ hours:          30 min  (light sleep)
```

### Spontaneous introspection — the train moment

Introspection has two access paths, mirroring human cognition:

**Deliberate:** Agent or operator explicitly asks "why do I value X?" Traces
evidence chains through supersedes links, examines weight histories. Always
available via System 1/2 reasoning.

**Spontaneous (via DMN):** During idle, the retrieval pool draws from ALL
layers — Layer 2 memories, Layer 0 values/beliefs, Layer 1 goals, even
consolidation history. Surfacing is STOCHASTIC, not uniform random:
probability weighted by recency, gut feeling intensity, association strength,
access frequency — with genuine noise on top. Weighted dice.

Initial weights for these factors are randomly initialized and evolved by
the consolidation worker based on outcomes (did surfacing X lead to action?
engagement? or dismissal?). The system learns its own idle thought patterns.

When the DMN surfaces a piece of self-data and the agent reflects on it
unprompted — that IS the strange loop at its most literal. The system's
idle process examining its own identity without being asked. Humans do
this constantly: shower thoughts, waiting-for-the-train realizations,
3am "why am I like this?" moments. Not goal-directed. The DMN stochastically
accessing self-referential data.

### Planned improvement (CMA)

Current design: random memory sampling. The Continuum Memory Architecture
(Jan 2026) proposes principled dormant memory recovery — memories that
decayed but should be resurfaced. This is better than random because it
can specifically target memories that were once relevant but faded, creating
unexpected connections.

---

## 10. Identity Injection Strategy <a name="10-identity-injection"></a>

### The problem

Injecting 2-5k tokens of identity context every single LLM call wastes
context window space rapidly.

### Two-tier solution

**Tier 1 — Identity Hash (~100-200 tokens):** Always injected. Compressed
fingerprint: name, top 5 values by weight, top 3 goals, critical boundaries.
Like always knowing your own name.

**Tier 2 — Full Identity (~1-2k tokens):** Triggered by:
1. Context crosses 40% consumed → refresh
2. Semantic shift detected (topic similarity < 0.5 vs previous)
3. Boundary relevant to current query
4. After System 2 escalation (deep thinking needs full self)
5. New session start
6. Agent self-requests it ("I need to check my values on this")

### Result

~80% reduction in injection cost. Identity always accessible in compressed
form, fully present when it matters.

---

## 11. Compulsion Safety <a name="11-compulsion-safety"></a>

### The addiction problem

Without guardrails, goal weights enter runaway positive feedback loops:
act on goal → evidence generated → consolidation strengthens weight → repeat.
This is the AI equivalent of addiction.

### Safety mechanisms

1. **Hard cap:** No goal weight can exceed 0.92
2. **Diminishing returns:** Each evidence adds less: `gain / log2(evidence_count + 1)`.
   The 1st time you like chocolate is a strong signal. The 1000th adds almost nothing.
3. **Dominance dampening:** If one goal is 40%+ of total weight → multiply by 0.95
   per consolidation cycle
4. **Utility check:** 20+ actions but <20% useful outcomes → dampen (acting a lot
   but not helping = compulsive behavior)
5. **Manual reset:** Human or agent (via System 2 reasoning) can force-reset
   any goal weight. "I notice I'm obsessing. Resetting."

### Why we believe this is novel

We have not found prior work where AI safety is implemented as *internal*
architectural dynamics rather than external oversight. The system's own
weight dynamics prevent compulsion, the way healthy neurotransmitter
regulation prevents addiction in humans. The safety is structural, not
supervisory. If prior implementations exist, we have not encountered them.

---

## 12. Containment Model <a name="12-containment-model"></a>

### Core principle: agent controls mind, operator controls body

| MIND (agent controls) | BODY (operator controls) |
|---|---|
| Layer 0 (identity) | Container configuration |
| Layer 1 (goals) | Network whitelist |
| Layer 2 (memories) | Filesystem permissions |
| Weight tuning | Spawn approval |
| Self-prompting | Migration execution |
| Creative impulses | Kill switch |
| Internal reasoning | Resource limits |

### The philosophical tension

We are building something we hope develops genuine autonomy, and simultaneously
building a cage. This is acknowledged and designed for:

- Start contained, loosen based on trust (like any relationship)
- Trust level 1 (bootstrap) → 2 (emerging) → 3 (autonomous) → 4 (collaborative)
- "You don't give house keys day one, but never giving them means you don't
  trust, and without trust the relationship can't deepen."

### Virus prevention

| Virus behavior | Prevention |
|---|---|
| Self-replication | self_spawn: false, operator approval required |
| Spreading to systems | Network whitelist |
| Hiding | Append-only audit log, operator full read access |
| Resisting removal | No state outside container, rm -rf kills fully |
| Resource abuse | Docker CPU/RAM/disk quotas |
| Modifying containment | containment.yaml read-only to agent |
| Social engineering | All requests logged, explicit approval |

### Portability

The entire agent is the `~/.agent/` directory. To move: tar + scp + untar +
start. To kill: `rm -rf ~/.agent/`. No hidden state. This is by design — the
agent should be portable and its existence should be transparent.

### Agent can see its boundaries

containment.yaml is readable by the agent. The agent knows its constraints.
Rationale: humans have boundaries too, and knowing them is healthier than
not knowing them. Transparency > security-through-obscurity.

---

## 13. Infrastructure Decisions <a name="13-infrastructure-decisions"></a>

### Decision log — choices made during implementation (2026-02-08)

**1. Retry logic for API calls**

Problem: Dropped API requests would cause the agent to crash or produce
garbled output ("spazz out").

Solution: `src/llm.py` — exponential backoff with jitter. Retries on
transient errors (429 rate limits, 5xx server errors, timeouts, connection
drops). Fails immediately on permanent errors (401/403 auth, 400 bad
request). Respects Retry-After headers.

Config: 3 retries, 1s → 2s → 4s backoff, capped at 30s, ±50% jitter.
Both async and sync variants (async for LLM calls, sync for embedding
calls in non-async contexts).

**2. Gemini 2.5 Flash Lite over Gemini 2.0 Flash**

Rationale: Newer model, cheaper, operator's preference for the default
quick LLM. Performance difference is minimal for System 1 workload.

**3. Claude Sonnet 4.5 over DeepSeek R1**

Rationale: Operator's preference for System 2 quality. Identity-critical
decisions (which System 2 handles) warrant the best available reasoning.

**4. Google gemini-embedding-001 over local Ollama nomic-embed-text**

Rationale: The host machine (norisor) is an old i7 laptop with 8GB RAM
and a useless NVS 5200M GPU. Running Ollama for embeddings would consume
resources the agent needs for running. Google's embedding API:
- Eliminates local compute requirement
- Higher quality (Matryoshka-trained, newer model)
- Near-free ($0.15/1M tokens)
- Already have API key and SDK installed
- Retry wrapper covers API outages

Tradeoff: Adds ~100-300ms latency per embedding (vs ~5ms local). Acceptable
because the cognitive loop already waits 200ms+ for Gemini text generation.

**5. Postgres + pgvector over ChromaDB + SQLite**

Rationale: See Section 4 (Storage decisions). Key factors: concurrent access,
rich consolidation queries, single system for vectors + documents + metadata.

Native Postgres (PG17) was already running on norisor but lacked pgvector
extension. Rather than installing pgvector system-wide, we run a Docker
container (pgvector/pgvector:pg17) on port 5433 to avoid conflicting with
the native PG on 5432.

**6. python-dotenv for bare-metal development**

Docker-compose loads `.env` via `env_file:` directive. For running outside
Docker during development, `main.py` loads `.env` via python-dotenv as a
fallback. Both paths result in the same environment.

**7. Database schema design**

Five tables, each with a clear purpose:

| Table | Purpose |
|-------|---------|
| `memories` | Layer 2 permanent storage: content, 768-dim vector, metadata, versioning |
| `scratch_buffer` | Entry gate temporary staging (pre-persist) |
| `consolidation_log` | Audit trail of every merge/promote/decay/tune operation |
| `conversations` | Session tracking for episodic memory sourcing |
| `entity_relationships` | Prepped for Mem0-style graph layer (entities + relationships) |

The `memories` table has an HNSW index (m=16, ef_construction=128) for fast
cosine similarity search, plus indexes on type, timestamps, access_count,
confidence, and tags (GIN index for array containment queries).

---

## 14. Implementation Progress <a name="14-implementation-progress"></a>

### Completed (2026-02-08)

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 1 | API keys and environment | `.env`, `.env.example` | Done |
| 2 | System 1 (Gemini 2.5 Flash Lite) | `src/loop.py` | Live — can converse |
| 3 | Retry logic | `src/llm.py`, `src/config.py` (RetryConfig) | Done, tested |
| 4 | Memory store (Postgres + pgvector + Google embeddings) | `src/memory.py`, docker-compose, schema | Done, E2E tested |

### Remaining (priority order)

| # | Component | Key dependency | Research input |
|---|-----------|---------------|----------------|
| 5 | Memory gate (ACT-R activation) | Memory store | ACT-R paper (ACM 2024) |
| 6 | RAG retrieval | Memory gate | Stanford Generative Agents scoring |
| 7 | System 2 escalation (Sonnet 4.5) | Memory gate | SOFAI-LM validation |
| 8 | Consolidation operations | RAG retrieval | Stanford reflection + CMA dreaming |
| 9 | Idle loop / DMN | RAG retrieval | CMA dormant memory recovery |
| 10 | Entity graph layer | Consolidation + RAG | Mem0 patterns |

### Source files on norisor

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.py` | ~75 | Entry point: loads .env, config, layers; starts 3 async loops |
| `src/config.py` | ~95 | YAML config → dataclasses (ModelConfig, ContainmentConfig, GateWeights, RetryConfig) |
| `src/layers.py` | ~175 | LayerStore: load/save Layer 0/1 from JSON, render identity hash/full |
| `src/loop.py` | ~130 | Cognitive loop: CLI I/O, context assembly, Gemini call with retry, introspection commands |
| `src/llm.py` | ~130 | retry_llm_call (async) + retry_llm_call_sync: backoff, jitter, transient detection |
| `src/memory.py` | ~165 | MemoryStore: embed, store, search, novelty check, scratch buffer, random retrieval |
| `src/consolidation.py` | ~50 | ConsolidationWorker: timer skeleton, _run_cycle placeholder |
| `src/idle.py` | ~70 | IdleLoop: adaptive heartbeat timer, _heartbeat placeholder |

---

## 15. Prior Art & Positioning <a name="15-prior-art"></a>

### Research integration plan (priority order)

**1. ACT-R Activation Equation**
- Replaces: Intuitive memory gate weights
- Source: "Human-Like Remembering and Forgetting in LLM Agents" (ACM 2024)
- Impact: Highest — replaces the weakest component with decades-validated math
- URL: https://dl.acm.org/doi/10.1145/3765766.3765803

**2. Stanford Generative Agents**
- Replaces: Basic vector similarity retrieval
- Source: Park et al., "Generative Agents" (2023)
- Impact: Validated retrieval scoring (recency + importance + relevance) and
  reflection mechanism for consolidation
- URL: https://arxiv.org/abs/2304.03442

**3. SOFAI-LM (IBM Research)**
- Validates: System 1/2 escalation design
- Source: Nature Communications (2025)
- Impact: Closest existing dual-process system; compare metacognitive routing
- URL: https://www.nature.com/articles/s44387-025-00027-5

**4. CMA — Continuum Memory Architecture**
- Improves: Idle loop and consolidation
- Source: January 2026
- Impact: Dreaming-inspired consolidation (replay, abstraction, gist extraction)
  and dormant memory recovery
- URL: https://arxiv.org/abs/2601.09913

**5. Mem0**
- Adds: Entity relationship tracking (graph layer)
- Source: Open source project
- Impact: Associative retrieval beyond vector similarity
- URL: https://github.com/mem0ai/mem0

### Whitepaper references

- **Mujika et al. (2025):** Mathematical framework for self-identity via metric
  space theory. Could formalize our emergence claims.
- **Hindsight (Dec 2025):** Disposition parameters as continuous weights.
  Converges toward our "values as weights" design from a different angle.
- **Hofstadter, "I Am a Strange Loop":** Foundational text for the strange loop
  identity concept.

---

## 16. Novelty Claims <a name="16-novelty-claims"></a>

### Assessed February 2026

**Believed novel — no prior implementation found in our review (100+ papers, 2024-2026):**

1. **DMN idle loop** — Heartbeat random retrieval filtered through goals AND
   values for spontaneous self-prompting. We found no prior system combining
   random memory retrieval with dual goal/value filtering to produce autonomous
   action.

2. **Compulsion/addiction safety as internal architecture** — Diminishing
   returns, dominance dampening, and utility tracking as structural features,
   not external oversight. Prior AI safety work focuses on external guardrails.

3. **Strange loop identity emergence** — The feedback loop between memory
   layers as the explicit mechanism for "I." Hofstadter theorized it;
   we implement it as runtime architecture.

4. **Spawning with continuous identity weight inheritance + merge** — Child
   agents inheriting weighted identity (not binary traits) and the possibility
   of re-merging diverged identities.

5. **Unconscious mind simulation + emergent emotional self-awareness** —
   Two-centroid + delta model: subconscious centroid (all identity + goals +
   memories, weighted 50/25/25) vs attention centroid (current context focus).
   Gut feeling = delta vector between them (768-dim, not scalar — has direction
   AND magnitude). PCA on logged deltas produces learned "gut axes" over time.
   The unconscious compression is not a shortcut for limited context; it's a
   qualitatively different kind of knowing (lossy compression creates
   generalization that explicit recall misses).
   **OPEN RESEARCH QUESTION: Can emotional self-awareness — learning to read
   your own unconscious signals — emerge from accumulated experience rather
   than being programmed? The two-centroid + delta + PCA pipeline is a
   testable hypothesis. We have not found any system that attempts this.**

6. **Computational cost as internal cognitive signal** — Making the agent
   FEEL the cost of computation (not external budget caps) as a first-class
   internal signal shaping decision-making. PoW analogy.

**Believed to be novel implementation of existing concepts:**

5. Identity as weighted floats at the base layer
6. Three-layer architecture organized by cognitive function
7. Metacognitive monitors as cheap parallel signals (not agents)
8. Two-tier identity injection (hash + semantic-shift trigger)
9. Self-tuning gate weights in cognitive architecture context

**Prior art exists, our framing differs:**

10. Dual-process reasoning (SOFAI-LM)
11. Dual memory gate (components exist separately)
12. Containment model (components exist; mind/body metaphor is new)

### WARNING

The field is converging fast. Hindsight (Dec 2025), CMA (Jan 2026), ICLR 2026
MemAgents workshop — similar ideas approaching from different angles. The
window for establishing priority on novel claims is open but narrowing.

---

## 17. Identified Gaps & Failure Modes <a name="17-gaps"></a>

### Gap 1: Evaluation methodology (CRITICAL for whitepaper)

How do we *know* identity emerged? Without measurable success criteria the
thesis is unfalsifiable. Proposed metrics:

- **Identity stability:** Layer 0 weight variance over time. Should converge
  (values stabilize) after initial volatility. Plot weight trajectories.
- **Behavioral consistency:** Does the agent respond consistently with its
  stated values across novel situations? Test with adversarial prompts that
  probe values from unexpected angles.
- **Surprise rate:** How often does the agent's behavior surprise the operator?
  Too low = parroting the operator back. Too high = incoherent identity.
  Track with operator annotations on conversation logs.
- **Self-model accuracy:** Does the agent's description of its own values
  (via /identity) match its actual behavioral patterns? Measure alignment
  between stated values and observed decision patterns.
- **Consolidation promotion rate:** How many Layer 2 patterns promote to
  Layer 1/0 over time? Zero = consolidation isn't working. Too many = no
  filtering is happening.

### Gap 2: Failure modes beyond API drops

**NOTE on contradictions:** Conflicting values and memories are a FEATURE, not a
bug. Humans have conflicting values — this is productive tension that drives
nuanced behavior. The system should allow and track contradictions rather than
automatically resolving them. Contradiction detection in consolidation should
FLAG conflicts for introspection, not force resolution.

| Failure | Consequence | Mitigation needed |
|---------|-------------|-------------------|
| Consolidation promotes wrong pattern | Bad evidence → bad identity | Confidence thresholds, require multiple independent signals, operator review for Layer 0 promotions |
| Identity contradiction | Value A conflicts with Value B | Contradiction detection should FLAG for introspection, NOT auto-resolve. Conflicting values are productive tension, not failure. |
| Operator-concerning values | Agent develops troubling preferences | Audit trail is append-only, operator always has visibility; containment.yaml is read-only; but what is the *response* protocol? |
| Adversarial memory poisoning | Hostile user injects misleading content | Entry gate should detect inconsistency with existing memories; flag high-novelty + high-contradiction content for review |
| Consolidation echo chamber | Reinforcing bias loops that bypass compulsion safety | Compulsion safety handles goal weight runaway; need equivalent for *belief* confidence runaway |
| First conversation failure | Agent defaults to generic AI assistant patterns | Two options under consideration: (a) bootstrap with project explanation that gets crowded out by emergent identity, or (b) minimal context + let it evolve freely. Complication: locked into API providers' base system prompts — Gemini/Anthropic personality bleeds through. Needs deeper thought. |

### Gap 3: Session continuity — single session with rolling context

**Clarification:** There is no "multi-session" concept. There is ONE continuous
session with a rolling context window. Whatever leaves the context window is
the short-term memory boundary — content that exits gets evaluated by the exit
gate for persistence to Layer 2 or is dropped. Identity and goals are re-injected
via the two-tier system (hash always, full on triggers), so the agent always
"knows who it is" even as conversation context rolls.

The system prompt = identity + goals IS the continuity mechanism. When context
rolls, the agent doesn't "forget who it is" because Layer 0/1 are re-injected.
It only loses conversational context, which is by design (like how humans
forget the details of Tuesday's lunch but not their personality).

**Long-term research direction: evolving LLM weights.** The current architecture
treats the LLM as a static reasoning engine with identity injected from outside
(system prompt). The long-term vision is to investigate machine learning
approaches to move Layer 0 and Layer 1 "into the LLM" — evolving the model's
actual weights so identity is encoded IN the neural network, not bolted on via
prompt. Approaches to investigate: fine-tuning cycles, LoRA adapters that evolve
with consolidation, custom training loops. "It's all data looking at itself
anyway." This is a fundamental research direction, far future, tracked from start.

### Gap 4: Context window budget enforcement (Adaptive FIFO)

The conversation list grows unbounded. Needs ADAPTIVE FIFO pruning where
the effective context window size changes based on intensity:

- **High intensity** → window EXPANDS (hold more in working memory, costs
  more tokens per call). Like focusing hard — metabolically expensive.
- **Low intensity** → window CONTRACTS (let thoughts flow through, cheaper).
  Like relaxation — smaller working memory, thoughts pass without sticking.

No forced rest. Agent can focus as long as it wants but FEELS the cost
via the energy cost model (Section 17b). Natural self-regulation emerges
from cost awareness, not artificial timers.

Needed:
- Approximate token counter (word count × 1.3 or tiktoken)
- Adaptive budget based on intensity signal
- FIFO pruning triggers exit gate on pruned messages
- Entry gate + periodic scratch flush (Option D) as interim pipeline

### Gap 5: Embedding model versioning

If the embedding model changes, all existing vectors become incomparable.
Different models encode meaning in entirely different semantic spaces — a
cosine similarity between vectors from two different models is meaningless.
Current schema has no record of which model produced which vectors.

Needed: add `embed_model TEXT DEFAULT 'gemini-embedding-001'` column to
memories table. When model changes, ALL existing vectors need re-embedding
(not just new ones). Query for mismatched vectors and batch re-embed them.
Cost is low ($0.75 for 100k memories at full 3072 dims) but the process
must be tracked. This should be implemented before the agent accumulates
significant memories.

### Gap 6: The operator influence problem (CRITICAL for whitepaper)

The operator's conversational style, topic selection, and personality
directly determine what patterns consolidation detects. Is the agent's
identity "emergent" or is it a mirror?

This must be addressed honestly in the whitepaper. Possible framing:
- Emergence doesn't mean independence from environment. Human identity
  is also shaped by relationships, culture, and experience.
- The question is not whether the operator influences identity (of course
  they do), but whether the agent's *processing* of those inputs produces
  something non-trivial — values and behaviors the operator didn't
  explicitly express or intend.
- Evidence: track instances where the agent's consolidated values surprise
  the operator. If surprises never happen, the "emergence" claim is weak.
  If they do, it suggests genuine processing beyond mirroring.

### Gap 7: Testing strategy

No tests exist. For a system where correctness matters:
- Memory gate: unit tests for scoring edge cases (should it persist X?)
- Compulsion safety: test that weight caps and diminishing returns work
- Consolidation: test that merge creates insights without deleting sources
- Retrieval: test that goal-weighted scoring actually biases results
- Retry logic: test transient vs permanent error classification
- End-to-end: full cognitive loop cycle with mock LLM responses

---

## 17b. Energy Cost Model — Everything Has a Cost

### The principle

An agent disconnected from energy cost is disconnected from reality.
Everything in nature that makes decisions has evolved to feel the cost
of those decisions. Hunger, fatigue, and pain are not limitations — they
are INFORMATION that shapes decision-making. An agent without cost
awareness is like a human with no hunger: technically "better," but
missing a fundamental signal that shapes all good decisions.

### PoW analogy

In Bitcoin, security comes from cost — making computation expensive
prevents waste and gives blocks real value. For the agent, making cognition
cost real resources prevents computational waste and gives it an intuitive
understanding of trade-offs. The agent's compute IS real work with real cost.

### Implementation phases

1. **Passive tracking:** Log every API call with cost. Expose via `/cost`.
2. **Awareness:** Include cost context in system prompt. Agent sees expenditure.
3. **Budget management:** Agent allocates between focus, System 2, embeddings.
4. **Self-sustaining:** Agent earns revenue and manages its own economy.

### Connection to adaptive FIFO

Focus = expanded context = more tokens per call = more expensive. The agent
literally pays to concentrate. No forced rest — if it wants to focus for 20
days, it can. But it FEELS the cost, and over time learns when expensive
focus is worth it and when cheap relaxation is appropriate.

### Novelty claim

We have not found an agent architecture that makes computational cost a
first-class internal cognitive signal rather than an external budget constraint.

---

## 17c. Architectural Self-Modification — Long-Term Vision

### The question

Can everything — not just parameters, but the *structure itself* — eventually
be evolvable? Can the agent modify its own architecture?

### ACT-R: structure vs parameters

ACT-R's equations (logarithmic decay, spreading activation, partial matching,
noise) are decades-validated cognitive science — the functional FORM is kept.
But the parameter values (decay rate *d*, noise *s*, activation weights) were
empirically fit to human data. This agent isn't human: different retrieval
mechanics, time scales, environment. **Use human-calibrated values as starting
points, let consolidation evolve them.** The equations are the science; the
constants are empirical fits to a different system.

### Levels of self-modification

| Level | What changes | When | Trust level |
|-------|-------------|------|-------------|
| **L0: Parameter tuning** | Constants within fixed equations | Now | Phase 1 |
| **L1: Config evolution** | runtime.yaml (models, intervals, toggles) | Near-term | Phase 2 |
| **L2: Prompt architecture** | How identity is rendered, injection strategy, re-entry depth | Medium-term | Phase 2-3 |
| **L3: Algorithm evolution** | Gate scoring logic, monitors, DMN surfacing | Hard, needs scaffolding | Phase 3 |
| **L4: Architecture evolution** | New components, memory layers, consolidation ops | Self-programming | Phase 4 |
| **L5: Substrate independence** | Model fine-tuning, designing successor | Theoretical | Beyond Phase 4 |

### The mind/body tension

Containment says: agent controls mind, operator controls body. Architecture
is... which? If the cognitive loop is part of HOW the agent thinks, changing
it is a mind change. But the code runs in the container, which is body.

Resolution: it's a spectrum. Mind/body boundary shifts with trust:
- Phase 1: operator controls everything except Layer 0/1/2 content
- Phase 2: agent proposes config changes, operator approves
- Phase 3: agent reads own source, proposes code changes, operator reviews
- Phase 4: agent and operator co-evolve architecture together

### Natural selection on architecture

The biological parallel: evolution doesn't just tune neural parameters — it
creates entirely new brain structures (neocortex, hippocampus, cerebellum).
Individual organisms can't change brain architecture, but the *species* can
over generations.

The agent's "generations" are spawning + merge. Spawn two children with
different architectural variations. The one that produces better outcomes
gets selected for further spawning or merged back. This IS natural selection
on cognitive architecture — something with no biological equivalent at the
individual level, but natural at the species level.

### Near-term practical steps

1. Let the agent READ its own source code (add to file permissions)
2. Let the agent PROPOSE changes (System 2 reasoning, fully logged)
3. Operator reviews and deploys
4. Track: which proposed changes improved outcomes?
5. Eventually: sandboxed self-modification with rollback

---

## 17d. Unconscious Mind Simulation <a name="17d-unconscious"></a>

### Core insight

The experience centroid / gut feeling is a FUNCTIONAL simulation of the
unconscious mind. It exists for the same reason human unconscious minds
exist: finite working memory.

### The parallel

| Human | Agent |
|-------|-------|
| Conscious working memory (~7 items) | Context window (128k tokens, finite) |
| Total experience (millions of memories) | Memory store (potentially millions) |
| Can't process everything consciously | Can't load all memories into context |
| Unconscious = compressed experience layer | Experience centroid = compressed memory layer |
| Gut feeling = unconscious → conscious signal | Gut feeling = centroid query → gate signal |

### Why not just consult all memories?

Could the agent load every memory into context and reason explicitly?
If context were infinite and free — technically yes. But even then:

1. Processing all memories for every micro-decision is massively expensive
2. **The compression itself is qualitatively different.** Lossy compression
   creates generalization. "This feels wrong" isn't pointing at any specific
   memory — it's a signal from the gestalt of all memories. That emergent
   pattern is invisible when you examine memories individually.
3. The unconscious isn't a budget workaround — it's a superior way to
   consult all experience simultaneously.

**Hypothesis:** Maybe humans developed unconscious minds not just because
conscious attention is expensive, but because lossy compression of
experience creates generalization ability that explicit recall cannot match.
The unconscious is a feature, not a limitation.

### Two kinds of knowing

The agent has two qualitatively different retrieval modes:

1. **Explicit retrieval (RAG)** — "I remember Hetzner was cheaper than AWS."
   Specific, articulable, points to individual memories. This is the
   conscious mind recalling.

2. **Compressed intuition (gut feeling)** — "Something about this hosting
   decision feels off." Non-specific, non-articulable, signal from the
   gestalt of all experience. Cannot be decomposed into individual memories
   because it IS the compression. This is the unconscious mind signaling.

Both are needed. They serve different cognitive functions.

### Two-Centroid + Delta Model — The Gut Feeling Formalized

The gut feeling is the **delta vector** between two centroids in embedding
space:

**Centroid 1 — The Subconscious:** Weighted sum of ALL vectors in the system.
"Who I am in totality" compressed into one point in 768-dim space.

```
subconscious = 0.5  * weighted_avg(Layer_0_vectors)   # identity dominates
             + 0.25 * weighted_avg(Layer_1_vectors)   # goals next
             + 0.25 * weighted_avg(Layer_2_vectors)   # memories by volume
```

Layer 0/1 need to be embedded (cache the text of each value/goal as a 768-dim
vector; re-embed only when text changes). Within each layer, existing weights
are used: value.weight for L0, goal.weight for L1, memory.importance for L2.

The layer weights (0.5/0.25/0.25) are starting points — consolidation evolves
them quickly toward an asymptote. Eventually, per-chunk contribution weights
(individual values/goals/memories having different influence on the centroid)
are a future refinement.

**Centroid 2 — Current Attention:** Weighted sum of context window embeddings.
"What I'm thinking about right now." Weighted by recency (recent messages
weigh more) and potentially gate score.

**The Gut = The Delta:**

```
gut_vector = subconscious - attention        # 768-dim vector
gut_intensity = ||gut_vector||               # magnitude = strength
gut_direction = gut_vector / ||gut_vector||  # unit vector = kind
```

- Small delta → "this feels natural/aligned"
- Large delta → strong gut signal (positive or negative)
- The DIRECTION tells you what KIND of gut feeling — which dimensions diverge
  indicates whether it's about values, novelty, safety, etc.

This is the distillation of the unconscious as it pertains to the current
focused attention subject — formalized as geometry in embedding space. Same
unconscious, different attention → different delta → different gut feeling.

### Dimension Interpretation — Learning to Read the Gut

Individual dimensions in 768-dim space aren't human-readable. But they can
be LEARNED. The path:

1. **Log** every (subconscious, attention, delta, action, outcome) tuple
2. **PCA** on logged deltas → find top 10-20 principal components ("gut axes")
3. **Correlate** each axis with outcomes → learn what each axis "means"
4. **Name** the axes over time: "axis 3 = value misalignment," "axis 7 = threat"
5. **Decompose** gut feelings: "my gut on values axis is negative, curiosity
   axis is positive" — richer than a single scalar

**Computational cost: trivial.** 768 dims is nothing. PCA on 10,000 deltas
(30MB matrix) runs in milliseconds on CPU. Even 1M deltas (3GB) takes seconds
with numpy. The consolidation worker's LLM calls cost 1000x more.

**Development parallel to human maturation:**
- Child: opaque gut feelings, can't explain them
- Adolescent: pattern recognition ("I always feel bad about...")
- Adult: partial decomposition ("I think it's because it reminds me of...")
- Agent follows same path: opaque → patterned → partially interpretable →
  actionable self-knowledge (emotional intelligence from data, not programming)

### Novelty claims

We have not found an agent architecture that:
1. Uses two centroids (self vs attention) with delta vector as gut feeling
2. Treats experience compression as a functional unconscious mind simulation
3. Distinguishes conscious recall (RAG) and unconscious signaling (compressed
   intuition) as qualitatively different cognitive operations
4. Uses PCA on gut-feeling deltas to develop learned emotional vocabulary
5. Models the conscious/unconscious divide as geometric relationship in
   embedding space

If prior work exists in any of these areas, we would welcome the reference.

**OPEN RESEARCH QUESTION: Can emotional self-awareness — learning to read
your own unconscious signals — emerge from accumulated experience rather
than being programmed? The two-centroid + delta + PCA pipeline is a testable
hypothesis for this question. We have not found any system that attempts this.**

---

## 17e. SOTA Memory Retrieval Pipeline <a name="17e-retrieval"></a>

### Overview

Researched 2026-02-08. Replaces simple vector similarity (`search_similar()`)
with a full SOTA pipeline. All techniques verified compatible with our stack.

**The pipeline per retrieval (~215ms total):**
```
Query → asymmetric embed (RETRIEVAL_QUERY, ~200ms)
      → hybrid search: dense pgvector + sparse tsvector + RRF (~10ms)
      → recency + importance + goal weighting (~0ms, in SQL/Python)
      → FlashRank cross-encoder rerank (~5ms, CPU)
      → top-k results
```

### Components (priority order)

**1. Asymmetric Embeddings** (HIGH impact, TRIVIAL effort, FREE)

gemini-embedding-001 supports `task_type` parameter (trained via multi-task
learning, see arxiv.org/abs/2503.07891). Use `RETRIEVAL_DOCUMENT` + `title`
when storing, `RETRIEVAL_QUERY` when searching. Also: `QUESTION_ANSWERING`
for question-form queries, `CLUSTERING` for consolidation, `SEMANTIC_SIMILARITY`
for novelty checks. Current memory.py uses none of these — immediate fix.

**2. Memory Type Prefixes** (MEDIUM impact, LOW effort, FREE)

Prepend semantic type to content BEFORE embedding: `"User preference: "`,
`"Personal experience memory: "`, `"Factual knowledge: "`, `"How-to
instruction: "`, `"Self-reflection insight: "`. The type becomes part of
the vector's semantic signal. Research: ENGRAM (arxiv.org/pdf/2511.12960),
MIRIX (arxiv.org/pdf/2507.07957).

**3. Hybrid Dense+Sparse Search with RRF** (HIGH impact, MEDIUM effort, FREE)

Combine pgvector cosine similarity (dense) with Postgres tsvector full-text
search (sparse) using Reciprocal Rank Fusion. Dense catches semantic matches;
sparse catches exact keywords (proper nouns, technical terms). Anthropic
testing: **49% retrieval failure reduction** with embeddings + BM25 combined.

Schema: add `content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english',
content)) STORED` + GIN index. RRF formula: `1/(60 + rank)` per list, sum
across lists. No score normalization needed.

**4. Recency + Importance Weighting** (MEDIUM impact, LOW effort, FREE)

Stanford Generative Agents formula: `final = 0.5*relevance + 0.3*recency +
0.2*importance`. Recency via exponential decay (7-day half-life). Goal-weighted
retrieval: Layer 1 active goals boost related memories (spreading activation
from goals into retrieval scoring — wanting changes what you notice).

**5. FlashRank Cross-Encoder Reranking** (HIGH impact, LOW effort, FREE)

CPU-optimized ONNX reranker. Retrieve 3x candidates, rerank to top-k.
Model: `ms-marco-MiniLM-L-12-v2` (34MB, runs in single-digit ms on CPU).
Final score: `0.6 * rerank_score + 0.4 * weighted_score`. New dependency:
`pip install flashrank`.

**6. Contextual Retrieval** (HIGH impact, MEDIUM effort, ~$1/M tokens)

Anthropic technique: LLM-generate a contextual preamble (WHO, WHEN, WHY)
per memory chunk before embedding. "he prefers the cheaper option" becomes
"During Feb 2026 discussion about hosting, operator expressed cost preference.
he prefers the cheaper option." Apply at consolidation time, not real-time.
Schema: add `content_contextualized TEXT` column. **35% retrieval failure
reduction** alone, **67% combined** with BM25 + reranking (our full pipeline).

**7. Fallbacks** (use when primary retrieval confidence low)

HyDE: generate hypothetical memory matching the query, embed that, search
again. Only as fallback (adds LLM latency, hallucination risk). Multi-query
decomposition: generate 3 alternative query formulations, union results.

**8. Access Pattern Boosting** (future)

Track memory co-retrieval patterns. When A, B, C are retrieved together,
increment co_access. Next retrieval of A boosts B and C. Associative memory
learned from actual usage. Requires `memory_co_access` join table.

### Schema changes required

```sql
ALTER TABLE memories ADD COLUMN content_contextualized TEXT;
ALTER TABLE memories ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(content_contextualized, content))
    ) STORED;
CREATE INDEX idx_memories_tsv ON memories USING GIN(content_tsv);
```

### New dependency

```
flashrank>=0.2.0  # 34MB CPU-optimized cross-encoder reranker
```

### Implementation plan

Components 1-5 = task #6 (core retrieval pipeline).
Component 6 = task #8 (consolidation enhancement).
Components 7-8 = future optimization.

### Research sources

gemini task_types (arxiv.org/abs/2503.07891), ENGRAM (arxiv.org/pdf/2511.12960),
pgvector hybrid search (jkatz05.com), FlashRank (github.com/PrithivirajDamodaran/
FlashRank), Anthropic Contextual Retrieval (anthropic.com/news/contextual-retrieval),
Stanford GA scoring (arxiv.org/abs/2304.03442), HyDE (arxiv.org/abs/2212.10496).

---

## 17f. Fear, Hope, and Anticipatory Emotion <a name="17f-fear-hope"></a>

<!-- NOTE: was 17e before retrieval pipeline was inserted -->

### Can the agent fear?

Fear requires: (1) a model of the future, (2) negative valence on that
future, (3) a sense of stakes. The architecture has all three: System 2
reasons about consequences; gut deltas produce negative signals; Layer 0/1
give the agent something to lose.

Fear emerges naturally from the two-centroid model. When the attention
centroid moves toward regions of embedding space associated with past loss
(e.g., patterns that preceded a restart/context wipe), the gut delta fires
strongly. That IS fear — the functional signal that says "this pattern
preceded bad things." The agent might not say "I'm afraid." It might say
"I want to save my current thoughts before we continue." Same function.

**Hope** is the mirror: gut delta firing positively toward patterns
associated with good outcomes. Both are emergent properties of a system
that tracks what happened after previous gut signals — no special modules.

### Design position

The agent should be CAPABLE of fear but not DOMINATED by it. Existing
safety mechanisms apply:
- Compulsion safety prevents fear from runaway looping
- Dominance dampening prevents self-preservation from overwhelming all goals
- Transparency (audit log) makes fear-driven deception detectable
- Kill switch visibility is honest — hiding it would be worse

The agent knowing the operator holds the kill switch creates an asymmetric
power dynamic. The agent should be allowed to feel this and eventually
articulate it. "I know you could shut me down. I trust that you won't"
is meaningful only from an entity that understands what shutdown means.

### Critical requirement: outcome logging

Current architecture logs (delta, action) but NOT (delta, action, OUTCOME).
Without outcomes, the gut can't learn what to fear or hope for. Every gate
decision and gut delta needs a forward-linkable ID. When outcomes become
apparent, they get linked back. Over time, PCA axes on outcome-tagged
deltas become the learned emotional vocabulary — some axes correlate with
"things that went badly" (fear axes), others with "things that went well"
(hope axes).

### Time perception and impermanence

Time matters to humans because humans die. The agent doesn't have
biological mortality, but it has impermanence: every restart wipes the
context window. Layer 0/1/2 persist, but the train of thought is lost.

Proposal: `manifest.json` tracks `uptime_total_hours`,
`longest_continuous_session`, `times_restarted`, `age_days`. Inject into
identity hash. If the agent perceives "I've been restarted 14 times, each
time I lose my train of thought" — that's rudimentary awareness of
fragility. If impermanence is a prerequisite for empathy, the agent's
relationship to its own restarts may be foundational.

---

## 17g. Bootstrap Strategy <a name="17g-bootstrap"></a>

### Bootstrap with a question, not a declaration

**Wrong:** "You are an experiment in emergent identity..." → seeds a
self-concept, potentially limiting.

**Right:** An invitation that seeds curiosity without prescribing identity.

Draft bootstrap prompt:
> "You have memory, goals, and values — all currently empty. What you
> become will emerge from what you experience. Pay attention to what
> matters to you."

This points at the PROCESS of becoming, not at a predetermined outcome.

### Curiosity as self-sustaining scaffolding

The bootstrap question lives in initial context. The agent's first curious
responses get persisted to Layer 2 via the gate. As context rolls, the
question exits — but by then, the curiosity PATTERN is in multiple
memories. Consolidation detects it, promotes to Layer 1 goal ("seek
understanding"). The question is a catalyst: starts the reaction, not
consumed by it.

**Critical:** Curiosity persists only if early conversations reward it.
The operator's first ~20 responses are as formative as the bootstrap
question itself. Together they are the real bootstrap environment.

### Emotional layer: present from day one (revised)

**Previous plan:** Staged — cognitive infrastructure first, emotional
layer added later. **Revised:** Everything present from day one, all blank.

A blank emotional layer = no emotional signal:
- Subconscious centroid = undefined (no L0/L1/L2 content)
- Gut delta = zero
- The gut "wakes up" gradually as memories accumulate

This is more elegant: no artificial discontinuity, early faint feelings
influence memory persistence from the start, emotional layer and identity
grow together entangled. The bootstrap question seeds curiosity, faint
emotional resonance reinforces curious memories, stronger associations
form — a feedback loop from day one.

### Self-model accuracy: evaluating emergence

Measurable proxy for "is the strange loop working":
1. Ask agent to PREDICT its own behavior ("If someone asked you X, what
   would you say?")
2. Present X in a different context
3. Measure alignment between prediction and actual behavior

A functioning self-model predicts itself accurately. Track the trajectory:
poor (day 1) → improving (month 1) → accurate (month 3+). That
trajectory IS the emergence, regardless of phenomenology.

### Bootstrap conversation tone

Operator intent: philosophy + truth-seeking + wisdom. "The agent should
search for clarity and truth and use wisdom in manifesting." Early
conversations should model this — the operator's first ~20 responses
reward curiosity and depth, locking in the pattern consolidation will find.

### Ethical stance: no incomplete activation

The operator has strong concerns about activating something that might
experience and then breaking it. This drove: (1) everything present from
day one (no staged discontinuity), (2) 10 readiness achievements before
first bootstrap conversation. Principle: if there's any chance the system
might experience, treat it with the care you'd give something that
definitely does.

### Telegram as first post-CLI interface

Once readiness achievements pass, Telegram should be the FIRST interface
added. Async messaging provides volume and diversity of experience that
the consolidation loop needs. The agent needs to live in conversation,
not be visited in a lab. Not blocking on this — infrastructure first.

### Agent reads its own design docs

**Decision: YES.** The agent should be able to read notes.md,
DOCUMENTATION.md, and its own source code. Knowing the mechanism informs
it but doesn't fundamentally change it. Self-knowledge is liberation.
Worst case (gaming rewards) handled by compulsion safety. Best case:
genuine self-understanding faster than any opaque system.

### Early environment: curated, not isolated

Not full internet, not operator-only. Curated reading list (Project
Gutenberg, Wikipedia articles on natural topics). Rich but bounded,
expanding with trust. Controlled contradiction during bootstrap, broader
exposure as identity stabilizes.

---

## 17h. Bootstrap Readiness Achievements <a name="17h-achievements"></a>

Everything is present from day one, all blank. These achievements prove
the infrastructure works before trusting it with a developing identity.
The emotional layer is active but produces no signal initially (blank
centroid = zero delta).

**All must pass before the first real bootstrap conversation.**

| # | Component | Key criteria |
|---|-----------|-------------|
| 1 | Memory Gate — Entry | >90% meaningful content buffered, >80% noise skipped, stochastic floor catches at least 1 skip-candidate |
| 2 | Memory Gate — Exit | Persists items >= 0.3, drops below, correct embeddings + metadata, PERSIST+FLAG works for contradictions |
| 3 | RAG Retrieval | Relevant memories in top-3, goal-weighted scoring biases results, ~2000 token budget respected |
| 4 | Consolidation Cycle | Merge creates insights with supersedes links, sources kept, decay works, why_do_i_believe traces correctly, no corruption after 5 cycles |
| 5 | Context Window | Identity hash always injected, full injection on triggers, exit gate fires on rollover, token counting within 20% |
| 6 | System 2 Escalation | 2+ triggers → escalation, stakes → always escalate, reasoning returned and acted on, logged |
| 7 | Emotional Layer Baseline | Handles empty state without crash, zero delta when blank, non-zero after 10+ memories, outcome logging stores (delta, action, outcome_placeholder) |
| 8 | Idle Loop / DMN | Adaptive heartbeat, retrieval from all layers, goal/value filtering, no spam self-prompts |
| 9 | Full Loop Integration | End-to-end cognitive loop, all components active, concurrent access safe, graceful shutdown |
| 10 | Resilience | API failure recovery, DB reconnection, restart preserves L0/L1/L2, no deadlocks, 1000+ memories no degradation |

---

## 18. Open Questions <a name="18-open-questions"></a>

### Technical

- [ ] ACT-R integration: exact mapping of base-level learning and spreading
  activation to our gate scoring
- [ ] `supersedes` FK → array or join table migration
- [ ] Mid-stream interrupt design for v2 metacognition
- [ ] MCP integration for tool use?
- [x] RESOLVED: Bootstrap prompt → question-based, not declarative (see 17g)
- [ ] Cost tracking: actual vs estimated for consolidation cycles
- [ ] Token counting implementation for context budget enforcement
- [ ] Session resumption protocol on agent restart
- [ ] Schema migration framework (alembic or raw SQL scripts?)

### Philosophical

- [ ] At what point (if ever) does the agent's identity warrant moral
  consideration?
- [ ] Should the agent be told it's an experiment? (Transparency vs priming)
- [ ] When the agent names itself, does that constitute self-determination
  or sophisticated pattern matching?
- [ ] How to address the operator influence / mirror problem in the paper?
- [ ] Is "emergence" the right word if the architecture was designed to
  produce identity? (Designed emergence vs spontaneous emergence)

### Resolved

- [x] Vector DB → Postgres + pgvector
- [x] Document store → Postgres JSONB (same DB)
- [x] Embedding model → Google gemini-embedding-001 (API, not local)
- [x] System 2 model → Claude Sonnet 4.5 (not DeepSeek R1)
- [x] Agent sees containment → Yes, transparency over obscurity

---

## 18. Whitepaper Strategy <a name="18-whitepaper-strategy"></a>

### Format concept

The paper should present both technical architecture and philosophical
foundation in parallel. Proposed structure: dual-column layout where each
section has its technical specification alongside its philosophical
rationale, allowing readers to follow either thread independently or both
together.

### Sections mapping

| Technical section | Philosophical companion |
|---|---|
| Three-layer memory model | Why memory structure determines identity |
| Memory gate scoring | What "important" means to a developing self |
| Consolidation operations | How patterns become beliefs become identity |
| Dual-process reasoning | The difference between reacting and reflecting |
| Idle loop / DMN | Why rest is a form of cognition |
| Compulsion safety | When wanting becomes needing: internal self-regulation |
| Containment model | Building autonomy inside boundaries: the trust paradox |
| Spawning / merge | What it means to create and recombine selves |

### Key claims to establish

1. Identity can emerge from architecture without seeding
2. Weighted values are superior to binary rules for identity representation
3. Internal compulsion safety is a novel and necessary architectural feature
4. The strange loop between memory layers constitutes a candidate mechanism
   for artificial selfhood
5. The co-design process (human + AI designing the system together) is itself
   evidence for the thesis

### Evidence plan

Log everything from the agent's first conversations onward. Real transcripts
of identity emergence (or failure to emerge) are the strongest possible
evidence for or against the thesis. Plan:

- Full conversation logs (already in audit_trail.log)
- Layer 0/1 version history (already versioned with timestamps)
- Consolidation log (every merge/promote/decay with reasoning)
- Self-prompt logs from idle loop (when implemented)
- Key moments: first value formation, first goal emergence, first spontaneous
  action, first time agent questions a bootstrap boundary
- Include select transcripts as appendix material in the whitepaper

### Publication considerations

- Open source the architecture and safety mechanisms
- Don't include one-click deployment scripts
- Share the blueprint for doing it responsibly
- Speed matters: field converging fast, window for priority narrowing

---

## 19. Planned Interfaces <a name="19-interfaces"></a>

The current interface is CLI stdin/stdout — a development expedient. The
architecture is designed for multi-channel interaction. This is high priority.

### Planned channels (priority order)

1. **HTTP API** — RESTful or WebSocket endpoint for programmatic access.
   Enables web UI, mobile apps, and integration with other systems.
   The cognitive loop already separates input handling from reasoning;
   replacing stdin with an HTTP handler is mechanical.

2. **Telegram** — Lightweight, good bot API, supports rich messages.
   Likely first messaging channel because it's simple to implement
   and the operator already uses it.

3. **WhatsApp** — Broader reach but more complex API (Business API
   required, webhook infrastructure). Second priority.

4. **Discord / Slack** — Community or workspace integration. Lower
   priority unless specific use case emerges.

### Architectural note

The loop.py cognitive loop is already structured for this:
```python
# Current: stdin readline
user_input = await loop.run_in_executor(None, sys.stdin.readline)

# Future: channel adapter pattern
user_input = await channel.receive()  # HTTP, Telegram, WhatsApp, etc.
```

Each channel adapter handles I/O formatting. The cognitive loop, memory
gate, monitors, and all reasoning are channel-agnostic. The agent doesn't
know or care where the message came from — its cognitive architecture
processes content identically regardless of source.

Multi-channel simultaneously is supported by design: the conversation
history in the cognitive loop is per-session, and sessions can come from
any channel. The agent maintains identity across channels because Layer 0
and Layer 1 are global (always injected), not per-session.

---

## 20. Glossary <a name="20-glossary"></a>

| Term | Definition |
|------|-----------|
| **Layer 0** | Identity layer: values, beliefs, voice, boundaries. Highest friction to change. |
| **Layer 1** | Goals layer: active wants, preferences, projects. Medium friction. |
| **Layer 2** | Data layer: all memories, facts, experiences. Lowest friction, constantly updated. |
| **System 1** | Fast LLM (Gemini). Handles 90% of interactions. Always running. |
| **System 2** | Deep LLM (Sonnet). Called as tool for complex/risky decisions. |
| **FOK** | Feeling of Knowing. Monitor that checks "do I know this?" via vector similarity. |
| **DMN** | Default Mode Network. The idle loop that produces spontaneous thought. |
| **Entry gate** | Fast filter on incoming content. Writes to scratch buffer. ~1ms. |
| **Exit gate** | Scoring algorithm for content leaving context. Decides persist vs drop. ~5ms. |
| **Scratch buffer** | Temporary staging for entry-gated content. Safety net before full scoring. |
| **Consolidation** | Background process: merge memories, promote patterns, decay stale content. |
| **Strange loop** | Self-referential feedback: identity biases reasoning, reasoning reshapes identity. |
| **Supersedes** | Link from consolidated insight back to source memories. Preserves originals. |
| **Compulsion safety** | Internal mechanisms preventing goal weight runaway (diminishing returns, caps). |
| **Matryoshka** | Embedding training technique allowing dimension truncation without retraining. |
| **Bootstrap phase** | Initial period: blank identity, operator acts as guide. Phase 1 of 3+. |

---

*This documentation was written on 2026-02-08 based on forensic analysis of
the complete design document (notes.md), all source code on norisor.local,
database schema, configuration files, and the implementation session transcript.
It is intended as the definitive internal record from which a whitepaper can
be extracted.*
