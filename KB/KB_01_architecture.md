# KB 01: System Architecture Overview

## Three Concurrent Loops

1. **Cognitive Loop** (`src/loop.py`) — processes one input at a time through attentional thread. All sources (user, DMN, consolidation, gut) compete for attention.
2. **Consolidation** (`src/consolidation.py`) — two-tier: constant background (decay, contradiction scan, pattern detection) + periodic deep cycles (merge, insight, promotion, reconsolidation).
3. **Idle Loop / DMN** (`src/idle.py`) — spontaneous thought during downtime, filtered through goals/values, queued for cognitive loop.

## Module Map

| Module | Role |
|--------|------|
| `loop.py` | Main cognitive cycle, escalation, commands |
| `memory.py` | Postgres + pgvector, embed, search, mutation |
| `safety.py` | 6 ceilings in 3 phases + OutcomeTracker |
| `consolidation.py` | Two-tier consolidation engine |
| `idle.py` | DMN idle loop |
| `attention.py` | Salience-based attention allocation |
| `relevance.py` | 5-component hybrid relevance + Dirichlet |
| `context_assembly.py` | Dynamic context injection + FIFO |
| `gate.py` | Entry gate (stochastic) + Exit gate (ACT-R 3x3) |
| `gut.py` | Two-centroid gut feeling (Free Energy Principle) |
| `bootstrap.py` | 10 readiness milestones |
| `stochastic.py` | StochasticWeight (Beta distribution) |
| `activation.py` | ACT-R activation equation |
| `metacognition.py` | Composite confidence scoring |
| `layers.py` | L0/L1 disk store + embedding cache |
| `config.py` | YAML config loader |
| `llm.py` | LLM client + EnergyTracker |
| `tokens.py` | Token counting |
| `main.py` | Entry point, initialization |

## Tech Stack

- Python 3.12 async
- PostgreSQL 17 + pgvector (halfvec 768-dim)
- Gemini Flash Lite (System 1), Claude Sonnet 4.5 (System 2)
- Gemini text-embedding-004 (embeddings)
- FlashRank (cross-encoder reranking)
- Docker Compose on norisor

## Data Flow (Simplified)

```
Input sources -> AttentionAllocator (salience competition)
  -> Embed winner
  -> Context Assembly (safety + identity + situational + cognitive state)
  -> FIFO prune
  -> System 1 LLM call
  -> Escalation check -> System 2 if needed
  -> Entry Gate (buffer to scratch)
  -> Exit Gate (persist/drop from scratch)
  -> Consolidation (background)
  -> Memory weights evolve
```
