# KB 02: Memory System

## Unified Store

All memories in one Postgres table. No discrete layers — depth_weight Beta distribution determines cognitive role:

| Weight Center | Role | Example Alpha/Beta |
|---------------|------|--------------------|
| 0.8 - 0.95 | Identity-equivalent | Beta(50, 2) |
| 0.6 - 0.8 | Goal-equivalent | Beta(10, 2) |
| 0.2 - 0.6 | Active data | Beta(1-10, 1-10) |
| < 0.2 | Dormant (never deleted) | Beta(1, 4) |

Only `immutable=true` memories (4 safety boundaries) are categorical.

## Memory Types

Stored with semantic prefixes baked into embeddings:
- episodic, semantic, procedural, preference, reflection, correction, narrative, tension

## Retrieval Pipelines

**Pipeline 1 — Gate (persist/drop):** ACT-R activation -> 3x3 decision matrix

**Pipeline 2 — Context injection:** pgvector pre-filter -> hybrid dense+sparse RRF -> FlashRank reranking -> 5-component hybrid relevance (Dirichlet-blended)

## Retrieval-Induced Mutation

Every retrieval modifies memory weights:
- Top-K retrieved: alpha += 0.1 (or 0.2 if dormant recovery)
- Near-misses (rank 6+): beta += 0.05 (mild suppression)
- All mutations go through SafetyMonitor ceilings

## Hebbian Co-Access

`memory_co_access` table tracks pairs retrieved together. Spreading activation: 1-hop default, 2-hop during DMN.
