# Supervisor Handoff

> **READ THIS FIRST.** You are the supervisor (queen agent) for this project.
>
> **Reading order (MANDATORY):**
> 1. This file (handoff.md) - bootstrap loader
> 2. `prompts/supervisor.md` - your supervisor contract
> 3. `state/charter.json` - project constraints (MANDATORY)
> 4. `python3 taskmaster.py ready` - available tasks
> 5. Sections below - previous session context

---

## Previous Sessions

### SESSION 2026-02-10 (A+B) - FOUNDATION + COGNITIVE PIPELINE

**STATUS:** DONE

**What was done:**
1. Fixed 15 syntax errors across gate.py, loop.py, main.py
2. Implemented tasks 1-12: embeddings, schema migration, stochastic weights, layers, hybrid search, reranking, ACT-R, gates, tokens, metacognition
3. Implemented tasks 13-17: hybrid relevance, attention allocation, context assembly, FIFO pruning, cognitive loop rewrite

**Verifications PASSED:**
- All src/*.py files py_compile clean
- Structural checks pass on all modules

| File | What was done |
|------|---------------|
| `src/memory.py` | task_type, prefixes, halfvec, embed_batch, search_hybrid, search_reranked |
| `src/stochastic.py` | NEW - Beta distribution StochasticWeight class |
| `src/activation.py` | NEW - ACT-R 4-component activation equation |
| `src/metacognition.py` | NEW - composite confidence scoring |
| `src/tokens.py` | NEW - token counting utilities |
| `src/gate.py` | REWRITTEN - 3x3 matrix exit gate + entry gate |
| `src/relevance.py` | NEW - 5-component hybrid relevance + Dirichlet |
| `src/attention.py` | NEW - salience-based attention allocation |
| `src/context_assembly.py` | NEW - dynamic context injection + FIFO |
| `src/loop.py` | REWRITTEN - attention-agnostic cognitive loop |

---

### SESSION 2026-02-11 (C+D+E) - SAFETY + CONSOLIDATION + PERIPHERALS

**STATUS:** DONE

**What was done:**
1. Tasks 18-22: escalation, System 2, reflection bank, retrieval mutation, safety ceilings
2. Tasks 23-28: two-tier consolidation engine (constant + deep)
3. Tasks 29-35: DMN idle loop, energy tracking, session restart, docs, gut feeling, bootstrap readiness, outcome tracking

**Verifications PASSED:**
- All src/*.py files py_compile clean
- Safety: hard ceiling blocks at >0.95, diminishing returns correct
- Consolidation: clustering finds correct groups
- Gut: centroid math correct, delta nonzero

| File | What was done |
|------|---------------|
| `src/safety.py` | SafetyMonitor + 6 ceilings (3 phases) + OutcomeTracker |
| `src/consolidation.py` | REWRITTEN - ConstantConsolidation + DeepConsolidation |
| `src/idle.py` | REWRITTEN - full DMN with 4 bias types, 3 output channels |
| `src/llm.py` | EnergyTracker class added |
| `src/gut.py` | NEW - two-centroid gut feeling model |
| `src/bootstrap.py` | NEW - 10 readiness milestones |
| `src/main.py` | ConsolidationEngine, DMN queue, session restart tracking |

---

### SESSION 2026-02-12 (F) - FRAMEWORK ADOPTION + WIRE PHASE

**STATUS:** DOING

**What was done:**
1. Adopted AI-DEV framework (taskmaster.py, state/, prompts/, KB/)
2. Filled charter.json with project context
3. Created roadmap.json with 4 done grouped tasks + 6 next-phase tasks
4. Migrated SESSION_HANDOFF.md to this file
5. FW-001 DONE — framework fully adopted
6. Added unattended execution directive to CLAUDE.md (temporary, removed by CLEANUP-001)
7. Added CLEANUP-001 task to roadmap (depends on TEST-002)

---

## What is this project?

Cognitive architecture for emergent AI identity. Three-layer memory unified into one Postgres store with continuous depth_weight (Beta distribution). Dual-process reasoning (System 1: Gemini Flash Lite, System 2: Claude Sonnet 4.5). Metacognitive monitoring. Consolidation sleep cycle. DMN idle loop. Two-centroid gut feeling model. Identity emerges from experience, not configuration. All 35 implementation plan tasks complete. Currently in integration wiring phase.

---

## Tasks DOING now

| Task ID | Status |
|---------|--------|
| FW-001 | done |
| WIRE-001 | next - Wire GutFeeling into cognitive loop |
| WIRE-002 | next - Wire BootstrapReadiness into cognitive loop |
| WIRE-003 | next - Wire OutcomeTracker into safety + consolidation |

## What exists

### Source files (src/)

```
src/
  __init__.py              empty
  config.py                working, clean
  llm.py                   EnergyTracker class (cost tracking)
  memory.py                Full memory store (embed, search_hybrid, search_reranked, retrieval mutation, safety integration)
  safety.py                SafetyMonitor + 6 ceiling classes + OutcomeTracker
  layers.py                L0/L1 disk store + embedding cache
  stochastic.py            StochasticWeight (Beta distribution)
  activation.py            ACT-R 4-component activation equation
  metacognition.py         Composite confidence scoring
  tokens.py                Token counting utilities
  gate.py                  3x3 exit gate + stochastic entry gate
  loop.py                  Attention-agnostic cognitive loop + dual-process escalation
  main.py                  Entry point, consolidation engine, DMN queue, session tracking
  relevance.py             5-component hybrid relevance + Dirichlet blend
  attention.py             Salience-based attention allocation
  context_assembly.py      Dynamic context injection + FIFO pruning
  consolidation.py         Two-tier: ConstantConsolidation + DeepConsolidation
  idle.py                  DMN with stochastic sampling, 3 output channels
  gut.py                   Two-centroid gut feeling model
  bootstrap.py             10 readiness milestones
```

### Remote database (norisor, port 5433)

Fully migrated. memories table has halfvec(768), Beta distribution columns, full-text search, access_timestamps, memory_co_access table. 6 test memories.

### Connection info

- **Server:** norisor (Debian, Docker)
- **Tailscale IP:** 100.66.170.31 (hostname `norisor`)
- **SSH:** `ssh norisor` (configured in ~/.ssh/config)
- **DB:** `postgresql://agent:agent_secret@localhost:5433/agent_memory`
- **Deploy:** Push to main -> GitHub Actions -> Docker -> norisor

---

## Docker/Prod Status

- Docker Compose on norisor: agent container (2 CPU/2GB) + postgres container (1 CPU/1GB)
- CI/CD: GitHub Actions builds on push to main (src/, Dockerfile, requirements.txt, docker-compose.yml)
- Image: ghcr.io/stonks-git/intuititive-identity-ai:latest

---

## Blockers or open questions

| Blocker/Question | Status |
|------------------|--------|
| GutFeeling, Bootstrap, OutcomeTracker not wired into loop | Next tasks (WIRE-001/002/003) |
| No runtime test yet (only py_compile) | Blocked by wiring |

---

## Useful commands (copy-paste ready)

```bash
# Validate framework state
python3 taskmaster.py validate

# Ready tasks
python3 taskmaster.py ready

# Local Python (use venv)
.venv/bin/python3 -m py_compile src/foo.py
.venv/bin/python3 -c "from src.foo import *"

# Deploy to norisor
ssh norisor
cd ~/agent-runtime
docker compose up -d postgres
export GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d= -f2)
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)
export DATABASE_URL="postgresql://agent:agent_secret@localhost:5433/agent_memory"
python3 -m src.main
```

---

## Key architectural decisions (resolved, don't revisit)

- Unified memory (not 3 discrete layers) -- depth_weight Beta distribution
- Identity is a rendered view of high-weight memories, not a stored artifact
- Stochastic everything -- Beta weights, Dirichlet blends, injection rolls
- ACT-R equations with human-calibrated starting points, evolved by consolidation
- Attention-agnostic loop -- all input sources feed same pipeline
- Build all safety from day one, enable incrementally
- Dual-process: System 1 (Gemini Flash Lite) drives, System 2 (Claude Sonnet 4.5) escalation
- Reflection bank: System 2 corrections stored as type="correction" memories

---

## Checklist before handoff

- [ ] Updated task statuses in handoff
- [ ] Completed current session section above
- [ ] devlog updated (+1 entry per significant change)
- [ ] **Kept only last 3 sessions** (older ones archived in git)
- [ ] KB updated if code was changed

---

## Git Status

- **Branch:** main
- **Last commit:** 506ab7e Add peripheral architecture design doc
- **Modified:** Framework adoption in progress

---

## Memory Marker

```
MEMORY_MARKER: 2026-02-12 | FW-001 (framework adoption) | WIRE-001 (gut wiring)
```
