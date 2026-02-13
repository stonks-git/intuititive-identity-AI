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

### SESSION 2026-02-11 (C+D+E) - SAFETY + CONSOLIDATION + PERIPHERALS

**STATUS:** DONE

**What was done:**
1. Tasks 18-22: escalation, System 2, reflection bank, retrieval mutation, safety ceilings
2. Tasks 23-28: two-tier consolidation engine (constant + deep)
3. Tasks 29-35: DMN idle loop, energy tracking, session restart, docs, gut feeling, bootstrap readiness, outcome tracking

---

### SESSION 2026-02-12 (F) - FRAMEWORK ADOPTION + WIRE PHASE

**STATUS:** DONE

**What was done:**
1. Adopted AI-DEV framework (taskmaster.py, state/, prompts/, KB/)
2. FW-001 DONE — framework fully adopted
3. WIRE-001 DONE — GutFeeling wired into cognitive loop
4. WIRE-002 DONE — BootstrapReadiness wired into cognitive loop
5. WIRE-003 DONE — OutcomeTracker wired into safety + consolidation

---

### SESSION 2026-02-13 (G) - PERIPHERAL ARCHITECTURE + TELEGRAM

**STATUS:** DONE

**What was done:**
1. Cleaned norisor of all old files (src/, .git, docs). Only .env, docker-compose.yml, agent-state/ remain.
2. Fixed Docker image name: `ghcr.io/stonks-git/intuitive-ai:latest` (was typo `intuititive-identity-ai`)
3. Fixed main.py: create `~/.agent/logs/` dir on startup (FileNotFoundError)
4. **PERIPHERAL ARCHITECTURE BUILT** — the big feature this session:
   - `src/stdin_peripheral.py` NEW — stdin factored out as a peripheral
   - `src/telegram_peripheral.py` NEW — raw httpx Telegram Bot API (long polling, owner-only auth)
   - `src/loop.py` MODIFIED — replaced hardcoded stdin with unified `input_queue: asyncio.Queue`
   - `src/idle.py` MODIFIED — renamed `dmn_queue` → `input_queue`
   - `src/main.py` MODIFIED — creates shared `input_queue(maxsize=50)`, wires peripherals
   - `src/context_assembly.py` MODIFIED — fixed empty query crash
5. Telegram bot: `@alecprats_ai_bot`, token + owner_id in norisor `.env`

**Commits:**
- `0f04799` Fix Docker image name and ensure logs directory exists
- `a5442e8` Add peripheral architecture: Telegram + stdin input, unified queue
- `130f26e` Fix empty query crash in context assembly + telegram offset

---

### SESSION 2026-02-14 (H) - DEPLOY + TEST + PERIPH-001 DONE

**STATUS:** DONE

**What was done:**
1. Completed session G documentation (KB_04, roadmap PERIPH-001, devlog entries)
2. Fixed stdin peripheral thread pool exhaustion bug
3. Deployed to norisor, verified Telegram end-to-end

**Commits:**
- `56bf262` Document peripheral architecture (session G): KB, roadmap, devlog, handoff
- `3cdfb95` Fix stdin peripheral thread pool exhaustion in Docker

---

### SESSION 2026-02-14 (I) - DASHBOARD v1 + v2 REWRITE

**STATUS:** v2 COMMITTED LOCALLY, NOT YET DEPLOYED

**What was done:**

**Phase 1: Dashboard v1 (built, deployed, verified)**
1. `src/dashboard.py` NEW — AgentState dataclass, SSE broadcast, REST API, inline HTML
2. `src/loop.py` modified: `agent_state=None` param, assigns objects after creation, shared conversation list, 4 SSE events (cycle_start, llm_response, escalation, gate_flush)
3. `src/main.py` modified: creates AgentState, passes to cognitive_loop, adds run_dashboard task
4. `_flush_scratch_through_exit_gate` now returns `(persisted, dropped)` tuple
5. Infrastructure: `requirements.txt` +aiohttp>=3.9, `Dockerfile` +EXPOSE 8080, `docker-compose.yml` +port 8080:8080, removed stdin_open/tty
6. Norisor docker-compose.yml: added port mapping 8080:8080 (via SSH sed)
7. **Deployed and verified**: `/api/status` returns correct JSON, 53 memories, bootstrap 5/10
8. Added System 1/2 model names + escalation stats to status API and frontend

**Phase 2: Dashboard v2 rewrite (committed locally, NOT deployed)**
User wanted terminal-style consciousness monitor, not a metrics dashboard. Reviewed by UX agent.

Changes to `src/dashboard.py` (complete rewrite of HTML/CSS/JS + new endpoints):
- **Layout**: Two-column asymmetric. Left 65% = Conscious Mind. Right 35% split: top = Attention Queue, bottom = Memory Search.
- **Header**: 28px bar with all stats compressed: gut, bootstrap, cost, escalations, queue, mem, model
- **Conscious Mind panel**: Each cycle streams as a block via SSE:
  - `INPUT` section: what won attention (full text)
  - `SYSTEM PROMPT` section: collapsible, full LLM context (identity + safety + gut + bootstrap + corrections)
  - `CONVERSATION` section: collapsible, rolling conversation window
  - `RESPONSE` section: full LLM output with `S1/S2 [model_name] conf:X` label, green border for S1, orange for S2
  - Auto-scrolls, pauses when user scrolls up manually
- **Attention Queue panel**: Full text of all competing candidates with winner marked green, losers listed
- **Memory Search panel**: search input with 300ms debounce, semantic search via `search_hybrid(mutate=False)`, click-to-expand inline detail. Shows latest 10 on empty search.
- **New API endpoint**: `GET /api/memories/search?q=...` — uses `search_hybrid(mutate=False)` for read-only semantic search
- **Removed**: v1 endpoints `/api/memories` (paginated), `/api/gut`, `/api/conversation`, `/api/energy` — replaced by SSE events and header polling

Changes to `src/loop.py`:
- **New SSE event `context_assembled`**: emitted after system prompt built, carries full `active_system_prompt` + conversation array + identity_tokens + context_shift
- **Expanded `cycle_start` event**: now includes `winner` object (source, full content, salience) + `losers` array (up to 5, each with source, full content, salience) + queue_size
- **Removed truncation**: `llm_response` event now carries full `reply` (was `reply[:200]`), plus `model` field
- **Added `model` field** to both S1 and S2 llm_response events
- **Consciousness log**: persistent NDJSON at `~/.agent/logs/consciousness.ndjson`, appended each cycle with: ts, source, salience, input, system_prompt_len, conversation_len, reply, escalated, confidence, context_shift, queue_size_after

**Commits (3 commits, all on main, only first 2 pushed):**
- `d685e49` Add agent consciousness dashboard (aiohttp, SSE, memory browser) — **PUSHED, DEPLOYED**
- `1c5c005` Show System 1/2 models and escalation stats in dashboard — **PUSHED, DEPLOYED**
- `427d213` Rewrite dashboard v2: terminal-style consciousness monitor — **LOCAL ONLY, NOT PUSHED**

**CRITICAL: Current norisor state**
- Norisor is running commit `1c5c005` (dashboard v1 with S1/S2 stats)
- Commit `427d213` (v2 rewrite) is LOCAL ONLY, not pushed, not deployed
- User was talking to the agent via Telegram when session ended
- User explicitly said "don't deploy until I tell you"
- To deploy v2: `git push origin main` then `ssh norisor "cd ~/agent-runtime && docker pull ghcr.io/stonks-git/intuitive-ai:latest && docker compose down && docker compose up -d"`

---

## What is this project?

Cognitive architecture for emergent AI identity. Three-layer memory unified into one Postgres store with continuous depth_weight (Beta distribution). Dual-process reasoning (System 1: Gemini Flash Lite, System 2: Claude Sonnet 4.5). Metacognitive monitoring. Consolidation sleep cycle. DMN idle loop. Two-centroid gut feeling model. Identity emerges from experience, not configuration. All 35 implementation plan tasks complete. Peripheral architecture built. Currently testing Telegram integration.

---

## Tasks DOING now

| Task ID | Status |
|---------|--------|
| FW-001 | done |
| WIRE-001/002/003 | done |
| PERIPH-001 (Telegram) | DONE |
| DASH-001 (Dashboard v1) | DONE — deployed on norisor, verified |
| DASH-002 (Dashboard v2) | COMMITTED LOCALLY — awaiting user signal to deploy |
| TEST-001 | NEXT — full end-to-end runtime test |

---

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
  loop.py                  Attention-agnostic cognitive loop (unified input_queue, reply_fn routing)
  main.py                  Entry point, consolidation engine, peripheral wiring, session tracking
  relevance.py             5-component hybrid relevance + Dirichlet blend
  attention.py             Salience-based attention allocation (AttentionCandidate, AttentionAllocator)
  context_assembly.py      Dynamic context injection + FIFO pruning (fixed empty query bug)
  consolidation.py         Two-tier: ConstantConsolidation + DeepConsolidation
  idle.py                  DMN with stochastic sampling, pushes to shared input_queue
  gut.py                   Two-centroid gut feeling model
  bootstrap.py             10 readiness milestones
  dashboard.py             NEW — localhost web dashboard (aiohttp, SSE, memory browser, port 8080)
  stdin_peripheral.py      stdin I/O as peripheral (single reader thread, disabled in Docker)
  telegram_peripheral.py   Telegram Bot API via raw httpx (long polling, owner auth, reply_fn)
```

### Peripheral Architecture (NEW)

```
                    ┌──────────────┐
                    │  input_queue  │  asyncio.Queue(maxsize=50)
                    │  (shared)     │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   StdinPeripheral   TelegramPeripheral    IdleLoop (DMN)
   (push external_user)  (push external_user)  (push internal_dmn)
        │                  │                  │
        │           reply_fn=sendMessage      │
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    cognitive_loop drains queue
                    → attention allocation
                    → context assembly
                    → LLM call
                    → reply via winner.metadata["reply_fn"]
```

Key design:
- All peripherals push `AttentionCandidate` objects into one queue
- Each candidate carries `metadata["reply_fn"]` — async callback to route response
- Telegram: reply_fn calls `sendMessage` API
- Stdin: reply_fn calls `print()`
- DMN: no reply_fn (internal thoughts log-only)
- Loop doesn't know or care where input came from

### Norisor setup (CLEAN)

```
~/agent-runtime/
  .env                 API keys + TELEGRAM_BOT_TOKEN + TELEGRAM_OWNER_ID
  docker-compose.yml   agent_001 + agent_postgres (correct image name)
  agent-state/         config/, identity/, goals/, logs/, manifest.json
```

- No src files on norisor (Docker-only deployment)
- Image: `ghcr.io/stonks-git/intuitive-ai:latest`
- Postgres data volume preserved (6 memories)
- CI/CD: push to main → GitHub Actions → build image → pull on norisor

### Connection info

- **Server:** norisor (Debian, Docker)
- **Tailscale IP:** 100.66.170.31 (hostname `norisor`)
- **SSH:** `ssh norisor` (configured in ~/.ssh/config)
- **DB:** `postgresql://agent:agent_secret@localhost:5433/agent_memory`
- **Deploy:** Push to main -> GitHub Actions -> Docker -> norisor
- **Telegram bot:** @alecprats_ai_bot (token in norisor .env)
- **Telegram owner ID:** 6639032827

---

## Docker/Prod Status

- Docker Compose on norisor: agent container (2 CPU/2GB) + postgres container (1 CPU/1GB)
- CI/CD: GitHub Actions builds on push to main (src/, Dockerfile, requirements.txt, docker-compose.yml)
- Image: ghcr.io/stonks-git/intuitive-ai:latest
- Norisor cleaned: only docker-compose.yml + .env + agent-state/ (no old src/docs)

---

## Blockers or open questions

| Blocker/Question | Status |
|------------------|--------|
| ~~GutFeeling, Bootstrap, OutcomeTracker not wired~~ | DONE |
| ~~Docker image name typo~~ | FIXED (0f04799) |
| ~~Empty query crash in context_assembly~~ | FIXED (130f26e) |
| ~~Redeploy and test Telegram~~ | DONE — verified end-to-end (session H) |
| ~~Stdin thread pool exhaustion in Docker~~ | FIXED (3cdfb95) |
| Multimodal perception layer (images/audio in attention loop) | FUTURE — discussed architecture, not started |

---

## Useful commands (copy-paste ready)

```bash
# Validate framework state
python3 taskmaster.py validate

# Ready tasks
python3 taskmaster.py ready

# Local Python (use venv)
.venv/bin/python3 -m py_compile src/foo.py

# Deploy to norisor (Docker only)
git push origin main  # triggers CI/CD
ssh norisor "cd ~/agent-runtime && docker pull ghcr.io/stonks-git/intuitive-ai:latest && docker compose down && docker compose up -d"

# Check agent logs
ssh norisor "docker logs --tail 30 agent_001"

# Check if Telegram is connected
ssh norisor "docker logs agent_001 2>&1 | grep -i telegram"
```

---

## Key architectural decisions (resolved, don't revisit)

- Unified memory (not 3 discrete layers) -- depth_weight Beta distribution
- Identity is a rendered view of high-weight memories, not a stored artifact
- Stochastic everything -- Beta weights, Dirichlet blends, injection rolls
- ACT-R equations with human-calibrated starting points, evolved by consolidation
- Attention-agnostic loop -- all input sources feed same pipeline via unified input_queue
- Build all safety from day one, enable incrementally
- Dual-process: System 1 (Gemini Flash Lite) drives, System 2 (Claude Sonnet 4.5) escalation
- Reflection bank: System 2 corrections stored as type="correction" memories
- Peripheral architecture: any I/O source pushes AttentionCandidate into shared queue, reply_fn routes responses back
- Telegram: raw httpx (no framework dependency), long polling (no public endpoint), owner-only auth
- Multimodal future: embedding stays text-based (subconscious); LLM gets full multimodal (conscious). Content type mismatch = gut delta signal.

---

## Checklist before handoff

- [x] Updated task statuses in handoff
- [x] Completed current session section above
- [x] devlog updated
- [x] **Kept only last 4 sessions** (C+D+E, F, G, H, I)
- [x] KB updated if code was changed
- [x] KB_05 needs v2 update (noted in devlog)

---

## Git Status

- **Branch:** main
- **Last commit (local):** 427d213 Rewrite dashboard v2: terminal-style consciousness monitor
- **Last commit (remote/norisor):** 1c5c005 Show System 1/2 models and escalation stats in dashboard
- **Uncommitted:** devlog, handoff, KB updates (this documentation commit)
- **Agent status:** Running on norisor with dashboard v1. Telegram connected. Dashboard v2 is local-only.

---

## Memory Marker

```
MEMORY_MARKER: 2026-02-14 | Dashboard v1 deployed, v2 local-only | Agent live on norisor, Telegram working | Next: deploy v2 when user says, then TEST-001
```
