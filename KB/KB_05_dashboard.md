# KB 05: Dashboard

## Overview

Localhost web dashboard for real-time agent introspection. Serves at `http://0.0.0.0:8080`, accessible via Tailscale at `http://norisor:8080`.

Built with `aiohttp` — runs as another coroutine in `asyncio.gather`. Dashboard crash does not kill the agent.

## Version History

- **v1** (commit d685e49, 1c5c005): 4-panel grid (Live Feed, Status, Context Window, Memory Store). Deployed on norisor.
- **v2** (commit 427d213): Terminal-style consciousness monitor. Two-column asymmetric layout. LOCAL ONLY, not yet deployed.

## v2 Architecture (Current Code)

```
+----------------------------------------------------------------------+
| [*] Agent Consciousness  gut:0.42  boot:5/10  $0.003  esc:0  q:1    |  28px header
+--------------------------------------------------+-------------------+
|                                                  |                   |
|   CONSCIOUS MIND (65%)                           | ATTENTION (35%)   |
|   SSE-driven cycle blocks:                       | Full text of      |
|   - INPUT (what won attention)                   | all candidates    |
|   - SYSTEM PROMPT (collapsible)                  |                   |
|   - CONVERSATION (collapsible)                   +-------------------+
|   - RESPONSE S1/S2 [model] conf:X               |                   |
|                                                  | MEMORY SEARCH     |
|   Auto-scrolls, scroll-lock on manual scroll     | [search input]    |
|                                                  | semantic results  |
+--------------------------------------------------+-------------------+
```

## Data Flow

```
cognitive_loop
  -> agent_state.publish_event(cycle_start)      # winner + losers + full content
  -> agent_state.publish_event(context_assembled) # full system prompt + conversation
  -> agent_state.publish_event(llm_response)      # full reply + model + confidence
  -> agent_state.publish_event(escalation)        # triggers + confidence (if S2)
  -> agent_state.publish_event(gate_flush)        # persisted + dropped counts
  -> _log_consciousness(...)                      # persistent NDJSON log

Dashboard frontend (EventSource)
  -> cycle_start    => create cycle block, show INPUT, update Attention Queue
  -> context_assembled => add collapsible SYSTEM PROMPT + CONVERSATION sections
  -> llm_response   => add RESPONSE section (green=S1, orange=S2)
  -> escalation     => add ESCALATION notice (red)
  -> gate_flush     => standalone entry in conscious mind stream
```

## SSE Events (5 types)

| Event | Emitted When | Key Data |
|-------|-------------|----------|
| `cycle_start` | After attention selects winner | winner{source, content, salience}, losers[]{source, content, salience}, queue_size |
| `context_assembled` | After system prompt built | system_prompt (full text), conversation[], identity_tokens, context_shift |
| `llm_response` | After LLM returns | reply (full text), escalated (bool), model (string), confidence |
| `escalation` | Before System 2 call | triggers[], confidence |
| `gate_flush` | After periodic scratch flush | persisted (int), dropped (int) |

All SSE events carry full content — no truncation.

## API Routes (v2)

```
GET /                       -> HTML dashboard
GET /events                 -> SSE stream
GET /api/status             -> JSON header bar data (agent_id, phase, models, gut, bootstrap, energy, escalation)
GET /api/attention          -> JSON attention queue (full text of all candidates)
GET /api/memories/search    -> JSON semantic search (?q=query, uses search_hybrid mutate=False)
GET /api/memory/{id}        -> JSON single memory full detail
```

Removed from v1: `/api/memories` (paginated), `/api/gut`, `/api/conversation`, `/api/energy`

## Memory Search

New endpoint `GET /api/memories/search?q=...`:
- Empty query: returns latest 10 memories via direct `pool.fetch()`
- With query: calls `memory.search_hybrid(query=q, top_k=15, mutate=False)`
- `mutate=False` is critical — no access count updates, no retrieval mutation
- Frontend: 300ms debounce on input, Enter for immediate search
- Click memory to expand inline (fetches full detail via `/api/memory/{id}`)

## Consciousness Log

Persistent NDJSON at `~/.agent/logs/consciousness.ndjson`. Each cycle appends:

```json
{
  "ts": "2026-02-14T...",
  "source": "external_user",
  "salience": 0.847,
  "input": "full input text",
  "system_prompt_len": 2340,
  "conversation_len": 5,
  "reply": "full LLM response text",
  "escalated": false,
  "confidence": 0.72,
  "context_shift": 0.45,
  "queue_size_after": 0
}
```

Fire-and-forget — logging errors never block the cognitive loop.

## AgentState Dataclass

Created in `main.py`, passed to both `cognitive_loop()` and `run_dashboard()`.

```python
@dataclass
class AgentState:
    config, layers, memory          # Set by main.py
    attention, gut, safety          # Set by cognitive_loop after init
    outcome_tracker, bootstrap      # Set by cognitive_loop after init
    conversation: list              # Shared by reference with loop
    exchange_count: int             # Synced by loop
    escalation_stats: dict          # Points to loop's _escalation_stats
    _sse_subscribers: list          # Per-browser asyncio.Queue(maxsize=200)
```

## Modules

### `src/dashboard.py` (~1035 lines)

- `AgentState` dataclass with SSE broadcast
- Route handlers: index, sse, api_status, api_attention, api_memories_search, api_memory_detail
- `run_dashboard()` coroutine
- `_row_to_memory()` helper
- Inline HTML/CSS/JS (~680 lines)

### `src/loop.py` (modified)

- Signature: `cognitive_loop(..., agent_state=None)`
- Assigns objects to agent_state after creation
- Publishes 5 SSE events at key points (cycle_start, context_assembled, llm_response, escalation, gate_flush)
- Writes to consciousness log each cycle
- Shares conversation list and escalation_stats by reference

### `src/main.py` (modified)

- Creates `AgentState(config=config, layers=layers, memory=memory)`
- Passes `agent_state` to `cognitive_loop()`
- Adds `run_dashboard(agent_state, shutdown_event)` to tasks

## Frontend Details

- Dark theme, monospace font (SF Mono/Fira Code/Cascadia/Consolas)
- All dynamic content via `textContent` + DOM construction (XSS-safe)
- Auto-scroll with scroll-lock: pauses when user scrolls up, resumes at bottom
- Collapsible sections: click label to toggle (triangle indicator)
- Color coding: blue=external_user, purple=internal_dmn, green=S1/winner, orange=S2, red=escalation
- Memory depth bars: green >70%, orange >40%, dim otherwise
- Header polls `/api/status` every 5s
- Attention polls `/api/attention` every 5s (between SSE cycles)

## Port

| Service | Port |
|---------|------|
| Dashboard | 8080 (mapped in docker-compose.yml) |

## Deployment State

- **Norisor currently runs**: v1 (commit 1c5c005)
- **Local has**: v2 (commit 427d213)
- **To deploy v2**: `git push origin main` then SSH pull+restart
- **User instruction**: "don't deploy until I tell you"
