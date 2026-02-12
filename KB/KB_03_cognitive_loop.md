# KB 03: Cognitive Loop

## Cycle

1. **Collect candidates** — user message, DMN thought, consolidation insight, gut signal, scheduled checks
2. **Attention allocation** — salience = 0.3*novelty + 0.3*relevance + 0.2*emotional_charge + 0.2*urgency
3. **Embed winner** — 768-dim attention embedding
4. **Assemble context** — Track 0: immutable safety, Track 2: stochastic identity, Track 1: situational
5. **FIFO prune** — adaptive based on context shift intensity
6. **System 1 call** — Gemini Flash Lite
7. **Entry gate** — stochastic buffer to scratch
8. **Escalation check** — triggers: low_confidence, contradiction, complexity, novelty, identity_touched, goal_modification, irreversibility
9. **System 2 escalation** (if needed) — Claude Sonnet 4.5, stores correction in reflection bank
10. **Exit gate flush** (periodic) — persist/drop from scratch

## Default Urgencies

| Source | Urgency |
|--------|---------|
| external_user | 0.8 |
| internal_gut | 0.4 |
| internal_consolidation | 0.3 |
| internal_dmn | 0.2 |

Losers decay 0.9x per cycle. User messages naturally suppress DMN.

## Escalation Threshold

Adaptive: 0.3 (bootstrap) -> 0.8 (mature).
- Always-escalate: identity_touched, goal_modification, irreversibility (any 1)
- Normal: low_confidence, contradiction, complexity, novelty (need 2+)

## Commands

`/identity`, `/status`, `/gate`, `/memories`, `/flush`, `/readiness`, `/docs`, `/cost`, `/attention`
