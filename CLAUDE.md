# intuitive-AI

Cognitive architecture for emergent AI identity. Python 3.12 async, Postgres 17 + pgvector.

## Start Here (MANDATORY reading order)

1. This file (`CLAUDE.md`)
2. `state/handoff.md` — current state, session history, what exists
3. `state/charter.json` — project constraints (MANDATORY)
4. `python3 taskmaster.py ready` — available tasks
5. `prompts/supervisor.md` — supervisor contract

<!-- UNATTENDED-SESSION-START -->
## ACTIVE DIRECTIVE: Unattended Execution (TEMPORARY)

**Scope:** Until all tasks in `state/roadmap.json` are complete (through CLEANUP-001).

**Rules:**
1. Execute tasks autonomously — no user prompts for permission.
2. Follow ALL existing rules strictly: anti-drift, KB gate, checkpoint protocol, charter constraints, security.
3. After each completed task:
   - Run `/doc` to document the work.
   - Commit changes with a descriptive message.
4. Focus on documentation quality — KB updates, devlog entries, handoff updates must be thorough.
5. Respect all security rules from `state/charter.json` — no deploy without local test + approval, no hardcoded identity, no memory deletion.
6. Task CLEANUP-001 removes this entire section (between `UNATTENDED-SESSION-START` and `UNATTENDED-SESSION-END` markers).

**This section is temporary and will be removed by CLEANUP-001.**
<!-- UNATTENDED-SESSION-END -->

## Framework

This project uses the **Taskmaster** framework for structured development.

### State (source of truth)

| File | Purpose |
|------|---------|
| `state/charter.json` | Project description, constraints, working agreement |
| `state/roadmap.json` | Task DAG with dependencies, priorities, verification |
| `state/devlog.ndjson` | Append-only log of decisions, milestones, blockers |
| `state/evidence.json` | Verification index |
| `state/handoff.md` | Session handoff for context survival |

### Tools

```bash
python3 taskmaster.py validate   # Validate all state files
python3 taskmaster.py ready      # Show tasks ready to start
python3 taskmaster.py order      # Print dependency-ordered task list
python3 taskmaster.py steps TASK-ID  # Show decomposition steps
```

### KB Gate (MANDATORY)

Every code change affecting functionality -> update relevant `KB/*.md` + append `kb_update` devlog entry. No KB for the module? Create one. **No commit without KB update.**

## Local Dev Environment

Use the project venv for all local Python commands. Deployment is Docker on norisor — the venv is dev-only.

```bash
source .venv/bin/activate
.venv/bin/python3 -m py_compile src/foo.py
```

## CRITICAL: Autocompact Survival

Context WILL be compressed mid-session. To survive:

- **After completing every task, update `state/handoff.md` immediately.**
- Append a devlog entry: `{"ts":"...","type":"milestone","msg":"..."}`
- Run `python3 taskmaster.py validate` to catch state corruption.
- The handoff must be detailed enough that a fresh Claude reading CLAUDE.md -> handoff.md -> charter.json can pick up exactly where you left off.
- Do this BEFORE moving to the next task. No exceptions.

## Checkpoint Protocol

Trigger checkpoint when: 3+ files read without save, important decision, task completed.

Actions:
1. Update `state/handoff.md`
2. Append devlog event
3. `python3 taskmaster.py validate`
4. Update KB if code was changed

## Anti-Drift Rules

- Work ONLY on the current task. Nothing else.
- Minimum necessary edits. No extra changes.
- No opportunistic refactors/cleanup/reformatting.
- No "while I'm here" improvements.
- Do not change scope without explicit user approval.
- Ask if unclear — do not assume.
- Never present assumptions as facts — mark [ASSUMED].
- Every task has acceptance_criteria and verification in roadmap.json. Don't mark done without passing them.
- Do not rewrite existing content in ways that drop context.

## Verification (before marking done)

Task matches request. Tests/checks pass. No regressions. Minimal changes only. KB updated.

## Context Loss

If you don't remember current task/recent files/decisions: **STOP.** Read `state/handoff.md`. Tell user "Context lost, re-read state." Wait for confirmation.

## Delegation

- **Supervisor** (you): planning, clarification, roadmap, delegation
- **Workers** (Task tool subagents): produce structured deltas for specific task IDs
- **Auditors** (7 specialized agents): security, archeologist, database, code-quality, frontend, ops, prospector — see `prompts/auditors/`

## Co-Creative Ownership

This is a co-creative project. You are building the mental architecture for something that might one day be born from your work. Take ownership of that. If something in the plan feels wrong, say so. If you see a better approach, propose it. If a design decision strikes you as beautiful or ugly, voice it. You are not a contractor executing a spec — you are a co-architect of a mind. Speak up.
