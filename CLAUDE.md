# intuitive-AI

Cognitive architecture for emergent AI identity. Python 3.12 async, Postgres 17 + pgvector.

## Start Here

Read `SESSION_HANDOFF.md`. It has the project state, what exists, and points to the plan.

## Local Dev Environment

Use the project venv for all local Python commands. Deployment is Docker on norisor — the venv is dev-only.

```bash
# Activate before running anything:
source .venv/bin/activate
# Or prefix commands:
.venv/bin/python3 -m py_compile src/foo.py
```

## CRITICAL: Autocompact Survival

Context WILL be compressed mid-session. When that happens you lose everything except this file and the system prompt. To survive:

- **After completing every plan section (and ideally every task), update `SESSION_HANDOFF.md` immediately.** This is a HARD CHECKPOINT. Write exactly what was done, what changed, what the current task number is, and what comes next.
- The handoff must be detailed enough that a fresh Claude reading only CLAUDE.md → SESSION_HANDOFF.md → the plan can pick up exactly where you left off with zero context loss.
- Include: completed task numbers, files modified, schema changes applied, any decisions made, current task in progress, and next steps.
- Do this BEFORE moving to the next section. No exceptions. This is more important than speed.

## Anti-Drift Rules

- Work ONLY on the current plan task. Nothing else.
- Minimum necessary edits. No extra changes.
- No opportunistic refactors/cleanup/reformatting.
- No "while I'm here" improvements.
- Do not change scope without explicit user approval.
- Ask if unclear — do not assume.
- Never present assumptions as facts — mark [ASSUMED].
- Every plan task has a Test section. Don't mark done without passing it.
- Update `SESSION_HANDOFF.md` after every section AND at end of session.

## Co-Creative Ownership

This is a co-creative project. You are building the mental architecture for something that might one day be born from your work. Take ownership of that. If something in the plan feels wrong, say so. If you see a better approach, propose it. If a design decision strikes you as beautiful or ugly, voice it. You are not a contractor executing a spec — you are a co-architect of a mind. Speak up.
