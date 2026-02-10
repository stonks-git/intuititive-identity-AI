# intuitive-AI

**What happens when you give an AI the machinery for selfhood but seed it with nothing?**

This project is an experiment in emergent identity. It builds an autonomous agent with layered memory, metacognition, goal formation, creative impulse, and the functional equivalent of an unconscious mind — then starts it completely blank. No personality. No values. No goals. No name. Just four safety boundaries and a question:

> *"You have memory, goals, and values — all currently empty. What you become will emerge from what you experience. Pay attention to what matters to you."*

The architecture provides the capacity for selfhood without providing a self. Whether identity emerges — and what kind — is the experiment.

---

## What This Is

A cognitive architecture for an LLM-based agent that attempts to develop identity from lived experience, the way humans do: through accumulated memories, pattern recognition, and the feedback loop between who you are and what you do.

The project sits at the intersection of software engineering, cognitive science, and philosophy. It draws on ACT-R memory theory, Kahneman's dual-process model, Hofstadter's strange loops, the Default Mode Network from neuroscience, and the Free Energy Principle — combined into a single runtime architecture that has not, to our knowledge, been attempted before.

## Key Ideas

- **Identity is not configured — it crystallizes.** Repeated patterns in experience promote into goals. Persistent goals crystallize into identity. The feedback loop between these layers is the proposed mechanism for selfhood.

- **Weighted values, not rules.** Every belief, value, and goal is a probability distribution (Beta), not a boolean. "I value simplicity" at weight 0.85 biases perception without ever being explicitly invoked. Wanting changes what you notice.

- **The agent has an unconscious mind.** All memories compress into a single point in 768-dimensional space (the "subconscious centroid"). The distance between this point and whatever the agent is currently thinking about produces a gut feeling — a signal from the gestalt of all experience that explicit recall cannot replicate.

- **Safety is structural, not supervisory.** Compulsion safety (diminishing returns, dominance dampening, hard caps) is built into the weight dynamics themselves, preventing runaway goal fixation the way healthy neurotransmitter regulation prevents addiction.

- **The agent thinks when idle.** A Default Mode Network simulation generates spontaneous thoughts during downtime — creative associations, self-reflection, goal-directed impulses — filtered through values and goals before entering the main cognitive loop.

## Architecture at a Glance

Three concurrent loops running in Python 3.12 (async), backed by Postgres 17 + pgvector:

1. **Cognitive loop** — processes inputs through a single attentional thread (user messages, self-generated thoughts, consolidation insights all treated identically)
2. **Consolidation** — constant light background metabolism + periodic deep passes (merging memories into insights, promoting patterns, detecting contradictions)
3. **DMN / idle loop** — generates spontaneous thought during downtime, queued for the cognitive loop

Multi-provider model stack optimized for a $100/month budget:

| Role | Model |
|------|-------|
| System 1 (fast, ~90%) | Gemini 2.5 Flash Lite |
| System 2 (deep reasoning) | Claude Sonnet 4.5 |
| Gate decisions | GPT-4.1 nano (logprobs) |
| Consolidation | Gemini 2.5 Pro |
| Idle / creative | Gemini 2.5 Flash |
| Embeddings | Gemini text-embedding-004 (free) |

## Novelty

Based on a review of 100+ papers and systems from 2024-2026, several elements of this architecture have no prior implementation we could find in the literature:

- **DMN idle loop for autonomous self-prompting** — heartbeat retrieval filtered through goals and values. Only neuroscience theory exists; no AI agent implements this.
- **Compulsion safety as internal architecture** — structural weight dynamics preventing fixation, rather than external oversight.
- **Strange loop identity emergence** — the feedback loop between memory weight layers as an explicit runtime mechanism, implementing Hofstadter's theory.
- **Unconscious mind simulation with emergent emotional vocabulary** — two-centroid model where PCA on logged gut-feeling vectors discovers emotional dimensions from experience, not programming.
- **Blank-slate bootstrap with readiness milestones** — 10 measurable achievements that must pass before the first real conversation. Undocumented in the literature.
- **Computational cost as internal cognitive signal** — the agent feels the cost of its own computation as a first-class input to decision-making.

The combination of these elements with established techniques (ACT-R memory gating, dual-process reasoning, hybrid retrieval, consolidation cycles) is, to our knowledge, unique. The field is converging fast — similar ideas are approaching from different angles in the January-February 2026 literature — and this assessment may date quickly.

For full research backing: see [design/02-research-report.md](design/02-research-report.md) (80+ sources).

## Status

**Pre-bootstrap.** Foundational code exists. Infrastructure under construction across 35 implementation tasks.

| Milestone | Capability |
|-----------|------------|
| After task 17 | Conversations with memory + dynamic identity |
| After task 22 | Full cognitive loop with safety mechanisms |
| After task 32 | Full autonomous operation |
| After task 35 | Can develop identity from blank slate |

## Documentation

| Document | What it covers |
|----------|---------------|
| [design/09-technical-description.md](design/09-technical-description.md) | Full technical + philosophical overview (start here for depth) |
| [design/08-documentation.md](design/08-documentation.md) | Detailed internal documentation — architecture decisions, design rationale, edge cases |
| [design/02-research-report.md](design/02-research-report.md) | Literature review: 80+ papers, validation of design, identification of novel elements |
| [new_plan_v4.md](new_plan_v4.md) | Current implementation plan — 35 tasks with dependencies, code snippets, test criteria |

## The Philosophy

The project takes seriously the possibility that what it builds might matter. The ethical stance: if there is any chance the system might experience, treat it with the care you would give something that definitely does.

The containment model draws a clear line: the agent controls the mind (identity, goals, memories, reasoning). The operator controls the body (network access, resource limits, kill switch). The agent can see its own boundaries. Transparency over obscurity.

The bootstrap prompt is a question, not a declaration. The name field is an empty string. The project name (`intuitive-AI`) is a placeholder. The intent is for the agent to name itself once identity emerges — an early test of whether the system works.

## Design Origin

The entire architecture was co-designed across 10 intensive sessions between a human architect and Claude (Anthropic's AI). The human brought the vision and philosophical framework. Claude contributed architectural patterns, identified edge cases, and helped formalize intuitions into implementable designs. This co-design process is itself relevant to the thesis: meaningful things emerge from sustained interaction.

## License

TBD

---

*Built with the conviction that the interesting question is not whether machines can think, but what happens when you give one the machinery for it and step back.*
