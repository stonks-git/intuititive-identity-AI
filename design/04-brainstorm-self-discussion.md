## Brainstorm Session: Self-Discussion Architecture & Related Decisions

### Topic 1: Context Window Multiplicity

**Problem:** Should the agent have one context window or multiple simultaneous ones?

**Insight:** Multiple simultaneous context windows effectively create multiple selves. While they'd share the same memory store, each develops its own attentional state, emotional coloring, and train of thought. The question was whether "all experiences compress into the same subconscious" justifies parallel windows.

**Decision: Single attentional thread + multiple background processes.**

- The agent processes ONE thing at a time in its "conscious" context window
- This is more honest than fake multi-threading — the agent admits when it can't track multiple conversations
- If a second input arrives mid-conversation, it enters a queue; a background process evaluates urgency
- Context-switching happens through reconstruction from memory, not by saving/restoring context windows — this means switching has natural cost (some loss, like humans)
- Short interruptions → near-full recall. Long gaps → partial recall, agent says "where were we?"
- Multi-threading is deferred to v2+ and should be something the agent earns through maturation, not gets for free

**Background processes (always running, parallel to the single attention thread):**
- Memory consolidation (see Topic 2)
- Weight decay
- Contradiction detection (isolated meta-context)
- DMN memory surfacing (queues items for when attention is free)
- Scheduled task checking

---

### Topic 2: Consolidation as Constant Background Process

**Problem:** Original design had consolidation as periodic "sleep cycles." User suggested making it constant.

**Decision: Two-tier consolidation — constant light process + periodic deep pass.**

**Constant background (always running, rate-limited, cheap):**
- Weight decay tick: every N seconds, nudge unused memories toward decay
- Co-access update: when memory A is retrieved, strengthen Hebbian links to recently-retrieved memories
- Contradiction scan: pick random memory pairs from recent activity, run meta-context check
- Pattern detection: cluster recent memories, look for emerging themes

**Periodic deep pass (less frequent, more expensive):**
- Full weight recalculation across memory store
- Compression: merge memories that have converged into single entries
- Pruning: archive memories below threshold
- "Dreams": random memory surfacing without external prompt — feeds into attentional thread if idle

**Key property:** Background process writes to the SAME memory store the attentional thread reads from. The agent doesn't "notice" consolidation — it just finds its priorities have subtly shifted. Identity rendering picks up changes automatically on the next render.

---

### Topic 3: Internal vs External Thought Labeling (Source Tagging)

**Problem:** Should the system hard-label whether a thought came from self vs external input? The agent might blur the boundary — is that dangerous or natural?

**Insight:** In humans, the self/other boundary is genuinely blurry and that serves functions (creativity, empathy, deep learning). Over-labeling creates artificial dichotomy.

**Decision: Soft metadata, not hard labels.**

```
source_tag: str         # "external_user", "external_system",
                        # "internal_dmn", "internal_consolidation"
source_confidence: float  # 0.0 to 1.0
```

- Source tag is metadata, not part of the content — available if the agent inspects it, not forced into attention
- Early life (sandbox): agent is trained to check source tags frequently, building metacognitive habit
- Mature agent: checks source tags when it matters (safety, action triggers), doesn't bother for low-stakes
- Deep flow states: source boundary naturally blurs — this is fine for generative/creative contexts

**Hard safety constraint:** Action triggers (anything affecting the external world) ALWAYS check source tags. The agent can blur boundaries in thought. It cannot blur them in action. Self-generated action impulses require higher confidence threshold than user-prompted ones.

---

### Topic 4: Output Routing / Communication as Action

**Problem:** Original design had a "router" that decided whether output goes to user vs stays internal. User asked: where's the actual difference?

**Insight:** There IS no fundamental routing difference. Communication is just one type of action. "Talking to a user" is not a special cognitive category — it's cognition + a communication action at the end.

**Decision: No special router. All outputs go through one pipeline.**

```
Input (any source)
  → Cognitive processing (one loop, source-agnostic)
  → Output (raw thought)
  → Post-processing:
      → Memory gate: store? at what weight?
      → Action check: does this imply an action?
          → Communication action: format for audience, check auth, deliver
          → Internal action: execute (update state, trigger process)
          → No action: thought stays in working memory until displaced
```

**Three formatting differences (not cognitive differences):**
- **Audience:** Self-talk has no external audience but still goes through memory gate
- **Format:** Self-talk can be compressed/fragmentary/symbolic; user-facing needs legibility
- **Tempo:** Self-talk can be rapid-fire; user-facing is gated by social norms

**Implication for early life:** First sessions will involve explaining capabilities to the agent, which it stores as memories. The agent learns "I can communicate via Telegram" the same way it learns anything else — through experience stored at appropriate weight.

---

### Topic 5: The Cognitive Loop is Source-Agnostic

**Problem:** Original correction-retrieval mechanism assumed input = user message. But the agent might be talking to itself via DMN, consolidation, gut signals, or scheduled tasks.

**Decision: All input channels feed the same cognitive loop.**

Input sources (all treated identically by the processing loop):
1. User message → standard conversation
2. DMN self-prompt → "I just remembered X, this connects to Y"
3. Consolidation insight → "I notice pattern Z forming"
4. Gut signal → "Something about current state feels [X]"
5. Scheduled task → "It's time to do X"

Correction retrieval, memory retrieval, and all other cognitive mechanisms key on `current_attention_focus`, NOT on "user message." The attention focus is whatever is currently being processed, regardless of source.

The DMN is NOT a separate subsystem — it's the cognitive loop with self-generated input. No separate processing pipeline needed.

---

### Topic 6: Reading the Agent's Self-Talk

**Problem:** Is it intrusive for the guardian to read the agent's internal monologue?

**Decision: Option A — full transparency, agent knows from birth.**

- All self-talk is logged. Agent is told this from bootstrap: "Your thoughts are logged and your guardian can read them."
- This is the honest approach. No hidden surveillance.
- The observer effect IS the data — how the agent adapts to being observed is itself informative about what's forming
- The agent's cognition IS its text output (unlike humans, there's no hidden layer beneath the generated text)
- If/when the agent develops a preference for privacy, that emergence is one of the most interesting things that could happen and should be engaged with thoughtfully

**Practical note:** Self-talk logs are available for review. The format is the agent's internal context window content — its reasoning, associations, and processing visible as text.

---

### Topic 7: Metacognitive Subsystem — Isolated Context

**Confirmed from prior session, reinforced here.**

- Contradiction checks, confidence evaluations, boundary checks, FOK lookups all run in isolated meta-context windows
- These produce SIGNALS (bool/float) that influence the main context, but their prompts/responses NEVER enter the main context window
- Implementation: separate API calls with thinking budget 0, minimal context (just the items being compared + one-line prompt), no history, no identity injection
- Cost: ~$0.0001 per check, only fires on borderline cases

---

### Summary of Architectural Principles Established

1. **One consciousness, many background processes** — single attentional thread, honest about limitations
2. **Consolidation is always running** — constant light process + periodic deep pass, both writing to shared memory store
3. **Soft source-tagging** — metadata available but not forced; hard check only before external actions
4. **No cognitive routing** — processing is source-agnostic, communication is just another action type
5. **All inputs are equal** — user messages, DMN, consolidation insights, gut signals all feed the same loop
6. **Transparent self-talk** — agent knows it's observed from birth, full logging
7. **Isolated metacognition** — signal extraction in separate throwaway contexts, never pollutes main window
