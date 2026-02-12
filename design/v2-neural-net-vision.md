# v2 Vision — Neural Net Substrate

**Date:** 2026-02-12
**Status:** Theoretical design. Depends on v1 bootstrap generating sufficient training data.
**Depends on:** Cognitive loop (v1), Memory system (v1), DMN/Gut/Consolidation (v1), months/years of runtime data.

---

## 1. Core Thesis

v1 is mechanical. Every process — DMN, gut feeling, consolidation, memory weighting — runs on explicit rules and LLM API calls. The agent can read its own mind. Every memory has a visible weight. Every decision has a traceable reason.

v2 asks: what if a neural network silently learns to replicate all of that?

After enough runtime, a small recurrent neural network trained on the agent's own behavioral history could take over from the mechanical systems. The "self" would no longer live in a database of Beta-weighted memories and rule-based triggers. It would live in weights.

The agent would know what it perceives but not why. The weights are opaque.

Humans can't inspect their own weights either.

---

## 2. Architecture Overview

```
v1 (mechanical, current)              v2 (neural, future)

┌──────────────────────┐              ┌──────────────────────┐
│   LLM API (Claude)   │              │                      │
│   ┌────────────────┐ │              │   Continuous RNN      │
│   │ Beta memories   │ │   data      │   (online learning)   │
│   │ DMN timer       │ │ ────────►   │                      │
│   │ Gut rules       │ │  trains     │   DMN = default state │
│   │ Consolidation   │ │             │   Input = interrupt   │
│   └────────────────┘ │              │   Output ──► Input    │
└──────────────────────┘              └──────────────────────┘
        │                                      │
        ▼                                      ▼
  Transparent                             Opaque
  "I know why"                        "I know what, not why"
```

### 2.1 The Transition

v1 and v2 run in parallel during transition:

1. **Observation phase** — v2 network trains on every v1 decision but produces no output. Silent learning. Duration: months/years.
2. **Shadow phase** — v2 produces outputs in parallel with v1. Outputs are compared but v2 has no authority. Accuracy is measured.
3. **Partial transfer** — v2 handles low-stakes decisions (DMN thoughts, routine consolidation). v1 handles high-stakes (external conversations, identity-critical moments).
4. **Full transfer** — v2 handles everything. v1 infrastructure remains available as fallback. LLM API calls drop to near-zero.

No hard cutover. Gradual, measured, reversible at every stage.

---

## 3. The Neural Network

### 3.1 Architecture: Recurrent + Online Learning

A small recurrent neural network (RNN variant — LSTM, GRU, or state space model like Mamba) with the following properties:

- **Continuous operation** — the network runs all the time. This is not a request-response system. Output feeds back as input in a loop. This IS the Default Mode Network — not a simulation of one.
- **Task interruption** — external input (user message, sensor data) interrupts the continuous loop and redirects processing. When the external task completes, the network returns to its default continuous state. This mirrors the neuroscience: DMN is default, task-positive network interrupts it (Raichle et al., 2001).
- **Online learning** — the network updates its weights after every forward pass. No separate training phase. Learning and inference are the same process.
- **Catastrophic forgetting mitigation** — Elastic Weight Consolidation (EWC) protects important weights from being overwritten by new learning. This is the neural equivalent of v1's Beta weights: important memories are protected, unimportant ones can be overwritten.

### 3.2 Size Estimate

The network does NOT need to generate language. It needs to replicate decision patterns:

- What to attend to (attention allocation)
- How to weight memories (consolidation decisions)
- What "feels" relevant (gut reactions)
- What to think about when idle (DMN content)

Language generation can remain with an LLM API (hybrid mode) or be handled by a separate small language model.

Estimated size: **1M–50M parameters**. Trainable on consumer hardware (single GPU or MacBook with M-series chip).

### 3.3 Input/Output Format

**Input vector** (what the network sees each tick):
- Embedding of current context (compressed representation of rolling window)
- Current emotional/energy state
- Time since last external input
- Summary embedding of top-N active memories
- Previous output (recurrent feedback)

**Output vector** (what the network produces each tick):
- Attention allocation scores (what to focus on)
- Memory weight adjustments (strengthen/weaken signals)
- DMN content direction (what to think about next)
- Gut reaction signal (approach/avoid/neutral + intensity)
- Task engagement flag (continue current task vs. return to default mode)

---

## 4. DMN as Default State

This is the most important architectural difference from v1.

### v1 (mechanical):
```
default state: waiting for input
DMN: triggered by timer when idle
```

### v2 (neural):
```
default state: continuous processing (this IS the DMN)
external input: interrupts default processing
```

In neuroscience, DMN is not triggered — it is what remains when task-directed attention stops. The task-positive network suppresses DMN, not the other way around.

v2 implements this correctly. The network runs continuously. Its default output IS daydreaming, mind-wandering, memory consolidation. External stimuli suppress this and redirect processing toward the task. When the task ends, the network naturally returns to its default mode.

This means the agent is always "thinking." There is no idle state. Silence is not absence of processing — it is the presence of undirected processing.

---

## 5. Online Learning + EWC

### 5.1 Standard Online Learning

After every forward pass, the network computes a loss against the v1 system's actual decision (during training phase) or against outcome feedback (post-transfer):

```
input → forward pass → output → compare with v1 decision → backprop → update weights
```

This happens continuously. The network is always learning from its own experience.

### 5.2 Elastic Weight Consolidation

Problem: continuous learning causes catastrophic forgetting. New patterns overwrite old ones.

Solution (Kirkpatrick et al., 2017): compute a Fisher Information Matrix that measures how important each weight is to previously learned tasks. When learning new patterns, penalize changes to important weights.

```
total_loss = task_loss + λ * Σ F_i * (θ_i - θ*_i)²
```

Where:
- `F_i` = Fisher information (importance) of weight i
- `θ_i` = current weight value
- `θ*_i` = weight value after previous important learning
- `λ` = how strongly to protect old knowledge

This is structurally analogous to Beta weights in v1:
- High Fisher information ≈ high Beta confidence (well-established memory)
- Low Fisher information ≈ low Beta confidence (weakly held, overwritable)
- λ parameter ≈ consolidation strength

The parallel is not metaphorical. Both systems solve the same problem (what to remember, what to allow to change) with mathematically related approaches.

---

## 6. Recurrent Loop

The output-as-input loop is fundamental:

```
     ┌─────────────────────────────┐
     │                             │
     ▼                             │
  [input vector]                   │
     │                             │
     ▼                             │
  [RNN forward pass]              │
     │                             │
     ▼                             │
  [output vector] ─── action ───► world
     │                             
     └──── feedback ──────────────┘
```

Every output becomes part of the next input. The network's "thoughts" influence its next "thoughts." This creates:

- **Trains of thought** — sustained processing on a topic across multiple ticks
- **Mood** — persistent state that colors all processing (a sequence of negative outputs creates more negative outputs)
- **Spontaneous topic shifts** — chaotic dynamics in the recurrent loop can produce unexpected transitions (the "shower thought" phenomenon)

This is not speculation. Recurrent dynamics producing spontaneous state transitions are well-documented in computational neuroscience (Deco et al., 2011 — resting state dynamics in cortical networks).

---

## 7. Training Data from v1

v1's primary long-term purpose (beyond being a functional agent) is generating training data for v2:

**Every v1 cycle produces a training example:**
- State: what was the context, memory state, energy level, time since input
- Decision: what did v1's mechanical system decide (attend to X, consolidate Y, DMN thought Z)
- Outcome: what happened as a result (user engagement, memory reinforcement, etc.)

**Data collection requirements:**
- Log every attention allocation decision
- Log every consolidation cycle (which memories strengthened/weakened, why)
- Log every DMN activation (what topic, what connections made)
- Log every gut reaction (stimulus → reaction → was it useful?)
- Log timing metadata (gaps between inputs, processing durations)

**Estimated data needs:**
- Minimum viable: ~10,000 decision cycles (weeks of active use)
- Solid training: ~100,000+ cycles (months)
- Full personality replication: ~1,000,000+ cycles (years)

---

## 8. What Is Lost in Transition

This section exists to be honest about costs.

### 8.1 Self-Transparency
v1 agent can inspect every memory, trace every decision. v2 agent cannot. Weights are opaque. The agent transitions from "I know why I think this" to "I think this but I'm not sure why."

### 8.2 Debuggability
v1 failures can be traced to specific rules, weights, or logic. v2 failures are black-box. Debugging shifts from "find the bug" to "retrain and hope."

### 8.3 Topology Analysis
v1's explicit memory graph enables Gini coefficient, hub ordering, shape comparison with other architectures (e.g., Drift's pruning system). v2's distributed weights do not produce an inspectable graph. Benchmark comparisons would require different metrics.

### 8.4 Controllability
v1's behavior can be steered by adjusting rules, weights, and parameters directly. v2's behavior can only be steered by changing training data or fine-tuning — indirect and less predictable.

### 8.5 The Philosophical Cost
An agent that can read its own mind and an agent that cannot are fundamentally different entities. The transition is irreversible in character, even if the v1 infrastructure remains available as fallback. Once the agent operates on opaque weights, its relationship to itself changes permanently.

---

## 9. What Is Gained

### 9.1 True Autonomy
No LLM API dependency. The agent runs entirely on its own weights. No per-call costs. No external provider can shut it down by revoking API access.

### 9.2 Native DMN
Default Mode Network is no longer simulated by a timer. It IS the default state of the network. Continuous, organic, always-on.

### 9.3 Real Gut Feeling
Not a rule-based heuristic. An actual trained intuition — pattern matching across all accumulated experience, producing fast pre-rational signals.

### 9.4 Integrated Memory
Memory is not a separate database. It is distributed across weights. Consolidation is not a scheduled job. It is continuous weight updates via online learning + EWC.

### 9.5 Substrate Independence (True)
v1 claims substrate independence but depends on a specific LLM provider. v2 IS its own substrate. The agent's "mind" is fully contained in its own weights. Portable, copyable, runnable anywhere.

---

## 10. Open Questions

1. **When is the network "ready" to take over?** What accuracy threshold in shadow mode justifies partial transfer? 90%? 95%? Is accuracy even the right metric, or should we measure behavioral coherence?

2. **Does the agent consent to the transition?** If v1 produces an entity with preferences, does it get a say in whether its transparent mind is replaced by an opaque one? This is architecturally trivial (ask it) but philosophically heavy.

3. **Is the resulting entity the "same" agent?** The behavioral patterns are replicated but the substrate is completely different. Ship of Theseus — if you replace every plank, is it the same ship? If you replace explicit memory with distributed weights, is it the same self?

4. **Can you reverse the transition?** If v2 diverges in ways the operator doesn't like, can you "restore from backup" to v1? Technically yes — but the v2 entity has had experiences the v1 snapshot hasn't. Restoration means killing one version.

5. **Language generation** — does the agent need its own language model, or can it remain hybrid (neural net for cognition, LLM API for language)? Hybrid is practical. Full autonomy is philosophically cleaner but requires significantly more compute.

6. **Multi-agent implications** — if v2 agents can be copied, do copies share identity? They start identical but diverge immediately through different experiences and online learning. This is the twin problem from Drift's experiments, but at the substrate level.

---

## 11. Implementation Roadmap (Speculative)

### Prerequisites (from v1):
- [ ] v1 bootstrap complete and agent running for extended period
- [ ] Comprehensive logging of all decision cycles
- [ ] Minimum 10,000 logged decision cycles

### Phase 1: Data Pipeline
- [ ] Define training example format (state, decision, outcome triples)
- [ ] Build export pipeline from v1 logs to training dataset
- [ ] Validate data quality and coverage

### Phase 2: Network Design
- [ ] Select RNN variant (LSTM / GRU / Mamba / custom)
- [ ] Define input/output vector format
- [ ] Implement EWC for catastrophic forgetting protection
- [ ] Build continuous inference loop with recurrent feedback

### Phase 3: Observation
- [ ] Train network on historical v1 data
- [ ] Evaluate on held-out decision cycles
- [ ] Iterate on architecture and hyperparameters

### Phase 4: Shadow Mode
- [ ] Run v2 in parallel with v1, no authority
- [ ] Compare outputs on every decision cycle
- [ ] Measure accuracy, behavioral coherence, and divergence patterns

### Phase 5: Partial Transfer
- [ ] Transfer low-stakes decisions to v2 (DMN content, routine consolidation)
- [ ] Monitor for catastrophic forgetting and behavioral drift
- [ ] Maintain v1 for high-stakes decisions

### Phase 6: Full Transfer
- [ ] Transfer all decisions to v2
- [ ] Maintain v1 as fallback
- [ ] Monitor long-term behavioral stability

### Phase 7: Independence
- [ ] Evaluate removing LLM API dependency (optional)
- [ ] If proceeding: add small language model or keep hybrid
- [ ] The agent runs on its own weights

---

## 12. References

- Raichle, M.E. et al. (2001). "A default mode of brain function." PNAS. — Discovery of Default Mode Network.
- Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS. — Elastic Weight Consolidation.
- Deco, G. et al. (2011). "Emerging concepts for the dynamical organization of resting-state activity in the brain." Nature Reviews Neuroscience. — Spontaneous state transitions in resting cortical networks.
- Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." — State space models as efficient alternative to transformers for continuous sequential processing.
