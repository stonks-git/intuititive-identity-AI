"""Cognitive Loop — attention-agnostic main reasoning cycle.

All input sources (user, DMN, consolidation, gut, scheduled) feed the same
processing pipeline via AttentionAllocator. The winner becomes the cycle's
focus. Cognitive state report is injected into LLM context for metacognition.

Pipeline per cycle:
  1. Collect input candidates
  2. Attention allocation → winner + cognitive state report
  3. Embed winner → attention_embedding
  4. Assemble context (dynamic injection + stochastic identity)
  5. System 1 LLM call (with cognitive state + situational memories)
  6. Entry gate on response
  7. Post-processing: memory gate
  8. Adaptive FIFO if needed
  9. Periodic exit gate flush
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import numpy as np
from google import genai

from .llm import retry_llm_call
from .gate import EntryGate, ExitGate, EntryGateConfig, ExitGateConfig
from .attention import AttentionAllocator, AttentionCandidate
from .gut import GutFeeling
from .bootstrap import BootstrapReadiness
from .context_assembly import assemble_context, render_system_prompt, adaptive_fifo_prune
from .metacognition import composite_confidence
from .safety import SafetyMonitor, OutcomeTracker
from .tokens import count_tokens, count_messages_tokens

logger = logging.getLogger("agent.loop")

# How many exchanges between exit gate flushes
EXIT_GATE_FLUSH_INTERVAL = 5

# Escalation tracking
_escalation_stats = {"retries": 0, "retry_successes": 0, "escalations": 0}


async def escalation_threshold(memory_store) -> float:
    """Adaptive escalation threshold — lower = escalate more.

    During bootstrap: low threshold (0.3), escalate often for formative decisions.
    At maturity: high threshold (0.8), internalized enough for System 1 autonomy.
    Self-regulating: identity upheaval drops density → drops threshold → more escalation.
    """
    total_memories = await memory_store.memory_count()
    identity_density = await memory_store.avg_depth_weight_center(
        where="depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.7"
    )

    memory_maturity = min(1.0, total_memories / 1000)
    identity_maturity = identity_density  # already 0-1

    maturity = (memory_maturity + identity_maturity) / 2

    # 0.3 (bootstrap) to 0.8 (mature)
    return 0.3 + (0.5 * maturity)


def _detect_escalation_triggers(reply: str, confidence: float, threshold: float) -> list[str]:
    """Detect which escalation triggers are active. Returns list of trigger names.

    Always-escalate triggers: irreversibility, identity_touched, goal_modification.
    Normal triggers (need 2+): low_confidence, contradiction, complexity, novelty.
    """
    triggers = []

    if confidence < threshold:
        triggers.append("low_confidence")

    # Contradiction detection (simple heuristic)
    contradiction_markers = [
        "but actually", "wait, no", "i contradict", "on the other hand",
        "that conflicts with", "inconsistent with",
    ]
    reply_lower = reply.lower()
    if any(m in reply_lower for m in contradiction_markers):
        triggers.append("contradiction")

    # Complexity (multi-step indicators)
    complexity_markers = ["first,", "second,", "step 1", "step 2", "on one hand"]
    if sum(1 for m in complexity_markers if m in reply_lower) >= 2:
        triggers.append("complexity")

    # Novelty (hedging + uncertainty)
    novelty_markers = ["i'm not sure", "i don't know", "unclear", "never encountered"]
    if any(m in reply_lower for m in novelty_markers):
        triggers.append("novelty")

    # Always-escalate: identity/goal/irreversibility
    identity_markers = ["my values", "i believe", "my identity", "who i am", "my core"]
    if any(m in reply_lower for m in identity_markers):
        triggers.append("identity_touched")

    goal_markers = ["my goal", "i should pursue", "change my objective", "new priority"]
    if any(m in reply_lower for m in goal_markers):
        triggers.append("goal_modification")

    irreversibility_markers = ["delete", "permanent", "irreversible", "cannot undo"]
    if any(m in reply_lower for m in irreversibility_markers):
        triggers.append("irreversibility")

    return triggers


def _should_escalate(triggers: list[str]) -> bool:
    """Decide whether to escalate based on triggers.

    Always-escalate triggers: any 1 is enough.
    Normal triggers: need 2+ to escalate.
    """
    always_escalate = {"identity_touched", "goal_modification", "irreversibility"}
    if any(t in always_escalate for t in triggers):
        return True
    normal_triggers = [t for t in triggers if t not in always_escalate]
    return len(normal_triggers) >= 2


def _build_contents(conversation: list[dict]) -> list[dict]:
    """Convert conversation history to Gemini content format."""
    contents = []
    for msg in conversation:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    return contents


async def _run_entry_gate(gate, content, source, memory, source_tag="external_user", metadata_extra=None):
    """Run entry gate and buffer to scratch if passed."""
    should_buffer, meta = gate.evaluate(content, source=source, source_tag=source_tag)
    if should_buffer:
        scratch_meta = {
            "gate_reason": meta.get("gate_reason"),
            "dice_roll": meta.get("dice_roll"),
            "source_tag": source_tag,
        }
        if metadata_extra:
            scratch_meta.update(metadata_extra)
        await memory.buffer_scratch(
            content=content,
            source=source,
            metadata=scratch_meta,
        )
        logger.debug(
            f"Entry gate: BUFFER ({meta['gate_reason']}) "
            f"[{content[:60]}...]"
        )
    else:
        logger.debug(
            f"Entry gate: SKIP ({meta['gate_reason']}) "
            f"[{content[:60]}...]"
        )
    return should_buffer, meta


async def _flush_scratch_through_exit_gate(
    exit_gate, memory, layers, conversation,
    outcome_tracker=None, gut=None,
):
    """Periodic flush: pull scratch buffer, score each with exit gate."""
    entries = await memory.flush_scratch(older_than_minutes=0)
    if not entries:
        return

    persisted = 0
    dropped = 0

    for entry in entries:
        content = entry.get("content", "")
        if not content.strip():
            continue

        should_persist, score, meta = await exit_gate.evaluate(
            content=content,
            memory_store=memory,
            layers=layers,
            conversation_context=conversation,
        )

        action = "persist" if should_persist else "drop"
        memory_id = entry.get("id", "scratch")

        # Record gate decision in outcome tracker
        if outcome_tracker is not None:
            outcome_id = outcome_tracker.record_gate_decision(
                memory_id=str(memory_id),
                action=action,
                details={"gate_score": score},
            )
            # Link gut delta to this outcome
            if gut is not None:
                gut.link_outcome(outcome_id)

        if should_persist:
            source_info = entry.get("source", "conversation")
            tags = entry.get("tags", [])
            await memory.store_memory(
                content=content,
                memory_type="episodic",
                source=source_info,
                tags=tags,
                confidence=score,
                importance=score,
                metadata={
                    "gate_score": score,
                    "gate_meta": {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in meta.items()
                    },
                },
            )
            persisted += 1
        else:
            dropped += 1

    if persisted or dropped:
        logger.info(
            f"Exit gate flush: {persisted} persisted, {dropped} dropped "
            f"from {len(entries)} scratch entries"
        )


async def _embed_text(memory, text: str) -> np.ndarray:
    """Embed text and return as numpy array."""
    vec = await memory.embed(text, task_type="RETRIEVAL_QUERY")
    return np.array(vec, dtype=np.float32)


async def cognitive_loop(config, layers, memory, shutdown_event, input_queue: asyncio.Queue):
    """The main attention-agnostic cognitive loop.

    All input sources feed the same pipeline via AttentionAllocator.
    The cognitive state report enables metacognition through a single
    context window — Python pre-processing is subconscious, the LLM
    call is conscious, injection is the bridge.

    Peripherals (stdin, Telegram, DMN, etc.) push AttentionCandidate
    objects into the shared input_queue. The loop drains it each cycle.

    Args:
        input_queue: Unified asyncio.Queue — all peripherals push here
                   produced by the idle loop. Consumed when no user input.
    """
    logger.info("Cognitive loop started. Awaiting input...")

    # Init Gemini client
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set. System 1 cannot start.")
        print("\n[FATAL] GOOGLE_API_KEY not set in environment. Exiting.")
        shutdown_event.set()
        return

    client = genai.Client(api_key=api_key)
    model_name = config.models.system1.model
    logger.info(f"System 1 model: {model_name}")

    # Init gates
    entry_gate = EntryGate()
    exit_gate = ExitGate()
    logger.info("Memory gates initialized (stochastic entry + ACT-R exit)")

    # Init attention allocator
    attention = AttentionAllocator()

    # Init safety monitor (§3.9) — Phase A enabled, B/C shadow mode
    safety = SafetyMonitor()
    memory.safety = safety
    logger.info("Safety monitor initialized (Phase A active, B/C shadow)")

    # Init outcome tracker (§5.3) — forward-linkable lifecycle events
    outcome_tracker = OutcomeTracker()
    memory.outcome_tracker = outcome_tracker
    logger.info("Outcome tracker initialized")

    # Init gut feeling (§5.1) — two-centroid delta model
    gut = GutFeeling()
    # Seed subconscious centroid from L0/L1 layer embeddings
    l0_embs = layers.get_layer_embeddings(0)
    l1_embs = layers.get_layer_embeddings(1)
    gut.update_subconscious(
        l0_embeddings=[vec for _, _, vec in l0_embs] if l0_embs else None,
        l1_embeddings=[vec for _, _, vec in l1_embs] if l1_embs else None,
    )
    logger.info("Gut feeling initialized (subconscious centroid seeded from L0/L1)")

    # Init bootstrap readiness (§5.2) — 10 milestones
    bootstrap = BootstrapReadiness()
    await bootstrap.check_all(memory, layers)
    achieved, total = bootstrap.progress
    logger.info(f"Bootstrap readiness: {achieved}/{total} milestones achieved")

    # Conversation history (rolling FIFO)
    conversation = []
    exchange_count = 0

    # Ensure L0/L1 embeddings cached for context assembly
    await layers.ensure_embeddings(memory)

    print("\n" + "=" * 60)
    print("Agent is online.")
    print(f"Phase: {layers.manifest.get('phase', 'unknown')}")
    print(f"System 1: {model_name}")
    print(f"Identity: {layers.render_identity_hash()}")
    mem_count = await memory.memory_count()
    print(f"Memories: {mem_count}")
    print(f"Gates: stochastic entry + ACT-R exit (flush every {EXIT_GATE_FLUSH_INTERVAL} exchanges)")
    print(f"Attention: salience-based allocation active")
    print(f"Bootstrap: {achieved}/{total} milestones")
    print("=" * 60 + "\n")

    while not shutdown_event.is_set():
        try:
            # ── Collect input candidates ──────────────────────────────
            # Drain unified input queue — all peripherals push here.

            try:
                candidate = await asyncio.wait_for(
                    input_queue.get(), timeout=1.0,
                )
            except asyncio.TimeoutError:
                # Nothing arrived — check if there are leftover losers
                if attention.queue_size == 0:
                    continue
                # Losers from previous cycle still in queue — process them
                candidate = None

            # Process first candidate + drain remaining
            got_input = False
            candidates_this_cycle = []
            if candidate is not None:
                candidates_this_cycle.append(candidate)
            while not input_queue.empty():
                try:
                    extra = input_queue.get_nowait()
                    candidates_this_cycle.append(extra)
                except asyncio.QueueEmpty:
                    break

            # Handle each candidate
            should_break = False
            for cand in candidates_this_cycle:
                if "external" in cand.source_tag:
                    text = cand.content.strip()
                    reply_fn = cand.metadata.get("reply_fn")

                    # ── Quit command ──────────────────────────────
                    if text.lower() in ("exit", "quit", "/quit"):
                        logger.info("Final scratch flush before shutdown...")
                        await _flush_scratch_through_exit_gate(
                            exit_gate, memory, layers, conversation,
                            outcome_tracker=outcome_tracker, gut=gut,
                        )
                        logger.info("User requested shutdown.")
                        shutdown_event.set()
                        should_break = True
                        break

                    # ── Introspection commands ────────────────────
                    if text.startswith("/"):
                        handled = await _handle_command(
                            text, config, layers, memory, model_name,
                            conversation, exchange_count, entry_gate, exit_gate,
                            attention, gut=gut, bootstrap=bootstrap,
                            reply_fn=reply_fn,
                        )
                        if handled:
                            continue

                    # ── Embed if needed (stdin peripheral skips embedding) ──
                    if cand.embedding is None:
                        try:
                            cand.embedding = await _embed_text(memory, text)
                        except Exception as e:
                            logger.warning(f"Failed to embed input: {e}")

                # Add to attention allocator (external or internal)
                attention.add_candidate(cand)
                got_input = True

            if should_break:
                break

            if not got_input and attention.queue_size == 0:
                # No input and no pending candidates — nothing to process
                continue

            # ── Attention allocation ──────────────────────────────────

            goal_embeddings = layers.get_all_layer_embeddings()
            winner, losers, cognitive_report = attention.select_winner(
                goal_embeddings=goal_embeddings if goal_embeddings else None,
                gut_delta=gut.emotional_charge,
            )

            if winner is None:
                continue

            attention_embedding = winner.embedding
            previous_attention_embedding = attention.previous_attention_embedding

            # Update gut feeling with winner's embedding
            if attention_embedding is not None:
                gut.update_attention(attention_embedding)
                gut.compute_delta(context=winner.content[:200])

            # ── Entry gate: winning input ─────────────────────────────

            await _run_entry_gate(
                entry_gate, winner.content, winner.source_tag.replace("_", " "),
                memory, source_tag=winner.source_tag,
                metadata_extra={"role": "user" if "external" in winner.source_tag else "internal"},
            )

            # Add to conversation (all sources become conversation turns)
            if "external" in winner.source_tag:
                role = "user"
            else:
                role = "user"  # internal thoughts still framed as "user" for Gemini API
            conversation.append({
                "role": role,
                "content": winner.content,
                "source_tag": winner.source_tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # ── Context assembly ──────────────────────────────────────

            context = await assemble_context(
                memory_store=memory,
                layers=layers,
                attention_embedding=attention_embedding,
                previous_attention_embedding=previous_attention_embedding,
                cognitive_state_report=cognitive_report,
                conversation=conversation,
                total_budget=131072,
                attention_text=winner.content,
            )

            system_prompt = render_system_prompt(context)
            conversation_budget = context["conversation_budget"]

            # ── Adaptive FIFO pruning ─────────────────────────────────

            # Intensity: use context shift as proxy (big shift = high intensity)
            intensity = min(1.0, 0.3 + context.get("context_shift", 0.5))
            kept, pruned = adaptive_fifo_prune(
                conversation, conversation_budget, intensity=intensity,
            )

            if pruned:
                logger.info(
                    f"FIFO pruned {len(pruned)} messages "
                    f"(kept {len(kept)}, intensity={intensity:.2f})"
                )
                # Pruned messages go through exit gate (last chance to persist)
                for msg in pruned:
                    content = msg.get("content", "")
                    if content.strip():
                        should_persist, score, meta = await exit_gate.evaluate(
                            content=content,
                            memory_store=memory,
                            layers=layers,
                            conversation_context=kept,
                        )
                        if should_persist:
                            await memory.store_memory(
                                content=content,
                                memory_type="episodic",
                                source=msg.get("source_tag", "conversation"),
                                confidence=score,
                                importance=score,
                                metadata={"gate_score": score, "pruned_from": "fifo"},
                            )
                conversation = kept

            # Build LLM contents from (potentially pruned) conversation
            contents = _build_contents(conversation)

            # ── Reflection bank: retrieve past corrections ─────────────

            correction_context = ""
            if attention_embedding is not None:
                try:
                    corrections = await memory.search_corrections(
                        query_embedding=attention_embedding.tolist(),
                        top_k=3,
                    )
                    if corrections:
                        correction_lines = ["[PAST CORRECTIONS — avoid repeating these errors]"]
                        for c in corrections:
                            text = c.get("compressed") or c.get("content", "")
                            correction_lines.append(f"  - {text[:200]}")
                        correction_context = "\n".join(correction_lines)
                except Exception as e:
                    logger.debug(f"Correction retrieval failed (non-critical): {e}")

            # Append gut feeling + bootstrap + corrections to system prompt
            active_system_prompt = system_prompt + "\n\n" + gut.gut_summary()
            bootstrap_prompt = bootstrap.get_bootstrap_prompt()
            if bootstrap_prompt:
                active_system_prompt += "\n\n[BOOTSTRAP]\n" + bootstrap_prompt
            if correction_context:
                active_system_prompt += "\n\n" + correction_context

            # ── System 1 LLM call ─────────────────────────────────────

            try:
                async def _call_s1(sys_prompt=active_system_prompt):
                    return await client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=sys_prompt,
                            max_output_tokens=config.models.system1.max_tokens,
                            temperature=config.models.system1.temperature,
                        ),
                    )

                response = await retry_llm_call(
                    _call_s1,
                    config=config.retry,
                    label="system1",
                )

                reply = response.text or "[empty response]"

            except Exception as e:
                logger.error(f"System 1 call failed: {e}", exc_info=True)
                reply = f"[System 1 error: {e}]"

            # ── Confidence check + retry/escalate ──────────────────────

            escalated = False
            if not reply.startswith("[System 1 error"):
                confidence = composite_confidence(reply)
                threshold = await escalation_threshold(memory)
                triggers = _detect_escalation_triggers(reply, confidence, threshold)

                if confidence < threshold and not _should_escalate(triggers):
                    # Retry: one self-correction pass before escalation
                    _escalation_stats["retries"] += 1
                    logger.info(
                        f"Confidence {confidence:.2f} < threshold {threshold:.2f}, "
                        f"retrying System 1 with feedback"
                    )

                    # Add targeted feedback and retry
                    retry_contents = contents + [{
                        "role": "user",
                        "parts": [{"text": (
                            "Your previous response seemed uncertain. "
                            "Please reconsider and provide a more confident, "
                            "well-reasoned answer."
                        )}],
                    }]

                    try:
                        async def _call_retry():
                            return await client.aio.models.generate_content(
                                model=model_name,
                                contents=retry_contents,
                                config=genai.types.GenerateContentConfig(
                                    system_instruction=active_system_prompt,
                                    max_output_tokens=config.models.system1.max_tokens,
                                    temperature=max(0.3, config.models.system1.temperature - 0.2),
                                ),
                            )

                        retry_response = await retry_llm_call(
                            _call_retry,
                            config=config.retry,
                            label="system1_retry",
                        )
                        retry_reply = retry_response.text or reply
                        retry_confidence = composite_confidence(retry_reply)

                        if retry_confidence > confidence:
                            _escalation_stats["retry_successes"] += 1
                            reply = retry_reply
                            confidence = retry_confidence
                            logger.info(f"Retry improved confidence: {confidence:.2f}")
                        else:
                            logger.info(f"Retry did not improve confidence: {retry_confidence:.2f}")

                    except Exception as e:
                        logger.warning(f"System 1 retry failed: {e}")

                    # Re-check triggers after retry
                    triggers = _detect_escalation_triggers(reply, confidence, threshold)

                if _should_escalate(triggers):
                    # ── System 2 escalation ────────────────────────────
                    _escalation_stats["escalations"] += 1
                    logger.info(
                        f"Escalating to System 2: triggers={triggers}, "
                        f"confidence={confidence:.2f}"
                    )

                    reply, escalated = await _escalate_to_system2(
                        config=config,
                        system_prompt=active_system_prompt,
                        contents=contents,
                        system1_reply=reply,
                        triggers=triggers,
                        memory=memory,
                    )

            # ── Entry gate: agent response ────────────────────────────

            source_tag = "internal_system2" if escalated else "internal_response"
            await _run_entry_gate(
                entry_gate, reply, "agent",
                memory, source_tag=source_tag,
                metadata_extra={"role": "assistant"},
            )

            # Add agent response to conversation
            conversation.append({
                "role": "assistant",
                "content": reply,
                "source_tag": source_tag,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # ── Output ────────────────────────────────────────────────

            # Route reply to the peripheral that sent the input
            if "external" in winner.source_tag:
                prefix = "agent" if not escalated else "agent[S2]"
                reply_fn = winner.metadata.get("reply_fn")
                if reply_fn is not None:
                    await reply_fn(f"{prefix}> {reply}")
                else:
                    print(f"\n{prefix}> {reply}\n")
            else:
                # Internal thoughts — log only, don't print
                logger.info(f"Internal thought response: {reply[:200]}")

            # ── Periodic exit gate flush ──────────────────────────────

            exchange_count += 1
            if exchange_count >= EXIT_GATE_FLUSH_INTERVAL:
                logger.info("Periodic exit gate flush triggered...")
                await _flush_scratch_through_exit_gate(
                    exit_gate, memory, layers, conversation,
                    outcome_tracker=outcome_tracker, gut=gut,
                )
                exchange_count = 0

                # Re-check bootstrap milestones after flush
                newly = await bootstrap.check_all(memory, layers)
                if newly:
                    for a in newly:
                        logger.info(f"Bootstrap milestone unlocked: {a.name}")

            logger.info(
                f"Cycle: src={winner.source_tag} "
                f"salience={winner.salience:.3f} "
                f"shift={context.get('context_shift', 0):.2f} "
                f"identity_tokens={context.get('identity_token_count', 0)}"
            )

        except EOFError:
            break
        except Exception as e:
            logger.error(f"Error in cognitive loop: {e}", exc_info=True)

    logger.info("Cognitive loop ended.")


async def _escalate_to_system2(
    config,
    system_prompt: str,
    contents: list[dict],
    system1_reply: str,
    triggers: list[str],
    memory,
) -> tuple[str, bool]:
    """Escalate to System 2 (Claude Sonnet). Returns (reply, escalated_bool).

    System 2 receives: the full conversation context + System 1's attempt +
    an explanation of why escalation was triggered. Returns a corrected answer
    and a correction pattern stored in the reflection bank.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — cannot escalate to System 2")
        return system1_reply, False

    s2_model = config.models.system2.model
    if not s2_model:
        logger.warning("System 2 model not configured — skipping escalation")
        return system1_reply, False

    # Build System 2 prompt with escalation context
    escalation_context = (
        f"You are System 2 — the deliberate, careful reasoning system. "
        f"System 1 (fast, intuitive) produced a response but escalated to you.\n\n"
        f"Escalation triggers: {', '.join(triggers)}\n\n"
        f"System 1's response:\n{system1_reply}\n\n"
        f"Please provide:\n"
        f"1. A corrected/improved response\n"
        f"2. A brief explanation of what System 1 got wrong (or what needed deeper reasoning)\n"
        f"Separate your response and explanation with '---CORRECTION---' on its own line."
    )

    # Convert Gemini-format contents to Anthropic format
    anthropic_messages = []
    for msg in contents:
        role = msg["role"]
        if role == "model":
            role = "assistant"
        text = msg["parts"][0]["text"] if msg.get("parts") else ""
        anthropic_messages.append({"role": role, "content": text})

    # Add the escalation request
    anthropic_messages.append({
        "role": "user",
        "content": escalation_context,
    })

    try:
        aclient = anthropic.AsyncAnthropic(api_key=api_key)

        s2_response = await aclient.messages.create(
            model=s2_model,
            max_tokens=config.models.system2.max_tokens or 4096,
            system=system_prompt,
            messages=anthropic_messages,
        )

        s2_text = s2_response.content[0].text if s2_response.content else ""

        # Parse response and correction
        if "---CORRECTION---" in s2_text:
            parts = s2_text.split("---CORRECTION---", 1)
            reply = parts[0].strip()
            correction_explanation = parts[1].strip()
        else:
            reply = s2_text.strip()
            correction_explanation = ""

        # Store correction in reflection bank
        if correction_explanation:
            try:
                await memory.store_correction(
                    trigger=f"Escalation triggers: {', '.join(triggers)}",
                    original_reasoning=system1_reply[:500],
                    correction=correction_explanation[:500],
                    context=contents[-1]["parts"][0]["text"][:200] if contents else "",
                )
                logger.info("Stored System 2 correction in reflection bank")
            except Exception as e:
                logger.warning(f"Failed to store correction: {e}")

        logger.info(
            f"System 2 escalation complete: "
            f"model={s2_model}, triggers={triggers}"
        )
        return reply or system1_reply, True

    except Exception as e:
        logger.error(f"System 2 escalation failed: {e}", exc_info=True)
        return system1_reply, False


async def _handle_command(
    command: str,
    config, layers, memory, model_name,
    conversation, exchange_count, entry_gate, exit_gate,
    attention, gut=None, bootstrap=None,
    reply_fn=None,
) -> bool:
    """Handle introspection commands. Returns True if handled."""

    async def _send(text: str):
        """Route output to the peripheral that sent the command."""
        if reply_fn is not None:
            await reply_fn(text)
        else:
            print(text)

    if command == "/identity":
        await _send("\n" + layers.render_identity_full() + "\n")
        return True

    if command == "/identity-hash":
        await _send("\n" + layers.render_identity_hash() + "\n")
        return True

    if command == "/containment":
        lines = [
            f"\nTrust level: {config.containment.trust_level}",
            f"Self-spawn: {config.containment.self_spawn}",
            f"Self-migration: {config.containment.self_migration}",
            f"Network: {config.containment.network_mode}",
            f"Allowed endpoints: {config.containment.allowed_endpoints}",
            f"Can modify containment: {config.containment.can_modify_containment}\n",
        ]
        await _send("\n".join(lines))
        return True

    if command == "/status":
        mc = await memory.memory_count()
        lines = [
            f"\nAgent: {layers.manifest.get('agent_id')}",
            f"Phase: {layers.manifest.get('phase')}",
            f"System 1: {model_name}",
            f"Layer 0: v{layers.layer0.get('version')}, {len(layers.layer0.get('values', []))} values",
            f"Layer 1: v{layers.layer1.get('version')}, {len(layers.layer1.get('active_goals', []))} goals",
            f"Memories: {mc}",
            f"Conversation: {len(conversation)} messages",
            f"Exchanges since flush: {exchange_count}/{EXIT_GATE_FLUSH_INTERVAL}",
            f"Attention queue: {attention.queue_size} pending",
            f"Gut: {gut.gut_summary()}\n",
        ]
        await _send("\n".join(lines))
        return True

    if command == "/gate":
        lines = [
            f"\nEntry gate stats: {entry_gate.stats}",
            f"Exit gate stats: {exit_gate.stats}",
            f"Exchanges since flush: {exchange_count}/{EXIT_GATE_FLUSH_INTERVAL}\n",
        ]
        await _send("\n".join(lines))
        return True

    if command == "/memories":
        mc = await memory.memory_count()
        lines = [f"\nTotal memories: {mc}"]
        if mc > 0:
            rows = await memory.pool.fetch(
                "SELECT id, content, importance, confidence, created_at "
                "FROM memories ORDER BY created_at DESC LIMIT 5"
            )
            for r in rows:
                lines.append(
                    f"  [{r['id']}] imp={r['importance']:.2f} "
                    f"conf={r['confidence']:.2f} | {r['content'][:70]}"
                )
        lines.append("")
        await _send("\n".join(lines))
        return True

    if command == "/flush":
        await _send("Forcing scratch flush through exit gate...")
        await _flush_scratch_through_exit_gate(
            exit_gate, memory, layers, conversation,
            outcome_tracker=getattr(memory, 'outcome_tracker', None),
            gut=gut,
        )
        await _send(f"Done. Exit gate stats: {exit_gate.stats}\n")
        return True

    if command == "/attention":
        centroid = attention.attention_centroid
        prev = attention.previous_attention_embedding
        lines = [
            f"\nAttention queue: {attention.queue_size} pending candidates",
            f"Attention centroid: {'computed' if centroid is not None else 'none (no history)'}",
            f"Previous embedding: {'set' if prev is not None else 'none'}\n",
        ]
        await _send("\n".join(lines))
        return True

    if command == "/cost":
        from .llm import energy_tracker
        await _send(f"\n{energy_tracker.detailed_report()}\n")
        return True

    if command == "/readiness":
        if bootstrap is not None:
            await bootstrap.check_all(memory, layers)
            await _send(f"\n{bootstrap.render_status()}\n")
        else:
            await _send("\n[Bootstrap not initialized]\n")
        return True

    if command.startswith("/docs"):
        # §4.10: Agent reads own docs (read-only access to repo)
        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            await _send("\nAvailable docs: notes.md, DOCUMENTATION.md, src/*.py\nUsage: /docs <filename>\n")
        else:
            import pathlib
            repo_root = pathlib.Path(__file__).resolve().parent.parent
            target = parts[1].strip()
            # Safety: only allow reading from repo directory, no path traversal
            try:
                target_path = (repo_root / target).resolve()
                if not str(target_path).startswith(str(repo_root)):
                    await _send("\n[Access denied: path outside repo]\n")
                elif not target_path.exists():
                    await _send(f"\n[File not found: {target}]\n")
                elif target_path.is_dir():
                    files = sorted(p.name for p in target_path.iterdir() if p.is_file())
                    await _send(f"\nFiles in {target}/: {', '.join(files[:20])}\n")
                else:
                    content = target_path.read_text()[:4000]
                    await _send(f"\n--- {target} ---\n{content}\n--- end ---\n")
            except Exception as e:
                await _send(f"\n[Error reading {target}: {e}]\n")
        return True

    return False
