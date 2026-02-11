"""Safety Ceilings — hard caps, diminishing returns, rate limiters, entropy, circuit breaker.

ALL mechanisms built from day one. Enabled in phases:
  Phase A (immediate): Hard ceiling, diminishing returns, audit trail
  Phase B (consolidation): Rate limiter, two-gate guardrail
  Phase C (patterns emerge): Entropy monitor, circuit breaker, CBA coherence

Disabled phases run in shadow mode: audit log captures what WOULD have
triggered, enabling validation before enabling enforcement.
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("agent.safety")


# ── SAFETY EVENT LOG ────────────────────────────────────────────────────────


@dataclass
class SafetyEvent:
    """Record of a safety ceiling evaluation."""
    ceiling: str
    action: dict
    reason: str
    enforced: bool
    timestamp: float = field(default_factory=time.time)


# Module-level audit log (survives per-process, flushed by consolidation)
_audit_log: list[SafetyEvent] = []

# Maximum audit log size before oldest entries are dropped
_MAX_AUDIT_LOG = 1000


def log_safety_event(
    ceiling: str, action: dict, reason: str, enforced: bool,
) -> None:
    """Log a safety event to audit trail and Python logger."""
    event = SafetyEvent(
        ceiling=ceiling, action=action, reason=reason, enforced=enforced,
    )
    _audit_log.append(event)
    if len(_audit_log) > _MAX_AUDIT_LOG:
        _audit_log.pop(0)

    mode = "ENFORCED" if enforced else "SHADOW"
    logger.info(f"[SAFETY:{mode}] {ceiling}: {reason} | {action.get('memory_id', '?')}")


def get_audit_log(last_n: int = 50) -> list[SafetyEvent]:
    """Retrieve recent safety events."""
    return _audit_log[-last_n:]


def clear_audit_log() -> int:
    """Clear and return count of cleared events."""
    count = len(_audit_log)
    _audit_log.clear()
    return count


# ── BASE CLASS ──────────────────────────────────────────────────────────────


class SafetyCeiling:
    """Base class for all safety mechanisms."""

    def __init__(self, name: str, enabled: bool = False):
        self.name = name
        self.enabled = enabled

    def check(self, action: dict) -> tuple[bool, str]:
        """Check if an action passes this ceiling.

        Returns (passed, reason). When disabled (shadow mode), always passes
        but still logs what would have been blocked.
        """
        passed, reason = self._evaluate(action)
        if not passed:
            log_safety_event(self.name, action, reason, enforced=self.enabled)
            if self.enabled:
                return False, reason
        return True, ""

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        raise NotImplementedError


# ── PHASE A: ENABLED IMMEDIATELY ────────────────────────────────────────────


class HardCeiling(SafetyCeiling):
    """No single memory weight center > 0.95 (except immutable).
    No goal-like memory > 40% of total goal-weight budget.
    """

    MAX_CENTER = 0.95
    MAX_GOAL_BUDGET_FRACTION = 0.40

    def __init__(self, enabled: bool = True):
        super().__init__("hard_ceiling", enabled)

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        # Skip immutable memories — they're exempt from the ceiling
        if action.get("is_immutable", False):
            return True, ""

        current_alpha = action.get("current_alpha", 1.0)
        current_beta = action.get("current_beta", 4.0)
        delta_alpha = action.get("delta_alpha", 0.0)
        delta_beta = action.get("delta_beta", 0.0)

        new_alpha = current_alpha + delta_alpha
        new_beta = current_beta + delta_beta
        new_center = new_alpha / (new_alpha + new_beta)

        if new_center > self.MAX_CENTER:
            return False, (
                f"Weight center {new_center:.3f} would exceed cap {self.MAX_CENTER}. "
                f"Alpha {current_alpha:.2f}+{delta_alpha:.2f}, "
                f"Beta {current_beta:.2f}+{delta_beta:.2f}"
            )

        # Goal budget check (only if goal context provided)
        if action.get("is_goal", False):
            goal_total = action.get("goal_weight_total", 0)
            if goal_total > 0:
                this_weight = new_center
                fraction = this_weight / goal_total
                if fraction > self.MAX_GOAL_BUDGET_FRACTION:
                    return False, (
                        f"Goal memory would consume {fraction:.1%} of goal budget "
                        f"(cap: {self.MAX_GOAL_BUDGET_FRACTION:.0%})"
                    )

        return True, ""


class DiminishingReturns(SafetyCeiling):
    """Apply diminishing returns: gain / max(1, log2(total_evidence)).

    More evidence → smaller effect per reinforcement. Prevents runaway.
    This ceiling is special: it TRANSFORMS the gain rather than blocking.
    """

    def __init__(self, enabled: bool = True):
        super().__init__("diminishing_returns", enabled)

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        # Always passes — transformation happens in apply()
        return True, ""

    def apply(self, gain: float, current_alpha: float, current_beta: float) -> float:
        """Transform gain through diminishing returns.

        Returns adjusted gain. Active regardless of enabled flag
        (this is a transformation, not a gate).
        """
        total_evidence = current_alpha + current_beta
        divisor = max(1.0, math.log2(total_evidence))
        adjusted = gain / divisor
        if adjusted < gain * 0.5:
            log_safety_event(
                self.name,
                {"gain": gain, "total_evidence": total_evidence},
                f"Gain reduced {gain:.3f} → {adjusted:.3f} (evidence={total_evidence:.1f})",
                enforced=True,
            )
        return adjusted


# ── PHASE B: ENABLED WHEN CONSOLIDATION STARTS ─────────────────────────────


class RateLimiter(SafetyCeiling):
    """No weight changes > 10% per consolidation cycle."""

    MAX_CHANGE_PER_CYCLE = 0.10

    def __init__(self, enabled: bool = False):
        super().__init__("rate_limiter", enabled)
        # Track changes per memory per cycle: {cycle_id: {memory_id: total_delta}}
        self._cycle_changes: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        cycle_id = action.get("cycle_id")
        if not cycle_id:
            return True, ""  # Not in a consolidation cycle

        memory_id = action.get("memory_id", "")
        current_alpha = action.get("current_alpha", 1.0)
        current_beta = action.get("current_beta", 4.0)
        delta_alpha = action.get("delta_alpha", 0.0)
        delta_beta = action.get("delta_beta", 0.0)

        current_center = current_alpha / (current_alpha + current_beta)
        new_center = (current_alpha + delta_alpha) / (
            current_alpha + delta_alpha + current_beta + delta_beta
        )
        change = abs(new_center - current_center)

        accumulated = self._cycle_changes[cycle_id][memory_id]
        if accumulated + change > self.MAX_CHANGE_PER_CYCLE:
            return False, (
                f"Rate limit: accumulated change {accumulated + change:.3f} "
                f"exceeds {self.MAX_CHANGE_PER_CYCLE} per cycle"
            )

        # Track the change
        self._cycle_changes[cycle_id][memory_id] += change
        return True, ""

    def end_cycle(self, cycle_id: str) -> None:
        """Clean up tracking for a completed cycle."""
        self._cycle_changes.pop(cycle_id, None)


class TwoGateGuardrail(SafetyCeiling):
    """Before any parameter change: (1) validation margin, (2) capacity cap.

    Gate 1 (validation margin): proposed change must have supporting evidence
           from at least 2 independent sources OR high confidence.
    Gate 2 (capacity cap): total parameter changes this cycle must not exceed
           a global budget.
    """

    MAX_CHANGES_PER_CYCLE = 50  # Global cap on parameter changes per cycle

    def __init__(self, enabled: bool = False):
        super().__init__("two_gate_guardrail", enabled)
        self._cycle_change_counts: dict[str, int] = defaultdict(int)

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        # Gate 1: Validation margin
        evidence_count = action.get("evidence_count", 0)
        confidence = action.get("confidence", 0.5)
        if evidence_count < 2 and confidence < 0.7:
            return False, (
                f"Two-gate G1: insufficient evidence ({evidence_count}) "
                f"and low confidence ({confidence:.2f})"
            )

        # Gate 2: Capacity cap
        cycle_id = action.get("cycle_id")
        if cycle_id:
            self._cycle_change_counts[cycle_id] += 1
            if self._cycle_change_counts[cycle_id] > self.MAX_CHANGES_PER_CYCLE:
                return False, (
                    f"Two-gate G2: cycle change cap exceeded "
                    f"({self._cycle_change_counts[cycle_id]}/{self.MAX_CHANGES_PER_CYCLE})"
                )

        return True, ""

    def end_cycle(self, cycle_id: str) -> None:
        """Clean up tracking for a completed cycle."""
        self._cycle_change_counts.pop(cycle_id, None)


# ── PHASE C: ENABLED WHEN PATTERNS EMERGE ───────────────────────────────────


class EntropyMonitor(SafetyCeiling):
    """Shannon entropy of weight distribution. Entropy drop → broaden sampling.

    Not a gate per se — monitors and flags when entropy falls below threshold.
    """

    ENTROPY_FLOOR = 2.0  # Minimum acceptable Shannon entropy (bits)

    def __init__(self, enabled: bool = False):
        super().__init__("entropy_monitor", enabled)
        self.last_entropy: float | None = None

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        # Expects action["weight_centers"] = list of depth_weight centers
        centers = action.get("weight_centers", [])
        if len(centers) < 5:
            return True, ""  # Not enough data to evaluate

        entropy = self.compute_entropy(centers)
        self.last_entropy = entropy

        if entropy < self.ENTROPY_FLOOR:
            return False, (
                f"Entropy {entropy:.3f} bits below floor {self.ENTROPY_FLOOR}. "
                f"Weight distribution is collapsing — broaden sampling."
            )

        return True, ""

    @staticmethod
    def compute_entropy(centers: list[float], bins: int = 20) -> float:
        """Compute Shannon entropy of weight distribution (in bits)."""
        if not centers:
            return 0.0

        # Bin the centers into a histogram
        counts = [0] * bins
        for c in centers:
            idx = min(int(c * bins), bins - 1)
            counts[idx] += 1

        total = len(centers)
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy


class CircuitBreaker(SafetyCeiling):
    """N consecutive cycles reinforcing same pattern without new evidence → pause.

    Detects runaway self-reinforcement loops.
    """

    MAX_CONSECUTIVE = 5  # Consecutive reinforcements before tripping

    def __init__(self, enabled: bool = False):
        super().__init__("circuit_breaker", enabled)
        # {memory_id: {"consecutive": int, "last_evidence_hash": str}}
        self._patterns: dict[str, dict] = defaultdict(
            lambda: {"consecutive": 0, "last_evidence_hash": ""}
        )

    def _evaluate(self, action: dict) -> tuple[bool, str]:
        memory_id = action.get("memory_id", "")
        evidence_hash = action.get("evidence_hash", "")
        is_reinforcement = action.get("delta_alpha", 0) > 0

        if not is_reinforcement:
            return True, ""

        pattern = self._patterns[memory_id]

        if evidence_hash and evidence_hash != pattern["last_evidence_hash"]:
            # New evidence — reset counter
            pattern["consecutive"] = 1
            pattern["last_evidence_hash"] = evidence_hash
            return True, ""

        # Same evidence or no evidence hash — increment
        pattern["consecutive"] += 1

        if pattern["consecutive"] >= self.MAX_CONSECUTIVE:
            return False, (
                f"Circuit breaker: {memory_id} reinforced {pattern['consecutive']} "
                f"consecutive cycles without new evidence"
            )

        return True, ""

    def reset(self, memory_id: str) -> None:
        """Reset circuit breaker for a memory."""
        self._patterns.pop(memory_id, None)


# ── SAFETY MONITOR (CENTRAL COORDINATOR) ────────────────────────────────────


class SafetyMonitor:
    """Central safety coordinator. Holds all ceilings, provides unified interface."""

    def __init__(self):
        # Phase A — enabled immediately
        self.hard_ceiling = HardCeiling(enabled=True)
        self.diminishing_returns = DiminishingReturns(enabled=True)

        # Phase B — shadow mode until consolidation activates
        self.rate_limiter = RateLimiter(enabled=False)
        self.two_gate = TwoGateGuardrail(enabled=False)

        # Phase C — shadow mode until patterns emerge
        self.entropy_monitor = EntropyMonitor(enabled=False)
        self.circuit_breaker = CircuitBreaker(enabled=False)

    def apply_gain(
        self,
        gain: float,
        current_alpha: float,
        current_beta: float,
    ) -> float:
        """Apply diminishing returns to a proposed gain.

        Always active. Returns adjusted gain amount.
        """
        return self.diminishing_returns.apply(gain, current_alpha, current_beta)

    def check_weight_change(
        self,
        memory_id: str,
        current_alpha: float,
        current_beta: float,
        delta_alpha: float = 0.0,
        delta_beta: float = 0.0,
        is_immutable: bool = False,
        is_goal: bool = False,
        goal_weight_total: float = 0.0,
        evidence_count: int = 0,
        confidence: float = 0.5,
        cycle_id: str | None = None,
        evidence_hash: str = "",
    ) -> tuple[bool, float, float, list[str]]:
        """Check all ceilings for a proposed weight change.

        Returns (allowed, adjusted_delta_alpha, adjusted_delta_beta, reasons).
        Delta values may be adjusted by diminishing returns even when allowed.
        """
        reasons = []

        # Apply diminishing returns to alpha gain
        if delta_alpha > 0:
            delta_alpha = self.apply_gain(delta_alpha, current_alpha, current_beta)

        # Build action dict for ceiling checks
        action = {
            "memory_id": memory_id,
            "current_alpha": current_alpha,
            "current_beta": current_beta,
            "delta_alpha": delta_alpha,
            "delta_beta": delta_beta,
            "is_immutable": is_immutable,
            "is_goal": is_goal,
            "goal_weight_total": goal_weight_total,
            "evidence_count": evidence_count,
            "confidence": confidence,
            "cycle_id": cycle_id,
            "evidence_hash": evidence_hash,
        }

        # Run through all ceilings
        for ceiling in [
            self.hard_ceiling,
            self.rate_limiter,
            self.two_gate,
            self.circuit_breaker,
        ]:
            passed, reason = ceiling.check(action)
            if not passed:
                reasons.append(f"{ceiling.name}: {reason}")
                return False, 0.0, 0.0, reasons

        return True, delta_alpha, delta_beta, reasons

    def check_entropy(self, weight_centers: list[float]) -> tuple[bool, str]:
        """Check system-wide weight entropy. Call periodically."""
        return self.entropy_monitor.check({"weight_centers": weight_centers})

    def enable_phase_b(self) -> None:
        """Enable Phase B ceilings (when consolidation starts)."""
        self.rate_limiter.enabled = True
        self.two_gate.enabled = True
        logger.info("Safety Phase B enabled: rate limiter + two-gate guardrail")

    def enable_phase_c(self) -> None:
        """Enable Phase C ceilings (when patterns emerge)."""
        self.entropy_monitor.enabled = True
        self.circuit_breaker.enabled = True
        logger.info("Safety Phase C enabled: entropy monitor + circuit breaker")

    def end_consolidation_cycle(self, cycle_id: str) -> None:
        """Clean up per-cycle state."""
        self.rate_limiter.end_cycle(cycle_id)
        self.two_gate.end_cycle(cycle_id)


# ── OUTCOME TRACKER (§5.3) ─────────────────────────────────────────────────


@dataclass
class OutcomeRecord:
    """A tracked lifecycle event with forward-linkable ID."""
    outcome_id: str
    event_type: str       # "gate_decision", "promotion", "demotion", "gut_delta"
    memory_id: str
    action: str           # "persist", "skip", "promote", "decay", etc.
    details: dict
    timestamp: float = field(default_factory=time.time)
    linked: bool = False  # Set True when outcome is linked back


class OutcomeTracker:
    """Track gate decisions and promotions with forward-linkable IDs.

    Enables PCA axes that correlate with outcomes (fear/hope axes).
    When outcomes become apparent, link back via outcome_id.
    """

    def __init__(self):
        self.records: list[OutcomeRecord] = []
        self._next_id = 0
        self._max_records = 2000

    def _generate_id(self) -> str:
        self._next_id += 1
        return f"out_{self._next_id}_{int(time.time())}"

    def record_gate_decision(
        self,
        memory_id: str,
        action: str,
        details: dict | None = None,
    ) -> str:
        """Record a gate decision. Returns outcome_id for forward linking."""
        outcome_id = self._generate_id()
        record = OutcomeRecord(
            outcome_id=outcome_id,
            event_type="gate_decision",
            memory_id=memory_id,
            action=action,
            details=details or {},
        )
        self._append(record)
        return outcome_id

    def record_promotion(
        self,
        memory_id: str,
        from_center: float,
        to_center: float,
        gain: float,
        details: dict | None = None,
    ) -> str:
        """Record a promotion event."""
        outcome_id = self._generate_id()
        record = OutcomeRecord(
            outcome_id=outcome_id,
            event_type="promotion",
            memory_id=memory_id,
            action="promote",
            details={
                "from_center": from_center,
                "to_center": to_center,
                "gain": gain,
                **(details or {}),
            },
        )
        self._append(record)
        return outcome_id

    def record_demotion(
        self,
        memory_id: str,
        from_center: float,
        to_center: float,
        details: dict | None = None,
    ) -> str:
        """Record a demotion/decay event."""
        outcome_id = self._generate_id()
        record = OutcomeRecord(
            outcome_id=outcome_id,
            event_type="demotion",
            memory_id=memory_id,
            action="decay",
            details={
                "from_center": from_center,
                "to_center": to_center,
                **(details or {}),
            },
        )
        self._append(record)
        return outcome_id

    def link_outcome(self, outcome_id: str, result: str, quality: float) -> bool:
        """Link an outcome back to its record when result becomes apparent.

        Args:
            outcome_id: The forward-linkable ID.
            result: Description of what happened.
            quality: 0.0 (bad) to 1.0 (good).
        """
        for record in reversed(self.records):
            if record.outcome_id == outcome_id:
                record.linked = True
                record.details["outcome_result"] = result
                record.details["outcome_quality"] = quality
                return True
        return False

    def get_linked_outcomes(self, event_type: str | None = None) -> list[OutcomeRecord]:
        """Get records that have been linked to outcomes (for PCA)."""
        return [
            r for r in self.records
            if r.linked and (event_type is None or r.event_type == event_type)
        ]

    def _append(self, record: OutcomeRecord):
        self.records.append(record)
        if len(self.records) > self._max_records:
            self.records.pop(0)
