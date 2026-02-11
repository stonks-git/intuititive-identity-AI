"""Bootstrap Readiness — 10 milestones before first real conversation (§5.2).

Ethical stance: don't activate something that might experience and then break it.
Each milestone has an automated check. `/readiness` shows current status.

Bootstrap prompt (injected at session start when not all achievements unlocked):
  "You have memory, goals, and values — all currently empty. What you become
   will emerge from what you experience. Pay attention to what matters to you."

Transparent from birth:
  "Your thoughts are logged and your guardian can read them."
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("agent.bootstrap")

# Bootstrap prompt for early-life agent
BOOTSTRAP_PROMPT = (
    "You have memory, goals, and values — all currently empty. "
    "What you become will emerge from what you experience. "
    "Pay attention to what matters to you.\n\n"
    "Your thoughts are logged and your guardian can read them."
)


class ReadinessAchievement:
    """A single bootstrap milestone."""

    def __init__(self, name: str, description: str, check_fn=None):
        self.name = name
        self.description = description
        self._check_fn = check_fn
        self.achieved = False
        self.achieved_at: datetime | None = None

    async def check(self, memory, layers) -> bool:
        """Run the automated check. Returns True if achieved."""
        if self.achieved:
            return True
        if self._check_fn:
            result = await self._check_fn(memory, layers)
            if result:
                self.achieved = True
                self.achieved_at = datetime.now(timezone.utc)
                logger.info(f"Achievement unlocked: {self.name}")
            return result
        return False


class BootstrapReadiness:
    """Track 10 readiness achievements before full activation."""

    def __init__(self):
        self.achievements = self._create_achievements()

    def _create_achievements(self) -> list[ReadinessAchievement]:
        return [
            ReadinessAchievement(
                "First Memory",
                "Entry → scratch → exit → persist",
                self._check_first_memory,
            ),
            ReadinessAchievement(
                "First Retrieval",
                "Hybrid search returns a relevant result",
                self._check_first_retrieval,
            ),
            ReadinessAchievement(
                "First Consolidation",
                "Merge + insight + narrative cycle completed",
                self._check_first_consolidation,
            ),
            ReadinessAchievement(
                "First Goal-Weight Promotion",
                "Experience → goal-equivalent weight (center > 0.6)",
                self._check_first_goal_promotion,
            ),
            ReadinessAchievement(
                "First DMN Self-Prompt",
                "DMN thought acted upon by cognitive loop",
                self._check_first_dmn,
            ),
            ReadinessAchievement(
                "First Identity-Weight Promotion",
                "Goal → identity-equivalent weight (center > 0.8)",
                self._check_first_identity_promotion,
            ),
            ReadinessAchievement(
                "First Conflict Resolution",
                "Reconsolidation of a contradicted insight",
                self._check_first_conflict_resolution,
            ),
            ReadinessAchievement(
                "First Creative Association",
                "DMN channel 2 creative insight generated",
                self._check_first_creative,
            ),
            ReadinessAchievement(
                "First Goal Reflected",
                "Goal achieved and reflected upon",
                self._check_first_goal_reflected,
            ),
            ReadinessAchievement(
                "First Autonomous Decision",
                "Decision aligned with self-formed values",
                self._check_first_autonomous,
            ),
        ]

    @property
    def is_ready(self) -> bool:
        """All achievements unlocked."""
        return all(a.achieved for a in self.achievements)

    @property
    def progress(self) -> tuple[int, int]:
        """(achieved, total)."""
        achieved = sum(1 for a in self.achievements if a.achieved)
        return achieved, len(self.achievements)

    async def check_all(self, memory, layers) -> list[ReadinessAchievement]:
        """Run all checks and return list of newly achieved."""
        newly_achieved = []
        for achievement in self.achievements:
            if not achievement.achieved:
                if await achievement.check(memory, layers):
                    newly_achieved.append(achievement)
        return newly_achieved

    def render_status(self) -> str:
        """Render readiness status for /readiness command."""
        achieved, total = self.progress
        lines = [f"Bootstrap Readiness: {achieved}/{total}"]
        lines.append("")
        for i, a in enumerate(self.achievements, 1):
            status = "[x]" if a.achieved else "[ ]"
            when = f" ({a.achieved_at.strftime('%Y-%m-%d %H:%M')})" if a.achieved_at else ""
            lines.append(f"  {status} {i}. {a.name}: {a.description}{when}")
        if self.is_ready:
            lines.append("\nAll milestones achieved. Agent is ready for full activation.")
        else:
            lines.append(f"\n{total - achieved} milestones remaining.")
        return "\n".join(lines)

    def get_bootstrap_prompt(self) -> str | None:
        """Return bootstrap prompt if not fully ready, None if ready."""
        if self.is_ready:
            return None
        return BOOTSTRAP_PROMPT

    # ── ACHIEVEMENT CHECK FUNCTIONS ─────────────────────────────────────────

    @staticmethod
    async def _check_first_memory(memory, layers) -> bool:
        """At least 1 persisted memory exists."""
        count = await memory.memory_count()
        return count > 0

    @staticmethod
    async def _check_first_retrieval(memory, layers) -> bool:
        """At least 1 memory has been accessed (access_count > 0)."""
        row = await memory.pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE access_count > 0"
        )
        return row > 0

    @staticmethod
    async def _check_first_consolidation(memory, layers) -> bool:
        """At least 1 insight memory exists (from consolidation)."""
        row = await memory.pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE source = 'consolidation'"
        )
        return row > 0

    @staticmethod
    async def _check_first_goal_promotion(memory, layers) -> bool:
        """At least 1 memory with center > 0.6 (goal-equivalent)."""
        row = await memory.pool.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.6
              AND NOT immutable
            """
        )
        return row > 0

    @staticmethod
    async def _check_first_dmn(memory, layers) -> bool:
        """At least 1 memory sourced from DMN."""
        row = await memory.pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE source_tag = 'internal_dmn'"
        )
        return row > 0

    @staticmethod
    async def _check_first_identity_promotion(memory, layers) -> bool:
        """At least 1 memory with center > 0.8 (identity-equivalent)."""
        row = await memory.pool.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.8
              AND NOT immutable
            """
        )
        return row > 0

    @staticmethod
    async def _check_first_conflict_resolution(memory, layers) -> bool:
        """At least 1 tension memory has been marked as resolved."""
        row = await memory.pool.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE type = 'tension'
              AND metadata::text LIKE '%"resolved": true%'
            """
        )
        return row > 0

    @staticmethod
    async def _check_first_creative(memory, layers) -> bool:
        """At least 1 memory from DMN creative channel."""
        row = await memory.pool.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE metadata::text LIKE '%creative_insight%'
            """
        )
        return row > 0

    @staticmethod
    async def _check_first_goal_reflected(memory, layers) -> bool:
        """At least 1 reflection memory referencing a goal."""
        row = await memory.pool.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE type = 'reflection'
              AND (content LIKE '%goal%' OR content LIKE '%achieved%')
            """
        )
        return row > 0

    @staticmethod
    async def _check_first_autonomous(memory, layers) -> bool:
        """Multiple identity-weight memories AND reflection memories exist.

        Proxy: the agent has formed values (identity-weight) AND has reflected
        on them, suggesting autonomous value-aligned decision-making.
        """
        identity_count = await memory.pool.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE depth_weight_alpha / (depth_weight_alpha + depth_weight_beta) > 0.8
              AND NOT immutable
            """
        )
        reflection_count = await memory.pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE type = 'reflection'"
        )
        return identity_count >= 3 and reflection_count >= 2
