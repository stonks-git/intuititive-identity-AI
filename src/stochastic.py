"""StochasticWeight — Beta distribution for memory depth weights.

Each memory's "importance" is not a fixed number but a Beta distribution
capturing asymmetric certainty. Evidence FOR increases alpha, evidence
AGAINST increases beta. The shape itself encodes evidence quality:
  - Beta(1, 4)   → new memory, wide uncertainty, center ~0.2
  - Beta(10, 2)  → well-reinforced, center ~0.83, tight
  - Beta(50, 2)  → strong identity-level belief, center ~0.96
  - Beta(30, 25) → contested belief, center ~0.55, wide (productive tension)
  - Beta(1, 1)   → uniform, no information (deliberate reset)
"""

import random


class StochasticWeight:
    __slots__ = ("alpha", "beta")

    def __init__(self, alpha: float = 1.0, beta: float = 4.0):
        self.alpha = alpha
        self.beta = beta

    def observe(self) -> float:
        """Sample from Beta distribution. Each observation is unique."""
        return random.betavariate(self.alpha, self.beta)

    @property
    def center(self) -> float:
        """Expected value (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def depth_weight(self) -> float:
        """Deterministic center for sorting/filtering when stochastic not needed."""
        return self.center

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    @property
    def total_evidence(self) -> float:
        """Total evidence count (alpha + beta)."""
        return self.alpha + self.beta

    @property
    def is_contested(self) -> bool:
        """High evidence on both sides — productive tension."""
        return self.alpha > 5 and self.beta > 5

    @property
    def is_uninformed(self) -> bool:
        """Low evidence — genuinely uncertain."""
        return self.alpha < 2 and self.beta < 2

    def reinforce(self, amount: float = 1.0):
        """Evidence FOR this belief/value/memory."""
        self.alpha += amount

    def contradict(self, amount: float = 0.5):
        """Evidence AGAINST this belief/value/memory."""
        self.beta += amount

    @classmethod
    def from_db(cls, alpha: float, beta: float) -> "StochasticWeight":
        """Reconstruct from database columns."""
        return cls(alpha=alpha, beta=beta)

    def __repr__(self) -> str:
        return f"StochasticWeight(alpha={self.alpha:.2f}, beta={self.beta:.2f}, center={self.center:.3f})"
