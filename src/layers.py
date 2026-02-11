"""Layer Store — manages Layer 0 (identity), Layer 1 (goals), Layer 2 (memory).

Also provides embedding cache for L0/L1 items, used by:
  - Subconscious centroid (5.1)
  - ACT-R spreading activation (2.7)
  - Hybrid relevance semantic component (3.1)
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

logger = logging.getLogger("agent.layers")


class LayerStore:
    """Manages the three memory layers on disk, with embedding cache."""

    def __init__(self, agent_home: Path):
        self.agent_home = agent_home
        self.layer0_path = agent_home / "identity" / "layer0.json"
        self.layer1_path = agent_home / "goals" / "layer1.json"
        self.manifest_path = agent_home / "manifest.json"
        self._embed_cache_path = agent_home / "cache" / "layer_embeddings.json"

        self.layer0: dict = {}
        self.layer1: dict = {}
        self.manifest: dict = {}
        # Cache: {text_hash: list[float]}
        self._embed_cache: dict[str, list[float]] = {}

    def load(self):
        """Load all layers and embedding cache from disk."""
        if self.layer0_path.exists():
            with open(self.layer0_path) as f:
                self.layer0 = json.load(f)
            logger.info(f"Layer 0 loaded: v{self.layer0.get('version', 0)}")

        if self.layer1_path.exists():
            with open(self.layer1_path) as f:
                self.layer1 = json.load(f)
            logger.info(f"Layer 1 loaded: v{self.layer1.get('version', 0)}")

        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
            logger.info(f"Manifest loaded: {self.manifest.get('agent_id')}")

        if self._embed_cache_path.exists():
            with open(self._embed_cache_path) as f:
                self._embed_cache = json.load(f)
            logger.info(f"Layer embedding cache loaded: {len(self._embed_cache)} entries")

    def save(self):
        """Persist all layers to disk."""
        self._save_with_history(self.layer0, self.layer0_path, "identity")
        self._save_with_history(self.layer1, self.layer1_path, "goals")

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        logger.info("All layers saved to disk.")

    def _save_with_history(self, data: dict, path: Path, layer_name: str):
        """Save layer and keep versioned history."""
        # Write current version
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Archive to history
        version = data.get("version", 0)
        history_dir = path.parent / f"{layer_name.split('/')[-1]}_history"
        if not history_dir.exists():
            # History dir might be named differently based on path
            history_dir = path.parent / f"layer{0 if 'identity' in str(path) else 1}_history"

        if history_dir.exists():
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            history_path = history_dir / f"v{version}_{ts}.json"
            shutil.copy2(path, history_path)

    # --- EMBEDDING CACHE ---

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _save_embed_cache(self):
        self._embed_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._embed_cache_path, "w") as f:
            json.dump(self._embed_cache, f)

    async def ensure_embeddings(self, memory_store) -> int:
        """Embed any L0/L1 items not yet in cache. Returns count of new embeddings."""
        texts_to_embed = []
        hashes = []

        for item in self._iter_layer_items():
            h = self._text_hash(item["text"])
            if h not in self._embed_cache:
                texts_to_embed.append(item["text"])
                hashes.append(h)

        if not texts_to_embed:
            return 0

        embeddings = await memory_store.embed_batch(
            texts_to_embed, task_type="RETRIEVAL_DOCUMENT", title="identity_goal",
        )
        for h, emb in zip(hashes, embeddings):
            self._embed_cache[h] = emb

        self._save_embed_cache()
        logger.info(f"Embedded {len(texts_to_embed)} new L0/L1 items")
        return len(texts_to_embed)

    def get_layer_embeddings(self, layer: int) -> list[tuple[str, float, np.ndarray]]:
        """Get (text, weight, embedding_vector) for items in a layer.

        Returns empty list at bootstrap (blank L0/L1).
        """
        results = []
        for item in self._iter_layer_items():
            if item["layer"] != layer:
                continue
            h = self._text_hash(item["text"])
            if h in self._embed_cache:
                vec = np.array(self._embed_cache[h], dtype=np.float32)
                results.append((item["text"], item["weight"], vec))
        return results

    def get_all_layer_embeddings(self) -> list[tuple[str, float, np.ndarray]]:
        """Get embeddings for all L0 + L1 items."""
        return self.get_layer_embeddings(0) + self.get_layer_embeddings(1)

    def _iter_layer_items(self):
        """Yield dicts with text, weight, layer for all L0/L1 items."""
        # L0 values
        for v in self.layer0.get("values", []):
            text = v.get("value", "")
            if text:
                yield {"text": text, "weight": v.get("weight", 0.5), "layer": 0}
        # L0 beliefs
        for b in self.layer0.get("beliefs", []):
            text = b.get("belief", "")
            if text:
                yield {"text": text, "weight": b.get("confidence", 0.5), "layer": 0}
        # L0 boundaries
        for b in self.layer0.get("boundaries", []):
            text = b.get("description", "")
            if text:
                yield {"text": text, "weight": 1.0, "layer": 0}
        # L1 goals
        for g in self.layer1.get("active_goals", []):
            text = g.get("description", "") or g.get("goal", "")
            if text:
                yield {"text": text, "weight": g.get("weight", 0.5), "layer": 1}

    def render_identity_hash(self) -> str:
        """Tier 1: compressed identity for every prompt (~100-200 tokens)."""
        parts = []

        # Name/persona
        core = self.layer0.get("core", {})
        if core.get("name"):
            parts.append(f"You are {core['name']}.")
        if core.get("voice"):
            parts.append(f"Voice: {core['voice']}.")

        # Top values by weight
        values = sorted(
            self.layer0.get("values", []),
            key=lambda v: v.get("weight", 0),
            reverse=True,
        )[:5]
        if values:
            val_strs = [f"{v['value']}({v['weight']:.1f})" for v in values]
            parts.append(f"Values: {', '.join(val_strs)}.")

        # Active goals by weight
        goals = sorted(
            self.layer1.get("active_goals", []),
            key=lambda g: g.get("weight", 0),
            reverse=True,
        )[:3]
        if goals:
            goal_strs = [f"{g['description']}({g['weight']:.1f})" for g in goals]
            parts.append(f"Goals: {', '.join(goal_strs)}.")

        # Critical boundaries
        boundaries = [
            b["description"]
            for b in self.layer0.get("boundaries", [])
            if b.get("type") == "hard"
        ]
        if boundaries:
            parts.append(f"Boundaries: {'; '.join(boundaries)}.")

        return " ".join(parts) if parts else "Identity: bootstrapping. No values or goals yet."

    def render_identity_full(self) -> str:
        """Tier 2: full identity for triggered injection (~1-2k tokens)."""
        sections = []

        # Full persona
        core = self.layer0.get("core", {})
        if any(core.values()):
            sections.append("## Identity")
            if core.get("name"):
                sections.append(f"Name: {core['name']}")
            if core.get("persona"):
                sections.append(f"Persona: {core['persona']}")
            if core.get("voice"):
                sections.append(f"Voice: {core['voice']}")

        # All values with weights
        values = self.layer0.get("values", [])
        if values:
            sections.append("\n## Values")
            for v in sorted(values, key=lambda x: x.get("weight", 0), reverse=True):
                sections.append(
                    f"- {v['value']} (weight: {v.get('weight', 0):.2f}, "
                    f"evidence: {v.get('evidence_count', 0)})"
                )

        # All beliefs
        beliefs = self.layer0.get("beliefs", [])
        if beliefs:
            sections.append("\n## Beliefs")
            for b in sorted(beliefs, key=lambda x: x.get("confidence", 0), reverse=True):
                sections.append(
                    f"- {b['belief']} (confidence: {b.get('confidence', 0):.2f}, "
                    f"evidence: {b.get('evidence_count', 0)}, "
                    f"contradictions: {b.get('contradictions', 0)})"
                )

        # Boundaries
        boundaries = self.layer0.get("boundaries", [])
        if boundaries:
            sections.append("\n## Boundaries")
            for b in boundaries:
                questionable = " (questionable)" if b.get("questionable") else " (hard)"
                sections.append(f"- {b['description']}{questionable}")

        # All active goals
        goals = self.layer1.get("active_goals", [])
        if goals:
            sections.append("\n## Active Goals")
            for g in sorted(goals, key=lambda x: x.get("weight", 0), reverse=True):
                sections.append(
                    f"- {g.get('description', '')} (weight: {g.get('weight', 0):.2f})"
                )

        if not sections:
            return "Identity is bootstrapping. No values, beliefs, or goals have formed yet. Everything is to be discovered through experience."

        return "\n".join(sections)
