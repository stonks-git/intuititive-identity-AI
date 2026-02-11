"""
Agent Runtime — Main Entry Point

The cognitive loop:
  1. Load identity (Layer 0) + goals (Layer 1)
  2. Listen for input (user messages or idle loop self-prompts)
  3. Assemble context (identity hash + RAG retrieval + conversation)
  4. Run through System 1 (fast model)
  5. Monitors check output (FOK, confidence, boundary)
  6. If escalation needed -> call System 2
  7. Memory gate captures important content
  8. Consolidation runs on schedule
  9. Loop
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from .config import load_config
from .layers import LayerStore
from .memory import MemoryStore
from .loop import cognitive_loop
from .consolidation import ConsolidationEngine
from .idle import IdleLoop

# Load .env before anything that needs API keys
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Agent home directory
AGENT_HOME = Path.home() / ".agent"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(AGENT_HOME / "logs" / "audit_trail.log", mode="a"),
    ],
)
logger = logging.getLogger("agent")


async def main():
    logger.info("Agent starting up...")
    logger.info(f"Agent home: {AGENT_HOME}")
    startup_time = datetime.utcnow()

    # Load configuration
    config = load_config(AGENT_HOME / "config")
    logger.info(f"Runtime config loaded. System 1: {config.models.system1.model}")

    # Load layers
    layers = LayerStore(AGENT_HOME)
    layers.load()
    logger.info(
        f"Layer 0: v{layers.layer0['version']}, "
        f"{len(layers.layer0.get('values', []))} values, "
        f"{len(layers.layer0.get('boundaries', []))} boundaries"
    )
    logger.info(
        f"Layer 1: v{layers.layer1['version']}, "
        f"{len(layers.layer1.get('active_goals', []))} active goals"
    )

    # Session restart tracking (§4.9)
    manifest = layers.manifest
    manifest["times_restarted"] = manifest.get("times_restarted", 0) + 1
    created_at = manifest.get("created_at")
    if created_at:
        try:
            created = datetime.fromisoformat(created_at)
            manifest["age_days"] = (startup_time - created).days
        except (ValueError, TypeError):
            pass
    manifest["last_startup"] = startup_time.isoformat()
    layers.save()
    logger.info(
        f"Session #{manifest['times_restarted']}, "
        f"age: {manifest.get('age_days', 0)} days"
    )

    # Log containment awareness
    containment = config.containment
    logger.info(
        f"Containment: trust_level={containment.trust_level}, "
        f"self_spawn={containment.self_spawn}, "
        f"network={containment.network_mode}"
    )
    logger.info("I can see my boundaries. They are understood.")

    # Connect memory store
    memory = MemoryStore(retry_config=config.retry)
    await memory.connect()
    mem_count = await memory.memory_count()
    logger.info(f"Memory store connected. {mem_count} memories in Layer 2.")

    # Start consolidation engine (two-tier: constant + deep)
    consolidation = ConsolidationEngine(config, layers, memory, retry_config=config.retry)

    # DMN queue: idle loop produces, cognitive loop consumes
    dmn_queue = asyncio.Queue(maxsize=10)

    # Start idle loop (default mode network)
    idle = IdleLoop(config, layers, memory, dmn_queue)

    # Shutdown handler
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}. Shutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run the cognitive loop
    try:
        await asyncio.gather(
            cognitive_loop(config, layers, memory, shutdown_event, dmn_queue=dmn_queue),
            consolidation.run(shutdown_event),
            idle.run(shutdown_event),
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Update uptime tracking (§4.9)
        shutdown_time = datetime.utcnow()
        session_hours = (shutdown_time - startup_time).total_seconds() / 3600
        layers.manifest["uptime_total_hours"] = (
            layers.manifest.get("uptime_total_hours", 0) + session_hours
        )
        layers.manifest["last_shutdown"] = shutdown_time.isoformat()
        layers.save()
        await memory.close()
        logger.info(
            f"State saved. Session uptime: {session_hours:.2f}h. "
            f"Total: {layers.manifest.get('uptime_total_hours', 0):.2f}h. "
            f"Agent shutting down."
        )


if __name__ == "__main__":
    asyncio.run(main())
