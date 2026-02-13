"""Telegram Peripheral — external user input/output via Telegram Bot API.

Receives messages from a single authorized Telegram user, creates
AttentionCandidate objects, and pushes them into the unified input queue.
Responses are routed back to the Telegram chat via reply_fn callbacks
carried in candidate metadata.

Uses raw httpx calls to the Telegram Bot API (long polling).
No framework dependency — just two endpoints: getUpdates + sendMessage.

Optional: starts only if TELEGRAM_BOT_TOKEN is set.
Security: only TELEGRAM_OWNER_ID can interact. Long polling = no public endpoint.
"""

import asyncio
import logging
import os

import httpx
import numpy as np

from .attention import AttentionCandidate

logger = logging.getLogger("agent.telegram")

TELEGRAM_API = "https://api.telegram.org"
POLL_TIMEOUT = 30  # seconds for long-poll
MSG_MAX_LEN = 4096  # Telegram message length limit


class TelegramPeripheral:
    """Telegram bot peripheral for the cognitive loop."""

    def __init__(self, input_queue: asyncio.Queue, memory):
        self.input_queue = input_queue
        self.memory = memory
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.owner_id = int(os.environ.get("TELEGRAM_OWNER_ID", "0"))
        self._base_url = f"{TELEGRAM_API}/bot{self.token}"
        self._client: httpx.AsyncClient | None = None
        self._offset = 0  # getUpdates offset for acknowledging processed messages

    @property
    def is_configured(self) -> bool:
        return bool(self.token) and self.owner_id != 0

    async def run(self, shutdown_event: asyncio.Event):
        """Run the Telegram bot. Long-polls until shutdown."""
        if not self.is_configured:
            logger.info(
                "Telegram peripheral not configured "
                "(set TELEGRAM_BOT_TOKEN + TELEGRAM_OWNER_ID)"
            )
            return

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(POLL_TIMEOUT + 10))

        try:
            # Verify token by calling getMe
            me = await self._api("getMe")
            if me:
                logger.info(
                    f"Telegram peripheral started: @{me.get('username')} "
                    f"(owner_id={self.owner_id})"
                )
            else:
                logger.error("Telegram: getMe failed — check TELEGRAM_BOT_TOKEN")
                return

            # Drop pending updates from before this session
            drop = await self._api("getUpdates", {"offset": -1})
            if drop:
                last_id = max(u.get("update_id", 0) for u in drop)
                self._offset = last_id + 1
            else:
                self._offset = 0

            # Main polling loop
            while not shutdown_event.is_set():
                try:
                    await self._poll_once()
                except httpx.TimeoutException:
                    # Normal — long poll timed out with no updates
                    continue
                except httpx.HTTPError as e:
                    logger.warning(f"Telegram HTTP error: {e}")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Telegram poll error: {e}", exc_info=True)
                    await asyncio.sleep(5)

        finally:
            logger.info("Telegram peripheral shutting down...")
            if self._client:
                await self._client.aclose()

    async def _api(self, method: str, params: dict | None = None) -> dict | None:
        """Call a Telegram Bot API method."""
        url = f"{self._base_url}/{method}"
        resp = await self._client.post(url, json=params or {})
        data = resp.json()
        if data.get("ok"):
            return data.get("result")
        logger.warning(f"Telegram API error: {method} -> {data}")
        return None

    async def _send_message(self, chat_id: int, text: str):
        """Send a message back to a Telegram chat (handles chunking)."""
        for i in range(0, len(text), MSG_MAX_LEN):
            chunk = text[i:i + MSG_MAX_LEN]
            await self._api("sendMessage", {
                "chat_id": chat_id,
                "text": chunk,
            })

    async def _poll_once(self):
        """One long-poll cycle: fetch updates, process messages."""
        params = {
            "offset": self._offset,
            "timeout": POLL_TIMEOUT,
            "allowed_updates": ["message"],
        }
        result = await self._api("getUpdates", params)
        if not result:
            return

        for update in result:
            update_id = update.get("update_id", 0)
            self._offset = max(self._offset, update_id + 1)

            message = update.get("message")
            if not message:
                continue

            await self._handle_message(message)

    async def _handle_message(self, message: dict):
        """Process a single incoming Telegram message."""
        # Auth check
        from_user = message.get("from", {})
        user_id = from_user.get("id", 0)
        if user_id != self.owner_id:
            logger.warning(
                f"Unauthorized Telegram message from user_id={user_id} "
                f"({from_user.get('username', 'unknown')})"
            )
            return

        # Extract text
        text = message.get("text", "").strip()
        if not text:
            return

        chat_id = message["chat"]["id"]

        # Embed the message (skip for commands — they bypass attention)
        embedding = None
        if not text.startswith("/"):
            try:
                vec = await self.memory.embed(text, task_type="RETRIEVAL_QUERY")
                embedding = np.array(vec, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to embed Telegram input: {e}")

        # Create reply callback bound to this chat
        reply_fn = self._make_reply_fn(chat_id)

        candidate = AttentionCandidate(
            content=text,
            source_tag="external_user",
            embedding=embedding,
            metadata={
                "reply_fn": reply_fn,
                "peripheral": "telegram",
                "telegram_user_id": user_id,
                "telegram_chat_id": chat_id,
                "telegram_message_id": message.get("message_id"),
            },
        )

        try:
            self.input_queue.put_nowait(candidate)
            logger.debug(f"Telegram input queued: {text[:60]}")
        except asyncio.QueueFull:
            await reply_fn("[Queue full — try again in a moment]")

    def _make_reply_fn(self, chat_id: int):
        """Create a reply callback bound to a specific Telegram chat."""
        peripheral = self

        async def reply_fn(text: str):
            await peripheral._send_message(chat_id, text)

        return reply_fn
