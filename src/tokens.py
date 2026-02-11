"""Token Counting — approximate token counter for context budget enforcement.

Fast approximation: word_count * 1.3 (within 20% of tiktoken).
Good enough for budget enforcement. No external dependency needed.
"""


def count_tokens(text: str) -> int:
    """Approximate token count for a string."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def count_message_tokens(message: dict) -> int:
    """Approximate token count for a conversation message dict."""
    content = message.get("content", "")
    # Role/metadata overhead ~4 tokens per message
    return count_tokens(content) + 4


def count_messages_tokens(messages: list[dict]) -> int:
    """Total token count for a list of messages."""
    return sum(count_message_tokens(m) for m in messages)


def fits_budget(text: str, budget: int) -> bool:
    """Check if text fits within a token budget."""
    return count_tokens(text) <= budget


def remaining_budget(used: int, total: int = 131072) -> int:
    """Tokens remaining in context window."""
    return max(0, total - used)
