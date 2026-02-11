"""Metacognition — composite confidence scoring.

Multi-source confidence signal. LLMs are overconfident in verbalized FOK,
so we use structural heuristics as objective signal.

All metacognitive checks run in ISOLATED META-CONTEXT — separate API calls,
no history, no identity, discarded after signal extraction.

Composite: C = w1*objective + w2*FOK + w3*verbalized (0.5/0.35/0.15)
"""

import re
import logging

logger = logging.getLogger("agent.metacognition")

# Hedging phrases that indicate uncertainty
HEDGING_PHRASES = [
    r"\bi think\b", r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b",
    r"\bprobably\b", r"\bmight\b", r"\bcould be\b", r"\bnot sure\b",
    r"\buncertain\b", r"\bguess\b", r"\bseem(?:s)?\b", r"\bappear(?:s)?\b",
    r"\broughly\b", r"\bapproximately\b", r"\baround\b",
    r"\bif i recall\b", r"\bif i remember\b",
]

# Self-correction phrases
CORRECTION_PHRASES = [
    r"\bactually\b", r"\bwait\b", r"\bcorrection\b", r"\bi was wrong\b",
    r"\blet me reconsider\b", r"\bon second thought\b", r"\brather\b",
    r"\binstead\b", r"\bmore accurately\b",
]

# Confident phrases (inverse signal)
CONFIDENT_PHRASES = [
    r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b",
    r"\bwithout doubt\b", r"\bclearly\b", r"\bobviously\b",
    r"\bof course\b", r"\bundoubtedly\b",
]


def detect_hedging(text: str) -> float:
    """Detect hedging language. Returns 0-1 (higher = more hedging)."""
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)
    matches = sum(1 for p in HEDGING_PHRASES if re.search(p, text_lower))
    return min(1.0, matches / max(word_count * 0.05, 1))


def detect_self_correction(text: str) -> float:
    """Detect self-correction patterns. Returns 0-1."""
    text_lower = text.lower()
    matches = sum(1 for p in CORRECTION_PHRASES if re.search(p, text_lower))
    return min(1.0, matches * 0.25)


def detect_confidence_language(text: str) -> float:
    """Detect explicit confidence language. Returns 0-1."""
    text_lower = text.lower()
    matches = sum(1 for p in CONFIDENT_PHRASES if re.search(p, text_lower))
    return min(1.0, matches * 0.2)


def detect_question_rate(text: str) -> float:
    """Ratio of questions in response. High question rate = low confidence."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    questions = sum(1 for s in sentences if s.rstrip().endswith('?') or s.strip().startswith(('Is ', 'Are ', 'Do ', 'Does ', 'Can ', 'Could ', 'Would ', 'Should ')))
    return questions / len(sentences)


def detect_length_anomaly(text: str, expected_min: int = 20, expected_max: int = 500) -> float:
    """Detect abnormally short or long responses. Returns 0-1 anomaly score."""
    word_count = len(text.split())
    if word_count < expected_min:
        return min(1.0, (expected_min - word_count) / expected_min)
    if word_count > expected_max:
        return min(1.0, (word_count - expected_max) / expected_max)
    return 0.0


def structural_confidence(text: str) -> float:
    """Compute objective confidence from structural heuristics. Returns 0-1."""
    hedging = detect_hedging(text)
    correction = detect_self_correction(text)
    confidence_lang = detect_confidence_language(text)
    question_rate = detect_question_rate(text)
    length_anomaly = detect_length_anomaly(text)

    # Higher hedging/correction/questions = lower confidence
    # Higher confidence language = higher confidence
    raw = (
        1.0
        - 0.3 * hedging
        - 0.25 * correction
        - 0.15 * question_rate
        - 0.1 * length_anomaly
        + 0.2 * confidence_lang
    )

    return max(0.0, min(1.0, raw))


def composite_confidence(
    response_text: str,
    fok_score: float | None = None,
    verbalized_confidence: float | None = None,
) -> float:
    """Compute composite confidence score.

    C = w1*objective + w2*FOK + w3*verbalized
    Weights: 0.5 / 0.35 / 0.15

    Args:
        response_text: The LLM's response text (for structural analysis).
        fok_score: Feeling-of-knowing from logprobs if available (0-1).
        verbalized_confidence: LLM's self-reported confidence (0-1).
    """
    objective = structural_confidence(response_text)

    # Default FOK to objective if not available (logprobs not supported)
    fok = fok_score if fok_score is not None else objective
    verbalized = verbalized_confidence if verbalized_confidence is not None else objective

    composite = 0.5 * objective + 0.35 * fok + 0.15 * verbalized
    return max(0.0, min(1.0, composite))
