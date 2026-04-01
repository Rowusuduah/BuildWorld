"""
Core context trimming engine.

Design principles:
- Zero hard dependencies (stdlib only)
- Token estimation via 4 chars/token heuristic (no tiktoken required)
- Strategies are deterministic and reproducible
- System messages are always preserved
- TrimResult is always complete and inspectable
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4  # conservative approximation, safe for all providers


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character count. No tokeniser required."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a single message dict."""
    content = message.get("content") or ""
    role = message.get("role") or ""
    # 4 overhead tokens per message (role + framing)
    return _estimate_tokens(str(content)) + _estimate_tokens(str(role)) + 4


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


@dataclass
class TokenBudget:
    """Manages token limits for a single LLM call.

    Args:
        max_tokens: Total context window size (e.g. 8192 for GPT-4 Turbo).
        reserved_tokens: Tokens to hold back for system prompt + expected response.
            Defaults to 512.
    """

    max_tokens: int
    reserved_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.reserved_tokens < 0:
            raise ValueError(f"reserved_tokens must be >= 0, got {self.reserved_tokens}")
        if self.reserved_tokens >= self.max_tokens:
            raise ValueError(
                f"reserved_tokens ({self.reserved_tokens}) must be < max_tokens ({self.max_tokens})"
            )

    @property
    def available_tokens(self) -> int:
        """Tokens available for conversation history."""
        return self.max_tokens - self.reserved_tokens

    def estimate(self, text: str) -> int:
        """Estimate token count for a text string."""
        return _estimate_tokens(text)

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total token count for a list of messages."""
        return sum(_message_tokens(m) for m in messages)

    def fits(self, messages: list[dict[str, Any]]) -> bool:
        """Return True if messages fit within the available token budget."""
        return self.estimate_messages(messages) <= self.available_tokens

    def tokens_over(self, messages: list[dict[str, Any]]) -> int:
        """Return how many tokens over budget the messages are (0 if fits)."""
        return max(0, self.estimate_messages(messages) - self.available_tokens)


# ---------------------------------------------------------------------------
# TrimStrategy
# ---------------------------------------------------------------------------


class TrimStrategy(Enum):
    """Strategy for choosing which messages to drop or compress.

    RECENCY_FIRST   — Drop oldest messages first. Best for short-horizon chats.
    IMPORTANCE      — Score each message; drop lowest-scored first.
    SLIDING_WINDOW  — Keep the last N messages that fit the budget.
    SUMMARY_POINTS  — Replace dropped messages with a bullet-point summary line.
    HYBRID          — Importance-ranked with recency bonus. Best general strategy.
    """

    RECENCY_FIRST = "recency_first"
    IMPORTANCE = "importance"
    SLIDING_WINDOW = "sliding_window"
    SUMMARY_POINTS = "summary_points"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# ImportanceScorer
# ---------------------------------------------------------------------------


class ImportanceScorer:
    """Scores messages by estimated importance. Higher = more important.

    Scoring dimensions (all normalised to [0, 1]):
    - role_weight:    system=1.0, user=0.7, assistant=0.5, tool=0.6
    - recency:        linear decay from newest (1.0) to oldest (0.1)
    - length:         longer messages are slightly more important (capped at 1.0)
    - question:       messages with '?' score slightly higher
    - keyword_boost:  messages with key phrases ('error', 'important', 'must') score higher
    """

    _ROLE_WEIGHTS: dict[str, float] = {
        "system": 1.0,
        "user": 0.7,
        "assistant": 0.5,
        "tool": 0.6,
        "function": 0.6,
    }

    _BOOST_PATTERNS = re.compile(
        r"\b(error|fail|important|critical|must|never|always|warning|exception)\b",
        re.IGNORECASE,
    )

    def score(self, message: dict[str, Any], index: int, total: int) -> float:
        """Return a composite importance score in [0, 1].

        Args:
            message: A message dict with 'role' and 'content' keys.
            index: 0-based position in the original message list.
            total: Total number of messages.
        """
        role = str(message.get("role", "user")).lower()
        content = str(message.get("content") or "")

        role_weight = self._ROLE_WEIGHTS.get(role, 0.5)

        # System messages always max score
        if role == "system":
            return 1.0

        # Recency: newest = 1.0, oldest = 0.1
        if total > 1:
            recency = 0.1 + 0.9 * (index / (total - 1))
        else:
            recency = 1.0

        # Length normalised to [0, 1] capped at 2000 chars
        length_score = min(len(content) / 2000.0, 1.0)

        # Question heuristic
        question_bonus = 0.1 if "?" in content else 0.0

        # Keyword boost
        boost = 0.15 if self._BOOST_PATTERNS.search(content) else 0.0

        score = (
            role_weight * 0.35
            + recency * 0.35
            + length_score * 0.15
            + question_bonus * 0.075
            + boost * 0.075
        )
        return min(1.0, score)

    def score_all(self, messages: list[dict[str, Any]]) -> list[float]:
        """Return importance scores for every message."""
        total = len(messages)
        return [self.score(m, i, total) for i, m in enumerate(messages)]


# ---------------------------------------------------------------------------
# TrimResult
# ---------------------------------------------------------------------------


@dataclass
class TrimResult:
    """Result of a trimming operation.

    Attributes:
        messages:         The trimmed message list, ready to send to an LLM.
        original_count:   Number of messages before trimming.
        final_count:      Number of messages after trimming.
        original_tokens:  Estimated token count before trimming.
        final_tokens:     Estimated token count after trimming.
        strategy:         Strategy that was applied.
        dropped_count:    Number of messages removed.
        trim_ratio:       Fraction of original tokens removed (0 if no trimming).
        budget:           The TokenBudget used.
        within_budget:    True if final_tokens <= budget.available_tokens.
    """

    messages: list[dict[str, Any]]
    original_count: int
    final_count: int
    original_tokens: int
    final_tokens: int
    strategy: TrimStrategy
    dropped_count: int
    trim_ratio: float
    budget: TokenBudget
    within_budget: bool

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        status = "OK" if self.within_budget else "OVER_BUDGET"
        return (
            f"[context-trim] {status} | strategy={self.strategy.value} "
            f"| messages {self.original_count}->{self.final_count} "
            f"| tokens {self.original_tokens}->{self.final_tokens} "
            f"| dropped={self.dropped_count} | ratio={self.trim_ratio:.1%}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dict representation."""
        return {
            "within_budget": self.within_budget,
            "strategy": self.strategy.value,
            "original_count": self.original_count,
            "final_count": self.final_count,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "dropped_count": self.dropped_count,
            "trim_ratio": round(self.trim_ratio, 4),
            "budget_max_tokens": self.budget.max_tokens,
            "budget_available": self.budget.available_tokens,
        }


# ---------------------------------------------------------------------------
# MessageTrimmer
# ---------------------------------------------------------------------------


class MessageTrimmer:
    """Trims a conversation history (list of message dicts) to fit a token budget."""

    def __init__(self) -> None:
        self._scorer = ImportanceScorer()

    def trim(
        self,
        messages: list[dict[str, Any]],
        budget: TokenBudget,
        strategy: TrimStrategy = TrimStrategy.RECENCY_FIRST,
    ) -> TrimResult:
        """Trim *messages* to fit within *budget* using *strategy*.

        System messages are always preserved regardless of strategy.

        Args:
            messages: List of dicts with at least 'role' and 'content' keys.
            budget:   Token budget to trim into.
            strategy: Trimming strategy to apply.

        Returns:
            A TrimResult with the trimmed message list and metadata.
        """
        if not messages:
            return self._build_result([], [], budget, strategy)

        original = list(messages)
        original_tokens = budget.estimate_messages(original)

        if budget.fits(original):
            return self._build_result(original, original, budget, strategy)

        if strategy == TrimStrategy.RECENCY_FIRST:
            trimmed = self._recency_first(original, budget)
        elif strategy == TrimStrategy.IMPORTANCE:
            trimmed = self._importance(original, budget)
        elif strategy == TrimStrategy.SLIDING_WINDOW:
            trimmed = self._sliding_window(original, budget)
        elif strategy == TrimStrategy.SUMMARY_POINTS:
            trimmed = self._summary_points(original, budget)
        elif strategy == TrimStrategy.HYBRID:
            trimmed = self._hybrid(original, budget)
        else:
            trimmed = self._recency_first(original, budget)

        return self._build_result(original, trimmed, budget, strategy)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _recency_first(
        self, messages: list[dict[str, Any]], budget: TokenBudget
    ) -> list[dict[str, Any]]:
        """Drop oldest non-system messages first."""
        system = [m for m in messages if str(m.get("role", "")).lower() == "system"]
        non_system = [m for m in messages if str(m.get("role", "")).lower() != "system"]

        # Work from oldest to newest, dropping until we fit
        result = list(non_system)
        while result and not budget.fits(system + result):
            result.pop(0)

        return system + result

    def _importance(
        self, messages: list[dict[str, Any]], budget: TokenBudget
    ) -> list[dict[str, Any]]:
        """Drop lowest-scored messages first (system always kept)."""
        system = [m for m in messages if str(m.get("role", "")).lower() == "system"]
        non_system = [m for m in messages if str(m.get("role", "")).lower() != "system"]

        # Score relative to the full list for recency signal
        scores = self._scorer.score_all(messages)
        non_sys_scores = [
            (i, scores[orig_i], m)
            for i, (orig_i, m) in enumerate(
                (orig_i, m)
                for orig_i, m in enumerate(messages)
                if str(m.get("role", "")).lower() != "system"
            )
        ]

        # Sort by score ascending — drop lowest first
        scored_result = sorted(non_sys_scores, key=lambda x: x[1])

        kept_indices: set[int] = set(range(len(non_system)))
        for idx, _score, _msg in scored_result:
            if budget.fits(system + [non_system[j] for j in sorted(kept_indices)]):
                break
            kept_indices.discard(idx)

        ordered = [non_system[j] for j in sorted(kept_indices)]
        return system + ordered

    def _sliding_window(
        self, messages: list[dict[str, Any]], budget: TokenBudget
    ) -> list[dict[str, Any]]:
        """Keep the last N messages that fit the budget (system always first)."""
        system = [m for m in messages if str(m.get("role", "")).lower() == "system"]
        non_system = [m for m in messages if str(m.get("role", "")).lower() != "system"]

        # Expand window from the tail until we exceed budget
        window: list[dict[str, Any]] = []
        for msg in reversed(non_system):
            candidate = [msg] + window
            if budget.fits(system + candidate):
                window = candidate
            else:
                break

        return system + window

    def _summary_points(
        self, messages: list[dict[str, Any]], budget: TokenBudget
    ) -> list[dict[str, Any]]:
        """Replace oldest messages with a compact summary bullet list."""
        system = [m for m in messages if str(m.get("role", "")).lower() == "system"]
        non_system = [m for m in messages if str(m.get("role", "")).lower() != "system"]

        if not non_system:
            return system

        # Try keeping more and more recent messages, summarising the rest
        for keep_count in range(len(non_system), -1, -1):
            recent = non_system[len(non_system) - keep_count :] if keep_count else []
            dropped = non_system[: len(non_system) - keep_count]

            if dropped:
                summary_lines = self._build_summary(dropped)
                summary_msg = {
                    "role": "system",
                    "content": f"[Context summary — {len(dropped)} earlier messages]\n{summary_lines}",
                }
                candidate = system + [summary_msg] + recent
            else:
                candidate = system + recent

            if budget.fits(candidate):
                return candidate

        # Fallback: truncated summary that is guaranteed to fit the budget
        return self._forced_summary(system, non_system, budget)

    def _forced_summary(
        self,
        system: list[dict[str, Any]],
        non_system: list[dict[str, Any]],
        budget: TokenBudget,
    ) -> list[dict[str, Any]]:
        """Build the shortest summary that fits, by capping bullet count and length."""
        system_tokens = budget.estimate_messages(system)
        available_for_summary = budget.available_tokens - system_tokens - 8  # 8 overhead

        if available_for_summary <= 0:
            return system

        # Try progressively fewer and shorter bullets until it fits
        max_snippet = 60
        for max_bullets in range(min(len(non_system), 20), 0, -1):
            sample = non_system[-max_bullets:]  # most recent dropped msgs
            lines: list[str] = []
            for msg in sample:
                role = str(msg.get("role", "user")).capitalize()
                content = str(msg.get("content") or "").strip()
                snippet = content[:max_snippet]
                lines.append(f"• {role}: {snippet}")
            header = f"[Summary: {len(non_system)} msgs]\n"
            body = "\n".join(lines)
            summary_text = header + body
            if _estimate_tokens(summary_text) <= available_for_summary:
                return system + [{"role": "system", "content": summary_text}]

        # Absolute fallback: one-liner summary
        summary_text = f"[Summary: {len(non_system)} earlier messages omitted to fit token budget]"
        return system + [{"role": "system", "content": summary_text}]

    def _hybrid(
        self, messages: list[dict[str, Any]], budget: TokenBudget
    ) -> list[dict[str, Any]]:
        """Importance-ranked with recency bonus. Preserves system messages."""
        system = [m for m in messages if str(m.get("role", "")).lower() == "system"]
        non_system = [m for m in messages if str(m.get("role", "")).lower() != "system"]

        if not non_system:
            return system

        scores = self._scorer.score_all(messages)
        non_sys_with_scores: list[tuple[float, int, dict[str, Any]]] = []
        for orig_i, m in enumerate(messages):
            if str(m.get("role", "")).lower() != "system":
                non_sys_with_scores.append((scores[orig_i], orig_i, m))

        # Sort by score descending, keep highest
        non_sys_with_scores.sort(key=lambda x: (-x[0], -x[1]))

        kept: list[tuple[int, dict[str, Any]]] = []
        for score, orig_i, msg in non_sys_with_scores:
            candidate_msgs = [m for _, m in sorted(kept + [(orig_i, msg)], key=lambda x: x[0])]
            if budget.fits(system + candidate_msgs):
                kept.append((orig_i, msg))

        ordered = [m for _, m in sorted(kept, key=lambda x: x[0])]
        return system + ordered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(messages: list[dict[str, Any]]) -> str:
        """Build a compact bullet-point summary from dropped messages."""
        lines: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).capitalize()
            content = str(msg.get("content") or "").strip()
            # Truncate long messages to first sentence or 120 chars
            first_sentence = re.split(r"(?<=[.!?])\s", content)[0] if content else ""
            snippet = first_sentence[:120] if first_sentence else content[:120]
            if snippet:
                lines.append(f"• {role}: {snippet}")
        return "\n".join(lines) if lines else "• (no content)"

    @staticmethod
    def _build_result(
        original: list[dict[str, Any]],
        trimmed: list[dict[str, Any]],
        budget: TokenBudget,
        strategy: TrimStrategy,
    ) -> TrimResult:
        orig_tokens = budget.estimate_messages(original)
        final_tokens = budget.estimate_messages(trimmed)
        dropped = len(original) - len(trimmed)
        ratio = (orig_tokens - final_tokens) / orig_tokens if orig_tokens > 0 else 0.0
        return TrimResult(
            messages=trimmed,
            original_count=len(original),
            final_count=len(trimmed),
            original_tokens=orig_tokens,
            final_tokens=final_tokens,
            strategy=strategy,
            dropped_count=dropped,
            trim_ratio=ratio,
            budget=budget,
            within_budget=budget.fits(trimmed),
        )


# ---------------------------------------------------------------------------
# DocumentTrimmer
# ---------------------------------------------------------------------------


class DocumentTrimmer:
    """Trims a long text document to fit within a token budget.

    Strategies:
    - RECENCY_FIRST   → keep the end of the document
    - IMPORTANCE      → keep paragraphs with the highest keyword density
    - SLIDING_WINDOW  → keep the last N paragraphs that fit
    - SUMMARY_POINTS  → truncate and append an ellipsis marker
    - HYBRID          → keyword + recency scoring per paragraph
    """

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")
    _PARA_RE = re.compile(r"\n{2,}")
    _KW_RE = re.compile(
        r"\b(result|conclusion|therefore|however|important|key|critical|summary|finally|in conclusion)\b",
        re.IGNORECASE,
    )

    def trim(
        self,
        text: str,
        budget: TokenBudget,
        strategy: TrimStrategy = TrimStrategy.RECENCY_FIRST,
    ) -> "DocumentTrimResult":
        """Trim *text* to fit within *budget*.

        Returns a DocumentTrimResult (has .text and .metadata).
        """
        original_tokens = budget.estimate(text)

        if budget.estimate(text) <= budget.available_tokens:
            return DocumentTrimResult(
                text=text,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                strategy=strategy,
                truncated=False,
                budget=budget,
            )

        paragraphs = [p.strip() for p in self._PARA_RE.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        if strategy in (TrimStrategy.RECENCY_FIRST, TrimStrategy.SLIDING_WINDOW):
            result_text = self._keep_tail(paragraphs, budget)
        elif strategy == TrimStrategy.IMPORTANCE:
            result_text = self._keep_important(paragraphs, budget)
        elif strategy == TrimStrategy.SUMMARY_POINTS:
            result_text = self._truncate_with_marker(text, budget)
        elif strategy == TrimStrategy.HYBRID:
            result_text = self._hybrid(paragraphs, budget)
        else:
            result_text = self._keep_tail(paragraphs, budget)

        final_tokens = budget.estimate(result_text)
        return DocumentTrimResult(
            text=result_text,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            strategy=strategy,
            truncated=True,
            budget=budget,
        )

    def _keep_tail(self, paragraphs: list[str], budget: TokenBudget) -> str:
        kept: list[str] = []
        for para in reversed(paragraphs):
            candidate = "\n\n".join([para] + kept)
            if budget.estimate(candidate) <= budget.available_tokens:
                kept = [para] + kept
            else:
                break
        return "\n\n".join(kept) if kept else paragraphs[-1][: budget.available_tokens * _CHARS_PER_TOKEN]

    def _keep_important(self, paragraphs: list[str], budget: TokenBudget) -> str:
        scored = [
            (len(self._KW_RE.findall(p)) / max(len(p.split()), 1), i, p)
            for i, p in enumerate(paragraphs)
        ]
        scored.sort(key=lambda x: -x[0])

        kept_indices: set[int] = set()
        for _s, idx, p in scored:
            candidate_indices = sorted(kept_indices | {idx})
            candidate = "\n\n".join(paragraphs[i] for i in candidate_indices)
            if budget.estimate(candidate) <= budget.available_tokens:
                kept_indices.add(idx)

        if not kept_indices:
            return paragraphs[-1][: budget.available_tokens * _CHARS_PER_TOKEN]
        return "\n\n".join(paragraphs[i] for i in sorted(kept_indices))

    def _truncate_with_marker(self, text: str, budget: TokenBudget) -> str:
        marker = "\n\n[... content trimmed to fit token budget ...]"
        marker_tokens = budget.estimate(marker)
        char_limit = (budget.available_tokens - marker_tokens) * _CHARS_PER_TOKEN
        return text[:char_limit] + marker

    def _hybrid(self, paragraphs: list[str], budget: TokenBudget) -> str:
        total = len(paragraphs)
        scored = []
        for i, p in enumerate(paragraphs):
            recency = (i + 1) / total
            kw_density = len(self._KW_RE.findall(p)) / max(len(p.split()), 1)
            score = recency * 0.5 + min(kw_density * 5, 1.0) * 0.5
            scored.append((score, i, p))
        scored.sort(key=lambda x: -x[0])

        kept_indices: set[int] = set()
        for _s, idx, _p in scored:
            candidate_indices = sorted(kept_indices | {idx})
            candidate = "\n\n".join(paragraphs[i] for i in candidate_indices)
            if budget.estimate(candidate) <= budget.available_tokens:
                kept_indices.add(idx)

        if not kept_indices:
            return paragraphs[-1][: budget.available_tokens * _CHARS_PER_TOKEN]
        return "\n\n".join(paragraphs[i] for i in sorted(kept_indices))


@dataclass
class DocumentTrimResult:
    """Result of a document trimming operation."""

    text: str
    original_tokens: int
    final_tokens: int
    strategy: TrimStrategy
    truncated: bool
    budget: TokenBudget

    def summary(self) -> str:
        status = "TRIMMED" if self.truncated else "OK"
        return (
            f"[context-trim] {status} | strategy={self.strategy.value} "
            f"| tokens {self.original_tokens}->{self.final_tokens}"
        )


# ---------------------------------------------------------------------------
# ContextTrim  —  Main public API
# ---------------------------------------------------------------------------


class ContextTrim:
    """High-level API for context trimming.

    Example::

        ct = ContextTrim(max_tokens=8192, reserved_tokens=512)

        if not ct.fits(messages):
            result = ct.trim(messages, strategy=TrimStrategy.HYBRID)
            messages = result.messages

        # Or use as a CI gate — raises RuntimeError if over budget
        ct.ci_gate(messages)
    """

    def __init__(
        self,
        max_tokens: int,
        reserved_tokens: int = 512,
        db_path: str | None = None,
    ) -> None:
        """Create a ContextTrim instance.

        Args:
            max_tokens:      Total context window size in tokens.
            reserved_tokens: Tokens to reserve for system prompt + response.
            db_path:         Optional path to SQLite history database.
        """
        self._budget = TokenBudget(max_tokens=max_tokens, reserved_tokens=reserved_tokens)
        self._trimmer = MessageTrimmer()
        self._doc_trimmer = DocumentTrimmer()
        self._store: "TrimStore | None" = None
        if db_path is not None:
            from .store import TrimStore  # lazy import
            self._store = TrimStore(db_path)

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        """Return estimated token count for *messages*."""
        return self._budget.estimate_messages(messages)

    def fits(self, messages: list[dict[str, Any]]) -> bool:
        """Return True if *messages* fit within the available budget."""
        return self._budget.fits(messages)

    def tokens_over(self, messages: list[dict[str, Any]]) -> int:
        """Return how many tokens *messages* exceed the budget by (0 if OK)."""
        return self._budget.tokens_over(messages)

    def trim(
        self,
        messages: list[dict[str, Any]],
        strategy: TrimStrategy = TrimStrategy.HYBRID,
        pipeline_id: str = "default",
    ) -> TrimResult:
        """Trim *messages* to fit within the token budget.

        Args:
            messages:    Conversation history as list of {role, content} dicts.
            strategy:    Which trimming strategy to apply.
            pipeline_id: Label stored in the history database (if db_path was set).

        Returns:
            TrimResult with trimmed messages and metadata.
        """
        result = self._trimmer.trim(messages, self._budget, strategy)
        if self._store is not None:
            self._store.record(result, pipeline_id)
        return result

    def trim_document(
        self,
        text: str,
        strategy: TrimStrategy = TrimStrategy.HYBRID,
    ) -> "DocumentTrimResult":
        """Trim a long text document to fit within the token budget."""
        return self._doc_trimmer.trim(text, self._budget, strategy)

    def ci_gate(
        self,
        messages: list[dict[str, Any]],
        fail_message: str | None = None,
    ) -> None:
        """Raise RuntimeError if *messages* exceed the token budget.

        Designed for use in CI pipelines to block builds that would send
        oversized context to an LLM.

        Args:
            messages:     Conversation history to check.
            fail_message: Optional custom error message.

        Raises:
            RuntimeError: If messages exceed the available token budget.
        """
        if not self._budget.fits(messages):
            over = self._budget.tokens_over(messages)
            default = (
                f"[context-trim] CI gate FAILED — "
                f"messages exceed token budget by {over} tokens "
                f"(budget={self._budget.available_tokens}, "
                f"actual≈{self._budget.estimate_messages(messages)})"
            )
            raise RuntimeError(fail_message or default)
