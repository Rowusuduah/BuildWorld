"""
context-trim CLI.

Commands:
    estimate   — estimate token count for messages in a JSON file
    trim       — trim messages to fit a token budget
    ci         — CI gate: exit code 1 if over budget, 0 if OK
    history    — show trim history from a SQLite store
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_messages(path: str) -> list[dict]:
    """Load messages from a JSON file (list of {role, content} dicts)."""
    p = Path(path)
    if not p.exists():
        print(f"[context-trim] ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[context-trim] ERROR: invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"[context-trim] ERROR: expected a JSON array of messages", file=sys.stderr)
        sys.exit(1)
    return data


def cmd_estimate(args: argparse.Namespace) -> None:
    from .core import TokenBudget

    messages = _load_messages(args.input)
    budget = TokenBudget(max_tokens=args.max_tokens, reserved_tokens=args.reserved)
    total = budget.estimate_messages(messages)
    fits = budget.fits(messages)
    status = "OK" if fits else "OVER_BUDGET"
    over = budget.tokens_over(messages)

    print(f"messages:  {len(messages)}")
    print(f"tokens:    ~{total}")
    print(f"budget:    {budget.available_tokens} available ({budget.max_tokens} max - {budget.reserved_tokens} reserved)")
    print(f"status:    {status}" + (f" (+{over} over)" if over else ""))


def cmd_trim(args: argparse.Namespace) -> None:
    from .core import ContextTrim, TrimStrategy

    messages = _load_messages(args.input)

    try:
        strategy = TrimStrategy(args.strategy)
    except ValueError:
        valid = [s.value for s in TrimStrategy]
        print(f"[context-trim] ERROR: invalid strategy '{args.strategy}'. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    ct = ContextTrim(max_tokens=args.max_tokens, reserved_tokens=args.reserved)
    result = ct.trim(messages, strategy=strategy)

    print(result.summary())

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(result.messages, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"trimmed messages written to: {args.output}")
    else:
        print(json.dumps(result.messages, indent=2, ensure_ascii=False))


def cmd_ci(args: argparse.Namespace) -> None:
    from .core import ContextTrim

    messages = _load_messages(args.input)
    ct = ContextTrim(max_tokens=args.max_tokens, reserved_tokens=args.reserved)

    if ct.fits(messages):
        tokens = ct.estimate(messages)
        print(f"[context-trim] CI gate PASSED — ~{tokens} tokens (budget: {ct.budget.available_tokens})")
        sys.exit(0)
    else:
        over = ct.tokens_over(messages)
        tokens = ct.estimate(messages)
        print(
            f"[context-trim] CI gate FAILED — ~{tokens} tokens exceed budget "
            f"by {over} (budget: {ct.budget.available_tokens})",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_history(args: argparse.Namespace) -> None:
    from .store import TrimStore

    store = TrimStore(db_path=args.db)
    if args.pipeline:
        records = store.history(pipeline_id=args.pipeline, limit=args.limit)
    else:
        records = store.all_history(limit=args.limit)

    if not records:
        print("No history found.")
        return

    for r in records:
        import datetime
        ts = datetime.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        status = "OK" if r["within_budget"] else "OVER"
        print(
            f"{ts} | {r['pipeline_id']:20s} | {r['strategy']:18s} | "
            f"{r['original_count']:3d}→{r['final_count']:3d} msgs | "
            f"{r['original_tokens']:5d}→{r['final_tokens']:5d} tok | "
            f"{status}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-trim",
        description="Trim LLM conversation history to fit a token budget.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- estimate ---
    p_est = sub.add_parser("estimate", help="Estimate token count for messages")
    p_est.add_argument("input", help="Path to JSON file with messages array")
    p_est.add_argument("--max-tokens", type=int, default=8192, help="Context window size (default: 8192)")
    p_est.add_argument("--reserved", type=int, default=512, help="Reserved tokens (default: 512)")

    # --- trim ---
    p_trim = sub.add_parser("trim", help="Trim messages to fit token budget")
    p_trim.add_argument("input", help="Path to JSON file with messages array")
    p_trim.add_argument("--max-tokens", type=int, default=8192)
    p_trim.add_argument("--reserved", type=int, default=512)
    p_trim.add_argument(
        "--strategy",
        default="hybrid",
        choices=[s.value for s in __import__("context_trim.core", fromlist=["TrimStrategy"]).TrimStrategy],
        help="Trimming strategy (default: hybrid)",
    )
    p_trim.add_argument("--output", "-o", default=None, help="Write trimmed messages to this file")

    # --- ci ---
    p_ci = sub.add_parser("ci", help="CI gate: exit 1 if over budget")
    p_ci.add_argument("input", help="Path to JSON file with messages array")
    p_ci.add_argument("--max-tokens", type=int, default=8192)
    p_ci.add_argument("--reserved", type=int, default=512)

    # --- history ---
    p_hist = sub.add_parser("history", help="Show trim history from SQLite store")
    p_hist.add_argument("--db", default="context_trim_history.db", help="Path to SQLite DB")
    p_hist.add_argument("--pipeline", default=None, help="Filter by pipeline ID")
    p_hist.add_argument("--limit", type=int, default=20, help="Max records to show")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "trim":
        cmd_trim(args)
    elif args.command == "ci":
        cmd_ci(args)
    elif args.command == "history":
        cmd_history(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
