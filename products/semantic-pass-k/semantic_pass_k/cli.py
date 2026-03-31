"""
semantic_pass_k.cli
-------------------
CLI entry point for semantic-pass-k.

Commands:
  sempass run     -- run a quick consistency check (stdin outputs)
  sempass report  -- display stored results
  sempass ci      -- CI gate: exit 1 if INCONSISTENT, exit 0 if CONSISTENT
  sempass budget  -- compare consistency scores across two agent labels
"""
from __future__ import annotations

import json
import sys
from typing import Optional

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

from .models import (
    CRITICALITY_THRESHOLDS,
    ConsistencyReport,
    CriticalityLevel,
)
from .engine import ConsistencyEngine
from .store import ConsistencyStore


def _no_click_error() -> None:
    print(
        "ERROR: 'click' is not installed. Install it with: pip install click",
        file=sys.stderr,
    )
    sys.exit(1)


if HAS_CLICK:
    @click.group()
    def cli() -> None:
        """semantic-pass-k — Semantic consistency testing for AI agents."""

    @cli.command("run")
    @click.argument("outputs", nargs=-1, required=False)
    @click.option("--prompt", "-p", default="<prompt>", help="Prompt text (for logging).")
    @click.option(
        "--criticality", "-c", default="HIGH",
        type=click.Choice(list(CRITICALITY_THRESHOLDS.keys()), case_sensitive=False),
        help="Criticality tier (CRITICAL/HIGH/MEDIUM/LOW).",
    )
    @click.option("--label", "-l", default="default", help="Agent label.")
    @click.option("--store-db", default=None, help="SQLite DB path for saving results.")
    @click.option("--json-output", is_flag=True, help="Output JSON instead of text.")
    def run_cmd(
        outputs: tuple,
        prompt: str,
        criticality: str,
        label: str,
        store_db: Optional[str],
        json_output: bool,
    ) -> None:
        """
        Evaluate semantic consistency across provided OUTPUTS.

        Pass outputs as positional arguments or pipe them via stdin (one per line).

        Examples:
            sempass run "Answer A." "Answer B." "Answer A." --criticality HIGH
            echo -e "Answer A\\nAnswer B\\nAnswer A" | sempass run --criticality HIGH
        """
        if not outputs:
            stdin_text = sys.stdin.read().strip()
            if not stdin_text:
                click.echo("ERROR: No outputs provided. Pass as arguments or via stdin.", err=True)
                sys.exit(1)
            outputs = tuple(line.strip() for line in stdin_text.splitlines() if line.strip())

        if len(outputs) < 2:
            click.echo("ERROR: At least 2 outputs required for consistency measurement.", err=True)
            sys.exit(1)

        engine = ConsistencyEngine(agent_label=label)
        result = engine.evaluate(prompt, list(outputs), criticality.upper())  # type: ignore[arg-type]

        if store_db:
            store = ConsistencyStore(store_db)
            store.save_result(result)

        if json_output:
            click.echo(json.dumps({
                "run_id": result.run_id,
                "consistency_score": result.consistency_score,
                "verdict": result.verdict,
                "criticality": result.criticality,
                "threshold": result.threshold,
                "k": result.k,
                "pairwise_scores": result.pairwise_scores,
            }, indent=2))
        else:
            _print_result(result)

        if result.verdict == "INCONSISTENT":
            sys.exit(1)

    @cli.command("report")
    @click.option("--store-db", default="consistency_history.db", help="SQLite DB path.")
    @click.option("--label", "-l", default=None, help="Filter by agent label.")
    @click.option("--limit", default=20, help="Max results to show.")
    def report_cmd(store_db: str, label: Optional[str], limit: int) -> None:
        """Display stored consistency results."""
        store = ConsistencyStore(store_db)
        results = (
            store.get_results_by_label(label) if label else store.list_results(limit)
        )
        if not results:
            click.echo("No results found.")
            return

        click.echo(f"\n{'RUN_ID'[:8]:8}  {'LABEL':12}  {'SCORE':6}  {'VERDICT':12}  TESTED_AT")
        click.echo("-" * 70)
        for r in results[:limit]:
            click.echo(
                f"{r.run_id[:8]:8}  {r.agent_label[:12]:12}  "
                f"{r.consistency_score:.3f}  {r.verdict:12}  "
                f"{r.tested_at.strftime('%Y-%m-%d %H:%M')}"
            )

    @cli.command("ci")
    @click.option("--store-db", default="consistency_history.db", help="SQLite DB path.")
    @click.option("--label", "-l", required=True, help="Agent label to check.")
    @click.option("--last", default=1, help="Number of most recent results to check.")
    def ci_cmd(store_db: str, label: str, last: int) -> None:
        """
        CI gate — exit 1 if any recent result is INCONSISTENT.

        Example (GitHub Actions):
            sempass ci --label my_agent --last 3
        """
        store = ConsistencyStore(store_db)
        results = store.get_results_by_label(label)[:last]

        if not results:
            click.echo(f"WARN: No results found for label '{label}'.", err=True)
            sys.exit(0)

        failed = [r for r in results if r.verdict == "INCONSISTENT"]
        if failed:
            click.echo(
                f"CI FAIL: {len(failed)}/{len(results)} results INCONSISTENT "
                f"for agent '{label}'.",
                err=True,
            )
            for r in failed:
                click.echo(f"  {r.run_id[:8]} score={r.consistency_score:.3f} "
                           f"threshold={r.threshold:.2f}", err=True)
            sys.exit(1)

        click.echo(
            f"CI PASS: {len(results)}/{len(results)} results CONSISTENT "
            f"for agent '{label}'."
        )

    @cli.command("budget")
    @click.option("--store-db", default="consistency_history.db", help="SQLite DB path.")
    @click.argument("label_a")
    @click.argument("label_b")
    def budget_cmd(store_db: str, label_a: str, label_b: str) -> None:
        """
        Compare consistency scores between two agent labels.

        Example:
            sempass budget gpt-4o claude-3-5-sonnet
        """
        store = ConsistencyStore(store_db)
        results_a = store.get_results_by_label(label_a)
        results_b = store.get_results_by_label(label_b)

        if not results_a:
            click.echo(f"No results for label '{label_a}'.", err=True)
            sys.exit(1)
        if not results_b:
            click.echo(f"No results for label '{label_b}'.", err=True)
            sys.exit(1)

        avg_a = sum(r.consistency_score for r in results_a) / len(results_a)
        avg_b = sum(r.consistency_score for r in results_b) / len(results_b)

        click.echo(f"\nConsistency Budget Comparison")
        click.echo(f"{'─' * 40}")
        click.echo(f"{label_a:20s}  avg={avg_a:.3f}  n={len(results_a)}")
        click.echo(f"{label_b:20s}  avg={avg_b:.3f}  n={len(results_b)}")
        click.echo(f"{'─' * 40}")
        winner = label_a if avg_a >= avg_b else label_b
        click.echo(f"More consistent: {winner} (+{abs(avg_a - avg_b):.3f})")


else:
    def cli() -> None:  # type: ignore[misc]
        _no_click_error()


def _print_result(result) -> None:  # type: ignore[no-untyped-def]
    """Pretty-print a ConsistencyResult."""
    verdict_color = {
        "CONSISTENT": "\033[92m",    # green
        "BORDERLINE": "\033[93m",    # yellow
        "INCONSISTENT": "\033[91m",  # red
    }
    reset = "\033[0m"
    color = verdict_color.get(result.verdict, "")

    print(f"\n{color}[{result.verdict}]{reset}  score={result.consistency_score:.3f}  "
          f"threshold={result.threshold:.2f}  criticality={result.criticality}  k={result.k}")
    print(f"  pairs: min={min(result.pairwise_scores):.3f}  "
          f"max={max(result.pairwise_scores):.3f}  "
          f"n={result.n_pairs}")
