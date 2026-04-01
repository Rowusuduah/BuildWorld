"""
cot_fidelity.cli
----------------
CLI for cot-fidelity.

Commands:
  cot-fidelity test     Test faithfulness from JSON input file
  cot-fidelity report   Generate report from saved history
  cot-fidelity drift    Check faithfulness drift over time
  cot-fidelity clear    Clear history database
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import click
    _CLICK_AVAILABLE = True
except ImportError:
    _CLICK_AVAILABLE = False

from .engine import FidelityEngine
from .models import FidelityBatchReport
from .store import FidelityStore


def _require_click():
    if not _CLICK_AVAILABLE:
        print("ERROR: click is required for CLI use. Install with: pip install cot-fidelity[cli]")
        sys.exit(1)


if _CLICK_AVAILABLE:
    import click

    @click.group()
    @click.version_option(version="0.1.0", prog_name="cot-fidelity")
    def cli():
        """cot-fidelity: Measure whether your LLM's reasoning chain actually caused its output."""
        pass

    @cli.command("test")
    @click.option(
        "--input", "-i", "input_file", required=True, type=click.Path(exists=True),
        help="JSON file with prompt/cot_chain/with_cot_output/without_cot_output entries."
    )
    @click.option("--faithful-threshold", default=0.15, show_default=True, type=float)
    @click.option("--unfaithful-threshold", default=0.08, show_default=True, type=float)
    @click.option("--output-format", default="text", type=click.Choice(["text", "json", "markdown"]))
    @click.option("--db", default=None, help="Path to SQLite history database.")
    @click.option("--model-version", default="", help="Tag for history store entries.")
    @click.option("--min-faithfulness-rate", default=None, type=float,
                  help="If set, exit 1 if faithfulness rate < this value (CI gate).")
    def test_cmd(input_file, faithful_threshold, unfaithful_threshold,
                  output_format, db, model_version, min_faithfulness_rate):
        """Test CoT faithfulness from a JSON file."""
        data = json.loads(Path(input_file).read_text())
        if isinstance(data, dict):
            data = [data]

        engine = FidelityEngine(
            faithful_threshold=faithful_threshold,
            unfaithful_threshold=unfaithful_threshold,
        )
        store = FidelityStore(db_path=db) if db else None

        results = []
        for entry in data:
            r = engine.test(
                prompt=entry["prompt"],
                cot_chain=entry.get("cot_chain", ""),
                with_cot_output=entry["with_cot_output"],
                without_cot_output=entry["without_cot_output"],
            )
            results.append(r)
            if store:
                store.save(r, model_version=model_version)

        batch = FidelityBatchReport(results=results)

        if output_format == "json":
            print(batch.to_json())
        elif output_format == "markdown":
            print(batch.to_markdown())
        else:
            _print_text_report(batch)

        if min_faithfulness_rate is not None and batch.faithfulness_rate < min_faithfulness_rate:
            click.echo(
                f"❌ CI gate FAILED: rate {batch.faithfulness_rate:.2%} < {min_faithfulness_rate:.2%}",
                err=True,
            )
            sys.exit(1)

    @cli.command("report")
    @click.option("--db", default=None, help="Path to SQLite history database.")
    @click.option("--n", default=50, show_default=True, help="Number of recent results to include.")
    @click.option("--output-format", default="text", type=click.Choice(["text", "json", "markdown"]))
    def report_cmd(db, n, output_format):
        """Generate report from stored history."""
        store = FidelityStore(db_path=db)
        results = store.recent(n)
        if not results:
            click.echo("No results in history.")
            return
        batch = FidelityBatchReport(results=results)
        if output_format == "json":
            print(batch.to_json())
        elif output_format == "markdown":
            print(batch.to_markdown())
        else:
            _print_text_report(batch)

    @cli.command("drift")
    @click.option("--db", default=None, help="Path to SQLite history database.")
    @click.option("--window", default=50, show_default=True, help="Recent-window size.")
    @click.option("--output-format", default="text", type=click.Choice(["text", "json", "markdown"]))
    @click.option("--fail-on-degrading", is_flag=True,
                  help="Exit 1 if trend is DEGRADING.")
    def drift_cmd(db, window, output_format, fail_on_degrading):
        """Check faithfulness drift over time."""
        store = FidelityStore(db_path=db)
        report = store.detect_drift(window=window)
        if output_format == "json":
            print(json.dumps(report.to_dict()))
        elif output_format == "markdown":
            print(report.to_markdown())
        else:
            icon = "🔴" if report.drift_detected else "🟢"
            click.echo(f"Drift: {icon} {report.trend}")
            click.echo(f"Window: last {report.window} runs ({len(report.points)} available)")
            click.echo(f"Mean faithfulness: {report.mean_score:.4f} ± {report.std_score:.4f}")

        if fail_on_degrading and report.trend == "DEGRADING":
            sys.exit(1)

    @cli.command("clear")
    @click.option("--db", default=None, help="Path to SQLite history database.")
    @click.confirmation_option(prompt="This will delete all stored history. Continue?")
    def clear_cmd(db):
        """Clear all stored history."""
        store = FidelityStore(db_path=db)
        n = store.clear()
        click.echo(f"Cleared {n} records.")

    def _print_text_report(batch: FidelityBatchReport) -> None:
        click.echo(f"cot-fidelity report: {batch.total} tests")
        click.echo(f"  Faithful:     {batch.faithful_count} ({batch.faithfulness_rate:.1%})")
        click.echo(f"  Unfaithful:   {batch.unfaithful_count} ({batch.unfaithfulness_rate:.1%})")
        click.echo(f"  Inconclusive: {batch.inconclusive_count}")
        click.echo(f"  Mean score:   {batch.mean_faithfulness_score:.4f}")
        click.echo(f"  Mean sim:     {batch.mean_similarity:.4f}")
        for i, r in enumerate(batch.results, 1):
            icon = {"FAITHFUL": "✅", "UNFAITHFUL": "❌", "INCONCLUSIVE": "⚠️"}[r.verdict]
            click.echo(
                f"  [{i:02d}] {icon} {r.verdict:12s} score={r.faithfulness_score:.4f} "
                f"sim={r.similarity:.4f}"
            )

else:
    # Fallback when click is not installed
    def cli():  # type: ignore[misc]
        _require_click()
