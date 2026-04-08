"""
livelock_probe.cli
------------------
lprobe CLI — command-line interface for livelock-probe.

Commands:
  lprobe estimate   Estimate minimum steps needed to detect livelock given k/epsilon.
  lprobe report     Generate a JSON/text report from a saved step-trace file.
  lprobe gate       Read a saved report and exit 1 if livelock detected above threshold.
  lprobe show       Pretty-print a saved livelock report.
  lprobe demo       Run a built-in demonstration showing livelock vs progressing agent.
"""
from __future__ import annotations

import json
import sys
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    click = None  # type: ignore

from .engine import LivelockEngine
from .models import (
    CriticalityLevel,
    ProgressConfig,
    LIVELOCK_THRESHOLDS,
)
from .suite import LivelockSuite


def _require_cli_deps() -> None:
    if not _HAS_RICH or click is None:
        print(
            "ERROR: lprobe CLI requires 'click' and 'rich'.\n"
            "Install with: pip install livelock-probe[cli]",
            file=sys.stderr,
        )
        sys.exit(1)


if click is not None:

    @click.group()
    @click.version_option(version="0.1.0", prog_name="lprobe")
    def cli():
        """lprobe — AI agent livelock detection CLI.

        Detect when an agent is active but making zero net progress toward its goal.
        """

    @cli.command()
    @click.option("--k", default=5, show_default=True, help="Consecutive stuck steps to trigger.")
    @click.option("--epsilon", default=0.05, show_default=True, help="Progress delta threshold.")
    @click.option(
        "--criticality",
        default="HIGH",
        show_default=True,
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
        help="Criticality tier.",
    )
    def estimate(k: int, epsilon: float, criticality: str):
        """Estimate minimum steps needed for livelock detection to fire.

        Given k consecutive stuck steps needed and epsilon threshold,
        shows the minimum sequence length and detection timing.
        """
        _require_cli_deps()
        console = Console()
        threshold = LIVELOCK_THRESHOLDS[criticality]

        console.print(f"\n[bold]livelock-probe — Estimate[/bold]")
        console.print(f"  k           = {k} consecutive stuck steps")
        console.print(f"  epsilon     = {epsilon} (|progress_delta| < epsilon = stuck)")
        console.print(f"  criticality = {criticality}")
        console.print(f"  threshold   = {threshold:.0%} max stuck fraction\n")

        min_steps_for_detection = k
        min_steps_for_gate_fail = int(threshold * k / 1.0) + 1 if threshold > 0 else k

        console.print(f"[yellow]Earliest livelock detection:[/yellow] step {min_steps_for_detection}")
        console.print(f"[yellow]Steps for gate to fail at {criticality}:[/yellow] ")
        console.print(
            f"  If {int(threshold * 100) + 1}% or more of steps are stuck, gate fails.\n"
        )
        console.print(
            "[dim]Example: at k=5, epsilon=0.05, HIGH criticality (15% threshold):\n"
            "  - 5 consecutive stuck steps → livelock_detected=True\n"
            "  - >15% of all steps stuck → gate_passed=False[/dim]"
        )

    @cli.command()
    @click.argument("trace_file", type=click.Path(exists=True))
    @click.option("--goal", required=True, help="Goal description for progress measurement.")
    @click.option("--k", default=5, show_default=True, help="Consecutive stuck steps threshold.")
    @click.option("--epsilon", default=0.05, show_default=True, help="Progress delta threshold.")
    @click.option(
        "--criticality",
        default="HIGH",
        show_default=True,
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
    )
    @click.option(
        "--format",
        "output_format",
        default="text",
        type=click.Choice(["text", "json"]),
        show_default=True,
    )
    @click.option("--output", "-o", default=None, help="Output file path. Defaults to stdout.")
    def report(
        trace_file: str,
        goal: str,
        k: int,
        epsilon: float,
        criticality: str,
        output_format: str,
        output: Optional[str],
    ):
        """Generate a livelock report from a saved step-trace file.

        TRACE_FILE should be a JSON file with a "steps" array of strings,
        e.g.: {"steps": ["step 1 output", "step 2 output", ...]}
        """
        _require_cli_deps()
        console = Console()

        with open(trace_file) as f:
            data = json.load(f)

        if "steps" not in data or not isinstance(data["steps"], list):
            console.print("[red]ERROR:[/red] trace file must have a 'steps' key with a list of strings.")
            sys.exit(1)

        steps = [str(s) for s in data["steps"]]
        config = ProgressConfig(
            goal=goal,
            k=k,
            epsilon=epsilon,
            criticality=criticality,  # type: ignore
        )
        engine = LivelockEngine()
        lreport = engine.compute(steps, config)

        if output_format == "json":
            result = json.dumps(lreport.to_dict(), indent=2)
            if output:
                with open(output, "w") as f:
                    f.write(result)
                console.print(f"[green]Report written to {output}[/green]")
            else:
                click.echo(result)
        else:
            _print_report(console, lreport)

    @cli.command()
    @click.argument("report_file", type=click.Path(exists=True))
    @click.option(
        "--max-livelock-score",
        default=None,
        type=float,
        help="Override max LivelockScore threshold. Defaults to criticality threshold from report.",
    )
    def gate(report_file: str, max_livelock_score: Optional[float]):
        """Exit with code 1 if livelock detected above threshold.

        REPORT_FILE is a JSON report previously generated by 'lprobe report --format json'.
        Use in CI to fail a pipeline if an agent run shows livelock.

        Exit codes:
          0 = gate passed (no livelock above threshold)
          1 = gate failed (livelock detected)
        """
        _require_cli_deps()
        console = Console()

        with open(report_file) as f:
            data = json.load(f)

        livelock_score = data.get("livelock_score", 1.0)
        verdict = data.get("verdict", "LIVELOCK_DETECTED")
        criticality = data.get("criticality", "HIGH")
        threshold = data.get("threshold", LIVELOCK_THRESHOLDS.get(criticality, 0.15))

        effective_threshold = max_livelock_score if max_livelock_score is not None else threshold
        gate_passed = livelock_score <= effective_threshold

        console.print(f"\n[bold]lprobe gate[/bold]")
        console.print(f"  LivelockScore = {livelock_score:.3f}")
        console.print(f"  Threshold     = {effective_threshold:.3f} ({criticality})")
        console.print(f"  Verdict       = {verdict}")

        if gate_passed:
            console.print("\n[green bold]GATE PASSED[/green bold] — livelock score within threshold.")
            sys.exit(0)
        else:
            console.print("\n[red bold]GATE FAILED[/red bold] — livelock score exceeds threshold.")
            sys.exit(1)

    @cli.command()
    @click.argument("report_file", type=click.Path(exists=True))
    def show(report_file: str):
        """Pretty-print a saved livelock report.

        REPORT_FILE is a JSON report previously generated by 'lprobe report --format json'.
        """
        _require_cli_deps()
        console = Console()

        with open(report_file) as f:
            data = json.load(f)

        console.print(f"\n[bold]livelock-probe Report[/bold]")
        console.print(f"  Agent:          {data.get('agent_label', 'unknown')}")
        console.print(f"  Goal:           {data.get('goal', '')[:80]}")
        console.print(f"  Total steps:    {data.get('total_steps', 0)}")
        console.print(f"  LivelockScore:  {data.get('livelock_score', 0):.3f}")
        console.print(f"  Threshold:      {data.get('threshold', 0):.3f}")
        console.print(f"  Verdict:        {data.get('verdict', 'UNKNOWN')}")
        console.print(f"  Max consecutive stuck: {data.get('max_consecutive_stuck', 0)}")
        console.print(f"  Stuck window:   {data.get('stuck_window_start')} — {data.get('stuck_window_end')}")
        console.print(f"\n[italic]{data.get('recommendation', '')}[/italic]\n")

    @cli.command()
    @click.option("--k", default=3, show_default=True, help="Consecutive stuck steps threshold.")
    @click.option(
        "--criticality",
        default="HIGH",
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
        show_default=True,
    )
    def demo(k: int, criticality: str):
        """Run a built-in demonstration of livelock detection.

        Shows the difference between a progressing agent and a livelock agent.
        No external dependencies required.
        """
        _require_cli_deps()
        console = Console()

        console.print("\n[bold]livelock-probe Demo[/bold]\n")

        # --- Progressing agent ---
        progressing_steps = [
            "I need to fix the database connection error.",
            "I found the error: wrong hostname in config.",
            "Updating the hostname to production-db.example.com.",
            "Configuration updated. Testing connection.",
            "Connection test passed. Database is reachable.",
        ]
        progressing_config = ProgressConfig(
            goal="resolve the database connection error",
            k=k,
            criticality=criticality,  # type: ignore
            agent_label="progressing_agent",
        )
        engine = LivelockEngine()
        report_a = engine.compute(progressing_steps, progressing_config)

        console.print("[green bold]--- PROGRESSING AGENT ---[/green bold]")
        _print_report(console, report_a)

        # --- Livelock agent ---
        stuck_steps = [
            "Searching docs for connection error.",
            "Docs search returned no results. Retrying.",
            "Docs search returned no results. Retrying again.",
            "Docs search returned no results. Trying different query.",
            "Docs search returned no results. Retrying with original query.",
            "Docs search returned no results. Retrying.",
        ]
        stuck_config = ProgressConfig(
            goal="resolve the database connection error",
            k=k,
            criticality=criticality,  # type: ignore
            agent_label="livelock_agent",
        )
        report_b = engine.compute(stuck_steps, stuck_config)

        console.print("[red bold]--- LIVELOCK AGENT ---[/red bold]")
        _print_report(console, report_b)


    def _print_report(console: "Console", lreport) -> None:
        """Print a LivelockReport to the console in a human-readable format."""
        verdict_color = {
            "LIVELOCK_FREE": "green",
            "BORDERLINE": "yellow",
            "LIVELOCK_DETECTED": "red",
        }.get(lreport.verdict, "white")

        console.print(f"\n  Verdict:        [{verdict_color} bold]{lreport.verdict}[/{verdict_color} bold]")
        console.print(f"  LivelockScore:  {lreport.livelock_score:.3f} (threshold: {lreport.threshold:.3f})")
        console.print(f"  Total steps:    {lreport.total_steps}")
        console.print(f"  Max consecutive stuck: {lreport.max_consecutive_stuck} (k={lreport.k})")
        console.print(f"  Mean progress:  {lreport.mean_progress:+.4f}")
        if lreport.stuck_window_start is not None:
            console.print(f"  Stuck window:   steps {lreport.stuck_window_start}–{lreport.stuck_window_end}")
        console.print(f"  [italic]{lreport.recommendation}[/italic]\n")

        table = Table(title="Step Progress", show_header=True, header_style="bold")
        table.add_column("Step", justify="right", width=6)
        table.add_column("Progress→Goal", justify="right", width=14)
        table.add_column("Delta", justify="right", width=10)
        table.add_column("Stuck?", justify="center", width=8)
        table.add_column("Output (truncated)", width=50)

        for step in lreport.steps:
            stuck_str = "[red]YES[/red]" if step.is_stuck else "[green]no[/green]"
            table.add_row(
                str(step.step_id),
                f"{step.progress_to_goal:.4f}",
                f"{step.progress_delta:+.4f}",
                stuck_str,
                step.output[:50],
            )
        console.print(table)

else:
    # click not available — define a stub cli for import safety
    def cli():  # type: ignore
        _require_cli_deps()
