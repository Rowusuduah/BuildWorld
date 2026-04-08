"""
pressure-gauge CLI: pgauge
--------------------------
Command-line interface for context pressure sweeps.
Requires: pip install pressure-gauge[cli]
"""
from __future__ import annotations

import json
import sys
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table

    _HAS_CLI = True
except ImportError:
    _HAS_CLI = False

from .gauge import PressureGauge
from .models import CriticalityLevel, PressureConfig, PressureReport

if _HAS_CLI:
    console = Console()

    @click.group()
    def cli() -> None:
        """pressure-gauge: Detect LLM behavioral drift under context fill pressure."""

    # ------------------------------------------------------------------ run
    @cli.command()
    @click.option(
        "--model-limit", default=8192, type=int,
        help="Model context window size in tokens.",
    )
    @click.option(
        "--fill-levels", default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated fill levels, e.g. 0.1,0.5,0.9.",
    )
    @click.option(
        "--stability-threshold", default=0.85, type=float,
        help="Minimum ContextPressureScore to pass gate.",
    )
    @click.option(
        "--criticality", default="HIGH",
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
    )
    @click.option(
        "--strategy", default="lorem_ipsum",
        type=click.Choice(["lorem_ipsum", "repeat_text", "inject_history"]),
        help="Context padding strategy.",
    )
    @click.option(
        "--context", default="Summarize the key points of the document above.",
        help="Base task context passed to the agent.",
    )
    @click.option("--json-output", is_flag=True, help="Output as JSON.")
    def run(
        model_limit: int,
        fill_levels: str,
        stability_threshold: float,
        criticality: str,
        strategy: str,
        context: str,
        json_output: bool,
    ) -> None:
        """Run a pressure sweep with a mock agent (for testing/demo)."""
        levels = [float(x.strip()) for x in fill_levels.split(",")]
        config = PressureConfig(
            model_context_limit=model_limit,
            fill_levels=levels,
            stability_threshold=stability_threshold,
            criticality=CriticalityLevel(criticality),
            padding_strategy=strategy,
        )
        gauge = PressureGauge(config=config)

        # Mock agent: echoes the last 20 words of context (stable behavior)
        def mock_agent(ctx: str) -> str:
            words = ctx.split()
            return " ".join(words[-20:]) if words else "empty"

        report = gauge.sweep(agent_fn=mock_agent, base_context=context)

        if json_output:
            click.echo(json.dumps(report.as_dict(), indent=2))
        else:
            _print_report(report)

    # ----------------------------------------------------------------- quick
    @cli.command()
    @click.option("--model-limit", default=8192, type=int)
    @click.option(
        "--criticality", default="HIGH",
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
    )
    @click.option("--context", default="What is the answer?")
    def quick(model_limit: int, criticality: str, context: str) -> None:
        """Quick 3-level sweep (10%, 50%, 90%)."""
        config = PressureConfig(
            model_context_limit=model_limit,
            criticality=CriticalityLevel(criticality),
        )
        gauge = PressureGauge(config=config)

        def mock_agent(ctx: str) -> str:
            words = ctx.split()
            return words[-1] if words else "empty"

        report = gauge.quick(agent_fn=mock_agent, base_context=context)
        _print_report(report)

    # ------------------------------------------------------------------ gate
    @cli.command()
    @click.option("--score", required=True, type=float,
                  help="ContextPressureScore to evaluate.")
    @click.option(
        "--criticality", default="HIGH",
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
    )
    @click.option("--fail-exit-code", default=1, type=int,
                  help="Exit code when gate fails.")
    def gate(score: float, criticality: str, fail_exit_code: int) -> None:
        """Check if a score passes the gate for a given criticality level."""
        from .models import PRESSURE_THRESHOLDS, score_to_verdict

        crit = CriticalityLevel(criticality)
        threshold = PRESSURE_THRESHOLDS[crit]
        passed = score >= threshold
        verdict = score_to_verdict(score, crit)

        if passed:
            console.print(
                f"[green]GATE PASSED[/green] — "
                f"score={score:.4f} >= {threshold} ({criticality})"
            )
        else:
            console.print(
                f"[red]GATE FAILED[/red] — "
                f"score={score:.4f} < {threshold} ({criticality}). "
                f"Verdict: {verdict.value}"
            )
            sys.exit(fail_exit_code)

    # --------------------------------------------------------------- estimate
    @cli.command()
    @click.option("--model-limit", default=8192, type=int)
    @click.option(
        "--criticality", default="HIGH",
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
    )
    def estimate(model_limit: int, criticality: str) -> None:
        """Estimate token counts for each fill level."""
        from .models import PRESSURE_THRESHOLDS

        config = PressureConfig(
            model_context_limit=model_limit,
            criticality=CriticalityLevel(criticality),
        )
        table = Table(title=f"Fill Level Token Estimates (limit={model_limit:,})")
        table.add_column("Fill Level", style="cyan")
        table.add_column("Target Tokens", style="green")
        table.add_column("Approx Chars (4 chars/token)", style="yellow")

        for fl in config.fill_levels:
            tokens = config.tokens_for_level(fl)
            chars = int(tokens * config.chars_per_token)
            table.add_row(f"{fl:.0%}", f"{tokens:,}", f"{chars:,}")

        console.print(table)
        threshold = PRESSURE_THRESHOLDS[CriticalityLevel(criticality)]
        console.print(
            f"\n[bold]Gate threshold ({criticality}):[/bold] "
            f"ContextPressureScore >= {threshold}"
        )

    # ------------------------------------------------------------------- demo
    @cli.command()
    def demo() -> None:
        """Run a demonstration with a stable and a pressure-sensitive agent."""
        config = PressureConfig(
            model_context_limit=2048,
            fill_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            criticality=CriticalityLevel.HIGH,
        )
        gauge = PressureGauge(config=config)

        console.print("\n[bold cyan]Demo 1: Stable agent[/bold cyan]")

        def stable_agent(ctx: str) -> str:
            return (
                "The answer is 42. Here is a complete and thorough analysis "
                "of the problem that covers all required aspects in detail."
            )

        report1 = gauge.sweep(agent_fn=stable_agent, base_context="What is the answer?")
        _print_report(report1)

        console.print("\n[bold red]Demo 2: Pressure-sensitive agent[/bold red]")

        _n = [0]

        def drifting_agent(ctx: str) -> str:
            _n[0] += 1
            if _n[0] <= 2:
                return "The answer is 42. Here is a thorough and complete analysis."
            elif _n[0] <= 4:
                return "42. Brief answer."
            else:
                return "Done."

        report2 = gauge.sweep(agent_fn=drifting_agent, base_context="What is the answer?")
        _print_report(report2)

    # ----------------------------------------------------------------- onset
    @cli.command()
    @click.option("--model-limit", default=8192, type=int)
    @click.option(
        "--criticality", default="HIGH",
        type=click.Choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
    )
    @click.option("--granularity", default=10, type=int,
                  help="Number of fill levels for fine-grained onset detection.")
    def onset(model_limit: int, criticality: str, granularity: int) -> None:
        """Estimate pressure onset token with fine granularity (mock agent)."""
        config = PressureConfig(
            model_context_limit=model_limit,
            criticality=CriticalityLevel(criticality),
        )
        gauge = PressureGauge(config=config)

        def mock_agent(ctx: str) -> str:
            return (
                "Complete thorough analysis with comprehensive detail and "
                "full coverage of all requirements."
            )

        onset_token = gauge.estimate_onset(mock_agent, granularity=granularity)
        if onset_token is None:
            console.print(
                "[green]No pressure onset detected[/green] — "
                "agent behavior stable across all fill levels."
            )
        else:
            console.print(
                f"[yellow]Pressure onset detected[/yellow] at "
                f"~{onset_token:,} tokens."
            )

    # ---------------------------------------------------------------- helpers
    def _print_report(report: PressureReport) -> None:
        table = Table(title="ContextDriftCurve")
        table.add_column("Fill Level", style="cyan")
        table.add_column("Tokens", style="yellow")
        table.add_column("Similarity to Baseline", justify="right")
        table.add_column("Verdict", style="magenta")

        for dp in report.drift_curve:
            color = "green" if dp.similarity_to_baseline >= 0.85 else "red"
            table.add_row(
                f"{dp.fill_level:.0%}",
                f"{dp.token_count:,}",
                f"[{color}]{dp.similarity_to_baseline:.4f}[/{color}]",
                dp.verdict.value,
            )

        console.print(table)

        score_color = "green" if report.gate_passed else "red"
        console.print(
            f"\n[bold]ContextPressureScore:[/bold] "
            f"[{score_color}]{report.context_pressure_score:.4f}[/{score_color}]"
        )
        console.print(f"[bold]Verdict:[/bold] {report.verdict.value}")

        if report.pressure_onset_token:
            console.print(
                f"[bold]Pressure onset:[/bold] ~{report.pressure_onset_token:,} tokens"
            )
        else:
            console.print("[bold]Pressure onset:[/bold] not detected")

        gate_str = (
            "[green]PASSED[/green]" if report.gate_passed else "[red]FAILED[/red]"
        )
        console.print(f"[bold]Gate:[/bold] {gate_str}")
        console.print(f"[bold]Recommendation:[/bold] {report.recommendation}")

else:
    def cli() -> None:  # type: ignore[misc]
        print("CLI requires: pip install pressure-gauge[cli]", file=sys.stderr)
        sys.exit(1)
