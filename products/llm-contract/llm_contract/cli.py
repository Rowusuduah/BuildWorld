"""CLI for llm-contract — validate and drift-report commands."""

from __future__ import annotations

import sys

try:
    import click
except ImportError:
    click = None  # type: ignore[assignment]


def _require_click() -> None:
    if click is None:
        print("click is required for CLI usage. Install with: pip install llm-contract[cli]")
        sys.exit(1)


def main() -> None:
    """Entry point for the llm-contract CLI."""
    _require_click()
    cli()


if click is not None:
    @click.group()
    @click.version_option()
    def cli() -> None:
        """llm-contract — Define, version, and enforce behavioral contracts on LLM function calls."""

    @cli.command("validate")
    @click.option("--db", default="./llm_contract_logs.db", show_default=True,
                  help="Path to the SQLite drift log database.")
    @click.option("--function", "function_name", default=None,
                  help="Filter to a specific function name.")
    @click.option("--min-pass-rate", default=0.90, show_default=True, type=float,
                  help="Minimum pass rate to consider healthy (0.0–1.0).")
    @click.option("--days", default=None, type=int,
                  help="Only consider evaluations from the last N days.")
    def validate_cmd(db: str, function_name: str | None, min_pass_rate: float,
                     days: int | None) -> None:
        """Check pass rates for all contracts in the drift log.

        Exits with code 1 if any contract is below the threshold.
        """
        from llm_contract.storage import get_pass_rate, list_contracts

        contracts = list_contracts(db)
        if not contracts:
            click.echo("No contract evaluations found in database.")
            sys.exit(0)

        if function_name:
            contracts = [c for c in contracts if c["function_name"] == function_name]
            if not contracts:
                click.echo(f"No evaluations found for function: {function_name!r}")
                sys.exit(0)

        any_failed = False
        for c in contracts:
            rate = get_pass_rate(c["function_name"], db, days=days)
            if rate is None:
                continue
            symbol = "✓" if rate >= min_pass_rate else "✗"
            status = "PASS" if rate >= min_pass_rate else "FAIL"
            click.echo(
                f"  {symbol} {c['function_name']} v{c['contract_version']} — "
                f"{rate:.1%} ({status}) "
                f"[threshold: {min_pass_rate:.0%}] "
                f"[{c['total']} evals, provider: {c['provider']}]"
            )
            if rate < min_pass_rate:
                any_failed = True

        if any_failed:
            click.echo("\nGATE FAIL — one or more contracts below threshold.", err=True)
            sys.exit(1)
        else:
            click.echo("\nAll contracts healthy.")

    @cli.command("drift-report")
    @click.option("--db", default="./llm_contract_logs.db", show_default=True,
                  help="Path to the SQLite drift log database.")
    @click.option("--function", "function_name", default=None,
                  help="Filter to a specific function name.")
    @click.option("--last", "days", default=30, show_default=True, type=int,
                  help="Number of days to analyze.")
    def drift_report_cmd(db: str, function_name: str | None, days: int) -> None:
        """Show behavioral drift for contracts over the last N days.

        Compares pass rate in the first half vs second half of the window.
        Flags contracts with drift >= 5 percentage points.
        """
        from llm_contract.storage import get_drift_report, list_contracts

        contracts = list_contracts(db)
        if not contracts:
            click.echo("No contract evaluations found in database.")
            sys.exit(0)

        if function_name:
            contracts = [c for c in contracts if c["function_name"] == function_name]

        any_drift = False
        for c in contracts:
            report = get_drift_report(c["function_name"], db, days=days)
            if report["current_pass_rate"] is None:
                continue

            drift_str = ""
            if report["drift_pp"] is not None:
                sign = "+" if report["drift_pp"] >= 0 else ""
                drift_str = f"{sign}{report['drift_pp']:.1f}pp"

            symbol = "!" if report["has_drift"] else "✓"
            drift_label = "DRIFT DETECTED" if report["has_drift"] else "stable"

            prior_str = (
                f"{report['prior_pass_rate']:.1%}"
                if report["prior_pass_rate"] is not None
                else "N/A"
            )
            click.echo(
                f"  {symbol} {c['function_name']} v{c['contract_version']} | "
                f"{prior_str} → {report['current_pass_rate']:.1%} "
                f"({drift_str}) | {drift_label} "
                f"[{report['evaluation_count']} evals, {days}d window]"
            )
            if report["has_drift"]:
                any_drift = True

        if any_drift:
            click.echo("\nDrift detected in one or more contracts. Review model/provider changes.")
        else:
            click.echo("\nNo significant drift detected.")

    @cli.command("list")
    @click.option("--db", default="./llm_contract_logs.db", show_default=True,
                  help="Path to the SQLite drift log database.")
    def list_cmd(db: str) -> None:
        """List all tracked contracts in the drift log."""
        from llm_contract.storage import list_contracts

        contracts = list_contracts(db)
        if not contracts:
            click.echo("No contract evaluations found.")
            return

        click.echo(f"{'FUNCTION':<40} {'VERSION':<10} {'PROVIDER':<12} {'PASS%':<8} {'EVALS':<8} LAST SEEN")
        click.echo("-" * 100)
        for c in contracts:
            rate = c["passed_count"] / c["total"] if c["total"] > 0 else 0
            click.echo(
                f"{c['function_name']:<40} {c['contract_version']:<10} "
                f"{c['provider']:<12} {rate:<8.1%} {c['total']:<8} {c['last_seen'][:19]}"
            )
