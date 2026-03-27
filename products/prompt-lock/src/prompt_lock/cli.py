"""Command-line interface for prompt-lock."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import PromptLockConfig
from .gate import evaluate_gate
from .runner import EvalResult, load_test_cases, run_eval
from .tracer import TraceLedger

console = Console()

# ─── Template files written by `init` ────────────────────────────────────────

_INIT_CONFIG = """\
# .prompt-lock.yml — prompt-lock configuration
# Docs: https://github.com/buildworld-ai/prompt-lock

version: "1"

# Default model for llm_judge evals (LiteLLM format: any provider)
model: gpt-4o-mini

# Judge calibration — verifies the LLM judge agrees with humans before trusting
# it as a CI gate. Unique to prompt-lock.
# judge:
#   enabled: true
#   human_labels_file: tests/human_labels.jsonl
#   model: gpt-4o-mini
#   criteria: "Rate the quality of this response on a scale of 0.0 to 1.0."
#   min_agreement: 0.80    # 80% of examples must be within ±0.15 of human score
#   min_spearman: 0.70     # Spearman rank correlation with human scores

# Prompt files to watch
prompts:
  - path: "prompts/*.txt"
    name: "My Prompts"
    test_cases_file: tests/test_cases.jsonl
    evals:
      - type: llm_judge
        criteria: "Is the response helpful, accurate, and well-structured?"
        threshold: 0.70
      # - type: semantic_similarity
      #   threshold: 0.80
      # - type: exact_match
      #   threshold: 1.0
      # - type: regex
      #   pattern: "\\\\b(yes|no)\\\\b"
      #   threshold: 1.0

# CI gate behavior
gate:
  mode: regression         # hard | regression | soft
  regression_threshold: 0.05   # fail if score drops more than 5% from baseline
  hard_threshold: 0.70         # used only in hard mode

# Trace ledger (SQLite, local)
tracer:
  db_path: .prompt-lock/traces.db
  enabled: true
"""

_EXAMPLE_TEST_CASES = """\
{"input": "What is 2+2?", "output": "The answer is 4.", "expected_output": "4"}
{"input": "Summarize Python in one sentence.", "output": "Python is a high-level, readable programming language known for its simplicity.", "expected_output": "Python is a versatile, readable programming language."}
{"input": "Translate 'hello' to Spanish.", "output": "Hola", "expected_output": "Hola"}
"""

_EXAMPLE_HUMAN_LABELS = """\
{"input": "What is 2+2?", "output": "The answer is 4.", "human_score": 1.0}
{"input": "What is 2+2?", "output": "It is roughly 5.", "human_score": 0.0}
{"input": "Summarize Python.", "output": "Python is a high-level, readable programming language.", "human_score": 0.9}
{"input": "Summarize Python.", "output": "Python is a snake species.", "human_score": 0.0}
{"input": "Explain gravity.", "output": "Gravity is the force that attracts objects with mass toward each other.", "human_score": 1.0}
{"input": "Explain gravity.", "output": "Gravity is when things fall down.", "human_score": 0.5}
"""


# ─── CLI entry point ──────────────────────────────────────────────────────────


@click.group()
@click.version_option(version="0.1.0", prog_name="prompt-lock")
def main():
    """prompt-lock — Git-native prompt regression testing with judge calibration.

    Guards at the gaps in your LLM CI/CD pipeline.
    """
    pass


# ─── init ─────────────────────────────────────────────────────────────────────


@main.command()
def init():
    """Initialize prompt-lock in the current directory."""
    config_path = Path(".prompt-lock.yml")
    if config_path.exists():
        console.print("[yellow]⚠[/yellow]  .prompt-lock.yml already exists — skipping")
    else:
        config_path.write_text(_INIT_CONFIG)
        console.print("[green]✓[/green]  Created .prompt-lock.yml")

    for dirpath in ["prompts", "tests"]:
        Path(dirpath).mkdir(exist_ok=True)
    console.print("[green]✓[/green]  Created prompts/ and tests/")

    tc = Path("tests/test_cases.jsonl")
    if not tc.exists():
        tc.write_text(_EXAMPLE_TEST_CASES)
        console.print("[green]✓[/green]  Created tests/test_cases.jsonl (example)")

    hl = Path("tests/human_labels.jsonl")
    if not hl.exists():
        hl.write_text(_EXAMPLE_HUMAN_LABELS)
        console.print("[green]✓[/green]  Created tests/human_labels.jsonl (example)")

    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Add your prompt files to [cyan]prompts/[/cyan]\n"
        "  2. Fill in [cyan]tests/test_cases.jsonl[/cyan] with real inputs/outputs\n"
        "  3. Run: [cyan bold]prompt-lock check[/cyan bold]"
    )


# ─── check ────────────────────────────────────────────────────────────────────


@main.command()
@click.option("--config", "-c", default=".prompt-lock.yml", help="Config file path")
@click.option(
    "--all-prompts",
    is_flag=True,
    help="Evaluate ALL prompts, not just git-changed ones",
)
@click.option("--base-ref", default="HEAD~1", help="Git ref to diff against")
@click.option("--no-calibrate", is_flag=True, help="Skip judge calibration check")
@click.option("--verbose", "-v", is_flag=True, help="Show per-test-case results")
def check(
    config: str,
    all_prompts: bool,
    base_ref: str,
    no_calibrate: bool,
    verbose: bool,
):
    """Run prompt regression checks. Exits 1 on gate failure, 2 on calibration failure."""
    try:
        cfg = PromptLockConfig.from_file(config)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red]  {e}")
        sys.exit(1)

    tracer = TraceLedger(cfg.tracer.db_path) if cfg.tracer.enabled else TraceLedger()
    any_failed = False

    # ── Judge calibration ──────────────────────────────────────────────────
    if cfg.judge and cfg.judge.enabled and not no_calibrate:
        console.rule("[bold]Judge Calibration[/bold]")
        from .judge.calibrate import calibrate_judge

        try:
            cal = calibrate_judge(
                human_labels_file=cfg.judge.human_labels_file,
                model=cfg.judge.model,
                criteria=cfg.judge.criteria,
                min_agreement=cfg.judge.min_agreement,
                min_spearman=cfg.judge.min_spearman,
                agreement_tolerance=cfg.judge.agreement_tolerance,
            )
            tracer.log_calibration(
                model=cal.model,
                criteria=cal.criteria,
                agreement_rate=cal.agreement_rate,
                spearman_correlation=cal.spearman_correlation,
                bias=cal.bias,
                n_examples=cal.n_examples,
                passed=cal.passed,
                details=cal.details,
            )
            if cal.passed:
                console.print(f"[green]✓[/green]  {cal.summary()}")
            else:
                console.print(f"[red]✗[/red]  {cal.summary()}")
                console.print(
                    "[red]  Judge failed calibration — LLM eval scores cannot be trusted "
                    "as a CI gate.[/red]\n"
                    "  Run [cyan]prompt-lock calibrate[/cyan] for details, "
                    "or [cyan]--no-calibrate[/cyan] to skip."
                )
                sys.exit(2)
        except Exception as e:
            console.print(f"[red]✗[/red]  Calibration error: {e}")
            if verbose:
                traceback.print_exc()
            sys.exit(1)

    # ── Prompt detection ───────────────────────────────────────────────────
    detector = None
    if not all_prompts:
        try:
            from .detector import ChangedPromptDetector

            detector = ChangedPromptDetector()
        except Exception:
            all_prompts = True  # fallback: not in a git repo

    console.rule("[bold]Eval Runs[/bold]")

    for prompt_cfg in cfg.prompts:
        if all_prompts or detector is None:
            prompt_files = list(Path(".").glob(prompt_cfg.path))
        else:
            prompt_files = detector.detect_changed_prompts(
                [prompt_cfg.path], base_ref=base_ref
            )

        if not prompt_files:
            if verbose:
                console.print(
                    f"[dim]  No changed prompts matching {prompt_cfg.path!r} — skipping[/dim]"
                )
            continue

        if not prompt_cfg.test_cases_file:
            console.print(
                f"[yellow]⚠[/yellow]  No test_cases_file for {prompt_cfg.path!r} — skipping"
            )
            continue

        try:
            test_cases = load_test_cases(prompt_cfg.test_cases_file)
        except Exception as e:
            console.print(f"[red]✗[/red]  Could not load test cases: {e}")
            any_failed = True
            continue

        for prompt_file in prompt_files:
            prompt_content = prompt_file.read_text()
            name = prompt_cfg.name or str(prompt_file)

            for eval_cfg in prompt_cfg.evals:
                results: list[EvalResult] = []

                for tc in test_cases:
                    output = tc.get("output", "")
                    if not output:
                        continue
                    try:
                        result = run_eval(
                            eval_config=eval_cfg,
                            input_text=tc["input"],
                            output_text=output,
                            expected=tc.get("expected_output"),
                            default_model=cfg.model,
                        )
                        results.append(result)

                        if cfg.tracer.enabled:
                            tracer.log_eval(
                                prompt_path=str(prompt_file),
                                prompt_content=prompt_content,
                                eval_type=result.eval_type,
                                score=result.score,
                                passed=result.passed,
                                threshold=result.threshold,
                                model=eval_cfg.model or cfg.model,
                                details=result.details,
                                input_text=result.input_text,
                                output_text=result.output_text,
                            )

                        if verbose:
                            mark = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
                            console.print(
                                f"    {mark} [{eval_cfg.type}] "
                                f"score={result.score:.3f} — {result.details[:60]}"
                            )
                    except Exception as e:
                        console.print(f"    [red]✗[/red]  Eval error: {e}")
                        if verbose:
                            traceback.print_exc()

                if not results:
                    continue

                avg_score = sum(r.score for r in results) / len(results)
                aggregate = EvalResult(
                    eval_type=eval_cfg.type,
                    score=avg_score,
                    passed=avg_score >= eval_cfg.threshold,
                    threshold=eval_cfg.threshold,
                    details=f"avg over {len(results)} test cases",
                    input_text="",
                    output_text="",
                )
                decision = evaluate_gate(aggregate, cfg.gate, tracer, str(prompt_file))

                mark = "[red]✗ FAIL[/red]" if decision.should_fail else "[green]✓ PASS[/green]"
                console.print(
                    f"  {mark}  [{eval_cfg.type}] {name} "
                    f"score={avg_score:.3f}  {decision.reason}"
                )

                if decision.should_fail:
                    any_failed = True

    console.rule()
    if any_failed:
        console.print("[red bold]✗  prompt-lock FAILED — regression detected[/red bold]")
        sys.exit(1)
    else:
        console.print("[green bold]✓  prompt-lock PASSED[/green bold]")
        sys.exit(0)


# ─── calibrate ────────────────────────────────────────────────────────────────


@main.command()
@click.option("--config", "-c", default=".prompt-lock.yml", help="Config file path")
def calibrate(config: str):
    """Run judge calibration and display detailed per-example results."""
    try:
        cfg = PromptLockConfig.from_file(config)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red]  {e}")
        sys.exit(1)

    if not cfg.judge:
        console.print("[yellow]⚠[/yellow]  No [cyan]judge:[/cyan] section in config")
        console.print("  Add judge calibration config to .prompt-lock.yml first.")
        sys.exit(0)

    console.print("[bold]Running judge calibration...[/bold]\n")

    from .judge.calibrate import calibrate_judge

    try:
        result = calibrate_judge(
            human_labels_file=cfg.judge.human_labels_file,
            model=cfg.judge.model,
            criteria=cfg.judge.criteria,
            min_agreement=cfg.judge.min_agreement,
            min_spearman=cfg.judge.min_spearman,
            agreement_tolerance=cfg.judge.agreement_tolerance,
        )
    except Exception as e:
        console.print(f"[red]✗[/red]  {e}")
        sys.exit(1)

    table = Table(
        title="Per-Example Results",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    table.add_column("Input (truncated)", max_width=45, no_wrap=False)
    table.add_column("Human", justify="right", width=7)
    table.add_column("Judge", justify="right", width=7)
    table.add_column("Agree?", justify="center", width=7)
    table.add_column("Reasoning", max_width=40, no_wrap=False)

    for d in result.details:
        agree = "[green]✓[/green]" if d["agreement"] else "[red]✗[/red]"
        table.add_row(
            d["input"][:45],
            f"{d['human_score']:.2f}",
            f"{d['judge_score']:.2f}",
            agree,
            str(d["reasoning"])[:40],
        )

    console.print(table)
    console.print()

    color = "green" if result.passed else "red"
    status = "PASSED" if result.passed else "FAILED"
    console.print(
        Panel(
            f"[{color} bold]{status}[/{color} bold]\n\n"
            f"Agreement rate   [bold]{result.agreement_rate:.1%}[/bold]  "
            f"(min: {cfg.judge.min_agreement:.0%})\n"
            f"Spearman r       [bold]{result.spearman_correlation:.3f}[/bold]  "
            f"(min: {cfg.judge.min_spearman:.2f})\n"
            f"Bias             [bold]{result.bias:+.3f}[/bold]  "
            f"(positive = judge inflates scores)\n"
            f"Examples         [bold]{result.n_examples}[/bold]",
            title="Calibration Summary",
            border_style=color,
        )
    )
    sys.exit(0 if result.passed else 1)


# ─── traces ───────────────────────────────────────────────────────────────────


@main.group()
def traces():
    """Inspect the eval trace ledger."""
    pass


@traces.command(name="show")
@click.option("--config", "-c", default=".prompt-lock.yml", help="Config file path")
@click.option("--limit", "-n", default=20, show_default=True, help="Number of runs to show")
def traces_show(config: str, limit: int):
    """Show the most recent eval runs."""
    ledger = _get_ledger(config)
    runs = ledger.get_recent_runs(limit)

    if not runs:
        console.print("[dim]No eval runs recorded yet. Run [cyan]prompt-lock check[/cyan].[/dim]")
        return

    table = Table(title=f"Recent {len(runs)} Eval Runs", show_header=True, header_style="bold")
    table.add_column("Timestamp", max_width=20)
    table.add_column("Commit", width=9)
    table.add_column("Prompt", max_width=32)
    table.add_column("Type", max_width=18)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Pass?", justify="center", width=6)

    for r in runs:
        mark = "[green]✓[/green]" if r["passed"] else "[red]✗[/red]"
        table.add_row(
            r["timestamp"][:19],
            r["commit"] or "—",
            ("…" + r["prompt"][-31:]) if len(r["prompt"]) > 32 else r["prompt"],
            r["eval_type"],
            f"{r['score']:.3f}",
            mark,
        )

    console.print(table)


@traces.command(name="diff")
@click.argument("commit_a")
@click.argument("commit_b")
@click.option("--config", "-c", default=".prompt-lock.yml", help="Config file path")
def traces_diff(commit_a: str, commit_b: str, config: str):
    """Compare eval scores between two commits (short SHAs)."""
    ledger = _get_ledger(config)
    diffs = ledger.diff_commits(commit_a, commit_b)

    if not diffs:
        console.print(f"[dim]No data found for commits {commit_a!r} or {commit_b!r}.[/dim]")
        return

    table = Table(
        title=f"Score Diff: {commit_a} → {commit_b}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Prompt", max_width=32)
    table.add_column("Type")
    table.add_column(commit_a, justify="right")
    table.add_column(commit_b, justify="right")
    table.add_column("Delta", justify="right")

    for d in diffs:
        delta = d["delta"]
        delta_str = f"{delta:+.3f}" if delta is not None else "—"
        delta_color = "green" if (delta or 0) >= 0 else "red"
        sa = d.get(f"score_{commit_a}")
        sb = d.get(f"score_{commit_b}")
        table.add_row(
            d["prompt"][-32:],
            d["eval_type"],
            f"{sa:.3f}" if sa is not None else "—",
            f"{sb:.3f}" if sb is not None else "—",
            f"[{delta_color}]{delta_str}[/{delta_color}]",
        )

    console.print(table)


def _get_ledger(config: str) -> TraceLedger:
    try:
        cfg = PromptLockConfig.from_file(config)
        return TraceLedger(cfg.tracer.db_path)
    except FileNotFoundError:
        return TraceLedger()
