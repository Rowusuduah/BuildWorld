"""context_trace.cli
~~~~~~~~~~~~~~~~~
ctrace — command-line interface for context-trace.

Commands:
    ctrace run      Run attribution from a YAML config
    ctrace show     Display a report as an ASCII heatmap
    ctrace gate     CI gate check (exits non-zero on failure)
    ctrace compare  Delta between two reports
    ctrace estimate Estimate API calls and cost before running
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option(package_name="context-trace")
def cli() -> None:
    """ctrace — Per-context-chunk causal attribution for LLM outputs."""


@cli.command("run")
@click.option("--config", "-c", required=True, help="Path to ctrace.yaml config file.")
@click.option("--output", "-o", default=None, help="Output JSON path (default: stdout).")
@click.option("--label", default="", help="Label for this run in the store.")
@click.option("--store", default=None, help="SQLite store path (optional).")
def run_command(config: str, output: Optional[str], label: str, store: Optional[str]) -> None:
    """Run attribution analysis from a YAML config file."""
    try:
        import yaml
    except ImportError:
        click.echo("pyyaml is required: pip install pyyaml", err=True)
        sys.exit(1)

    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Config file not found: {config}", err=True)
        sys.exit(1)

    cfg = yaml.safe_load(config_path.read_text())

    runner = _build_runner(cfg.get("runner", {}))
    chunks = _load_chunks(cfg.get("chunks", {}), config_path.parent)
    prompt = _load_text(cfg.get("prompt", ""), config_path.parent)
    original_output = _load_text(cfg.get("original_output", ""), config_path.parent)

    if not prompt:
        click.echo("Error: 'prompt' is required in config.", err=True)
        sys.exit(1)
    if not original_output:
        click.echo("Error: 'original_output' is required in config.", err=True)
        sys.exit(1)

    from context_trace.tracer import ContextTracer, CostBudget

    budget_cfg = cfg.get("budget", {})
    budget = CostBudget(
        max_api_calls=budget_cfg.get("max_api_calls", 500),
        max_cost_usd=budget_cfg.get("max_cost_usd", 10.0),
        cost_per_call_usd=budget_cfg.get("cost_per_call_usd", 0.001),
    )

    tracer = ContextTracer(runner=runner, k=cfg.get("k", 3), budget=budget)
    report = tracer.trace(prompt=prompt, original_output=original_output, chunks=chunks)

    report_dict = report.to_dict()

    if store:
        from context_trace.store import AttributionStore
        with AttributionStore(store) as s:
            run_id = s.save(report, label=label)
        click.echo(f"Saved to store '{store}': run_id={run_id}", err=True)

    if output:
        Path(output).write_text(json.dumps(report_dict, indent=2))
        click.echo(f"Report written to {output}")
    else:
        click.echo(json.dumps(report_dict, indent=2))


@cli.command("show")
@click.option("--report", "-r", required=True, help="Path to JSON report file.")
@click.option("--top", default=5, help="Number of top contributors to show (default: 5).")
def show_command(report: str, top: int) -> None:
    """Display an attribution report as an ASCII heatmap."""
    report_path = Path(report)
    if not report_path.exists():
        click.echo(f"Report file not found: {report}", err=True)
        sys.exit(1)

    data = json.loads(report_path.read_text())
    chunk_scores = data.get("chunk_scores", {})

    if not chunk_scores:
        click.echo("No chunk scores in report.")
        return

    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: x[1]["attribution_score"],
        reverse=True,
    )

    max_name_len = max(len(n) for n in chunk_scores)
    max_score = max(v["attribution_score"] for v in chunk_scores.values())
    if max_score == 0:
        max_score = 1.0

    click.echo("\nAttribution Heatmap (sorted by causal contribution):")
    click.echo("-" * 55)
    for name, score_data in sorted_chunks[:top]:
        score = score_data["attribution_score"]
        bar_len = round((score / max_score) * 10)
        bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
        click.echo(f"  {name:<{max_name_len}} [{bar}] {score:.3f}")

    click.echo(f"\nTotal API calls : {data.get('total_api_calls', 'N/A')}")
    click.echo(f"Estimated cost  : ${data.get('estimated_cost_usd', 0):.4f}")
    click.echo(f"Elapsed         : {data.get('elapsed_seconds', 0):.2f}s")
    if data.get("skipped_chunks"):
        click.echo(f"Skipped chunks  : {', '.join(data['skipped_chunks'])}")


@cli.command("gate")
@click.option("--report", "-r", required=True, help="Path to JSON report file.")
@click.option("--max-score", default=None, type=float, help="Max attribution_score for any chunk.")
@click.option("--min-contributors", default=None, type=int, help="Min chunks with score >= 0.3.")
@click.option("--min-top-score", default=None, type=float, help="Min top contributor score.")
@click.option("--max-api-calls", default=None, type=int, help="Max API calls budget check.")
def gate_command(
    report: str,
    max_score: Optional[float],
    min_contributors: Optional[int],
    min_top_score: Optional[float],
    max_api_calls: Optional[int],
) -> None:
    """CI gate check on a report. Exits non-zero on failure."""
    report_path = Path(report)
    if not report_path.exists():
        click.echo(f"Report not found: {report}", err=True)
        sys.exit(1)

    data = json.loads(report_path.read_text())

    from context_trace.tracer import AttributionReport, ChunkScore
    from context_trace.gate import AttributionGate, AttributionGateFailure

    chunk_scores = {
        name: ChunkScore(
            chunk_name=name,
            attribution_score=v["attribution_score"],
            mean_similarity=v["mean_similarity"],
            std_similarity=v["std_similarity"],
            runs=v["runs"],
        )
        for name, v in data.get("chunk_scores", {}).items()
    }

    report_obj = AttributionReport(
        chunk_scores=chunk_scores,
        original_output="",
        prompt="",
        k=data.get("k", 3),
        total_api_calls=data.get("total_api_calls", 0),
        estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
        elapsed_seconds=data.get("elapsed_seconds", 0.0),
        skipped_chunks=data.get("skipped_chunks", []),
    )

    gate = AttributionGate(
        max_single_chunk_score=max_score,
        min_chunks_contributing=min_contributors,
        min_top_contributor_score=min_top_score,
        max_total_api_calls=max_api_calls,
    )

    try:
        gate.check(report_obj)
        click.echo("GATE PASSED")
        sys.exit(0)
    except AttributionGateFailure as e:
        click.echo(f"GATE FAILED: {e}", err=True)
        for v in e.violations:
            click.echo(f"  - {v}", err=True)
        sys.exit(1)


@cli.command("compare")
@click.option("--baseline", required=True, help="Baseline JSON report path.")
@click.option("--current", required=True, help="Current JSON report path.")
def compare_command(baseline: str, current: str) -> None:
    """Compare two attribution reports (delta per chunk)."""
    base_data = json.loads(Path(baseline).read_text())
    curr_data = json.loads(Path(current).read_text())

    base_chunks = base_data.get("chunk_scores", {})
    curr_chunks = curr_data.get("chunk_scores", {})
    all_chunks = set(base_chunks) | set(curr_chunks)

    deltas = []
    for name in all_chunks:
        base_score = base_chunks.get(name, {}).get("attribution_score", 0.0)
        curr_score = curr_chunks.get(name, {}).get("attribution_score", 0.0)
        delta = curr_score - base_score
        deltas.append((name, base_score, curr_score, delta))

    deltas.sort(key=lambda x: abs(x[3]), reverse=True)

    click.echo("\nAttribution Delta (current - baseline, sorted by |delta|):")
    click.echo("-" * 65)
    for name, base, curr, delta in deltas:
        sign = "+" if delta >= 0 else ""
        click.echo(
            f"  {name:<25}  baseline={base:.3f}  current={curr:.3f}  "
            f"delta={sign}{delta:.3f}"
        )


@cli.command("estimate")
@click.option("--config", "-c", required=True, help="Path to ctrace.yaml.")
def estimate_command(config: str) -> None:
    """Estimate API calls and cost without running."""
    try:
        import yaml
    except ImportError:
        click.echo("pyyaml is required: pip install pyyaml", err=True)
        sys.exit(1)

    cfg = yaml.safe_load(Path(config).read_text())
    chunks = cfg.get("chunks", {})
    k = cfg.get("k", 3)
    cost_per_call = cfg.get("budget", {}).get("cost_per_call_usd", 0.001)

    n_chunks = len(chunks)
    total_calls = n_chunks * k
    total_cost = total_calls * cost_per_call

    click.echo(f"Chunks         : {n_chunks}")
    click.echo(f"k (runs/chunk) : {k}")
    click.echo(f"Total API calls: {total_calls}")
    click.echo(f"Estimated cost : ${total_cost:.4f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(runner_cfg: dict):
    runner_type = runner_cfg.get("type", "anthropic")
    model = runner_cfg.get("model", "claude-haiku-4-5-20251001")
    max_tokens = runner_cfg.get("max_tokens", 512)
    if runner_type == "anthropic":
        from context_trace.runners import anthropic_runner
        return anthropic_runner(model=model, max_tokens=max_tokens)
    elif runner_type == "openai":
        from context_trace.runners import openai_runner
        return openai_runner(model=model, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown runner type: {runner_type!r}. Use 'anthropic' or 'openai'.")


def _load_text(source: object, base_dir: Path) -> str:
    if isinstance(source, str):
        candidate = base_dir / source
        if candidate.exists():
            return candidate.read_text()
        return source
    return ""


def _load_chunks(chunks_cfg: dict, base_dir: Path) -> dict:
    chunks = {}
    for name, spec in chunks_cfg.items():
        if isinstance(spec, str):
            chunks[name] = spec
        elif isinstance(spec, dict):
            if "source" in spec:
                chunks[name] = (base_dir / spec["source"]).read_text()
            elif "inline" in spec:
                chunks[name] = spec["inline"]
    return chunks


def main() -> None:
    cli()
