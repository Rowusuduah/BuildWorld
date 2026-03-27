# drift-guard

**Verify that pull requests do what they say.**

drift-guard is a git-native semantic PR intent verifier. You write what a PR is supposed to do — drift-guard checks whether the actual code changes fulfill it. Catches silent contract violations, scope creep, and incomplete implementations before they hit production.

```bash
pip install drift-guard
```

---

## The Problem

AI generates 42% of code in 2026. AI-assisted PRs create 1.7x more issues than human-only PRs. The core failure mode is **intent drift**: the PR says "fix null pointer in user registration" but the diff also silently removes an auth check. Tests pass. Code review misses it. Production breaks.

No existing tool verifies **whether a PR's code changes actually fulfill the stated intent**. Code review tools (CodeRabbit, PR-Agent) suggest improvements. Linters check syntax. CI runs tests. **Nobody checks: does this code do what the PR description says it will?**

drift-guard does exactly that.

---

## Quick Start

### CLI — verify a PR before merge

```bash
# From inside a git repo
drift-guard verify \
  --title "Fix null pointer in user registration when email is None" \
  --description "Adds None check before calling .lower() on email field in register_user(). Adds test for None email case." \
  --base origin/main \
  --head HEAD \
  --format text
```

**Output:**
```
drift-guard result: PASS
Drift score: 0.04 (threshold: 0.30)
Confidence: 92%

Intent summary:
  The PR claims to add a None check for the email field in user registration
  and add a corresponding test. The diff shows exactly this: a None guard
  added in register_user() and a new test_register_null_email() test function.

Clauses (2):
  [+] Adds None check before calling .lower() on email field
       -> Diff shows `if email is None: raise ValueError` added in register_user()
  [+] Adds test for None email case
       -> New test_register_null_email test added in tests/test_auth.py
```

### Python API

```python
from drift_guard import verify

report = verify(
    pr_title="Refactor payment processor to use new Stripe SDK",
    pr_description="""
    - Updates stripe.charge() calls to stripe.PaymentIntent.create()
    - Removes deprecated customer.sources usage
    - Does not change the public API of charge_customer()
    """,
    base="origin/main",
    head="HEAD",
)

if not report.passed():
    print(report.to_markdown())
    exit(1)

print(f"Drift score: {report.drift_score:.2f}")
print(f"Confidence: {report.overall_confidence:.0%}")
```

### GitHub Action

```yaml
# .github/workflows/drift-guard.yml
name: drift-guard

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  verify-intent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install drift-guard
        run: pip install drift-guard

      - name: Verify PR intent
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          drift-guard verify \
            --title "${{ github.event.pull_request.title }}" \
            --description "${{ github.event.pull_request.body }}" \
            --base origin/${{ github.base_ref }} \
            --head ${{ github.event.pull_request.head.sha }} \
            --format markdown \
            --threshold 0.3
```

---

## How It Works

**Three-layer verification:**

```
PR title + description
       │
       ▼
┌─────────────────┐
│  Intent Parser  │  Extracts structured clauses from natural language:
│                 │  "adds X", "removes Y", "ensures Z", "does not..."
└────────┬────────┘
         │ clauses
         ▼
┌─────────────────┐
│  Diff Fetcher   │  git diff base..head → structured DiffHunks
│  + Parser       │  File paths, added/removed lines, hunk boundaries
└────────┬────────┘
         │ diff
         ▼
┌─────────────────┐
│  LLM Verifier   │  Claude (claude-haiku-4-5 by default) checks each
│  (Claude)       │  clause against the actual diff. Returns:
│                 │  - PASS / FAIL / WARN / SKIP per clause
│                 │  - Evidence quote from the diff
│                 │  - Drift score (0.0–1.0)
└────────┬────────┘
         │ report
         ▼
┌─────────────────┐
│  SQLite Trace   │  Every report persisted. Drift trends visible.
│  Log            │  history subcommand shows all PRs over time.
└─────────────────┘
```

**Drift Score:**
- `0.00` — All clauses fulfilled. PR does exactly what it says.
- `0.30` — Default threshold. Above this, CI gate fails.
- `1.00` — Complete mismatch. PR says one thing; code does another.

---

## What drift-guard Catches

| Scenario | What happens | drift-guard verdict |
|----------|-------------|---------------------|
| PR says "fix null check", also silently removes auth guard | Tests pass | FAIL — auth removal not mentioned |
| PR says "add feature X", only adds skeleton | Tests pass | WARN — partial implementation |
| PR says "does not change public API", but renames a method | CI passes | FAIL — API change not in description |
| PR says "removes deprecated Y", only comments it out | Tests pass | WARN — removal incomplete |
| PR says "fix bug in payment flow", diff only touches logging | Tests pass | FAIL — no payment code changed |
| PR says "add validation for email", adds it correctly | Tests pass | PASS |

---

## Configuration

### `.drift-guard.toml` (project root)

```toml
[drift-guard]
threshold = 0.25                      # Drift score above which CI fails (default: 0.30)
model = "claude-haiku-4-5-20251001"   # LLM used for verification
warn_on_scope_creep = true            # Warn when diff exceeds stated scope
skip_on_no_description = true         # Skip verification if PR has no body
db_path = ".drift-guard.db"           # SQLite trace log location

[drift-guard.ignore_files]
# Files to exclude from scope-creep analysis
patterns = ["*.lock", "*.generated.*", "CHANGELOG.md"]
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required. Your Anthropic API key. |
| `DRIFT_GUARD_MODEL` | Override default model. |
| `DRIFT_GUARD_THRESHOLD` | Override default drift threshold. |
| `DRIFT_GUARD_DB` | Override SQLite database path. |

---

## Commands

```bash
# Verify a PR
drift-guard verify --title "..." --description "..." [options]

# Show verification history
drift-guard history --n 20

# Output formats
drift-guard verify ... --format text      # Human-readable (default)
drift-guard verify ... --format json      # Machine-readable JSON
drift-guard verify ... --format markdown  # GitHub comment-ready markdown

# Adjust CI threshold
drift-guard verify ... --threshold 0.2   # Stricter gate
drift-guard verify ... --threshold 0.5   # Looser gate
```

---

## Comparison with Existing Tools

| Tool | What it does | Does it check intent? |
|------|-------------|----------------------|
| CodeRabbit ($24-30/dev/month) | Reviews code style, bugs, best practices | No |
| PR-Agent / Qodo ($19-30/dev/month) | Suggests improvements, summarizes diff | No |
| DeepEval | Evaluates LLM outputs against metrics | No |
| ESLint / Ruff | Lints syntax | No |
| Codecov | Measures test coverage | No |
| **drift-guard (free, MIT)** | **Verifies code changes fulfill PR intent** | **Yes** |

drift-guard is not a replacement for any of these. It fills the gap they all leave: the semantic gap between what a PR claims to do and what the code actually does.

---

## Why This Matters in 2026

> "A change like 'add a required field to a shared request schema' looked 'small' in the PR, but silently broke dozens of downstream services and jobs." — Salesforce Engineering Blog, 2025

With 42% of code now AI-generated:
- PRs move faster than human reviewers can track
- AI code often passes tests while violating unstated assumptions
- Silent scope creep in AI-assisted PRs has increased 38% YoY
- Intent drift is the #1 source of "this should have been caught in review" incidents

drift-guard gives every team a systematic, automated check that every PR's code matches its stated intent — before merge.

---

## Roadmap

**v0.1 (current)**
- CLI + Python API
- Claude-powered intent verification
- SQLite trace log
- GitHub Action support

**v0.2**
- GitLab CI integration
- PR comment posting (post drift report as PR comment automatically)
- Slack/Teams alerts for high drift scores
- Team dashboard (aggregated drift metrics)

**v0.3**
- Multi-clause dependency analysis
- Custom verification rules per file type
- Integration with Linear, Jira (link PR to ticket intent)
- OpenAI/Gemini model support

---

## Installation

```bash
pip install drift-guard

# Development install
git clone https://github.com/buildworld-ai/drift-guard
cd drift-guard
pip install -e ".[dev]"
```

**Requires:** Python 3.10+, an Anthropic API key, git in PATH.

---

## Contributing

drift-guard is MIT licensed. Contributions welcome.

```bash
git clone https://github.com/buildworld-ai/drift-guard
cd drift-guard
pip install -e ".[dev]"
pytest tests/
```

Please ensure new features include tests and that all PRs pass their own drift-guard verification.

---

## Biblical Foundation

> "I would not have known what sin was had it not been for the law." — Romans 7:7

The law in Romans 7 does not fix behavior — it makes violations *visible*. Before the law, sin was undetected. The law creates the measurement standard that exposes the gap between reality and requirement.

drift-guard operates on the same principle: without an explicit intent contract, code drift is invisible. The PR description becomes the law. The diff is measured against it. Violations that were previously invisible — passing tests, passing review — become detectable.

This is the **Romans Verification Pattern** — Law exposes; measurement enables; quality follows.

---

*Built by BuildWorld — Cycle 003*
*Pattern: PAT-036 — Romans 7:7 — The Law makes violations visible.*
