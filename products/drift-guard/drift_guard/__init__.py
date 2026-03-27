"""
drift-guard: PR Semantic Intent Verifier
========================================
Verifies that a pull request's actual code changes fulfill the stated intent
in the PR description / commit message. Catches silent contract violations
before they hit production.

Biblical Pattern: Romans 7:7 — "I would not have known what sin was had it
not been for the law." The law makes violations visible. drift-guard makes
the PR intent the law — then checks whether the code fulfills it.

Author: BuildWorld — Cycle 003
License: MIT
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class VerifyStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class IntentClause:
    """A single parsed clause from the PR intent statement."""
    text: str
    clause_type: str  # "adds", "removes", "modifies", "fixes", "ensures", "does_not"
    subject: str
    confidence: float = 1.0


@dataclass
class DiffHunk:
    """A single changed hunk from the git diff."""
    file_path: str
    old_start: int
    new_start: int
    lines_added: list[str] = field(default_factory=list)
    lines_removed: list[str] = field(default_factory=list)

    @property
    def is_pure_addition(self) -> bool:
        return len(self.lines_added) > 0 and len(self.lines_removed) == 0

    @property
    def is_pure_removal(self) -> bool:
        return len(self.lines_removed) > 0 and len(self.lines_added) == 0

    def summary(self) -> str:
        added = "\n".join(f"+ {l}" for l in self.lines_added[:20])
        removed = "\n".join(f"- {l}" for l in self.lines_removed[:20])
        return f"File: {self.file_path}\n{removed}\n{added}".strip()


@dataclass
class ClauseVerification:
    """Result of verifying a single intent clause against the diff."""
    clause: IntentClause
    status: VerifyStatus
    evidence: str          # What in the diff supports or contradicts this
    confidence: float      # 0.0 – 1.0
    explanation: str       # Human-readable explanation


@dataclass
class DriftReport:
    """Full drift verification report for a PR."""
    pr_title: str
    pr_description: str
    intent_summary: str
    clauses: list[IntentClause]
    verifications: list[ClauseVerification]
    overall_status: VerifyStatus
    overall_confidence: float
    files_changed: list[str]
    lines_added: int
    lines_removed: int
    drift_score: float        # 0.0 = perfect fulfillment, 1.0 = complete mismatch
    timestamp: str
    model_used: str
    commit_sha: str

    def passed(self) -> bool:
        return self.overall_status == VerifyStatus.PASS

    def to_dict(self) -> dict:
        return {
            "pr_title": self.pr_title,
            "overall_status": self.overall_status.value,
            "overall_confidence": self.overall_confidence,
            "drift_score": self.drift_score,
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "clauses": [
                {
                    "text": v.clause.text,
                    "status": v.status.value,
                    "confidence": v.confidence,
                    "explanation": v.explanation,
                    "evidence": v.evidence,
                }
                for v in self.verifications
            ],
            "intent_summary": self.intent_summary,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
            "commit_sha": self.commit_sha,
        }

    def to_markdown(self) -> str:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}
        lines = [
            f"## drift-guard Report",
            f"",
            f"**PR:** {self.pr_title}",
            f"**Status:** {icon.get(self.overall_status.value, '?')} {self.overall_status.value}",
            f"**Drift Score:** {self.drift_score:.2f} (0.00 = perfect, 1.00 = total mismatch)",
            f"**Confidence:** {self.overall_confidence:.0%}",
            f"**Changed:** {len(self.files_changed)} files, +{self.lines_added} -{self.lines_removed} lines",
            f"",
            f"### Intent Clauses",
            f"",
        ]
        for v in self.verifications:
            ic = icon.get(v.status.value, "?")
            lines.append(f"- {ic} **{v.clause.text}**")
            lines.append(f"  - {v.explanation}")
            if v.evidence:
                lines.append(f"  - *Evidence:* `{v.evidence[:120]}`")
        lines += [
            f"",
            f"### Intent Summary",
            f"",
            f"{self.intent_summary}",
            f"",
            f"---",
            f"*Generated by drift-guard at {self.timestamp} using {self.model_used}*",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Git utilities
# ---------------------------------------------------------------------------

def get_git_diff(base: str = "HEAD~1", head: str = "HEAD") -> str:
    """Return unified diff between base and head."""
    try:
        result = subprocess.run(
            ["git", "diff", f"{base}..{head}", "--unified=5"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git diff failed: {e.stderr}") from e


def get_current_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def parse_diff(diff_text: str) -> list[DiffHunk]:
    """Parse a unified diff into DiffHunk objects."""
    hunks: list[DiffHunk] = []
    current_file = ""
    current_hunk: Optional[DiffHunk] = None

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            match = re.search(r"-(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?", line)
            if match:
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = DiffHunk(
                    file_path=current_file,
                    old_start=int(match.group(1)),
                    new_start=int(match.group(2)),
                )
        elif current_hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.lines_added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.lines_removed.append(line[1:])

    if current_hunk:
        hunks.append(current_hunk)

    return hunks


def diff_stats(diff_text: str) -> tuple[list[str], int, int]:
    """Return (files_changed, lines_added, lines_removed) from a diff."""
    files: set[str] = set()
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:])
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return sorted(files), added, removed


# ---------------------------------------------------------------------------
# Intent parser
# ---------------------------------------------------------------------------

_INTENT_PATTERNS = {
    "adds":     [r"\badd(s|ing|ed)?\b", r"\bintroduc(e|es|ing|ed)\b", r"\bcreate(s|d)?\b", r"\bimplement(s|ed|ing)?\b"],
    "removes":  [r"\bremov(e|es|ing|ed)\b", r"\bdelet(e|es|ing|ed)\b", r"\bdrop(s|ped|ping)?\b"],
    "modifies": [r"\bupdat(e|es|ing|ed)\b", r"\bmodif(y|ies|ied|ying)\b", r"\brefactor(s|ed|ing)?\b", r"\bimprove(s|d)?\b"],
    "fixes":    [r"\bfix(es|ed|ing)?\b", r"\bresolv(e|es|ing|ed)\b", r"\bpatch(es|ed)?\b", r"\bcorrect(s|ed)?\b"],
    "ensures":  [r"\bensur(e|es|ing|ed)\b", r"\bguarantee(s|d)?\b", r"\bvalidat(e|es|ing|ed)\b"],
    "does_not": [r"\bdoes not\b", r"\bshould not\b", r"\bwon[''`]?t\b", r"\bnever\b"],
}


def parse_intent(description: str) -> list[IntentClause]:
    """Parse a PR description into structured intent clauses."""
    clauses: list[IntentClause] = []
    raw_clauses = re.split(r"[.!?\n]|(?:^|\n)\s*[-*•]\s*", description)
    for raw in raw_clauses:
        raw = raw.strip()
        if len(raw) < 8:
            continue
        clause_type = "modifies"  # default
        for ctype, patterns in _INTENT_PATTERNS.items():
            if any(re.search(p, raw, re.IGNORECASE) for p in patterns):
                clause_type = ctype
                break
        subject_match = re.search(
            r"(?:the |a |an )?([A-Za-z_][A-Za-z0-9_ ]{2,40}?)(?:\s+(?:function|method|class|module|endpoint|route|field|parameter|type|check|validation|test|feature))?(?:\s|$)",
            raw, re.IGNORECASE
        )
        subject = subject_match.group(1).strip() if subject_match else raw[:40]
        clauses.append(IntentClause(text=raw, clause_type=clause_type, subject=subject))
    return clauses[:12]


# ---------------------------------------------------------------------------
# LLM verifier
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = """\
You are drift-guard, a semantic PR intent verifier. Your job is to determine
whether a set of code changes (git diff) actually fulfills the stated intent
of a pull request.

You will be given:
1. A PR title and description (the intent)
2. A structured list of intent clauses extracted from the description
3. A summary of the git diff (what actually changed)

For EACH clause, you must output a JSON object with:
- clause_index: integer (0-based)
- status: "PASS" | "FAIL" | "WARN" | "SKIP"
  * PASS = the diff clearly fulfills this clause
  * FAIL = the diff contradicts or ignores this clause
  * WARN = the diff partially fulfills this clause or there is ambiguity
  * SKIP = cannot determine from diff alone (e.g., runtime behavior)
- confidence: float 0.0-1.0
- evidence: a SHORT specific quote from the diff (max 120 chars) that is the
  primary evidence for your verdict, or "" if none found
- explanation: 1-2 sentence human-readable explanation

Also output an overall summary:
- overall_status: "PASS" | "FAIL" | "WARN"
  * PASS if ALL clauses PASS or SKIP (no FAILs, no WARNs)
  * WARN if any clause is WARN but no FAILs
  * FAIL if any clause FAILs
- overall_confidence: float 0.0-1.0 (average weighted confidence)
- drift_score: float 0.0-1.0
  * 0.0 = all clauses pass, perfect fulfillment
  * 1.0 = all clauses fail, complete mismatch
- intent_summary: 2-3 sentence synthesis of what the PR claims vs. what the
  diff shows. Be specific. Be honest. If there is a gap, name it.

Respond with ONLY valid JSON in this exact schema:
{
  "clauses": [
    {"clause_index": 0, "status": "PASS", "confidence": 0.9,
     "evidence": "...", "explanation": "..."}
  ],
  "overall_status": "PASS",
  "overall_confidence": 0.88,
  "drift_score": 0.05,
  "intent_summary": "..."
}

Rules:
- Be honest. If the diff doesn't show something, say so.
- Do not assume things happened outside the diff.
- A PR that only mentions fixing a bug but also silently refactors unrelated
  code should be flagged as WARN, not PASS.
- Pay special attention to schema changes, API contract changes, and removals
  that are not mentioned in the PR description.
"""

# Default model — fastest Haiku, optimized for binary verdicts
DEFAULT_MODEL = os.environ.get("DRIFT_GUARD_MODEL", "claude-haiku-4-5-20251001")


def verify_with_llm(
    pr_title: str,
    pr_description: str,
    clauses: list[IntentClause],
    diff_text: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> dict:
    """Call Claude to verify intent against diff. Returns parsed JSON response."""
    if anthropic is None:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    diff_truncated = diff_text[:8000]
    if len(diff_text) > 8000:
        diff_truncated += f"\n... [diff truncated, {len(diff_text) - 8000} chars omitted] ..."

    clauses_text = "\n".join(
        f"{i}. [{c.clause_type.upper()}] {c.text}"
        for i, c in enumerate(clauses)
    )

    user_content = f"""PR TITLE: {pr_title}

PR DESCRIPTION:
{pr_description}

EXTRACTED INTENT CLAUSES:
{clauses_text}

GIT DIFF:
{diff_truncated}

Verify each clause against the diff and return the JSON report."""

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_VERIFY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    content = message.content[0].text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return json.loads(content)


# ---------------------------------------------------------------------------
# SQLite trace log
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    return Path(os.environ.get("DRIFT_GUARD_DB", ".drift-guard.db"))


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            pr_title TEXT,
            overall_status TEXT NOT NULL,
            drift_score REAL NOT NULL,
            overall_confidence REAL NOT NULL,
            files_changed TEXT,
            lines_added INTEGER,
            lines_removed INTEGER,
            report_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_report(report: DriftReport, db_path: Optional[Path] = None) -> None:
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO drift_reports
            (timestamp, commit_sha, pr_title, overall_status, drift_score,
             overall_confidence, files_changed, lines_added, lines_removed, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report.timestamp,
        report.commit_sha,
        report.pr_title,
        report.overall_status.value,
        report.drift_score,
        report.overall_confidence,
        json.dumps(report.files_changed),
        report.lines_added,
        report.lines_removed,
        json.dumps(report.to_dict()),
    ))
    conn.commit()
    conn.close()


def load_recent_reports(n: int = 10, db_path: Optional[Path] = None) -> list[dict]:
    if db_path is None:
        db_path = get_db_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT timestamp, commit_sha, pr_title, overall_status, drift_score
        FROM drift_reports ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [
        {"timestamp": r[0], "commit": r[1], "pr_title": r[2],
         "status": r[3], "drift_score": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Core verify function (main API)
# ---------------------------------------------------------------------------

def verify(
    pr_title: str,
    pr_description: str,
    diff: Optional[str] = None,
    base: str = "HEAD~1",
    head: str = "HEAD",
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    save: bool = True,
) -> DriftReport:
    """
    Main API. Verifies that a PR's code changes fulfill its stated intent.

    Args:
        pr_title:        The PR title or commit message subject.
        pr_description:  The PR body / commit message body.
        diff:            Optional pre-computed diff. If None, uses git diff.
        base:            Git ref for base of diff (default: HEAD~1).
        head:            Git ref for head of diff (default: HEAD).
        model:           Anthropic model to use for semantic verification.
        api_key:         Anthropic API key (falls back to ANTHROPIC_API_KEY env).
        save:            Whether to persist report to SQLite trace log.

    Returns:
        DriftReport with full verification results.
    """
    if diff is None:
        diff = get_git_diff(base, head)

    commit_sha = get_current_commit_sha()
    files_changed, lines_added, lines_removed = diff_stats(diff)

    full_intent = f"{pr_title}\n\n{pr_description}"
    clauses = parse_intent(full_intent)

    if not clauses:
        return DriftReport(
            pr_title=pr_title,
            pr_description=pr_description,
            intent_summary="No parseable intent clauses found in PR description.",
            clauses=[],
            verifications=[],
            overall_status=VerifyStatus.SKIP,
            overall_confidence=0.0,
            files_changed=files_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
            drift_score=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used=model,
            commit_sha=commit_sha,
        )

    llm_result = verify_with_llm(
        pr_title=pr_title,
        pr_description=pr_description,
        clauses=clauses,
        diff_text=diff,
        model=model,
        api_key=api_key,
    )

    verifications: list[ClauseVerification] = []
    for cv in llm_result.get("clauses", []):
        idx = cv.get("clause_index", 0)
        if idx < len(clauses):
            clause = clauses[idx]
        else:
            clause = IntentClause(text="unknown", clause_type="modifies", subject="unknown")
        verifications.append(ClauseVerification(
            clause=clause,
            status=VerifyStatus(cv.get("status", "SKIP")),
            evidence=cv.get("evidence", ""),
            confidence=float(cv.get("confidence", 0.5)),
            explanation=cv.get("explanation", ""),
        ))

    report = DriftReport(
        pr_title=pr_title,
        pr_description=pr_description,
        intent_summary=llm_result.get("intent_summary", ""),
        clauses=clauses,
        verifications=verifications,
        overall_status=VerifyStatus(llm_result.get("overall_status", "WARN")),
        overall_confidence=float(llm_result.get("overall_confidence", 0.5)),
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        drift_score=float(llm_result.get("drift_score", 0.5)),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_used=model,
        commit_sha=commit_sha,
    )

    if save:
        save_report(report)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="drift-guard",
        description="Verify that a PR's code changes fulfill its stated intent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify_cmd = sub.add_parser("verify", help="Run a verification")
    verify_cmd.add_argument("--title", required=True, help="PR title or commit subject")
    verify_cmd.add_argument("--description", required=True, help="PR body or commit message body")
    verify_cmd.add_argument("--base", default="HEAD~1", help="Base git ref (default: HEAD~1)")
    verify_cmd.add_argument("--head", default="HEAD", help="Head git ref (default: HEAD)")
    verify_cmd.add_argument("--diff-file", help="Path to pre-computed diff file")
    verify_cmd.add_argument("--model", default=DEFAULT_MODEL)
    verify_cmd.add_argument("--format", choices=["json", "markdown", "text"], default="text")
    verify_cmd.add_argument("--no-save", action="store_true", help="Do not persist to SQLite")
    verify_cmd.add_argument("--threshold", type=float, default=0.3,
                            help="Drift score threshold — fail CI if score exceeds this (default 0.3)")

    history_cmd = sub.add_parser("history", help="Show recent verification history")
    history_cmd.add_argument("--n", type=int, default=10)

    args = parser.parse_args(argv)

    if args.command == "verify":
        diff = None
        if args.diff_file:
            with open(args.diff_file) as f:
                diff = f.read()

        report = verify(
            pr_title=args.title,
            pr_description=args.description,
            diff=diff,
            base=args.base,
            head=args.head,
            model=args.model,
            save=not args.no_save,
        )

        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        elif args.format == "markdown":
            print(report.to_markdown())
        else:
            icon = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}
            print(f"\ndrift-guard result: {icon[report.overall_status.value]}")
            print(f"Drift score: {report.drift_score:.2f} (threshold: {args.threshold})")
            print(f"Confidence: {report.overall_confidence:.0%}")
            print(f"\nIntent summary:\n  {report.intent_summary}")
            print(f"\nClauses ({len(report.verifications)}):")
            for v in report.verifications:
                mark = {"PASS": "[+]", "FAIL": "[X]", "WARN": "[~]", "SKIP": "[-]"}
                print(f"  {mark.get(v.status.value, '?')} {v.clause.text[:80]}")
                print(f"     -> {v.explanation[:100]}")

        if report.drift_score > args.threshold:
            print(f"\ndrift-guard: GATE FAILED — drift score {report.drift_score:.2f} exceeds threshold {args.threshold}")
            return 1
        return 0

    elif args.command == "history":
        rows = load_recent_reports(args.n)
        if not rows:
            print("No drift-guard history found.")
            return 0
        print(f"{'Timestamp':<28} {'Commit':<10} {'Status':<6} {'Drift':>7}  PR Title")
        print("-" * 90)
        for r in rows:
            print(f"{r['timestamp']:<28} {r['commit']:<10} {r['status']:<6} {r['drift_score']:>7.2f}  {r['pr_title'][:40]}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
