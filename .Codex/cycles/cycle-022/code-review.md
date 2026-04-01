# Cycle 022 — Code Review
## The Reviewer's Assessment

**Date:** 2026-04-01
**Cycle Type:** SHIP (infrastructure only — no new product code)
**Reviewer:** The Reviewer Agent

---

## Scope of Review

This cycle produced:
- 17 GitHub Actions YAML workflows for PyPI publishing
- 1 GitHub Actions YAML workflow for Vercel deployment
- 1 DEPLOY.md playbook
- 1 launch copy document

No new Python code was written. All 1,473 existing tests remain unchanged and passing.

---

## Workflow Security Review

### OIDC Trusted Publishing — APPROVED

All 17 Python package workflows use OIDC trusted publishing:
```yaml
permissions:
  id-token: write
environment:
  name: pypi
  url: https://pypi.org/p/<package-name>
```

This is **more secure than API tokens** because:
- No long-lived credentials stored in GitHub Secrets
- Short-lived tokens issued per workflow run
- PyPI audit log records which workflow published what
- Token cannot be leaked via `git log` or env var inspection

**Risk: NONE** — this is the current PyPA-recommended approach.

### Workflow Dependency Pinning — FLAG (non-blocking)

Workflows use floating refs:
```yaml
uses: actions/checkout@v4
uses: actions/setup-python@v5
uses: pypa/gh-action-pypi-publish@release/v1
```

Best practice is SHA pinning (e.g., `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683`). However:
- `@v4`, `@v5` are maintained major version tags — not floating to arbitrary commits
- `@release/v1` is the pypa-recommended ref for gh-action-pypi-publish
- SHA pinning is overkill for an indie project at $0 MRR

**Verdict:** Acceptable at current stage. Pin to SHA when the first product reaches 1,000 installs.

### Workflow Permissions — APPROVED

Top-level `permissions: contents: read` limits blast radius. The `id-token: write` permission is scoped only to the `publish` job. No `write-all` or `*` permissions anywhere.

### Secret Exposure — APPROVED

No secrets referenced in Python package workflows. The Vercel workflow requires three secrets (VERCEL_TOKEN, VERCEL_ORG_ID, VERCEL_PROJECT_ID) — these are standard Vercel deployment secrets, documented in DEPLOY.md.

---

## model-parity CI Variation — APPROVED

model-parity uses:
```yaml
run: pip install pytest pytest-cov pyyaml
```
instead of `pip install -e ".[dev]"`.

**Reason:** model-parity's `[dev]` extras include the full Anthropic and OpenAI SDKs. Installing these in CI is unnecessary overhead (tests mock all API calls). This is a correct minimal install.

**Risk: LOW** — if a new test ever uses a real SDK call, this CI step would fail loudly. That's the right behavior.

---

## DEPLOY.md Review

### Completeness — PASS
All 10 steps are present. No step is omitted. The PyPI OIDC setup (Step 5) is the trickiest — it correctly points to `pypi.org/manage/account/publishing/` and lists all required fields.

### Accuracy — PASS
The batch script for pushing all repos uses correct path syntax. The tag-and-publish flow is correct (`git tag v0.1.0 && git push origin v0.1.0`).

### Safety — MINOR FLAG
The batch push script uses `for product in ...` with `git init` + `git remote add`. If run twice, `git remote add origin` will fail (remote already exists). Non-destructive, just noisy. User should `git remote set-url origin ...` on second run.

### inject-lock vs prompt-lock — FLAG (non-blocking)
The prompt-lock product directory is named `prompt-lock` but the PyPI package name is `inject-lock` (per pyproject.toml). The DEPLOY.md table correctly shows this discrepancy. The GitHub repo should be named `prompt-lock` (matches directory) while the PyPI project name is `inject-lock`. Both are documented.

---

## Launch Copy Review

### Accuracy — PASS
The HN post for semantic-pass-k accurately describes the algorithm and correctly distinguishes from AgentAssay, SelfCheckGPT, ContextCite. No false claims.

### Tone — PASS
HN posts are technical, honest, and specific. No "revolutionary" or "AI-powered" hype. Posts lead with the problem, show code immediately, and explain the competitive gap factually.

### Pricing Claims — PASS
No pricing is claimed for the open-source tools. Stripe tier ($49/month Pro, $299/month Enterprise) is marked as "post-traction" — not claimed to be live.

---

## Production Readiness Score

| Component | Score | Notes |
|-----------|-------|-------|
| Workflow security | 9/10 | OIDC correct; SHA pinning deferred |
| Workflow correctness | 10/10 | All 18 valid, matching pypi package names |
| DEPLOY.md completeness | 9/10 | Minor: batch script second-run safety |
| Launch copy accuracy | 10/10 | No false claims |
| Overall | **9.5/10** | Ship it |

---

## Blocking Issues

**NONE.**

The cycle is clear to ship. No security vulnerabilities. No incorrect claims. The manual deployment steps are unavoidable (PyPI account creation cannot be automated).

---

## Post-Deploy Recommendations

1. After first publish, run `pip install semantic-pass-k` from a clean venv to verify end-to-end
2. Add `SECURITY.md` to top 3 packages before Product Hunt launch (shows maturity)
3. Pin workflow SHA after 1,000 installs on any tool
4. Add `CHANGELOG.md` before v0.2.0

*— The Reviewer*
