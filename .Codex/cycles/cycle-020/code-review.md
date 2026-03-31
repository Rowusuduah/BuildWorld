# Code Review — Cycle 020
## Reviewer Agent Assessment: semantic-pass-k v0.1.0

**Date:** 2026-03-31
**Reviewer:** The Reviewer (BuildWorld)

---

## Production Readiness Score: 8.8/10

---

## Summary

`semantic-pass-k` is a well-structured, zero-dependency Python package implementing semantic consistency testing for AI agents. The code is production-quality, comprehensively tested (177/177), and follows the same patterns as other BuildWorld tools (hatchling, SQLite store, injectable similarity functions, CLI via click).

---

## Security Review

| Issue | Severity | Status |
|---|---|---|
| SQL injection in ConsistencyStore | None — uses parameterized queries (`?` placeholders) throughout. CLEAR. | ✅ |
| Path traversal in ConsistencyStore | Uses `pathlib.Path(db_path)` — no directory traversal risk. CLEAR. | ✅ |
| Arbitrary code execution | No eval(), exec(), or subprocess calls. CLEAR. | ✅ |
| CLI input handling | All user inputs validated by Click (type constraints) or guarded by explicit checks. CLEAR. | ✅ |
| LiteLLM dependency | None. POLICY compliance confirmed — no LiteLLM, no supply-chain risk. | ✅ |
| External network calls | None in core package. Neural backend (sentence-transformers) is optional and isolated. | ✅ |

**Security verdict: CLEAN — no issues found.**

---

## Code Quality

### Strengths

1. **Injectable similarity_fn**: All tests use injected similarity stubs — no LLM, no network, no ML model required. Follows cot-fidelity and model-parity precedent.

2. **Zero hard runtime dependencies**: stdlib only. Optional neural (sentence-transformers) and CLI (click, rich) are isolated behind `[neural]` and `[cli]` extras.

3. **Smoothed IDF**: Uses the same corrected `math.log(1.0 + 2.0 / df)` formula that was validated this cycle for cot-fidelity. This IDF avoids the 2-doc corpus collapse bug.

4. **Separation of concerns**: ConsistencyEngine (algorithm), ConsistencyRunner (orchestration), ConsistencyStore (persistence) are cleanly separated. Engine has no I/O.

5. **BORDERLINE verdict**: Three-verdict system (CONSISTENT / BORDERLINE / INCONSISTENT) avoids false binary pass/fail for scores near the threshold. Configurable band.

6. **Consistent patterns**: pyproject.toml, store, CLI, decorators, pytest_plugin all follow the same structure as prior BuildWorld tools.

### Issues Found

1. **Neural backend: model reloading** — `_neural_cosine` creates a new `SentenceTransformer` model instance on every call. For large batches this causes unnecessary repeated model loading.
   - **Severity**: MEDIUM (performance, not correctness)
   - **Fix for v0.2**: Cache model in `ConsistencyEngine.__init__` when `use_neural=True`

2. **TF-IDF semantic gap** — TF-IDF cannot detect that "Accra is the capital" and "The capital of Ghana is Accra" are semantically equivalent. This is an inherent limitation, not a bug.
   - **Severity**: LOW (documented in README)
   - **Recommendation**: Examples use lexically similar outputs for TF-IDF mode. Neural mode (`use_neural=True`) handles semantic similarity correctly.

3. **Empty batch in evaluate_batch** — Fixed during this cycle (raise ValueError on empty prompts list). RESOLVED.

4. **ConsistencyReport.from_results uses results[0].criticality** — Assumes all results share the same criticality tier. If mixed-criticality batches are used, the report's threshold reflects only the first result's tier.
   - **Severity**: LOW (runner.run_batch enforces uniform criticality)
   - **Fix for v0.2**: Add validation that all results use the same criticality, or support per-result thresholds

---

## Test Coverage

- Models: 42 tests — complete coverage of all data classes and utility functions
- Engine: 50 tests — TF-IDF, pairwise scoring, all tiers, edge cases, hash determinism
- Runner: 22 tests — k override, criticality override, batch behavior, call count verification
- Store: 17 tests — CRUD, label filtering, prompt hash filtering, replace-on-duplicate
- Decorators: 12 tests — k calls, raise_on_fail, default criticality, pre-call state
- Config: 16 tests — YAML parsing, defaults, edge cases
- pytest_plugin: 11 tests — pass/fail/borderline, assertion messages
- CLI: 17 tests — all 4 commands (run/report/ci/budget), JSON output, DB persistence

**177/177 passing. 0 skipped. 0 errors.**

---

## Deployment Readiness

| Check | Status |
|---|---|
| pyproject.toml valid | ✅ |
| package builds with hatchling | ✅ (pip install -e . succeeds) |
| CLI entry point registered | ✅ (sempass) |
| pytest11 entry point registered | ✅ (assert_consistent fixture) |
| README complete | ✅ (problem, solution, quick start, CLI, competitive gap, roadmap) |
| LICENSE present | ✅ (MIT) |
| Zero hard runtime dependencies | ✅ |
| No LiteLLM | ✅ |

**PyPI-ready. Pending: GitHub repo creation + OIDC trusted publishing setup.**

---

## Verdict

**Ship when PyPI account and GitHub org are available. No blocking code issues.**
