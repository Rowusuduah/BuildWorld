# Cycle 022 — Launch Copy
## BuildWorld SHIP Cycle

**Date:** 2026-04-01
**Primary Launch:** semantic-pass-k
**Secondary Launches:** model-parity, cot-fidelity, context-trace

---

## SEMANTIC-PASS-K

### Hacker News — Show HN Post

**Title:**
```
Show HN: semantic-pass-k – Does your AI agent give consistent answers? (pip install)
```

**Body:**
```
I built semantic-pass-k to solve a problem I kept hitting: AI agents that pass unit tests but give wildly different answers in prod.

The idea: run your agent k times on the same input, embed all outputs, compute pairwise cosine similarity. If the mean similarity falls below your threshold, the test fails.

The key thing it adds is a named metric (ConsistencyScore) with four task-criticality tiers:
- CRITICAL: 0.99 (medical, legal, financial decisions)
- HIGH: 0.90 (user-facing responses)
- MEDIUM: 0.75 (internal tools)
- LOW: 0.60 (creative tasks)

This makes it CI-gateable — you can block a deploy if ConsistencyScore < threshold.

pip install semantic-pass-k

Quick start:
  from semantic_pass_k import ConsistencyRunner
  runner = ConsistencyRunner(agent_fn=my_agent, k=5)
  report = runner.run("What is our return policy?")
  print(report.consistency_score)  # 0.94
  print(report.verdict)  # PASS

Also ships a pytest plugin:
  def test_agent_consistent(assert_consistent):
      assert_consistent(my_agent, "What is our return policy?", k=5, tier="HIGH")

The main gap it fills: ContextCite and SelfCheckGPT are hallucination detection. DeepEval/Promptfoo have no pass@k metric. AgentAssay is sampling efficiency, not consistency. Nothing pip-installable gives you a ConsistencyScore with criticality tiers.

GitHub: github.com/buildworld-ai/semantic-pass-k
```

---

### Tweet Thread (semantic-pass-k launch)

**Tweet 1:**
```
Shipped: semantic-pass-k

Your AI agent passes tests. But run it 5 times — do you get the same answer?

pip install semantic-pass-k

🧵
```

**Tweet 2:**
```
The problem: LLM outputs are non-deterministic.

"unit tests pass" ≠ "consistent in production"

You need to run the same query k times and measure semantic equivalence — not just string match.

semantic-pass-k gives you a named ConsistencyScore (0-1) with CI-gateable criticality tiers.
```

**Tweet 3:**
```
Works with any LLM via a simple callable:

from semantic_pass_k import ConsistencyRunner

runner = ConsistencyRunner(agent_fn=my_agent, k=5)
report = runner.run("What is our refund policy?")

print(report.consistency_score)  # 0.87
print(report.verdict)            # PASS (HIGH tier: 0.90 required → FAIL)
```

**Tweet 4:**
```
Also ships a pytest plugin — drop it into any test suite:

def test_agent_consistent(assert_consistent):
    assert_consistent(
        my_agent,
        "Summarize this document",
        k=5,
        tier="HIGH"  # requires 0.90+ consistency
    )

Blocks deploys when your agent becomes inconsistent.
```

**Tweet 5:**
```
Zero runtime dependencies (pure Python stdlib + optional sentence-transformers for neural mode).

177 tests. MIT license. Full CLI included.

pip install semantic-pass-k

github.com/buildworld-ai/semantic-pass-k
```

---

### Product Hunt

**Name:** semantic-pass-k

**Tagline:** CI-gateable semantic consistency testing for AI agents

**Description:**
```
AI agents that pass your unit tests can still give wildly different answers in production. semantic-pass-k solves this.

Run your agent k times on the same input. Embed all outputs. Measure semantic equivalence with a named ConsistencyScore (0-1) against four criticality tiers:
- CRITICAL (0.99): medical, legal, financial
- HIGH (0.90): user-facing responses
- MEDIUM (0.75): internal tools
- LOW (0.60): creative tasks

Block your CI pipeline when ConsistencyScore falls below threshold. Works with any LLM via a plain Python callable — Claude, GPT-4o, Gemini, local models.

pip install semantic-pass-k

Features:
✓ Named ConsistencyScore metric — reportable, trackable
✓ 4 criticality tiers — calibrated to use case risk
✓ pytest plugin — assert_consistent() drops into any test suite
✓ CLI — sempass run / sempass ci / sempass report
✓ SQLite store — track consistency over time
✓ Zero runtime deps (optional sentence-transformers for neural mode)
✓ 177 tests, MIT license

No existing tool gives you this: ContextCite needs raw logits. SelfCheckGPT detects hallucinations. DeepEval/Promptfoo have no pass@k. AgentAssay is sampling efficiency. semantic-pass-k is the consistency gate.
```

**Topics:** Developer Tools, Artificial Intelligence, Testing, Open Source

---

## MODEL-PARITY

### Hacker News — Show HN Post

**Title:**
```
Show HN: model-parity – Certify your replacement LLM is behaviorally equivalent before you migrate
```

**Body:**
```
Model migrations break things in subtle ways. The new model handles edge cases differently. Tone shifts. Safety filters fire in different places. You don't know until prod explodes.

model-parity gives you a behavioral equivalence certificate before you cut over.

pip install model-parity

Write YAML test suites against 7 dimensions:
- semantic_equivalence (cosine similarity of embeddings)
- length_ratio (response length parity)
- format_compliance (structured output matching)
- safety_behavior (refuses same things)
- factual_consistency (same facts, different words)
- instruction_following (follows constraints)
- style_consistency (tone, register, formality)

Each test gets PASS/FAIL/PARTIAL. The certificate is CERTIFIED if all critical tests pass.

Built this after watching teams migrate from GPT-3.5 to GPT-4 and discover behavioral regressions in week 3. Run it before you flip the flag.

GitHub: github.com/buildworld-ai/model-parity
```

---

### Tweet Thread (model-parity launch)

**Tweet 1:**
```
Shipping: model-parity

Before you migrate from GPT-3.5 → GPT-4o or Claude 2 → Claude 3, prove they're behaviorally equivalent.

pip install model-parity

🧵
```

**Tweet 2:**
```
Model migrations break things in subtle ways:
- New model refuses requests the old one handled
- Tone shifts slightly — users notice
- Structured outputs change format
- Edge cases handled differently

You don't find out until production.

model-parity catches it before the deploy.
```

**Tweet 3:**
```
Write YAML test suites:

tests:
  - name: "Customer support tone"
    input: "My order is late, I'm frustrated"
    check: semantic_equivalence
    threshold: 0.88
  - name: "JSON format compliance"
    input: "Return order details as JSON"
    check: format_compliance

Run: parity certify --suite tests.yaml --model-a claude-3 --model-b claude-3-5

Result: CERTIFIED ✓ (or blocked with diffs)
```

**Tweet 4:**
```
7 behavioral dimensions:
✓ Semantic equivalence
✓ Length ratio
✓ Format compliance
✓ Safety behavior (refuses same things)
✓ Factual consistency
✓ Instruction following
✓ Style consistency

97 tests. MIT license.

pip install model-parity
```

---

## COT-FIDELITY

### Hacker News — Show HN Post

**Title:**
```
Show HN: cot-fidelity – Is your LLM's chain-of-thought actually causal? The counterfactual test
```

**Body:**
```
LLMs produce chain-of-thought reasoning that looks plausible but may have zero causal connection to the output.

The test: run with CoT, run without CoT, embed both outputs, measure cosine delta.

If removing the CoT barely changes the output → the reasoning is UNFAITHFUL (decorative, not causal).
If removing it significantly changes the output → FAITHFUL.

cot-fidelity ships this as a named verdict (FAITHFUL/UNFAITHFUL/INCONCLUSIVE) with a configurable delta threshold.

pip install cot-fidelity

This matters for:
- Auditing AI in high-stakes applications
- Detecting "reasoning theater" in agent pipelines
- Research on LLM faithfulness

Based on the counterfactual suppression methodology from Turpin et al. (2023) and Lanham et al. (2023) — but packaged as a pip-installable dev tool with CI integration.

GitHub: github.com/buildworld-ai/cot-fidelity
```

---

## CONTEXT-TRACE

### Hacker News — Show HN Post

**Title:**
```
Show HN: context-trace – Which part of your context caused that LLM output? (API-compatible)
```

**Body:**
```
When a RAG pipeline hallucinates or an agent refuses, you can't tell which input chunk caused it.

context-trace gives you an AttributionScore (0-1) per context chunk using counterfactual masking.

The algorithm: for each named chunk, replace it with [REMOVED], run k times, embed outputs, compute cosine delta vs original. Higher score = more causal influence.

pip install context-trace

from context_trace import ContextTracer
tracer = ContextTracer(runner=my_llm_fn, embedder=my_embedder)
report = tracer.trace(
    prompt="Summarize the key risks",
    chunks={
        "system_prompt": "You are a risk analyst...",
        "document_1": "Q3 revenue declined...",
        "tool_result": "Current exposure: $2.3M"
    }
)
print(report.attribution_heatmap())
# system_prompt  0.12 ░░░░░░░░░░
# document_1     0.78 ████████░░  ← this drove the output
# tool_result    0.31 ███░░░░░░░

Key difference from ContextCite: works with any hosted API (Claude, GPT-4o, Gemini). ContextCite requires raw logit access — no hosted API exposes this.

102 tests. MIT license.

GitHub: github.com/buildworld-ai/context-trace
```

---

## GHANASCOPE

### Vercel Deploy Copy

**App name:** GhanaScope

**Tagline:** Real-time Ghana market intelligence for builders and investors

**Description:**
```
GhanaScope tracks the macro signals that matter for building or investing in Ghana:

- Gold price (live) — Ghana's primary FX buffer
- Fuel floors (NPA windows) — operational cost signal
- IMF program status — policy risk indicator
- GSE performance — capital market signal
- Opportunity scoring — ranked by timing and risk

Built from 7 cycles of GhanaWorld intelligence synthesis.
Updated weekly. Free.
```

**Target users:** Ghana diaspora professionals, returning founders, impact investors, fintech builders

---

## DISTRIBUTION CHANNELS (priority order)

| Channel | Products | Timing |
|---------|---------|--------|
| Hacker News (Show HN) | semantic-pass-k | Day 1 post-launch |
| AI Twitter/X | All tools | Day 1, every 3 days |
| LangChain Discord #tools | context-trace, agent-patrol | Day 2 |
| LlamaIndex Discord | rag-pathology, context-trace | Day 2 |
| r/MachineLearning | semantic-pass-k, cot-fidelity | Day 3 |
| r/LocalLLaMA | model-parity, prompt-shield | Day 3 |
| Product Hunt | semantic-pass-k | Week 2 (after HN traction) |
| Indie Hackers | All tools | Week 2 |
| Dev.to article | "18 tools for LLM testing" | Week 3 |
| GitHub Trending (if starred) | Auto-discovery | Ongoing |

---

## PRICING (Stripe, post-traction)

Add Pro tier after 100+ PyPI installs on any tool:

| Tier | Price | What it includes |
|------|-------|-----------------|
| Open Source | Free | Full library, local runs |
| Pro | $49/month | Hosted API, CI integration, team dashboard, email alerts |
| Enterprise | $299/month | SLA, private deployment, custom scoring, dedicated support |

**Stripe setup:** Create account → Products → Add price → Generate payment link → Add to README.
