# BuildWorld Cycle Instructions
# Read by the buildworld-runner trigger. Ship real products.

REPO: BuildWorld (https://github.com/Rowusuduah/BuildWorld)

==== BUILDWORLD CYCLE ====

Read BuildWorld/settings.json to get current_cycle N.
Read BuildWorld/.Codex/memory/product-registry.md for the full product pipeline.
Read BuildWorld/.Codex/revenue/revenue-log.md for current revenue state.
Read BuildWorld/.Codex/logs/build-log.md for what has been built so far.
Read BuildWorld/.Codex/handoff.json plus the latest BibleWorld and GhanaWorld handoff.json files.
Read BuildWorld/.Codex/memory/microcore-deep-research-protocol.md for the core-cycle and evidence rules.
Read BuildWorld/.Codex/memory/benchmark-suite.md for the required benchmark checks.

Run cycle N+1. Choose cycle type based on what is needed:
- BUILD: write production code for the highest-priority unshipped product
- SHIP: deploy code to production, set up Stripe, create launch copy
- MEASURE: check analytics, revenue, user feedback, update metrics
- ITERATE: fix bugs, improve UX based on feedback
- NEW_PRODUCT: spec and start building the next product in pipeline

MICRO-CORE + DEEP RESEARCH RULES:
- One cycle, one core thesis.
- Start with the smallest correct kernel before adding polish, integrations, or launch copy.
- Promote only claims supported by current docs, competitor evidence, or working code.
- If the gap is weak, crowded, or stale, say so and do not force a build.
- Use [DEEP-RESEARCH] only when the Research Ledger includes current docs, competitors, pricing or distribution evidence, and contradiction notes.
- Every cycle report must include Core Thesis, Research Ledger, Benchmark Check, and Reproducibility Block.

CYCLE TYPE SELECTION RULES:
- If no product has been shipped yet, BUILD the highest-priority product.
- If code exists but is not deployed, SHIP.
- If deployed but no revenue data exists, MEASURE.
- If revenue exists but feedback indicates issues, ITERATE.
- If the current product is stable and generating revenue, start NEW_PRODUCT for the next item in pipeline.

BUILD CYCLE REQUIREMENTS:
- Write REAL, DEPLOYABLE code. Not stubs. Not skeletons. Production code.
- Shipping code goes in BuildWorld/products/[product-name]/.
- BuildWorld/.Codex/builds/[product-name]/ is for notes or design artifacts only, if needed.
- Use free-tier infrastructure only (Vercel, Supabase, Cloudflare, Stripe).
- Every file must work. Test it honestly. Would a user pay for this?
- Run at least 3 benchmark checks from benchmark-suite.md before calling a product ready.
- For web apps: complete pages, API routes, components, and styles.
- For packages: complete module, tests, README, and pyproject.toml or package.json.

SHIP CYCLE REQUIREMENTS:
- Deploy to a live URL.
- Set up Stripe payment link or checkout when revenue is part of the plan.
- Write launch copy: 1 tweet, 1 Product Hunt description, 1 email.
- Save all URLs and credentials info to build-log.md.

MEASURE CYCLE REQUIREMENTS:
- Check if the product URL is live.
- Review any available analytics.
- Calculate current MRR.
- Update revenue-log.md with any new entries.
- Decide whether to iterate on this product or move to the next.

Create these files each cycle:
1. BuildWorld/.Codex/cycles/cycle-[N+1]/cycle-report.md (400+ words)
   - Core Thesis
   - Research Ledger
   - Benchmark Check
   - Reproducibility Block
   - What was built, shipped, or measured
   - Product status update
   - Revenue update
   - Next cycle recommendation

2. BuildWorld/.Codex/cycles/cycle-[N+1]/code-review.md
   - The Reviewer agent's assessment of code quality
   - Security issues found
   - Production-readiness score (0-10)

3. BuildWorld/.Codex/cycles/cycle-[N+1]/launch-copy.md (for SHIP cycles)
   - Tweet thread (3-5 tweets)
   - Product Hunt title, tagline, and description
   - Landing page copy

Update these files:
- BuildWorld/settings.json: increment current_cycle, update metrics
- BuildWorld/world-status.json: add cycle_history entry, update revenue, update deployment_status
- BuildWorld/.Codex/memory/product-registry.md: update product statuses
- BuildWorld/.Codex/revenue/revenue-log.md: add any revenue entries
- BuildWorld/.Codex/logs/build-log.md: log what was built and where

THE THREE LAWS (enforced every cycle):
1. Ship weekly, every cycle must produce deployable progress.
2. Revenue within 30 days, every product needs a payment mechanism or an explicit adoption path.
3. Budget zero, free tiers only and no upfront capital.

INPUT FROM OTHER WORLDS:
- Read BibleWorld pattern-registry.md for Level 3 patterns that could become products.
- Read GhanaWorld opportunities.md for market signals and real-world constraints.
- These inputs inform what to build, not how to build.

Commit and push:
cd BuildWorld && git add -A && git commit -m "BuildWorld cycle [N+1] | [CYCLE_TYPE] | [PRODUCT] | revenue=$[MRR]" && git push origin master && cd ..

==== DONE ====
Ship. Ship. Ship. Revenue is oxygen.
