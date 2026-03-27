# BuildWorld Cycle Instructions
# Read by the buildworld-runner trigger. Ship or die.

REPO: BuildWorld (https://github.com/Rowusuduah/BuildWorld)

==== BUILDWORLD CYCLE ====

Read BuildWorld/settings.json to get current_cycle N.
Read BuildWorld/.Codex/memory/product-registry.md for the full product pipeline.
Read BuildWorld/.Codex/revenue/revenue-log.md for current revenue state.
Read BuildWorld/.Codex/logs/build-log.md for what has been built so far.

Run cycle N+1. Choose cycle type based on what's needed:
- BUILD: Write production code for the highest-priority unshipped product
- SHIP: Deploy code to production, set up Stripe, create launch copy
- MEASURE: Check analytics, revenue, user feedback, update metrics
- ITERATE: Fix bugs, improve UX based on feedback
- NEW_PRODUCT: Spec and start building the next product in pipeline

CYCLE TYPE SELECTION RULES:
- If no product has been shipped yet → BUILD the highest priority product (PROD-001 LogosSchema)
- If code exists but isn't deployed → SHIP
- If deployed but no revenue data → MEASURE
- If revenue exists but feedback indicates issues → ITERATE
- If current product is stable and generating revenue → NEW_PRODUCT for next in pipeline

BUILD CYCLE REQUIREMENTS:
- Write REAL, DEPLOYABLE code. Not stubs. Not skeletons. Production code.
- All code goes in BuildWorld/.Codex/builds/[product-name]/
- Use free-tier infrastructure only (Vercel, Supabase, Cloudflare, Stripe)
- Every file must work. Test it mentally. Would a user pay for this?
- For web apps: complete pages, API routes, components, styles
- For packages: complete module, tests, README, setup.py/package.json

SHIP CYCLE REQUIREMENTS:
- Deploy to a live URL
- Set up Stripe payment link or checkout
- Write launch copy: 1 tweet, 1 Product Hunt description, 1 email
- Save all URLs and credentials info to build-log.md

MEASURE CYCLE REQUIREMENTS:
- Check if the product URL is live
- Review any available analytics
- Calculate current MRR
- Update revenue-log.md with any new entries
- Decide: iterate on this product or move to next?

Create these files each cycle:
1. BuildWorld/.Codex/cycles/cycle-[N+1]/cycle-report.md (400+ words)
   - What was built/shipped/measured
   - Product status update
   - Revenue update
   - Next cycle recommendation

2. BuildWorld/.Codex/cycles/cycle-[N+1]/code-review.md
   - The Reviewer agent's assessment of code quality
   - Security issues found
   - Production-readiness score (0-10)

3. BuildWorld/.Codex/cycles/cycle-[N+1]/launch-copy.md (for SHIP cycles)
   - Tweet thread (3-5 tweets)
   - Product Hunt title + tagline + description
   - Landing page copy

Update these files:
- BuildWorld/settings.json: increment current_cycle, update metrics
- BuildWorld/world-status.json: add cycle_history entry, update revenue, update deployment_status
- BuildWorld/.Codex/memory/product-registry.md: update product statuses
- BuildWorld/.Codex/revenue/revenue-log.md: add any revenue entries
- BuildWorld/.Codex/logs/build-log.md: log what was built and where

THE THREE LAWS (enforced every cycle):
1. Ship Weekly — every cycle must produce deployable code
2. Revenue Within 30 Days — every product needs a payment mechanism
3. Budget Zero — free tiers only, no upfront capital

INPUT FROM OTHER WORLDS:
- Read BibleWorld pattern-registry.md for Level 3 patterns that could become products
- Read GhanaWorld opportunities.md for Africa-specific product-market fit data
- These inputs inform WHAT to build, not HOW to build

Commit and push:
cd BuildWorld && git add -A && git commit -m "BuildWorld cycle [N+1] | [CYCLE_TYPE] | [PRODUCT] | revenue=$[MRR]" && git push origin master && cd ..

==== DONE ====
Ship. Ship. Ship. Revenue is oxygen.
