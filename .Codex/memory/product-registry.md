# BuildWorld Product Registry
## Every Product — From Idea to Revenue

**Last Updated:** Cycle 000 (initialization)

---

## ACTIVE PIPELINE

### PROD-001: LogosSchema
**Source Pattern:** BibleWorld PAT-012 (John 1:1-14 — Logos as constitutional architecture)
**Status:** SPEC_COMPLETE — skeleton code exists in BibleWorld `.Codex/builds/logosschema/`
**Priority:** 1 (ship first)

**What it does:** Founder describes business in plain English → gets ER diagram, API spec, governance rules
**Who pays:** Non-technical founders, accelerator cohorts, Africa tech startups
**Revenue model:**
- Free: 1 schema
- Pro: $49/month (unlimited schemas, export to code scaffolding)
- Africa Pack: $15/month (lighter interface, offline-capable)

**Build plan:**
- Cycle 1: Next.js app with Claude API integration, Mermaid diagram output, Vercel deploy
- Cycle 2: Stripe integration, user auth (Supabase), schema save/load
- Launch: Product Hunt, Indie Hackers, Twitter/X, Ghana tech WhatsApp groups

**Revenue target:** $100 MRR within 60 days of launch
**What kills it:** AI coding assistants add native schema generation
**What saves it:** Africa Pack pricing creates a segment no one else serves

---

### PROD-002: EvalGate
**Source Pattern:** BibleWorld PAT-010 (Genesis 1:2-3 — logical before physical)
**Status:** SPEC_COMPLETE — Python code exists in BibleWorld `.Codex/builds/evalgate/`
**Priority:** 2

**What it does:** Evaluates AI agent outputs before production deployment
**Who pays:** AI developers, enterprise AI teams
**Revenue model:**
- Open source core (PyPI package — free)
- Hosted API: $0.001/evaluation call
- Standard: $199/month (500K evaluations)
- Enterprise: $500/month (unlimited + custom test suites)

**Build plan:**
- Cycle 1: Polish Python package, publish to PyPI, write docs
- Cycle 2: Hosted API on Railway free tier, Stripe metered billing
- Launch: Hacker News, AI Twitter, r/MachineLearning, r/ClaudeAI

**Revenue target:** $50 MRR within 90 days (developer adoption is slower)
**What kills it:** Anthropic/OpenAI build native evaluation
**What saves it:** Open source creates adoption; enterprise tier monetizes power users

---

### PROD-003: GhanaFounder Intelligence Report
**Source:** GhanaWorld cycle outputs (direct productization)
**Status:** CONCEPT
**Priority:** 3

**What it does:** Weekly email: top Ghana opportunities, macro updates, failure warnings, pricing data
**Who pays:** Diaspora Ghanaians considering investments, returning professionals
**Revenue model:** $29/month subscription via Stripe

**Build plan:**
- Cycle 1: Template email in HTML, Stripe checkout link, Resend integration, landing page
- The "product" is the intelligence GhanaWorld already produces — just formatted and delivered

**Revenue target:** $290 MRR (10 subscribers) within 30 days
**What kills it:** Free alternatives (blogs, podcasts covering Ghana investment)
**What saves it:** Depth and rigor no free source provides; GhanaWorld's 21 cycles of accumulated intelligence

---

### PROD-004: AI Business Schema Generator
**Source:** BibleWorld PAT-012 + PAT-001
**Status:** CONCEPT
**Priority:** 4 (ship after LogosSchema proves product-market fit)

**What it does:** Broader LogosSchema for global market — "3 sentences → full tech architecture"
**Revenue model:** $9.99/month, freemium (3 free schemas/month)

---

### PROD-005: Scripture Pattern Engine
**Source:** BibleWorld all patterns
**Status:** CONCEPT
**Priority:** 5

**What it does:** Enter a Bible passage → get technology pattern analysis + build opportunity
**Revenue model:** $4.99/month individuals, $19.99/month churches/ministries

---

### PROD-006: Africa SME Toolkit
**Source:** GhanaWorld + BibleWorld PAT-004
**Status:** CONCEPT
**Priority:** 6 (requires more development)

**What it does:** Inventory + customer + payment tracking for Ghana SMEs, mobile-first, offline-capable
**Revenue model:** $5/month (GHS 60), mobile money payment

---

## SHIPPED PRODUCTS (CODE-COMPLETE)

| Product | Tests | Pivot Score | PyPI Status |
|---------|-------|-------------|-------------|
| model-parity | 97/97 | 8.90 | READY (pending deploy) |
| chain-probe | 45/45 | 8.85 | READY (pending deploy) |
| cot-fidelity | 124/124 | 8.85 | READY (pending deploy) |
| context-lens | 80/80 | 8.80 | READY (pending deploy) |
| prompt-shield | 130/130 | 8.75 | READY (pending deploy) |
| prompt-lock | 34/34 | 8.70 | READY (pending deploy) |
| rag-pathology | 41/41 | 8.65 | READY (pending deploy) |
| llm-mutation | 90/90 | 8.65 | READY (pending deploy) |
| semantic-pass-k | 177/177 | 8.65 | READY (pending deploy) |
| spec-drift | 67/67 | 8.63 | READY (pending deploy) |
| drift-guard | 41/41 | 8.60 | READY (pending deploy) |
| agent-patrol | 51/51 | 8.50 | READY (pending deploy) |
| llm-contract | 66/66 | 8.30 | READY (pending deploy) |
| cot-coherence | 84/84 | 8.00 | READY (pending deploy) |
| context-trim | 103/103 | — | READY (pending deploy) |
| llmguardrail | 41/41 | — | READY (pending deploy) |
| ghanascope | Astro | — | VERCEL (pending deploy) |

**Total Python tests: 1,371 passing**

---

## PRODUCT METRICS

| Product | Status | Revenue | Users | Live URL |
|---------|--------|---------|-------|----------|
| semantic-pass-k | CODE_COMPLETE | $0 | 0 | PyPI pending |
| All 16 Python tools | CODE_COMPLETE | $0 | 0 | PyPI pending |
| ghanascope | CODE_COMPLETE | $0 | 0 | Vercel pending |

**Total MRR:** $0
**Total products shipped:** 17 (code-complete)
**Total products generating revenue:** 0
**Critical blocker:** PyPI account + GitHub org + OIDC trusted publishing (manual, outside build loop)
