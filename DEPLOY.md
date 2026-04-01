# BuildWorld Deployment Playbook
## The Exact Steps to Go from $0 to Live

**Status:** 18 products code-complete. 1,473 tests passing. $0 revenue.
**Blocker:** Manual setup required for PyPI accounts and GitHub org.
**Time to execute this playbook:** ~2 hours.

---

## STEP 1: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Username: `buildworld-ai`
3. Email: your email
4. Enable 2FA (required for Trusted Publishing)

---

## STEP 2: Create GitHub Organization

1. Go to https://github.com/organizations/plan
2. Org name: `buildworld-ai`
3. Plan: Free

---

## STEP 3: Create GitHub Repos (Priority Order)

Create these repos in the `buildworld-ai` GitHub org. Each is one product.

| Repo Name | PyPI Package | Priority |
|-----------|-------------|----------|
| `semantic-pass-k` | semantic-pass-k | 1 — Launch HN first |
| `model-parity` | model-parity | 2 |
| `cot-fidelity` | cot-fidelity | 3 |
| `chain-probe` | chain-probe | 4 |
| `context-trace` | context-trace | 5 |
| `prompt-shield` | prompt-shield | 6 |
| `context-lens` | context-lens | 7 |
| `rag-pathology` | rag-pathology | 8 |
| `llm-mutation` | llm-mutation | 9 |
| `spec-drift` | spec-drift | 10 |
| `drift-guard` | drift-guard | 11 |
| `agent-patrol` | agent-patrol | 12 |
| `llm-contract` | llm-contract | 13 |
| `cot-coherence` | cot-coherence | 14 |
| `context-trim` | context-trim | 15 |
| `llmguardrail` | llmguardrail | 16 |
| `prompt-lock` | inject-lock | 17 |

Settings for each repo:
- Visibility: Public
- Add README: NO (already exists)
- License: NO (already exists)

---

## STEP 4: Push Code to Each Repo

For each product (example: semantic-pass-k):

```bash
cd products/semantic-pass-k
git init
git add .
git commit -m "Ship semantic-pass-k v0.1.0 — 177/177 tests pass"
git branch -M main
git remote add origin https://github.com/buildworld-ai/semantic-pass-k.git
git push -u origin main
```

**Shortcut — push all at once:**
```bash
for product in semantic-pass-k model-parity cot-fidelity chain-probe context-trace prompt-shield context-lens rag-pathology llm-mutation spec-drift drift-guard agent-patrol llm-contract cot-coherence context-trim llmguardrail prompt-lock; do
  cd "E:/BuildWorld/products/$product"
  git init
  git add .
  git commit -m "Ship $product v0.1.0"
  git branch -M main
  git remote add origin "https://github.com/buildworld-ai/$product.git"
  git push -u origin main
  cd "E:/BuildWorld"
done
```

---

## STEP 5: Configure OIDC Trusted Publishing on PyPI

For each package, do this on PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI project name:** `semantic-pass-k` (or the package name)
   - **Owner:** `buildworld-ai`
   - **Repository name:** `semantic-pass-k`
   - **Workflow filename:** `publish.yml`
   - **Environment name:** `pypi`
4. Click "Add"

Repeat for all 17 packages.

**Why OIDC?** No stored API tokens. GitHub Actions gets a short-lived token from PyPI. More secure, no credential rotation needed.

---

## STEP 6: Create `pypi` Environment in Each GitHub Repo

For each repo in `buildworld-ai`:
1. Go to repo Settings → Environments
2. Click "New environment"
3. Name: `pypi`
4. No required reviewers (automated publish)
5. Save

---

## STEP 7: Tag v0.1.0 to Trigger Publish

```bash
# For semantic-pass-k (do first — highest HN priority)
cd products/semantic-pass-k
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow will:
1. Run 177 tests
2. Build the distribution (wheel + sdist)
3. Publish to PyPI via OIDC

Within 2-3 minutes: `pip install semantic-pass-k` works globally.

**Then tag the others in priority order:**
```bash
for product in model-parity cot-fidelity chain-probe context-trace prompt-shield; do
  cd "E:/BuildWorld/products/$product"
  git tag v0.1.0
  git push origin v0.1.0
  cd "E:/BuildWorld"
done
```

---

## STEP 8: Deploy GhanaScope to Vercel

1. Go to https://vercel.com/new
2. Import GitHub repo: `buildworld-ai/ghanascope`
3. Framework: Astro (auto-detected)
4. Deploy
5. Get your `.vercel.app` URL

**Or via CLI:**
```bash
cd products/ghanascope
npx vercel --prod
```

---

## STEP 9: Launch Sequence

**Day 1 (semantic-pass-k):**
- Post to Hacker News: "Show HN: semantic-pass-k — Does your AI agent give consistent answers?"
- Tweet thread (see `.Codex/cycles/cycle-022/launch-copy.md`)
- Post in LangChain Discord #tools channel
- Post in LlamaIndex Discord

**Day 3 (model-parity):**
- Post to Hacker News: "Show HN: model-parity — Certify your replacement LLM is behaviorally equivalent before you migrate"

**Day 7 (cot-fidelity):**
- Post to Hacker News: "Show HN: cot-fidelity — Is your LLM's chain-of-thought actually causal?"

---

## STEP 10: Add Stripe to Top Products (After 100+ installs)

Once semantic-pass-k reaches 100+ installs on PyPI:
1. Create Stripe account
2. Add Pro tier to README: `$49/month — hosted API, unlimited runs, team dashboard`
3. Create Stripe payment link
4. Add to GitHub README and PyPI description

---

## Verification

After tagging v0.1.0 and waiting ~3 minutes:

```bash
pip install semantic-pass-k
python -c "from semantic_pass_k import ConsistencyRunner; print('Live!')"
```

If this works: **first product is live. The system works.**

---

*18 products. 1,473 tests. 2 hours from this playbook to first install.*
*Ship it.*
