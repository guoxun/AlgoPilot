# AlgoPilot — A Self-Driving Algorithm Iteration Org for Search / Ads / Recommendation

> **English quick look.** Full docs are in Chinese; code comments are bilingual-friendly (Chinese, but tiny and mechanical).

AlgoPilot ports [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to the search / ads / recommendation (SAR) domain, and upgrades it into a **domain-agnostic iteration engine**.

**Core ideas**

- **Ratchet loop**: agent proposes → engineer patches the *only* editable file (`model.py`) → frozen evaluator runs under a fixed time budget → keep only statistically real improvements, revert otherwise.
- **Three gates**: offline ratchet (fast) → **end-to-end delivery gate** (a 400-session replay where the model picks its own exposures and users respond via the true causal pCTR — offline-only gains that hurt the ecosystem get blocked) → human launch gate.
- **Alignment ratio**: every keep records `offline Δ / delivery Δ`. In our mock run: feature crossing 3.07, more epochs 0.078 — a 40× gap in realized value, visible to the org.
- **Limited self-iteration of the org itself**: `pilot propose` distills ledger lessons into `program.md` edits; hard-coded guardrails (immutable rules block, anchor whitelist, idempotence) + human approval.
- **L4 parallel branches**: fork-point settlement, best-branch merge, append-only merge events.

**Run it (no API key needed)**

```bash
python tests/run_all.py                       # 80 tests, incl. full org loops
python -m pilot run --domain domains/rec --engine mock --max-experiments 10
python -m pilot propose --domain domains/rec  # org-code proposal from the ledger
python -m pilot approve --domain domains/rec --yes
```

**LLM mode**: set `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (any OpenAI-compatible endpoint). Researchers/engineers become real LLMs; the reviewer stays pure code.

License MIT. 中文完整文档见 [README.md](README.md) 与 [docs/](docs/).
