# AGENTS.md — MinTok (open layer)

> Operating instructions for AI coding agents and human contributors working in this
> repository. MinTok is a **local developer tool** running on standard CPython.

MinTok is the open layer of an inference compiler that turns software-engineering tasks
into the minimum frontier-model computation required to produce a verified change.
**Don't compress expensive AI work. Eliminate it.**

---

## 1. Hard targets

1. **≥4× accepted changes per total dollar** vs. a strong baseline agent.
2. **≤5% success-rate degradation** at worst (ideally improvement).
3. **Frontier cost sublinear in repository size** after indexing.

Compression alone is capped (~1.78× work/$ in the best published benchmark; AST-level
structured actions give 12–38% fewer tokens). The rest must come from **frontier
avoidance**: deterministic tools answer everything they can before a frontier model is
involved.

## 2. Objective function (never optimize raw tokens)

```text
efficiency = accepted_changes / (frontier$ + local_model$ + indexing$ + storage$ + CPU$)
```

- Raw tokens are a **diagnostic**, not the score. Compression that breaks prompt-cache
  locality, adds turns, or increases output can cost more.
- Round trips are a hidden cost: optimize `cost + latency + turns`.
- First-class metrics: `frontier_calls / successful_task`, cache-aware $/solved task,
  cost-vs-repo-size curve `C(N)`.

## 3. Two layers, in this order

1. **Frontier avoidance (always first).** Can a parser, compiler, codemod, static
   analyzer, test runner, cached result, or transformation template do it? Navigation,
   symbol/type/caller lookup, AST edits, imports, and migrations must not reach a
   frontier model.
2. **Frontier compression.** Minimize context only for unresolved semantic decisions
   (ambiguous requirements, tradeoffs, architecture, business logic). Separate retrieval
   compression (100k → 8k evidence) from reasoning compression (8 turns → 2 turns).

## 4. Open-core boundary (mandatory for all contributors)

This repo is the **open ecosystem layer**. The commercial **MinTok Inference Compiler**
is built and hosted elsewhere. **Never commit any of the following to this repository:**

- program-slicing implementations beyond trivial graph reads,
- task-specific context planning or token-allocation logic,
- learned context policies, ranking weights, or cost/success predictors,
- model-routing engines or model-specific context backends,
- optimization corpora, trajectory datasets, or transformation memory,
- business plans, pricing, or revenue strategy,
- secrets, credentials, `.env*` files, or any personal data (PII).

Work on those components belongs in the private commercial repository. PRs that add them
here will be declined. High-level thesis statements are fine; recipes are not.

## 5. Core architecture rules (open layer)

- **Agent Program IR v1** is open, versioned, model-independent: symbols, types, effects,
  relationships, source refs, confidence, provenance, hashes. Framework adapters compile
  into it.
- **Agent ABI**: one compact interface (`query` / `change` / `verify`). Tool schemas are
  context; keep the surface tiny and measure its token cost. The protocol reserves the
  `slice` op for the commercial build; the open reference build reports that clearly.
- **Cache verifiable facts, not summaries.** Facts carry the source hash they were
  derived from and are invalidated by dependency tracking. Never store AI-written prose
  as truth.
- **Two hashes per symbol**: `body_hash` (normalized AST; ignores formatting, comments,
  docstrings) and `interface_hash` (signature + effects + calls). Text change with
  unchanged interface hash ⇒ the agent need not relearn the component.
- The compiler/IR stays dependency-free (stdlib `ast`); add a dependency only with a
  measured benchmark reason.

## 6. Evaluation discipline

- Temperature-0 runs flip outcomes on ~9% of benchmark instances between identical runs;
  small gains are noise. Require repeated runs, **paired** task evaluation, bootstrap
  CIs, multiple models.
- Compare against good agents (grep/paging agents), not straw men.
- Track task classes separately: tiny algorithm bugs (~1.1× headroom) vs. schema/API
  propagation, broad refactors, dependency upgrades, monorepo navigation (10×+ possible).
- **Kill criterion (run first):** the oracle experiment. If oracle-context headroom is
  < ~2–3×, a generalized 4× from context architecture is unlikely.

## 7. Testing contract (pytest-bdd, mandatory)

All behavior is specified in Gherkin (`features/*.feature`) and bound with pytest-bdd
(`tests/acceptance/`). Scenario tags are a binding contract:

| Tag | Means | Must not | Budget |
|---|---|---|---|
| `@domain` | Pure logic via library calls | No HTTP, no browser, no subprocesses | <5 ms |
| `@integration` | CLI / compiler against real files on disk | No browser | <100 ms |

Cardinal rule: **the cheapest test level that genuinely proves it.**

```bash
uv run pytest -m domain        # fast inner loop
uv run pytest -m integration   # CLI over real files on disk
uv run pytest                  # everything (CI gate)
```

Every new capability ships with a scenario first. Do not weaken an assertion to make a
scenario pass; change the implementation.

## 8. Code layout

```text
src/mintok/
  ir.py               Agent Program IR v1 (symbols, facts, hashes, JSON)
  tokens.py           token estimation (pluggable; chars/4 default)
  compiler/python.py  Python AST → IR (symbols, calls, raises, attribute writes)
  cache.py            hash-invalidated fact cache
  diff.py             semantic diff between two compiled repositories
  metrics.py          run records, accepted-changes/$, paired bootstrap, oracle verdict, OCU metering
  records.py          open JSONL record formats (agent sessions, benchmark runs)
  profiler.py         avoidable-spend profiler (published formulas; capped at total spend)
  benchmark.py        Agent Efficiency Benchmark runner, paired report, savings summary
  abi.py              Agent ABI + compact tool surface (query / change / verify)
  cli.py              mintok compile | diff | slice (commercial) | --version
features/*.feature    Gherkin specs (@domain / @integration)
tests/acceptance/     pytest-bdd step definitions
```

## 9. Git and commit discipline

- Commit coherent, self-contained units of work; imperative messages
  (`Add package re-export resolution`), not `update` / `wip`.
- Run the relevant scenarios before committing.
- Never rewrite shared history; never force-push `main`.
- Keep commits reviewable and revertible.

## 10. Security and privacy policy (all contributors and agents)

- **Never commit secrets**: API keys, tokens, passwords, `.env*` files, connection
  strings, or credentials of any kind. `.gitignore` blocks `.env*`; do not force-add.
- **Never commit personal data (PII)**: real names, personal emails, phone numbers, home
  directories, or customer data — in code, fixtures, logs, commit messages, issues, or
  PRs. Use synthetic fixtures (e.g. `billing.py`) instead of real repositories.
- If a secret is accidentally committed, rotate it immediately and notify maintainers via
  `SECURITY.md`; history scrubbing is secondary to rotation.
- Vulnerabilities go through private vulnerability reporting, never public issues.
