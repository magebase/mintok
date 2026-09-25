# Contributing to MinTok

Contributions are welcome: bug fixes, new scenarios, better slicing, framework adapters,
benchmark infrastructure. This document covers the ground rules.

## Ground rules

1. **Open-core boundary.** This repo contains only the open ecosystem layer (IR spec,
   protocol, CLI, basic parsers, semantic diff, fact cache, cost-accounting
   methodology). The commercial MinTok Inference Compiler — context planning, program
   slicing, learned policies, model routing, semantic caching, transformation memory —
   is developed privately. Do not submit or commit those components, their weights or
   datasets, business/pricing plans, secrets, or any personal data here.
2. **BDD first.** Every behavior change ships with a Gherkin scenario in `features/` before
   or with the implementation. Scenario tags are a binding contract (see `AGENTS.md` §7):
   `@domain` = pure logic, no subprocesses; `@integration` = CLI over real files on disk.
3. **Stay dependency-free.** The compiler and IR use only the Python standard library.
   Add a runtime dependency only with a measured benchmark reason.
4. **No secrets, no PII.** Never commit API keys, tokens, passwords, `.env*` files, or any
   personal data (real names, personal emails, phone numbers). Use synthetic fixtures.
   Do not paste secrets into issues, PRs, or commit messages. See `SECURITY.md` for
   vulnerability disclosure.
5. **Tests must pass.** Run `uv run pytest` locally. CI must pass on your PR.
6. **Imperative, scoped commits** (`Add oracle arm to benchmark runner`), one conceptual
   change per commit.

## Development setup

```bash
git clone https://github.com/magebase/mintok
cd mintok
uv sync
uv run pytest -m domain        # fast inner loop
uv run pytest                  # full suite
```

Python 3.12+ (3.14 recommended). No virtualenv wrangling needed; `uv` handles it.

## Pull request checklist

- [ ] Gherkin scenario added/updated for the behavior change
- [ ] `uv run pytest` passes locally
- [ ] No secrets, credentials, or personal data in the diff
- [ ] Commit messages are imperative and scoped
- [ ] Documentation updated (`README.md` / `AGENTS.md`) when commands or contracts change

## Review process

Maintainers review PRs as capacity allows. CI (compile + full pytest suite) must pass
before merge; the `main` branch is protected against force pushes and deletion.

## Licensing

By contributing, you agree that your contributions are licensed under the Apache-2.0
license of this repository (`LICENSE`).
