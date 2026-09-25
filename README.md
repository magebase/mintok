# MinTok

The open layer of an **inference compiler**: a system that turns software-engineering
tasks into the minimum frontier-model computation required to produce a verified change.

**Don't compress expensive AI work. Eliminate it.**

- **Frontier avoidance first** — parsers, static analysis, cached semantic facts, and
  deterministic transforms answer everything they can. Only genuine judgment reaches a
  frontier model.
- **The score is accepted changes per total dollar** — not raw tokens — measured with
  paired bootstrap confidence intervals, because small deltas are usually noise.
- **Hard targets**: ≥4× accepted changes per dollar vs. a strong baseline agent, ≤5%
  success-rate degradation, and frontier inference cost that grows **sublinearly with
  repository size** after indexing.

## Open core

This repository is the open ecosystem layer. The optimizer that produces the savings is
the commercial **MinTok Inference Compiler** and is not published here.

| Open in this repo | Commercial (not published) |
|---|---|
| Agent Program IR v1 (spec + JSON + reference compiler) | Task-specific context planner |
| Agent ABI protocol (`query` / `change` / `verify`) | Program slicer |
| CLI (`compile`, `diff`) | Learned context policies |
| Basic language parsers (Python) | Semantic cache optimizer |
| Semantic diff | Model-specific context backends |
| Hash-invalidated fact cache | Model routing engine |
| Cost-accounting methodology (OCU metering) | Transformation memory / learned policies |
| Agent Efficiency Benchmark runner + profiler | Enterprise distributed indexing |

The open protocol reserves the `slice` operation and the full context-planning surface
for the commercial build; the reference implementation here is complete enough to
compile real repositories, emit the IR, produce semantic diffs, and account for
benchmark economics.

## Quickstart

```bash
git clone https://github.com/magebase/mintok
cd mintok
uv sync

# Compile a repository into Agent Program IR (JSON)
uv run mintok compile path/to/repo --out ir.json

# Semantic diff: meaning-level changes, not line diffs
uv run mintok diff path/to/old path/to/new

# Relearn pack: only what agents must relearn between two versions
uv run mintok relearn path/to/old path/to/new

# Deterministic context-token benchmark vs a grep-and-paging baseline
uv run mintok bench path/to/repo

# Profile agent session records for avoidable inference spend (free)
uv run mintok profile sessions.jsonl

# Compare two benchmark arms: cache-aware $/solved, work/$, raw traces published
uv run mintok benchmark baseline.jsonl optimizer.jsonl

uv run mintok --version
```

The compiler is pure stdlib (`ast`) — no dependencies at runtime.

## Agent ABI

Agents learn one compact interface over the semantic IR instead of dozens of
verbose tools and raw source reads:

- `query` — semantic facts: `symbol`, `effects`, `callers`, `find`, `writers`, `summary`
  (`slice` in the commercial build)
- `change` — replace one symbol's definition, with parse validation and interface-impact report
- `verify` — run a check command, get pass/fail plus a compact output tail

## Semantic IR v1

An open, versioned, model-independent representation: symbols, signatures, effects
(calls / raises / writes), confidence, provenance, and two hashes per symbol
(`body_hash` ignores formatting, comments, and docstrings; `interface_hash` covers
signature + effects). A text change with an unchanged interface hash means an agent does
not need to relearn the component.

## Token minimization (deterministic, no model calls)

The open layer eliminates frontier reads before any compression is needed:

- **Rendering ladder** — `find` (one signature line per name match), `symbol` /
  `effects` / `callers` / `writers` (signature + compact facts), `summary` (adds the
  docstring summary line); raw source is the last resort.
- **Relearn packs** — `mintok relearn old new` emits only symbols whose
  `interface_hash` changed, rendered as compact facts. Body-only changes are
  provably free to skip and are omitted by count.
- **`mintok bench`** — deterministic context-token benchmark. For representative
  questions (locate / understand / attribute callers / attribute writers / relearn),
  it compares a competent grep-and-paging baseline (grep hit lines + sources of the
  innermost matching definitions) against the ABI answers, using the pluggable
  chars/4 estimator. On this repository it measures **~33× fewer context tokens**
  (97%); the relearn task alone drops a one-line body-only edit from a full-file
  re-read to an 18-token omission note.

## Development

```bash
uv sync
uv run pytest -m domain        # fast inner loop
uv run pytest -m integration   # CLI over real files on disk
uv run pytest                  # full suite
```

All behavior is specified in Gherkin (`features/`) and bound with pytest-bdd
(`tests/acceptance/`). See [AGENTS.md](AGENTS.md) for the engineering rules, the
testing contract, and the open-core boundary that contributors must respect.

## Contributing

Contributions to the open layer are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and
the [Code of Conduct](CODE_OF_CONDUCT.md). Security issues: [SECURITY.md](SECURITY.md)
(private disclosure, never public issues).

## License

The open layer is [Apache-2.0](LICENSE). The MinTok Inference Compiler (context
planning, slicing, learned policies, routing, semantic caching) is commercial software;
see the [magebase GitHub organization](https://github.com/magebase) for contact.
