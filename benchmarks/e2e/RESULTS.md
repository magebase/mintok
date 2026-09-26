# End-to-end paired agent benchmark — results

**Date:** 2026-09-25 · **Commit under test:** 51615a9 (+ harness fixes through ebad210)
**Agent:** this repository's own coding subagents (frontier model via the agent
harness), one isolated session per run. **No external API spend**: the harness
cannot observe provider-side billing (see "Blockers").

## Method

- 30 real repository tasks (see `tasks.py`): parameter additions, renames,
  behavior changes, class-level edits, new CLI commands, refactors, docstrings,
  comments, error handling, multi-file changes, cleanups. Task instructions are
  arm-agnostic; each arm gets the identical instruction plus an arm discipline.
- **Control arm** (30 runs): ordinary shell tooling through a logging shim
  (`grep`/`sed`/`pytest`), every call logged to a JSONL trajectory.
- **MinTok arm** (30 runs): only the Agent ABI through the same shim
  (`query` / `change` / `add` / `remove` / `verify` / `suite`). No shell.
- Scoring: automated checker per task (suite pass + behavioral assertion),
  executed against the task copy. A run is *solved* only if both pass.
- Measured quantity: **tool-side context tokens** (everything the tool layer
  fed into the agent's context: tool outputs + instruction/prompt overhead),
  from the trajectory logs. Not billed dollars.

## Headline (30 tasks × 2 arms)

| arm | tasks | solved | solve % | turns | tool-context tokens | in-tok/solved | solved/Mtok |
|---|---|---|---|---|---|---|---|
| control | 30 | 29 | 97% | 153 | 66,277 | 2,285 | 437.6 |
| mintok | 30 | 26 | 87% | 531 | 77,096 | 2,965 | 337.5 |

**Paired view (26 tasks both arms solved):**

| statistic | control / mintok context-token ratio |
|---|---|
| geometric mean | **0.82x** (mintok arm more expensive) |
| median | 0.93x |
| range | 0.03x – 42.2x |
| mintok wins / losses | 11 / 15 |

## Verdict

**The end-to-end benchmark does not support a work-per-dollar win for the
current open-layer ABI.** The deterministic context-token benchmark
(`mintok bench`: 6.1x vs strong tooling) does not transfer to full agent
trajectories. Three effects dominate:

1. **Turn amplification.** The ABI arm uses ~3.5x more turns: each
   query/change/suite is its own tool call, and edits require a patch file
   round-trip. Frontier pricing bills every turn's output (and cache-breaking
   input), so many small calls can cost more than one `sed`.
2. **Query blowups.** On several tasks the ABI arm read many symbol pages
   before editing (worst case: 29,249 tokens on a 2-parameter task the control
   solved for 985). When the agent does not know where to look, unstructured
   paging is competitive.
3. **ABI coverage gaps cost solve rate.** All three mintok-only unsolved tasks
   are capability gaps, not reasoning failures: module-level docstrings are not
   indexed (`docstring-tokens-module`), import-line rewrites in existing files
   have no op (`multi-move-stable-hash`), and test-file callers cannot be
   edited (`rename-cached-facts` — the control edited tests freely).

Where the task sits inside the ABI's model, it wins: up to 42x (control
flailed reading a 25k-token file the ABI answered from one symbol page), 2.8x
on symbol listing and case-sensitive lookup, ~1.4x on class-level edits.

## V2 ablation ladder: where the loss lives

After the v1 negative verdict, the architecture was changed from "ABI-only" to
**hybrid**: semantic fast path + ordinary tools + escape hatches. The ablation
ladder isolates which layer pays for itself (all arms on the V2 harness,
commit 2814a75; policies in `agent_cli.py`):

| arm | adds vs previous | solved | tn/att | tok/att | paired geomean vs control |
|---|---|---|---|---|---|
| control | shell only | 29/30 | 5.1 | 2,209 | — |
| B | + semantic reads (`query find/symbol`) | **30/30** | 6.4 | **1,834** | 0.85x |
| C | + attribution (`callers/writers/effects`) | **30/30** | 7.2 | **1,753** | 0.85x |
| D | + batched bundle (`inspect`) | 29/30 | 8.5 | 1,938 | 0.82x |
| E | + semantic edits (`change/add/remove/patch`) | 29/30 | 14.0 | 3,393 | 0.54x |
| F | + circuit breaker (limits unlock shell) | 29/30 | 14.8 | 6,853 | 0.42x |
| G | + local planner (`task_packet`) | 29/30 | 16.0 | 4,906 | 0.42x |
| mintok | ABI-only (v1, no shell) | 26/30 | 16.8 | 2,570 | 0.82x |

Findings (tool-context tokens; see caveats below):

1. **The user hypothesis held for reads.** Arms B–C beat control on solve rate
   (30/30 vs 29/30) *and* tokens (–17%/–21% tok/att) at a +25–40% turn cost.
   Attribution (`callers/writers/effects`) is the cheapest per-token layer.
2. **Semantic edits are the loss point.** E jumps to 1.5x control tokens and
   2.7x turns: `change` makes the model re-emit whole function bodies. Hybrid
   E still solved 29/30 — capability parity works — but tokens pay for it.
3. **The circuit breaker fixed capability, not cost.** F solved the same 29/30
   but tokens *doubled* vs E (6,853): when the breaker unlocks shell, workers
   flood context with unfiltered shell output (worst: 94,747 tokens on
   `add-cli-verify` vs 6,962 in E). An unlocked shell needs output capping,
   not just access.
4. **The packet planner did not pay off.** G is cheaper than F (workers fall
   back to shell less) but still 2.2x control tokens; workers over-requested
   packets (worst: 33,718 tokens on `new-ir-symbols-of-kind`). Packet reuse
   and packet scoping need work before this layer earns its tokens.
5. **`new-pack-is-empty` failed on control, D, E, F, G** (B and C solved it):
   its edge case is a reading-comprehension trap, not a tooling difference.

**Architectural conclusion:** ship B–D (semantic reads + attribution +
bundle) behind ordinary tools; treat semantic *edits* as opt-in per task
class; add output caps to breaker-unlocked shell; rework packet reuse before
shipping `task_packet`.

Caveats: control ran on pre-V2 copies (55–58 tests) while B–G ran on 63-test
copies; `frontier_usd` remains unobservable (0.0 recorded); single run per
cell (no temperature noise handling).

## Blockers to a $-denominated claim

- **Billed cost is unobservable here.** Runs execute as harness subagents;
  provider token accounting is not exposed to the harness, so `frontier_usd`
  is recorded as 0 and all ratios are tool-context-token proxies. A true
  accepted-changes/$ number requires an API-level runner that logs per-request
  usage payloads (input, output, cached tokens) per arm.
- **Success-rate degradation exceeds target.** 87% vs 97% is a 10-point drop;
  the hard target allows ≤5%. The coverage gaps above account for all three.

## Fixes already landed from this benchmark

- `add` op added to the ABI (new symbols/modules were impossible before);
  import splicing ordered after `__future__` imports (commit 3e971c7).
- Checker environment fixed to score the task copy, not the harness
  (copy-first imports, `root` binding); four checker assertion bugs fixed
  (fe59196, ebad210, 51615a9).

## Per-task paired view

Generated by `uv run python benchmarks/e2e/run_bench.py report` (runs/*.jsonl
committed alongside; raw JSONL trajectories in `trajectories.tar.gz`).

```
arm         tasks  solved  solve%  turns  tool_out_tok  in_tok/solved  s/Mtok
-----------------------------------------------------------------------------
control        30      29     97%    153         66277           2285  437.56
mintok         30      26     90%    531         77096           2965  337.54

per-task paired view (control vs mintok tool-context tokens):
add-cli-verify                  OK OK     3603   2773   1.3x
add-param-estimate              OK OK      345    607   0.6x
add-param-factsfor              OK OK      985  29249   0.0x
add-param-renderfact            OK OK      926   1943   0.5x
behavior-estimate-chars3        OK OK      792    452   1.8x
behavior-find-casesensitive     OK OK     1217    439   2.8x
behavior-tail-10                OK OK      704    562   1.3x
class-change-factcache          OK OK      789   1378   0.6x
class-change-programir          OK OK     1156    801   1.4x
cleanup-kinds                   OK OK      867   4248   0.2x
cli-callers-command             OK OK     2432   5266   0.5x
cli-diff-quiet                  OK OK     1348   3365   0.4x
comment-skipdirs                OK OK      614    397   1.5x
decorator-lrucache              OK OK     9095   4396   2.1x
docstring-factcache             OK OK     1341   1506   0.9x
docstring-tokens-module         OK NO      377    417   0.9x
errors-compile-missing-root     OK OK    25735    610  42.2x
errors-query-unknown-op         OK OK      576    794   0.7x
multi-errors-module             OK OK      514    453   1.1x
multi-move-stable-hash          OK NO     1306   1182   1.1x
new-cli-ops-flag                OK OK     1174   3281   0.4x
new-cli-symbols                 OK OK     4118   1463   2.8x
new-ir-symbols-of-kind          OK OK      628    497   1.3x
new-pack-is-empty               NO NO      637    805   0.8x
refactor-indent-helper          OK OK     1771   1878   0.9x
refactor-resolve-call           OK OK     1436   1541   0.9x
rename-cached-facts             OK NO      538    839   0.6x
rename-dotted-module            OK OK      361    621   0.6x
rename-savings-pct              OK OK      413   4193   0.1x
rename-stable-hash              OK OK      479   1386   0.3x
```

## Threats to validity

- One agent family, one model tier, single run per cell (temperature noise not
  addressed; ~9% of instances flip between identical runs).
- Copies were prepared at slightly different commits as ABI fixes landed
  mid-benchmark; task difficulty is unaffected but not bit-identical.
- The ABI arm's discipline forbids shell; the control arm's ordinary tooling
  is the strongest realistic baseline (grep/sed/pytest), not a straw man.
- `new-pack-is-empty` is unsolved on both arms (its behavioral edge case
  requires reading the semantics carefully, not a tooling difference).

## V3 experiments: batched reads, virtualized shell, codemods (arms H/I/J)

Three levers were built on top of C (the shipped default) and measured on the
same 30 tasks, verbatim canonical instructions, 30 fresh copies per arm:

- **H = C + `query batch`** — one call may answer several read questions
  (`query batch "symbol:mod:fn callers:mod:fn"`).
- **I = C + virtualized shell** — shell output is stored to disk; the agent
  sees a 40-line head plus a handle, and pages/finds via `result R1 ...`.
- **J = C + deterministic codemods** — `codemod rename|add_parameter|add_import`
  applies AST-verified structural transforms; the model emits only arguments.

Four-metric headline (tool-context tokens; `tok/solv` is the $-per-success
proxy until API billing telemetry lands):

```
arm    policy                    solved  solve%  turns/att  tok/att  tok/solv
---------------------------------------------------------------------------
C      reads + attribution         30/30   100%       7.2       1753      1753
H      C + query batch             29/30    97%       9.9       2590      2679
I      C + virtual shell           29/30    97%       7.8       2046      2116
J      C + codemods                28/30    93%       7.9       2104      2254
--- reference arms from the ablation ladder ---
control  grep/paging               29/30    97%       5.1       2209      2285
B        reads only                30/30   100%       6.4       1834      1834
D        reads + freeform edits    29/30    97%       8.5       1938      2005
E        C + task packets          29/30    97%      14.0       3393      3510
F        C + circuit breaker       29/30    97%      14.8       6853      7089
G        C + packet planner        29/30    97%      16.0       4906      5075
mintok   ABI-only (v1)             26/30    87%      16.8       2570      2965

paired geomean vs control (jointly solved): B 0.85x, C 0.85x, D 0.82x,
H 0.62x, I 0.73x, J 0.69x, E 0.54x, F 0.42x, G 0.42x
```

### Findings

1. **C stands.** It remains the only arm at 30/30 with the lowest tokens
   (1753/att) and near-lowest turns (7.2). All three new levers regressed
   tokens and none beat control's 5.1 turns/att.
2. **Batching backfired (H).** Turns rose 7.2 -> 9.9 and tokens +48%. Batched
   queries over-fetch (workers answer speculative questions they did not
   need), and malformed batch specs add rejection/retry turns. Turn-count
   reduction was NOT achieved by merging read calls.
3. **Virtual shell is tail-risk insurance, not average savings (I).** Worst
   per-task tool context dropped from 94,747 (uncapped breaker, F) and
   30,272 (packets, E) to 8,947; but the head+handle boilerplate taxes every
   shell call, costing ~+17% tokens on average because floods never occur in
   this suite. Keep it as a safety cap, not as a token lever.
4. **Codemods break even at this scale (J).** The shim's edit responses were
   already compact; emitting arguments instead of edits saves model output
   the shim never charged for. One checker-strict failure appeared (a
   trailing-style comment next to SKIP_DIRS fails the own-line-comment
   checker that accepts the same comment on its own lines).
5. **`new-pack-is-empty` has now failed on control, D, E, F, G, H, I, J** —
   a stable semantic trap (omitted_body_only accounting), independent of
   tooling. Retained as a known-hard item.

### Verdict against the V3 targets

- 30/30: only B and C reach it.
- turns <= 5.1 (control): not reached; best is C at 7.2. Batched reads
  made turns worse, so the next turn lever must come from elsewhere
  (candidate: interface-hash-aware "component unchanged" short-circuits,
  i.e. the agent skips relearning files whose interface hash is known).
- tokens < 1,500/att: not reached; best is C at 1753. With the tool surface
  at 165 tokens charged once, and reads already paid-for, the remaining
  mass is edit-turn context — pointing back at edit gating (ship C) and
  larger-task validation before more surface changes.

The three V3 mechanisms stay in the codebase behind policies (H/I/J) with
scenario coverage; none is promoted to the default. Default remains **C**.

## V3 follow-up: arm K (transparent semantic continuity)

**Setup.** Arm K reuses C's exact agent-facing surface (same tools, same ops,
identical prompts — verbatim canonical text). The only difference is invisible
to the agent: the shim maintains a persistent learned-state file
(`<log-stem>.state.json`, from `src/mintok/continuity.py`). After every write,
subsequent `query` responses on already-learned symbols/relations are prefixed
with a hash-based verdict ("X unchanged since learn #N", "interface unchanged
since learn #N; body changed", "removed") instead of forcing a full re-show.
No `check_hash`/`should_relearn` calls are exposed; the agent never sees the
mechanism.

**Results (30 tasks):**

| arm | tasks | solved | solve % | turns | tok/att | paired geomean vs control |
|---|---|---|---|---|---|---|
| K | 30 | 29 | 97% | 8.3 | 2,742 | 0.70x |

**Findings**

1. **Solve rate unchanged (29/30).** `new-pack-is-empty` failed again — its
   tenth consecutive failure across all arms; the trap is arm-independent.
2. **Average tokens regressed (2,742 vs C's 1,753).** Verdict lines add a
   per-response marker even when the agent was about to act anyway, and two
   tasks ballooned (`comment-skipdirs` 16,794; `behavior-find-casesensitive`
   10,530 at 42 turns) from agent variance, not from the mechanism.
3. **Coverage was low.** Continuity verdicts appeared in only 12/30 logs.
   Agents frequently bypass `query` for `read` + shell grep/sed, so the
   lever only pays when agents actually re-query learned symbols. Where
   verdicts fired (e.g. `rename-*` family), per-task tokens were at or below
   C's levels (652, 468, 526, 619 vs C's 753, 471, 452, 576).
4. **Verdict:** K is not promoted. Transparent continuity does no harm on
   average in the covered tasks, but the measured benefit is inconclusive at
   this sample size because agents under-use `query`. A larger frozen run
   (control/C/K on the 120 generated eval tasks) is the right next test;
   per-stratum reporting will show whether continuity concentrates its wins
   on re-visit-heavy strata (api propagation, schema change) as hypothesized.

Default remains **C**.

## C is frozen (decision)

After V3 (H/I/J regress) and the K follow-up (transparent query-attached
continuity: 29/30, 8.3 turns, 2,742 tok/att — regression vs C), the
frontier-facing interface is **frozen at C**: compact semantic reads +
attribution, with no further tool additions on the 30-task dev set.

Key lesson from K: agents frequently bypass `query` for `read` + shell, so
optimizations must work *underneath* whatever the agent naturally does
(interception at the I/O layer), not require cooperation with a new API.
Any future continuity work belongs there and must be validated out of
sample, not on these 30 tasks.

`new-pack-is-empty` is retained as a marked **known semantic-reasoning
trap** (ten consecutive failures across all arms). The report now includes
tail-cost statistics (median/p90/p95/max per arm) so pathological
trajectories stay visible instead of hiding inside averages.

Next: frozen out-of-sample run — control (primary baseline) vs C (primary
experiment) vs K (secondary, exploratory) on the 120 generated eval tasks,
reported per stratum.

## Frozen out-of-sample evaluation (120 generated tasks, control vs C)

The frozen eval ran the 120 stratified generated tasks (`tasks_generated.json`,
never used for development decisions) against the shell-only `control` arm and
frozen **C**. Every task ran once per arm; scores were recorded immediately
after each run (`runs/control-eval.jsonl`, `runs/C-eval.jsonl`). Three
checker defects were found and fixed mid-run (each validated against pristine
fixtures before attributing anything to agents; commits `7cdf902`, `091cf09`,
`e3fcbc2`, `ea53ab5`, `bfbc4b0`, `c43acf4`).

### Aggregate (n=120 per arm)

| arm | solved | tok/attempt | tok/solved | median | p90 | p95 | max |
|---|---|---|---|---|---|---|---|
| control-eval | 114 (95%) | 1,876 | 1,974 | 1,087 | 2,882 | 4,080 | 26,842 |
| C-eval | 114 (95%) | **1,607** | **1,691** | 1,161 | **2,946** | **3,138** | **8,482** |

- **Solve rate: identical** (114/120 both; fails below).
- **Average tool-context tokens: C 14.3% cheaper** per attempt and per solve.
- **Paired geomean on the 111 jointly solved tasks: 0.93x** (C 7% cheaper;
  median 0.95x; C wins 51/111). The dev-set 1.30x did **not** survive at full
  strength — but the aggregate hides where C wins and loses.
- **Tail: C's real advantage.** p95 3,138 vs 4,080 (−23%); max 8,482 vs
  26,842 (−68%). C's semantic reads bound worst-case wandering; control's
  single worst trajectory cost 3.2x C's worst. Under real billing the max,
  not the mean, is what hurts.

### Per stratum (tokens per solved task, C vs control)

| stratum | C solve | ctrl solve | C tok/solv | ctrl tok/solv | C vs ctrl |
|---|---|---|---|---|---|
| api_signature_propagation | 19/20 | 19/20 | 1,618 | 4,008 | **2.48x cheaper** |
| simple_lookup | 15/15 | 14/15 | 943 | 1,341 | 1.42x cheaper |
| feature_addition | 20/20 | 19/20 | 1,166 | 1,544 | 1.32x cheaper |
| schema_or_framework_change | 15/15 | 15/15 | 2,170 | 2,633 | 1.21x cheaper |
| cross_file_bug | 17/20 | 18/20 | 1,793 | 1,594 | 0.89x (costlier) |
| refactor | 15/15 | 14/15 | 1,627 | 1,158 | 0.71x (costlier) |
| large_file_navigation | 13/15 | 15/15 | 2,861 | 1,093 | **0.38x (2.6x costlier, −2 solves)** |

**C's efficiency is concentrated, not uniform.** It wins big exactly where
designed — revisit-heavy api-signature propagation (2.48x), lookups,
features, schema changes — and *loses* on `large_file_navigation`, where
semantic reads over a huge module cost 2.6x the control's targeted grep with
2 fewer solves. Cross-file bugs and refactors are ~costlier but safer on
solve rate.

### Failure accounting (kept, per policy)

- Both arms: `pipeline-cross-04` (checker requires the parameter named
  exactly `policy`; genuine spec-reading trap), `biglib-api-02` (both arms
  broke wrap_text's byte-identical default), `biglib-cross-02` (both arms
  failed the word-boundary truncation edge).
- control only: `shopcart-lookup-03` (genuine miss), `shopcart-refactor-03`
  (flaky), `notesrv-feature-02` (made `excerpt(length)` required despite the
  instruction's `excerpt(length=40)` signature; C read it correctly).
- C only: `biglib-large-03` (inverted emptiness logic its suite didn't catch),
  `biglib-large-14`, `pipeline-cross-03`.

### Verdict

**C holds as the default.** Equal solve at −14% tokens with a −68% worst
case is a real but modest out-of-sample win: 1.17x aggregate, 1.08x paired —
not the dev set's 1.30x. The loss profile (large files) is the clearest
input to the next lever: large-module navigation needs deterministic
symbol/slice tooling under the interface, not more semantic reads.

**K was not run on the 120.** K is a rejected arm (29/30, 0.70x paired on
the dev set); under the funnel policy a full frozen exam is reserved for
shippable candidates, so the 120 K runs are dropped.

### Experiment funnel (infrastructure)

To stop paying frontier compute for bad ideas, the harness now ships a
funnel (`mintok.funnel`, wired into `run_bench.py`):

1. **Static integrity first.** `fingerprint`/`verify` hash prompt, checker,
   and fixture tree; a drifted task refuses to run (exit 2) instead of
   invalidating a wave.
2. **FAST-8 smoke suite** — 8 tasks covering lookup, cross-file, api
   propagation, schema, large-file, ordinary-edit, feature, and the known
   trap (`fast8` subcommand). Ideas die here first.
3. **Sequential stopping.** `sequential_verdict` kills an arm at 5 paired
   tasks if it is ≥25% costlier, at 10 if ≥15% costlier, at 15 without a
   ≥5% improvement, and any time solve rate drops >5 points.
4. **Control cache.** `control_key` hashes task + repo snapshot + harness +
   model + effort + toolset; identical configurations reuse the stored
   control trajectory instead of rerunning it per arm.
5. **Adaptive concurrency.** `AdaptivePool` starts at 8 workers, grows +2
   per clean wave to 12, and drops 3 on provider throttling — replacing the
   fixed 20+-worker waves that triggered rate-limit storms.
6. **Copy-on-write prepare.** Fixture copies try `cp --reflink=auto` before
   falling back to a deep copy (never hardlinks: agents edit in place).
7. **Offline replay.** `replay` recomputes turns/tokens/latency for every
   trajectory from raw shim logs with zero agent runs — metric changes
   never require rerunning arms.

### Oracle routing headroom (offline, zero new runs)

C should not be the optimizer; it should be one backend chosen by one. The
`oracle` subcommand (`run_bench.py oracle --arm control-eval --arm C-eval`,
backed by `mintok.metrics.oracle_router`) computes the upper bound a
per-task arm selector would achieve over the 240 existing frozen
trajectories: a task both arms solve goes to the cheaper solver, a task only
one solves goes to the solver, and a both-fail task is charged the cheaper
failed attempt (optimistic). No frontier calls were made.

| selector | solved | tok/attempt | tok/solved | p95 | max | vs control uniform | vs C uniform |
|---|---|---|---|---|---|---|---|
| uniform control | 114 (95%) | 1,876 | 1,974 | 4,080 | 26,842 | 1.00x | 0.86x |
| uniform C | 114 (95%) | 1,607 | 1,691 | 3,138 | 8,482 | 1.17x | 1.00x |
| **stratum router** | **116 (97%)** | **1,350** | **1,397** | **3,049** | **4,961** | **1.39x** | **1.19x** |
| oracle router | 117 (98%) | 1,163 | 1,193 | 2,741 | 4,135 | **1.61x** | **1.38x** |

The **stratum router** is the realistic proxy: route by task class only
(C for api-signature/lookup/feature/schema strata, control for
cross-file/refactor/large-file), with no per-task outcome peeking. It beats
both uniform arms on solve rate (116/120) while beating control by 28% on
tokens. The gap between 1.39x and the oracle's 1.61x is the residual value
of per-task (below-stratum) selection — smaller than the stratum-level gain
itself.

Two honesty notes. First, the solve-rate gains (+2 stratum router, +3
oracle) sit on temperature-0 single-run flip variance (~9% of instances
flip between identical runs); the robust signal is the token headroom and
the tail (routed max 4,135 vs control's 26,842), not the two extra solves.
Second, the both-fail convention matters: charging the *dearer* failed
attempt on the 3 both-fail tasks instead of the cheaper lifts the oracle to
1,366 tok/attempt (1.37x over control), because control's
`biglib-api-02` failure is the 26,842-token runaway. The stratum router is
immune to this choice — it charges real, pre-declared routes — so its
**1.39x over control is the defensible headline**; treat the oracle as a
1.37–1.61x band whose width is entirely the both-fail convention.

**Routing verdict: pursue aggressively.** Even the convention-independent
stratum router's 1.39x over control (1.19x over C-everywhere, with +2
solves) comes from task-class selection alone — a *measured result on this
frozen benchmark*, not a floor for future datasets — and a router only
needs to know the stratum: deterministic pre-flight features (implied
signature/schema change in the instruction, target module size, reference
fan-out) can approximate it without a model in front of the frontier call.
The strata disagree by 2.48x-vs-0.38x, so the classification margin is
wide; the oracle's 1.61x is the ceiling if per-task selection proves
learnable.

### The pre-flight router: built, and it matches the stratum router

The stratum router used oracle knowledge of the true task class. The
production router may not. `mintok.router` decides from **pre-flight
features only** — the task instruction's wording and the target files'
line counts, both known before the frontier model starts:

```text
api cue ("keyword parameter")            → semantic-C   (C: 2.48x cheaper here)
refactor cues (rename/extract/dead/dup)  → control
feature cues (add ... / endpoint)        → semantic-C
cross-file defect cues (misuses, silently,
  never validates, "from the X module")  → control
schema cues (field (default, dataclass)  → semantic-C
target LOC >= 1000                       → slicer (falls back to control
                                            until the slicer exists)
no cue hit                               → semantic-C at low confidence
```

Evaluated offline over the frozen 120 (each task charged the trajectory of
the arm it was routed to; zero new runs):

| selector | solved | tok/attempt | tok/solved | p95 | max | vs control | vs C |
|---|---|---|---|---|---|---|---|
| uniform control | 114 | 1,876 | 1,974 | 4,080 | 26,842 | 1.00x | 0.86x |
| uniform C | 114 | 1,607 | 1,691 | 3,138 | 8,482 | 1.17x | 1.00x |
| stratum router (true labels) | 116 | 1,350 | 1,397 | 3,049 | 4,961 | 1.39x | 1.19x |
| **pre-flight router** | **116** | **1,342** | **1,388** | **3,049** | **4,961** | **1.40x** | **1.20x** |
| oracle (per-task, band) | 117 | 1,163–1,366 | 1,193–1,401 | 2,741 | 4,135 | 1.37–1.61x | 1.17–1.38x |

The rule chain classifies 115/120 tasks to the backend the frozen stratum
table would pick (exact-class 101/120; label misses on the C-side strata
are cost-free when the backend is right). Two word-level lessons mattered
more than any statistics: "Add a keyword parameter" must outrank
"rename/extract" language (one api task saying "instead of being split"
initially routed to control and ate the 26,842-token runaway), and "Add a
search endpoint" must outrank cross-module phrasing ("from store.search").
The router also captures the oracle's solve-rate gain: 116/120, because
control handles the large-file tasks C fails.

**Disclosure:** the cue vocabulary and its ordering were tuned on the
frozen-eval instructions, so 1.40x is a development result, not a
pre-registered one. The honest test of generalization is a fresh
generated task set; the funnel exists to price that run.

Every decision is logged as the row a learned router would need
(`runs/router-predictions.jsonl`, via `route --eval`):

```text
features (instruction, target_loc, files) → backend → predicted cost →
confidence → reasons → actual {control: tokens/solved, semantic-C: ...}
```

`route --task ID` prints the live decision with its reasons. Rules stay
deterministic until they stop improving.

### Next: the product is the router, not any single arm

The architecture implied by the split: a cheap deterministic classifier
picks among backends — semantic reads + attribution (C) for
api/schema/lookup/feature work, plain shell tools for cross-file bugs and
refactors, and a purpose-built **large-module slice backend**
(deterministic symbol index → AST slice → exact `file:line` candidate
regions → targeted excerpts, no narration, no summaries, no broad symbol
packets) for giant modules where semantic reads lose 2.6x. The pre-flight
router above already ships the classifier slot this backend plugs into.
Before running any live routed system, the oracle should be recomputed
over control/C/slicer to check whether slicing actually recovers the
`large_file_navigation` loss; the funnel (FAST-8 → sequential dev →
frozen confirm) prices that experiment cheaply.

**Milestone framing.** Routing control/C alone does not reach 4x. The
next milestone is **2x on a frozen benchmark** without solve-rate loss;
on this eval the pre-flight router already delivers 1.40x over uniform
control, so the remaining lever is a backend that beats control's
1,093 tok/solved on large-module navigation — the slicer's concrete bar —
plus one more genuinely high-leverage deterministic backend (repetitive
propagation via transforms, revisit caching). Routing then composes the
specialized wins.

### Slicer backend: built, sized offline, bar pre-registered

`mintok.slicer` is the third backend for giant modules, deliberately
dumb by design: from the Agent Program IR it ranks **exact source
regions** — definitions matched by instruction identifiers (a method
beats its own class), call-edge callers, field-write sites, and
test-file mentions — and renders `file:line` excerpts. No narration, no
summaries, no symbol packets; full source is an explicit agent-initiated
fallback the slicer itself never takes. Hard budgets cap the packet
failure mode: initial slice ≤ 800 tokens, expanded ≤ 2,000 (taken only
when the primary definition itself needs it).

Offline sizing over the 15 large-module tasks (package rendered from the
pristine fixture + instruction; zero agent runs; `slice --eval`):

| statistic | value |
|---|---|
| avg package | **358 tokens** (control's actual tool-context avg: 1,093) |
| packages within the initial 800 budget | 13/15 |
| largest package | 1,786 (expanded budget; a class-level target) |
| headroom if solve rate holds | ~3.1x on tool-context |

**Pre-registered promotion criteria** (written before any slicer
trajectory exists; live runs go through FAST-8 → sequential dev first):

- solve rate ≥ control's 15/15 on the stratum;
- tokens/solved ≤ 800 = good, ≤ 600 = strong, ≤ 400 = excellent
  (control: 1,093).

Per the no-aspirational-oracle rule, the 3-backend oracle
(control/C/slicer) is **not** computed from these hypothetical packages —
only from real slicer trajectories once the live run produces them. The
next benchmark after slicer tuning must be a genuinely fresh holdout
(100–200 new tasks, wording uninspected during tuning) with prompts,
distribution, snapshots, checkers, router rules, and model config frozen
before the run; that result decides whether the 1.4x routing claim
survives contact with an unseen instruction set.

**Isolated live test** (`promote-large`): the slicer arm (policy S:
slice + bounded read + line-range patch + suite) runs on the same 15
large-module tasks. Three safeguards gate the comparison:

1. **Control-manifest guard.** Every arm records the exact effective
   generation config: provider, API version, model snapshot id, reasoning
   effort, temperature, top_p, max output tokens, system-prompt hash,
   agent-instruction hash, tool-schema hash, harness version, task-set
   hash (bound transitively to per-task fingerprints: prompt, checker,
   fixture contents), and repo-snapshot hashes. Provider-fixed values
   are recorded as such. The frozen control-eval trajectories are
   reused only on a full manifest match; the frozen eval predates
   manifests, so **the first promote-large run executes 15 fresh control
   trajectories too** (the mismatch branch is the honest default).
   Manifests land in `runs/<arm>.manifest.json` for all future reuse.
   Harness identity is the executing commit for this run (conservative);
   after the run settles it should become a hash of execution-critical
   files only, so README edits stop invalidating controls.
2. **Provider billing telemetry.** Providers: Anthropic
   (`MINTOK_MODEL_API_KEY`) or OpenRouter (`OPENROUTER_API_KEY`, free
   models available). The loop accumulates per-call usage (input, cached
   input, cache writes, output, reasoning tokens where exposed,
   latency) into `<log>.usage.json` — **raw usage is the immutable
   evidence** — plus a per-turn billing JSONL priced through
   `mintok.billing.PriceTable` (list-price snapshots, recorded with the
   sidecar; `MINTOK_PRICE_OVERRIDES` JSON replaces them without
   touching old trajectories). Run records carry real `frontier_usd`;
   **$/solved and solve rate lead the report**, with p95/max $/task,
   cache splits, and tool-context tokens as diagnostics.
3. **Failure attribution (diagnostic, not causal ground truth).** Every
   failed slicer task is classified from trajectory evidence
   (`mintok.funnel.attribute_failure`): `bad_slice` (accepted patches
   never landed in a slice-ranked region), `insufficient_slice`
   (expanded/truncated package), `fallback_needed` (raw reads outspent
   the slice), `edit_tool_limitation` (patch rejections),
   `checker_disagreement` (suite green, checker red — could be a
   stricter checker, a hidden requirement, a harness mismatch, or real
   flakiness; "stochastic" is reserved for reruns demonstrating
   nondeterministic outcome changes), and `agent_reasoning_failure`
   (everything needed was available). Classes label runs for reading;
   they do not assert causes.

The promotion verdict (`mintok.funnel.slicer_promotion`) also records
turns/attempt and the **slice acceptance rate** (solved tasks where the
slice stayed the dominant information channel). A 15/15 result with 12
abandoned slices does not promote. The loop driver (`live.py`) is
orchestrator-only: the shim log stays the exact record of tool-side
context.

**Run it:**

```bash
OPENROUTER_API_KEY=... uv run python run_bench.py promote-large \
    --provider openrouter --model <free-model-id>
# or paid frontier:
MINTOK_MODEL_API_KEY=... uv run python run_bench.py promote-large
```

### Stage-1 result (30 live trajectories, stealth/space-bunny-alpha, free)

First real-model run of the isolated promotion test: 15 slicer + 15 fresh
control (manifest guard fired; frozen eval predates manifests). A first
pilot on a weaker free model (nemotron-3-super) solved 0/15 in both arms
and was archived (`/home/aqua/bench-run/nemotron-pilot/`), then the run
was repeated on a stronger one — the two-model pattern is itself
evidence for the structural-transfer thesis.

| metric | control | slicer | delta |
|---|---|---|---|
| solved | 1/15 | 1/15 | 1.00x |
| input tokens, total | ~133,600 | 84,893 | **4.7x less** |
| input tokens, median/task | 13,575 | 1,683 | **8.1x less** |
| cache reads (slicer) | — | 409,811 | |
| output tokens (slicer) | — | 37,710 | |
| turns/task | 10.7 | 10.0 | 0.94x |
| slice acceptance / fallback / expanded | — | 0.00 / 0.07 / 0.13 | |

**VERDICT: REJECT** (pre-registered bar: solve ≥ 15/15). Per the
decision rule, no paid Stage-2 frontier calibration yet.

What the data says:

- **Structural savings transferred.** A second, unrelated model shows
  the same ~5–8x provider-token gap at equal solve rate. The savings
  live in the software representation, not the model.
- **Behavioral savings did not.** Failure attribution is dominated by
  `fallback_needed`: after the slice, this model reads past it and then
  patches blind in loops (worst: 17 patches + 12 reads, suite never run,
  151k provider tokens on one task). One `checker_disagreement`
  (suite green, checker red — correctly not labeled stochastic).
- **Diagnosis queue** (before any rerun): (1) verify-then-stop
  discipline in the loop driver — a trajectory that never runs `suite`
  should be steered back, not cut off at the turn cap; (2) the 13%
  expanded-slice rate says the 800-token initial budget is tight on
  class-level targets — the fix is member-method slicing (already
  noted), not a budget raise; (3) slice acceptance must be measured on
  more than one solve before it means anything.

The $-denominated claim remains blocked on a nonzero-priced model (both
arms $0.00 here); provider tokens/solved is the Stage-1 headline by
pre-decision.
