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
