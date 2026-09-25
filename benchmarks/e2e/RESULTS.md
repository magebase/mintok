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
