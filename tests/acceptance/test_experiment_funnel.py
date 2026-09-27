from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from mintok.funnel import (
    FAST8_IDS,
    AdaptivePool,
    ControlCache,
    PairedOutcome,
    RunManifest,
    SlicerRun,
    TaskIntegrityError,
    attribute_failure,
    control_key,
    fast8_classes,
    fast8_suite,
    manifests_compatible,
    next_phase,
    phase_suite,
    sequential_verdict,
    slicer_promotion,
    task_fingerprint,
    verify_fingerprint,
)

scenarios("experiment_funnel.feature")


def make_pair(task: str, ratio: float = 1.0, arm_solved: bool = True, control_solved: bool = True):
    base = 1000
    return PairedOutcome(
        task_id=task,
        control_solved=control_solved,
        arm_solved=arm_solved,
        control_tokens=base,
        arm_tokens=int(base * ratio),
    )


@given("the full task set contains the FAST-8 tasks")
def full_task_set(ctx: SimpleNamespace) -> None:
    ctx.available = sorted(FAST8_IDS) + ["shopcart-api-01", "notesrv-lookup-02"]


@when("the FAST-8 suite is selected from the full task set")
def select_fast8(ctx: SimpleNamespace) -> None:
    ctx.fast8 = fast8_suite(ctx.available)
    ctx.classes = fast8_classes()


@then("it contains exactly 8 tasks")
def eight_tasks(ctx: SimpleNamespace) -> None:
    assert len(ctx.fast8) == 8
    assert len(set(ctx.fast8)) == 8


@then(parsers.parse("it covers {klass}"))
def covers_class(ctx: SimpleNamespace, klass: str) -> None:
    name = klass
    for article in ("an ", "a ", "the "):
        if name.startswith(article):
            name = name[len(article):]
            break
    if name.endswith(" task"):
        name = name[: -len(" task")]
    assert name.replace(" ", "_").replace("-", "_") in ctx.classes.values(), ctx.classes


@then("every selected task exists in the full task set")
def all_exist(ctx: SimpleNamespace) -> None:
    assert set(ctx.fast8) <= set(ctx.available)


@given(
    parsers.parse(
        "{n:d} paired tasks where the arm uses {ratio}x the control tokens "
        "and {solve_constraint}"
    )
)
def paired_tasks(ctx: SimpleNamespace, n: int, ratio: str, solve_constraint: str) -> None:
    arm_solved = True
    ctx.pairs = []
    for i in range(n):
        if "fewer" in solve_constraint and i == 0:
            arm_solved = False  # 4/5 vs 5/5 at n=5: 20 points fewer
        ctx.pairs.append(
            make_pair(f"t{i}", ratio=float(ratio), arm_solved=arm_solved, control_solved=True)
        )


@when(parsers.parse("the sequential verdict is computed after the {n:d}th pair"))
def compute_verdict(ctx: SimpleNamespace, n: int) -> None:
    ctx.verdict = sequential_verdict(ctx.pairs[:n])


@then("the arm is killed")
def killed(ctx: SimpleNamespace) -> None:
    assert ctx.verdict.action == "kill", ctx.verdict


@then("the arm continues")
def continues(ctx: SimpleNamespace) -> None:
    assert ctx.verdict.action == "continue", ctx.verdict


@then(parsers.parse("the reason mentions {fragment}"))
def reason_mentions(ctx: SimpleNamespace, fragment: str) -> None:
    fragment = fragment.removeprefix("the ")
    assert fragment in ctx.verdict.reason, ctx.verdict.reason


@when(parsers.parse("the arm reaches 15 paired tasks at {ratio}x the control tokens"))
def arm_reaches_15(ctx: SimpleNamespace, ratio: str) -> None:
    ctx.pairs = [make_pair(f"t{i}", ratio=float(ratio)) for i in range(15)]


@given('a completed control run for task "t1" with model "m" and effort "high"')
def completed_control(ctx: SimpleNamespace, tmp_path) -> None:
    ctx.cache = ControlCache(tmp_path / "control-cache")
    ctx.key = control_key("t1", "repo-hash", "harness-1", "m", "high", "control")
    ctx.cache.put(ctx.key, {"task_id": "t1", "solved": True, "input_tokens": 1200})


@when("a control run for the same task, model, effort, harness, and toolset is requested")
def request_same_control(ctx: SimpleNamespace) -> None:
    runs: list[dict] = []

    def run() -> dict:
        runs.append({"task_id": "t1", "fresh": True})
        return runs[0]

    ctx.record, ctx.from_cache = ctx.cache.get_or_run(ctx.key, run)
    ctx.fresh_runs = len(runs)


@then("the cache serves the stored record without a new run")
def cache_hit(ctx: SimpleNamespace) -> None:
    assert ctx.from_cache is True and ctx.fresh_runs == 0


@when('a control run for the same task with model "other" is requested')
def request_other_control(ctx: SimpleNamespace) -> None:
    ctx.other_key = control_key("t1", "repo-hash", "harness-1", "other", "high", "control")


@then("the cache reports a miss")
def cache_miss(ctx: SimpleNamespace) -> None:
    assert ctx.cache.get(ctx.other_key) is None


@given("a fingerprinted task with instruction, checker, and fixture files")
def fingerprinted_task(ctx: SimpleNamespace, tmp_path) -> None:
    fixture_root = tmp_path / "fixture"
    (fixture_root / "src").mkdir(parents=True)
    (fixture_root / "src" / "mod.py").write_text("x = 1\n")
    ctx.task = {
        "id": "demo",
        "instruction": "do the thing",
        "check": "assert True",
        "repo": "demo",
    }
    ctx.fixture_root = fixture_root
    ctx.stored = task_fingerprint(ctx.task, fixture_root)


@when("the same task is verified against its fingerprint")
def verify_same(ctx: SimpleNamespace) -> None:
    verify_fingerprint(ctx.task, ctx.stored, ctx.fixture_root)  # must not raise
    ctx.verified = True


@then("verification passes")
def verification_passes(ctx: SimpleNamespace) -> None:
    assert ctx.verified is True


@when("the checker text is edited and the task is verified again")
def verify_drifted(ctx: SimpleNamespace) -> None:
    ctx.task["check"] = "assert False"
    try:
        verify_fingerprint(ctx.task, ctx.stored, ctx.fixture_root)
        ctx.drift_error = None
    except TaskIntegrityError as exc:
        ctx.drift_error = exc


@then("verification fails with a drift error naming the changed part")
def drift_named(ctx: SimpleNamespace) -> None:
    assert ctx.drift_error is not None
    assert "checker_hash" in str(ctx.drift_error)


@when("a fresh adaptive pool is created")
def fresh_pool(ctx: SimpleNamespace) -> None:
    ctx.pool = AdaptivePool()


@then(parsers.parse("it allows {n:d} concurrent workers"))
def pool_size(ctx: SimpleNamespace, n: int) -> None:
    assert ctx.pool.size == n


@when("three clean waves complete without throttling")
def clean_waves(ctx: SimpleNamespace) -> None:
    for _ in range(3):
        ctx.pool.wave_completed(throttled=False)


@when("the provider throttles a wave")
def throttled_wave(ctx: SimpleNamespace) -> None:
    ctx.pool.wave_completed(throttled=True)


@then("it never exceeds 12 or drops below 4")
def pool_bounds(ctx: SimpleNamespace) -> None:
    assert 4 <= ctx.pool.size <= 12


@given(parsers.parse('an arm with verdict history "{history}"'))
def verdict_history(ctx: SimpleNamespace, history: str) -> None:
    ctx.history = history


@when("the funnel advances the arm")
def advance(ctx: SimpleNamespace) -> None:
    ctx.phase = ctx.history


@then(parsers.parse('the arm is in phase "{phase}"'))
def phase_is(ctx: SimpleNamespace, phase: str) -> None:
    mapping = {
        "fresh": None,
        "smoke8_pass": ("smoke8", False),
        "dev15_pass": ("dev15", False),
        "confirm30_pass": ("confirm30", False),
        "eval120_pass": ("eval120", False),
        "smoke8_killed": ("smoke8", True),
        "dev15_killed": ("dev15", True),
        "confirm30_killed": ("confirm30", True),
    }
    if ctx.history == "fresh":
        assert phase_suite("smoke8", sorted(FAST8_IDS))  # smoke is runnable
        return
    current, killed = mapping[ctx.history]
    assert next_phase(current, killed) == phase


@given("slicer runs on 15 large-module tasks:")
def slicer_runs(ctx: SimpleNamespace, datatable: list[list[str]]) -> None:
    from tests.acceptance.helpers import table_rows

    ctx.slicer_runs = [
        SlicerRun(
            task_id=row["task"],
            solved=row["solved"] == "yes",
            tokens=int(row["tokens"]),
            turns=int(row["turns"]),
            slice_tokens=int(row["slice_tok"]),
            fallback_tokens=int(row["fallback_tok"]),
            expanded=row["expanded"] == "yes",
        )
        for row in table_rows(datatable)
    ]


@when(parsers.parse("the slicer promotion verdict is computed with solve target {target:d}"))
def compute_promotion(ctx: SimpleNamespace, target: int) -> None:
    ctx.promotion = slicer_promotion(ctx.slicer_runs, solve_target=target)


@then(parsers.parse('the verdict is "{verdict}"'))
def promotion_verdict(ctx: SimpleNamespace, verdict: str) -> None:
    assert ctx.promotion.verdict == verdict, ctx.promotion


@then(parsers.parse("the slice acceptance rate is {value:f}"))
def acceptance_rate(ctx: SimpleNamespace, value: float) -> None:
    assert round(ctx.promotion.acceptance_rate, 2) == value, ctx.promotion


@then(parsers.parse("the raw-source fallback rate is {value:f}"))
def fallback_rate(ctx: SimpleNamespace, value: float) -> None:
    assert round(ctx.promotion.fallback_rate, 2) == value, ctx.promotion


@then(parsers.parse("the expanded-slice rate is {value:f}"))
def expanded_rate(ctx: SimpleNamespace, value: float) -> None:
    assert round(ctx.promotion.expanded_rate, 2) == value, ctx.promotion


@given(parsers.parse('a frozen control manifest with model "{model}" and prompt hash "{prompt}"'))
def frozen_manifest(ctx: SimpleNamespace, model: str, prompt: str) -> None:
    ctx.frozen_manifest = RunManifest(
        provider="anthropic", model=model, reasoning_effort="none",
        system_prompt_hash=prompt, toolset_hash="t", harness_version="h",
        task_set_hash="tasks",
    )


@when(parsers.parse('a live slicer manifest arrives with model "{model}" and prompt hash "{prompt}"'))
def live_manifest(ctx: SimpleNamespace, model: str, prompt: str) -> None:
    live = RunManifest(
        provider="anthropic", model=model, reasoning_effort="none",
        system_prompt_hash=prompt, toolset_hash="t", harness_version="h",
        task_set_hash="tasks",
    )
    ctx.compatible, ctx.reasons = manifests_compatible(ctx.frozen_manifest, live)


@then("the frozen control is compatible")
def control_compatible(ctx: SimpleNamespace) -> None:
    assert ctx.compatible, ctx.reasons


@then(parsers.parse('the frozen control is refused with reason "{fragment}"'))
def control_refused(ctx: SimpleNamespace, fragment: str) -> None:
    assert not ctx.compatible, ctx.reasons
    assert any(fragment in r for r in ctx.reasons), ctx.reasons


@given("no frozen control manifest exists")
def no_frozen_manifest(ctx: SimpleNamespace) -> None:
    ctx.frozen_manifest = None


@when("the comparison is planned")
def plan_comparison(ctx: SimpleNamespace) -> None:
    ctx.compatible, ctx.reasons = manifests_compatible(
        ctx.frozen_manifest,
        RunManifest(
            provider="anthropic", model="m", reasoning_effort="none",
            system_prompt_hash="p", toolset_hash="t", harness_version="h",
            task_set_hash="tasks",
        ),
    )


@then("the run demands fresh control trajectories")
def demands_fresh(ctx: SimpleNamespace) -> None:
    assert not ctx.compatible, ctx.reasons


@given(parsers.parse(
    "a failed slicer run where target_in_slice is {target}, slice_dominated is {dominated}, "
    "slice_truncated is {truncated}, edit_rejections is {rejects:d}, and suite_ok is {suite}"
))
def failed_run_flags(ctx: SimpleNamespace, target: str, dominated: str, truncated: str, rejects: int, suite: str) -> None:
    ctx.failure_flags = dict(
        solved=False,
        target_in_slice=target == "yes",
        slice_dominated=dominated == "yes",
        slice_truncated=truncated == "yes",
        edit_rejections=rejects,
        suite_ok=suite == "yes",
    )


@then(parsers.parse('the failure class is "{klass}"'))
def failure_class_is(ctx: SimpleNamespace, klass: str) -> None:
    assert attribute_failure(**ctx.failure_flags) == klass


_REPO_ROOT = Path(__file__).resolve().parents[2]


@given('two task copies checked in sequence, both defining module "biglib"')
def two_copies(ctx: SimpleNamespace, tmp_path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mintok_bench_run_bench", _REPO_ROOT / "benchmarks" / "e2e" / "run_bench.py"
    )
    assert spec is not None and spec.loader is not None
    ctx.run_bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx.run_bench)

    def make_copy(name: str, value: str):
        root = tmp_path / name
        (root / "src" / "biglib").mkdir(parents=True)
        (root / "src" / "biglib" / "__init__.py").write_text("")
        (root / "src" / "biglib" / "val.py").write_text(f"VALUE = {value!r}\n")
        tdir = root / "tests"
        tdir.mkdir()
        (tdir / "test_ok.py").write_text("def test_ok():\n    assert True\n")
        log = root / "run.jsonl"
        log.write_text("")  # empty trajectory: check still must run
        return root, log

    # first copy's module state satisfies only the first checker
    ctx.copy_a, ctx.log_a = make_copy("copy-a", "one")
    ctx.copy_b, ctx.log_b = make_copy("copy-b", "two")
    ctx.checker_a = "import sys\nfrom biglib.val import VALUE\nassert VALUE == 'one'\n"
    ctx.checker_b = "import sys\nfrom biglib.val import VALUE\nassert VALUE == 'two'\n"


@given("the first copy's module state satisfies only the first checker")
def first_copy_satisfies_first_only(ctx: SimpleNamespace) -> None:
    assert "one" in (ctx.copy_a / "src" / "biglib" / "val.py").read_text()
    assert "two" in (ctx.copy_b / "src" / "biglib" / "val.py").read_text()


@when("each copy's checker runs after its trajectory")
def run_both_checkers(ctx: SimpleNamespace) -> None:
    ctx.score_a = ctx.run_bench.check_task(ctx.copy_a, "iso-a", ctx.checker_a, ctx.log_a)
    ctx.score_b = ctx.run_bench.check_task(ctx.copy_b, "iso-b", ctx.checker_b, ctx.log_b)


@then("the second checker sees the second copy's module state")
def second_checker_sees_own_copy(ctx: SimpleNamespace) -> None:
    assert ctx.score_b["check_ok"] is True, ctx.score_b["detail"]


@then("a checker that passes is recorded as solved regardless of position")
def pass_is_solved_at_any_position(ctx: SimpleNamespace) -> None:
    assert ctx.score_b["solved"] is True


def _make_live(tmp_path, replies):
    """Build a run_loop with scripted completions over a tiny repo."""
    import importlib.util
    import json as jsonlib

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "big.py").write_text("def f():\n    return 1\n")
    log = tmp_path / "t.jsonl"

    spec = importlib.util.spec_from_file_location(
        "live_for_verify", _REPO_ROOT / "benchmarks" / "e2e" / "live.py"
    )
    assert spec is not None and spec.loader is not None
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    iterator = iter(replies)

    def completion(model, system, messages, tools=None):
        return next(iterator)

    return live, root, log, completion


@pytest.fixture
def patch_only_replies():
    """A model that patches forever and never verifies."""
    return [
        (
            "tool_use",
            [{"type": "tool_use", "id": f"t{i}", "name": "patch",
              "input": {"file": "src/big.py", "start": 1, "end": 1,
                        "source": f"def f():\n    return {i}\n"}}],
            None,
        )
        for i in range(10)
    ]


@given("a live loop with a 4-turn budget whose model only ever patches")
def live_patch_only(ctx: SimpleNamespace, tmp_path, patch_only_replies) -> None:
    ctx.live, ctx.root, ctx.log, ctx.completion = _make_live(tmp_path, patch_only_replies)
    ctx.max_turns = 4


@given("a live loop with a 4-turn budget whose model runs the suite after each patch")
def live_verifying_model(ctx: SimpleNamespace, tmp_path) -> None:
    replies = []
    for i in range(4):
        replies.append((
            "tool_use",
            [{"type": "tool_use", "id": f"p{i}", "name": "patch",
              "input": {"file": "src/big.py", "start": 1, "end": 1,
                        "source": f"def f():\n    return {i}\n"}}],
            None,
        ))
        replies.append((
            "tool_use",
            [{"type": "tool_use", "id": f"s{i}", "name": "suite", "input": {}}],
            None,
        ))
    ctx.live, ctx.root, ctx.log, ctx.completion = _make_live(tmp_path, replies)
    ctx.max_turns = 4


@when("the trajectory exhausts the budget without ever running the suite")
def exhaust_budget_dirty(ctx: SimpleNamespace) -> None:
    ctx.summary = ctx.live.run_loop(
        ctx.root, ctx.log, "S", "fix f", "fake", max_turns=ctx.max_turns,
        completion=ctx.completion,
    )


@when("the trajectory exhausts the budget")
def exhaust_budget_clean(ctx: SimpleNamespace) -> None:
    ctx.summary = ctx.live.run_loop(
        ctx.root, ctx.log, "S", "fix f", "fake", max_turns=ctx.max_turns,
        completion=ctx.completion,
    )


@then("the harness runs the suite itself exactly once")
def forced_suite_once(ctx: SimpleNamespace) -> None:
    import json as jsonlib

    entries = [jsonlib.loads(l) for l in ctx.log.read_text().splitlines() if l.strip()]
    forced = [e for e in entries if e["tool"] == "suite"]
    assert len(forced) == 1, [e["tool"] for e in entries]


@then("the model gets one reaction turn after the forced verification")
def reaction_turn(ctx: SimpleNamespace) -> None:
    # turns = budget + forced verification turn; the loop then terminates
    assert ctx.summary["turns"] == ctx.max_turns + 1, ctx.summary


@then("the forced suite call is recorded in the shim log")
def forced_in_shim_log(ctx: SimpleNamespace) -> None:
    import json as jsonlib

    entries = [jsonlib.loads(l) for l in ctx.log.read_text().splitlines() if l.strip()]
    assert any(e["tool"] == "suite" for e in entries)


@then("the harness adds no forced verification")
def no_forced_verification(ctx: SimpleNamespace) -> None:
    import json as jsonlib

    entries = [jsonlib.loads(l) for l in ctx.log.read_text().splitlines() if l.strip()]
    model_suites = sum(1 for e in entries if e["tool"] == "suite")
    assert model_suites == 2, [e["tool"] for e in entries]  # exactly the model's own


@when("the holdout task suite is verified against its frozen fingerprints")
def verify_holdout_suite(ctx: SimpleNamespace) -> None:
    import json as jsonlib

    bench_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "e2e"
    tasks_path = bench_dir / "tasks_holdout.json"
    fp_path = bench_dir / "holdout_fingerprints.json"
    fixtures_dir = bench_dir / "fixtures"

    tasks = jsonlib.loads(tasks_path.read_text())
    fingerprints = jsonlib.loads(fp_path.read_text())

    ctx.verified_holdout = []
    for t in tasks:
        tid = t["id"]
        fp = fingerprints[tid]
        repo_dir = fixtures_dir / t["repo"]
        verify_fingerprint(t, fp, repo_dir)
        ctx.verified_holdout.append(tid)


@then("all 40 holdout tasks pass integrity verification")
def all_40_pass(ctx: SimpleNamespace) -> None:
    assert len(ctx.verified_holdout) == 40


@given("a holdout task prepared for benchmark execution")
def prepare_holdout_task(ctx: SimpleNamespace, tmp_path: Path) -> None:
    import shutil

    bench_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "e2e"
    fixtures_dir = bench_dir / "fixtures"
    dest = tmp_path / "workspace"
    src = fixtures_dir / "ledger"
    shutil.copytree(src, dest)
    ctx.workspace = dest


@when("the workspace directory is inspected")
def inspect_workspace(ctx: SimpleNamespace) -> None:
    pass


@then("no solution files exist in the workspace")
def no_solutions(ctx: SimpleNamespace) -> None:
    forbidden = ["holdout_solutions.json", "tasks_holdout.json", "holdout_fingerprints.json"]
    for name in forbidden:
        assert not (ctx.workspace / name).exists()
        assert not any(ctx.workspace.rglob(name))


@then("no checker source exists in the workspace")
def no_checkers(ctx: SimpleNamespace) -> None:
    for py_file in ctx.workspace.rglob("*.py"):
        content = py_file.read_text()
        assert "assert clip_" not in content
        assert "assert flags_" not in content
        assert "assert cap_" not in content


