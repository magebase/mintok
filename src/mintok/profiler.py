"""Inference profiler: quantify avoidable spend in agent session records.

The profiler optimizes nothing. It makes waste visible — the free first stage
of the open-core funnel. Its formulas are published and intentionally simple
heuristics; categories may overlap, so they are never multiplied together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mintok.records import SessionRecord

DETERMINISTIC_TASK_CLASSES = frozenset({"deterministic"})
POOR_CACHE_THRESHOLD = 0.5
POOR_CACHE_SHARE = 0.25

FORMULAS = {
    "repeated context": "price every read beyond the first at the session's average cost per token",
    "overpowered model": "flags the full cost of frontier sessions on deterministic task classes",
    "poor cache use": "flags 25% of cost when fewer than half the input tokens hit cache",
}
CATEGORY_ORDER = ("repeated context", "overpowered model", "poor cache use")


@dataclass(frozen=True, slots=True)
class Profile:
    sessions: int
    total: float
    categories: dict[str, float] = field(default_factory=dict)

    @property
    def raw_categories_total(self) -> float:
        return sum(self.categories.values())

    @property
    def categories_overlap(self) -> bool:
        """True when the same dollar was flagged by more than one heuristic."""
        return self.raw_categories_total > self.total

    @property
    def avoidable(self) -> float:
        """Upper bound on recoverable spend, capped at total spend."""
        return min(self.raw_categories_total, self.total)

    @property
    def improvement_multiple(self) -> float:
        remaining = self.total - self.avoidable
        return self.total / remaining if remaining > 0 else float("inf")


def profile_sessions(sessions: Iterable[SessionRecord]) -> Profile:
    sessions = list(sessions)

    repeated = 0.0
    for session in sessions:
        extra_tokens = sum((read.count - 1) * read.tokens for read in session.context_reads)
        repeated += extra_tokens * session.cost_per_token

    overpowered = sum(s.usd for s in sessions if s.task_class in DETERMINISTIC_TASK_CLASSES)
    poor_cache = sum(s.usd * POOR_CACHE_SHARE for s in sessions if s.cache_read_ratio < POOR_CACHE_THRESHOLD)

    return Profile(
        sessions=len(sessions),
        total=sum(s.usd for s in sessions),
        categories={
            "repeated context": repeated,
            "overpowered model": overpowered,
            "poor cache use": poor_cache,
        },
    )


def render_profile(profile: Profile) -> str:
    lines = [f"Inference profile ({profile.sessions} sessions)"]
    lines.append(f"Total spend              ${profile.total:,.2f}")
    for category in CATEGORY_ORDER:
        lines.append(f"  {category:<23}${profile.categories.get(category, 0.0):,.2f}")
    lines.append(f"Estimated avoidable      ${profile.avoidable:,.2f}")
    multiple = profile.improvement_multiple
    lines.append(
        "Potential improvement    n/a (all spend flagged)"
        if multiple == float("inf")
        else f"Potential improvement    {multiple:.2f}x"
    )
    if profile.categories_overlap:
        lines.append("note: waste categories may overlap; avoidable is capped at total spend")
    return "\n".join(lines)


@dataclass
class TokenWaterfall:
    """Itemized token allocation waterfall for an agent trajectory."""

    frontier_instructions: int = 0
    task: int = 0
    repo_profile: int = 0
    source: int = 0
    semantic_packets: int = 0
    shell_digests: int = 0
    expanded_observations: int = 0
    conversation_state: int = 0
    history_replay: int = 0
    frontier_output: int = 0
    verification: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return (
            self.frontier_instructions
            + self.task
            + self.repo_profile
            + self.source
            + self.semantic_packets
            + self.shell_digests
            + self.expanded_observations
            + self.conversation_state
            + self.history_replay
            + self.frontier_output
            + self.verification
            + self.other
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "frontier_instructions": self.frontier_instructions,
            "task": self.task,
            "repo_profile": self.repo_profile,
            "source": self.source,
            "semantic_packets": self.semantic_packets,
            "shell_digests": self.shell_digests,
            "expanded_observations": self.expanded_observations,
            "conversation_state": self.conversation_state,
            "history_replay": self.history_replay,
            "frontier_output": self.frontier_output,
            "verification": self.verification,
            "other": self.other,
        }

    def render(self) -> str:
        lines = [
            f"TOTAL                    {self.total:>10,d}",
            f"frontier instructions    {self.frontier_instructions:>10,d}",
            f"task                     {self.task:>10,d}",
            f"repo profile             {self.repo_profile:>10,d}",
            f"source                   {self.source:>10,d}",
            f"semantic packets         {self.semantic_packets:>10,d}",
            f"shell digests            {self.shell_digests:>10,d}",
            f"expanded observations    {self.expanded_observations:>10,d}",
            f"conversation state       {self.conversation_state:>10,d}",
            f"history replay           {self.history_replay:>10,d}",
            f"frontier output          {self.frontier_output:>10,d}",
            f"verification             {self.verification:>10,d}",
            f"other                    {self.other:>10,d}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class OracleMinimum:
    """Post-hoc minimal known sufficient evidence chain for a task."""

    task_id: str
    task_tokens: int
    decisive_source_tokens: int
    decisive_failure_tokens: int
    patch_tokens: int
    verification_tokens: int

    @property
    def minimum_tokens(self) -> int:
        return (
            self.task_tokens
            + self.decisive_source_tokens
            + self.decisive_failure_tokens
            + self.patch_tokens
            + self.verification_tokens
        )

    @property
    def known_sufficient_tokens(self) -> int:
        """Alias for post-hoc minimal known sufficient evidence chain."""
        return self.minimum_tokens

    def amplification_factor(self, actual_tokens: int) -> float:
        """Inference amplification factor: A = T_actual / T_known-sufficient."""
        min_tok = max(1, self.minimum_tokens)
        return actual_tokens / min_tok

    def inference_amplification_factor(self, actual_tokens: int) -> float:
        return self.amplification_factor(actual_tokens)


def profile_trajectory_waterfall(events: list[dict[str, Any]]) -> TokenWaterfall:
    """Classify trajectory events into token waterfall categories."""
    wf = TokenWaterfall()
    for e in events:
        tool = e.get("tool", "")
        toks = e.get("tokens", e.get("package_tokens", 0))
        output = str(e.get("output", ""))

        if tool in ("instruction", "system"):
            wf.frontier_instructions += toks
        elif tool == "task":
            wf.task += toks
        elif tool == "repo_profile" or "[repository profile:" in output:
            wf.repo_profile += toks
        elif tool == "read" or "diff --git" in output:
            wf.source += toks
        elif tool in ("slice", "investigate_failure", "localize_symbol") or "[evidence packet:" in output or "[symbol packet:" in output:
            wf.semantic_packets += toks
        elif tool in ("shell", "grep", "find_files") and ("obs:" in output or "unchanged:" in output):
            wf.shell_digests += toks
        elif tool == "expand" or "[obs:" in output:
            wf.expanded_observations += toks
        elif "Canonical Working State" in output or tool == "state":
            wf.conversation_state += toks
        elif tool == "assistant" or e.get("role") == "assistant":
            wf.frontier_output += toks
        elif tool in ("suite", "verify") or "test session starts" in output:
            wf.verification += toks
        else:
            wf.other += toks
    return wf


@dataclass(frozen=True, slots=True)
class CostWaterfall:
    """Itemized dollar cost breakdown across provider billing dimensions."""

    fresh_input_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0
    output_usd: float = 0.0
    reasoning_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return (
            self.fresh_input_usd
            + self.cache_read_usd
            + self.cache_write_usd
            + self.output_usd
            + self.reasoning_usd
        )


@dataclass(frozen=True, slots=True)
class TrajectoryEfficiencyMetrics:
    """Detailed waste, tax, and timing metrics for an agent trajectory."""

    task_id: str
    total_tokens: int
    dead_token_ratio: float = 0.0
    tokens_to_first_decisive_evidence: int = 0
    post_decisive_waste_tokens: int = 0
    rediscovery_tax_tokens: int = 0
    navigation_tax_tokens: int = 0
    recovery_tax_tokens: int = 0
    policy_regret_usd: float = 0.0


class LostSolveCategory:
    """Attribution taxonomy for tasks solved by Control baseline but missed by MinTok."""

    VIRTUALIZATION_OMISSION = "virtualization_omission"
    STATE_COMPACTION_OMISSION = "state_compaction_omission"
    SOURCE_DEDUP_SUPPRESSION = "source_dedup_suppression"
    SEMANTIC_PACKET_PRUNING = "semantic_packet_pruning"
    VERIFICATION_UNDER_TESTING = "verification_under_testing"
    PREMATURE_EARLY_STOP = "premature_early_stop"
    CONTROLLER_MISROUTING = "controller_misrouting"
    MODEL_VARIANCE = "model_variance"


@dataclass(frozen=True, slots=True)
class LostSolveAttribution:
    """Attribution record explaining a lost solve."""

    task_id: str
    category: str
    explanation: str
    tokens_saved: int = 0


def attribute_lost_solve(
    task_id: str,
    control_solved: bool,
    mintok_solved: bool,
    early_stopped: bool = False,
    verification_missed: bool = False,
    packet_omitted: bool = False,
    source_suppressed: bool = False,
    virtualization_omitted: bool = False,
    tokens_saved: int = 0,
) -> LostSolveAttribution | None:
    """Classify the root cause of a Control-only solve."""
    if not control_solved or mintok_solved:
        return None

    if early_stopped:
        return LostSolveAttribution(
            task_id=task_id,
            category=LostSolveCategory.PREMATURE_EARLY_STOP,
            explanation="Early stop continuation hazard aborted before task resolution",
            tokens_saved=tokens_saved,
        )
    if verification_missed:
        return LostSolveAttribution(
            task_id=task_id,
            category=LostSolveCategory.VERIFICATION_UNDER_TESTING,
            explanation="Targeted test passed while broader test suite had unresolved regressions",
            tokens_saved=tokens_saved,
        )
    if packet_omitted:
        return LostSolveAttribution(
            task_id=task_id,
            category=LostSolveCategory.SEMANTIC_PACKET_PRUNING,
            explanation="AST neighborhood pruning omitted critical caller containing bug",
            tokens_saved=tokens_saved,
        )
    if source_suppressed:
        return LostSolveAttribution(
            task_id=task_id,
            category=LostSolveCategory.SOURCE_DEDUP_SUPPRESSION,
            explanation="Source deduplication suppressed code span required for semantic fix",
            tokens_saved=tokens_saved,
        )
    if virtualization_omitted:
        return LostSolveAttribution(
            task_id=task_id,
            category=LostSolveCategory.VIRTUALIZATION_OMISSION,
            explanation="Tool output digest omitted key error message line",
            tokens_saved=tokens_saved,
        )

    return LostSolveAttribution(
        task_id=task_id,
        category=LostSolveCategory.MODEL_VARIANCE,
        explanation="Stochastic model completion variance on identical evidence",
        tokens_saved=tokens_saved,
    )

