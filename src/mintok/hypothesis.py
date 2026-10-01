"""Hypothesis Graph and Negative Evidence Memory for MinTok.

Enables MinTok to operate as an empirical information acquisition optimizer:
1. HypothesisGraph:
   - Hierarchical hypothesis tree: H1 -> H1a -> H1a-i.
   - Confidence scoring and Bayesian / log-odds posterior updates.
   - Action scoring by discriminating power (information gain / entropy reduction).
2. NegativeEvidenceMemory:
   - Explicit ledger of verified negative facts (e.g. NO_CALLER_WRITES_TIMEOUT,
     NO_IMPORT_CYCLE, NO_ATTRIBUTE_OVERRIDE).
   - Prevents catastrophic memory amnesia and redundant re-exploration of ruled-out code paths.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class HypothesisStatus(str, Enum):
    """Lifecycle status of a diagnostic hypothesis."""

    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class HypothesisNode:
    """A node in the hierarchical hypothesis tree."""

    id: str
    description: str
    confidence: float = 0.5  # 0.0 to 1.0
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    refuting_evidence: list[str] = field(default_factory=list)
    discriminating_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "supporting_evidence": list(self.supporting_evidence),
            "refuting_evidence": list(self.refuting_evidence),
            "discriminating_features": dict(self.discriminating_features),
        }


class HypothesisGraph:
    """Hierarchical hypothesis tree tracking candidate failure root causes."""

    def __init__(self) -> None:
        self.nodes: dict[str, HypothesisNode] = {}
        self.root_ids: list[str] = []

    def add_hypothesis(
        self,
        hypothesis_id: str,
        description: str,
        confidence: float = 0.5,
        parent_id: str | None = None,
        discriminating_features: dict[str, Any] | None = None,
    ) -> HypothesisNode:
        """Add a hypothesis node into the graph."""
        if hypothesis_id in self.nodes:
            raise ValueError(f"Hypothesis '{hypothesis_id}' already exists")

        node = HypothesisNode(
            id=hypothesis_id,
            description=description,
            confidence=max(0.01, min(0.99, confidence)),
            status=HypothesisStatus.ACTIVE,
            parent_id=parent_id,
            discriminating_features=discriminating_features or {},
        )
        self.nodes[hypothesis_id] = node

        if parent_id is not None:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent hypothesis '{parent_id}' does not exist")
            self.nodes[parent_id].children_ids.append(hypothesis_id)
        else:
            self.root_ids.append(hypothesis_id)

        self._normalize_confidences()
        return node

    def get_active_hypotheses(self) -> list[HypothesisNode]:
        """Return all hypotheses currently marked as ACTIVE."""
        return [node for node in self.nodes.values() if node.status == HypothesisStatus.ACTIVE]

    def update_confidence(
        self,
        hypothesis_id: str,
        likelihood_ratio: float,
        evidence: str,
        supports: bool = True,
    ) -> None:
        """Update posterior confidence of a hypothesis using evidence likelihood ratio."""
        if hypothesis_id not in self.nodes:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found")

        node = self.nodes[hypothesis_id]
        if supports:
            node.supporting_evidence.append(evidence)
        else:
            node.refuting_evidence.append(evidence)

        # Bayesian odds update: posterior_odds = prior_odds * likelihood_ratio
        prior_p = max(0.001, min(0.999, node.confidence))
        prior_odds = prior_p / (1.0 - prior_p)
        post_odds = prior_odds * max(0.01, likelihood_ratio)
        post_p = post_odds / (1.0 + post_odds)

        node.confidence = max(0.01, min(0.99, post_p))

        # Check threshold status changes
        if node.confidence >= 0.90:
            node.status = HypothesisStatus.CONFIRMED
        elif node.confidence <= 0.05:
            node.status = HypothesisStatus.REFUTED

        self._normalize_confidences()

    def _normalize_confidences(self) -> None:
        """Normalize confidences across competing siblings under the same parent."""
        parent_groups: dict[str | None, list[HypothesisNode]] = {}
        for node in self.nodes.values():
            if node.status == HypothesisStatus.ACTIVE:
                parent_groups.setdefault(node.parent_id, []).append(node)

        for p_id, group in parent_groups.items():
            if len(group) > 1:
                total = sum(n.confidence for n in group)
                if total > 0:
                    for n in group:
                        n.confidence = max(0.01, min(0.99, n.confidence / total))

    def compute_entropy(self) -> float:
        """Compute Shannon entropy across active leaf hypotheses."""
        actives = self.get_active_hypotheses()
        if not actives:
            return 0.0
        tot = sum(n.confidence for n in actives)
        if tot <= 0:
            return 0.0
        entropy = 0.0
        for n in actives:
            p = n.confidence / tot
            if p > 1e-6:
                entropy -= p * math.log2(p)
        return entropy

    def evaluate_discriminating_power(
        self,
        action_type: str,
        target_features: dict[str, Any] | None = None,
    ) -> float:
        """Score an action by how effectively it partitions or tests active hypotheses.
        
        Actions that directly address features distinguishing rival hypotheses receive high scores.
        """
        actives = self.get_active_hypotheses()
        if len(actives) <= 1:
            return 0.1  # Minimal discriminating power needed if 1 or 0 hypotheses remain

        features = target_features or {}
        # Count how many active hypotheses specify discriminators matched by the action
        matched = 0
        differing = 0
        seen_vals = set()

        for h in actives:
            for k, v in h.discriminating_features.items():
                if k in features:
                    matched += 1
                    seen_vals.add(str(v))

        # If the action touches a feature where active hypotheses hold differing expectations,
        # discriminating power is maximized
        if len(seen_vals) > 1:
            return 0.95
        if matched > 0:
            return 0.70

        # General action heuristic
        if action_type in ("run_verifier", "query_symbol", "slice_diagnose_failure"):
            return 0.60
        return 0.30

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_ids": list(self.root_ids),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "entropy": round(self.compute_entropy(), 4),
        }


@dataclass(frozen=True, slots=True)
class NegativeFact:
    """A verified negative observation that rules out a hypothesis or search path."""

    fact_key: str
    target: str
    rationale: str
    confidence: float = 1.0
    recorded_at_turn: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_key": self.fact_key,
            "target": self.target,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 4),
            "recorded_at_turn": self.recorded_at_turn,
        }


class NegativeEvidenceMemory:
    """Ledger of verified negative evidence preventing redundant exploration."""

    def __init__(self) -> None:
        self.facts: dict[str, NegativeFact] = {}

    def record_negative_fact(
        self,
        fact_key: str,
        target: str,
        rationale: str,
        confidence: float = 1.0,
        turn: int = 1,
    ) -> NegativeFact:
        """Store a negative fact (e.g. NO_CALLER_WRITES_TIMEOUT, target='billing/service.py')."""
        key = f"{fact_key}:{target}".lower()
        fact = NegativeFact(
            fact_key=fact_key,
            target=target,
            rationale=rationale,
            confidence=max(0.0, min(1.0, confidence)),
            recorded_at_turn=turn,
        )
        self.facts[key] = fact
        return fact

    def is_ruled_out(self, fact_key: str, target: str) -> bool:
        """Check if a particular hypothesis or action target is already ruled out."""
        key = f"{fact_key}:{target}".lower()
        return key in self.facts

    def get_negative_facts(self) -> list[NegativeFact]:
        return list(self.facts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts_count": len(self.facts),
            "facts": [f.to_dict() for f in self.facts.values()],
        }
