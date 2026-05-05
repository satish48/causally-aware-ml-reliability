"""
src/canary_controller.py
------------------------
Adaptive canary deployment controller with SLO-gated traffic shifting.

Industry context:
    Google's SRE book and internal deploy systems (Borg, GKE deploy pipelines)
    use canary deployments with automatic promotion or rollback based on golden
    signals (latency, error rate, saturation). This controller applies the same
    pattern to ML model deployments, using SLO burn rate as the gating signal.

Traffic progression (one stage at a time):
    staged (0%) → CANARY (10%) → EXPANDING (30%) → MAJORITY (50%) → STABLE (100%)

Promotion criteria (ALL must hold for N consecutive evaluations):
    • SLO worst burn rate ≤ 1.0  (not consuming error budget faster than allowed)
    • No fast-burn alerts in current evaluation window
    • Canary health score ≥ HEALTH_PROMOTE_THRESHOLD (80)

Rollback criteria (ANY one triggers automatic rollback):
    • Fast-burn alert emitted this batch (burn_rate > 14.4×)
    • Canary health score < HEALTH_ROLLBACK_THRESHOLD (30) after min observations
    • Drift severity == "critical" while canary is active

Rollback is auto-executed (safety default).
Promotion is advisory — emitted as a decision the API serves; the controller
also executes it immediately since it only shifts a weight.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from src.drift_detector import DriftResult
from src.model_registry import ModelRegistry, ModelVersion
from src.slo_engine import BurnRateAlert, SLOEngine

logger = logging.getLogger(__name__)

__all__ = [
    "CanaryStage",
    "CanaryDecisionType",
    "CanaryEvaluation",
    "CanaryController",
    "get_canary_controller",
]


# ────────────────────────────────────────────────────────────────
# DOMAIN TYPES
# ────────────────────────────────────────────────────────────────

class CanaryStage(Enum):
    CANARY = 0.10
    EXPANDING = 0.30
    MAJORITY = 0.50
    STABLE = 1.00

    @classmethod
    def next_stage(cls, current: "CanaryStage") -> "CanaryStage | None":
        stages = list(cls)
        idx = stages.index(current)
        return stages[idx + 1] if idx + 1 < len(stages) else None

    @classmethod
    def from_weight(cls, weight: float) -> "CanaryStage":
        for stage in cls:
            if abs(stage.value - weight) < 0.05:
                return stage
        return cls.CANARY


CanaryDecisionType = Literal["promote", "hold", "rollback", "no_canary"]


@dataclass(slots=True)
class CanaryEvaluation:
    evaluation_id: str
    batch_id: int
    evaluated_at: str
    model_name: str
    canary_version_id: str | None
    decision: CanaryDecisionType
    current_stage: str | None       # e.g. "CANARY", "EXPANDING"
    next_stage: str | None
    current_weight: float
    next_weight: float
    worst_burn_rate: float
    health_score: float
    consecutive_healthy: int
    rationale: str
    auto_executed: bool              # rollbacks and promotions are auto-executed


# ────────────────────────────────────────────────────────────────
# CONTROLLER
# ────────────────────────────────────────────────────────────────

class CanaryController:
    """
    Evaluate canary health every batch and emit deployment decisions.

    Requires a ModelRegistry (truth about versions) and SLOEngine (burn rates).
    Call evaluate() after every batch once the SLO engine has been updated.
    """

    CONSECUTIVE_HEALTHY_FOR_PROMOTE: int = 5
    MIN_OBSERVATIONS_BEFORE_ROLLBACK: int = 3
    FAST_BURN_ROLLBACK_THRESHOLD: float = 14.4
    HEALTH_PROMOTE_THRESHOLD: float = 80.0
    HEALTH_ROLLBACK_THRESHOLD: float = 30.0

    MAX_EVALUATIONS: int = 500

    def __init__(
        self,
        model_registry: ModelRegistry,
        slo_engine: SLOEngine,
    ) -> None:
        self._registry = model_registry
        self._slo = slo_engine
        self._consecutive_healthy: dict[str, int] = {}
        self._observations: dict[str, int] = {}
        self._evaluations: list[CanaryEvaluation] = []
        logger.info("CanaryController initialized")

    # ── PUBLIC API ────────────────────────────────────────────

    def evaluate(
        self,
        *,
        model_name: str,
        batch_id: int,
        drift_result: DriftResult,
        burn_rate_alerts: list[BurnRateAlert],
    ) -> CanaryEvaluation:
        """
        Evaluate the active canary (if any) and return a deployment decision.

        Side effects:
            - On ROLLBACK: transitions canary version to "rolled_back" and
              restores the previous stable version's weight to 1.0.
            - On PROMOTE: advances canary_weight to next stage, or finalises
              it as "stable" when reaching 100%.
        """
        canary = self._registry.get_canary_version(model_name)

        if canary is None:
            # No canary is active — evaluate the current production model health
            # and emit a deployment posture recommendation instead of a dead stub.
            budgets = self._slo.get_budgets_snapshot()
            worst_burn = max((v["burn_rate"] for v in budgets.values()), default=0.0)
            min_budget = min((v["budget_remaining_pct"] for v in budgets.values()), default=100.0)
            stable = self._registry.get_stable_version(model_name)
            health = stable.health_score if stable else 100.0

            if drift_result.severity == "critical" or min_budget <= 0.0 or worst_burn >= 14.4:
                posture: CanaryDecisionType = "rollback"
                rationale = (
                    f"Production model critically degraded — drift severity={drift_result.severity}, "
                    f"budget_remaining={min_budget:.0f}%, burn={worst_burn:.1f}×. "
                    "Recommend immediate rollback to last known-good checkpoint or emergency retrain."
                )
            elif drift_result.drift_detected or worst_burn > 1.0:
                posture = "hold"
                rationale = (
                    f"Drift detected (PSI={drift_result.overall_psi:.3f}, Z={drift_result.max_z_score:.2f}). "
                    f"SLO burn={worst_burn:.1f}×. Hold all new deployments; "
                    "investigate root cause before promoting any candidate."
                )
            else:
                posture = "promote"
                rationale = (
                    f"Model health={health:.0f}/100, no drift, burn={worst_burn:.1f}×. "
                    "Production posture is stable — safe window for new deployments."
                )

            evaluation = self._make_evaluation(
                batch_id=batch_id,
                model_name=model_name,
                canary=None,
                decision=posture,
                current_stage="production",
                next_stage=None,
                current_weight=1.0,
                next_weight=1.0,
                worst_burn_rate=round(worst_burn, 3),
                health_score=round(health, 1),
                consecutive_healthy=0,
                rationale=rationale,
                auto_executed=False,
            )
            self._store(evaluation)
            return evaluation

        vid = canary.version_id
        self._observations[vid] = self._observations.get(vid, 0) + 1

        # ── Derive signals ────────────────────────────────────
        budgets = self._slo.get_budgets_snapshot()
        worst_burn = max((v["burn_rate"] for v in budgets.values()), default=0.0)
        has_fast_burn = any(a.alert_type == "fast_burn" for a in burn_rate_alerts)
        has_budget_exhausted = any(a.alert_type == "budget_exhausted" for a in burn_rate_alerts)
        critical_drift = drift_result.drift_detected and drift_result.severity == "critical"
        health_collapse = (
            canary.health_score < self.HEALTH_ROLLBACK_THRESHOLD
            and self._observations[vid] >= self.MIN_OBSERVATIONS_BEFORE_ROLLBACK
        )

        current_stage = CanaryStage.from_weight(canary.canary_weight)
        next_stage_enum = CanaryStage.next_stage(current_stage)

        # ── ROLLBACK gate ─────────────────────────────────────
        if has_fast_burn or has_budget_exhausted or health_collapse:
            self._consecutive_healthy[vid] = 0
            self._execute_rollback(model_name, canary)

            reasons: list[str] = []
            if has_fast_burn:
                reasons.append(f"fast-burn SLO (burn_rate={worst_burn:.1f}×)")
            if has_budget_exhausted:
                reasons.append("SLO error budget exhausted")
            if health_collapse:
                reasons.append(f"health score collapse ({canary.health_score:.0f}/100)")

            evaluation = self._make_evaluation(
                batch_id=batch_id,
                model_name=model_name,
                canary=canary,
                decision="rollback",
                current_stage=current_stage.name,
                next_stage=None,
                current_weight=canary.canary_weight,
                next_weight=0.0,
                worst_burn_rate=round(worst_burn, 3),
                health_score=round(canary.health_score, 1),
                consecutive_healthy=self._consecutive_healthy.get(vid, 0),
                rationale=(
                    f"AUTO-ROLLBACK: {' | '.join(reasons)}. "
                    "Traffic restored to previous stable version."
                ),
                auto_executed=True,
            )
            logger.critical(
                "CANARY ROLLBACK | model=%s | version=%s | reason=%s | burn=%.1f×",
                model_name, vid, " | ".join(reasons), worst_burn,
            )
            self._store(evaluation)
            return evaluation

        # ── PROMOTE gate ──────────────────────────────────────
        batch_healthy = (
            worst_burn <= self.FAST_BURN_ROLLBACK_THRESHOLD
            and not critical_drift
            and canary.health_score >= self.HEALTH_PROMOTE_THRESHOLD
        )

        if batch_healthy:
            self._consecutive_healthy[vid] = self._consecutive_healthy.get(vid, 0) + 1
        else:
            self._consecutive_healthy[vid] = 0

        consecutive = self._consecutive_healthy.get(vid, 0)
        can_promote = (
            consecutive >= self.CONSECUTIVE_HEALTHY_FOR_PROMOTE
            and next_stage_enum is not None
            and worst_burn <= 1.0
        )

        if can_promote and next_stage_enum is not None:
            new_weight = next_stage_enum.value
            self._execute_promote(model_name, canary, next_stage_enum)
            self._consecutive_healthy[vid] = 0  # reset counter after each promotion

            evaluation = self._make_evaluation(
                batch_id=batch_id,
                model_name=model_name,
                canary=canary,
                decision="promote",
                current_stage=current_stage.name,
                next_stage=next_stage_enum.name if new_weight < 1.0 else "STABLE",
                current_weight=current_stage.value,
                next_weight=new_weight,
                worst_burn_rate=round(worst_burn, 3),
                health_score=round(canary.health_score, 1),
                consecutive_healthy=consecutive,
                rationale=(
                    f"PROMOTE: {consecutive} consecutive healthy batches "
                    f"(burn_rate={worst_burn:.2f}× ≤ 1.0, health={canary.health_score:.0f}). "
                    f"Advancing {current_stage.name} → {next_stage_enum.name} "
                    f"({new_weight:.0%} traffic)."
                ),
                auto_executed=True,
            )
            logger.info(
                "CANARY PROMOTE | model=%s | version=%s | %s → %s | weight=%.0f%%",
                model_name, vid, current_stage.name, next_stage_enum.name, new_weight * 100,
            )
            self._store(evaluation)
            return evaluation

        # ── HOLD ─────────────────────────────────────────────
        needed = self.CONSECUTIVE_HEALTHY_FOR_PROMOTE - consecutive
        evaluation = self._make_evaluation(
            batch_id=batch_id,
            model_name=model_name,
            canary=canary,
            decision="hold",
            current_stage=current_stage.name,
            next_stage=next_stage_enum.name if next_stage_enum else None,
            current_weight=canary.canary_weight,
            next_weight=next_stage_enum.value if next_stage_enum else canary.canary_weight,
            worst_burn_rate=round(worst_burn, 3),
            health_score=round(canary.health_score, 1),
            consecutive_healthy=consecutive,
            rationale=(
                f"HOLD at {current_stage.name} ({canary.canary_weight:.0%} traffic). "
                f"Need {needed} more healthy batches to promote. "
                f"Burn rate: {worst_burn:.2f}×. Health: {canary.health_score:.0f}/100."
            ),
            auto_executed=False,
        )
        self._store(evaluation)
        return evaluation

    def deploy_canary(
        self,
        model_name: str,
        *,
        notes: str = "",
        baseline_metrics: dict[str, float] | None = None,
    ) -> ModelVersion:
        """
        Register a new canary version at 10% traffic.

        Returns the new ModelVersion so callers can reference version_id.
        """
        stable = self._registry.get_stable_version(model_name)
        if stable:
            stable.canary_weight = 0.90  # reserve 10% for canary

        canary = self._registry.register_version(
            model_name=model_name,
            status="canary",
            canary_weight=0.10,
            baseline_metrics=baseline_metrics or {},
            notes=notes or f"Canary deployed at {datetime.now(timezone.utc).isoformat()}",
        )
        logger.info(
            "Canary deployed | model=%s | version=%s | weight=10%%",
            model_name, canary.version_id,
        )
        return canary

    def get_evaluations_snapshot(self, limit: int = 30) -> list[dict[str, Any]]:
        return [
            {
                "evaluation_id": e.evaluation_id,
                "batch_id": e.batch_id,
                "evaluated_at": e.evaluated_at,
                "model_name": e.model_name,
                "canary_version_id": e.canary_version_id,
                "decision": e.decision,
                "current_stage": e.current_stage,
                "next_stage": e.next_stage,
                "current_weight": e.current_weight,
                "next_weight": e.next_weight,
                "worst_burn_rate": e.worst_burn_rate,
                "health_score": e.health_score,
                "consecutive_healthy": e.consecutive_healthy,
                "rationale": e.rationale,
                "auto_executed": e.auto_executed,
            }
            for e in self._evaluations[-limit:]
        ]

    def get_latest_evaluation(self) -> CanaryEvaluation | None:
        return self._evaluations[-1] if self._evaluations else None

    def get_stats(self) -> dict[str, Any]:
        evals = self._evaluations
        return {
            "total_evaluations": len(evals),
            "promotes": sum(1 for e in evals if e.decision == "promote"),
            "rollbacks": sum(1 for e in evals if e.decision == "rollback"),
            "holds": sum(1 for e in evals if e.decision == "hold"),
            "no_canary": sum(1 for e in evals if e.decision == "no_canary"),
        }

    # ── INTERNAL ──────────────────────────────────────────────

    def _execute_rollback(self, model_name: str, canary: ModelVersion) -> None:
        self._registry.transition_status(canary.version_id, "rolled_back")
        canary.canary_weight = 0.0
        stable = self._registry.get_stable_version(model_name)
        if stable and stable.version_id != canary.version_id:
            stable.canary_weight = 1.0

    def _execute_promote(
        self,
        model_name: str,
        canary: ModelVersion,
        next_stage: CanaryStage,
    ) -> None:
        new_weight = next_stage.value
        if new_weight >= 1.0:
            old_stable = self._registry.get_stable_version(model_name)
            if old_stable and old_stable.version_id != canary.version_id:
                old_stable.status = "deprecated"
                old_stable.canary_weight = 0.0
            canary.status = "stable"
            canary.canary_weight = 1.0
        else:
            canary.canary_weight = new_weight
            stable = self._registry.get_stable_version(model_name)
            if stable:
                stable.canary_weight = round(1.0 - new_weight, 4)

    @staticmethod
    def _make_evaluation(
        *,
        batch_id: int,
        model_name: str,
        canary: ModelVersion | None,
        decision: CanaryDecisionType,
        current_stage: str | None,
        next_stage: str | None,
        current_weight: float,
        next_weight: float,
        worst_burn_rate: float,
        health_score: float,
        consecutive_healthy: int,
        rationale: str,
        auto_executed: bool,
    ) -> CanaryEvaluation:
        return CanaryEvaluation(
            evaluation_id=f"ce-{uuid.uuid4().hex[:8]}",
            batch_id=batch_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            canary_version_id=canary.version_id if canary else None,
            decision=decision,
            current_stage=current_stage,
            next_stage=next_stage,
            current_weight=current_weight,
            next_weight=next_weight,
            worst_burn_rate=worst_burn_rate,
            health_score=health_score,
            consecutive_healthy=consecutive_healthy,
            rationale=rationale,
            auto_executed=auto_executed,
        )

    def _store(self, evaluation: CanaryEvaluation) -> None:
        self._evaluations.append(evaluation)
        if len(self._evaluations) > self.MAX_EVALUATIONS:
            self._evaluations = self._evaluations[-self.MAX_EVALUATIONS:]


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_controller: CanaryController | None = None


def get_canary_controller() -> CanaryController:
    global _controller
    if _controller is None:
        from src.model_registry import get_model_registry
        from src.slo_engine import get_slo_engine
        _controller = CanaryController(
            model_registry=get_model_registry(),
            slo_engine=get_slo_engine(),
        )
    return _controller
