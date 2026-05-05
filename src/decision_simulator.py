"""
src/decision_simulator.py
-------------------------
Decision simulation engine for ML incident response.

Industry context:
    At Stripe, before an on-call engineer takes action on a degrading model,
    they run through a mental model of "what happens if I do X?". This engine
    makes that simulation explicit and quantified.

    The four canonical responses to ML model degradation:
        1. rollback         — immediate but cold-start risk, interrupts experiments
        2. manual_review    — safe but costly, ops-team bottleneck at scale
        3. trigger_retraining — correct long-term but has a recovery lag
        4. ignore           — valid if false-positive, catastrophic if real drift

    Each action produces an ActionSimulation with estimated recovery time,
    loss reduction, risk factors, and a confidence score. When signals are
    ambiguous (low causal confidence, borderline drift), the engine explicitly
    returns lower confidence and flags the ambiguity rather than forcing a
    high-confidence recommendation.

Design:
    - Heuristic model (not ML on ML — that adds latency and training complexity)
    - Inputs: current risk forecast, SLO state, causal attribution confidence,
      canary stage, drift severity
    - Outputs: ranked list of ActionSimulation objects with recommended=True
      for the top action
    - Uncertainty: when causal_confidence < 0.4, all simulations have capped
      confidence to force human review
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ActionType",
    "ActionSimulation",
    "SimulationResult",
    "DecisionSimulator",
    "get_decision_simulator",
]

ActionType = str  # "rollback" | "manual_review" | "trigger_retraining" | "ignore"

_ACTION_ORDER: list[ActionType] = [
    "rollback",
    "manual_review",
    "trigger_retraining",
    "ignore",
]


@dataclass(slots=True)
class ActionSimulation:
    """
    Estimated outcome of taking a specific incident response action.

    recovery_time_minutes: 0 means instant, None means unknown.
    estimated_loss_reduction_pct: 0-100, how much of current loss/hr is mitigated.
    confidence: 0-1, how reliable this simulation is given available signals.
    recommended: True for the single highest-value action.
    """
    action: ActionType
    display_name: str
    recovery_time_minutes: float | None
    estimated_loss_reduction_pct: float
    risk_level: str                     # "low" | "moderate" | "high" | "critical"
    risk_factors: list[str]
    upside_factors: list[str]
    confidence: float
    recommended: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "display_name": self.display_name,
            "recovery_time_minutes": self.recovery_time_minutes,
            "estimated_loss_reduction_pct": round(self.estimated_loss_reduction_pct, 1),
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "upside_factors": self.upside_factors,
            "confidence": round(self.confidence, 3),
            "recommended": self.recommended,
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class SimulationResult:
    """
    Output of DecisionSimulator.simulate_all().

    simulations: all four actions, sorted by estimated_loss_reduction_pct desc.
    recommended_action: action string for the top-ranked simulation.
    ambiguous: True when signals are insufficient to make a confident recommendation.
    causal_confidence: the attribution confidence that drove the simulation.
    generated_at: ISO timestamp.
    """
    simulations: list[ActionSimulation]
    recommended_action: ActionType | None
    ambiguous: bool
    causal_confidence: float
    risk_score: float
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulations": [s.to_dict() for s in self.simulations],
            "recommended_action": self.recommended_action,
            "ambiguous": self.ambiguous,
            "causal_confidence": round(self.causal_confidence, 3),
            "risk_score": round(self.risk_score, 1),
            "generated_at": self.generated_at,
        }


# ────────────────────────────────────────────────────────────────
# SIMULATOR
# ────────────────────────────────────────────────────────────────

class DecisionSimulator:
    """
    Simulate the outcomes of all four canonical incident response actions.

    Call simulate_all() with current observability signals to get a ranked
    simulation for each action. The simulator is stateless — safe to call
    concurrently and repeatedly without side effects.

    Key design decision: when causal_confidence is low or drift signals are
    borderline, the simulator caps all confidence values at 0.55 and sets
    ambiguous=True. This forces a human into the loop rather than auto-acting
    on weak signals.
    """

    AMBIGUITY_THRESHOLD: float = 0.40    # below this causal_confidence → ambiguous
    CAP_CONFIDENCE_WHEN_AMBIGUOUS: float = 0.55

    def simulate_all(
        self,
        risk_score: float,                  # 0-100 from RiskForecaster
        causal_confidence: float,           # 0-1 from CausalEngine
        budget_remaining_pct: float,        # 0-100 from SLOEngine
        worst_burn_rate: float,             # from SLOEngine
        drift_detected: bool,
        drift_severity: str,               # "stable" | "moderate" | "severe"
        canary_stage: str | None,          # None if no canary
        fraud_rate: float,
        avg_confidence: float,
    ) -> SimulationResult:
        """
        Generate outcome simulations for all four response actions.

        Returns:
            SimulationResult with all simulations ranked by loss reduction.
        """
        ambiguous = causal_confidence < self.AMBIGUITY_THRESHOLD or not drift_detected
        is_critical = risk_score >= 75 or worst_burn_rate >= 14.4

        simulations: list[ActionSimulation] = [
            self._simulate_rollback(
                risk_score=risk_score,
                causal_confidence=causal_confidence,
                budget_remaining_pct=budget_remaining_pct,
                worst_burn_rate=worst_burn_rate,
                canary_stage=canary_stage,
                ambiguous=ambiguous,
            ),
            self._simulate_manual_review(
                risk_score=risk_score,
                causal_confidence=causal_confidence,
                fraud_rate=fraud_rate,
                is_critical=is_critical,
                ambiguous=ambiguous,
            ),
            self._simulate_retraining(
                risk_score=risk_score,
                causal_confidence=causal_confidence,
                drift_severity=drift_severity,
                ambiguous=ambiguous,
            ),
            self._simulate_ignore(
                risk_score=risk_score,
                causal_confidence=causal_confidence,
                drift_detected=drift_detected,
                worst_burn_rate=worst_burn_rate,
                ambiguous=ambiguous,
            ),
        ]

        # Cap confidence when ambiguous
        if ambiguous:
            for s in simulations:
                s.confidence = min(s.confidence, self.CAP_CONFIDENCE_WHEN_AMBIGUOUS)

        simulations.sort(key=lambda s: s.estimated_loss_reduction_pct, reverse=True)

        # Mark recommended: highest loss reduction that meets confidence threshold
        recommended_action: ActionType | None = None
        for sim in simulations:
            if sim.confidence >= 0.50:
                sim.recommended = True
                recommended_action = sim.action
                break

        logger.debug(
            "Decision simulation | risk_score=%.1f | causal_conf=%.2f | "
            "ambiguous=%s | recommended=%s",
            risk_score, causal_confidence, ambiguous, recommended_action,
        )

        return SimulationResult(
            simulations=simulations,
            recommended_action=recommended_action,
            ambiguous=ambiguous,
            causal_confidence=causal_confidence,
            risk_score=risk_score,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── ACTION SIMULATIONS ────────────────────────────────────

    def _simulate_rollback(
        self,
        *,
        risk_score: float,
        causal_confidence: float,
        budget_remaining_pct: float,
        worst_burn_rate: float,
        canary_stage: str | None,
        ambiguous: bool,
    ) -> ActionSimulation:
        has_canary = canary_stage is not None and canary_stage not in ("stable", None)

        # Recovery time: canary rollback is instant; stable rollback needs traffic draining
        recovery_minutes = 1.0 if has_canary else 8.0

        # Loss reduction: rollback fully mitigates if we're in canary; partial if on stable
        loss_reduction = 90.0 if has_canary else 75.0

        risk_factors: list[str] = []
        upside_factors: list[str] = []

        if has_canary:
            upside_factors.append("Canary at partial traffic — rollback impact is limited.")
            upside_factors.append(f"Current stage: {canary_stage} — clean rollback path exists.")
        else:
            risk_factors.append("No active canary — rolling back stable model causes cold-start.")
            risk_factors.append("Previous stable version may not be current in model registry.")

        if worst_burn_rate >= 14.4:
            upside_factors.append(
                f"Fast burn at {worst_burn_rate:.1f}× — rollback is urgently warranted."
            )
        if budget_remaining_pct < 20:
            upside_factors.append(f"Budget at {budget_remaining_pct:.0f}% — immediate action reduces blast radius.")

        if causal_confidence < 0.4:
            risk_factors.append(
                "Causal confidence is LOW — root cause unconfirmed. "
                "Rolling back may not resolve the underlying issue."
            )

        confidence = min(1.0, 0.50 + (risk_score / 100.0) * 0.35 + causal_confidence * 0.15)

        reasoning = (
            f"Rollback {'canary' if has_canary else 'stable model'}. "
            f"Estimated recovery: {recovery_minutes:.0f} min. "
            f"Loss reduction: {loss_reduction:.0f}%. "
            + ("Recommended when burn rate exceeds fast-burn threshold." if worst_burn_rate >= 14.4
               else "Consider if SLO budget continues to deplete.")
        )

        return ActionSimulation(
            action="rollback",
            display_name="Rollback Model" + (" (Canary)" if has_canary else ""),
            recovery_time_minutes=recovery_minutes,
            estimated_loss_reduction_pct=loss_reduction,
            risk_level="low" if has_canary else "moderate",
            risk_factors=risk_factors,
            upside_factors=upside_factors,
            confidence=round(confidence, 3),
            recommended=False,
            reasoning=reasoning,
        )

    def _simulate_manual_review(
        self,
        *,
        risk_score: float,
        causal_confidence: float,
        fraud_rate: float,
        is_critical: bool,
        ambiguous: bool,
    ) -> ActionSimulation:
        # Manual review adds latency but keeps model live
        recovery_minutes = 20.0

        # Catches ~50% of bad transactions at current fraud rate
        loss_reduction = min(55.0, fraud_rate * 300.0)

        risk_factors: list[str] = []
        upside_factors: list[str] = []

        upside_factors.append("Non-destructive — model continues serving while reviewing.")
        upside_factors.append("Gives engineers time to validate the root cause before rollback.")

        if is_critical:
            risk_factors.append("At critical risk levels, manual review alone is insufficient.")
            risk_factors.append("Ops team throughput may be overwhelmed at high fraud rates.")
        if fraud_rate > 0.25:
            risk_factors.append(
                f"Fraud rate at {fraud_rate:.1%} — manual review queue will back up."
            )

        confidence = 0.60 if ambiguous else 0.55 + (1.0 - risk_score / 100.0) * 0.2

        reasoning = (
            f"Flag transactions for manual review. Catches ~{loss_reduction:.0f}% of fraud. "
            "Best used when root cause is unclear and rollback risk is high."
        )

        return ActionSimulation(
            action="manual_review",
            display_name="Increase Manual Review Threshold",
            recovery_time_minutes=recovery_minutes,
            estimated_loss_reduction_pct=round(loss_reduction, 1),
            risk_level="moderate",
            risk_factors=risk_factors,
            upside_factors=upside_factors,
            confidence=round(confidence, 3),
            recommended=False,
            reasoning=reasoning,
        )

    def _simulate_retraining(
        self,
        *,
        risk_score: float,
        causal_confidence: float,
        drift_severity: str,
        ambiguous: bool,
    ) -> ActionSimulation:
        # Retraining has high recovery latency (infra pipeline) but is the right fix for real drift
        recovery_minutes = 90.0

        # Retraining on recent data fully resolves distribution drift
        loss_reduction = 85.0 if drift_severity == "severe" else 65.0

        risk_factors: list[str] = []
        upside_factors: list[str] = []

        upside_factors.append("Addresses root cause if drift is due to distribution shift.")
        upside_factors.append("Produces a new model version — can be deployed as canary.")

        risk_factors.append(f"Training pipeline takes ~{recovery_minutes:.0f} min to produce a new model.")
        risk_factors.append("New model may introduce regressions if training data is also drifted.")

        if causal_confidence >= 0.6 and drift_severity in ("moderate", "severe"):
            upside_factors.append(
                f"High causal confidence ({causal_confidence:.2f}) supports training-data root cause."
            )
        else:
            risk_factors.append(
                "Low causal confidence — training on current data may encode the corrupted distribution."
            )

        confidence = min(0.85, 0.40 + causal_confidence * 0.30 + (risk_score / 100.0) * 0.15)

        reasoning = (
            f"Retrain on fresh data. Expected recovery: ~{recovery_minutes:.0f} min. "
            "Correct long-term fix for genuine distribution drift. "
            "Not recommended as immediate mitigation."
        )

        return ActionSimulation(
            action="trigger_retraining",
            display_name="Trigger Model Retraining",
            recovery_time_minutes=recovery_minutes,
            estimated_loss_reduction_pct=loss_reduction,
            risk_level="low",
            risk_factors=risk_factors,
            upside_factors=upside_factors,
            confidence=round(confidence, 3),
            recommended=False,
            reasoning=reasoning,
        )

    def _simulate_ignore(
        self,
        *,
        risk_score: float,
        causal_confidence: float,
        drift_detected: bool,
        worst_burn_rate: float,
        ambiguous: bool,
    ) -> ActionSimulation:
        risk_factors: list[str] = []
        upside_factors: list[str] = []

        # Ignoring is only valid when causal confidence is very low and drift borderline
        is_safe_to_ignore = causal_confidence < 0.25 and worst_burn_rate < 1.0 and not drift_detected

        loss_reduction = 0.0

        if is_safe_to_ignore:
            upside_factors.append("No confirmed drift — false-positive alert is plausible.")
            upside_factors.append("Avoids unnecessary rollback churn on a stable model.")
            confidence = 0.65
        else:
            risk_factors.append(
                f"Burn rate at {worst_burn_rate:.1f}× — ignoring will exhaust error budget."
            )
            if drift_detected:
                risk_factors.append("Drift is confirmed — ignoring will compound losses.")
            confidence = max(0.10, 0.30 - (risk_score / 100.0) * 0.25)

        reasoning = (
            "Take no action and continue monitoring. "
            + ("Acceptable when drift is borderline and causal confidence is low."
               if is_safe_to_ignore
               else "HIGH RISK: continuing to ignore at current burn rate will exhaust SLO budget.")
        )

        return ActionSimulation(
            action="ignore",
            display_name="Continue Monitoring (No Action)",
            recovery_time_minutes=None,
            estimated_loss_reduction_pct=loss_reduction,
            risk_level="low" if is_safe_to_ignore else "critical",
            risk_factors=risk_factors,
            upside_factors=upside_factors,
            confidence=round(confidence, 3),
            recommended=False,
            reasoning=reasoning,
        )


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_simulator: DecisionSimulator | None = None


def get_decision_simulator() -> DecisionSimulator:
    global _simulator
    if _simulator is None:
        _simulator = DecisionSimulator()
    return _simulator
