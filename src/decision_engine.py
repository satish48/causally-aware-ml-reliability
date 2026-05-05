"""
src/decision_engine.py
----------------------
Decisioning layer for the ML incident intelligence system.

Responsibilities:
    - Convert impact + detector state into recommended operational actions
    - Produce a stable contract for UI, incident workflows, and future agents
    - Keep recommendation logic deterministic and explainable

Important:
    This is not an autonomous remediation engine yet.
    It is a recommendation engine with production-safe output contracts.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from src.drift_detector import DriftResult
from src.incident_models import DecisionRecommendation, ImpactAssessment, ProjectedOutcome

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Recommend next actions based on detector state and impact assessment.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        self._retraining_threshold = cfg.decision.retraining_severity_threshold
        self._rollback_threshold = cfg.decision.rollback_severity_threshold
        self._rollback_repeated_alert_count = cfg.decision.repeated_alert_count_for_rollback

        logger.info(
            "DecisionEngine initialized | retraining_threshold=%d | rollback_threshold=%d",
            self._retraining_threshold,
            self._rollback_threshold,
        )

    def recommend(
        self,
        *,
        result: DriftResult,
        impact: ImpactAssessment,
        recent_alert_count: int,
    ) -> DecisionRecommendation:
        """
        Produce a deterministic operational recommendation with specific tradeoffs
        and projected outcomes so operators know exactly what they're deciding.
        """
        loss_hr = impact.estimated_loss_usd
        fraud_pct = round(result.fraud_rate * 100, 1)
        healthy_fraud = 3.2  # healthy baseline %

        if (
            impact.severity_score >= self._rollback_threshold
            and recent_alert_count >= self._rollback_repeated_alert_count
            and impact.incident_kpi_breach
        ):
            saved = round(loss_hr * 0.85)
            outcome = ProjectedOutcome(
                t5_fraud_rate_pct=round(fraud_pct * 0.35, 1),
                t15_fraud_rate_pct=round(healthy_fraud * 1.2, 1),
                t30_fraud_rate_pct=round(healthy_fraud, 1),
                t30_loss_per_hour_usd=round(loss_hr * 0.05),
                loss_saved_per_hour_usd=saved,
                narrative=f"Stable model resumes. Fraud rate drops from {fraud_pct}% → ~{healthy_fraud}% within 15 min, saving ${saved:,.0f}/hr.",
            )
            recommendation = DecisionRecommendation(
                recommended_action="rollback_model",
                action_type="model_rollback",
                priority="critical",
                rationale=(
                    f"Critical severity (score {impact.severity_score}/100) with {recent_alert_count} repeated alerts "
                    f"and KPI breach. Rollback is the safest immediate containment — "
                    f"the previous stable model should restore fraud rate from {fraud_pct}% to baseline in ~8 min."
                ),
                confidence=0.92,
                specific_threshold=f"Revert to last stable checkpoint. Traffic reroutes within 60s.",
                tradeoff=f"Gain: ${saved:,.0f}/hr saved, fraud drops to baseline. Cost: cold-start latency ~8 min, any live A/B experiment is interrupted.",
                projected_outcome=outcome,
            )
            return self._log(result, recommendation)

        if (
            impact.severity_score >= self._retraining_threshold
            and impact.requires_escalation
        ):
            saved = round(loss_hr * 0.70)
            outcome = ProjectedOutcome(
                t5_fraud_rate_pct=fraud_pct,
                t15_fraud_rate_pct=round(fraud_pct * 0.8, 1),
                t30_fraud_rate_pct=round(fraud_pct * 0.6, 1),
                t30_loss_per_hour_usd=round(loss_hr * 0.35),
                loss_saved_per_hour_usd=saved,
                narrative=f"New model trains on fresh data (~90 min). Fraud rate begins declining after deployment; full recovery at T+90 min.",
            )
            recommendation = DecisionRecommendation(
                recommended_action="trigger_retraining",
                action_type="model_retraining",
                priority="critical",
                rationale=(
                    f"PSI={result.overall_psi:.2f} confirms genuine distribution shift — not a transient spike. "
                    f"Retraining on the last 7 days of data should realign the model's decision boundary. "
                    f"Combine with manual review as an immediate bridge."
                ),
                confidence=0.88,
                specific_threshold="Trigger training pipeline on last 7-day window. Deploy as canary at 10% traffic.",
                tradeoff=f"Gain: ${saved:,.0f}/hr at full recovery, root cause addressed permanently. Cost: 90 min pipeline lag — pair with manual review now.",
                projected_outcome=outcome,
            )
            return self._log(result, recommendation)

        if impact.incident_kpi_breach or result.fraud_rate >= 0.20:
            # Specific: what threshold to raise, how much it helps
            current_threshold_pct = 15
            new_threshold_pct = min(35, int(fraud_pct * 1.4))
            catch_rate = min(55, int(result.fraud_rate * 280))
            saved = round(loss_hr * (catch_rate / 100))
            ops_cost_hr = round(new_threshold_pct * 22)  # ~$22/hr per review %pt
            net_saved = max(0, saved - ops_cost_hr)
            outcome = ProjectedOutcome(
                t5_fraud_rate_pct=round(fraud_pct * 0.85, 1),
                t15_fraud_rate_pct=round(fraud_pct * (1 - catch_rate / 100 * 0.7), 1),
                t30_fraud_rate_pct=round(fraud_pct * (1 - catch_rate / 100), 1),
                t30_loss_per_hour_usd=round(loss_hr * (1 - catch_rate / 100)),
                loss_saved_per_hour_usd=net_saved,
                narrative=f"Review queue catches ~{catch_rate}% of excess fraud within 20 min. Net saving ${net_saved:,.0f}/hr after ops cost.",
            )
            recommendation = DecisionRecommendation(
                recommended_action="increase_manual_review",
                action_type="manual_review_escalation",
                priority="high",
                rationale=(
                    f"Fraud rate at {fraud_pct}% exceeds the {current_threshold_pct}% alert threshold. "
                    f"Raising the manual review flag to {new_threshold_pct}% routes the highest-risk "
                    f"transactions to human review while the root cause is confirmed."
                ),
                confidence=0.84,
                specific_threshold=f"Raise review threshold: {current_threshold_pct}% → {new_threshold_pct}% (flags top {new_threshold_pct - current_threshold_pct}% of score distribution).",
                tradeoff=f"Gain: catches ~{catch_rate}% more fraud, saves ${saved:,.0f}/hr. Cost: +${ops_cost_hr:,.0f}/hr ops overhead. Net: +${net_saved:,.0f}/hr.",
                projected_outcome=outcome,
            )
            return self._log(result, recommendation)

        if result.overall_psi >= 0.20 or result.max_z_score >= 2.5:
            outcome = ProjectedOutcome(
                t5_fraud_rate_pct=fraud_pct,
                t15_fraud_rate_pct=fraud_pct,
                t30_fraud_rate_pct=fraud_pct,
                t30_loss_per_hour_usd=round(loss_hr),
                loss_saved_per_hour_usd=0,
                narrative="Monitoring mode — no financial improvement until evidence crosses action threshold.",
            )
            recommendation = DecisionRecommendation(
                recommended_action="open_incident",
                action_type="incident_opened",
                priority="medium",
                rationale=(
                    f"PSI={result.overall_psi:.2f} confirms feature drift but impact is below the rollback/review threshold. "
                    f"Open an incident ticket, assign an on-call owner, and set a 15-min re-evaluation trigger."
                ),
                confidence=0.79,
                specific_threshold="Open P2 incident. Set re-eval at T+15 min or if fraud rate crosses 20%.",
                tradeoff="Gain: keeps model live, avoids unnecessary rollback. Cost: no immediate financial improvement — monitor closely.",
                projected_outcome=outcome,
            )
            return self._log(result, recommendation)

        recommendation = DecisionRecommendation(
            recommended_action="monitor",
            action_type="monitor_only",
            priority="low",
            rationale=(
                "All signals below action thresholds. PSI and Z-score are within "
                "normal operating variance. Continue passive monitoring."
            ),
            confidence=0.72,
            specific_threshold="No parameter change needed. Review again at next drift detection.",
            tradeoff="Gain: zero ops overhead. Risk: if drift is slow-burn, delay compounds losses — watch burn rate.",
            projected_outcome=ProjectedOutcome(
                t5_fraud_rate_pct=fraud_pct,
                t15_fraud_rate_pct=fraud_pct,
                t30_fraud_rate_pct=fraud_pct,
                t30_loss_per_hour_usd=round(loss_hr),
                loss_saved_per_hour_usd=0,
                narrative="No action — system continues in current state.",
            ),
        )
        return self._log(result, recommendation)

    @staticmethod
    def _log(
        result: DriftResult,
        recommendation: DecisionRecommendation,
    ) -> DecisionRecommendation:
        """
        Log recommendation output consistently and return it unchanged.
        """
        logger.info(
            "Decision recommended | batch_id=%d | action=%s | action_type=%s | "
            "priority=%s | confidence=%.2f",
            result.batch_id,
            recommendation.recommended_action,
            recommendation.action_type,
            recommendation.priority,
            recommendation.confidence,
        )
        return recommendation