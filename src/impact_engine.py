"""
src/impact_engine.py
--------------------
Business-impact estimation layer for the ML incident intelligence system.

Responsibilities:
    - Convert technical detector signals into business-facing impact
    - Produce a normalized severity score
    - Estimate rough financial exposure
    - Indicate whether escalation is warranted

Important:
    This module is deterministic by design.
    It is intentionally heuristic and explainable, not opaque.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import get_settings
from src.drift_detector import DriftResult
from src.incident_models import ImpactAssessment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ImpactWeights:
    """
    Relative weights used to compose the final severity score.
    """

    psi: float = 0.30
    z_score: float = 0.20
    fraud_rate: float = 0.25
    confidence_drop: float = 0.15
    recurrence: float = 0.10


class ImpactEngine:
    """
    Estimate business impact from drift detector output.

    The goal is to translate:
        technical degradation -> business consequence

    Inputs:
        - DriftResult from the detector
        - recent alert count as a recurrence signal

    Outputs:
        - ImpactAssessment with severity score, business label,
          estimated loss, and escalation flag
    """

    def __init__(self) -> None:
        cfg = get_settings()

        self._fraud_alert_threshold = cfg.monitoring.alert_fraud_rate
        self._base_incident_cost = cfg.impact.base_incident_cost_usd
        self._cost_per_excess_fraud_event = cfg.impact.cost_per_excess_fraud_event_usd
        self._cost_per_confidence_penalty = cfg.impact.cost_per_confidence_drop_pct_usd
        self._weights = ImpactWeights()

        logger.info(
            "ImpactEngine initialized | fraud_alert_threshold=%.2f | base_cost=%.2f",
            self._fraud_alert_threshold,
            self._base_incident_cost,
        )

    def assess(
        self,
        *,
        result: DriftResult,
        recent_alert_count: int,
    ) -> ImpactAssessment:
        """
        Build a business-facing impact assessment from one detector result.
        """
        fraud_rate_pct = result.fraud_rate * 100.0
        confidence_drop_pct = max(0.0, (0.90 - result.avg_confidence) * 100.0)

        psi_score = min(100.0, result.overall_psi * 100.0)
        z_score_score = min(100.0, (result.max_z_score / 4.0) * 100.0)

        excess_fraud = max(0.0, result.fraud_rate - self._fraud_alert_threshold)
        fraud_impact_score = min(
            100.0,
            (excess_fraud / max(0.01, 1.0 - self._fraud_alert_threshold)) * 100.0,
        )

        confidence_impact_score = min(100.0, confidence_drop_pct * 4.0)
        recurrence_score = min(100.0, recent_alert_count * 15.0)

        severity_score = int(round(
            psi_score * self._weights.psi
            + z_score_score * self._weights.z_score
            + fraud_impact_score * self._weights.fraud_rate
            + confidence_impact_score * self._weights.confidence_drop
            + recurrence_score * self._weights.recurrence
        ))

        affected_segment_count = self._estimate_affected_segments(result)
        # A KPI breach is not just a fraud-rate crossing — drift at PSI≥0.5 or
        # Z-score≥3.0 represent material model quality degradation regardless of
        # whether the raw fraud rate exceeds the SLO target.
        incident_kpi_breach = (
            result.fraud_rate > self._fraud_alert_threshold
            or result.overall_psi >= 0.50
            or result.max_z_score >= 3.0
        )

        estimated_loss_usd = self._estimate_loss_usd(
            result=result,
            excess_fraud=excess_fraud,
            confidence_drop_pct=confidence_drop_pct,
            severity_score=severity_score,
            affected_segment_count=affected_segment_count,
        )

        requires_escalation = (
            severity_score >= 70
            or incident_kpi_breach
            or result.overall_psi >= 0.50
        )
        business_impact_label = self._label_from_score(severity_score)
        # Invariant: if escalation is required, the label must be at least "material".
        # Anything below "material" paired with requires_escalation=True contradicts
        # the escalation signal and confuses incident responders.
        _LABEL_RANK = {"minimal": 0, "moderate": 1, "material": 2, "severe": 3}
        if requires_escalation and _LABEL_RANK.get(business_impact_label, 2) < 2:
            business_impact_label = "material"

        assessment = ImpactAssessment(
            estimated_loss_usd=round(estimated_loss_usd, 2),
            severity_score=severity_score,
            affected_segment_count=affected_segment_count,
            business_impact_label=business_impact_label,
            requires_escalation=requires_escalation,
            confidence_drop_pct=round(confidence_drop_pct, 2),
            fraud_rate_pct=round(fraud_rate_pct, 2),
            incident_kpi_breach=incident_kpi_breach,
        )

        logger.info(
            "Impact assessed | batch_id=%d | severity_score=%d | impact=%s | "
            "loss=$%.2f | escalation=%s",
            result.batch_id,
            assessment.severity_score,
            assessment.business_impact_label,
            assessment.estimated_loss_usd,
            assessment.requires_escalation,
        )

        return assessment

    def _estimate_affected_segments(self, result: DriftResult) -> int:
        """
        Estimate the number of impacted segments from feature-level degradation.
        """
        materially_shifted_features = sum(
            1
            for score in result.feature_scores
            if score.drifted or score.psi >= 0.20 or score.z_score >= 2.0
        )

        if materially_shifted_features >= 5:
            return 4
        if materially_shifted_features >= 3:
            return 3
        if materially_shifted_features >= 1:
            return 2
        return 1

    def _estimate_loss_usd(
        self,
        *,
        result: DriftResult,
        excess_fraud: float,
        confidence_drop_pct: float,
        severity_score: int,
        affected_segment_count: int,
    ) -> float:
        """
        Estimate rough financial exposure.

        This is intentionally heuristic:
            - excess fraud contributes directly
            - confidence degradation contributes indirectly
            - severity adds a general incident burden
            - affected segments amplify impact
        """
        excess_fraud_events = excess_fraud * 1000.0
        fraud_loss = excess_fraud_events * self._cost_per_excess_fraud_event
        confidence_penalty = confidence_drop_pct * self._cost_per_confidence_penalty
        severity_penalty = severity_score * 8.0
        segment_multiplier = 1.0 + (affected_segment_count - 1) * 0.20

        total = (
            self._base_incident_cost
            + fraud_loss
            + confidence_penalty
            + severity_penalty
        ) * segment_multiplier

        if not result.drift_detected and result.overall_psi < 0.10:
            total *= 0.10

        return max(0.0, total)

    @staticmethod
    def _label_from_score(score: int) -> str:
        """
        Map normalized severity score to a business impact label.
        """
        if score >= 85:
            return "severe"
        if score >= 65:
            return "material"
        if score >= 35:
            return "moderate"
        return "minimal"