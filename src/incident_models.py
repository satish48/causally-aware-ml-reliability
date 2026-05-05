"""
src/incident_models.py
----------------------
Typed domain models for the ML incident intelligence layer.

These models sit above raw drift detection and below the dashboard/API layer.

Responsibilities:
    - Define stable contracts for explanation, impact, decisioning, and incident summary
    - Keep the business/incident layer decoupled from raw detector internals
    - Provide a clean operator-facing incident summary builder

Design principles:
    - Explicit enums via Literal for stable downstream contracts
    - Pydantic validation for API- and UI-safe payloads
    - Small, focused models with descriptive field names
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


SeverityLabel = Literal["low", "medium", "high", "critical"]
ImpactLabel = Literal["minimal", "moderate", "material", "severe"]
PriorityLabel = Literal["low", "medium", "high", "critical"]

RecommendedAction = Literal[
    "monitor",
    "open_incident",
    "increase_manual_review",
    "trigger_retraining",
    "rollback_model",
]

ActionType = Literal[
    "monitor_only",
    "incident_opened",
    "manual_review_escalation",
    "threshold_hardening",
    "model_retraining",
    "model_rollback",
]


class ExplanationPayload(BaseModel):
    """
    Technical explanation produced from detector output.

    This is intentionally technical, not business-facing.
    """

    reason: str = Field(..., description="Human-readable technical explanation.")
    top_features: list[str] = Field(
        default_factory=list,
        description="Top shifted or highest-contributing features.",
    )
    summary: str = Field(
        ...,
        description="Compact numeric explanation summary for operators.",
    )


class ImpactAssessment(BaseModel):
    """
    Business-facing impact assessment derived from detector signals.
    """

    estimated_loss_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated financial exposure or impact in USD.",
    )
    severity_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Normalized severity score on a 0-100 scale.",
    )
    affected_segment_count: int = Field(
        ...,
        ge=0,
        description="Estimated number of affected traffic or user segments.",
    )
    business_impact_label: ImpactLabel = Field(
        ...,
        description="Business-oriented severity label.",
    )
    requires_escalation: bool = Field(
        ...,
        description="Whether this incident should be escalated immediately.",
    )
    confidence_drop_pct: float = Field(
        ...,
        ge=0.0,
        description="Estimated drop in model confidence versus healthy baseline.",
    )
    fraud_rate_pct: float = Field(
        ...,
        ge=0.0,
        description="Current fraud-rate percentage used in impact calculation.",
    )
    incident_kpi_breach: bool = Field(
        ...,
        description="Whether a business KPI threshold breach occurred.",
    )


class ProjectedOutcome(BaseModel):
    """What the system predicts will happen after an action is taken."""

    t5_fraud_rate_pct: float = Field(..., description="Projected fraud rate % at T+5 min.")
    t15_fraud_rate_pct: float = Field(..., description="Projected fraud rate % at T+15 min.")
    t30_fraud_rate_pct: float = Field(..., description="Projected fraud rate % at T+30 min.")
    t30_loss_per_hour_usd: float = Field(..., description="Projected loss/hr at T+30 min.")
    loss_saved_per_hour_usd: float = Field(..., description="Loss reduction vs doing nothing.")
    narrative: str = Field(..., description="One-sentence plain-English consequence.")


class DecisionRecommendation(BaseModel):
    """
    Recommended operational action derived from impact + detector state.
    """

    recommended_action: RecommendedAction = Field(
        ...,
        description="Human-meaningful next action.",
    )
    action_type: ActionType = Field(
        ...,
        description="Stable machine-readable action contract.",
    )
    priority: PriorityLabel = Field(
        ...,
        description="Operational priority of the recommendation.",
    )
    rationale: str = Field(
        ...,
        description="Human-readable explanation for the recommendation.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the recommendation, on a 0-1 scale.",
    )
    specific_threshold: str | None = Field(
        default=None,
        description="Concrete operational parameter change, e.g. 'raise review threshold 15% → 28%'.",
    )
    tradeoff: str | None = Field(
        default=None,
        description="Explicit tradeoff statement: what you gain vs what it costs.",
    )
    projected_outcome: ProjectedOutcome | None = Field(
        default=None,
        description="Projected system state after this action is taken.",
    )


class IncidentSummary(BaseModel):
    """
    Compact operator-facing incident brief.

    This summary is assembled from:
        - explanation
        - impact
        - decision
        - detector severity context

    It is not an independent decision system.
    """

    batch_id: int = Field(..., description="Batch identifier associated with the incident.")
    created_at: str = Field(..., description="UTC timestamp for summary creation.")
    title: str = Field(..., description="Short title for the incident.")
    status: Literal["open", "monitoring", "resolved"] = Field(
        ...,
        description="Operator-facing incident state.",
    )
    severity: SeverityLabel = Field(
        ...,
        description="Normalized incident severity.",
    )
    operator_brief: str = Field(
        ...,
        description="Compact incident brief for operators or reviewers.",
    )
    top_features: list[str] = Field(
        default_factory=list,
        description="Top affected features carried into the summary.",
    )
    recommended_action: str = Field(
        ...,
        description="Human-facing recommended action.",
    )
    action_type: str = Field(
        ...,
        description="Stable action type identifier.",
    )
    estimated_loss_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated financial impact in USD.",
    )
    requires_escalation: bool = Field(
        ...,
        description="Whether the incident requires escalation.",
    )

    @classmethod
    def build(
        cls,
        *,
        batch_id: int,
        detector_severity: str,
        explanation: ExplanationPayload,
        impact: ImpactAssessment,
        decision: DecisionRecommendation,
    ) -> "IncidentSummary":
        """
        Build an operator-facing incident summary from lower-level payloads.
        """
        title = f"ML incident at batch {batch_id}"
        status = "open" if impact.requires_escalation else "monitoring"

        operator_brief = (
            f"{explanation.reason}. "
            f"{explanation.summary}. "
            f"Estimated loss: ${impact.estimated_loss_usd:,.0f}. "
            f"Business impact: {impact.business_impact_label}. "
            f"Recommended action: {decision.recommended_action}."
        )

        return cls(
            batch_id=batch_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            title=title,
            status=status,
            severity=_normalize_severity(detector_severity, impact.severity_score),
            operator_brief=operator_brief,
            top_features=explanation.top_features,
            recommended_action=decision.recommended_action,
            action_type=decision.action_type,
            estimated_loss_usd=round(impact.estimated_loss_usd, 2),
            requires_escalation=impact.requires_escalation,
        )


def _normalize_severity(
    detector_severity: str,
    severity_score: int,
) -> SeverityLabel:
    """
    Merge detector severity and normalized impact severity into a stable label.
    """
    normalized = (detector_severity or "").strip().lower()

    if normalized == "critical" or severity_score >= 85:
        return "critical"
    if normalized in {"severe", "high"} or severity_score >= 70:
        return "high"
    if normalized in {"moderate", "medium"} or severity_score >= 40:
        return "medium"
    return "low"