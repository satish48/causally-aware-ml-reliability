"""
src/causal_engine.py
--------------------
Causal attribution engine: correlates ML drift events with upstream pipeline events.

Industry context:
    The gap in current MLOps tooling (Evidently, WhyLogs, Fiddler, Arize) is that
    they tell you WHAT drifted but not WHY. At Google, Stripe, and Meta, the first
    question an on-call engineer asks is "what changed upstream?" This engine
    maintains a timestamped event log and — when drift is detected — generates
    ranked causal hypotheses using time-lagged correlation.

    Inspired by Google's ML Metadata causal tracing, LinkedIn's FAME framework,
    and Uber's Databand root-cause tooling.

Algorithm:
    For each drift alert at time T:
        candidates = upstream events where 0 ≤ (T - event_time) ≤ LOOKBACK_WINDOW
        for each candidate:
            temporal_score  = exp(-ln(2) * lag / HALF_LIFE)   # exponential decay
            overlap_score   = |drifted_features ∩ event_fields| / |drifted_features|
            severity_score  = {critical:1.0, high:0.75, moderate:0.5, info:0.25}
            confidence      = 0.40*temporal + 0.45*overlap + 0.15*severity
        return top-K sorted by confidence descending
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.drift_detector import DriftResult
from src.schema_registry import BreakingChange

logger = logging.getLogger(__name__)

__all__ = [
    "UpstreamEventType",
    "UpstreamEvent",
    "CausalHypothesis",
    "CausalAttribution",
    "CausalEngine",
    "get_causal_engine",
]


UpstreamEventType = Literal[
    "schema_change",
    "deployment",
    "pipeline_anomaly",
    "config_change",
    "data_source_switch",
    "traffic_shift",
]

_SEVERITY_SCORES: dict[str, float] = {
    "critical": 1.00,
    "high": 0.75,
    "moderate": 0.50,
    "info": 0.25,
}


# ────────────────────────────────────────────────────────────────
# DOMAIN TYPES
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class UpstreamEvent:
    event_id: str
    event_type: UpstreamEventType
    source: str
    timestamp: str
    timestamp_unix: float
    affected_fields: list[str]
    severity: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CausalHypothesis:
    rank: int
    event: UpstreamEvent
    confidence: float        # composite [0, 1]
    lag_seconds: float       # drift_time - event_time
    field_overlap_score: float
    temporal_score: float
    explanation: str
    component: str = ""                          # owning component (e.g. "payment_pipeline")
    evidence: list[str] = field(default_factory=list)  # human-readable evidence items


@dataclass(slots=True)
class CausalAttribution:
    drift_batch_id: int
    attributed_at: str
    hypotheses: list[CausalHypothesis]
    root_cause_summary: str
    causal_confidence: float   # max confidence across hypotheses
    attributed: bool           # True when at least one hypothesis found with confidence ≥ 0.3


# ────────────────────────────────────────────────────────────────
# ENGINE
# ────────────────────────────────────────────────────────────────

class CausalEngine:
    """
    Maintain an upstream event timeline and attribute drift to root causes.

    Key design choices:
        - Events are ingested from two sources:
            1. SchemaRegistry violations (auto-converted via ingest_schema_violation)
            2. External events pushed via ingest_upstream_event (deployments, config changes)
        - Temporal scoring uses exponential decay so very recent events score highest.
        - Field overlap is the strongest signal: if the drifted features exactly
          match the fields changed in an upstream event, confidence is high.
        - Minimum confidence 0.3 is required to call an attribution "attributed".
    """

    LOOKBACK_WINDOW_SECONDS: float = 3600.0   # 1 hour — real incidents last longer than 5 min
    TEMPORAL_HALF_LIFE_SECONDS: float = 600.0  # 10-min half-life; old events decay but stay attributable
    TOP_K: int = 5
    MAX_EVENTS: int = 500
    MAX_ATTRIBUTIONS: int = 100

    WEIGHT_TEMPORAL: float = 0.40
    WEIGHT_OVERLAP: float = 0.45
    WEIGHT_SEVERITY: float = 0.15

    ATTRIBUTION_THRESHOLD: float = 0.30
    HIGH_CONFIDENCE_THRESHOLD: float = 0.60

    def __init__(self) -> None:
        self._event_log: deque[UpstreamEvent] = deque(maxlen=self.MAX_EVENTS)
        self._attributions: list[CausalAttribution] = []
        logger.info(
            "CausalEngine initialized | lookback=%.0fs | half_life=%.0fs | top_k=%d",
            self.LOOKBACK_WINDOW_SECONDS,
            self.TEMPORAL_HALF_LIFE_SECONDS,
            self.TOP_K,
        )

    # ── INGESTION ─────────────────────────────────────────────

    def ingest_upstream_event(self, event: UpstreamEvent) -> None:
        """Register an upstream event (deployment, config change, etc.)."""
        self._event_log.append(event)
        logger.info(
            "Upstream event ingested | id=%s | type=%s | source=%s | fields=%s | severity=%s",
            event.event_id, event.event_type, event.source,
            event.affected_fields, event.severity,
        )

    def ingest_schema_violation(self, violation: BreakingChange) -> None:
        """Convert a SchemaRegistry violation into an upstream event and ingest it."""
        ts = violation.detected_at
        try:
            ts_unix = datetime.fromisoformat(ts).timestamp()
        except ValueError:
            ts_unix = datetime.now(timezone.utc).timestamp()

        event = UpstreamEvent(
            event_id=f"se-{violation.change_id}",
            event_type="schema_change",
            source=violation.source,
            timestamp=ts,
            timestamp_unix=ts_unix,
            affected_fields=[violation.field_name],
            severity=violation.severity,
            description=violation.description,
            metadata={
                "change_type": violation.change_type.value,
                "old_value": violation.old_value,
                "new_value": violation.new_value,
            },
        )
        self.ingest_upstream_event(event)

    # ── ATTRIBUTION ───────────────────────────────────────────

    def attribute(self, drift_result: DriftResult) -> CausalAttribution:
        """
        Generate ranked causal hypotheses for a drift detection result.

        Only call this when drift_detected is True. Returns an attribution object
        with hypotheses sorted by confidence descending.
        """
        drift_ts = self._parse_ts(drift_result.timestamp)
        drifted_features = {
            s.feature
            for s in drift_result.feature_scores
            if s.drifted or s.psi >= 0.15
        }

        candidates = [
            e for e in self._event_log
            if 0.0 <= (drift_ts - e.timestamp_unix) <= self.LOOKBACK_WINDOW_SECONDS
        ]

        if not candidates:
            attribution = CausalAttribution(
                drift_batch_id=drift_result.batch_id,
                attributed_at=datetime.now(timezone.utc).isoformat(),
                hypotheses=[],
                root_cause_summary=(
                    "No upstream events found within the lookback window. "
                    "Drift may be a gradual distribution shift or a pipeline change "
                    "not yet registered in the event log. Consider retraining."
                ),
                causal_confidence=0.0,
                attributed=False,
            )
            self._store(attribution)
            return attribution

        scored: list[tuple[float, CausalHypothesis]] = []
        for event in candidates:
            lag = drift_ts - event.timestamp_unix
            temporal = math.exp(-math.log(2) * lag / self.TEMPORAL_HALF_LIFE_SECONDS)

            if drifted_features and event.affected_fields:
                overlap = len(drifted_features & set(event.affected_fields)) / len(drifted_features)
            else:
                overlap = 0.0

            sev = _SEVERITY_SCORES.get(event.severity, 0.25)
            confidence = (
                self.WEIGHT_TEMPORAL * temporal
                + self.WEIGHT_OVERLAP * overlap
                + self.WEIGHT_SEVERITY * sev
            )

            explanation = (
                f"{event.event_type.replace('_', ' ').title()} on '{event.source}' "
                f"occurred {lag:.0f}s before drift onset "
                f"(temporal={temporal:.2f}, field_overlap={overlap:.2f}, "
                f"severity={event.severity}, confidence={confidence:.2f}). "
                f"{event.description}"
            )

            evidence_items: list[str] = [
                f"Temporal alignment: event occurred {lag:.0f}s before drift onset "
                f"(score={temporal:.2f}, half-life={self.TEMPORAL_HALF_LIFE_SECONDS:.0f}s).",
            ]
            if drifted_features and event.affected_fields:
                overlap_fields = sorted(drifted_features & set(event.affected_fields))
                if overlap_fields:
                    evidence_items.append(
                        f"Field overlap: {len(overlap_fields)} shared field(s) — "
                        f"{', '.join(overlap_fields[:5])}."
                    )
                else:
                    evidence_items.append(
                        "No direct field overlap with drifted features; "
                        "indirect dependency is possible."
                    )
            elif not drifted_features:
                evidence_items.append(
                    "Drifted feature set is unknown; overlap could not be computed."
                )
            change_type = event.metadata.get("change_type")
            if change_type:
                old_v = event.metadata.get("old_value", "?")
                new_v = event.metadata.get("new_value", "?")
                evidence_items.append(
                    f"Schema change type: {change_type} "
                    f"('{old_v}' → '{new_v}')."
                )
            evidence_items.append(
                f"Severity: {event.severity} "
                f"(score={sev:.2f} of 1.0)."
            )

            scored.append((confidence, CausalHypothesis(
                rank=0,
                event=event,
                confidence=round(confidence, 3),
                lag_seconds=round(lag, 1),
                field_overlap_score=round(overlap, 3),
                temporal_score=round(temporal, 3),
                explanation=explanation,
                component=event.source,
                evidence=evidence_items,
            )))

        scored.sort(key=lambda x: x[0], reverse=True)
        hypotheses: list[CausalHypothesis] = []
        for rank, (_, hyp) in enumerate(scored[:self.TOP_K], start=1):
            hyp.rank = rank
            hypotheses.append(hyp)

        best_conf = hypotheses[0].confidence if hypotheses else 0.0
        best = hypotheses[0] if hypotheses else None
        summary = self._build_summary(best, drift_result.batch_id)

        attribution = CausalAttribution(
            drift_batch_id=drift_result.batch_id,
            attributed_at=datetime.now(timezone.utc).isoformat(),
            hypotheses=hypotheses,
            root_cause_summary=summary,
            causal_confidence=best_conf,
            attributed=best_conf >= self.ATTRIBUTION_THRESHOLD,
        )
        self._store(attribution)

        logger.info(
            "Causal attribution | batch_id=%d | candidates=%d | hypotheses=%d | "
            "best_confidence=%.2f | attributed=%s",
            drift_result.batch_id, len(candidates), len(hypotheses),
            best_conf, attribution.attributed,
        )
        return attribution

    # ── QUERIES ───────────────────────────────────────────────

    def get_latest_attribution(self) -> CausalAttribution | None:
        return self._attributions[-1] if self._attributions else None

    def get_attributions_snapshot(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "drift_batch_id": a.drift_batch_id,
                "attributed_at": a.attributed_at,
                "causal_confidence": a.causal_confidence,
                "attributed": a.attributed,
                "root_cause_summary": a.root_cause_summary,
                "hypotheses": [
                    {
                        "rank": h.rank,
                        "confidence": h.confidence,
                        "lag_seconds": h.lag_seconds,
                        "field_overlap_score": h.field_overlap_score,
                        "temporal_score": h.temporal_score,
                        "event_type": h.event.event_type,
                        "source": h.event.source,
                        "severity": h.event.severity,
                        "affected_fields": h.event.affected_fields,
                        "explanation": h.explanation,
                        "component": h.component,
                        "evidence": h.evidence,
                    }
                    for h in a.hypotheses
                ],
            }
            for a in self._attributions[-limit:]
        ]

    def get_event_log_snapshot(self, limit: int = 50) -> list[dict[str, Any]]:
        events = list(self._event_log)[-limit:]
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "source": e.source,
                "timestamp": e.timestamp,
                "affected_fields": e.affected_fields,
                "severity": e.severity,
                "description": e.description,
            }
            for e in reversed(events)
        ]

    def get_event_objects(self, limit: int | None = None) -> list[UpstreamEvent]:
        events = list(self._event_log)
        return events if limit is None else events[-limit:]

    def get_stats(self) -> dict[str, Any]:
        return {
            "events_in_log": len(self._event_log),
            "attributions_made": len(self._attributions),
            "attributions_successful": sum(1 for a in self._attributions if a.attributed),
            "lookback_window_seconds": self.LOOKBACK_WINDOW_SECONDS,
        }

    # ── INTERNAL ──────────────────────────────────────────────

    def _store(self, attribution: CausalAttribution) -> None:
        self._attributions.append(attribution)
        if len(self._attributions) > self.MAX_ATTRIBUTIONS:
            self._attributions = self._attributions[-self.MAX_ATTRIBUTIONS:]

    @staticmethod
    def _parse_ts(ts: str) -> float:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            return datetime.now(timezone.utc).timestamp()

    def _build_summary(self, best: CausalHypothesis | None, batch_id: int) -> str:
        if best is None:
            return (
                "No upstream events found. Likely gradual distribution shift. "
                "Consider scheduling a retraining run."
            )
        if best.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return (
                f"HIGH CONFIDENCE: {best.event.event_type.replace('_', ' ').title()} "
                f"on '{best.event.source}' (lag={best.lag_seconds:.0f}s, "
                f"confidence={best.confidence:.2f}) is the most probable root cause "
                f"of drift in batch {batch_id}. "
                f"Fields affected: {', '.join(best.event.affected_fields) or 'unknown'}."
            )
        if best.confidence >= self.ATTRIBUTION_THRESHOLD:
            return (
                f"MODERATE CONFIDENCE: {best.event.event_type.replace('_', ' ').title()} "
                f"on '{best.event.source}' may have contributed to drift "
                f"(confidence={best.confidence:.2f}). Additional investigation recommended."
            )
        return (
            "LOW CONFIDENCE: No upstream event strongly correlates with this drift. "
            "Most likely a gradual distribution shift. Schedule retraining."
        )


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_engine: CausalEngine | None = None


def get_causal_engine() -> CausalEngine:
    global _engine
    if _engine is None:
        _engine = CausalEngine()
    return _engine
