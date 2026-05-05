"""
src/causal_timeline.py
----------------------
Causal timeline reconstruction engine.

Industry context:
    When an on-call engineer gets paged at 3am, they need a single chronological
    chain — not five separate dashboards. Google SRE calls this the "incident
    timeline". This module merges events from four distinct signal sources
    (upstream events, schema violations, feature drift, SLO burn alerts) into
    one sorted timeline with a human-readable narrative.

    Each event is typed, severity-stamped, and annotated so the engineer can
    trace the propagation: upstream deployment → schema change → feature drift
    → model degradation → SLO burn.

Design:
    - Merge-sort by timestamp_unix across all four source lists
    - Temporal window filter (default 30 min) keeps timeline actionable
    - Narrative is generated from the sorted chain, identifying the root trigger
      vs. downstream symptoms
    - Inconclusive timelines are explicitly flagged rather than silently omitted
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.causal_engine import UpstreamEvent
from src.drift_detector import DriftResult
from src.schema_registry import BreakingChange
from src.slo_engine import BurnRateAlert

logger = logging.getLogger(__name__)

__all__ = [
    "TimelineEvent",
    "CausalTimeline",
    "CausalTimelineEngine",
    "get_causal_timeline_engine",
]

_SEVERITY_ORDER: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "info": 1,
    "stable": 0,
}

_EVENT_TYPE_LABELS: dict[str, str] = {
    "schema_change": "Schema Change",
    "deployment": "Deployment",
    "pipeline_anomaly": "Pipeline Anomaly",
    "config_change": "Config Change",
    "data_source_switch": "Data Source Switch",
    "traffic_shift": "Traffic Shift",
    "feature_drift": "Feature Drift",
    "model_degradation": "Model Degradation",
    "slo_alert": "SLO Alert",
}


# ────────────────────────────────────────────────────────────────
# DOMAIN TYPES
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class TimelineEvent:
    time_label: str          # "14:28:03"
    timestamp_unix: float
    event_type: str          # one of _EVENT_TYPE_LABELS keys
    severity: str            # critical | high | moderate | info
    title: str
    description: str
    source: str
    batch_id: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_label": self.time_label,
            "timestamp_unix": self.timestamp_unix,
            "event_type": self.event_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "batch_id": self.batch_id,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CausalTimeline:
    events: list[TimelineEvent]
    narrative: str
    window_minutes: float
    generated_at: str
    inconclusive: bool        # True when evidence is sparse or contradictory
    trigger_event: TimelineEvent | None   # earliest high-severity event (likely root)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "narrative": self.narrative,
            "window_minutes": self.window_minutes,
            "generated_at": self.generated_at,
            "inconclusive": self.inconclusive,
            "trigger_event": self.trigger_event.to_dict() if self.trigger_event else None,
            "event_count": len(self.events),
        }


# ────────────────────────────────────────────────────────────────
# ENGINE
# ────────────────────────────────────────────────────────────────

class CausalTimelineEngine:
    """
    Build chronological incident timelines from heterogeneous signal sources.

    Call build_timeline() with lists of events from each signal source.
    The engine handles deduplication (same source + timestamp + type),
    window filtering, severity-aware narrative, and trigger identification.

    Thread-safety: stateless per call — safe for concurrent use.
    """

    MAX_EVENTS_IN_TIMELINE: int = 100

    def build_timeline(
        self,
        upstream_events: list[UpstreamEvent],
        schema_violations: list[BreakingChange],
        drift_results: list[DriftResult],
        slo_alerts: list[BurnRateAlert],
        window_minutes: float = 30.0,
    ) -> CausalTimeline:
        """
        Merge all signal sources into a chronological incident timeline.

        Args:
            upstream_events:    ingested upstream events (deployments, config changes)
            schema_violations:  schema breaking changes from SchemaRegistry
            drift_results:      drift results — ALL included (not just drift_detected=True)
                                so the timeline shows the full health trajectory
            slo_alerts:         burn rate alerts from SLOEngine
            window_minutes:     only include events within this many minutes of now

        Returns:
            CausalTimeline with sorted events and narrative.
        """
        now_unix = datetime.now(timezone.utc).timestamp()
        cutoff = now_unix - (window_minutes * 60.0)

        all_events: list[TimelineEvent] = []

        all_events.extend(self._from_upstream_events(upstream_events, cutoff))
        all_events.extend(self._from_schema_violations(schema_violations, cutoff))
        all_events.extend(self._from_drift_results(drift_results, cutoff))
        all_events.extend(self._from_slo_alerts(slo_alerts, cutoff))

        all_events.sort(key=lambda e: e.timestamp_unix)

        # Deduplicate: same source + event_type within a 60-second window.
        # A 1-second bucket was too fine — repeated SLO alerts fired every batch
        # (every ~2 s) all landed in unique buckets, flooding the timeline with
        # identical entries. 60 s keeps one representative event per minute.
        seen: set[str] = set()
        deduped: list[TimelineEvent] = []
        for ev in all_events:
            bucket = int(ev.timestamp_unix) // 60
            key = f"{ev.source}:{ev.event_type}:{bucket}"
            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        deduped = deduped[-self.MAX_EVENTS_IN_TIMELINE:]

        trigger = self._find_trigger(deduped)
        narrative = self._build_narrative(deduped, trigger, window_minutes)
        inconclusive = self._is_inconclusive(deduped)

        timeline = CausalTimeline(
            events=deduped,
            narrative=narrative,
            window_minutes=window_minutes,
            generated_at=datetime.now(timezone.utc).isoformat(),
            inconclusive=inconclusive,
            trigger_event=trigger,
        )

        logger.debug(
            "Timeline built | events=%d | window=%.0fm | inconclusive=%s | trigger=%s",
            len(deduped), window_minutes, inconclusive,
            trigger.event_type if trigger else "none",
        )
        return timeline

    # ── SOURCE CONVERTERS ──────────────────────────────────────

    def _from_upstream_events(
        self,
        events: list[UpstreamEvent],
        cutoff: float,
    ) -> list[TimelineEvent]:
        out = []
        for e in events:
            if e.timestamp_unix < cutoff:
                continue
            label = _format_time(e.timestamp_unix)
            fields_str = ", ".join(e.affected_fields[:5]) or "unknown"
            out.append(TimelineEvent(
                time_label=label,
                timestamp_unix=e.timestamp_unix,
                event_type=e.event_type,
                severity=e.severity,
                title=f"{_EVENT_TYPE_LABELS.get(e.event_type, e.event_type)}: {e.source}",
                description=e.description,
                source=e.source,
                batch_id=None,
                metadata={
                    "event_id": e.event_id,
                    "affected_fields": e.affected_fields,
                    "fields_str": fields_str,
                    **e.metadata,
                },
            ))
        return out

    def _from_schema_violations(
        self,
        violations: list[BreakingChange],
        cutoff: float,
    ) -> list[TimelineEvent]:
        out = []
        for v in violations:
            try:
                ts_unix = datetime.fromisoformat(v.detected_at).timestamp()
            except ValueError:
                continue
            if ts_unix < cutoff:
                continue
            label = _format_time(ts_unix)
            out.append(TimelineEvent(
                time_label=label,
                timestamp_unix=ts_unix,
                event_type="schema_change",
                severity=v.severity,
                title=f"Schema {v.change_type.value.replace('_', ' ').title()}: {v.field_name}",
                description=v.description,
                source=v.source,
                batch_id=v.batch_id,
                metadata={
                    "change_id": v.change_id,
                    "change_type": v.change_type.value,
                    "old_value": v.old_value,
                    "new_value": v.new_value,
                    "field_name": v.field_name,
                },
            ))
        return out

    def _from_drift_results(
        self,
        results: list[DriftResult],
        cutoff: float,
    ) -> list[TimelineEvent]:
        out = []
        for r in results:
            try:
                ts_unix = datetime.fromisoformat(r.timestamp).timestamp()
            except ValueError:
                continue
            if ts_unix < cutoff:
                continue

            if r.drift_detected:
                event_type = "model_degradation"
                severity = r.severity if r.severity in _SEVERITY_ORDER else "moderate"
                drifted = [s.feature for s in r.feature_scores if s.drifted]
                desc = (
                    f"Drift detected in batch {r.batch_id} "
                    f"(PSI={r.overall_psi:.3f}, max_z={r.max_z_score:.2f}). "
                    f"Affected features: {', '.join(drifted[:5]) or 'unknown'}. "
                    f"Fraud rate: {r.fraud_rate:.1%}, Confidence: {r.avg_confidence:.1%}."
                )
                title = f"Model Drift Detected — batch {r.batch_id}"
            else:
                event_type = "feature_drift"
                severity = "info"
                desc = (
                    f"Batch {r.batch_id} stable "
                    f"(PSI={r.overall_psi:.3f}, fraud_rate={r.fraud_rate:.1%})."
                )
                title = f"Healthy Batch — {r.batch_id}"

            label = _format_time(ts_unix)
            out.append(TimelineEvent(
                time_label=label,
                timestamp_unix=ts_unix,
                event_type=event_type,
                severity=severity,
                title=title,
                description=desc,
                source="fraud_detection_v1",
                batch_id=r.batch_id,
                metadata={
                    "overall_psi": r.overall_psi,
                    "max_z_score": r.max_z_score,
                    "fraud_rate": r.fraud_rate,
                    "avg_confidence": r.avg_confidence,
                    "drift_detected": r.drift_detected,
                },
            ))
        return out

    def _from_slo_alerts(
        self,
        alerts: list[BurnRateAlert],
        cutoff: float,
    ) -> list[TimelineEvent]:
        out = []
        for a in alerts:
            try:
                ts_unix = datetime.fromisoformat(a.triggered_at).timestamp()
            except ValueError:
                continue
            if ts_unix < cutoff:
                continue

            severity = "critical" if a.alert_type in ("fast_burn", "budget_exhausted") else "high"
            label = _format_time(ts_unix)
            out.append(TimelineEvent(
                time_label=label,
                timestamp_unix=ts_unix,
                event_type="slo_alert",
                severity=severity,
                title=f"SLO {a.alert_type.replace('_', ' ').title()}: {a.slo_name}",
                description=a.message,
                source=a.slo_name,
                batch_id=a.batch_id,
                metadata={
                    "alert_id": a.alert_id,
                    "alert_type": a.alert_type,
                    "burn_rate": a.burn_rate,
                    "budget_remaining_pct": a.budget_remaining_pct,
                    "exhaustion_eta_batches": a.exhaustion_eta_batches,
                },
            ))
        return out

    # ── NARRATIVE & ANALYSIS ──────────────────────────────────

    def _find_trigger(self, events: list[TimelineEvent]) -> TimelineEvent | None:
        """Identify the earliest high-severity event as the likely root trigger."""
        high_sev = [
            e for e in events
            if _SEVERITY_ORDER.get(e.severity, 0) >= _SEVERITY_ORDER["high"]
        ]
        return high_sev[0] if high_sev else (events[0] if events else None)

    def _is_inconclusive(self, events: list[TimelineEvent]) -> bool:
        """
        Flag the timeline as inconclusive when evidence is sparse or contradictory:
        - Fewer than 2 events
        - No upstream events (only model-side signals; can't attribute to pipeline)
        - Conflicting severity signals within 60 seconds
        """
        if len(events) < 2:
            return True

        upstream_types = {"schema_change", "deployment", "pipeline_anomaly",
                          "config_change", "data_source_switch", "traffic_shift"}
        has_upstream = any(e.event_type in upstream_types for e in events)
        if not has_upstream:
            return True

        # Detect conflicting signals: critical followed by info within 60s
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            if (b.timestamp_unix - a.timestamp_unix <= 60.0
                    and _SEVERITY_ORDER.get(a.severity, 0) >= 3
                    and _SEVERITY_ORDER.get(b.severity, 0) <= 1):
                return True

        return False

    def _build_narrative(
        self,
        events: list[TimelineEvent],
        trigger: TimelineEvent | None,
        window_minutes: float,
    ) -> str:
        if not events:
            return (
                f"No events recorded in the past {window_minutes:.0f} minutes. "
                "The system appears stable, or event ingestion may be lagging."
            )

        severity_counts: dict[str, int] = {}
        for e in events:
            severity_counts[e.severity] = severity_counts.get(e.severity, 0) + 1

        critical_count = severity_counts.get("critical", 0)
        high_count = severity_counts.get("high", 0)

        parts: list[str] = []

        if trigger:
            parts.append(
                f"Incident timeline ({len(events)} events, last {window_minutes:.0f} min). "
                f"Likely trigger: {trigger.title} at {trigger.time_label} [{trigger.severity.upper()}]."
            )
        else:
            parts.append(
                f"Timeline of {len(events)} events in the past {window_minutes:.0f} minutes."
            )

        if critical_count:
            parts.append(f"{critical_count} critical signal(s) detected.")
        if high_count:
            parts.append(f"{high_count} high-severity signal(s) detected.")

        # Describe propagation chain if we have upstream→drift→slo
        event_types = {e.event_type for e in events}
        if {"schema_change", "model_degradation", "slo_alert"}.issubset(event_types):
            parts.append(
                "Propagation chain confirmed: upstream schema change → model drift → SLO burn."
            )
        elif {"schema_change", "model_degradation"}.issubset(event_types):
            parts.append(
                "Partial chain: upstream schema change preceded model drift. "
                "SLO budget not yet exhausted."
            )
        elif "model_degradation" in event_types and "slo_alert" in event_types:
            parts.append(
                "Model degradation escalated to SLO burn. "
                "No upstream event registered — check pipeline logs manually."
            )

        return " ".join(parts)


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def _format_time(ts_unix: float) -> str:
    """Format unix timestamp as HH:MM:SS in UTC."""
    return datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%H:%M:%S")


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_timeline_engine: CausalTimelineEngine | None = None


def get_causal_timeline_engine() -> CausalTimelineEngine:
    global _timeline_engine
    if _timeline_engine is None:
        _timeline_engine = CausalTimelineEngine()
    return _timeline_engine
