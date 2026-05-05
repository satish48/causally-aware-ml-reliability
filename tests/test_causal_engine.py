"""
tests/test_causal_engine.py
----------------------------
Unit tests for CausalEngine: event ingestion, attribution scoring,
hypothesis ranking, and edge cases.
"""

import math
import time
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from src.causal_engine import CausalEngine, UpstreamEvent
from src.drift_detector import DriftResult, FeatureDriftScore
from src.schema_registry import BreakingChange, BreakingChangeType


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def make_engine() -> CausalEngine:
    return CausalEngine()


def _ts(offset_seconds: float = 0.0) -> tuple[str, float]:
    """Return (iso_string, unix_float) for now + offset_seconds."""
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.isoformat(), dt.timestamp()


def make_event(
    event_type: str = "schema_change",
    source: str = "payments",
    affected_fields: list[str] | None = None,
    severity: str = "high",
    seconds_ago: float = 30.0,
) -> UpstreamEvent:
    ts_str, ts_unix = _ts(-seconds_ago)
    return UpstreamEvent(
        event_id=f"ev-{uuid.uuid4().hex[:8]}",
        event_type=event_type,  # type: ignore[arg-type]
        source=source,
        timestamp=ts_str,
        timestamp_unix=ts_unix,
        affected_fields=affected_fields or ["amount"],
        severity=severity,
        description="test event",
    )


def make_drift_result(
    batch_id: int = 1,
    drift_detected: bool = True,
    severity: str = "high",
    drifted_features: list[str] | None = None,
    seconds_ago: float = 0.0,
) -> DriftResult:
    features = drifted_features or ["amount", "hour"]
    ts_str, _ = _ts(-seconds_ago)
    return DriftResult(
        batch_id=batch_id,
        timestamp=ts_str,
        overall_psi=0.30,
        max_z_score=3.5,
        drift_detected=drift_detected,
        severity=severity,
        feature_scores=[
            FeatureDriftScore(
                feature=f,
                z_score=3.0,
                psi=0.25,
                drifted=True,
                severity="high",
            )
            for f in features
        ],
        fraud_rate=0.20,
        avg_confidence=0.80,
        processing_ms=5.0,
        warmed_up=True,
    )


# ────────────────────────────────────────────────────────────────
# INGESTION
# ────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_ingest_upstream_event(self):
        eng = make_engine()
        event = make_event()
        eng.ingest_upstream_event(event)
        assert eng.get_stats()["events_in_log"] == 1

    def test_ingest_schema_violation(self):
        eng = make_engine()
        violation = BreakingChange(
            change_id="vc-000001",
            change_type=BreakingChangeType.FIELD_REMOVED,
            source="payments",
            field_name="amount",
            old_value="numeric",
            new_value="absent",
            severity="critical",
            detected_at=datetime.now(timezone.utc).isoformat(),
            batch_id=1,
            description="Test violation",
        )
        eng.ingest_schema_violation(violation)
        assert eng.get_stats()["events_in_log"] == 1

    def test_event_log_bounded(self):
        eng = make_engine()
        eng.MAX_EVENTS = 3
        from collections import deque
        eng._event_log = deque(maxlen=3)
        for i in range(5):
            eng.ingest_upstream_event(make_event())
        assert len(eng._event_log) == 3


# ────────────────────────────────────────────────────────────────
# ATTRIBUTION — NO CANDIDATES
# ────────────────────────────────────────────────────────────────

class TestAttributionNoCandidates:
    def test_no_events_returns_unattributed(self):
        eng = make_engine()
        result = eng.attribute(make_drift_result())
        assert result.attributed is False
        assert result.causal_confidence == 0.0
        assert result.hypotheses == []

    def test_event_outside_lookback_window_not_attributed(self):
        eng = make_engine()
        # Event happened 2 hours ago — outside the 1-hour (3600s) lookback window.
        eng.ingest_upstream_event(make_event(seconds_ago=7300.0))
        result = eng.attribute(make_drift_result())
        assert result.attributed is False


# ────────────────────────────────────────────────────────────────
# ATTRIBUTION — WITH CANDIDATES
# ────────────────────────────────────────────────────────────────

class TestAttributionWithCandidates:
    def test_recent_event_produces_hypothesis(self):
        eng = make_engine()
        eng.ingest_upstream_event(make_event(seconds_ago=30.0, affected_fields=["amount", "hour"]))
        result = eng.attribute(make_drift_result(drifted_features=["amount", "hour"]))
        assert len(result.hypotheses) >= 1

    def test_hypothesis_confidence_between_0_and_1(self):
        eng = make_engine()
        eng.ingest_upstream_event(make_event(seconds_ago=20.0, affected_fields=["amount"]))
        result = eng.attribute(make_drift_result(drifted_features=["amount"]))
        for h in result.hypotheses:
            assert 0.0 <= h.confidence <= 1.0

    def test_hypothesis_rank_starts_at_1(self):
        eng = make_engine()
        eng.ingest_upstream_event(make_event(seconds_ago=20.0))
        eng.ingest_upstream_event(make_event(seconds_ago=60.0))
        result = eng.attribute(make_drift_result())
        assert result.hypotheses[0].rank == 1

    def test_hypotheses_sorted_by_confidence_descending(self):
        eng = make_engine()
        # Very recent event with matching field
        eng.ingest_upstream_event(make_event(seconds_ago=5.0, affected_fields=["amount"], severity="critical"))
        # Older event with no field overlap
        eng.ingest_upstream_event(make_event(seconds_ago=200.0, affected_fields=["unrelated"], severity="info"))
        result = eng.attribute(make_drift_result(drifted_features=["amount"]))
        confidences = [h.confidence for h in result.hypotheses]
        assert confidences == sorted(confidences, reverse=True)

    def test_perfect_field_overlap_boosts_confidence(self):
        """Event affecting exactly the same fields as drift should score higher."""
        eng1 = make_engine()
        eng1.ingest_upstream_event(make_event(seconds_ago=30.0, affected_fields=["amount", "hour"]))
        r1 = eng1.attribute(make_drift_result(drifted_features=["amount", "hour"]))

        eng2 = make_engine()
        eng2.ingest_upstream_event(make_event(seconds_ago=30.0, affected_fields=["unrelated"]))
        r2 = eng2.attribute(make_drift_result(drifted_features=["amount", "hour"]))

        assert r1.hypotheses[0].confidence > r2.hypotheses[0].confidence

    def test_closer_event_scores_higher_than_distant(self):
        """Temporal decay: a 10s-lag event should beat a 200s-lag event (same fields)."""
        eng = make_engine()
        eng.ingest_upstream_event(make_event(seconds_ago=10.0, affected_fields=["amount"]))
        eng.ingest_upstream_event(make_event(seconds_ago=200.0, affected_fields=["amount"]))
        result = eng.attribute(make_drift_result(drifted_features=["amount"]))
        assert result.hypotheses[0].lag_seconds < result.hypotheses[1].lag_seconds

    def test_high_confidence_attributions_marked_attributed(self):
        eng = make_engine()
        # Close event, full overlap, critical severity → confidence should exceed 0.30
        eng.ingest_upstream_event(
            make_event(seconds_ago=5.0, affected_fields=["amount", "hour"], severity="critical")
        )
        result = eng.attribute(make_drift_result(drifted_features=["amount", "hour"]))
        assert result.attributed is True

    def test_top_k_limit_respected(self):
        eng = make_engine()
        for i in range(10):
            eng.ingest_upstream_event(make_event(seconds_ago=float(i * 10 + 5)))
        result = eng.attribute(make_drift_result())
        assert len(result.hypotheses) <= eng.TOP_K


# ────────────────────────────────────────────────────────────────
# TEMPORAL SCORING (math)
# ────────────────────────────────────────────────────────────────

class TestTemporalDecay:
    def test_zero_lag_score_is_1(self):
        eng = make_engine()
        score = math.exp(-math.log(2) * 0 / eng.TEMPORAL_HALF_LIFE_SECONDS)
        assert abs(score - 1.0) < 1e-9

    def test_half_life_lag_score_is_half(self):
        eng = make_engine()
        lag = eng.TEMPORAL_HALF_LIFE_SECONDS
        score = math.exp(-math.log(2) * lag / eng.TEMPORAL_HALF_LIFE_SECONDS)
        assert abs(score - 0.5) < 1e-6


# ────────────────────────────────────────────────────────────────
# SNAPSHOTS & STATS
# ────────────────────────────────────────────────────────────────

class TestSnapshots:
    def test_attributions_snapshot(self):
        eng = make_engine()
        eng.ingest_upstream_event(make_event(seconds_ago=20.0))
        eng.attribute(make_drift_result(batch_id=1))
        eng.attribute(make_drift_result(batch_id=2))
        snapshot = eng.get_attributions_snapshot(limit=1)
        assert len(snapshot) == 1

    def test_stats_track_attributions(self):
        eng = make_engine()
        eng.ingest_upstream_event(make_event(seconds_ago=5.0, affected_fields=["amount"], severity="critical"))
        eng.attribute(make_drift_result(batch_id=1, drifted_features=["amount"]))
        stats = eng.get_stats()
        assert stats["attributions_made"] == 1
        assert stats["events_in_log"] == 1
