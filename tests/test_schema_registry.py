"""
tests/test_schema_registry.py
------------------------------
Unit tests for SchemaRegistry: fingerprinting, version tracking,
and breaking-change detection.
"""

import pytest

from src.schema_registry import (
    BreakingChangeType,
    SchemaRegistry,
    _fingerprint,
    FieldType,
)


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def make_registry() -> SchemaRegistry:
    return SchemaRegistry()


BASE_FEATURES = {"amount": 85.0, "hour": 13.0, "tx_count": 3.0}


# ────────────────────────────────────────────────────────────────
# FINGERPRINT
# ────────────────────────────────────────────────────────────────

class TestFingerprint:
    def test_same_fields_same_fingerprint(self):
        fp1 = _fingerprint({"a": FieldType.NUMERIC, "b": FieldType.CATEGORICAL})
        fp2 = _fingerprint({"b": FieldType.CATEGORICAL, "a": FieldType.NUMERIC})
        assert fp1 == fp2, "Fingerprint must be order-independent"

    def test_different_fields_different_fingerprint(self):
        fp1 = _fingerprint({"a": FieldType.NUMERIC})
        fp2 = _fingerprint({"b": FieldType.NUMERIC})
        assert fp1 != fp2

    def test_fingerprint_length(self):
        fp = _fingerprint({"x": FieldType.NUMERIC})
        assert len(fp) == 16, "Fingerprint must be 16 hex chars"


# ────────────────────────────────────────────────────────────────
# STABLE SCHEMA (NO VIOLATIONS)
# ────────────────────────────────────────────────────────────────

class TestStableSchema:
    def test_first_registration_no_violations(self):
        reg = make_registry()
        violations = reg.register_batch_schema(
            source="payments", batch_id=1, feature_means=BASE_FEATURES
        )
        assert violations == []

    def test_same_schema_repeated_no_violations(self):
        reg = make_registry()
        reg.register_batch_schema(source="payments", batch_id=1, feature_means=BASE_FEATURES)
        violations = reg.register_batch_schema(
            source="payments", batch_id=2, feature_means=BASE_FEATURES
        )
        assert violations == []

    def test_version_history_grows(self):
        reg = make_registry()
        for i in range(1, 4):
            reg.register_batch_schema(source="payments", batch_id=i, feature_means=BASE_FEATURES)
        history = reg.get_version_history("payments")
        assert len(history) == 3

    def test_different_sources_are_independent(self):
        reg = make_registry()
        reg.register_batch_schema(source="source_a", batch_id=1, feature_means={"x": 1.0})
        violations = reg.register_batch_schema(
            source="source_b", batch_id=1, feature_means={"y": 2.0}
        )
        assert violations == []


# ────────────────────────────────────────────────────────────────
# FIELD REMOVED (critical)
# ────────────────────────────────────────────────────────────────

class TestFieldRemoved:
    def test_field_removed_emits_critical_violation(self):
        reg = make_registry()
        reg.register_batch_schema(source="p", batch_id=1, feature_means=BASE_FEATURES)

        shrunk = {"amount": 85.0, "hour": 13.0}  # tx_count removed
        violations = reg.register_batch_schema(source="p", batch_id=2, feature_means=shrunk)

        assert len(violations) == 1
        v = violations[0]
        assert v.change_type == BreakingChangeType.FIELD_REMOVED
        assert v.field_name == "tx_count"
        assert v.severity == "critical"
        assert v.source == "p"
        assert v.batch_id == 2

    def test_multiple_fields_removed_multiple_violations(self):
        reg = make_registry()
        reg.register_batch_schema(source="p", batch_id=1, feature_means=BASE_FEATURES)
        violations = reg.register_batch_schema(
            source="p", batch_id=2, feature_means={"amount": 85.0}
        )
        removed = [v for v in violations if v.change_type == BreakingChangeType.FIELD_REMOVED]
        assert len(removed) == 2


# ────────────────────────────────────────────────────────────────
# FIELD ADDED (info)
# ────────────────────────────────────────────────────────────────

class TestFieldAdded:
    def test_field_added_emits_info_violation(self):
        reg = make_registry()
        reg.register_batch_schema(source="p", batch_id=1, feature_means=BASE_FEATURES)

        expanded = {**BASE_FEATURES, "new_feature": 42.0}
        violations = reg.register_batch_schema(source="p", batch_id=2, feature_means=expanded)

        added = [v for v in violations if v.change_type == BreakingChangeType.FIELD_ADDED]
        assert len(added) == 1
        assert added[0].field_name == "new_feature"
        assert added[0].severity == "info"


# ────────────────────────────────────────────────────────────────
# NULL RATE SPIKE (high)
# ────────────────────────────────────────────────────────────────

class TestNullRateSpike:
    def test_null_rate_spike_emits_high_violation(self):
        reg = make_registry()
        reg.register_batch_schema(
            source="p", batch_id=1,
            feature_means=BASE_FEATURES,
            feature_null_rates={"amount": 0.01},
        )
        violations = reg.register_batch_schema(
            source="p", batch_id=2,
            feature_means=BASE_FEATURES,
            feature_null_rates={"amount": 0.20},  # 19% spike
        )
        null_violations = [v for v in violations if v.change_type == BreakingChangeType.NULL_RATE_SPIKE]
        assert len(null_violations) == 1
        assert null_violations[0].severity == "high"
        assert null_violations[0].field_name == "amount"

    def test_small_null_rate_change_no_violation(self):
        reg = make_registry()
        reg.register_batch_schema(
            source="p", batch_id=1,
            feature_means=BASE_FEATURES,
            feature_null_rates={"amount": 0.01},
        )
        violations = reg.register_batch_schema(
            source="p", batch_id=2,
            feature_means=BASE_FEATURES,
            feature_null_rates={"amount": 0.05},  # only 4% delta
        )
        null_violations = [v for v in violations if v.change_type == BreakingChangeType.NULL_RATE_SPIKE]
        assert null_violations == []


# ────────────────────────────────────────────────────────────────
# STATS
# ────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_reflect_violations(self):
        reg = make_registry()
        reg.register_batch_schema(source="p", batch_id=1, feature_means=BASE_FEATURES)
        reg.register_batch_schema(source="p", batch_id=2, feature_means={"amount": 1.0})
        stats = reg.get_stats()
        assert stats["critical_violations"] >= 2  # hour and tx_count removed
        assert stats["total_violations"] >= 2

    def test_violations_snapshot_bounded(self):
        reg = make_registry()
        reg.register_batch_schema(source="p", batch_id=1, feature_means=BASE_FEATURES)
        reg.register_batch_schema(source="p", batch_id=2, feature_means={"amount": 1.0})
        snapshot = reg.get_violations_snapshot(limit=1)
        assert len(snapshot) == 1
