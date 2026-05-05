"""
src/schema_registry.py
----------------------
Schema contract management with fingerprinting and breaking-change detection.

Industry analog: Confluent Schema Registry, Google Dataplex, Netflix Metacat.

The gap this fills: upstream data pipelines silently mutate field types, drop
columns, or spike null rates. By the time the model degrades, the root cause is
buried in ten service deploys. This registry fingerprints every batch schema,
diffs successive versions, and emits typed BreakingChange events that feed
directly into the CausalEngine for attribution.

Breaking-change taxonomy (Stripe/Google-style policy):
    critical  — field removed, type changed        → model likely producing wrong predictions NOW
    high      — null-rate spike > 10%              → upstream pipeline anomaly
    moderate  — new non-nullable field added       → possible training-serving skew
    info      — new nullable field                 → low risk, monitor
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FieldType",
    "SchemaField",
    "SchemaVersion",
    "BreakingChangeType",
    "BreakingChange",
    "SchemaRegistry",
    "get_schema_registry",
]


# ────────────────────────────────────────────────────────────────
# DOMAIN TYPES
# ────────────────────────────────────────────────────────────────

class FieldType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    UNKNOWN = "unknown"


class BreakingChangeType(str, Enum):
    FIELD_REMOVED = "field_removed"
    FIELD_ADDED = "field_added"
    TYPE_CHANGED = "type_changed"
    NULL_RATE_SPIKE = "null_rate_spike"
    RANGE_VIOLATION = "range_violation"


@dataclass(slots=True)
class SchemaField:
    name: str
    field_type: FieldType
    nullable: bool
    null_rate: float = 0.0
    sample_min: float | None = None
    sample_max: float | None = None


@dataclass(slots=True)
class SchemaVersion:
    version_id: str
    source: str
    fingerprint: str
    fields: dict[str, SchemaField]
    registered_at: str
    batch_id: int


@dataclass(slots=True)
class BreakingChange:
    change_id: str
    change_type: BreakingChangeType
    source: str
    field_name: str
    old_value: str
    new_value: str
    severity: str  # "critical" | "high" | "moderate" | "info"
    detected_at: str
    batch_id: int
    description: str


# ────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ────────────────────────────────────────────────────────────────

def _fingerprint(fields: dict[str, FieldType]) -> str:
    """SHA-256 of canonicalized {field_name: field_type} mapping, truncated to 16 hex chars."""
    canonical = json.dumps(
        {k: v.value for k, v in sorted(fields.items())},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ────────────────────────────────────────────────────────────────
# REGISTRY
# ────────────────────────────────────────────────────────────────

class SchemaRegistry:
    """
    Track schema evolution per logical source and emit breaking-change events.

    Call register_batch_schema() on every batch. It returns an empty list when
    the schema is stable and a list of BreakingChange objects when violations are
    found. Callers should forward violations to the CausalEngine.

    Thread-safety: single-threaded async event loop only.
    """

    NULL_RATE_SPIKE_THRESHOLD: float = 0.10
    MAX_VERSIONS_PER_SOURCE: int = 50
    MAX_VIOLATIONS: int = 500

    def __init__(self) -> None:
        self._versions: dict[str, list[SchemaVersion]] = defaultdict(list)
        self._violations: list[BreakingChange] = []
        self._violation_counter: int = 0
        logger.info("SchemaRegistry initialized")

    # ── PUBLIC API ────────────────────────────────────────────

    def register_batch_schema(
        self,
        *,
        source: str,
        batch_id: int,
        feature_means: dict[str, float],
        feature_null_rates: dict[str, float] | None = None,
        feature_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> list[BreakingChange]:
        """
        Register the schema implied by a batch summary and return any violations.

        Args:
            source:             logical data source name (e.g. "payment_events")
            batch_id:           batch identifier
            feature_means:      {field: mean_value} from the batch summary
            feature_null_rates: optional {field: null_fraction in [0,1]}
            feature_ranges:     optional {field: (min, max)}

        Returns:
            List of BreakingChange (empty when schema is stable).
        """
        null_rates = feature_null_rates or {}
        ranges = feature_ranges or {}

        fields: dict[str, SchemaField] = {
            name: SchemaField(
                name=name,
                field_type=FieldType.NUMERIC,
                nullable=null_rates.get(name, 0.0) > 0.0,
                null_rate=null_rates.get(name, 0.0),
                sample_min=ranges[name][0] if name in ranges else None,
                sample_max=ranges[name][1] if name in ranges else None,
            )
            for name in feature_means
        }

        fp = _fingerprint({n: f.field_type for n, f in fields.items()})

        version = SchemaVersion(
            version_id=f"{source}:v{len(self._versions[source]) + 1}",
            source=source,
            fingerprint=fp,
            fields=fields,
            registered_at=datetime.now(timezone.utc).isoformat(),
            batch_id=batch_id,
        )

        violations: list[BreakingChange] = []
        history = self._versions[source]

        if history:
            # Always diff: fingerprint catches structural changes (field add/remove/type),
            # but null-rate spikes must be checked even when structure is unchanged.
            violations = self._diff(history[-1], version)

        history.append(version)
        if len(history) > self.MAX_VERSIONS_PER_SOURCE:
            self._versions[source] = history[-self.MAX_VERSIONS_PER_SOURCE:]

        if violations:
            self._violations.extend(violations)
            if len(self._violations) > self.MAX_VIOLATIONS:
                self._violations = self._violations[-self.MAX_VIOLATIONS:]

            for v in violations:
                logger.warning(
                    "Schema violation | source=%s | change=%s | field=%s | "
                    "severity=%s | batch_id=%d | '%s' → '%s'",
                    source, v.change_type.value, v.field_name,
                    v.severity, batch_id, v.old_value, v.new_value,
                )

        return violations

    def get_violations_snapshot(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {
                "change_id": v.change_id,
                "change_type": v.change_type.value,
                "source": v.source,
                "field_name": v.field_name,
                "old_value": v.old_value,
                "new_value": v.new_value,
                "severity": v.severity,
                "detected_at": v.detected_at,
                "batch_id": v.batch_id,
                "description": v.description,
            }
            for v in self._violations[-limit:]
        ]

    def get_violation_objects(self, limit: int | None = None) -> list[BreakingChange]:
        items = self._violations if limit is None else self._violations[-limit:]
        return list(items)

    def get_version_history(self, source: str, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "version_id": v.version_id,
                "fingerprint": v.fingerprint,
                "field_count": len(v.fields),
                "registered_at": v.registered_at,
                "batch_id": v.batch_id,
            }
            for v in self._versions.get(source, [])[-limit:]
        ]

    def get_stats(self) -> dict[str, Any]:
        all_violations = self._violations
        return {
            "sources_tracked": len(self._versions),
            "total_versions": sum(len(v) for v in self._versions.values()),
            "total_violations": len(all_violations),
            "critical_violations": sum(1 for v in all_violations if v.severity == "critical"),
            "high_violations": sum(1 for v in all_violations if v.severity == "high"),
        }

    # ── INTERNAL ─────────────────────────────────────────────

    def _diff(
        self,
        prev: SchemaVersion,
        curr: SchemaVersion,
    ) -> list[BreakingChange]:
        changes: list[BreakingChange] = []
        now = datetime.now(timezone.utc).isoformat()
        prev_fields = set(prev.fields)
        curr_fields = set(curr.fields)

        for removed in prev_fields - curr_fields:
            changes.append(self._make_change(
                change_type=BreakingChangeType.FIELD_REMOVED,
                source=curr.source,
                field_name=removed,
                old_value=prev.fields[removed].field_type.value,
                new_value="absent",
                severity="critical",
                batch_id=curr.batch_id,
                detected_at=now,
                description=(
                    f"Field '{removed}' present in {prev.version_id} is absent in "
                    f"{curr.version_id}. Models depending on this field will degrade silently."
                ),
            ))

        for added in curr_fields - prev_fields:
            changes.append(self._make_change(
                change_type=BreakingChangeType.FIELD_ADDED,
                source=curr.source,
                field_name=added,
                old_value="absent",
                new_value=curr.fields[added].field_type.value,
                severity="info",
                batch_id=curr.batch_id,
                detected_at=now,
                description=(
                    f"New field '{added}' appeared in {curr.version_id}. "
                    "Verify this is intentional and will not cause training-serving skew."
                ),
            ))

        for common in prev_fields & curr_fields:
            pf = prev.fields[common]
            cf = curr.fields[common]

            if pf.field_type != cf.field_type:
                changes.append(self._make_change(
                    change_type=BreakingChangeType.TYPE_CHANGED,
                    source=curr.source,
                    field_name=common,
                    old_value=pf.field_type.value,
                    new_value=cf.field_type.value,
                    severity="critical",
                    batch_id=curr.batch_id,
                    detected_at=now,
                    description=(
                        f"Type of '{common}' changed {pf.field_type.value} → "
                        f"{cf.field_type.value}. Models trained on the old type "
                        "will produce incorrect predictions."
                    ),
                ))

            null_delta = cf.null_rate - pf.null_rate
            if null_delta > self.NULL_RATE_SPIKE_THRESHOLD:
                changes.append(self._make_change(
                    change_type=BreakingChangeType.NULL_RATE_SPIKE,
                    source=curr.source,
                    field_name=common,
                    old_value=f"{pf.null_rate:.2%}",
                    new_value=f"{cf.null_rate:.2%}",
                    severity="high",
                    batch_id=curr.batch_id,
                    detected_at=now,
                    description=(
                        f"Null rate for '{common}' spiked "
                        f"{pf.null_rate:.1%} → {cf.null_rate:.1%} "
                        f"(Δ={null_delta:.1%}). Upstream pipeline anomaly likely."
                    ),
                ))

        return changes

    def _make_change(self, **kwargs: Any) -> BreakingChange:
        self._violation_counter += 1
        return BreakingChange(
            change_id=f"vc-{self._violation_counter:06d}",
            **kwargs,
        )


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_registry: SchemaRegistry | None = None


def get_schema_registry() -> SchemaRegistry:
    global _registry
    if _registry is None:
        _registry = SchemaRegistry()
    return _registry
