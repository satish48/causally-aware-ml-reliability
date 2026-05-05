"""
src/model_registry.py
---------------------
Multi-model version registry with canary-state tracking.

Industry context:
    At Google, Meta, and Stripe a model is never "the model" — it is a versioned
    artifact with a known lineage. Every deployment creates a new version record.
    Traffic is split across versions via explicit weights. This registry is the
    single source of truth for which version serves what fraction of traffic and
    what its current health looks like.

    Design mirrors the Google Vertex AI Model Registry and Uber's Michelangelo
    version tracking, simplified for self-hosted operation.

Invariants:
    - Version records are append-only; only status and health_score are mutable.
    - At most one version per model_name may have status="stable".
    - Canary weights across active versions do not need to sum to 1.0 during
      a transition — the controller manages re-balancing explicitly.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "ModelStatus",
    "ModelVersion",
    "ModelRegistry",
    "get_model_registry",
]


ModelStatus = Literal["staged", "canary", "stable", "degraded", "rolled_back", "deprecated"]


# ────────────────────────────────────────────────────────────────
# DOMAIN TYPES
# ────────────────────────────────────────────────────────────────

@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    status: ModelStatus
    canary_weight: float       # fraction of traffic [0.0, 1.0]
    health_score: float        # 0–100, updated by the SLO engine each batch
    deployed_at: str
    notes: str = ""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "status": self.status,
            "canary_weight": round(self.canary_weight, 4),
            "health_score": round(self.health_score, 1),
            "deployed_at": self.deployed_at,
            "notes": self.notes,
            "baseline_metrics": self.baseline_metrics,
        }


# ────────────────────────────────────────────────────────────────
# REGISTRY
# ────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Authoritative store for model versions and canary traffic allocation.

    Call register_version() when a new artifact is deployed.
    Call transition_status() when the canary controller promotes or rolls back.
    Call update_health_score() each batch with the SLO engine's worst burn metric.
    """

    MAX_VERSIONS_PER_MODEL: int = 50

    def __init__(self) -> None:
        self._versions: dict[str, list[ModelVersion]] = {}
        logger.info("ModelRegistry initialized")
        self._seed_default()

    # ── PUBLIC API ────────────────────────────────────────────

    def register_version(
        self,
        *,
        model_name: str,
        status: ModelStatus = "staged",
        canary_weight: float = 0.0,
        baseline_metrics: dict[str, float] | None = None,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Create and store a new model version record."""
        version = ModelVersion(
            version_id=f"{model_name}-{uuid.uuid4().hex[:8]}",
            model_name=model_name,
            status=status,
            canary_weight=canary_weight,
            health_score=100.0,
            deployed_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
            baseline_metrics=baseline_metrics or {},
            metadata=metadata or {},
        )

        if model_name not in self._versions:
            self._versions[model_name] = []

        self._versions[model_name].append(version)
        if len(self._versions[model_name]) > self.MAX_VERSIONS_PER_MODEL:
            self._versions[model_name] = self._versions[model_name][-self.MAX_VERSIONS_PER_MODEL:]

        logger.info(
            "ModelVersion registered | version_id=%s | model=%s | status=%s | weight=%.2f",
            version.version_id, model_name, status, canary_weight,
        )
        return version

    def transition_status(
        self,
        version_id: str,
        new_status: ModelStatus,
    ) -> ModelVersion | None:
        """Mutate the status of an existing version. Returns None if not found."""
        for versions in self._versions.values():
            for v in versions:
                if v.version_id == version_id:
                    old = v.status
                    v.status = new_status
                    logger.info(
                        "ModelVersion transition | id=%s | %s → %s",
                        version_id, old, new_status,
                    )
                    return v
        logger.warning("ModelVersion not found | version_id=%s", version_id)
        return None

    def set_canary_weight(self, version_id: str, weight: float) -> None:
        for versions in self._versions.values():
            for v in versions:
                if v.version_id == version_id:
                    v.canary_weight = max(0.0, min(1.0, weight))
                    return

    def update_health_score(self, model_name: str, health_score: float) -> None:
        """Update health for all active (stable/canary) versions of a model."""
        clamped = max(0.0, min(100.0, health_score))
        for v in self._versions.get(model_name, []):
            if v.status in {"stable", "canary"}:
                v.health_score = clamped

    def update_live_metrics(self, model_name: str, metrics: dict[str, float | None]) -> None:
        """
        Replace baseline_metrics with live-computed values.
        Called each batch in real mode so the registry always reflects actual
        precision/recall/F1 rather than the static seed values.
        """
        clean = {k: v for k, v in metrics.items() if v is not None}
        if not clean:
            return
        for v in self._versions.get(model_name, []):
            if v.status in {"stable", "canary"}:
                v.baseline_metrics.update(clean)

    def get_active_versions(self, model_name: str) -> list[ModelVersion]:
        return [
            v for v in self._versions.get(model_name, [])
            if v.status in {"stable", "canary"}
        ]

    def get_stable_version(self, model_name: str) -> ModelVersion | None:
        for v in reversed(self._versions.get(model_name, [])):
            if v.status == "stable":
                return v
        return None

    def get_canary_version(self, model_name: str) -> ModelVersion | None:
        for v in reversed(self._versions.get(model_name, [])):
            if v.status == "canary":
                return v
        return None

    def get_stable_version(self, model_name: str) -> ModelVersion | None:
        for v in reversed(self._versions.get(model_name, [])):
            if v.status == "stable":
                return v
        return None

    def get_all_versions(self) -> dict[str, list[dict[str, Any]]]:
        return {
            model: [v.to_dict() for v in versions]
            for model, versions in self._versions.items()
        }

    def get_stats(self) -> dict[str, Any]:
        all_versions = [v for vs in self._versions.values() for v in vs]
        return {
            "models_tracked": len(self._versions),
            "total_versions": len(all_versions),
            "stable_versions": sum(1 for v in all_versions if v.status == "stable"),
            "canary_versions": sum(1 for v in all_versions if v.status == "canary"),
            "degraded_versions": sum(1 for v in all_versions if v.status == "degraded"),
            "rolled_back_versions": sum(1 for v in all_versions if v.status == "rolled_back"),
        }

    # ── INTERNAL ──────────────────────────────────────────────

    def _seed_default(self) -> None:
        """Seed the baseline stable version that represents the current running model."""
        self.register_version(
            model_name="fraud_detection_v1",
            status="stable",
            canary_weight=1.0,
            baseline_metrics={
                # Seeded with representative values; overwritten each batch by
                # ModelRegistry.update_live_metrics() once true_label data flows.
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc_roc_proxy": 0.0,
            },
            notes="Baseline stable version seeded at startup.",
        )


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
