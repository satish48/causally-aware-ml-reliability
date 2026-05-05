"""
src/drift_detector.py
----------------------
Real-time drift detection engine for the ML Observability Platform.

Responsibilities:
    - Load baseline feature distributions from validated settings
    - Consume batch summaries from the event broker
    - Compute feature-level z-score + rolling PSI drift signals
    - Produce bounded drift results and alerts
    - Expose detector state for API, dashboard, and downstream engines

Design notes:
    - Drift detection and alerting are related but not identical
    - Result history and alert history are bounded in memory
    - Warmup and cooldown reduce noisy or premature alerts
    - Detector remains technical: business impact and decisions belong elsewhere
    - Aggregate-level PSI and business KPI breaches can trigger drift even if
      no single feature crosses both z-score and PSI thresholds
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.stats import norm

from config.settings import get_settings
from src.stream_producer import get_event_broker

logger = logging.getLogger(__name__)

__all__ = [
    "DriftDetector",
    "DriftResult",
    "Alert",
    "FeatureDriftScore",
    "compute_z_score",
    "compute_rolling_psi",
    "severity_from_psi",
    "get_detector",
]


# ────────────────────────────────────────────────────────────────
# DATA MODELS
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FeatureDriftScore:
    feature: str
    z_score: float
    psi: float
    drifted: bool
    severity: str


@dataclass(slots=True)
class DriftResult:
    batch_id: int
    timestamp: str
    overall_psi: float
    max_z_score: float
    drift_detected: bool
    severity: str
    feature_scores: list[FeatureDriftScore]
    fraud_rate: float
    avg_confidence: float
    processing_ms: float
    warmed_up: bool


@dataclass(slots=True)
class Alert:
    alert_id: int
    timestamp: str
    batch_id: int
    severity: str
    message: str
    psi: float
    fraud_rate: float
    features: list[str]


# ────────────────────────────────────────────────────────────────
# STATISTICAL HELPERS
# ────────────────────────────────────────────────────────────────

def compute_z_score(
    current_mean: float,
    baseline_mean: float,
    baseline_std: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute absolute z-score for current batch mean vs baseline mean.
    """
    return abs(current_mean - baseline_mean) / (baseline_std + epsilon)


def compute_rolling_psi(
    baseline_mean: float,
    baseline_std: float,
    window_means: list[float],
    bins: int = 10,
    epsilon: float = 1e-6,
    baseline_means: list[float] | None = None,
) -> float:
    """
    Compute PSI over a rolling window of observed batch means.

    Two-sample mode (preferred, used when baseline_means is provided):
        Uses symmetric KL divergence between two Gaussians fit to the
        baseline window and current window batch means. This is robust
        for small N (10-30 batch means) because it uses moment estimates
        rather than sparse histograms.

        Symmetric KL divergence is zero when distributions are identical
        and grows with both mean shift and variance change. It is scaled
        via tanh to [0, 1] so the existing PSI thresholds continue to
        work without changes. A 1-SEM mean shift ≈ PSI 0.46.

    Legacy single-sample mode (fallback when baseline_means is None):
        Normalises using baseline_mean/std and compares against the
        theoretical normal. Kept for backwards compatibility.

    Returns:
        A float in [0.0, 1.0] for dashboard-friendly interpretation.
    """
    if len(window_means) < 5:
        return 0.0

    # ── TWO-SAMPLE PSI via Cohen's d ────────────────────────────
    # Histogram PSI requires N≫bins (breaks at 10-30 batch means, 10 bins).
    # Sym-KL breaks because variance estimates from 5 warmup samples have
    # chi-squared(4) noise — 90% CI spans 5× the true value.
    #
    # Correct approach: express the mean shift as a fraction of the
    # INDIVIDUAL observation std (baseline_std, known precisely from
    # training data). This is Cohen's d: a pure effect-size metric that
    # is invariant to sample size and does not require variance estimation
    # from small warmup windows.
    #
    # d = |mu_current - mu_baseline| / sigma_individual
    # PSI = tanh(d / 1.5)   →   d=0→0, d=0.4→0.26, d=2.0→0.83, d≥4→≈1.0
    if baseline_means and len(baseline_means) >= 2:
        mu_b = float(np.mean(baseline_means))
        mu_c = float(np.mean(window_means))
        sigma = max(baseline_std, epsilon)   # individual observation std from config

        d = abs(mu_c - mu_b) / sigma

        # tanh(d/1.5): d=0→0, d=0.4→0.26, d=1→0.58, d=2→0.83, d≥4→≈1.0
        return float(math.tanh(d / 1.5))

    # ── LEGACY SINGLE-SAMPLE PSI (synthetic mode fallback) ──────
    normalized = [
        (value - baseline_mean) / (baseline_std + epsilon)
        for value in window_means
    ]
    bin_edges = np.linspace(-3.0, 3.0, bins + 1)
    expected_pct = np.diff(norm.cdf(bin_edges))
    expected_pct = np.clip(expected_pct, epsilon, None)
    expected_pct /= expected_pct.sum()
    actual_counts, _ = np.histogram(normalized, bins=bin_edges)
    if actual_counts.sum() == 0:
        return 0.0
    actual_pct = actual_counts / actual_counts.sum()
    actual_pct = np.clip(actual_pct, epsilon, None)
    actual_pct /= actual_pct.sum()
    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return min(abs(psi), 1.0)


def severity_from_psi(psi: float) -> str:
    """
    Map PSI to a qualitative severity label.
    """
    if psi >= 0.50:
        return "critical"
    if psi >= 0.25:
        return "high"
    if psi >= 0.10:
        return "moderate"
    return "stable"


# ────────────────────────────────────────────────────────────────
# BASELINE LOADING
# ────────────────────────────────────────────────────────────────

def _load_baseline() -> dict[str, dict[str, float]]:
    """
    Load baseline distributions from validated settings.

    synthetic -> from config.settings baseline config
    real      -> from unified simulator baseline stats
    """
    cfg = get_settings()
    data_mode = cfg.model.data_mode

    if data_mode == "real":
        logger.info("DriftDetector loading REAL baseline from unified simulator")
        from src.model_simulator import FraudModelSimulator

        simulator = FraudModelSimulator()
        baseline = simulator.get_baseline_stats()
        logger.info("Real baseline loaded | features=%s", list(baseline.keys()))
        return baseline

    baseline = cfg.baseline.as_dict()
    logger.info("Synthetic baseline loaded | features=%s", list(baseline.keys()))
    return baseline


# ────────────────────────────────────────────────────────────────
# DETECTOR
# ────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Consume batch summaries and perform feature-level drift analysis.

    Detection conditions (after warmup):
        - any feature where z-score >= threshold and PSI >= threshold
        - OR strong aggregate PSI shift
        - OR fraud rate exceeds monitoring alert threshold

    Alert conditions:
        - drift detected
        - OR high/critical severity
        - AND alert cooldown has expired
    """

    SUBSCRIBER_NAME = "drift_detector"

    def __init__(self) -> None:
        cfg = get_settings()

        self._baseline_dist = _load_baseline()
        self._alert_fraud_rate = cfg.monitoring.alert_fraud_rate

        # Tunable detection policy.
        self._warmup_batches = cfg.detector.warmup_batches
        self._z_score_threshold = cfg.detector.z_score_threshold
        self._psi_threshold = cfg.detector.psi_threshold
        self._aggregate_psi_threshold = max(
            cfg.detector.psi_threshold,
            cfg.monitoring.drift_threshold,
        )
        self._window_size = cfg.detector.window_size
        self._alert_cooldown_batches = cfg.detector.alert_cooldown_batches

        self._max_results = cfg.detector.max_results
        self._max_alerts = cfg.detector.max_alerts

        self._window: dict[str, deque[float]] = {
            feature: deque(maxlen=self._window_size)
            for feature in self._baseline_dist
        }

        # Warmup accumulator: collects batch means during the warmup period.
        # After warmup, _warmup_means is used to compute an empirical std of
        # BATCH MEANS — which is std_individual / sqrt(N) by CLT. PSI must
        # be normalized against this, not individual observation std, or
        # every feature shows PSI≈1.0 because batch means cluster centrally
        # against a theoretical normal with full individual-observation spread.
        self._warmup_means: dict[str, list[float]] = {
            feature: [] for feature in self._baseline_dist
        }
        self._psi_baseline: dict[str, dict[str, float]] = {}   # set after warmup

        self.results: list[DriftResult] = []
        self.alerts: list[Alert] = []

        self._alert_counter = 0
        self._batches_seen = 0
        self._last_alert_batch = -(self._alert_cooldown_batches + 1)

        self._running = False
        self._queue: asyncio.Queue[dict[str, Any]] | None = None

        logger.info(
            "DriftDetector initialized | features=%d | z_threshold=%.2f | "
            "psi_threshold=%.2f | aggregate_psi_threshold=%.2f | warmup=%d | "
            "window=%d | cooldown=%d",
            len(self._baseline_dist),
            self._z_score_threshold,
            self._psi_threshold,
            self._aggregate_psi_threshold,
            self._warmup_batches,
            self._window_size,
            self._alert_cooldown_batches,
        )

    # ── PUBLIC API ─────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            logger.warning("DriftDetector.start() called while already running")
            return

        self._queue = await get_event_broker().subscribe(self.SUBSCRIBER_NAME)
        self._running = True

        logger.info(
            "DriftDetector started | subscriber=%s | warmup=%d | z_threshold=%.2f | "
            "psi_threshold=%.2f | aggregate_psi_threshold=%.2f",
            self.SUBSCRIBER_NAME,
            self._warmup_batches,
            self._z_score_threshold,
            self._psi_threshold,
            self._aggregate_psi_threshold,
        )

        try:
            while self._running:
                await self._consume_one_batch()
        except asyncio.CancelledError:
            logger.warning("DriftDetector task cancelled")
            self._running = False
            raise
        finally:
            await get_event_broker().unsubscribe(self.SUBSCRIBER_NAME)
            logger.info(
                "DriftDetector stopped | batches_analysed=%d | alerts_raised=%d",
                len(self.results),
                len(self.alerts),
            )

    def stop(self) -> None:
        self._running = False
        logger.info("DriftDetector stop signal received")

    def get_latest_result(self) -> DriftResult | None:
        return self.results[-1] if self.results else None

    def get_result_objects(self, limit: int | None = None) -> list[DriftResult]:
        items = self.results if limit is None else self.results[-limit:]
        return list(items)

    def get_results_snapshot(self) -> list[dict[str, Any]]:
        return [self._result_to_dict(result) for result in self.results[-50:]]

    def get_alerts_snapshot(self) -> list[dict[str, Any]]:
        return [self._alert_to_dict(alert) for alert in self.alerts[-20:]]

    def get_stats(self) -> dict[str, Any]:
        latest = self.get_latest_result()

        psi_scores: dict[str, float] = {}
        z_scores: dict[str, float] = {}
        top_drifted: list[str] = []
        if latest and latest.feature_scores:
            for fs in latest.feature_scores:
                psi_scores[fs.feature] = round(fs.psi, 4)
                z_scores[fs.feature] = round(fs.z_score, 3)
            top_drifted = [
                fs.feature for fs in sorted(
                    latest.feature_scores, key=lambda x: x.psi, reverse=True
                ) if fs.drifted
            ][:5]

        return {
            "running": self._running,
            "subscriber_name": self.SUBSCRIBER_NAME,
            "queue_attached": self._queue is not None,
            "feature_count": len(self._baseline_dist),
            "batches_analysed": len(self.results),
            "alerts_raised": len(self.alerts),
            "warmed_up": self._batches_seen > self._warmup_batches,
            "last_alert_batch": self._last_alert_batch if self.alerts else None,
            "latest_psi": latest.overall_psi if latest else 0.0,
            "max_z_score": latest.max_z_score if latest else 0.0,
            "latest_severity": latest.severity if latest else "stable",
            "drift_detected": latest.drift_detected if latest else False,
            "psi_scores": psi_scores,
            "z_scores": z_scores,
            "top_drifted_features": top_drifted,
        }

    # ── INTERNAL LOOP ──────────────────────────────────────────

    async def _consume_one_batch(self) -> None:
        if self._queue is None:
            logger.error("DriftDetector queue is not initialized")
            await asyncio.sleep(0.1)
            return

        try:
            batch_summary = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return

        started = time.perf_counter()

        try:
            result = self._analyse(batch_summary)
            result.processing_ms = round((time.perf_counter() - started) * 1000.0, 2)

            self.results.append(result)
            if len(self.results) > self._max_results:
                self.results = self.results[-self._max_results:]

            should_alert = (
                result.drift_detected
                or result.severity in {"high", "critical", "severe"}
            )

            alert_emitted = False
            if should_alert:
                alert_emitted = self._raise_alert(result)

            if should_alert:
                logger.warning(
                    "DRIFT DETECTED | batch_id=%03d | max_z=%.2f | psi=%.4f | "
                    "severity=%s | fraud=%.1f%% | ms=%.1f | alert_emitted=%s",
                    result.batch_id,
                    result.max_z_score,
                    result.overall_psi,
                    result.severity,
                    result.fraud_rate * 100,
                    result.processing_ms,
                    alert_emitted,
                )
            else:
                status = "WARMING UP" if not result.warmed_up else "healthy"
                logger.info(
                    "Batch OK | batch_id=%03d | max_z=%.2f | psi=%.4f | "
                    "status=%s | fraud=%.1f%% | ms=%.1f",
                    result.batch_id,
                    result.max_z_score,
                    result.overall_psi,
                    status,
                    result.fraud_rate * 100,
                    result.processing_ms,
                )

        except Exception:
            logger.exception(
                "DriftDetector failed on batch_id=%s",
                batch_summary.get("batch_id"),
            )

    def _analyse(self, batch_summary: dict[str, Any]) -> DriftResult:
        feature_means = batch_summary.get("feature_means", {})
        batch_id = int(batch_summary["batch_id"])
        fraud_rate = float(batch_summary["fraud_rate"])
        avg_confidence = float(batch_summary["avg_confidence"])

        self._batches_seen += 1
        warmed_up = self._batches_seen > self._warmup_batches

        for feature, value in feature_means.items():
            if feature in self._window:
                fv = float(value)
                self._window[feature].append(fv)
                # Accumulate warmup batch means for empirical PSI baseline
                if not warmed_up and feature in self._warmup_means:
                    self._warmup_means[feature].append(fv)

        # After warmup: finalise PSI baseline from empirical batch mean distribution.
        # This is critical — normalising with individual-observation std makes every
        # feature show PSI≈1.0 because batch means (std/√N) cluster in the central
        # bins of the theoretical normal, with empty tails driving PSI to 1.
        if warmed_up and not self._psi_baseline:
            for feature, means in self._warmup_means.items():
                if len(means) >= 2:
                    import statistics
                    em = statistics.mean(means)
                    es = statistics.stdev(means)
                    self._psi_baseline[feature] = {
                        "mean": em,
                        "std": max(es, 1e-4),   # floor prevents division by zero
                    }
                else:
                    # Fallback: approximate SEM from individual observation std
                    dist = self._baseline_dist.get(feature, {})
                    self._psi_baseline[feature] = {
                        "mean": dist.get("mean", 0.0),
                        "std": max(dist.get("std", 1.0) / 7.0, 1e-4),
                    }
            logger.info(
                "PSI baseline finalised from warmup batch means | features=%d",
                len(self._psi_baseline),
            )

        feature_scores: list[FeatureDriftScore] = []
        max_z = 0.0

        for feature, dist in self._baseline_dist.items():
            current_mean = feature_means.get(feature)
            if current_mean is None:
                continue

            z_score = compute_z_score(
                current_mean=float(current_mean),
                baseline_mean=float(dist["mean"]),
                baseline_std=float(dist["std"]),
            )
            max_z = max(max_z, z_score)

            # Two-sample mode: compare current window means against warmup batch
            # means, normalised by individual observation std (known precisely from
            # training config). Using individual std (not estimated batch-mean std)
            # avoids huge variance from estimating sigma from only 5 warmup samples.
            psi_dist = self._psi_baseline.get(feature, dist)
            psi = compute_rolling_psi(
                baseline_mean=float(psi_dist["mean"]),
                baseline_std=float(dist["std"]),   # individual obs std — precise, not estimated
                window_means=list(self._window[feature]),
                baseline_means=self._warmup_means.get(feature) or None,
            )

            feature_level_drift = warmed_up and (
                z_score >= self._z_score_threshold
                and psi >= self._psi_threshold
            )

            feature_scores.append(
                FeatureDriftScore(
                    feature=feature,
                    z_score=round(z_score, 3),
                    psi=round(psi, 4),
                    drifted=feature_level_drift,
                    severity=severity_from_psi(psi),
                )
            )

        overall_psi = round(
            sum(score.psi for score in feature_scores) / max(len(feature_scores), 1),
            4,
        )
        max_z = round(max_z, 3)

        aggregate_drift = warmed_up and (overall_psi >= self._aggregate_psi_threshold)
        business_kpi_breach = warmed_up and (fraud_rate > self._alert_fraud_rate)
        feature_drift_detected = any(score.drifted for score in feature_scores)

        drift_detected = warmed_up and (
            feature_drift_detected
            or aggregate_drift
            or business_kpi_breach
        )

        if not warmed_up:
            overall_severity = "stable"
        elif overall_psi >= 0.50:
            overall_severity = "critical"
        elif overall_psi >= 0.25:
            overall_severity = "high"
        elif overall_psi >= 0.10:
            overall_severity = "moderate"
        else:
            overall_severity = "stable"

        return DriftResult(
            batch_id=batch_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_psi=overall_psi,
            max_z_score=max_z,
            drift_detected=drift_detected,
            severity=overall_severity,
            feature_scores=feature_scores,
            fraud_rate=fraud_rate,
            avg_confidence=avg_confidence,
            processing_ms=0.0,
            warmed_up=warmed_up,
        )

    def _raise_alert(self, result: DriftResult) -> bool:
        batches_since_last = result.batch_id - self._last_alert_batch
        if batches_since_last < self._alert_cooldown_batches:
            logger.debug(
                "Alert suppressed due to cooldown | batch_id=%d | "
                "batches_since_last=%d | cooldown=%d",
                result.batch_id,
                batches_since_last,
                self._alert_cooldown_batches,
            )
            return False

        self._last_alert_batch = result.batch_id
        self._alert_counter += 1

        # Rank shifted features by PSI descending — top 5 only.
        # Listing all 10 features in every alert is noise; operators care about
        # which features shifted *most*, not a flat membership list.
        shifted = sorted(
            [s for s in result.feature_scores if s.drifted or s.psi >= self._psi_threshold],
            key=lambda s: s.psi,
            reverse=True,
        )
        unique_features: list[str] = [
            f"{s.feature} (PSI={s.psi:.2f}, Z={s.z_score:.2f})"
            for s in shifted[:5]
        ]
        if result.overall_psi >= self._aggregate_psi_threshold and "aggregate_psi" not in unique_features:
            unique_features.append(f"aggregate_psi (PSI={result.overall_psi:.2f})")
        if result.fraud_rate > self._alert_fraud_rate:
            unique_features.append(f"fraud_rate ({result.fraud_rate:.1%})")

        alert = Alert(
            alert_id=self._alert_counter,
            timestamp=result.timestamp,
            batch_id=result.batch_id,
            severity=result.severity,
            message=(
                f"Drift detected at batch {result.batch_id} — "
                f"max_z={result.max_z_score:.2f} | "
                f"psi={result.overall_psi:.4f} | "
                f"severity={result.severity} | "
                f"fraud={result.fraud_rate:.1%}"
            ),
            psi=result.overall_psi,
            fraud_rate=result.fraud_rate,
            features=unique_features,
        )

        self.alerts.append(alert)
        if len(self.alerts) > self._max_alerts:
            self.alerts = self.alerts[-self._max_alerts:]

        logger.info(
            "Alert raised | alert_id=%d | batch_id=%d | severity=%s | features=%s",
            alert.alert_id,
            alert.batch_id,
            alert.severity,
            alert.features,
        )
        return True

    # ── SERIALIZATION HELPERS ──────────────────────────────────

    @staticmethod
    def _result_to_dict(result: DriftResult) -> dict[str, Any]:
        return {
            "batch_id": result.batch_id,
            "timestamp": result.timestamp,
            "overall_psi": result.overall_psi,
            "max_z_score": result.max_z_score,
            "drift_detected": result.drift_detected,
            "severity": result.severity,
            "fraud_rate": result.fraud_rate,
            "avg_confidence": result.avg_confidence,
            "processing_ms": result.processing_ms,
            "warmed_up": result.warmed_up,
            "feature_scores": [asdict(score) for score in result.feature_scores],
        }

    @staticmethod
    def _alert_to_dict(alert: Alert) -> dict[str, Any]:
        return asdict(alert)


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_detector: DriftDetector | None = None


def get_detector() -> DriftDetector:
    global _detector
    if _detector is None:
        _detector = DriftDetector()
        logger.info("DriftDetector singleton created")
    return _detector


# ────────────────────────────────────────────────────────────────
# ENTRYPOINT TEST
# ────────────────────────────────────────────────────────────────

async def _test() -> None:
    import json
    from src.stream_producer import StreamProducer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = get_settings()
    duration = 16 * cfg.model.interval_seconds

    producer = StreamProducer()
    detector = get_detector()

    producer_task = asyncio.create_task(producer.start())
    detector_task = asyncio.create_task(detector.start())

    await asyncio.sleep(duration)

    producer.stop()
    detector.stop()

    await asyncio.gather(
        producer_task,
        detector_task,
        return_exceptions=True,
    )

    print("\n── Detector stats ──")
    print(json.dumps(detector.get_stats(), indent=2))

    print("\n── Latest result ──")
    latest = detector.get_latest_result()
    if latest:
        print(json.dumps(DriftDetector._result_to_dict(latest), indent=2))


if __name__ == "__main__":
    asyncio.run(_test())
