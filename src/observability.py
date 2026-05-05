from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any

LabelKey = tuple[tuple[str, str], ...]


def _label_key(labels: dict[str, str] | None) -> LabelKey:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _escape_label(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _format_labels(labels: LabelKey) -> str:
    if not labels:
        return ""
    body = ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels)
    return f"{{{body}}}"


class MetricsRegistry:
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, dict[LabelKey, float]] = defaultdict(dict)
        self._gauges: dict[str, dict[LabelKey, float]] = defaultdict(dict)
        self._histograms: dict[str, dict[LabelKey, dict[str, Any]]] = defaultdict(dict)
        self._meta: dict[str, tuple[str, str]] = {}

    def inc_counter(
        self,
        name: str,
        description: str,
        *,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = _label_key(labels)
        with self._lock:
            self._meta[name] = ("counter", description)
            current = self._counters[name].get(key, 0.0)
            self._counters[name][key] = current + value

    def set_gauge(
        self,
        name: str,
        description: str,
        *,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = _label_key(labels)
        with self._lock:
            self._meta[name] = ("gauge", description)
            self._gauges[name][key] = value

    def observe_histogram(
        self,
        name: str,
        description: str,
        *,
        value: float,
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        key = _label_key(labels)
        bucket_values = buckets or self.DEFAULT_BUCKETS
        with self._lock:
            self._meta[name] = ("histogram", description)
            series = self._histograms[name].setdefault(
                key,
                {
                    "buckets": bucket_values,
                    "counts": {bucket: 0 for bucket in bucket_values},
                    "count": 0,
                    "sum": 0.0,
                },
            )
            series["count"] += 1
            series["sum"] += value
            for bucket in series["buckets"]:
                if value <= bucket:
                    series["counts"][bucket] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name in sorted(self._meta):
                metric_type, description = self._meta[name]
                lines.append(f"# HELP {name} {description}")
                lines.append(f"# TYPE {name} {metric_type}")

                if metric_type == "counter":
                    for labels, value in sorted(self._counters[name].items()):
                        lines.append(f"{name}{_format_labels(labels)} {value}")
                elif metric_type == "gauge":
                    for labels, value in sorted(self._gauges[name].items()):
                        lines.append(f"{name}{_format_labels(labels)} {value}")
                elif metric_type == "histogram":
                    for labels, series in sorted(self._histograms[name].items()):
                        for bucket in series["buckets"]:
                            bucket_labels = labels + (("le", str(bucket)),)
                            lines.append(
                                f"{name}_bucket{_format_labels(bucket_labels)} {series['counts'][bucket]}"
                            )
                        inf_labels = labels + (("le", "+Inf"),)
                        lines.append(
                            f"{name}_bucket{_format_labels(inf_labels)} {series['count']}"
                        )
                        lines.append(f"{name}_count{_format_labels(labels)} {series['count']}")
                        lines.append(f"{name}_sum{_format_labels(labels)} {series['sum']}")

            return "\n".join(lines) + "\n"


_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry
