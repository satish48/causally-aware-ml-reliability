"""
src/dependency_graph.py
-----------------------
Lightweight dependency DAG for ML pipeline component tracing.

Industry context:
    Google's ML Metadata (MLMD) and LinkedIn's Feathr maintain a full
    lineage graph for every artifact. The gap is that when a model degrades,
    the first question is "which service owns the pipeline feeding this feature?"
    This module implements a minimal DAG that answers exactly that:
        "Fraud model depends on feature X → feature X from pipeline P
         → pipeline P owned by service S → service S last deployed at T"

    This is NOT a general-purpose graph library. It's scoped specifically to
    the five node types that appear in a standard ML serving stack:
        model → feature → pipeline → service → deployment

Design:
    - Adjacency list using dict[str, DependencyNode]
    - Edges are typed and carry a label describing the dependency relationship
    - trace_upstream() does BFS to find all ancestors of a node
    - get_downstream() does BFS to find impact radius (what does this node affect)
    - mark_degraded() propagates a "degraded" health signal upstream
    - Pre-seeded with the fraud detection DAG for immediate use
    - Nodes carry a health_score (0-1) and last_event fields for real-time annotation
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "NodeType",
    "DependencyNode",
    "DependencyEdge",
    "DependencyGraph",
    "get_dependency_graph",
]

NodeType = str   # "model" | "feature" | "pipeline" | "service" | "deployment"

_NODE_TYPE_PRIORITY: dict[str, int] = {
    "model": 5,
    "feature": 4,
    "pipeline": 3,
    "service": 2,
    "deployment": 1,
}


@dataclass
class DependencyNode:
    """
    A node in the ML pipeline dependency graph.

    health_score: 0-1, updated by mark_degraded() and external health events.
    last_event:   description of the most recent significant event on this node.
    degraded:     True when health_score < 0.70 or explicitly marked.
    """
    node_id: str
    node_type: NodeType
    display_name: str
    owner_team: str          # team responsible for this component
    health_score: float = 1.0
    last_event: str = ""
    last_event_at: str = ""
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "display_name": self.display_name,
            "owner_team": self.owner_team,
            "health_score": round(self.health_score, 3),
            "last_event": self.last_event,
            "last_event_at": self.last_event_at,
            "degraded": self.degraded,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class DependencyEdge:
    """A directed edge: source depends on target."""
    source_id: str
    target_id: str
    label: str          # e.g. "reads_feature", "produced_by", "deployed_via"
    weight: float = 1.0   # relative dependency strength 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "weight": self.weight,
        }


@dataclass(slots=True)
class TraceResult:
    """Result of trace_upstream() or get_downstream()."""
    origin_id: str
    nodes: list[DependencyNode]    # BFS-ordered, not including origin
    edges: list[DependencyEdge]    # edges traversed
    depth: int                     # max depth reached
    degraded_nodes: list[str]      # node_ids that are currently degraded

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "depth": self.depth,
            "degraded_nodes": self.degraded_nodes,
            "node_count": len(self.nodes),
        }


# ────────────────────────────────────────────────────────────────
# GRAPH
# ────────────────────────────────────────────────────────────────

class DependencyGraph:
    """
    Directed dependency graph: source depends on target (edges point upstream).

    Methods:
        add_node()          — register a component
        add_edge()          — declare a dependency relationship
        trace_upstream()    — BFS from a node to all its ancestors
        get_downstream()    — BFS from a node to all components that depend on it
        mark_degraded()     — annotate a node as unhealthy
        update_health()     — set health_score and propagate
        get_stats()         — graph summary

    Thread-safety: not thread-safe — designed for single-threaded async event loop.
    """

    MAX_BFS_DEPTH: int = 10

    def __init__(self) -> None:
        self._nodes: dict[str, DependencyNode] = {}
        self._edges: list[DependencyEdge] = []
        # Adjacency: node_id → list of target node_ids (upstream deps)
        self._adj_upstream: dict[str, list[str]] = {}
        # Reverse adjacency: node_id → list of source node_ids (downstream consumers)
        self._adj_downstream: dict[str, list[str]] = {}
        logger.info("DependencyGraph initialized")

    # ── GRAPH CONSTRUCTION ────────────────────────────────────

    def add_node(self, node: DependencyNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adj_upstream:
            self._adj_upstream[node.node_id] = []
        if node.node_id not in self._adj_downstream:
            self._adj_downstream[node.node_id] = []

    def add_edge(self, edge: DependencyEdge) -> None:
        """Register that source depends on target."""
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise ValueError(
                f"Both nodes must be registered before adding edge: "
                f"{edge.source_id} → {edge.target_id}"
            )
        self._edges.append(edge)
        self._adj_upstream.setdefault(edge.source_id, []).append(edge.target_id)
        self._adj_downstream.setdefault(edge.target_id, []).append(edge.source_id)

    # ── TRAVERSAL ─────────────────────────────────────────────

    def trace_upstream(self, node_id: str, max_depth: int | None = None) -> TraceResult:
        """
        BFS from node_id following upstream dependency edges.

        Returns all ancestors in BFS order with the traversed edges.
        Use this to answer: "what does fraud_detection_v1 depend on?"
        """
        depth_limit = min(max_depth or self.MAX_BFS_DEPTH, self.MAX_BFS_DEPTH)
        return self._bfs(
            origin_id=node_id,
            adj=self._adj_upstream,
            depth_limit=depth_limit,
        )

    def get_downstream(self, node_id: str, max_depth: int | None = None) -> TraceResult:
        """
        BFS from node_id following downstream consumer edges.

        Returns all dependents in BFS order.
        Use this to answer: "if payment_pipeline degrades, what models are affected?"
        """
        depth_limit = min(max_depth or self.MAX_BFS_DEPTH, self.MAX_BFS_DEPTH)
        return self._bfs(
            origin_id=node_id,
            adj=self._adj_downstream,
            depth_limit=depth_limit,
        )

    def _bfs(
        self,
        origin_id: str,
        adj: dict[str, list[str]],
        depth_limit: int,
    ) -> TraceResult:
        if origin_id not in self._nodes:
            return TraceResult(
                origin_id=origin_id,
                nodes=[],
                edges=[],
                depth=0,
                degraded_nodes=[],
            )

        visited: set[str] = {origin_id}
        queue: deque[tuple[str, int]] = deque([(origin_id, 0)])
        result_nodes: list[DependencyNode] = []
        result_edges: list[DependencyEdge] = []
        max_depth_reached = 0

        while queue:
            current_id, depth = queue.popleft()
            if depth >= depth_limit:
                continue
            for neighbor_id in adj.get(current_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1))
                max_depth_reached = max(max_depth_reached, depth + 1)
                if neighbor_id in self._nodes:
                    result_nodes.append(self._nodes[neighbor_id])

                # Find the edge connecting current → neighbor
                for e in self._edges:
                    if (e.source_id == current_id and e.target_id == neighbor_id) or \
                       (e.source_id == neighbor_id and e.target_id == current_id):
                        result_edges.append(e)
                        break

        degraded = [n.node_id for n in result_nodes if n.degraded]
        return TraceResult(
            origin_id=origin_id,
            nodes=result_nodes,
            edges=result_edges,
            depth=max_depth_reached,
            degraded_nodes=degraded,
        )

    # ── HEALTH ANNOTATION ─────────────────────────────────────

    def mark_degraded(self, node_id: str, reason: str = "", health_score: float = 0.0) -> None:
        """Mark a node as degraded and annotate with the reason."""
        if node_id not in self._nodes:
            logger.warning("mark_degraded: unknown node %s", node_id)
            return
        node = self._nodes[node_id]
        node.degraded = True
        node.health_score = max(0.0, min(1.0, health_score))
        node.last_event = reason or "Marked degraded"
        node.last_event_at = datetime.now(timezone.utc).isoformat()
        logger.info("Node degraded | id=%s | score=%.2f | reason=%s", node_id, health_score, reason)

    def update_health(self, node_id: str, health_score: float, event: str = "") -> None:
        """Update health score for a node. Scores below 0.70 automatically set degraded=True."""
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        node.health_score = max(0.0, min(1.0, health_score))
        node.degraded = node.health_score < 0.70
        if event:
            node.last_event = event
            node.last_event_at = datetime.now(timezone.utc).isoformat()

    def recover_node(self, node_id: str) -> None:
        """Mark a node as recovered."""
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        node.degraded = False
        node.health_score = 1.0
        node.last_event = "Recovered"
        node.last_event_at = datetime.now(timezone.utc).isoformat()

    # ── QUERIES ───────────────────────────────────────────────

    def get_node(self, node_id: str) -> DependencyNode | None:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[dict[str, Any]]:
        return [n.to_dict() for n in self._nodes.values()]

    def get_all_edges(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._edges]

    def get_degraded_nodes(self) -> list[dict[str, Any]]:
        return [n.to_dict() for n in self._nodes.values() if n.degraded]

    def get_stats(self) -> dict[str, Any]:
        nodes = list(self._nodes.values())
        return {
            "total_nodes": len(nodes),
            "total_edges": len(self._edges),
            "degraded_nodes": sum(1 for n in nodes if n.degraded),
            "avg_health_score": (
                sum(n.health_score for n in nodes) / len(nodes) if nodes else 1.0
            ),
            "node_types": {
                t: sum(1 for n in nodes if n.node_type == t)
                for t in ("model", "feature", "pipeline", "service", "deployment")
            },
        }


# ────────────────────────────────────────────────────────────────
# DEFAULT FRAUD DETECTION DAG SEED
# ────────────────────────────────────────────────────────────────

def _seed_fraud_detection_dag(graph: DependencyGraph) -> None:
    """
    Pre-seed the fraud detection pipeline dependency graph.

    Topology:
        fraud_detection_v1 (model)
            ↑ reads_feature: txn_amount_feature, velocity_feature, merchant_risk_feature
        txn_amount_feature (feature)
            ↑ produced_by: payment_pipeline
        velocity_feature (feature)
            ↑ produced_by: event_stream_pipeline
        merchant_risk_feature (feature)
            ↑ produced_by: merchant_data_pipeline
        payment_pipeline (pipeline)
            ↑ owned_by: payments_service
        event_stream_pipeline (pipeline)
            ↑ owned_by: streaming_service
        merchant_data_pipeline (pipeline)
            ↑ owned_by: merchant_service
        payments_service (service)
            ↑ deployed_via: payments_deployment_v3
        streaming_service (service)
            ↑ deployed_via: streaming_deployment_v7
        merchant_service (service)
            ↑ deployed_via: merchant_deployment_v2
    """
    nodes = [
        DependencyNode("fraud_detection_v1", "model", "Fraud Detection v1", "ml-platform",
                       metadata={"version": "1.0.0", "framework": "xgboost"}),
        DependencyNode("txn_amount_feature", "feature", "Transaction Amount Features", "data-eng",
                       metadata={"feature_group": "transaction", "update_freq": "real-time"}),
        DependencyNode("velocity_feature", "feature", "Transaction Velocity Features", "data-eng",
                       metadata={"feature_group": "velocity", "update_freq": "real-time"}),
        DependencyNode("merchant_risk_feature", "feature", "Merchant Risk Score", "risk-eng",
                       metadata={"feature_group": "merchant", "update_freq": "hourly"}),
        DependencyNode("payment_pipeline", "pipeline", "Payment Events Pipeline", "payments-infra",
                       metadata={"language": "Java", "sla_ms": 50}),
        DependencyNode("event_stream_pipeline", "pipeline", "Event Stream Pipeline", "streaming-infra",
                       metadata={"language": "Flink", "sla_ms": 30}),
        DependencyNode("merchant_data_pipeline", "pipeline", "Merchant Data Pipeline", "merchant-infra",
                       metadata={"language": "Python", "sla_ms": 200}),
        DependencyNode("payments_service", "service", "Payments Service", "payments",
                       metadata={"repo": "stripe/payments", "language": "Go"}),
        DependencyNode("streaming_service", "service", "Streaming Service", "streaming",
                       metadata={"repo": "stripe/stream-proc", "language": "Scala"}),
        DependencyNode("merchant_service", "service", "Merchant Service", "merchant",
                       metadata={"repo": "stripe/merchants", "language": "Ruby"}),
        DependencyNode("payments_deployment_v3", "deployment", "Payments Deploy v3", "deploys",
                       metadata={"commit": "a1b2c3d", "deployed_at": "2026-04-20T10:00:00Z"}),
        DependencyNode("streaming_deployment_v7", "deployment", "Streaming Deploy v7", "deploys",
                       metadata={"commit": "e4f5g6h", "deployed_at": "2026-04-22T14:30:00Z"}),
        DependencyNode("merchant_deployment_v2", "deployment", "Merchant Deploy v2", "deploys",
                       metadata={"commit": "i7j8k9l", "deployed_at": "2026-04-18T09:15:00Z"}),
    ]
    for node in nodes:
        graph.add_node(node)

    edges = [
        DependencyEdge("fraud_detection_v1", "txn_amount_feature", "reads_feature", 0.9),
        DependencyEdge("fraud_detection_v1", "velocity_feature", "reads_feature", 0.85),
        DependencyEdge("fraud_detection_v1", "merchant_risk_feature", "reads_feature", 0.7),
        DependencyEdge("txn_amount_feature", "payment_pipeline", "produced_by", 1.0),
        DependencyEdge("velocity_feature", "event_stream_pipeline", "produced_by", 1.0),
        DependencyEdge("merchant_risk_feature", "merchant_data_pipeline", "produced_by", 1.0),
        DependencyEdge("payment_pipeline", "payments_service", "owned_by", 1.0),
        DependencyEdge("event_stream_pipeline", "streaming_service", "owned_by", 1.0),
        DependencyEdge("merchant_data_pipeline", "merchant_service", "owned_by", 1.0),
        DependencyEdge("payments_service", "payments_deployment_v3", "deployed_via", 1.0),
        DependencyEdge("streaming_service", "streaming_deployment_v7", "deployed_via", 1.0),
        DependencyEdge("merchant_service", "merchant_deployment_v2", "deployed_via", 1.0),
    ]
    for edge in edges:
        graph.add_edge(edge)

    logger.info("DependencyGraph seeded with fraud detection DAG | nodes=%d | edges=%d",
                len(nodes), len(edges))


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_graph: DependencyGraph | None = None


def get_dependency_graph() -> DependencyGraph:
    global _graph
    if _graph is None:
        _graph = DependencyGraph()
        _seed_fraud_detection_dag(_graph)
    return _graph
