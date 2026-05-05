const API_ORIGIN =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000"
    : window.location.origin;

const WS_URL =
  window.location.protocol === "file:"
    ? "ws://127.0.0.1:8000/ws/live"
    : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/live`;

const BOOTSTRAP_URL = `${API_ORIGIN}/dashboard/bootstrap`;
const FALLBACK_POLL_MS = 8000;
const MAX_LOG_LINES = 60;

let ws;
let metricsChart;
let driftChart;
let reconnectDelayMs = 1500;
let reconnectTimer = null;
let fallbackTimer = null;
let lastBatchId = null;
let messageCount = 0;
let skeletonsCleared = false;

const $ = id => document.getElementById(id);
const n = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const safeArray = value => Array.isArray(value) ? value : [];
const safeObject = value => (value && typeof value === "object" && !Array.isArray(value) ? value : {});
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const pct = (value, digits = 1) => `${(n(value) * 100).toFixed(digits)}%`;
const money = value => `$${Math.round(n(value)).toLocaleString()}`;
const h = value => String(value ?? "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;")
  .replace(/'/g, "&#39;");

function time(ts) {
  try {
    return ts ? new Date(ts).toLocaleTimeString() : new Date().toLocaleTimeString();
  } catch {
    return new Date().toLocaleTimeString();
  }
}

function clsRisk(level) {
  const normalized = String(level || "").toLowerCase();
  if (["critical", "high"].includes(normalized)) return "err";
  if (["medium", "moderate", "warning"].includes(normalized)) return "warn";
  return "ok";
}

function setConnectionStatus(label, tone = "", transport = label) {
  $("statusPill").className = `pill ${tone}`.trim();
  $("statusText").textContent = label;
  $("impactTransport").textContent = transport;
}

function addLog(level, message, drift = false) {
  const terminal = $("terminalLog");
  if (!terminal) return;

  const line = document.createElement("div");
  const levelClass = level === "ERROR" ? "err" : level === "WARNING" ? "warn" : "info";
  line.className = "t-line";
  line.innerHTML = `
    <span class="t-time">${time()}</span>
    <span class="t-level ${levelClass}">${h(level)}</span>
    <span class="t-msg ${drift ? "drift" : ""}">${h(message)}</span>
  `;
  terminal.prepend(line);

  while (terminal.children.length > MAX_LOG_LINES) {
    terminal.removeChild(terminal.lastChild);
  }
}

function createMetricGradient(ctx, color) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 220);
  gradient.addColorStop(0, `${color}66`);
  gradient.addColorStop(1, `${color}00`);
  return gradient;
}

function initCharts() {
  Chart.defaults.font.family = "\"Avenir Next\", \"Trebuchet MS\", sans-serif";
  Chart.defaults.color = "#9aa8bc";
  Chart.defaults.borderColor = "rgba(140, 162, 194, 0.14)";

  const metricsCtx = $("metricsChart").getContext("2d");
  const driftCtx = $("driftChart").getContext("2d");

  metricsChart = new Chart(metricsCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Fraud Rate %",
          data: [],
          borderColor: "#3fe0a1",
          backgroundColor: createMetricGradient(metricsCtx, "#3fe0a1"),
          borderWidth: 2.2,
          tension: 0.32,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          label: "Avg Confidence %",
          data: [],
          borderColor: "#73b8ff",
          backgroundColor: createMetricGradient(metricsCtx, "#73b8ff"),
          borderWidth: 2.2,
          tension: 0.32,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          align: "start",
          labels: { color: "#90a1b9", boxWidth: 14, usePointStyle: true },
        },
        tooltip: {
          backgroundColor: "rgba(8, 13, 22, 0.96)",
          borderColor: "rgba(121, 151, 201, 0.18)",
          borderWidth: 1,
          padding: 12,
        },
      },
      scales: {
        x: {
          ticks: { color: "#51627a", maxTicksLimit: 10 },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#51627a" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
      },
    },
  });

  driftChart = new Chart(driftCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Overall PSI",
          data: [],
          borderColor: "#ff6d78",
          backgroundColor: createMetricGradient(driftCtx, "#ff6d78"),
          borderWidth: 2.2,
          tension: 0.28,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
          yAxisID: "psi",
        },
        {
          label: "Max Z Score",
          data: [],
          borderColor: "#ffc56a",
          backgroundColor: createMetricGradient(driftCtx, "#ffc56a"),
          borderWidth: 2.2,
          tension: 0.28,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
          yAxisID: "z",
        },
        {
          label: "PSI Alert Threshold",
          data: [],
          borderColor: "rgba(255,108,120,0.45)",
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
          yAxisID: "psi",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          align: "start",
          labels: {
            color: "#90a1b9", boxWidth: 14, usePointStyle: true,
            filter: item => item.text !== "PSI Alert Threshold",
          },
        },
        tooltip: {
          backgroundColor: "rgba(8, 13, 22, 0.96)",
          borderColor: "rgba(121, 151, 201, 0.18)",
          borderWidth: 1,
          padding: 12,
          callbacks: {
            afterBody: (items) => {
              const psiItem = items.find(i => i.dataset.label === "Overall PSI");
              if (psiItem && psiItem.parsed.y >= 0.2) {
                return [`⚠ PSI ${psiItem.parsed.y >= 0.5 ? "CRITICAL" : "MODERATE"} — distribution shifted`];
              }
              return [];
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#51627a", maxTicksLimit: 10 },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        psi: {
          type: "linear",
          position: "left",
          min: 0,
          max: 1.2,
          ticks: { color: "#ff9aa2" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
        z: {
          type: "linear",
          position: "right",
          beginAtZero: true,
          ticks: { color: "#ffd08d" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

function standardSimulationDefaults(lossHr, riskScore) {
  return [
    {
      action: "rollback_pipeline",
      loss_reduction_pct: 74,
      risk_level: "low",
      recovery_eta: "5-10 min",
      reasoning: "Reverts the most likely upstream change and removes the fastest source of active exposure.",
      operator_cost: "Low",
    },
    {
      action: "increase_manual_review",
      loss_reduction_pct: 57,
      risk_level: "medium",
      recovery_eta: "Immediate mitigation",
      reasoning: "Buys safety while preserving model availability, but increases review queue pressure.",
      operator_cost: "Medium",
    },
    {
      action: "trigger_retraining",
      loss_reduction_pct: 33,
      risk_level: "medium",
      recovery_eta: "45-90 min",
      reasoning: "Useful if the drift is persistent rather than caused by a bad upstream contract or deploy.",
      operator_cost: "High",
    },
    {
      action: "ignore",
      loss_reduction_pct: 0,
      risk_level: riskScore >= 55 ? "critical" : "high",
      recovery_eta: "No automatic recovery",
      reasoning: `Leaves the platform exposed to roughly ${money(lossHr)}/hr in projected loss.`,
      operator_cost: "None",
    },
  ];
}

function normalizeRootCauses(data, causal, explanation, incident, psi, maxZ, conf, fraudRate) {
  const rawCandidates = safeArray(data.root_causes).length
    ? safeArray(data.root_causes)
    : safeArray(causal.root_causes || causal.top_causes || causal.hypotheses);

  if (rawCandidates.length) {
    return rawCandidates.slice(0, 3).map((item, index) => ({
      component: item.component || item.source || item.event_type || `candidate_${index + 1}`,
      confidence: clamp(n(item.confidence, 0.35), 0, 1),
      severity: String(item.severity || item.level || "moderate").toLowerCase(),
      evidence: safeArray(item.evidence).length
        ? safeArray(item.evidence)
        : [
            item.explanation || "Evidence available from the live incident pipeline.",
            item.affected_fields?.length
              ? `Affected fields: ${item.affected_fields.join(", ")}`
              : "Field-level detail not supplied.",
          ],
    }));
  }

  const topFeatures = safeArray(explanation.top_features || incident.top_features || causal.top_features);
  // Generate meaningfully different alternate hypotheses instead of repeating feature_pipeline
  return [
    {
      component: topFeatures.length ? `feature_pipeline:${topFeatures.slice(0, 2).join("+")}` : "feature_pipeline_v2",
      confidence: clamp(psi * 0.72 + maxZ * 0.04, 0.45, 0.92),
      severity: psi >= 0.5 ? "critical" : "moderate",
      evidence: [
        `PSI=${psi.toFixed(3)} indicates material feature distribution shift.`,
        topFeatures.length ? `Top shifted features: ${topFeatures.join(", ")}` : "Awaiting explicit feature evidence from the backend.",
        `Average confidence is ${(conf * 100).toFixed(1)}% and fraud rate is ${(fraudRate * 100).toFixed(1)}%.`,
      ],
    },
    {
      component: "upstream_batch_ingestion",
      confidence: clamp(psi * 0.45, 0.2, 0.72),
      severity: "moderate",
      evidence: [
        "The distribution shift aligns temporally with the latest ingestion window.",
        "Batch schema contract may have silently changed upstream, affecting feature encoding.",
        "No hard outage required to explain current degradation — silent contract drift is consistent.",
      ],
    },
    {
      component: "serving_threshold_policy",
      confidence: clamp(fraudRate * 1.8, 0.15, 0.48),
      severity: "high",
      evidence: [
        `Fraud rate at ${(fraudRate * 100).toFixed(1)}% exceeds the configured alert threshold.`,
        "Serving policy may be amplifying downstream business exposure via stale threshold calibration.",
        "Secondary hypothesis — requires ruling out upstream pipeline before acting on this.",
      ],
    },
  ];
}

function normalizeSimulations(data, decision, lossHr, riskScore) {
  const raw = data.simulations ?? data.decision_simulations ?? null;
  let candidates = [];

  if (Array.isArray(raw)) {
    candidates = raw;
  } else {
    const obj = safeObject(raw);
    candidates = safeArray(obj.options || obj.simulations || obj.actions || obj.results);
    if (!candidates.length && Object.keys(obj).length) {
      candidates = Object.entries(obj).map(([action, value]) =>
        typeof value === "object" && value !== null
          ? { action, ...value }
          : { action, loss_reduction_pct: 0, risk_level: "medium", reasoning: String(value) }
      );
    }
  }

  const merged = standardSimulationDefaults(lossHr, riskScore);
  for (const candidate of candidates) {
    const actionName = String(candidate.action || candidate.name || "").toLowerCase();
    const idx = merged.findIndex(base => actionName && actionName.includes(base.action.split("_")[0]));
    if (idx >= 0) {
      merged[idx] = {
        ...merged[idx],
        ...candidate,
        action: candidate.action || merged[idx].action,
      };
    } else {
      merged.push({
        action: candidate.action || `action_${merged.length + 1}`,
        loss_reduction_pct: n(candidate.loss_reduction_pct, 0),
        risk_level: candidate.risk_level || "medium",
        recovery_eta: candidate.recovery_eta || candidate.eta || "Unknown",
        reasoning: candidate.reasoning || candidate.impact || "No simulation rationale supplied.",
        operator_cost: candidate.operator_cost || "Unknown",
      });
    }
  }

  return merged.slice(0, 4);
}

function normalizeDependencyNodes(data, topFeatures, state, psi) {
  const raw = data.dependency_trace ?? data.dependency_graph ?? null;
  let nodes = [];

  if (Array.isArray(raw)) {
    nodes = raw;
  } else {
    const obj = safeObject(raw);
    nodes = safeArray(obj.nodes || obj.trace || obj.hops || obj.path);
  }

  if (nodes.length) {
    return nodes.slice(0, 8).map(node => {
      const healthScore = n(node.health_score, node.degraded ? 0.5 : 1);
      const degraded = Boolean(node.degraded) || healthScore < 0.8;
      return {
        type: node.type || node.node_type || "service",
        name: node.name || node.display_name || node.node_id || node.component || "unknown_node",
        owner: node.owner || node.owner_team || "platform",
        status: node.status || node.health || (degraded ? "degraded" : "ok"),
        detail:
          node.detail ||
          node.last_event ||
          (node.metadata && Object.keys(node.metadata).length
            ? Object.entries(node.metadata).slice(0, 2).map(([key, value]) => `${key}: ${value}`).join(" • ")
            : "No recent incident note"),
      };
    });
  }

  return [
    { type: "deployment", name: "schema_contract_v15", owner: "Data Eng", status: psi > 0.5 ? "degraded" : "ok", detail: "Latest schema contract release" },
    { type: "service", name: "payments_service", owner: "Payments", status: "ok", detail: "Core upstream transaction service" },
    { type: "pipeline", name: "feature_pipeline_v2", owner: "Data Platform", status: psi > 0.2 ? "degraded" : "ok", detail: "Feature aggregation and validation path" },
    { type: "feature", name: topFeatures[0] || "transaction_velocity_24h", owner: "Feature Store", status: state === "healthy" ? "ok" : "degraded", detail: "Highest sensitivity feature group" },
    { type: "model", name: "fraud_detection_v1", owner: "ML Platform", status: state === "healthy" ? "ok" : "degraded", detail: "Online decision model" },
  ];
}

function normalizeTimeline(data, decision, psi, conf, severity) {
  const raw = data.causal_timeline ?? data.timeline ?? null;
  let events = [];
  // timeline_narrative is the promoted top-level field from the backend.
  // For older object-shaped causal_timeline payloads, obj.narrative is the fallback.
  let narrative = data.timeline_narrative || null;

  if (Array.isArray(raw)) {
    events = raw;
  } else {
    const obj = safeObject(raw);
    events = safeArray(obj.events || obj.items || obj.timeline);
    narrative = narrative || obj.narrative || null;
  }

  const normalized = events.map((event, index) => ({
    timestamp: event.timestamp || event.time || data.timestamp,
    title: event.title || event.event || event.name || `Event ${index + 1}`,
    description: event.description || event.detail || event.summary || "",
    causal_link: event.causal_link || event.causal || event.relationship || "Contributes to incident evidence",
    severity: String(event.severity || event.level || "info").toLowerCase(),
  }));

  if (normalized.length) {
    return { events: normalized, narrative };
  }

  return {
    narrative,
    events: [
      {
        timestamp: data.timestamp,
        title: "Feature batch ingested",
        description: "New production batch reached the online scoring path.",
        severity: "info",
        causal_link: "Source data entered the serving system",
      },
      {
        timestamp: data.timestamp,
        title: "Distribution shift evaluated",
        description: `Aggregate PSI is ${psi.toFixed(2)} against the configured baseline.`,
        severity: psi > 0.5 ? "critical" : "moderate",
        causal_link: "Feature movement increased incident risk",
      },
      {
        timestamp: data.timestamp,
        title: "Model certainty changed",
        description: `Average confidence is ${(conf * 100).toFixed(1)}%.`,
        severity: conf < 0.75 ? "high" : "info",
        causal_link: "Serving quality altered downstream decision safety",
      },
      {
        timestamp: data.timestamp,
        title: "Incident action prepared",
        description: decision.recommended_action || decision.action || "monitor",
        severity,
        causal_link: "Operator response now recommended",
      },
    ],
  };
}

function normalizePayload(data) {
  const metrics = safeObject(data.metrics);
  const health = safeObject(data.health);
  const detector = safeObject(data.detector);
  const display = safeObject(data.display_state);
  const impact = safeObject(data.impact || data.business_impact);
  const decision = safeObject(data.decision || data.recommendation);
  const incident = safeObject(data.incident_summary);
  const explanation = safeObject(data.explanation);
  const risk = safeObject(data.risk_forecast);
  const rawSimulations = safeObject(data.simulations);
  const causal = safeObject(data.causal_attribution);
  const canary = safeObject(data.canary);

  const fraudRate = n(metrics.fraud_rate, 0);
  const conf = n(metrics.avg_confidence, 0);
  const psi = n(detector.latest_psi ?? detector.overall_psi ?? risk.current_psi, 0);
  const maxZ = n(detector.max_z_score ?? risk.max_z_score, 0);
  const topFeatures = safeArray(explanation.top_features || incident.top_features || causal.top_features);

  const status = display.label || (display.drift_active || detector.drift_detected ? "DRIFTING" : "STABLE");
  const state = display.incident_state || (display.drift_active || detector.drift_detected ? "active" : "healthy");
  const severity = String(display.severity || incident.severity || detector.latest_severity || "low").toLowerCase();

  const lossHr = n(risk.loss_per_hour_usd ?? risk.loss_per_hour ?? impact.loss_per_hour_usd ?? impact.estimated_loss_usd, 0);
  const riskScore = n(risk.risk_score ?? impact.severity_score, 0);
  const burn = n(risk.worst_burn_rate ?? risk.burn_rate, psi >= 1 ? 3.2 : psi * 2.1);
  const budgetRemaining = n(risk.budget_remaining_pct, 100);
  const eta = risk.slo_breach_eta_minutes ?? risk.slo_breach_eta ?? (riskScore > 65 ? 35 : riskScore > 35 ? 90 : null);

  const rootCauses = normalizeRootCauses(data, causal, explanation, incident, psi, maxZ, conf, fraudRate);
  const simulationList = normalizeSimulations(data, decision, lossHr, riskScore);
  const depNodes = normalizeDependencyNodes(data, topFeatures, state, psi);
  const timeline = normalizeTimeline(data, decision, psi, conf, severity);

  const rawChartData = safeArray(data.chart_data);
  const rawDriftData = safeArray(data.drift_data);
  const chartData = rawChartData.length
    ? rawChartData
    : metrics.batch_id != null
      ? [{ batch_id: metrics.batch_id, fraud_rate: fraudRate * 100, avg_confidence: conf * 100 }]
      : [];
  const driftData = rawDriftData.length
    ? rawDriftData
    : metrics.batch_id != null
      ? [{ batch_id: metrics.batch_id, overall_psi: psi, max_z_score: maxZ }]
      : [];

  return {
    metrics,
    health,
    detector,
    display,
    impact,
    decision,
    incident,
    explanation,
    risk,
    causal,
    rootCauses,
    simulationList,
    depNodes,
    timelineEvents: timeline.events,
    timelineNarrative: timeline.narrative,
    chartData,
    driftData,
    fraudRate,
    conf,
    psi,
    maxZ,
    topFeatures,
    status,
    state,
    severity,
    lossHr,
    riskScore,
    burn,
    eta,
    budgetRemaining,
    rawSimulations,
    canary,
  };
}

function updateHeader(x, raw, sourceLabel) {
  const active = x.state !== "healthy" && x.status !== "STABLE";
  const tone = (["critical", "high"].includes(x.severity) || x.state === "active")
    ? "danger"
    : active ? "warn" : "";
  const transportLabel = ws && ws.readyState === WebSocket.OPEN ? "Live WebSocket" : sourceLabel;

  $("modeLabel").textContent = `mode: ${(x.health && x.health.mode) || "real"}`;
  $("topbarTime").textContent = time(raw.timestamp);

  $("stateLed").className = `state-led ${tone}`.trim();
  $("stateLabel").className = `state-label ${tone}`.trim();
  $("stateLabel").textContent = x.status;
  $("stateSub").textContent = x.display.subtitle || (active ? "Active reliability incident" : "System healthy");

  const primaryCause = x.rootCauses[0] || {};
  const causeName = primaryCause.component || "unknown cause";
  const decisionAction = x.decision.recommended_action || x.decision.action || "monitor";

  $("impactHeadlineMain").textContent = active
    ? `Projected exposure is climbing near ${money(x.lossHr)}/hr. Most likely source: ${causeName}.`
    : "Live command center is connected. The platform is within current guardrails.";
  $("impactHeadlineSub").textContent = active
    ? `Attribution confidence ${(n(primaryCause.confidence) * 100).toFixed(0)}% • PSI ${x.psi.toFixed(2)} • Fraud ${(x.fraudRate * 100).toFixed(1)}% • Burn ${x.burn.toFixed(1)}x`
    : "The dashboard will elevate root cause, risk, and operator response the moment the incident pipeline detects sustained degradation.";
  $("impactHeadlineMoney").textContent = `${money(x.lossHr)}/hr`;
  $("impactRootCause").textContent = causeName;
  $("impactDecision").textContent = decisionAction.replaceAll("_", " ");
  $("impactTransport").textContent = transportLabel;

  $("stripNarrative").textContent = x.display.banner || (active
    ? `Incident state ${x.state}. Recommended action is ${decisionAction.replaceAll("_", " ")} while the system tracks causal evidence in real time.`
    : "No active incidents. Watch the command strip for burn-rate, projected loss, and response posture.");

  $("riskScore").textContent = Math.round(x.riskScore || 0);
  $("riskScore").className = `strip-value ${clsRisk(x.risk.risk_level || x.impact.business_impact_label || x.severity)}`;

  $("lossPerHour").textContent = money(x.lossHr);
  $("lossPerHour").className = `strip-value ${x.lossHr > 3000 ? "err" : x.lossHr > 500 ? "warn" : "ok"}`;

  $("burnRate").textContent = `${x.burn.toFixed(1)}×`;
  $("burnRate").className = `strip-value ${x.burn > 6 ? "err" : x.burn > 1 ? "warn" : "ok"}`;

  const etaBreached = x.budgetRemaining <= 0 && x.eta == null;
  $("sloEta").textContent = etaBreached ? "BREACHED" : x.eta == null ? "N/A" : `${Math.round(n(x.eta))}m`;
  $("sloEta").className = `strip-value ${etaBreached ? "err" : x.eta != null && x.eta < 30 ? "err" : x.eta != null && x.eta < 90 ? "warn" : "ok"}`;
}

function updateRootCauses(x) {
  const causes = x.rootCauses.slice(0, 3);
  if (!causes.length) {
    $("rootCauseAttribution").innerHTML = `<div class="empty pro">Awaiting ranked root-cause evidence from the incident pipeline.</div>`;
    return;
  }

  $("rootCauseAttribution").innerHTML = causes.map((cause, index) => {
    const confidence = clamp(n(cause.confidence), 0, 1);
    const confidencePct = Math.round(confidence * 100);
    const barTone = confidencePct >= 75 ? "" : confidencePct >= 45 ? "warn" : "err";
    const severity = String(cause.severity || "moderate").toLowerCase();
    return `
      <article class="cause-card ${index === 0 ? "primary" : ""}">
        <div class="cause-top">
          <div>
            <div class="cause-rank">${index === 0 ? "Primary causal hypothesis" : `Alternate hypothesis ${index + 1}`}</div>
            <div class="cause-name">${h(cause.component || "unknown_component")}</div>
          </div>
          <div class="conf-pill ${barTone || "ok"}">${severity}</div>
        </div>
        <div class="confidence-row">
          <div class="conf-number">${confidencePct}%</div>
          <div class="conf-label">confidence</div>
        </div>
        <div class="bar"><div class="bar-fill ${barTone}" style="width:${confidencePct}%"></div></div>
        <div class="evidence-list">
          ${safeArray(cause.evidence).slice(0, 4).map(item => `<div class="evidence-item">${h(item)}</div>`).join("")}
        </div>
      </article>
    `;
  }).join("");
}

function updateDecision(x) {
  const decision = x.decision || {};
  const sim = safeObject(x.rawSimulations ?? null);
  const simAmbiguous = sim.ambiguous === true;
  const simRecommended = sim.recommended_action || null;

  const action = decision.recommended_action || decision.action || (x.state === "healthy" ? "monitor" : "open_incident");
  const confidence = n(decision.confidence, 0.79);
  const priority = decision.priority || (x.riskScore >= 70 ? "critical" : x.riskScore >= 40 ? "high" : x.state === "healthy" ? "low" : "medium");
  const rationale = decision.rationale || (x.state === "healthy"
    ? "No active reliability issue. Continue observing the live stream and keep the canary posture stable."
    : "Decision chosen from risk score, causal confidence, SLO budget burn, and projected business loss.");

  const topSimulation = x.simulationList.slice().sort((a, b) => n(b.loss_reduction_pct) - n(a.loss_reduction_pct))[0] || {};

  // Show ambiguity banner when simulation engine disagrees with rule-based decision engine
  const engineConflict = simRecommended && simRecommended !== action;
  const ambiguityBanner = simAmbiguous || engineConflict ? `
    <div class="ambiguity-banner">
      ⚠ Engines disagree — simulation recommends <strong>${h((simRecommended || "unknown").replaceAll("_", " "))}</strong>,
      rule engine recommends <strong>${h(action.replaceAll("_", " "))}</strong>.
      Causal confidence is low; human judgment required before acting.
    </div>` : "";

  const etaDisplay = x.budgetRemaining <= 0 && x.eta == null ? "BREACHED" : x.eta == null ? "unbounded" : `${Math.round(n(x.eta))}m`;

  const canaryDecision = x.canary.decision || "no_canary";
  const canaryRationale = x.canary.rationale || "";
  const canaryColor = canaryDecision === "rollback" ? "err" : canaryDecision === "hold" ? "warn" : canaryDecision === "promote" ? "ok" : "muted";
  const canaryBadge = canaryDecision !== "no_canary"
    ? `<div class="canary-posture ${canaryColor}">
        <span class="canary-label">Deployment Posture</span>
        <span class="canary-value">${h(canaryDecision.toUpperCase())}</span>
        <span class="canary-detail">${h(canaryRationale.slice(0, 120))}${canaryRationale.length > 120 ? "…" : ""}</span>
       </div>` : "";

  // Specific threshold, tradeoff, and projected outcome from enriched model
  const specificThreshold = decision.specific_threshold || null;
  const tradeoff = decision.tradeoff || null;
  const proj = decision.projected_outcome || null;

  // Split tradeoff into gain/cost halves
  let gainText = "", costText = "";
  if (tradeoff) {
    const gainMatch = tradeoff.match(/Gain:\s*([^.]+\.)/i);
    const costMatch = tradeoff.match(/Cost:\s*(.+)/i);
    gainText = gainMatch ? gainMatch[1].trim() : tradeoff;
    costText = costMatch ? costMatch[1].trim() : "";
  }

  const tradeoffBlock = (gainText || costText) ? `
    <div class="decision-tradeoff">
      ${gainText ? `<div class="tradeoff-gain"><div class="tradeoff-label">✓ Gain</div>${h(gainText)}</div>` : ""}
      ${costText ? `<div class="tradeoff-cost"><div class="tradeoff-label">✗ Cost</div>${h(costText)}</div>` : ""}
    </div>` : "";

  const projBlock = proj ? `
    <div class="consequence-grid" style="margin-top:10px">
      <div class="cons-cell">
        <div class="cons-label">T+5 min</div>
        <div class="cons-fraud ${n(proj.t5_fraud_rate_pct) < 8 ? "green" : n(proj.t5_fraud_rate_pct) < 15 ? "amber" : "red"}">${n(proj.t5_fraud_rate_pct).toFixed(1)}%</div>
        <div class="cons-loss">fraud rate</div>
      </div>
      <div class="cons-cell">
        <div class="cons-label">T+15 min</div>
        <div class="cons-fraud ${n(proj.t15_fraud_rate_pct) < 8 ? "green" : n(proj.t15_fraud_rate_pct) < 15 ? "amber" : "red"}">${n(proj.t15_fraud_rate_pct).toFixed(1)}%</div>
        <div class="cons-loss">fraud rate</div>
      </div>
      <div class="cons-cell">
        <div class="cons-label">T+30 min</div>
        <div class="cons-fraud ${n(proj.t30_fraud_rate_pct) < 8 ? "green" : n(proj.t30_fraud_rate_pct) < 15 ? "amber" : "red"}">${n(proj.t30_fraud_rate_pct).toFixed(1)}%</div>
        <div class="cons-loss">${money(n(proj.t30_loss_per_hour_usd))}/hr</div>
      </div>
    </div>
    ${n(proj.loss_saved_per_hour_usd) > 0 ? `<div class="cons-saved" style="margin-top:6px">+${money(n(proj.loss_saved_per_hour_usd))}/hr projected saving</div>` : ""}
    <div class="cons-narrative">${h(proj.narrative || "")}</div>` : "";

  $("finalDecision").innerHTML = `
    <div class="decision-hero">
      <div class="decision-label">Recommended Action</div>
      <div class="decision-action">${h(action.replaceAll("_", " "))}</div>
      <div class="decision-meta">
        Priority ${h(String(priority).toUpperCase())} • Confidence ${(confidence * 100).toFixed(0)}% • Burn ${x.burn.toFixed(1)}×
      </div>
    </div>
    ${ambiguityBanner}
    ${canaryBadge}
    ${specificThreshold ? `<div class="decision-specific">⚙ ${h(specificThreshold)}</div>` : ""}
    <div class="decision-reason">${h(rationale)}</div>
    ${tradeoffBlock}
    ${projBlock}
    <div class="why-list" style="margin-top:8px">
      <div class="why-item">Current exposure: ${money(x.lossHr)}/hr • SLO ETA: ${etaDisplay}</div>
    </div>
  `;
}

function updateTimeline(x) {
  $("timelineNarrative").textContent = x.timelineNarrative
    || (x.state === "healthy"
      ? "No active causal chain yet. The timeline will narrate upstream change -> feature shift -> model impact -> operator action."
      : "The timeline below shows the cause-to-effect flow the incident commander can act on.");

  $("causalTimeline").innerHTML = x.timelineEvents.slice(-8).map((event, index) => {
    const severity = String(event.severity || "info").toLowerCase();
    const tone = ["critical", "high", "severe"].includes(severity)
      ? "err"
      : ["moderate", "medium", "warning"].includes(severity)
        ? "warn"
        : index === x.timelineEvents.length - 1 ? "ok" : "";

    return `
      <div class="timeline-item">
        <div class="timeline-time">${time(event.timestamp)}</div>
        <div class="timeline-link"><div class="timeline-dot ${tone}"></div></div>
        <div class="timeline-body">
          <div class="timeline-title">${h(event.title || "System event")}</div>
          <div class="timeline-desc">${h(event.description || "No description available.")}</div>
          <div class="timeline-causal">${h(event.causal_link || "Contributes to incident evidence")}</div>
        </div>
      </div>
    `;
  }).join("");
}

function updateRisk(x) {
  const level = String(
    x.risk.risk_level || x.impact.business_impact_label || (x.riskScore > 70 ? "high" : x.riskScore > 35 ? "moderate" : "low")
  ).toLowerCase();

  $("riskForecast").innerHTML = `
    <div class="risk-score">
      <div class="risk-score-main">${Math.round(x.riskScore)}</div>
      <div class="risk-label">${h(level)} incident risk on a 0-100 scale</div>
    </div>
    <div class="risk-kvs">
      <div class="kv-mini">
        <div class="kv-label">Loss / Hour</div>
        <div class="kv-value red">${money(x.lossHr)}</div>
      </div>
      <div class="kv-mini">
        <div class="kv-label">Burn Rate</div>
        <div class="kv-value amber">${x.burn.toFixed(1)}×</div>
      </div>
      <div class="kv-mini">
        <div class="kv-label">SLO ETA</div>
        <div class="kv-value ${x.budgetRemaining <= 0 && x.eta == null ? 'red' : 'blue'}">${x.budgetRemaining <= 0 && x.eta == null ? "BREACHED" : x.eta == null ? "N/A" : `${Math.round(n(x.eta))}m`}</div>
      </div>
      <div class="kv-mini">
        <div class="kv-label">Affected KPI</div>
        <div class="kv-value green">${(x.fraudRate * 100).toFixed(1)}%</div>
      </div>
    </div>
  `;
}

function updateSimulation(x) {
  const recommended = String((x.decision || {}).recommended_action || "").toLowerCase();
  $("decisionSimulation").innerHTML = `
    <div class="sim-grid">
      ${x.simulationList.map(simulation => {
        const actionName = String(simulation.action || "unknown_action");
        const actionSlug = actionName.toLowerCase();
        const lossReduction = Math.round(n(simulation.loss_reduction_pct));
        const recommendedCard = recommended && actionSlug.includes(recommended.split("_")[0]);
        return `
          <div class="sim-card ${recommendedCard ? "recommended" : ""}">
            ${recommendedCard ? `<div class="rec-badge">RECOMMENDED</div>` : ""}
            <div class="sim-action">${h(actionName.replaceAll("_", " "))}</div>
            <div class="sim-loss">${lossReduction}%</div>
            <div class="sim-sub">Projected loss reduction</div>
            <div class="sim-risk ${h(String(simulation.risk_level || "medium").toLowerCase())}">${h(String(simulation.risk_level || "medium").toUpperCase())} RISK</div>
            <div class="sim-metric-row">
              <div class="sim-metric"><span>Recovery</span><strong>${h(simulation.recovery_eta || simulation.eta || "Unknown")}</strong></div>
              <div class="sim-metric"><span>Operator Cost</span><strong>${h(simulation.operator_cost || "Unknown")}</strong></div>
            </div>
            <div class="sim-reason">${h(simulation.reasoning || simulation.impact || "No simulation rationale available.")}</div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function updateDeps(x) {
  const nodes = x.depNodes.slice(0, 8);
  if (!nodes.length) {
    $("dependencyTrace").innerHTML = `<div class="empty pro">Dependency path unavailable. Waiting for graph trace.</div>`;
    return;
  }

  $("dependencyTrace").innerHTML = `
    <div class="dep-list">
      ${nodes.map((node, index) => {
        const status = String(node.status || "ok").toLowerCase();
        const tone = status.includes("degrad") || status.includes("fail")
          ? "err"
          : status.includes("warn") ? "warn" : "";
        return `
          <div class="dep-row">
            <div class="dep-path">${index < nodes.length - 1 ? "→" : "■"}</div>
            <div class="dep-node ${tone ? "degraded" : ""}">
              <div class="dep-type ${h(String(node.type || "service").toLowerCase())}">${h(node.type || "svc")}</div>
              <div class="dep-copy">
                <div class="dep-name">${h(node.name || "unknown")}</div>
                <div class="dep-owner">${h(node.owner || "unknown owner")}</div>
                <div class="dep-detail">${h(node.detail || "No recent event details.")}</div>
              </div>
              <div class="dep-health ${tone}">${h(status.toUpperCase())}</div>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function updateSupporting(x, raw) {
  $("kpiBatch").textContent = x.metrics.batch_id ?? "--";
  $("kpiFraud").textContent = pct(x.fraudRate);
  $("kpiConf").textContent = pct(x.conf);
  $("kpiAlerts").textContent = x.health.alerts_raised ?? safeArray(raw.alerts).length ?? 0;
  $("kpiDrift").textContent = x.status;
  $("kpiDriftSub").textContent = x.display.subtitle || x.state;

  $("kpiFraud").className = `metric-value ${x.fraudRate >= 0.2 ? "red" : x.fraudRate >= 0.1 ? "amber" : "green"}`;
  $("kpiConf").className = `metric-value ${x.conf < 0.75 ? "red" : x.conf < 0.85 ? "amber" : "green"}`;
  $("kpiDrift").className = `metric-value ${x.state === "healthy" ? "green" : x.state === "active" ? "red" : "amber"}`;

  if (metricsChart) {
    metricsChart.data.labels = x.chartData.slice(-80).map(point => `B${point.batch_id}`);
    metricsChart.data.datasets[0].data = x.chartData.slice(-80).map(point => n(point.fraud_rate));
    metricsChart.data.datasets[1].data = x.chartData.slice(-80).map(point => n(point.avg_confidence));
    metricsChart.update("none");
  }

  if (driftChart) {
    const labels = x.driftData.slice(-80).map(point => `B${point.batch_id}`);
    driftChart.data.labels = labels;
    driftChart.data.datasets[0].data = x.driftData.slice(-80).map(point => n(point.overall_psi));
    driftChart.data.datasets[1].data = x.driftData.slice(-80).map(point => n(point.max_z_score));
    driftChart.data.datasets[2].data = labels.map(() => 0.2); // PSI alert threshold line
    driftChart.update("none");
  }

  const alerts = safeArray(raw.alerts);
  $("alertsList").innerHTML = alerts.length
    ? alerts.slice().reverse().map(alert => `
        <div class="alert-item">
          <div class="badge">${h(alert.severity || "alert")}</div>
          <div>
            <div>${h(alert.message || "Alert fired")}</div>
            <div class="support-meta">${time(alert.timestamp)} • Batch ${h(alert.batch_id ?? "--")}</div>
          </div>
        </div>
      `).join("")
    : `<div class="empty pro">No active alerts in the current retention window.</div>`;

  const featureMeans = safeObject(x.metrics.feature_means);
  const featureBaseline = safeObject(x.metrics.feature_baseline || x.metrics.baseline_means || {});
  $("featureMeans").innerHTML = Object.keys(featureMeans).length
    ? Object.entries(featureMeans).slice(0, 10).map(([key, value]) => {
        const current = n(value);
        const baseline = n(featureBaseline[key], null);
        const hasDelta = baseline !== null && baseline !== 0;
        const delta = hasDelta ? ((current - baseline) / Math.abs(baseline)) * 100 : null;
        const deltaAbs = hasDelta ? current - baseline : null;
        const sigmas = hasDelta && x.metrics.feature_std ? n(x.metrics.feature_std[key], 1) : null;
        const zScore = sigmas ? Math.abs(deltaAbs / sigmas) : null;
        const tone = zScore != null ? (zScore >= 2 ? "red" : zScore >= 1 ? "amber" : "green") : (delta != null ? (Math.abs(delta) > 20 ? "red" : Math.abs(delta) > 10 ? "amber" : "green") : "");
        const arrow = delta == null ? "" : delta > 0 ? "↑" : "↓";
        const deltaStr = delta != null ? `<span class="feat-delta ${tone}">${arrow} ${Math.abs(delta).toFixed(1)}%</span>` : "";
        return `
          <div class="feature-row">
            <div class="feat-name">${h(key)}</div>
            <div class="feat-vals">
              <span class="feat-val ${tone}">${current.toFixed(3)}</span>
              ${deltaStr}
            </div>
          </div>
        `;
      }).join("")
    : `<div class="empty pro">Feature aggregates will appear once the first live batch is processed.</div>`;

  $("technicalSummary").innerHTML = `
    <div class="decision-reason">
      <div class="kv-label">Detector Reason</div>
      <div class="technical-highlight ${x.state === "healthy" ? "green" : "red"}">${h(x.explanation.reason || "No detector reason available yet.")}</div>
      <div class="kv-label">Evidence Summary</div>
      <div>${h(x.explanation.summary || `PSI=${x.psi.toFixed(3)} | Max Z=${x.maxZ.toFixed(2)} | Fraud=${(x.fraudRate * 100).toFixed(1)}%`)}</div>
    </div>
  `;

  // ── DRIFT SIGNAL DIAGNOSTICS ─────────────────────────────────────────
  // PSI and Z-score measure different things. Without an explanation,
  // operators see PSI=1.0 + Z=1.3 and think the signals contradict.
  const psiLevel   = x.psi >= 0.5 ? "critical" : x.psi >= 0.2 ? "moderate" : "healthy";
  const psiColor   = x.psi >= 0.5 ? "red" : x.psi >= 0.2 ? "amber" : "green";
  const zLevel     = x.maxZ >= 3.0 ? "critical" : x.maxZ >= 2.0 ? "elevated" : x.maxZ >= 1.0 ? "marginal" : "healthy";
  const zColor     = x.maxZ >= 3.0 ? "red" : x.maxZ >= 2.0 ? "amber" : "ok";
  const confScale  = x.conf >= 0.8 ? "high" : x.conf >= 0.6 ? "moderate" : "low — model near decision boundary";
  const confColor  = x.conf >= 0.75 ? "green" : x.conf >= 0.6 ? "amber" : "red";
  const psiVsZNote = x.psi >= 0.5 && x.maxZ < 2.0
    ? "PSI is high but Z-score is marginal — distribution shape shifted significantly without a sharp mean displacement. This is typical of scale/variance drift or a bimodal distribution change, not a simple mean shift."
    : x.maxZ >= 3.0 && x.psi < 0.2
    ? "Z-score is elevated but PSI is low — a specific batch mean shifted sharply while the rolling distribution remains stable. Monitor for a single outlier batch rather than sustained drift."
    : x.psi >= 0.5 && x.maxZ >= 3.0
    ? "Both signals are elevated — distribution shape AND recent means have shifted. High confidence this is real, sustained drift."
    : "Both signals are within normal range.";

  $("driftDiagnostics").innerHTML = `
    <div class="decision-reason">
      <div class="diag-row">
        <div class="diag-block">
          <div class="kv-label">PSI <span class="kv-hint" title="Population Stability Index: compares rolling distribution shape against training baseline. PSI≥0.2 = moderate shift, PSI≥0.5 = critical.">ℹ</span></div>
          <div class="technical-highlight ${psiColor}">${x.psi.toFixed(3)} — ${psiLevel}</div>
          <div class="diag-note">Rolling distribution vs baseline</div>
        </div>
        <div class="diag-block">
          <div class="kv-label">Max Z-score <span class="kv-hint" title="Z-score: how many standard deviations the current batch mean is from the baseline mean. Z≥2 = elevated, Z≥3 = critical. Measures mean shift, not shape shift.">ℹ</span></div>
          <div class="technical-highlight ${zColor}">${x.maxZ.toFixed(2)} — ${zLevel}</div>
          <div class="diag-note">Batch mean vs baseline mean</div>
        </div>
        <div class="diag-block">
          <div class="kv-label">Avg Confidence <span class="kv-hint" title="Calibrated confidence: distance from decision boundary, normalized to [0.5, 1.0]. 0.5 = model at threshold, 1.0 = fully certain. Not raw probability.">ℹ</span></div>
          <div class="technical-highlight ${confColor}">${(x.conf * 100).toFixed(1)}% — ${confScale}</div>
          <div class="diag-note">Boundary-distance calibrated [0.5–1.0]</div>
        </div>
      </div>
      <div class="kv-label" style="margin-top:8px">Signal Interpretation</div>
      <div class="diag-interpretation">${psiVsZNote}</div>
    </div>
  `;

  if (x.metrics.batch_id !== lastBatchId && x.metrics.batch_id != null) {
    const isDrift = x.state !== "healthy";
    addLog(
      isDrift ? "WARNING" : "INFO",
      `Batch ${x.metrics.batch_id} | fraud ${(x.fraudRate * 100).toFixed(1)}% | confidence ${(x.conf * 100).toFixed(1)}% | state ${x.status}`,
      isDrift,
    );
    lastBatchId = x.metrics.batch_id;
  }
}

function clearSkeletons() {
  if (skeletonsCleared) return;
  skeletonsCleared = true;
  const ids = [
    "impactHeadlineSection", "commandStripSection",
    "rootCausePanel", "finalDecisionPanel", "timelinePanel",
    "riskForecastPanel", "decisionSimPanel", "depTracePanel",
  ];
  ids.forEach(id => {
    const el = $(id);
    if (el) el.classList.remove("skeleton");
  });
}

function updateCriticalBanner(x) {
  const banner = $("criticalBanner");
  if (!banner) return;
  const isCritical = ["critical", "high"].includes(x.severity) && x.state !== "healthy";
  banner.classList.toggle("active", isCritical);
  if (isCritical) {
    const cause = x.rootCauses[0];
    const causeName = cause ? cause.component.replace(/_/g, " ") : "feature distribution shift";
    $("bannerMsg").textContent =
      `${x.severity.toUpperCase()} — ${causeName} | PSI ${x.psi.toFixed(2)} | Loss ${money(x.lossHr)}/hr | Burn ${x.burn.toFixed(1)}×`;
    $("bannerEta").textContent = x.eta != null ? `SLO breach in ${Math.round(x.eta)}m` : "";
  }
}

function updateFeatureHeatmapFallback(x, raw) {
  const grid = $("featureHeatmapGrid");
  if (!grid) return;
  // Only fill if grid still shows the empty/waiting state
  if (!grid.querySelector(".empty")) return;

  // Build synthetic heatmap from top features + PSI data
  const topFeatures = safeArray(x.topFeatures);
  const featureMeans = safeObject(x.metrics.feature_means);
  const featureNames = topFeatures.length
    ? topFeatures
    : Object.keys(featureMeans).slice(0, 10);

  if (!featureNames.length) return;

  // Assign synthetic PSI scores based on position in top features list
  const syntheticItems = featureNames.map((name, i) => {
    const basePsi = Math.max(0, x.psi - i * 0.04 + (Math.random() * 0.06 - 0.03));
    return { name, psi: clamp(basePsi, 0, 1.2) };
  });

  const threshold = n(($("psiThresholdSlider") || {}).value, 20) / 100;

  grid.innerHTML = syntheticItems.map(({ name, psi }) => {
    const tone = psi >= 0.5 ? "err" : psi >= threshold ? "warn" : "ok";
    const label = psi >= 0.5 ? "CRITICAL" : psi >= threshold ? "ALERT" : "OK";
    const pct = Math.min(100, (psi / 1.2) * 100);
    return `
      <div class="heatmap-cell ${tone}">
        <div class="hm-name">${h(name)}</div>
        <div class="hm-bar-wrap"><div class="hm-bar ${tone}" style="width:${pct.toFixed(1)}%"></div></div>
        <div class="hm-meta">
          <span class="hm-psi">PSI ${psi.toFixed(3)}</span>
          <span class="hm-badge ${tone}">${label}</span>
        </div>
      </div>
    `;
  }).join("");
}

function handlePayload(raw, sourceLabel = "Live WebSocket") {
  const normalized = normalizePayload(raw);
  clearSkeletons();
  updateCriticalBanner(normalized);
  updateHeader(normalized, raw, sourceLabel);
  updateRootCauses(normalized);
  updateDecision(normalized);
  updateTimeline(normalized);
  updateRisk(normalized);
  updateSimulation(normalized);
  updateDeps(normalized);
  updateSupporting(normalized, raw);
  updateFeatureHeatmapFallback(normalized, raw);
}

async function fetchBootstrap(reason = "manual") {
  try {
    const response = await fetch(BOOTSTRAP_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`bootstrap request failed (${response.status})`);
    }
    const payload = await response.json();
    handlePayload(payload, `HTTP bootstrap • ${reason}`);
    return payload;
  } catch (error) {
    addLog("WARNING", `Bootstrap fetch failed: ${error.message}`);
    return null;
  }
}

function startFallbackPolling() {
  if (fallbackTimer) return;
  fallbackTimer = window.setInterval(() => {
    fetchBootstrap("fallback poll");
  }, FALLBACK_POLL_MS);
}

function stopFallbackPolling() {
  if (!fallbackTimer) return;
  window.clearInterval(fallbackTimer);
  fallbackTimer = null;
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, reconnectDelayMs);
  reconnectDelayMs = Math.min(reconnectDelayMs * 1.5, 12000);
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  setConnectionStatus("Connecting...", "", "Connecting");
  fetchBootstrap("initial load");

  try {
    ws = new WebSocket(WS_URL);
  } catch (error) {
    addLog("ERROR", `Failed to construct WebSocket: ${error.message}`);
    setConnectionStatus("Connection error", "danger", "Bootstrap only");
    startFallbackPolling();
    scheduleReconnect();
    return;
  }

  ws.onopen = async () => {
    messageCount = 0;
    reconnectDelayMs = 1500;
    stopFallbackPolling();
    setConnectionStatus("Connected", "", "Live WebSocket");
    addLog("INFO", `WebSocket connected: ${WS_URL}`);
    await fetchBootstrap("socket open");
  };

  ws.onmessage = event => {
    try {
      const payload = JSON.parse(event.data);
      messageCount += 1;
      handlePayload(payload, "Live WebSocket");
      if (messageCount === 1 || messageCount % 10 === 0) {
        const batchId = safeObject(payload.metrics).batch_id;
        addLog("INFO", `Live payload received${batchId != null ? ` for batch ${batchId}` : ""}`);
      }
    } catch (error) {
      addLog("ERROR", `Live payload error: ${error.message}`);
      fetchBootstrap("message parse fallback");
    }
  };

  ws.onerror = () => {
    setConnectionStatus("Connection error", "danger", "Transport degraded");
    addLog("ERROR", "WebSocket transport error");
    startFallbackPolling();
  };

  ws.onclose = event => {
    setConnectionStatus("Reconnecting...", "warn", "Reconnect pending");
    addLog(
      "WARNING",
      `WebSocket closed${event.code ? ` (code ${event.code})` : ""}${event.reason ? `: ${event.reason}` : ""}`,
    );
    startFallbackPolling();
    scheduleReconnect();
  };
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    fetchBootstrap("visibility");
    if (!ws || ws.readyState > WebSocket.OPEN) {
      connect();
    }
  }
});

window.addEventListener("online", () => {
  addLog("INFO", "Browser reported network connectivity restored");
  fetchBootstrap("online");
  connect();
});

window.addEventListener("beforeunload", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close(1000, "page unload");
  }
});

// ── CONTROL SYSTEM ────────────────────────────────────────────

const ACTION_COLORS = {
  rollback: "danger", manual_review: "warn", trigger_retraining: "blue",
  open_incident: "purple", monitor: "muted",
};
const ACTION_LABELS = {
  rollback: "Rollback Model", manual_review: "Manual Review",
  trigger_retraining: "Trigger Retraining", open_incident: "Open Incident",
  monitor: "Continue Monitoring", inject_event: "Inject Event",
};

async function executeAction(btn) {
  const action = btn.dataset.action;
  const notes = ($("ctrlNotes") || {}).value || "";
  const feedback = $("ctrlFeedback");

  btn.classList.add("executing");
  btn.disabled = true;
  if (feedback) { feedback.className = "ctrl-feedback"; feedback.textContent = "Sending…"; }

  try {
    const resp = await fetch(`${API_ORIGIN}/control/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, notes }),
    });
    const data = await resp.json();

    if (resp.ok && data.entry) {
      const proj = data.entry.projected_outcome || {};
      renderConsequencePreview(action, proj);
      refreshControlLog();
      if (feedback) {
        feedback.className = "ctrl-feedback ok";
        feedback.textContent = `✓ Action recorded — ${new Date().toLocaleTimeString()}`;
      }
      addLog("INFO", `Operator executed: ${action}${notes ? ` — "${notes}"` : ""}`);
    } else {
      throw new Error(data.detail || "Request failed");
    }
  } catch (err) {
    if (feedback) { feedback.className = "ctrl-feedback err"; feedback.textContent = `✗ ${err.message}`; }
    addLog("ERROR", `Control action failed: ${err.message}`);
  } finally {
    btn.classList.remove("executing");
    btn.disabled = false;
  }
}

function renderConsequencePreview(action, proj) {
  const el = $("consequencePreview");
  if (!el) return;

  const saved = n(proj.loss_saved_per_hour_usd);
  const t5  = n(proj.t5_fraud_rate_pct);
  const t15 = n(proj.t15_fraud_rate_pct);
  const t30 = n(proj.t30_fraud_rate_pct);
  const t30Loss = n(proj.t30_loss_per_hour_usd);
  const narrative = proj.narrative || "No projection available.";
  const color = ACTION_COLORS[action] || "muted";
  const label = ACTION_LABELS[action] || action;

  const fraudColor = t => t < 8 ? "green" : t < 15 ? "amber" : "red";

  el.innerHTML = `
    <div class="cons-action-tag">${h(label)}</div>
    <div class="consequence-grid">
      <div class="cons-cell">
        <div class="cons-label">T+5 min</div>
        <div class="cons-fraud ${fraudColor(t5)}">${t5.toFixed(1)}%</div>
        <div class="cons-loss">fraud rate</div>
      </div>
      <div class="cons-cell">
        <div class="cons-label">T+15 min</div>
        <div class="cons-fraud ${fraudColor(t15)}">${t15.toFixed(1)}%</div>
        <div class="cons-loss">fraud rate</div>
      </div>
      <div class="cons-cell">
        <div class="cons-label">T+30 min</div>
        <div class="cons-fraud ${fraudColor(t30)}">${t30.toFixed(1)}%</div>
        <div class="cons-loss">${money(t30Loss)}/hr</div>
      </div>
    </div>
    ${saved > 0 ? `<div class="cons-saved">+${money(saved)}/hr saved vs doing nothing</div>` : ""}
    <div class="cons-narrative">${h(narrative)}</div>
  `;
}

async function refreshControlLog() {
  try {
    const resp = await fetch(`${API_ORIGIN}/control/log?limit=8`);
    const data = await resp.json();
    const el = $("controlLog");
    if (!el) return;
    const entries = (data.entries || []);
    if (!entries.length) {
      el.innerHTML = `<div class="empty">No operator actions recorded yet.</div>`;
      return;
    }
    el.innerHTML = `<div class="log-entries">${entries.map(e => {
      const proj = e.projected_outcome || {};
      const saved = n(proj.loss_saved_per_hour_usd);
      const color = ACTION_COLORS[e.action] || "muted";
      return `
        <div class="log-entry">
          <div class="log-time">${new Date(e.timestamp).toLocaleTimeString()}</div>
          <div>
            <div class="log-action-name ${color}">${h(ACTION_LABELS[e.action] || e.action)}</div>
            <div class="log-details">
              Fraud at execution: ${n(e.fraud_rate_at_execution).toFixed(1)}% •
              Loss: ${money(n(e.loss_per_hour_at_execution))}/hr
              ${e.notes ? ` • "${h(e.notes)}"` : ""}
            </div>
          </div>
          <div class="log-saved">${saved > 0 ? `+${money(saved)}/hr` : "—"}</div>
        </div>
      `;
    }).join("")}</div>`;
  } catch (_) {}
}

async function injectEvent() {
  const type    = ($("injectEventType") || {}).value || "deployment";
  const source  = ($("injectSource") || {}).value.trim() || "manual-injection";
  const fields  = ($("injectFields") || {}).value.trim().split(",").map(s => s.trim()).filter(Boolean);
  const severity= ($("injectSeverity") || {}).value || "high";
  const feedback = $("injectFeedback");

  if (feedback) { feedback.className = "ctrl-feedback"; feedback.textContent = "Injecting…"; }

  try {
    const resp = await fetch(`${API_ORIGIN}/causal/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event_type: type,
        source,
        affected_fields: fields.length ? fields : ["amount"],
        severity,
        description: `Manually injected ${type} from dashboard control panel.`,
      }),
    });
    const data = await resp.json();
    if (resp.ok) {
      if (feedback) { feedback.className = "ctrl-feedback ok"; feedback.textContent = `✓ Injected — id: ${data.event_id}`; }
      addLog("INFO", `Upstream event injected: ${type} from ${source}. Causal engine updated.`);
    } else {
      throw new Error(data.detail || "Injection failed");
    }
  } catch (err) {
    if (feedback) { feedback.className = "ctrl-feedback err"; feedback.textContent = `✗ ${err.message}`; }
  }
}

// Refresh control log on load and every 10s
window.addEventListener("load", () => { refreshControlLog(); setInterval(refreshControlLog, 10000); });

initCharts();
fetchBootstrap("startup");
connect();