/**
 * viewer.js — Main application orchestrator
 * Wires Parser → Engine → Charts → UI. Zero hardcoded column names.
 */
(function (global) {
  'use strict';

  // ── Safe value helper — never show NaN / undefined ────────────────────
  function sv(v, fallback, digits) {
    if (v == null || (typeof v === 'number' && !isFinite(v))) return fallback != null ? fallback : '—';
    if (typeof v === 'number' && digits != null) return v.toFixed(digits);
    return v;
  }

  const $ = id => document.getElementById(id);
  const fmtMoney = v => '$' + Math.round(+v || 0).toLocaleString();
  // Auto-precision: 3dp for tiny rates (< 1%), 1dp otherwise — never shows "0.0%" for real signal
  const fmtPct = (v, d) => {
    const n = +v;
    if (!isFinite(n)) return '—';
    if (d != null) return n.toFixed(d) + '%';
    return n.toFixed(Math.abs(n) < 1 && n !== 0 ? 3 : 1) + '%';
  };
  const fmtPSI = v => sv(+v, '—', 3);
  const fmtZ = v => sv(+v, '—', 2);
  const nowTime = () => new Date().toLocaleTimeString();

  // ── Chart references ──────────────────────────────────────────────────
  let metricsChart = null, driftChart = null, lossChart = null, drilldownChart = null;
  const CHART_MAX_POINTS = 80;
  let alertsList = [];
  let injectedEvents = [];
  let lastDrilldownFeature = null;
  let lastCurrentValsMap = {}; // real current-batch means for heatmap tooltip
  let lastHeatmapBatch = -1;   // throttle — only rebuild heatmap when batch advances

  // ── Chart init ────────────────────────────────────────────────────────
  function initCharts() {
    Chart.defaults.font.family = '"Avenir Next","Trebuchet MS",sans-serif';
    Chart.defaults.color = '#9aa8bc';
    Chart.defaults.borderColor = 'rgba(140,162,194,0.14)';

    const D = global.DATASET;
    const targetLabel = D ? D.targetName + ' Rate %' : 'Event Rate %';

    const grad = (ctx, color) => {
      const g = ctx.createLinearGradient(0, 0, 0, 220);
      g.addColorStop(0, color + '55'); g.addColorStop(1, color + '00');
      return g;
    };

    // Metrics chart: fraud rate + loss
    const mc = $('metricsChart').getContext('2d');
    metricsChart = new Chart(mc, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: targetLabel, data: [], borderColor: '#3fe0a1', backgroundColor: grad(mc, '#3fe0a1'), borderWidth: 2.2, tension: 0.32, fill: true, pointRadius: 0, pointHoverRadius: 4 },
          { label: 'Loss/hr ($k)', data: [], borderColor: '#ff6c78', backgroundColor: grad(mc, '#ff6c78'), borderWidth: 2.2, tension: 0.32, fill: true, pointRadius: 0, pointHoverRadius: 4, yAxisID: 'loss' },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: '#90a1b9', boxWidth: 14, usePointStyle: true } }, tooltip: { backgroundColor: 'rgba(8,13,22,0.96)', borderColor: 'rgba(121,151,201,0.18)', borderWidth: 1, padding: 12 } },
        scales: {
          x: { ticks: { color: '#51627a', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { beginAtZero: true, ticks: { color: '#51627a' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          loss: { type: 'linear', position: 'right', beginAtZero: true, ticks: { color: '#ff9aa2', callback: v => '$' + v + 'k' }, grid: { drawOnChartArea: false } },
        },
      },
    });

    // Drift chart: PSI + max Z
    const dc = $('driftChart').getContext('2d');
    driftChart = new Chart(dc, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Overall PSI', data: [], borderColor: '#ff6d78', backgroundColor: grad(dc, '#ff6d78'), borderWidth: 2.2, tension: 0.28, fill: true, pointRadius: 0, pointHoverRadius: 4, yAxisID: 'psi' },
          { label: 'Max Z-Score', data: [], borderColor: '#ffc56a', backgroundColor: grad(dc, '#ffc56a'), borderWidth: 2.2, tension: 0.28, fill: true, pointRadius: 0, pointHoverRadius: 4, yAxisID: 'z' },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: '#90a1b9', boxWidth: 14, usePointStyle: true } }, tooltip: { backgroundColor: 'rgba(8,13,22,0.96)', borderColor: 'rgba(121,151,201,0.18)', borderWidth: 1, padding: 12 } },
        scales: {
          x: { ticks: { color: '#51627a', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.04)' } },
          psi: { type: 'linear', position: 'left', min: 0, max: 0.6, ticks: { color: '#ff9aa2' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          z: { type: 'linear', position: 'right', beginAtZero: true, ticks: { color: '#ffd08d' }, grid: { drawOnChartArea: false } },
        },
      },
    });

    // Loss trajectory chart (3 scenarios)
    const lc = $('lossChart').getContext('2d');
    lossChart = new Chart(lc, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Ignore (no action)', data: [], borderColor: '#ff6c78', backgroundColor: 'rgba(255,108,120,0.08)', borderWidth: 2, tension: 0.3, fill: true, pointRadius: 0 },
          { label: 'Rollback model', data: [], borderColor: '#3fe0a1', backgroundColor: 'rgba(63,224,161,0.05)', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 0 },
          { label: 'Increase review', data: [], borderColor: '#ffc46b', backgroundColor: 'rgba(255,196,107,0.05)', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 0 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#90a1b9', boxWidth: 14, usePointStyle: true } },
          tooltip: { backgroundColor: 'rgba(8,13,22,0.96)', borderColor: 'rgba(121,151,201,0.18)', borderWidth: 1, padding: 10,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${fmtMoney(ctx.raw)}` },
          },
          annotation: {},
        },
        scales: {
          x: { ticks: { color: '#51627a', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Batches from now', color: '#51627a' } },
          y: { beginAtZero: true, ticks: { color: '#51627a', callback: v => fmtMoney(v) }, grid: { color: 'rgba(255,255,255,0.05)' } },
        },
      },
    });

    updateChartLabels();
  }

  function updateChartLabels() {
    const D = global.DATASET;
    if (!D || !metricsChart) return;
    metricsChart.data.datasets[0].label = D.targetName + ' Rate %';
    metricsChart.update('none');
  }

  // ── All chart updates — single pass ───────────────────────────────────
  function updateAllCharts(tickData) {
    const SIM = Engine.SIM;
    const D = global.DATASET;
    if (!D || !metricsChart) return;

    const hist = SIM.metrics;
    const n = hist.batchIds.length;
    const labels = hist.batchIds.slice(-CHART_MAX_POINTS).map(b => 'B' + b);

    // Metrics chart
    const fraudSlice = hist.fraudRateHistory.slice(-CHART_MAX_POINTS);
    const lossSlice = hist.lossHistory.slice(-CHART_MAX_POINTS).map(v => +(v / 1000).toFixed(2));
    metricsChart.data.labels = labels;
    metricsChart.data.datasets[0].data = fraudSlice;
    metricsChart.data.datasets[1].data = lossSlice;
    metricsChart.update('none');

    // Drift chart: overall PSI (max across features) + max Z
    const psiHistory = [];
    const zHistory = [];
    const idxSlice = hist.batchIds.slice(-CHART_MAX_POINTS);
    idxSlice.forEach((_, i) => {
      const absI = Math.max(0, n - CHART_MAX_POINTS) + i;
      let maxP = 0, maxZ = 0;
      D.numericCols.forEach(col => {
        const pv = (hist.psiPerFeature[col] || [])[absI];
        const zv = (hist.zscorePerFeature[col] || [])[absI];
        if (pv != null && pv > maxP) maxP = pv;
        if (zv != null && zv > maxZ) maxZ = zv;
      });
      psiHistory.push(+maxP.toFixed(4));
      zHistory.push(+maxZ.toFixed(3));
    });
    driftChart.data.labels = labels;
    driftChart.data.datasets[0].data = psiHistory;
    driftChart.data.datasets[1].data = zHistory;
    driftChart.update('none');

    // Loss trajectory chart
    const traj = Engine.buildLossTrajectory(30);
    if (traj && lossChart) {
      lossChart.data.labels = traj.labels;
      lossChart.data.datasets[0].data = traj.ignoreLine;
      lossChart.data.datasets[1].data = traj.rollbackLine;
      lossChart.data.datasets[2].data = traj.reviewLine;
      // Highlight active action — always reset to canonical colors to prevent accumulation
      const activeAction = SIM.actionState.active;
      const LOSS_COLORS = ['#ff6c78', '#3fe0a1', '#ffc46b'];
      const ACTION_MAP  = { 0: 'monitor', 1: 'rollback', 2: 'increase_review' };
      lossChart.data.datasets.forEach((ds, i) => {
        const isActive = ACTION_MAP[i] === activeAction;
        ds.borderWidth = isActive ? 3.5 : 1.5;
        ds.borderColor = activeAction ? (isActive ? LOSS_COLORS[i] : LOSS_COLORS[i] + '44') : LOSS_COLORS[i];
      });
      lossChart.update('none');
    }

    // Update drilldown if open
    if (lastDrilldownFeature) updateDrilldown(lastDrilldownFeature);
  }

  // ── Hero metrics ──────────────────────────────────────────────────────
  function updateHeroMetrics(tickData) {
    const D = global.DATASET;
    const SIM = Engine.SIM;
    const { batchId, displayFraudRate, lossPerHour, maxPSI, drifting, topDriftedFeatures, incidentActive } = tickData;

    const riskScore = Engine.computeRiskScore(maxPSI, displayFraudRate, 0, 0);
    const aliases = D ? D.featureAliases || {} : {};

    // Batch counter — absolute (never resets) with replay suffix after first wrap
    const absB = tickData.absoluteBatch || batchId;
    const replayLabel = tickData.replayCount >= 1 ? ` · Replay ${tickData.replayCount}` : '';
    $('batchCounter').textContent = `${absB}${replayLabel}`;
    $('batchProgressLabel').textContent = `Batch ${absB}${replayLabel}`;
    const pct = SIM.batches.length > 0 ? (SIM.currentIdx / SIM.batches.length) * 100 : 0;
    $('batchBarFill').style.width = pct + '%';

    // Status
    const state = incidentActive ? 'active' : (drifting ? 'warning' : 'healthy');
    const tone = incidentActive ? 'err' : (drifting ? 'warn' : '');
    $('stateLed').className = 'state-led ' + tone;
    $('stateLabel').className = 'state-label ' + tone;
    $('stateLabel').textContent = incidentActive ? 'INCIDENT' : (drifting ? 'DRIFTING' : 'STABLE');
    $('stateSub').textContent = incidentActive ? 'Causal investigation active' : (drifting ? 'Distribution shift detected' : 'System healthy');

    $('statusPill').className = 'pill ' + (incidentActive ? 'danger' : (drifting ? 'warn' : ''));
    $('statusText').textContent = incidentActive ? 'INCIDENT ACTIVE' : (drifting ? 'DRIFT DETECTED' : 'Monitoring');

    // Strip values
    $('heroRiskScore').textContent = sv(riskScore, '—');
    $('heroRiskScore').className = 'strip-value ' + (riskScore > 70 ? 'err' : riskScore > 35 ? 'warn' : 'ok');

    $('heroLoss').textContent = fmtMoney(lossPerHour);
    $('heroLoss').className = 'strip-value ' + (lossPerHour > 5000 ? 'err' : lossPerHour > 1000 ? 'warn' : 'ok');

    $('heroFraud').textContent = fmtPct(displayFraudRate * 100);
    $('heroFraud').className = 'strip-value ' + (displayFraudRate > 0.1 ? 'err' : displayFraudRate > 0.03 ? 'warn' : 'ok');

    $('heroMaxPSI').textContent = fmtPSI(maxPSI);
    $('heroMaxPSI').className = 'strip-value ' + (maxPSI > 0.25 ? 'err' : maxPSI > 0.1 ? 'warn' : 'ok');

    // Strip narrative
    const topF = topDriftedFeatures[0];
    const topAlias = topF ? (aliases[topF] || topF) : null;
    $('stripNarrative').textContent = incidentActive
      ? `Incident active — ${topAlias ? topAlias + ' shows highest drift (PSI=' + fmtPSI(tickData.psiMap[topF]) + ')' : 'investigating root cause'}. Recommended: check causal hypotheses.`
      : (drifting ? `Drift detected — ${topAlias || 'features'} shifting above threshold. Monitor or act using the control panel below.`
        : 'No active incidents. All features within SLO bounds. Watching for distribution shift.');

    // Critical banner + incident-active class for Loss pulse (Bug 5)
    // Severity is driven by riskScore, not just incidentActive boolean:
    //   riskScore >= 70 → red  + "Incident Active"
    //   riskScore < 70  → amber + "Drift Detected"
    //   riskScore < 30  → amber + "Drift Detected" (same, just lower intensity implied by score)
    const wrapper = $('mainWrapper');
    const showBanner = incidentActive || drifting;
    if (showBanner) {
      const banner = $('criticalBanner');
      banner.style.display = 'flex';

      // Severity class
      const isRedSeverity = riskScore >= 70;
      banner.className = 'critical-banner' + (isRedSeverity ? '' : ' banner-amber');

      // Title — reserve "Incident Active" for risk >= 70
      $('bannerTitle').textContent = isRedSeverity ? 'Incident Active' : 'Drift Detected';

      // Technical detail line (unchanged — PSI, attribution, fraud rate, loss)
      const topCount = topDriftedFeatures.filter(f => (tickData.psiMap[f] || 0) > Engine.SIM.driftThreshold).length;
      $('bannerMsg').textContent = `${topCount} feature${topCount !== 1 ? 's' : ''} drifting — ${topAlias || 'features'} leading (PSI=${fmtPSI(maxPSI)}) · ${D ? D.targetName : 'Event'} rate ${fmtPct(displayFraudRate * 100)} · Loss ${fmtMoney(lossPerHour)}/hr`;
      $('bannerEta').textContent = `Risk: ${riskScore}/100`;

      // Recommended action secondary line — shown only when decision confidence > 70%
      const topHyp = Engine.SIM.hypotheses && Engine.SIM.hypotheses[0];
      const decisionConf = topHyp ? (topHyp.confidence || 0) : 0;
      const bannerAction = $('bannerAction');
      if (bannerAction) {
        if (decisionConf > 0.70 && lossPerHour > 0) {
          // Mirror decision panel action selection
          let recActionKey = 'monitor';
          if (incidentActive && riskScore >= 70)   recActionKey = 'rollback';
          else if (drifting   && riskScore >= 40)   recActionKey = 'increase_review';
          else if (drifting)                         recActionKey = 'open_incident';
          const proj = Engine.projectOutcome(recActionKey, displayFraudRate, lossPerHour);
          const savedPerHr = Math.max(0, Math.round(lossPerHour - proj.t15LossPerHour));
          const recLabel = recActionKey.replace(/_/g, ' ');
          bannerAction.textContent = savedPerHr > 0
            ? `Recommended: ${recLabel} — saves ${fmtMoney(savedPerHr)}/hr`
            : `Recommended: ${recLabel}`;
          bannerAction.style.display = 'block';
        } else {
          bannerAction.style.display = 'none';
        }
      }

      if (wrapper) wrapper.classList.add('incident-active');
    } else {
      $('criticalBanner').style.display = 'none';
      if (wrapper) wrapper.classList.remove('incident-active');
    }

    // KPI cards
    $('kpiBatch').textContent = absB;
    $('kpiBatchSub').textContent = tickData.replayCount >= 1 ? `Replay ${tickData.replayCount}` : 'live';
    $('kpiFraud').textContent = fmtPct(displayFraudRate * 100);
    $('kpiFraud').className = 'metric-value ' + (displayFraudRate > 0.1 ? 'red' : displayFraudRate > 0.03 ? 'amber' : 'green');
    $('kpiPSI').textContent = fmtPSI(maxPSI);
    $('kpiPSI').className = 'metric-value ' + (maxPSI > 0.25 ? 'red' : maxPSI > 0.1 ? 'amber' : 'green');

    if (topF) {
      $('kpiTopFeature').textContent = aliases[topF] || topF;
      $('kpiTopPSI').textContent = 'PSI=' + fmtPSI(tickData.psiMap[topF]);
    }

    // Hero labels use dataset target name
    if (D) {
      $('kpiFraudLabel').textContent = D.targetName + ' Rate';
      $('heroFraudLabel').textContent = D.targetName + ' Rate';
      $('fraudChartTitle').textContent = D.targetName + ' Rate & Loss Trend';
    }

    // Feature means table
    const means = $('featureMeans');
    const cols = D ? D.numericCols.slice(0, 8) : [];
    if (cols.length === 0) { means.innerHTML = '<div class="empty">No numeric features</div>'; }
    else {
      means.innerHTML = cols.map(c => {
        const curr = (tickData.currentValsMap[c] || []);
        const cm = curr.length ? curr.reduce((a, b) => a + b, 0) / curr.length : null;
        const bm = Engine.SIM.baseline.means[c];
        const delta = cm != null && bm != null ? cm - bm : null;
        const deltaStr = delta != null ? (delta >= 0 ? '+' : '') + sv(delta, '—', 2) : '—';
        const cls = delta != null && Math.abs(delta) > Math.abs(bm || 1) * 0.2 ? 'amber' : 'muted';
        return `<div class="feature-row">
          <div class="feature-name">${(D.featureAliases || {})[c] || c}</div>
          <div class="feature-val">${sv(cm, '—', 2)}</div>
          <div class="feature-delta ${cls}">${deltaStr}</div>
        </div>`;
      }).join('');
    }

    // Drift diagnostics
    const maxZ = Math.max(...Object.values(tickData.zscoreMap).filter(v => v != null), 0);
    const psiLvl = maxPSI >= 0.25 ? 'Critical' : maxPSI >= 0.2 ? 'Alert' : maxPSI >= 0.1 ? 'Warning' : 'Stable';
    const psiCls = maxPSI >= 0.25 ? 'red' : maxPSI >= 0.1 ? 'amber' : 'green';
    const zLvl = maxZ >= 3 ? 'Critical' : maxZ >= 2 ? 'Elevated' : maxZ >= 1 ? 'Marginal' : 'Normal';
    const zCls = maxZ >= 3 ? 'red' : maxZ >= 2 ? 'amber' : 'ok';
    const interp = (maxPSI >= 0.25 && maxZ >= 2)
      ? 'Both PSI and Z-score elevated — confirmed sustained drift in distribution shape AND mean.'
      : (maxPSI >= 0.2 && maxZ < 1.5) ? 'High PSI but moderate Z — shape shift without sharp mean displacement. Typical of variance drift or bimodal change.'
      : (maxZ >= 2.5 && maxPSI < 0.1) ? 'Sharp Z but low PSI — single batch mean spike. Monitor for persistence.'
      : 'Both signals within normal bounds.';
    $('driftDiagnostics').innerHTML = `
      <div class="diag-row">
        <div class="diag-block"><div class="kv-label">Peak PSI <span class="kv-hint" title="PSI &lt;0.1=Stable · 0.1-0.2=Warning · &gt;0.2=Alert · &gt;0.25=Critical">ℹ</span></div>
          <div class="technical-highlight ${psiCls}">${fmtPSI(maxPSI)} — ${psiLvl}</div>
          <div class="diag-note">Rolling distribution vs baseline</div></div>
        <div class="diag-block"><div class="kv-label">Max Z-Score <span class="kv-hint" title="Z≥2=Elevated · Z≥3=Critical. Measures mean displacement from baseline.">ℹ</span></div>
          <div class="technical-highlight ${zCls}">${fmtZ(maxZ)} — ${zLvl}</div>
          <div class="diag-note">Batch mean vs baseline mean</div></div>
        <div class="diag-block"><div class="kv-label">${D ? D.targetName : 'Event'} Rate</div>
          <div class="technical-highlight ${displayFraudRate > 0.1 ? 'red' : displayFraudRate > 0.03 ? 'amber' : 'green'}">${fmtPct(displayFraudRate * 100)} ${D ? '(baseline: ' + fmtPct((D.positiveRate || 0) * 100, 2) + ')' : ''}</div>
          <div class="diag-note">Action modifier: ${sv(Engine.SIM.actionState.fraudModifier, 1, 2)}×</div></div>
      </div>
      <div class="diag-interpretation">${interp}</div>`;

    addLog(drifting ? 'WARNING' : 'INFO',
      `Batch ${batchId} | ${D ? D.targetName : 'Event'}=${fmtPct(displayFraudRate * 100)} | PSI=${fmtPSI(maxPSI)} | Loss=${fmtMoney(lossPerHour)}/hr | State=${$('stateLabel').textContent}`,
      drifting);
  }

  // ── Root cause panel ──────────────────────────────────────────────────
  function updateRootCauses(hypotheses) {
    const D = global.DATASET;
    const aliases = D ? D.featureAliases || {} : {};
    if (!hypotheses || hypotheses.length === 0) {
      $('rootCauseAttribution').innerHTML = '<div class="empty">No drift detected — awaiting causal evidence.</div>';
      return;
    }
    $('rootCauseAttribution').innerHTML = hypotheses.map((h, i) => {
      const pct = Math.round(h.confidence * 100);
      const tone = pct >= 80 ? '' : pct >= 60 ? 'warn' : 'err';
      return `<article class="cause-card ${i === 0 ? 'primary' : ''}">
        <div class="cause-top">
          <div>
            <div class="cause-rank">${i === 0 ? 'Primary causal hypothesis' : 'Alternate hypothesis ' + (i + 1)}</div>
            <div class="cause-name">${h.label}</div>
            <div class="cause-source">${h.source}</div>
          </div>
          <div class="conf-pill ${tone || 'ok'}">${h.type.replace(/_/g, ' ')}</div>
        </div>
        <div class="confidence-row">
          <div class="conf-number">${pct}%</div>
          <div class="conf-label">confidence · lag ${h.lagSeconds}s</div>
        </div>
        <div class="bar"><div class="bar-fill ${tone}" style="width:${pct}%"></div></div>
        <div class="evidence-list">
          ${h.evidence.map(e => '<div class="evidence-item">' + e + '</div>').join('')}
        </div>
      </article>`;
    }).join('');
  }

  // ── Decision panel ────────────────────────────────────────────────────
  function updateDecision(tickData) {
    const D = global.DATASET;
    const SIM = Engine.SIM;
    const { maxPSI, displayFraudRate, lossPerHour, drifting, incidentActive } = tickData;
    const riskScore = Engine.computeRiskScore(maxPSI, displayFraudRate, 0, 0);
    const aliases = D ? D.featureAliases || {} : {};
    const targetName = D ? D.targetName : 'Event';
    const topF = tickData.topDriftedFeatures[0];
    const topAlias = topF ? (aliases[topF] || topF) : 'top feature';
    const topPSI = topF ? fmtPSI(tickData.psiMap[topF]) : '—';

    let action = 'monitor', priority = 'low', rationale = 'System within normal bounds. Continue observing.';
    let specific = null;
    if (incidentActive && riskScore >= 70) {
      action = 'rollback'; priority = 'critical';
      specific = `Revert to last stable model checkpoint. ${topAlias} PSI=${topPSI} exceeds critical threshold.`;
      rationale = `Risk score ${riskScore}/100. ${targetName} rate at ${fmtPct(displayFraudRate * 100)} with active causal evidence. Rollback is fastest recovery path.`;
    } else if (drifting && riskScore >= 40) {
      action = 'increase_review'; priority = 'high';
      specific = `Raise review threshold — route top ${Math.min(35, Math.round(displayFraudRate * 200))}% score distribution to manual queue.`;
      const catchRate = Math.min(55, Math.round(displayFraudRate * 280));
      const saved = Math.round(lossPerHour * (catchRate / 100));
      rationale = `Drift in ${topAlias} (PSI=${topPSI}) with ${fmtPct(displayFraudRate * 100)} ${targetName.toLowerCase()} rate. Manual review catches ~${catchRate}% — saves est. ${fmtMoney(saved)}/hr.`;
    } else if (drifting) {
      action = 'open_incident'; priority = 'medium';
      specific = `Open P2 incident. Re-evaluate at T+15 min or when ${targetName.toLowerCase()} rate crosses ${fmtPct((displayFraudRate * 1.5) * 100)}.`;
      rationale = `Drift detected (PSI=${topPSI}) but risk score ${riskScore} is below rollback threshold. Monitor causal timeline.`;
    }

    const proj = Engine.projectOutcome(action, displayFraudRate, lossPerHour);
    const canaryDecision = SIM.actionState.active
      ? SIM.actionState.active.replace(/_/g, ' ')
      : (incidentActive ? 'rollback' : 'hold');

    $('finalDecision').innerHTML = `
      <div class="decision-hero">
        <div class="decision-label">Recommended Action</div>
        <div class="decision-action">${action.replace(/_/g, ' ')}</div>
        <div class="decision-meta">Priority ${priority.toUpperCase()} · Risk ${riskScore}/100</div>
      </div>
      ${specific ? `<div class="decision-specific">⚙ ${specific}</div>` : ''}
      <div class="decision-reason">${rationale}</div>
      <div class="consequence-grid" style="margin-top:10px">
        <div class="cons-cell">
          <div class="cons-label">T+5 min</div>
          <div class="cons-fraud ${proj.t5FraudPct < 3 ? 'green' : proj.t5FraudPct < 8 ? 'amber' : 'red'}">${fmtPct(proj.t5FraudPct)}</div>
          <div class="cons-loss">${targetName} rate</div>
        </div>
        <div class="cons-cell">
          <div class="cons-label">T+15 min</div>
          <div class="cons-fraud ${proj.t15FraudPct < 3 ? 'green' : proj.t15FraudPct < 8 ? 'amber' : 'red'}">${fmtPct(proj.t15FraudPct)}</div>
          <div class="cons-loss">${fmtMoney(proj.t15LossPerHour)}/hr</div>
        </div>
        <div class="cons-cell">
          <div class="cons-label">T+30 min</div>
          <div class="cons-fraud ${proj.t30FraudPct < 3 ? 'green' : proj.t30FraudPct < 8 ? 'amber' : 'red'}">${fmtPct(proj.t30FraudPct)}</div>
          <div class="cons-loss">${fmtMoney(proj.t30LossPerHour)}/hr</div>
        </div>
      </div>
      ${proj.savedVsIgnore > 0 ? `<div class="cons-saved">+${fmtMoney(proj.savedVsIgnore)}/hr saved vs ignoring over 30 min</div>` : ''}
      <div class="canary-posture ${incidentActive ? 'err' : drifting ? 'warn' : 'ok'}">
        <span class="canary-label">Deployment Posture</span>
        <span class="canary-value">${canaryDecision.toUpperCase()}</span>
      </div>`;
  }

  // ── Risk forecast panel ───────────────────────────────────────────────
  function updateRiskForecast(tickData) {
    const D = global.DATASET;
    const { maxPSI, displayFraudRate, lossPerHour, drifting } = tickData;
    const riskScore = Engine.computeRiskScore(maxPSI, displayFraudRate, 0, 0);
    const level = riskScore >= 70 ? 'critical' : riskScore >= 40 ? 'high' : riskScore >= 20 ? 'moderate' : 'low';
    const levelCls = riskScore >= 70 ? 'err' : riskScore >= 40 ? 'warn' : 'ok';
    $('riskForecast').innerHTML = `
      <div class="risk-score">
        <div class="risk-score-main ${levelCls}">${sv(riskScore, '—')}</div>
        <div class="risk-label">${level} risk on a 0–100 scale</div>
      </div>
      <div class="risk-kvs">
        <div class="kv-mini"><div class="kv-label">Loss / Hour</div><div class="kv-value red">${fmtMoney(lossPerHour)}</div></div>
        <div class="kv-mini"><div class="kv-label">Peak PSI</div><div class="kv-value ${maxPSI >= 0.2 ? 'amber' : 'ok'}">${fmtPSI(maxPSI)}</div></div>
        <div class="kv-mini"><div class="kv-label">${D ? D.targetName : 'Event'} Rate</div><div class="kv-value ${displayFraudRate > 0.05 ? 'red' : 'green'}">${fmtPct(displayFraudRate * 100)}</div></div>
        <div class="kv-mini"><div class="kv-label">Baseline Rate</div><div class="kv-value">${D ? fmtPct((D.positiveRate || 0) * 100, 2) : '—'}</div></div>
      </div>`;
  }

  // ── Decision simulation panel ─────────────────────────────────────────
  function updateSimulation(tickData) {
    const { lossPerHour, displayFraudRate } = tickData;
    const SIM = Engine.SIM;
    const recommended = SIM.actionState.active || (displayFraudRate > 0.05 ? 'rollback' : 'monitor');
    const sims = [
      { action: 'rollback',           loss_pct: 74, risk: 'low',    eta: '2–5 min',  cost: 'Low' },
      { action: 'increase_review',    loss_pct: 55, risk: 'medium', eta: 'Immediate',cost: 'Medium' },
      { action: 'trigger_retraining', loss_pct: 33, risk: 'medium', eta: '45–90 min',cost: 'High' },
      { action: 'open_incident',      loss_pct: 10, risk: 'high',   eta: 'Page now', cost: 'None' },
    ];
    $('decisionSimulation').innerHTML = `<div class="sim-grid">${sims.map(s => {
      const isRec = s.action === recommended;
      const saved = Math.round(lossPerHour * (s.loss_pct / 100));
      return `<div class="sim-card ${isRec ? 'recommended' : ''}">
        ${isRec ? '<div class="rec-badge">RECOMMENDED</div>' : ''}
        <div class="sim-action">${s.action.replace(/_/g, ' ')}</div>
        <div class="sim-loss">${s.loss_pct}%</div>
        <div class="sim-sub">Projected loss reduction · saves ${fmtMoney(saved)}/hr</div>
        <div class="sim-risk ${s.risk}">${s.risk.toUpperCase()} RISK</div>
        <div class="sim-metric-row">
          <div class="sim-metric"><span>Time to effect</span><strong>${s.eta}</strong></div>
          <div class="sim-metric"><span>Operator cost</span><strong>${s.cost}</strong></div>
        </div>
      </div>`;
    }).join('')}</div>`;
  }

  // ── Timeline panel ────────────────────────────────────────────────────
  function updateTimeline(tickData, hypotheses) {
    const D = global.DATASET;
    const { batchId, maxPSI, displayFraudRate, drifting, incidentActive } = tickData;
    const targetName = D ? D.targetName : 'Event';
    const aliases = D ? D.featureAliases || {} : {};
    const topF = tickData.topDriftedFeatures[0];

    $('timelineNarrative').textContent = incidentActive
      ? `Causal chain confirmed — drift onset at batch ${Engine.SIM.incidentStartBatch || batchId}. ${hypotheses.length} hypotheses ranked by confidence.`
      : 'No active incident. Timeline will populate when drift is detected.';

    const events = [];
    if (D) {
      events.push({ title: 'Dataset loaded', desc: `${D.filename} · ${D.rowCount} rows · ${D.numericCols.length} numeric features`, sev: 'info', causal: 'Baseline established from first ' + D.baselineBatches + ' batches' });
    }
    if (incidentActive && hypotheses[0]) {
      const h = hypotheses[0];
      events.push({ title: `Root cause: ${h.label}`, desc: h.description, sev: 'critical', causal: `Confidence ${Math.round(h.confidence * 100)}% · lag ${h.lagSeconds}s` });
    }
    if (drifting && topF) {
      events.push({ title: `Distribution shift: ${aliases[topF] || topF}`, desc: `PSI=${fmtPSI(tickData.psiMap[topF])} exceeds threshold ${fmtPSI(Engine.SIM.driftThreshold)}`, sev: maxPSI >= 0.25 ? 'critical' : 'warning', causal: 'Feature movement increases incident risk' });
    }
    events.push({ title: `Batch ${batchId} processed`, desc: `${targetName} rate ${fmtPct(displayFraudRate * 100)} · Loss ${fmtMoney(tickData.lossPerHour)}/hr`, sev: incidentActive ? 'high' : 'info', causal: 'Live batch scored by simulation engine' });
    // Injected events
    injectedEvents.slice(-2).forEach(e => {
      events.push({ title: `Injected: ${e.type.replace(/_/g, ' ')}`, desc: `Source: ${e.source}`, sev: 'warning', causal: 'Operator-registered upstream event' });
    });
    // Action log entries
    Engine.SIM.actionLog.slice(0, 2).forEach(e => {
      events.push({ title: `Action: ${e.action.replace(/_/g, ' ')}`, desc: e.notes || 'Operator decision recorded', sev: 'info', causal: `Projected savings: ${fmtMoney(e.projectedSavings)}/hr vs inaction` });
    });

    const sev2tone = s => s === 'critical' ? 'err' : s === 'warning' || s === 'high' ? 'warn' : s === 'info' ? 'ok' : '';
    $('causalTimeline').innerHTML = events.slice(-6).map((e, i) => `
      <div class="timeline-item">
        <div class="timeline-time">${nowTime()}</div>
        <div class="timeline-link"><div class="timeline-dot ${sev2tone(e.sev)}"></div></div>
        <div class="timeline-body">
          <div class="timeline-title">${e.title}</div>
          <div class="timeline-desc">${e.desc}</div>
          <div class="timeline-causal">${e.causal}</div>
        </div>
      </div>`).join('');
  }

  // ── Heatmap ───────────────────────────────────────────────────────────
  function z2color(z) {
    if (z < 0.5) return '#0f2744';
    if (z < 1.0) return '#854d0e';
    if (z < 1.5) return '#c2410c';
    if (z < 2.0) return '#dc2626';
    if (z < 2.5) return '#991b1b';
    return '#7f1d1d';
  }

  function buildHeatmap() {
    const D = global.DATASET;
    const { matrix, cols } = Engine.getHeatmapMatrix(20);
    const container = $('heatmapContainer');
    if (!D || matrix.length === 0) {
      container.innerHTML = '<div class="empty">Heatmap will populate after drift onset…</div>';
      return;
    }

    const aliases = D.featureAliases || {};
    let html = '<div class="heatmap-table"><div class="heatmap-header-row"><div class="heatmap-feature-label"></div>';
    cols.forEach(b => { html += `<div class="heatmap-col-header">B${b}</div>`; });
    html += '</div>';

    matrix.forEach(row => {
      const psis = row.psis || [];
      const maxPSI = Math.max(...psis.filter(p => p != null), 0);
      const isDrifted = maxPSI > Engine.SIM.driftThreshold;
      html += `<div class="heatmap-row ${isDrifted ? 'drifted-row' : ''}">
        <div class="heatmap-feature-label" title="${row.feature}">${row.alias}</div>`;
      row.zscores.forEach((z, ci) => {
        const zv = z != null ? z : 0;
        const pv = psis[ci] != null ? psis[ci] : 0;
        const bm = Engine.SIM.baseline.means[row.feature] != null ? Engine.SIM.baseline.means[row.feature] : 0;
        // Use actual batch mean from last tick — no random approximation
        const curVals = lastCurrentValsMap[row.feature] || [];
        const curMean = curVals.length
          ? curVals.reduce((a, b) => a + b, 0) / curVals.length
          : bm;
        html += `<div class="heatmap-cell"
          style="background:${z2color(zv)};transition:background-color 600ms ease"
          onclick="App.openDrilldown('${row.feature}')"
          onmouseenter="App.showCellTip(event,'${row.feature}',${cols[ci]},${zv.toFixed(3)},${pv.toFixed(4)},${curMean.toFixed(2)},${bm.toFixed(2)},'${row.alias}')"
          onmouseleave="Tooltip.hide()">
        </div>`;
      });
      html += '</div>';
    });
    html += '</div>';
    container.innerHTML = html;
  }

  // ── Drilldown ─────────────────────────────────────────────────────────
  function openDrilldown(feature) {
    const D = global.DATASET;
    if (!D) return;
    lastDrilldownFeature = feature;
    const alias = (D.featureAliases || {})[feature] || feature;
    $('drilldownTitle').textContent = alias + ' — Feature Drilldown';
    $('heatmapDrilldown').style.display = 'block';
    updateDrilldown(feature);
  }

  function updateDrilldown(feature) {
    const D = global.DATASET;
    const SIM = Engine.SIM;
    const history = SIM.metrics.batchIds.slice(-30);
    const psis = (SIM.metrics.psiPerFeature[feature] || []).slice(-30);
    const zs = (SIM.metrics.zscorePerFeature[feature] || []).slice(-30);
    const bm = SIM.baseline.means[feature] || 0;
    const lastPSI = psis.at(-1);
    const lastZ = zs.at(-1);
    const alias = (D ? D.featureAliases || {} : {})[feature] || feature;

    if (!drilldownChart) {
      const ctx = $('drilldownChart').getContext('2d');
      drilldownChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: history.map(b => 'B' + b),
          datasets: [
            { label: 'Z-Score', data: zs, borderColor: '#ff6c78', backgroundColor: 'rgba(255,108,120,0.1)', borderWidth: 2, tension: 0.3, fill: true, pointRadius: 2, yAxisID: 'z' },
            { label: 'PSI', data: psis, borderColor: '#ffc46b', backgroundColor: 'rgba(255,196,107,0.08)', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 2, yAxisID: 'psi' },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#90a1b9', boxWidth: 12, usePointStyle: true } }, tooltip: { backgroundColor: 'rgba(8,13,22,0.96)', padding: 10 } },
          scales: {
            x: { ticks: { color: '#51627a' }, grid: { color: 'rgba(255,255,255,0.04)' } },
            z: { type: 'linear', position: 'left', beginAtZero: true, ticks: { color: '#ff9aa2' }, title: { display: true, text: 'Z-Score', color: '#ff9aa2' } },
            psi: { type: 'linear', position: 'right', min: 0, max: 0.5, ticks: { color: '#ffd08d' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'PSI', color: '#ffd08d' } },
          },
        },
      });
    } else {
      drilldownChart.data.labels = history.map(b => 'B' + b);
      drilldownChart.data.datasets[0].data = zs;
      drilldownChart.data.datasets[1].data = psis;
      drilldownChart.update('none');
    }

    $('drilldownStats').innerHTML = `
      <div class="dd-stats-grid">
        <div class="dd-stat"><div class="kv-label">${alias}</div><div class="kv-value">${Engine.psiLabel(lastPSI)}</div></div>
        <div class="dd-stat"><div class="kv-label">Current PSI</div><div class="kv-value ${lastPSI >= 0.2 ? 'amber' : 'green'}">${fmtPSI(lastPSI)}</div></div>
        <div class="dd-stat"><div class="kv-label">Current Z</div><div class="kv-value ${lastZ >= 2 ? 'amber' : 'ok'}">${fmtZ(lastZ)}</div></div>
        <div class="dd-stat"><div class="kv-label">Baseline Mean</div><div class="kv-value">${sv(bm, '—', 2)}</div></div>
      </div>`;
  }

  function closeDrilldown() {
    $('heatmapDrilldown').style.display = 'none';
    lastDrilldownFeature = null;
    if (drilldownChart) { drilldownChart.destroy(); drilldownChart = null; }
  }

  // ── Log ───────────────────────────────────────────────────────────────
  const MAX_LOG = 60;
  function addLog(level, msg, isDrift) {
    const t = $('terminalLog');
    if (!t) return;
    const line = document.createElement('div');
    const lvlCls = level === 'ERROR' ? 'err' : level === 'WARNING' ? 'warn' : 'info';
    line.className = 't-line';
    line.innerHTML = `<span class="t-time">${nowTime()}</span><span class="t-level ${lvlCls}">${level}</span><span class="t-msg ${isDrift ? 'drift' : ''}">${msg}</span>`;
    t.prepend(line);
    while (t.children.length > MAX_LOG) t.removeChild(t.lastChild);
  }

  // ── Alerts list ───────────────────────────────────────────────────────
  function addAlert(batchId, maxPSI, topFeature, alias) {
    alertsList.unshift({ batchId, maxPSI, topFeature, alias, time: nowTime() });
    if (alertsList.length > 20) alertsList.pop();
    $('alertsList').innerHTML = alertsList.map(a => `
      <div class="alert-item">
        <div class="badge ${a.maxPSI >= 0.25 ? 'critical' : 'warning'}">${a.maxPSI >= 0.25 ? 'critical' : 'warning'}</div>
        <div>
          <div>Drift alert: ${a.alias || a.topFeature} | PSI=${fmtPSI(a.maxPSI)}</div>
          <div class="support-meta">${a.time} · Batch ${a.batchId}</div>
        </div>
      </div>`).join('');
  }

  // ── Main tick callback ─────────────────────────────────────────────────
  let lastIncident = false;
  function onTick(tickData) {
    lastCurrentValsMap = tickData.currentValsMap || {};  // store for heatmap tooltip
    updateHeroMetrics(tickData);
    updateDecision(tickData);
    updateRiskForecast(tickData);
    updateSimulation(tickData);
    updateTimeline(tickData, Engine.SIM.hypotheses);
    updateAllCharts(tickData);
    // Throttle heatmap rebuild — only when batch ID changes (debounce heavy DOM work)
    if (tickData.batchId !== lastHeatmapBatch) {
      lastHeatmapBatch = tickData.batchId;
      buildHeatmap();
    }
    if (tickData.drifting && !lastIncident) {
      const D = global.DATASET;
      const topF = tickData.topDriftedFeatures[0];
      addAlert(tickData.batchId, tickData.maxPSI, topF, D ? (D.featureAliases || {})[topF] : topF);
    }
    lastIncident = tickData.incidentActive;
  }

  function onIncident(batchId, maxPSI, hypotheses) {
    addLog('ERROR', `INCIDENT at batch ${batchId} — PSI=${fmtPSI(maxPSI)} — ${hypotheses.length} causal hypotheses`, true);
    updateRootCauses(hypotheses);
  }

  // ── PSI slider ────────────────────────────────────────────────────────
  function initSlider() {
    const slider = $('psiSlider');
    if (!slider) return;
    slider.addEventListener('input', e => {
      const val = (+e.target.value) / 100;
      Engine.SIM.driftThreshold = val;
      $('thresholdDisplay').textContent = val.toFixed(2);
    });
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────────
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); Engine.SIM.playing ? App.pause() : App.play(); }
    if (e.code === 'KeyR') App.resetSim();
  });

  // ── Dataset load flow ─────────────────────────────────────────────────
  function showLoading(show) {
    $('loadingOverlay').style.display = show ? 'flex' : 'none';
  }

  function setProgress(pct, title, msg) {
    $('progressFill').style.width = pct + '%';
    if (title) $('loadingTitle').textContent = title;
    if (msg) $('loadingMsg').textContent = msg;
  }

  async function onFileSelected(file) {
    if (!file) return;
    showLoading(true);
    try {
      const dataset = await Parser.loadDataset(file, ({ stage, pct, message }) => {
        setProgress(pct, stage === 'done' ? 'Ready!' : 'Parsing dataset…', message);
      });
      startWithDataset(dataset);
    } catch (err) {
      showLoading(false);
      alert('Error loading CSV: ' + err.message);
      addLog('ERROR', 'CSV load failed: ' + err.message);
    }
  }

  function startWithDataset(dataset) {
    showLoading(false);
    $('landingState').style.display = 'none';
    $('dashboardState').style.display = 'block';
    $('topbarDataset').style.display = 'flex';
    $('datasetLabel').textContent = dataset.filename;
    const sampleNote = dataset.isLarge ? ' · SAMPLED first 10MB' : '';
    const excNote = dataset.ignoredCols.length
      ? ` · ${dataset.ignoredCols.length} cols excluded (ID/monotonic)` : '';
    $('datasetSubtitle').textContent =
      `${dataset.filename} · ${dataset.rowCount.toLocaleString()} rows · ${dataset.numericCols.length} features tracked · target: ${dataset.targetCol}${sampleNote}${excNote}`;
    $('kpiDataset').textContent = dataset.filename.replace('.csv', '');
    $('kpiDatasetSub').textContent = `${dataset.rowCount.toLocaleString()} rows · ${dataset.numericCols.length} features`;
    $('demoBanner').style.display = dataset.isDemo ? 'flex' : 'none';

    // Destroy existing charts
    [metricsChart, driftChart, lossChart, drilldownChart].forEach(c => { if (c) c.destroy(); });
    metricsChart = driftChart = lossChart = drilldownChart = null;
    alertsList = []; injectedEvents = []; lastDrilldownFeature = null; lastIncident = false;
    lastCurrentValsMap = {}; lastHeatmapBatch = -1;

    Engine.initSim(dataset);
    Engine.SIM.callbacks.onTick = onTick;
    Engine.SIM.callbacks.onIncident = onIncident;

    initCharts();
    initSlider();
    Tooltip.init();
    Tooltip.wireAll();

    addLog('INFO', `Dataset loaded: ${dataset.filename} | ${dataset.rowCount} rows | target: ${dataset.targetCol} | ${dataset.numericCols.length} numeric features`);
    addLog('INFO', `Numeric cols: ${dataset.numericCols.join(', ')}`);
    if (dataset.ignoredCols.length) addLog('INFO', `Ignored (high-cardinality ID cols): ${dataset.ignoredCols.join(', ')}`);
    if (dataset.isLarge) addLog('WARNING', `Large file (>${Math.round(dataset.fileSize/1e6)||'?'}MB) — using first 10MB as representative sample. Positive rate may be underestimated.`);
    if (dataset.isDemo) addLog('INFO', 'Demo mode — drift injected at batch ' + (dataset.demoConfig.driftStartBatch || 50));
    seedActionLog();  // Bug 8: populate action log immediately on load
  }

  // ── Public API (called from HTML onclick) ─────────────────────────────
  const App = {
    play() {
      if (!global.DATASET) return;
      Engine.play();
      $('btnPlay').style.display = 'none';
      $('btnPause').style.display = '';
      $('statusText').textContent = Engine.SIM.incidentActive ? 'INCIDENT ACTIVE' : 'Live';
    },
    pause() {
      Engine.pause();
      $('btnPlay').style.display = '';
      $('btnPause').style.display = 'none';
    },
    resetSim() {
      App.pause();
      Engine.reset();
      alertsList = []; injectedEvents = []; lastIncident = false;
      $('statusText').textContent = 'Paused';
      $('heroRiskScore').textContent = '—';
      $('heroLoss').textContent = '—';
      $('heroFraud').textContent = '—';
      $('heroMaxPSI').textContent = '—';
      $('rootCauseAttribution').innerHTML = '<div class="empty">Reset — awaiting drift onset.</div>';
      $('alertsList').innerHTML = '<div class="empty">No alerts yet</div>';
      $('controlLog').innerHTML = '<div class="empty">No operator actions recorded yet.</div>';
      $('criticalBanner').style.display = 'none';
      const mw = $('mainWrapper'); if (mw) mw.classList.remove('incident-active');
      addLog('INFO', 'Simulation reset');
    },
    setSpeed(s, btn) {
      Engine.setSpeed(s);
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');
    },
    executeAction(btn) {
      const action = btn.dataset.action;
      const notes = ($('ctrlNotes') || {}).value || '';
      const entry = Engine.applyAction(action, notes);
      const feedback = $('ctrlFeedback');
      feedback.className = 'ctrl-feedback ok';
      feedback.textContent = '✓ Action recorded — ' + nowTime();
      renderConsequencePreview(action, entry.projectedOutcome);
      renderActionLog();
      addLog('INFO', `Operator executed: ${action}${notes ? ' — "' + notes + '"' : ''} · projected savings ${fmtMoney(entry.projectedSavings)}/hr`);
    },
    injectEvent() {
      const type   = ($('injectEventType') || {}).value || 'distribution_shift';
      const source = ($('injectSource') || {}).value.trim() || 'manual-injection';
      const fields = ($('injectFields') || {}).value.trim().split(',').map(s => s.trim()).filter(Boolean);
      const evt = { type, source, fields, time: nowTime() };
      injectedEvents.push(evt);
      const fb = $('injectFeedback');
      fb.className = 'ctrl-feedback ok';
      fb.textContent = '✓ Injected — causal engine updated';
      // Refresh hypotheses to pick up injected event
      if (Engine.SIM.hypotheses.length > 0) {
        const SIM = Engine.SIM;
        const last = SIM.metrics.batchIds.at(-1) || SIM.currentIdx;
        SIM.hypotheses = Engine.generateHypotheses(last, SIM.metrics.topDriftedFeatures, {}, {});
        updateRootCauses(SIM.hypotheses);
      }
      addLog('INFO', `Upstream event injected: ${type} from ${source}`);
    },
    loadDemo(type) {
      let cfg = { targetName: 'Fraud' };
      if (type === 'churn')   cfg = { targetName: 'Churn' };
      if (type === 'anomaly') cfg = { targetName: 'Anomaly' };
      const dataset = Parser.generateDemoDataset(cfg);
      startWithDataset(dataset);
    },
    openDrilldown, closeDrilldown,
    showCellTip(e, feature, batchId, z, psi, curMean, baseMean, alias) {
      Tooltip.show(e.clientX, e.clientY, Tooltip.heatmapCellTip(feature, batchId, z, psi, curMean, baseMean, alias));
    },
    exportReport() {
      const D = global.DATASET;
      const SIM = Engine.SIM;
      if (!D) { alert('No dataset loaded'); return; }
      const topF = SIM.metrics.topDriftedFeatures.slice(0, 3);
      const aliases = D.featureAliases || {};
      const lines = [
        '═══ ML INCIDENT COMMAND CENTER — INCIDENT REPORT ═══',
        '',
        'Dataset:       ' + D.filename,
        'Target:        ' + D.targetCol + ' (' + D.targetName + ')',
        'Rows sampled:  ' + D.rowCount,
        'Sim batches:   ' + SIM.metrics.batchIds.length + ' of ' + SIM.batches.length,
        'Generated at:  ' + new Date().toISOString(),
        '',
        '─── PEAK METRICS ───',
        'Peak Loss/hr:  ' + fmtMoney(Math.max(...SIM.metrics.lossHistory, 0)),
        'Peak ' + D.targetName + ' Rate: ' + fmtPct(Math.max(...SIM.metrics.fraudRateHistory, 0)),
        'Peak PSI:      ' + fmtPSI(Object.values(SIM.metrics.psiPerFeature).flat().reduce((m, v) => v > m ? v : m, 0)),
        '',
        '─── TOP DRIFTED FEATURES ───',
        ...topF.map(f => {
          const psis = SIM.metrics.psiPerFeature[f] || [];
          const maxP = Math.max(...psis, 0);
          const lastZ = (SIM.metrics.zscorePerFeature[f] || []).at(-1) || 0;
          return `  ${(aliases[f] || f).padEnd(28)} PSI=${fmtPSI(maxP)}  Z=${fmtZ(lastZ)}`;
        }),
        '',
        '─── OPERATOR ACTIONS ───',
        ...SIM.actionLog.map(e => `  [${new Date(e.timestamp).toLocaleTimeString()}] ${e.action.replace(/_/g, ' ')} · ${e.notes || 'No note'} · Est. savings ${fmtMoney(e.projectedSavings)}/hr`),
        SIM.actionLog.length === 0 ? '  No operator actions taken' : '',
        '',
        '─── CAUSAL HYPOTHESES (most recent) ───',
        ...(SIM.hypotheses || []).map(h => `  H${h.rank}: ${h.label} (${h.source}) — ${Math.round(h.confidence * 100)}% confidence`),
        '',
        '═══ END REPORT ═══',
      ];
      const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'incident_report_' + Date.now() + '.txt';
      a.click();
    },
  };

  // ── Consequence preview ───────────────────────────────────────────────
  function renderConsequencePreview(action, proj) {
    const el = $('consequencePreview');
    if (!el || !proj) return;
    const D = global.DATASET;
    const targetName = D ? D.targetName : 'Event';
    const fc = v => v < 3 ? 'green' : v < 8 ? 'amber' : 'red';
    const labels = { rollback: 'Rollback Model', increase_review: 'Increase Manual Review', trigger_retraining: 'Trigger Retraining', open_incident: 'Open Incident', monitor: 'Continue Monitoring' };
    el.innerHTML = `
      <div class="cons-action-tag">${labels[action] || action}</div>
      <div class="consequence-grid">
        <div class="cons-cell"><div class="cons-label">T+5 min</div><div class="cons-fraud ${fc(proj.t5FraudPct)}">${fmtPct(proj.t5FraudPct)}</div><div class="cons-loss">${targetName} rate</div></div>
        <div class="cons-cell"><div class="cons-label">T+15 min</div><div class="cons-fraud ${fc(proj.t15FraudPct)}">${fmtPct(proj.t15FraudPct)}</div><div class="cons-loss">${fmtMoney(proj.t15LossPerHour)}/hr</div></div>
        <div class="cons-cell"><div class="cons-label">T+30 min</div><div class="cons-fraud ${fc(proj.t30FraudPct)}">${fmtPct(proj.t30FraudPct)}</div><div class="cons-loss">${fmtMoney(proj.t30LossPerHour)}/hr</div></div>
      </div>
      ${proj.savedVsIgnore > 0 ? `<div class="cons-saved">+${fmtMoney(proj.savedVsIgnore)}/hr saved vs doing nothing over 30 min</div>` : ''}
      <div class="cons-narrative">Fraud modifier → ${fmtPct(Engine.SIM.actionState.targetModifier * 100)} of baseline over next ticks.</div>`;
  }

  // ── Bug 8: Seed action log with system events on dataset load ────────
  function seedActionLog() {
    const D = global.DATASET;
    const el = $('controlLog');
    if (!D || !el) return;
    const now = new Date();
    const prev = new Date(now - 5000);
    const fmt = d => d.toLocaleTimeString();
    el.innerHTML = `<div class="log-entries">
      <div class="log-entry">
        <div class="log-time">${fmt(now)}</div>
        <div>
          <div class="log-action-name blue">Monitoring Started</div>
          <div class="log-details">Baseline computed from first ${D.baselineBatches || 20} batches · ${(D.sampleSize || D.rowCount || 0).toLocaleString()} rows sampled from ${D.filename}</div>
        </div>
        <div class="log-saved">AUTO</div>
      </div>
      <div class="log-entry">
        <div class="log-time">${fmt(prev)}</div>
        <div>
          <div class="log-action-name muted">Dataset Loaded</div>
          <div class="log-details">${(D.numericCols || []).length} features tracked · target: ${D.targetName || D.targetCol} · positive rate: ${((D.positiveRate || 0) * 100).toFixed(2)}%</div>
        </div>
        <div class="log-saved">SYSTEM</div>
      </div>
    </div>`;
  }

  // ── Action log ────────────────────────────────────────────────────────
  const ACTION_COLORS = { rollback: 'danger', increase_review: 'warn', trigger_retraining: 'blue', open_incident: 'purple', monitor: 'muted' };
  const ACTION_LABELS = { rollback: 'Rollback Model', increase_review: 'Manual Review', trigger_retraining: 'Trigger Retraining', open_incident: 'Open Incident', monitor: 'Continue Monitoring' };
  function renderActionLog() {
    const D = global.DATASET;
    const entries = Engine.SIM.actionLog;
    const el = $('controlLog');
    if (!entries.length) { el.innerHTML = '<div class="empty">No operator actions recorded yet.</div>'; return; }
    el.innerHTML = `<div class="log-entries">${entries.slice(0, 8).map(e => `
      <div class="log-entry">
        <div class="log-time">${new Date(e.timestamp).toLocaleTimeString()}</div>
        <div>
          <div class="log-action-name ${ACTION_COLORS[e.action] || 'muted'}">${ACTION_LABELS[e.action] || e.action}</div>
          <div class="log-details">${D ? D.targetName : 'Event'} rate at exec: ${fmtPct(e.fraudRateAtExecution)} · Loss: ${fmtMoney(e.lossAtExecution)}/hr${e.notes ? ' · "' + e.notes + '"' : ''}</div>
        </div>
        <div class="log-saved">${e.projectedSavings > 0 ? '+' + fmtMoney(e.projectedSavings) + '/hr' : '—'}</div>
      </div>`).join('')}</div>`;
  }

  // ── Wire file inputs ───────────────────────────────────────────────────
  function wireFileInputs() {
    ['csvFileInput', 'csvDropInput', 'demoBannerUpload'].forEach(id => {
      const el = $(id);
      if (el) el.addEventListener('change', e => onFileSelected(e.target.files[0]));
    });

    const dropZone = $('dropZone');
    if (dropZone) {
      dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
      dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
      dropZone.addEventListener('drop', e => {
        e.preventDefault(); dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) onFileSelected(file);
        else alert('Please drop a CSV file');
      });
    }
  }

  // ── Boot ──────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    // Enable engine debug logs via ?debug=1 — no code change needed in prod
    window.__ML_OBS_DEBUG = new URLSearchParams(location.search).get('debug') === '1';
    wireFileInputs();
    Tooltip.init();
    addLog('INFO', 'ML Incident Command Center initialized — drop a CSV or try demo mode');
  });

  global.App = App;

})(window);
