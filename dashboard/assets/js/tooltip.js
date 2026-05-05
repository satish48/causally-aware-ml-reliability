/**
 * tooltip.js — Live tooltip system with real-time data from SIM state
 */
(function (global) {
  'use strict';

  let activeTooltip = null;
  let tooltipEl = null;
  const wiredElements = new WeakSet(); // idempotency guard — never double-wire

  function init() {
    tooltipEl = document.createElement('div');
    tooltipEl.id = 'liveTooltip';
    tooltipEl.className = 'live-tooltip';
    tooltipEl.style.cssText = 'display:none;position:fixed;z-index:9999;pointer-events:none;';
    document.body.appendChild(tooltipEl);
  }

  function show(x, y, html) {
    if (!tooltipEl) init();
    tooltipEl.innerHTML = html;
    tooltipEl.style.display = 'block';

    const vw = window.innerWidth, vh = window.innerHeight;
    const rect = tooltipEl.getBoundingClientRect();
    let left = x + 14, top = y - 10;
    if (left + rect.width > vw - 10) left = x - rect.width - 14;
    if (top + rect.height > vh - 10) top = vh - rect.height - 10;
    tooltipEl.style.left = left + 'px';
    tooltipEl.style.top = top + 'px';
  }

  function hide() {
    if (tooltipEl) tooltipEl.style.display = 'none';
  }

  // ── Formatters ─────────────────────────────────────────────────────────
  const fmt = {
    money: v => '$' + Math.round(+v || 0).toLocaleString(),
    pct: (v, d) => (+v || 0).toFixed(d != null ? d : 1) + '%',
    psi: v => v != null ? (+v).toFixed(3) : '—',
    z: v => v != null ? (+v).toFixed(2) : '—',
    num: (v, d) => v != null ? (+v).toFixed(d != null ? d : 2) : '—',
  };

  // ── Tooltip content builders (all data-driven from SIM) ────────────────
  function heatmapCellTip(feature, batchId, z, psi, currentMean, baselineMean, alias) {
    const delta = currentMean - baselineMean;
    const deltaPct = baselineMean !== 0 ? ((delta / baselineMean) * 100) : 0;
    const zLvl = z >= 2.5 ? 'Critical' : z >= 2 ? 'High' : z >= 1.5 ? 'Elevated' : z >= 1 ? 'Marginal' : 'Normal';
    const zClass = z >= 2 ? 'tip-red' : z >= 1 ? 'tip-amber' : 'tip-green';

    return `
      <div class="tip-row tip-header"><span>${alias || feature}</span></div>
      <div class="tip-divider"></div>
      <div class="tip-row"><span class="tip-label">Batch</span><span class="tip-val">${batchId}</span></div>
      <div class="tip-row"><span class="tip-label">Z-Score</span><span class="tip-val ${zClass}">${fmt.z(z)} <em>${zLvl}</em></span></div>
      <div class="tip-row"><span class="tip-label">PSI</span><span class="tip-val ${psi >= 0.2 ? 'tip-red' : psi >= 0.1 ? 'tip-amber' : 'tip-green'}">${fmt.psi(psi)}</span></div>
      <div class="tip-row"><span class="tip-label">Live mean</span><span class="tip-val">${fmt.num(currentMean, 2)}</span></div>
      <div class="tip-row"><span class="tip-label">Baseline</span><span class="tip-val">${fmt.num(baselineMean, 2)}</span></div>
      <div class="tip-row"><span class="tip-label">Δ</span><span class="tip-val ${delta > 0 ? 'tip-amber' : 'tip-green'}">${delta >= 0 ? '+' : ''}${fmt.num(delta, 2)} (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%)</span></div>
    `;
  }

  function actionTip(actionName) {
    const DS = global.DATASET || {};
    const SIM = (global.Engine || {}).SIM || {};
    const currentLoss = fmt.money((SIM.metrics && SIM.metrics.lossHistory) ? SIM.metrics.lossHistory.at(-1) : 0);
    const targetName = DS.targetName || 'Event';
    const positiveRate = DS.positiveRate != null ? (DS.positiveRate * 100).toFixed(2) : '—';

    const TIPS = {
      rollback: {
        title: 'Rollback Model',
        body: `Reverts to previous model checkpoint. Most effective for sudden distribution shifts — estimated ${targetName.toLowerCase()} rate drops to ~5% of current within 2 ticks.`,
        cost: 'Low operational cost — automatic in CI/CD',
        eta: '2–5 minutes',
      },
      manual_review: {
        title: 'Increase Manual Review',
        body: `Flags top-risk transactions for human review. Effective given ${positiveRate}% base ${targetName.toLowerCase()} rate — review queue remains manageable at this volume.`,
        cost: 'Medium — $22/hr per reviewer added to queue',
        eta: 'Immediate — next batch',
      },
      trigger_retraining: {
        title: 'Trigger Retraining',
        body: `Schedules model retraining on latest data. Best for gradual drift — ${targetName.toLowerCase()} rate unchanged for ~18 batches, then drops to 12% of current.`,
        cost: 'High — compute cost + 45–90 min downtime risk',
        eta: '45–90 minutes',
      },
      open_incident: {
        title: 'Open Incident',
        body: `Opens a formal P2 incident ticket and pages on-call engineer. Appropriate when causal source is unclear and human investigation is needed.`,
        cost: 'None — escalation only',
        eta: 'Immediate page',
      },
      monitor: {
        title: 'Continue Monitoring',
        body: `Maintains current state. Appropriate when drift is within acceptable SLO bounds. Current exposure: ${currentLoss}/hr.`,
        cost: 'None — maintains current ${targetName.toLowerCase()} exposure',
        eta: 'N/A',
      },
    };

    const tip = TIPS[actionName] || { title: actionName, body: 'Action details unavailable.', cost: '—', eta: '—' };
    return `
      <div class="tip-row tip-header"><span>${tip.title}</span></div>
      <div class="tip-divider"></div>
      <div class="tip-body">${tip.body}</div>
      <div class="tip-divider"></div>
      <div class="tip-row"><span class="tip-label">Operator cost</span><span class="tip-val">${tip.cost}</span></div>
      <div class="tip-row"><span class="tip-label">Time to effect</span><span class="tip-val">${tip.eta}</span></div>
      <div class="tip-row"><span class="tip-label">Current loss</span><span class="tip-val tip-red">${currentLoss}/hr</span></div>
    `;
  }

  function psiTip(feature, psi, alias) {
    const lvl = (global.Engine || {}).psiLabel ? global.Engine.psiLabel(psi) : '—';
    const sev = (global.Engine || {}).psiSeverity ? global.Engine.psiSeverity(psi) : 'stable';
    const cls = sev === 'critical' ? 'tip-red' : sev === 'alert' ? 'tip-amber' : sev === 'warning' ? 'tip-amber' : 'tip-green';
    const interp = psi >= 0.25
      ? 'Severe distribution shift — investigate immediately.'
      : psi >= 0.2 ? 'Significant shift — alert-level drift detected.'
      : psi >= 0.1 ? 'Moderate shift — monitor closely.'
      : 'No significant distribution change from baseline.';
    return `
      <div class="tip-row tip-header"><span>${alias || feature} — PSI</span></div>
      <div class="tip-divider"></div>
      <div class="tip-row"><span class="tip-label">Value</span><span class="tip-val ${cls}">${(+psi || 0).toFixed(3)}</span></div>
      <div class="tip-row"><span class="tip-label">Status</span><span class="tip-val ${cls}">${lvl}</span></div>
      <div class="tip-row"><span class="tip-label">Thresholds</span><span class="tip-val">&lt;0.10 Stable · 0.10–0.20 Warning · &gt;0.20 Alert</span></div>
      <div class="tip-body">${interp}</div>
    `;
  }

  function riskScoreTip() {
    const Eng = global.Engine || {};
    const SIM = Eng.SIM || {};
    const m   = SIM.metrics || {};
    // Derive max PSI from per-feature histories
    const psiVals = Object.values(m.psiPerFeature || {}).map(arr => arr.at(-1) || 0);
    const maxPSI  = psiVals.length ? Math.max(...psiVals) : 0;
    // fraudRateHistory is already in % — convert to fraction
    const fraudPct = m.fraudRateHistory ? (m.fraudRateHistory.at(-1) || 0) : 0;
    const fraudFrac = fraudPct / 100;
    const score = Eng.computeRiskScore ? Eng.computeRiskScore(maxPSI, fraudFrac, 0, 0) : 0;
    const lvl = score >= 75 ? 'Critical' : score >= 50 ? 'High' : score >= 25 ? 'Elevated' : 'Normal';
    const cls = score >= 75 ? 'tip-red' : score >= 50 ? 'tip-amber' : score >= 25 ? 'tip-amber' : 'tip-green';
    const psiPts   = Math.min(maxPSI  / 0.5, 1.0) * 40;
    const fraudPts = Math.min(fraudFrac / 0.5, 1.0) * 35;
    return `
      <div class="tip-row tip-header"><span>Risk Score</span><span class="${cls}">${Math.round(score)}</span></div>
      <div class="tip-divider"></div>
      <div class="tip-row"><span class="tip-label">Level</span><span class="tip-val ${cls}">${lvl}</span></div>
      <div class="tip-divider"></div>
      <div class="tip-row"><span class="tip-label">PSI component (40pt)</span><span class="tip-val">${psiPts.toFixed(1)}</span></div>
      <div class="tip-row"><span class="tip-label">Event-rate component (35pt)</span><span class="tip-val">${fraudPts.toFixed(1)}</span></div>
      <div class="tip-row"><span class="tip-label">Confidence component (15pt)</span><span class="tip-val">—</span></div>
      <div class="tip-row"><span class="tip-label">Burn-rate component (10pt)</span><span class="tip-val">—</span></div>
      <div class="tip-divider"></div>
      <div class="tip-body">Score is the weighted sum of four independent risk signals (max 100). Action recommended above 50.</div>
    `;
  }

  function psiHeroTip() {
    const SIM = (global.Engine || {}).SIM || {};
    const m = SIM.metrics || {};
    // Build current max PSI from per-feature trailing values
    const psiVals = Object.values(m.psiPerFeature || {}).map(arr => arr.at(-1) || 0);
    const maxPSI  = psiVals.length ? Math.max(...psiVals) : 0;
    const topArr  = m.topDriftedFeatures || [];
    const topFeat = topArr.length ? (topArr[0].alias || topArr[0].col || '—') : '—';
    const topPSI  = topArr.length ? (topArr[0].psi  || 0) : 0;
    const slider  = document.getElementById('psiSlider');
    const threshold = slider ? (+slider.value / 100) : 0.20;
    const lvl = maxPSI >= 0.25 ? 'Critical' : maxPSI >= 0.20 ? 'Alert' : maxPSI >= 0.10 ? 'Warning' : 'Stable';
    const cls = maxPSI >= 0.20 ? 'tip-red' : maxPSI >= 0.10 ? 'tip-amber' : 'tip-green';
    return `
      <div class="tip-row tip-header"><span>Peak PSI</span><span class="${cls}">${fmt.psi(maxPSI)}</span></div>
      <div class="tip-divider"></div>
      <div class="tip-row"><span class="tip-label">Status</span><span class="tip-val ${cls}">${lvl}</span></div>
      <div class="tip-row"><span class="tip-label">Alert threshold</span><span class="tip-val">${threshold.toFixed(2)}</span></div>
      <div class="tip-row"><span class="tip-label">Top drifted feature</span><span class="tip-val tip-amber">${topFeat} (${fmt.psi(topPSI)})</span></div>
      <div class="tip-divider"></div>
      <div class="tip-body">Population Stability Index measures how much the feature distribution has shifted from baseline. PSI &gt; 0.20 triggers an alert.</div>
    `;
  }

  // ── Attach tooltip to an element with dynamic content fn ──────────────
  function attach(el, contentFn) {
    el.addEventListener('mouseenter', e => {
      const html = contentFn(e);
      if (html) show(e.clientX, e.clientY, html);
    });
    el.addEventListener('mousemove', e => {
      const html = contentFn(e);
      if (html) show(e.clientX, e.clientY, html);
    });
    el.addEventListener('mouseleave', hide);
  }

  // ── Auto-wire elements with data-tooltip-type ─────────────────────────
  // Idempotent: skips already-wired elements via WeakSet so calling multiple
  // times (e.g. after dataset reload) never stacks duplicate listeners.
  function wireAll() {
    document.querySelectorAll('[data-tooltip-type]').forEach(el => {
      if (wiredElements.has(el)) return; // already wired — skip
      const type = el.dataset.tooltipType;
      if (type === 'action') {
        attach(el, () => actionTip(el.dataset.action || ''));
      } else if (type === 'psi') {
        attach(el, () => psiTip(el.dataset.feature, +el.dataset.psi, el.dataset.alias));
      } else if (type === 'risk-score') {
        attach(el, () => riskScoreTip());
      } else if (type === 'psi-hero') {
        attach(el, () => psiHeroTip());
      }
      wiredElements.add(el);
    });
  }

  global.Tooltip = { init, show, hide, attach, wireAll, heatmapCellTip, actionTip, psiTip, riskScoreTip, psiHeroTip, fmt };

})(window);
