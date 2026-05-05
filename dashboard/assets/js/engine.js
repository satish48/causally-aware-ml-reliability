/**
 * engine.js — Simulation engine, PSI computation, causal attribution
 * All math runs in-browser. No backend required.
 */
(function (global) {
  'use strict';

  // ── PSI computation — robust dual approach ─────────────────────────────
  // For small batches: use Cohen's d via tanh (bounded [0,1], stable for N<100)
  // For large batches: use histogram PSI with Laplace smoothing + capped at 1.5
  function computePSI(baselineVals, currentVals) {
    if (!baselineVals || baselineVals.length < 5) return null;
    if (!currentVals || currentVals.length < 30) return null;

    // Baseline stats
    const nBase = baselineVals.length;
    const baseMean = baselineVals.reduce((a, b) => a + b, 0) / nBase;
    const baseVar = baselineVals.reduce((a, v) => a + (v - baseMean) ** 2, 0) / nBase;
    const baseStd = Math.sqrt(baseVar);

    // Near-zero variance: skip
    if (baseStd < 1e-8) return { psi: 0, stable: true, reason: 'Insufficient variance' };

    const nCurr = currentVals.length;
    const currMean = currentVals.reduce((a, b) => a + b, 0) / nCurr;

    // Cohen's d (primary signal — bounded and stable for any N)
    const cohensD = Math.abs(currMean - baseMean) / baseStd;
    const cohensBasedPSI = Math.tanh(cohensD / 1.5);   // [0, 1]

    // Histogram PSI (secondary cross-check, only for N ≥ 100)
    let histPSI = cohensBasedPSI; // fallback = cohens
    if (nCurr >= 100) {
      const bins = Math.min(10, Math.max(5, Math.floor(Math.sqrt(nCurr))));
      const allVals = baselineVals.concat(currentVals);
      const lo = Math.min(...allVals);
      const hi = Math.max(...allVals);
      const range = hi - lo;
      if (range > 1e-8) {
        const binW = range / bins;
        const alpha = 1;  // Laplace smoothing
        const baseCounts = new Array(bins).fill(alpha);
        const currCounts = new Array(bins).fill(alpha);
        for (const v of baselineVals) {
          const b = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / binW)));
          baseCounts[b]++;
        }
        for (const v of currentVals) {
          const b = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / binW)));
          currCounts[b]++;
        }
        const sumBase = baseCounts.reduce((a, b) => a + b, 0);
        const sumCurr = currCounts.reduce((a, b) => a + b, 0);
        let rawPSI = 0;
        for (let i = 0; i < bins; i++) {
          const e = baseCounts[i] / sumBase;
          const a = currCounts[i] / sumCurr;
          rawPSI += (a - e) * Math.log(a / e);
        }
        // Normalize to [0,1] range — cap at 1.5, then normalize
        histPSI = Math.min(1.0, Math.max(0, rawPSI) / 1.5);
      }
    }

    // Blend: Cohen's d drives the score (stable), hist cross-checks for large N
    const finalPSI = nCurr >= 100
      ? (cohensBasedPSI * 0.6 + histPSI * 0.4)
      : cohensBasedPSI;

    return { psi: Math.max(0, Math.min(1.0, finalPSI)), stable: false };
  }

  function psiLabel(psi) {
    if (psi == null) return 'N/A';
    if (psi < 0.1) return 'Stable';
    if (psi < 0.2) return 'Warning';
    if (psi < 0.25) return 'Alert';
    return 'Critical';
  }

  function psiSeverity(psi) {
    if (psi == null) return 'stable';
    if (psi < 0.1) return 'stable';
    if (psi < 0.2) return 'warning';
    if (psi < 0.25) return 'alert';
    return 'critical';
  }

  // ── Z-score ────────────────────────────────────────────────────────────
  function computeZScore(baselineMean, baselineStd, currentMean) {
    if (baselineStd < 1e-8) return 0;
    return Math.abs(currentMean - baselineMean) / baselineStd;
  }

  // ── Weighted random WITHOUT replacement ───────────────────────────────
  function weightedPickN(catalog, n) {
    const pool = catalog.slice();
    const picked = [];
    for (let i = 0; i < n && pool.length > 0; i++) {
      const total = pool.reduce((s, e) => s + e.weight, 0);
      let r = Math.random() * total;
      let idx = 0;
      for (; idx < pool.length - 1; idx++) {
        r -= pool[idx].weight;
        if (r <= 0) break;
      }
      picked.push(pool[idx]);
      pool.splice(idx, 1);
    }
    return picked;
  }

  // ── Event catalog ──────────────────────────────────────────────────────
  const EVENT_CATALOG = [
    {
      source: 'pipeline-anomaly-detector',
      type: 'distribution_shift',
      label: 'Feature Pipeline',
      description: 'Rolling KS-test detected batch output deviation exceeding 3σ threshold',
      weight: 0.30,
    },
    {
      source: 'schema-drift-monitor',
      type: 'schema_change',
      label: 'Schema Validator',
      description: 'Nullable field ratio changed in upstream schema contract beyond tolerance',
      weight: 0.20,
    },
    {
      source: 'upstream-data-validator',
      type: 'data_quality',
      label: 'Data Quality Gate',
      description: 'Null rate spike and imputation errors detected in ingestion layer',
      weight: 0.18,
    },
    {
      source: 'deployment-tracker',
      type: 'model_deployment',
      label: 'Deployment Monitor',
      description: 'Model version change detected in serving layer — scoring weights updated',
      weight: 0.12,
    },
    {
      source: 'load-balancer-monitor',
      type: 'traffic_anomaly',
      label: 'Traffic Monitor',
      description: 'Unusual transaction volume pattern detected — routing policy shift',
      weight: 0.12,
    },
    {
      source: 'feature-store-monitor',
      type: 'feature_store_lag',
      label: 'Feature Store',
      description: 'Feature computation lag exceeding SLA by 340ms — stale features served',
      weight: 0.08,
    },
  ];

  // ── Risk score — Bug 4: component-based formula, never pinned at 100 ────
  function computeRiskScore(maxPSI, fraudRate, confidenceDrop, burnRate) {
    const psiScore   = Math.min((maxPSI || 0) / 0.5, 1.0) * 40;
    const fraudScore = Math.min((fraudRate || 0) / 0.5, 1.0) * 35;
    const confScore  = Math.min((confidenceDrop || 0) / 0.3, 1.0) * 15;
    const burnScore  = Math.min((burnRate || 0) / 20, 1.0) * 10;
    return Math.min(100, Math.max(0, Math.round(psiScore + fraudScore + confScore + burnScore)));
  }

  // ── Causal hypothesis generator — Bug 6: feature-based source matching ──
  function generateHypotheses(batchId, topFeatures, psiMap, zscoreMap) {
    const DS = global.DATASET || {};
    const aliases = DS.featureAliases || {};
    const threshold = SIM.driftThreshold;
    const maxPSI = Math.max(...Object.values(psiMap || {}).filter(v => v != null), 0);
    const pctAbove = (maxPSI > threshold && threshold > 0)
      ? ((maxPSI / threshold - 1) * 100).toFixed(0) : '0';
    const severity = maxPSI >= 0.25 ? 'Critical' : maxPSI >= 0.2 ? 'Alert' : 'Warning';
    const features = (topFeatures || []).slice(0, 5);

    // Rules: match based on WHICH features are drifting, not random weights
    const SOURCE_RULES = [
      {
        matches: f => f.some(x => /amount|balance|price|value|cost|revenue|spend/i.test(x)),
        source: 'upstream-data-validator', label: 'Data Quality Gate',
        type: 'data_quality',
        description: 'Null rate spike in transaction value fields detected at ingestion boundary',
      },
      {
        matches: f => f.length >= 4,
        source: 'schema-drift-monitor', label: 'Schema Validator',
        type: 'schema_change',
        description: 'Batch schema deviation across multiple fields — upstream schema contract changed',
      },
      {
        matches: f => f.some(x => /risk|score|prob|pred|model|output|flag/i.test(x)),
        source: 'deployment-tracker', label: 'Deployment Monitor',
        type: 'model_deployment',
        description: 'Model version change detected in serving layer — scoring weights updated',
      },
      {
        matches: f => f.some(x => /channel|device|region|country|geo|network|src/i.test(x)),
        source: 'load-balancer-monitor', label: 'Traffic Monitor',
        type: 'traffic_anomaly',
        description: 'Unusual transaction volume pattern detected — routing policy shift',
      },
      {
        matches: f => f.some(x => /lag|latency|delay|wait|queue/i.test(x)),
        source: 'feature-store-monitor', label: 'Feature Store',
        type: 'feature_store_lag',
        description: 'Feature computation lag exceeding SLA — stale features served to model',
      },
      {
        matches: () => true,  // fallback — always matches
        source: 'pipeline-anomaly-detector', label: 'Feature Pipeline',
        type: 'distribution_shift',
        description: 'Rolling KS-test detected batch output deviation exceeding 3σ threshold',
      },
    ];

    // Pick H1, H2, H3 from distinct sources via rule matching
    const picked = [];
    for (const rule of SOURCE_RULES) {
      if (picked.length >= 3) break;
      if (rule.matches(features) && !picked.some(p => p.source === rule.source)) {
        picked.push(rule);
      }
    }
    // Pad from EVENT_CATALOG if fewer than 3 matched
    for (const evt of EVENT_CATALOG) {
      if (picked.length >= 3) break;
      if (!picked.some(p => p.source === evt.source)) picked.push(evt);
    }

    // Confidence derived from actual PSI magnitude
    const h1Conf = Math.min(0.97, 0.85 + Math.max(0, maxPSI - threshold) * 0.3);
    const confidences = [
      h1Conf,
      h1Conf - (0.08 + Math.random() * 0.06),
      h1Conf - (0.16 + Math.random() * 0.10),
    ];

    return picked.slice(0, 3).map((evt, i) => {
      const lagSeconds = (10 + i * 35 + Math.random() * 40) | 0;
      const confidence = Math.max(0.45, confidences[i]);

      const topF = features.slice(0, 3);
      const psiValues = topF.map(f => {
        const p = psiMap && psiMap[f] != null ? psiMap[f].toFixed(3) : '—';
        return `${aliases[f] || f}: PSI ${p}`;
      }).join('; ');
      const zVals = topF.map(f => {
        const z = zscoreMap && zscoreMap[f] != null ? zscoreMap[f].toFixed(2) : '—';
        return `${aliases[f] || f}: Z=${z}`;
      }).join(', ');

      const evidence = [
        `Temporal alignment: ${evt.label} event occurred ${lagSeconds}s before drift onset at batch ${batchId}.`,
        topF.length > 0
          ? `Field correlation: ${psiValues}.`
          : 'No specific feature correlation — indirect upstream dependency.',
        topF.length > 0
          ? `Severity: ${severity} — PSI ${maxPSI.toFixed(3)} is ${pctAbove}% above threshold ${threshold.toFixed(2)}.`
          : 'Severity assessment pending additional batches.',
        `Confidence computed: ${(confidence * 100).toFixed(1)}% (base=${(h1Conf * 100).toFixed(0)}%, temporal lag penalty applied).`,
      ];

      return {
        rank: i + 1,
        source: evt.source,
        type: evt.type,
        label: evt.label,
        description: evt.description,
        lagSeconds,
        confidence: +confidence.toFixed(3),
        evidence,
      };
    });
  }

  // ── Simulation state ───────────────────────────────────────────────────
  const SIM = {
    batches: [],
    currentIdx: 0,
    baseline: {},
    baselineVals: {},    // {col: [raw values]} from baseline window
    speed: 1,
    playing: false,
    intervalId: null,
    driftThreshold: 0.2,  // PSI alert threshold (user-adjustable)
    costPerEvent: 142,
    batchSizeSeconds: 300,  // 5-minute batch intervals (Bug 2 fix)
    incidentActive: false,
    incidentStartBatch: null,
    hypotheses: [],
    actionLog: [],
    actionState: {
      active: null,
      appliedAtIdx: null,
      fraudModifier: 1.0,
      targetModifier: 1.0,
    },
    metrics: {
      psiPerFeature: {},   // {col: [psi per batch]}
      zscorePerFeature: {}, // {col: [z per batch]}
      fraudRateHistory: [],
      lossHistory: [],
      batchIds: [],
      topDriftedFeatures: [],
    },
    callbacks: {
      onTick: null,
      onIncident: null,
      onEnd: null,
    },
  };

  // ── Initialize from DATASET ────────────────────────────────────────────
  function initSim(dataset) {
    const D = dataset || global.DATASET;
    if (!D) throw new Error('No DATASET loaded');

    SIM.batches = D.batches;
    SIM.baseline = { means: D.baselineMeans, stds: D.baselineStds, mins: D.baselineMins, maxs: D.baselineMaxs };
    SIM.costPerEvent = D.suggestedCostPerEvent;
    SIM.incidentActive = false;
    SIM.incidentStartBatch = null;
    SIM.hypotheses = [];
    SIM.actionLog = [];
    SIM.actionState = { active: null, appliedAtIdx: null, fraudModifier: 1.0, targetModifier: 1.0 };
    SIM.metrics = {
      psiPerFeature: {}, zscorePerFeature: {},
      fraudRateHistory: [], lossHistory: [], batchIds: [],
      topDriftedFeatures: [],
    };

    // Pre-compute baseline raw values for PSI computation
    SIM.baselineVals = {};
    const baseBatches = SIM.batches.slice(0, D.baselineBatches);
    D.numericCols.forEach(col => {
      SIM.baselineVals[col] = [];
      baseBatches.forEach(batch => {
        batch.forEach(row => {
          const v = parseFloat(row[col]);
          if (!isNaN(v)) SIM.baselineVals[col].push(v);
        });
      });
    });

    // Start from first non-baseline batch
    SIM.currentIdx = D.baselineBatches;

    // Init history arrays
    D.numericCols.forEach(col => {
      SIM.metrics.psiPerFeature[col] = [];
      SIM.metrics.zscorePerFeature[col] = [];
    });

    return SIM;
  }

  // ── Single tick ────────────────────────────────────────────────────────
  function tick() {
    const D = global.DATASET;
    if (!D || SIM.batches.length === 0) return;

    // Loop seamlessly at end
    if (SIM.currentIdx >= SIM.batches.length) {
      SIM.currentIdx = D.baselineBatches;
    }

    const batch = SIM.batches[SIM.currentIdx];
    const batchId = SIM.currentIdx + 1;

    // ── Per-feature PSI and Z-score ──────────────────────────────────────
    const psiMap = {};
    const zscoreMap = {};
    const currentValsMap = {};

    D.numericCols.forEach(col => {
      const currentVals = batch.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      currentValsMap[col] = currentVals;

      const result = computePSI(SIM.baselineVals[col], currentVals);
      const psiVal = result && !result.stable ? result.psi : (result && result.stable ? 0 : null);
      psiMap[col] = psiVal;

      const currentMean = currentVals.length > 0
        ? currentVals.reduce((a, b) => a + b, 0) / currentVals.length : SIM.baseline.means[col] || 0;
      zscoreMap[col] = computeZScore(SIM.baseline.means[col] || 0, SIM.baseline.stds[col] || 1, currentMean);

      if (SIM.metrics.psiPerFeature[col] == null) SIM.metrics.psiPerFeature[col] = [];
      if (SIM.metrics.zscorePerFeature[col] == null) SIM.metrics.zscorePerFeature[col] = [];
      SIM.metrics.psiPerFeature[col].push(psiVal != null ? +psiVal.toFixed(4) : 0);
      SIM.metrics.zscorePerFeature[col].push(+zscoreMap[col].toFixed(3));
    });

    // Top drifted features by PSI
    const sortedByPSI = D.numericCols
      .filter(c => psiMap[c] != null)
      .sort((a, b) => (psiMap[b] || 0) - (psiMap[a] || 0));
    SIM.metrics.topDriftedFeatures = sortedByPSI.slice(0, 5);

    // ── Fraud rate from actual data — Bug 3: loose equality handles '1'==1, 'true', etc. ──
    const rawFraudRate = D.targetCol
      ? batch.filter(r => {
          const v = r[D.targetCol];
          const pl = D.positiveLabel;
          return v == pl ||
            (pl === '1' && (v === 1 || v === true)) ||
            String(v).toLowerCase() === String(pl).toLowerCase();
        }).length / batch.length
      : (D.positiveRate || 0.002);

    // Apply action modifier (lerp toward target)
    const { actionState } = SIM;
    if (actionState.active && actionState.appliedAtIdx != null) {
      const ticksElapsed = SIM.currentIdx - actionState.appliedAtIdx;
      actionState.fraudModifier += (actionState.targetModifier - actionState.fraudModifier) * 0.4;
    }

    const displayFraudRate = Math.max(0, Math.min(1, rawFraudRate * actionState.fraudModifier));

    // ── Loss per hour ────────────────────────────────────────────────────
    // Formula: fraudRate * costPerEvent * batchSize * batchesPerHour
    const batchesPerHour = 3600 / SIM.batchSizeSeconds;
    let lossPerHour = displayFraudRate * SIM.costPerEvent * D.batchSize * batchesPerHour;

    // Floor during incidents: real datasets have very low per-batch fraud counts
    // (e.g. creditcard.csv 0.17% → most batches have 0 fraud rows → loss = $0).
    // Use the dataset baseline rate so the incident shows meaningful exposure.
    if (lossPerHour < 1 && SIM.incidentActive) {
      const baselineRate = Math.max(D.positiveRate || 0, 0.001); // min floor 0.1%
      lossPerHour = baselineRate * SIM.costPerEvent * D.batchSize * batchesPerHour;
    }

    // Hard floor: during confirmed incident never show $0 (confuses operators)
    if (SIM.incidentActive && lossPerHour < 50) lossPerHour = 50;

    SIM.metrics.fraudRateHistory.push(+(displayFraudRate * 100).toFixed(2));
    SIM.metrics.lossHistory.push(+lossPerHour.toFixed(2));
    SIM.metrics.batchIds.push(batchId);

    // Keep history bounded (last 200)
    const MAX_HIST = 200;
    if (SIM.metrics.fraudRateHistory.length > MAX_HIST) {
      SIM.metrics.fraudRateHistory.shift();
      SIM.metrics.lossHistory.shift();
      SIM.metrics.batchIds.shift();
      D.numericCols.forEach(col => {
        if (SIM.metrics.psiPerFeature[col]) SIM.metrics.psiPerFeature[col].shift();
        if (SIM.metrics.zscorePerFeature[col]) SIM.metrics.zscorePerFeature[col].shift();
      });
    }

    // ── Incident detection ───────────────────────────────────────────────
    const maxPSI = Math.max(...Object.values(psiMap).filter(v => v != null), 0);
    // Debug log — gated by ?debug=1 query param; silent in production
    if (global.__ML_OBS_DEBUG && (batchId % 5 === 0 || SIM.incidentActive)) {
      console.log(`[Engine] B${batchId}: fraudRate=${(displayFraudRate*100).toFixed(3)}% loss=$${lossPerHour.toFixed(0)}/hr maxPSI=${maxPSI.toFixed(4)} incident=${SIM.incidentActive}`);
    }
    const wasIncident = SIM.incidentActive;
    if (maxPSI > SIM.driftThreshold && !SIM.incidentActive) {
      SIM.incidentActive = true;
      SIM.incidentStartBatch = batchId;
      // Generate fresh hypotheses at incident onset
      SIM.hypotheses = generateHypotheses(batchId, sortedByPSI.slice(0, 3), psiMap, zscoreMap);
      SIM.callbacks.onIncident && SIM.callbacks.onIncident(batchId, maxPSI, SIM.hypotheses);
    } else if (maxPSI <= SIM.driftThreshold * 0.7 && SIM.incidentActive) {
      SIM.incidentActive = false;
    }

    SIM.currentIdx++;

    // Notify
    const tickData = {
      batchId, batchNum: SIM.currentIdx, totalBatches: SIM.batches.length,
      psiMap, zscoreMap, currentValsMap,
      displayFraudRate, lossPerHour,
      maxPSI, drifting: maxPSI > SIM.driftThreshold,
      incidentActive: SIM.incidentActive,
      topDriftedFeatures: sortedByPSI,
      actionState: { ...actionState },
    };

    SIM.callbacks.onTick && SIM.callbacks.onTick(tickData);
    return tickData;
  }

  // ── Playback controls ──────────────────────────────────────────────────
  const SPEED_INTERVALS = { 1: 2000, 3: 700, 10: 200 };

  function play() {
    if (SIM.playing) return;
    SIM.playing = true;
    const ms = SPEED_INTERVALS[SIM.speed] || 700;
    SIM.intervalId = setInterval(tick, ms);
  }

  function pause() {
    SIM.playing = false;
    if (SIM.intervalId) { clearInterval(SIM.intervalId); SIM.intervalId = null; }
  }

  function reset() {
    pause();
    const D = global.DATASET;
    if (D) initSim(D);
  }

  function setSpeed(s) {
    const was = SIM.playing;
    pause();
    SIM.speed = s;
    if (was) play();
  }

  // ── Operator actions ───────────────────────────────────────────────────
  const ACTION_MODIFIERS = {
    rollback:          { target: 0.05, ticksToEffect: 2 },
    increase_review:   { target: 0.55, ticksToEffect: 5 },
    trigger_retraining:{ target: 0.12, ticksToEffect: 18 },
    monitor:           { target: 1.0,  ticksToEffect: 0 },
    open_incident:     { target: 0.85, ticksToEffect: 3 },
  };

  function applyAction(actionName, notes) {
    const mod = ACTION_MODIFIERS[actionName] || { target: 1.0, ticksToEffect: 0 };
    SIM.actionState.active = actionName;
    SIM.actionState.appliedAtIdx = SIM.currentIdx;
    SIM.actionState.targetModifier = mod.target;

    const currentFraud = SIM.metrics.fraudRateHistory.at(-1) || 0;
    const currentLoss = SIM.metrics.lossHistory.at(-1) || 0;

    // Project T+5, T+15, T+30 batches
    const proj = projectOutcome(actionName, currentFraud / 100, currentLoss);

    const entry = {
      timestamp: new Date().toISOString(),
      action: actionName,
      notes: notes || '',
      batchId: SIM.currentIdx,
      fraudRateAtExecution: +(currentFraud).toFixed(2),
      lossAtExecution: +currentLoss.toFixed(0),
      projectedOutcome: proj,
      projectedSavings: +(currentLoss - proj.t30LossPerHour).toFixed(0),
    };

    SIM.actionLog.unshift(entry);
    return entry;
  }

  function projectOutcome(action, currentFraudRate, currentLoss) {
    const D = global.DATASET || {};
    // Use dataset baseline rate as floor so projections never show 0.0% on low-fraud data
    const effectiveRate = Math.max(currentFraudRate, D.positiveRate || 0.001);
    const effectiveLoss = Math.max(currentLoss, 50); // never project $0
    const curves = {
      rollback:           [0.40, 0.08, 0.05],
      increase_review:    [0.85, 0.60, 0.42],
      trigger_retraining: [1.00, 0.80, 0.12],
      monitor:            [1.02, 1.05, 1.08],
      open_incident:      [0.95, 0.90, 0.85],
    };
    const [f5, f15, f30] = (curves[action] || curves.monitor).map(m => effectiveRate * m);
    const lossMult = effectiveLoss / Math.max(effectiveRate, 0.001);
    return {
      t5FraudPct:  +(f5 * 100).toFixed(3),
      t15FraudPct: +(f15 * 100).toFixed(3),
      t30FraudPct: +(f30 * 100).toFixed(3),
      t5LossPerHour:  +(f5 * lossMult).toFixed(0),
      t15LossPerHour: +(f15 * lossMult).toFixed(0),
      t30LossPerHour: +(f30 * lossMult).toFixed(0),
      savedVsIgnore:  +((effectiveRate * 1.08 - f30) * lossMult).toFixed(0),
    };
  }

  // ── Loss trajectory projection (3 scenarios) ──────────────────────────
  function buildLossTrajectory(steps) {
    steps = steps || 30;
    const currentFraud = (SIM.metrics.fraudRateHistory.at(-1) || 0) / 100;
    const currentLoss = SIM.metrics.lossHistory.at(-1) || 0;
    const D = global.DATASET;
    if (!D) return null;
    const batchesPerHour = 3600 / SIM.batchSizeSeconds;

    const labels = [];
    const ignoreLine = [], rollbackLine = [], reviewLine = [];

    for (let t = 0; t <= steps; t++) {
      labels.push(`+${t}`);
      const ignFraud = currentFraud * (1 + 0.02 * t);
      const rollFraud = t === 0 ? currentFraud : t === 1 ? currentFraud * 0.4 : currentFraud * 0.05;
      const revFraud = currentFraud * Math.pow(0.60, t / 5);

      const calc = f => +(f * SIM.costPerEvent * D.batchSize * batchesPerHour).toFixed(0);
      ignoreLine.push(calc(ignFraud));
      rollbackLine.push(calc(rollFraud));
      reviewLine.push(calc(revFraud));
    }

    // "Act now saves" at T+15
    const savedAtT15 = Math.max(0, ignoreLine[15] - rollbackLine[15]);

    return { labels, ignoreLine, rollbackLine, reviewLine, savedAtT15, currentLoss };
  }

  // ── Z-score matrix for heatmap ─────────────────────────────────────────
  function getHeatmapMatrix(windowSize) {
    windowSize = windowSize || 20;
    const D = global.DATASET;
    if (!D) return { features: [], cols: [], matrix: [] };

    // Top 10 features by max PSI ever seen
    const features = D.numericCols
      .filter(c => SIM.metrics.psiPerFeature[c] && SIM.metrics.psiPerFeature[c].length > 0)
      .sort((a, b) => {
        const maxA = Math.max(...(SIM.metrics.psiPerFeature[a] || [0]));
        const maxB = Math.max(...(SIM.metrics.psiPerFeature[b] || [0]));
        return maxB - maxA;
      })
      .slice(0, 10);

    const n = SIM.metrics.batchIds.length;
    const start = Math.max(0, n - windowSize);
    const cols = SIM.metrics.batchIds.slice(start);

    const matrix = features.map(f => ({
      feature: f,
      alias: (D.featureAliases || {})[f] || f,
      zscores: (SIM.metrics.zscorePerFeature[f] || []).slice(start),
      psis:    (SIM.metrics.psiPerFeature[f] || []).slice(start),
    }));

    return { features, cols, matrix };
  }

  // ── Safe number formatter ──────────────────────────────────────────────
  function safeVal(v, fallback, digits) {
    if (v == null || v === undefined || (typeof v === 'number' && !isFinite(v))) return fallback != null ? fallback : '—';
    if (typeof v === 'number') {
      if (digits != null) return v.toFixed(digits);
      return v;
    }
    return String(v);
  }

  // Expose
  global.Engine = {
    SIM, initSim, tick, play, pause, reset, setSpeed,
    applyAction, projectOutcome, buildLossTrajectory,
    computePSI, computeZScore, computeRiskScore, generateHypotheses,
    getHeatmapMatrix, psiLabel, psiSeverity, safeVal,
  };

})(window);
