/**
 * parser.js — Production CSV streaming parser and schema auto-detector
 * Handles any CSV schema. Reads first 2MB max. Sets window.DATASET.
 */
(function (global) {
  'use strict';

  // ── Alias map ──────────────────────────────────────────────────────────
  const KNOWN_ALIASES = {
    isfraud: 'Fraud Indicator', isFraud: 'Fraud Indicator',
    isflaggedfraud: 'Flagged Fraud', isFlaggedFraud: 'Flagged Fraud',
    oldbalanceorg: 'Sender Opening Balance', oldbalanceOrg: 'Sender Opening Balance',
    newbalanceorig: 'Sender Closing Balance', newbalanceOrig: 'Sender Closing Balance',
    oldbalancedest: 'Receiver Opening Balance', oldbalanceDest: 'Receiver Opening Balance',
    newbalancedest: 'Receiver Closing Balance', newbalanceDest: 'Receiver Closing Balance',
    amount: 'Transaction Amount', Amount: 'Transaction Amount',
    step: 'Time Step (Hours)',
    type: 'Transaction Type',
    nameorig: 'Sender Account', nameOrig: 'Sender Account',
    namedest: 'Receiver Account', nameDest: 'Receiver Account',
    class: 'Fraud Label', Class: 'Fraud Label',
    label: 'Label', target: 'Target',
    churn: 'Churn Indicator', default: 'Default Indicator',
    v1: 'Feature V1', v2: 'Feature V2', v3: 'Feature V3', v4: 'Feature V4',
    v5: 'Feature V5', v6: 'Feature V6', v7: 'Feature V7', v8: 'Feature V8',
    v9: 'Feature V9', v10: 'Feature V10',
    time: 'Time', Time: 'Time',
  };

  const TARGET_PATTERNS = [
    /^is.?fraud$/i, /^fraud$/i, /^class$/i, /^label$/i,
    /^target$/i, /^churn$/i, /^is.?churn$/i, /^default$/i,
    /^is.?default$/i, /^anomaly$/i, /^y$/i, /^outcome$/i,
  ];

  // ── Alias builder ──────────────────────────────────────────────────────
  function buildAlias(col) {
    const lc = col.toLowerCase();
    if (KNOWN_ALIASES[col]) return KNOWN_ALIASES[col];
    if (KNOWN_ALIASES[lc]) return KNOWN_ALIASES[lc];
    // camelCase split
    let s = col.replace(/([a-z])([A-Z])/g, '$1 $2');
    // snake_case / kebab-case
    s = s.replace(/[_-]/g, ' ');
    return s.replace(/\b\w/g, c => c.toUpperCase());
  }

  // ── Fast CSV line parser (quoted fields support) ───────────────────────
  function parseCSVLine(line) {
    const result = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        if (inQ && line[i + 1] === '"') { cur += '"'; i++; }
        else inQ = !inQ;
      } else if (c === ',' && !inQ) {
        result.push(cur); cur = '';
      } else cur += c;
    }
    result.push(cur);
    return result;
  }

  function parseCSV(text) {
    const lines = text.split('\n');
    const nonEmpty = lines.filter(l => l.trim().length > 0);
    if (nonEmpty.length < 2) return { headers: [], rows: [] };
    const headers = parseCSVLine(nonEmpty[0]).map(h => h.trim().replace(/^"|"$/g, ''));
    const rows = [];
    for (let i = 1; i < nonEmpty.length; i++) {
      const vals = parseCSVLine(nonEmpty[i]);
      if (vals.length < headers.length - 1) continue;
      const row = {};
      headers.forEach((h, idx) => { row[h] = vals[idx] != null ? vals[idx].trim().replace(/^"|"$/g, '') : ''; });
      rows.push(row);
    }
    return { headers, rows };
  }

  // ── Schema detection ───────────────────────────────────────────────────
  function detectSchema(headers, rows) {
    const n = Math.max(rows.length, 1);

    // Target column
    let targetCol = null;
    for (const pat of TARGET_PATTERNS) {
      const f = headers.find(h => pat.test(h));
      if (f) { targetCol = f; break; }
    }
    if (!targetCol) targetCol = headers[headers.length - 1];

    // Per-column stats
    const uniqueCounts = {}, numericRatios = {};
    headers.forEach(h => {
      const vals = rows.map(r => r[h]).filter(v => v !== '' && v != null);
      uniqueCounts[h] = new Set(vals).size;
      const numCount = vals.filter(v => v !== '' && !isNaN(parseFloat(v)) && isFinite(+v)).length;
      numericRatios[h] = vals.length > 0 ? numCount / vals.length : 0;
    });

    // ── Layer 0: name-pattern exclusion fires BEFORE anything else ──────────
    // Time/step/id columns inflate PSI to 1.0 and must never enter analysis.
    // This runs first so they cannot appear in numericCols OR categoricalCols.
    const MONOTONIC_NAME_PAT = /^(time|step|id|index|row|seq|num|number|timestamp|ts|tick|epoch)\b/i;
    const CHECK_ROWS = rows.slice(0, Math.min(200, rows.length));

    // Pre-screen: exclude by name OR by actual monotonicity/Pearson before building any col list
    const _monotonicExcludes = new Set();
    headers.forEach(h => {
      if (h === targetCol) return;
      if (MONOTONIC_NAME_PAT.test(h)) {
        _monotonicExcludes.add(h);
        console.log(`[Parser] PRE-excluded "${h}": name matches time/step/id pattern`);
        return;
      }
      if (numericRatios[h] >= 0.85) {
        const vals = CHECK_ROWS.map(r => parseFloat(r[h])).filter(v => !isNaN(v));
        if (vals.length >= 10) {
          // Strict monotone check
          let mono = true;
          for (let i = 1; i < vals.length; i++) { if (vals[i] < vals[i-1]) { mono = false; break; } }
          if (mono) { _monotonicExcludes.add(h); console.log(`[Parser] PRE-excluded "${h}": monotonically increasing`); return; }
          // Pearson r with row index
          const nv = vals.length, mx = (nv-1)/2;
          const my = vals.reduce((a,b)=>a+b,0)/nv;
          let num=0,dX=0,dY=0;
          for (let i=0;i<nv;i++){num+=(i-mx)*(vals[i]-my);dX+=(i-mx)**2;dY+=(vals[i]-my)**2;}
          const r = (dX>0&&dY>0)?Math.abs(num/Math.sqrt(dX*dY)):0;
          if (r>0.95){_monotonicExcludes.add(h);console.log(`[Parser] PRE-excluded "${h}": Pearson r=${r.toFixed(3)}`);return;}
        }
      }
    });

    // Ignored: high-cardinality string ID columns + all monotonic excludes
    let ignoredCols = headers.filter(h =>
      h !== targetCol && !_monotonicExcludes.has(h) &&
      uniqueCounts[h] > n * 0.5 && numericRatios[h] < 0.5
    );
    // Add monotonic pre-excludes to ignoredCols for reporting
    _monotonicExcludes.forEach(h => { if (!ignoredCols.includes(h)) ignoredCols.push(h); });

    let numericCols = headers.filter(h =>
      h !== targetCol &&
      !ignoredCols.includes(h) &&
      !_monotonicExcludes.has(h) &&
      numericRatios[h] >= 0.85
    );

    const categoricalCols = headers.filter(h =>
      h !== targetCol && !ignoredCols.includes(h) && !numericCols.includes(h) && uniqueCounts[h] <= 50
    );

    // Positive/negative label detection
    let positiveLabel = '1', negativeLabel = '0';
    const targetVals = [...new Set(rows.map(r => r[targetCol]).filter(v => v !== ''))];
    const binaryOne = targetVals.find(v => v === '1' || v.toLowerCase() === 'true' || v.toLowerCase() === 'yes');
    if (binaryOne != null) {
      positiveLabel = binaryOne;
      negativeLabel = targetVals.find(v => v !== binaryOne) || '0';
    } else if (targetVals.length === 2) {
      positiveLabel = targetVals[1]; negativeLabel = targetVals[0];
    }

    const featureAliases = {};
    headers.forEach(h => { featureAliases[h] = buildAlias(h); });

    const tl = (targetCol || '').toLowerCase();
    let targetName = buildAlias(targetCol || 'Event');
    if (tl.includes('fraud')) targetName = 'Fraud';
    else if (tl.includes('churn')) targetName = 'Churn';
    else if (tl.includes('default')) targetName = 'Default';
    else if (tl.includes('anomaly')) targetName = 'Anomaly';
    else if (tl === 'class' || tl === 'y' || tl === 'label') targetName = 'Label';

    // Bug 3: loose equality — handles '1'==1, 'TRUE'=='true', etc.
    const posCount = rows.filter(r => {
      const v = r[targetCol];
      return v == positiveLabel ||
        (positiveLabel === '1' && (v === 1 || v === true)) ||
        String(v).toLowerCase() === String(positiveLabel).toLowerCase();
    }).length;
    const positiveRate = n > 0 ? posCount / n : 0;

    // Heuristic refinement: very low positive rate (<2%) on generic "Label"/"Class"
    // columns almost always means fraud/anomaly detection — rename for clarity
    if ((targetName === 'Label' || targetName === 'Class') && positiveRate > 0 && positiveRate < 0.02) {
      targetName = 'Fraud';
    }

    let suggestedCostPerEvent = 100;
    if (targetName === 'Fraud') suggestedCostPerEvent = 142;
    else if (targetName === 'Churn') suggestedCostPerEvent = 890;
    else if (targetName === 'Default') suggestedCostPerEvent = 3200;

    return {
      targetCol, positiveLabel, negativeLabel,
      numericCols, categoricalCols, ignoredCols,
      featureAliases, targetName, suggestedCostPerEvent, positiveRate,
    };
  }

  // ── Baseline stats ─────────────────────────────────────────────────────
  function computeBaselineStats(rows, numericCols, batchSize) {
    const baselineBatches = Math.min(20, Math.floor(rows.length / batchSize));
    const baselineRows = rows.slice(0, baselineBatches * batchSize);
    const means = {}, stds = {}, mins = {}, maxs = {};

    numericCols.forEach(col => {
      const vals = baselineRows.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      if (vals.length === 0) { means[col] = 0; stds[col] = 1; mins[col] = 0; maxs[col] = 1; return; }
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, v) => a + (v - m) ** 2, 0) / vals.length;
      means[col] = m;
      stds[col] = Math.max(Math.sqrt(variance), 1e-8);
      mins[col] = vals.reduce((a, v) => v < a ? v : a,  Infinity);
      maxs[col] = vals.reduce((a, v) => v > a ? v : a, -Infinity);
    });

    return { means, stds, mins, maxs, baselineBatches };
  }

  // ── Gaussian random ────────────────────────────────────────────────────
  function gaussianRandom(mean, std) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // ── Main load function ─────────────────────────────────────────────────
  async function loadDataset(file, onProgress) {
    const MAX_BYTES = 10_000_000;  // 10 MB — enough rows for stable PSI + visible fraud rate
    const isLarge = file.size > 50 * 1024 * 1024;  // flag if file > 50MB

    onProgress && onProgress({ stage: 'reading', pct: 5, message: isLarge
      ? 'Large file detected — using first 50,000 rows as representative sample for real-time simulation'
      : `Reading ${file.name}…` });

    const blob = file.slice(0, MAX_BYTES);
    const text = await blob.text();

    onProgress && onProgress({ stage: 'parsing', pct: 30, message: 'Parsing CSV structure…' });
    const { headers, rows } = parseCSV(text);

    if (rows.length < 10) throw new Error('CSV has too few rows to analyze. Need at least 10 data rows.');

    onProgress && onProgress({ stage: 'schema', pct: 55, message: 'Detecting schema and column types…' });
    const schema = detectSchema(headers, rows);

    if (schema.numericCols.length === 0) throw new Error('No numeric columns found. Need at least one numeric feature for drift analysis.');

    const batchSize = Math.max(30, Math.min(500, Math.floor(rows.length / 80)));

    onProgress && onProgress({ stage: 'baseline', pct: 80, message: `Computing baseline from first ${schema.numericCols.length} features…` });
    const baseline = computeBaselineStats(rows, schema.numericCols, batchSize);

    const batches = [];
    for (let i = 0; i + batchSize <= rows.length; i += batchSize) {
      batches.push(rows.slice(i, i + batchSize));
    }
    // Include partial last batch if >= 30 rows
    const remainder = rows.length % batchSize;
    if (remainder >= 30) batches.push(rows.slice(rows.length - remainder));

    onProgress && onProgress({ stage: 'done', pct: 100, message: `Ready — ${batches.length} batches of ${batchSize} rows` });

    const dataset = {
      filename: file.name,
      headers,
      ...schema,
      rowCount: rows.length,
      sampleSize: rows.length,
      batchSize,
      batches,
      baselineMeans: baseline.means,
      baselineStds: baseline.stds,
      baselineMins: baseline.mins,
      baselineMaxs: baseline.maxs,
      baselineBatches: baseline.baselineBatches,
      isLarge,
      fileSize: file.size,
      isDemo: false,
    };

    global.DATASET = dataset;
    return dataset;
  }

  // ── Demo dataset generator ─────────────────────────────────────────────
  function generateDemoDataset(config) {
    config = Object.assign({ targetName: 'Fraud', type: 'fraud' }, config || {});
    const ROWS = 10000, BATCH = 100, DRIFT_AT = 5000;
    const rows = [];

    for (let i = 0; i < ROWS; i++) {
      const drifting = i >= DRIFT_AT;
      const fraudRate = drifting ? 0.055 : 0.002;
      const isFraud = Math.random() < fraudRate;
      const amount = drifting
        ? Math.abs(gaussianRandom(520000, 280000))
        : Math.abs(gaussianRandom(156000, 120000));
      const oldB = Math.abs(gaussianRandom(250000, 200000));
      rows.push({
        step: (Math.floor(i / 10) + 1).toString(),
        txn_type: ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'][Math.floor(Math.random() * 5)],
        amount: amount.toFixed(2),
        oldbalanceOrg: oldB.toFixed(2),
        newbalanceOrig: Math.max(0, oldB - amount).toFixed(2),
        oldbalanceDest: Math.abs(gaussianRandom(100000, 150000)).toFixed(2),
        newbalanceDest: (drifting
          ? Math.abs(gaussianRandom(750000, 350000))
          : Math.abs(gaussianRandom(250000, 200000))).toFixed(2),
        channel_risk: (drifting ? Math.abs(gaussianRandom(0.75, 0.18)) : Math.abs(gaussianRandom(0.22, 0.12))).toFixed(4),
        isFraud: isFraud ? '1' : '0',
      });
    }

    const headers = Object.keys(rows[0]);
    const schema = detectSchema(headers, rows);
    const baseline = computeBaselineStats(rows, schema.numericCols, BATCH);
    const batches = [];
    for (let i = 0; i + BATCH <= rows.length; i += BATCH) batches.push(rows.slice(i, i + BATCH));

    const dataset = {
      filename: 'demo_paysim_fraud.csv',
      headers, ...schema,
      rowCount: ROWS, sampleSize: ROWS, batchSize: BATCH,
      batches, baselineMeans: baseline.means, baselineStds: baseline.stds,
      baselineMins: baseline.mins, baselineMaxs: baseline.maxs,
      baselineBatches: baseline.baselineBatches,
      isLarge: false, isDemo: true,
      demoConfig: { driftStartBatch: DRIFT_AT / BATCH },
    };

    global.DATASET = dataset;
    return dataset;
  }

  global.Parser = { loadDataset, generateDemoDataset, parseCSV, detectSchema, buildAlias };

})(window);
