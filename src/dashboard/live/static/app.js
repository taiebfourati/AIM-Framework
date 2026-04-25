/* app.js — WS client + live Chart.js updaters */
(() => {
  const WIN = 500;                 // rolling window of samples per chart
  const ACC_WIN = 200;              // rolling accuracy window

  // ── element refs ────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  const statusBadge = $('status-badge');
  const footerStatus = $('footer-status');
  const fsStep = $('fs-step');
  const fsAcc  = $('fs-acc');
  const fsInj  = $('fs-inj');

  // v2: hero-tile current-value displays + boot overlay
  const heroVals = {
    rsrp:  $('hero-rsrp'),
    sinr:  $('hero-sinr'),
    tput:  $('hero-tput'),
    delay: $('hero-delay'),
  };
  const heroTrends = {
    rsrp:  $('trend-rsrp'),
    sinr:  $('trend-sinr'),
    tput:  $('trend-tput'),
    delay: $('trend-delay'),
  };
  const bootOverlay = $('boot-overlay');
  const bootSub     = $('boot-sub');
  // Track the previous hero value per KPI so we can render a trend chip
  // (Δ vs. ~50 samples ago) without recomputing per-frame.
  const heroBaseline = { rsrp: null, sinr: null, tput: null, delay: null };

  function setStatus(text, cls) {
    statusBadge.textContent = text;
    statusBadge.className = 'status' + (cls ? ' ' + cls : '');
  }
  function showBoot(visible, sub) {
    if (!bootOverlay) return;
    if (visible) {
      if (sub && bootSub) bootSub.textContent = sub;
      bootOverlay.removeAttribute('hidden');
    } else {
      bootOverlay.setAttribute('hidden', '');
    }
  }

  // Closed-loop RAN element refs (resolved lazily — kept here for clarity)
  const ranPanel        = $('ran-panel');
  const ranModeBadge    = $('ran-mode-badge');
  const ranActuatorEl   = $('ran-actuator-state');
  const ranFireCountEl  = $('ran-fire-count');
  const ranEffectsEl    = $('ran-effects');
  const ranActionLog    = $('ran-action-log');
  const ranCells = {
    tx_power_offset_db:     $('ran-tx-offset'),
    interference_offset_db: $('ran-interf-offset'),
    sched_priority_boost:   $('ran-sched-boost'),
    mcs_robustness:         $('ran-mcs-robust'),
    serving_cell_id:        $('ran-cell-id'),
    ue_distance_m:          $('ran-ue-dist'),
  };
  // Track previous ran_state values so we can flash cells that changed.
  const ranPrev = {};

  // ── Chart.js config helpers ─────────────────────────────────────────
  const axisColor = 'rgba(200,210,220,0.08)';
  const textColor = '#8b949e';

  function makeLine(ctx, label, color, opts = {}) {
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label,
          data: [],
          borderColor: color,
          backgroundColor: color + '22',
          borderWidth: 1.3,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: true, labels: { color: textColor, font: { size: 10 } } },
          title: { display: false },
        },
        scales: {
          x: { display: false, grid: { color: axisColor } },
          y: {
            grid: { color: axisColor },
            ticks: { color: textColor, font: { size: 10 }, maxTicksLimit: 4 },
            ...(opts.y ?? {}),
          },
        },
      },
    });
  }

  const charts = {
    rsrp:  makeLine($('chart-rsrp'),  'RSRP (dBm)',     '#58a6ff', { y: { suggestedMin: -130, suggestedMax: -70 } }),
    sinr:  makeLine($('chart-sinr'),  'SINR (dB)',      '#3fb950', { y: { suggestedMin: -10,  suggestedMax: 45 } }),
    tput:  makeLine($('chart-tput'),  'Throughput (Mbps)', '#ffbc42', { y: { suggestedMin: 0 } }),
    delay: makeLine($('chart-delay'), 'Delay (ms)',     '#f85149', { y: { suggestedMin: 0, suggestedMax: 80 } }),
    acc:   makeLine($('chart-acc'),   'Rolling accuracy', '#58a6ff', { y: { suggestedMin: 0, suggestedMax: 1 } }),
  };

  // ── rolling buffers ────────────────────────────────────────────────
  const buffers = {
    steps: [], rsrp: [], sinr: [], tput: [], delay: [],
    accSteps: [], accVals: [],
    recent: [],   // {correct} for rolling accuracy
  };

  function pushSample(s) {
    buffers.steps.push(s.step);
    buffers.rsrp.push(s.x[0]);
    buffers.sinr.push(s.x[1]);
    buffers.tput.push(s.x[2]);
    buffers.delay.push(s.x[3]);

    buffers.recent.push(s.correct);
    if (buffers.recent.length > ACC_WIN) buffers.recent.shift();
    const acc = buffers.recent.reduce((a,b)=>a+b,0) / buffers.recent.length;
    buffers.accSteps.push(s.step);
    buffers.accVals.push(acc);

    // v2: paint hero-tile current values + trend chip
    paintHero('rsrp',  s.x[0], 'dBm');
    paintHero('sinr',  s.x[1], 'dB');
    paintHero('tput',  s.x[2], 'Mbps');
    paintHero('delay', s.x[3], 'ms');

    // Trim rolling windows
    while (buffers.steps.length > WIN) {
      buffers.steps.shift();
      buffers.rsrp.shift(); buffers.sinr.shift();
      buffers.tput.shift(); buffers.delay.shift();
    }
    while (buffers.accSteps.length > WIN) {
      buffers.accSteps.shift(); buffers.accVals.shift();
    }
  }

  // v2: hero-tile painter — current value + Δ-vs-baseline chip.
  // Baseline rebases every ~50 samples so the chip stays meaningful as
  // the stream evolves rather than diverging forever from the first sample.
  let __heroFrame = 0;
  function paintHero(key, val, _unit) {
    const valEl = heroVals[key];
    const trEl  = heroTrends[key];
    if (!valEl) return;
    // Choose decimals per metric so each number looks proportional
    const dec = (key === 'tput' || key === 'delay') ? 1 : 1;
    valEl.textContent = Number(val).toFixed(dec);
    if (heroBaseline[key] === null) heroBaseline[key] = val;
    if (trEl) {
      const d = val - heroBaseline[key];
      const sign = d >= 0 ? '+' : '';
      trEl.textContent = `${sign}${d.toFixed(1)}`;
      trEl.classList.toggle('up',   d > 0.05);
      trEl.classList.toggle('down', d < -0.05);
    }
    __heroFrame++;
    if (__heroFrame % 50 === 0) heroBaseline[key] = val;
  }

  function refreshCharts() {
    charts.rsrp.data.labels  = buffers.steps;
    charts.rsrp.data.datasets[0].data = buffers.rsrp;
    charts.sinr.data.labels  = buffers.steps;
    charts.sinr.data.datasets[0].data = buffers.sinr;
    charts.tput.data.labels  = buffers.steps;
    charts.tput.data.datasets[0].data = buffers.tput;
    charts.delay.data.labels = buffers.steps;
    charts.delay.data.datasets[0].data = buffers.delay;
    charts.acc.data.labels   = buffers.accSteps;
    charts.acc.data.datasets[0].data = buffers.accVals;

    charts.rsrp.update('none');
    charts.sinr.update('none');
    charts.tput.update('none');
    charts.delay.update('none');
    charts.acc.update('none');

    const last = buffers.recent;
    const acc = last.length ? (last.reduce((a,b)=>a+b,0) / last.length) : 0;
    fsAcc.textContent = (acc * 100).toFixed(1) + '%';
  }

  // ~30 FPS refresh loop
  setInterval(refreshCharts, 66);

  // ── detectors ──────────────────────────────────────────────────────
  function fmt(n, d = 3) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    return Number(n).toFixed(d);
  }

  function onDetector(ev) {
    const apply = (id, metric, sub, fire) => {
      const card = $('det-' + id);
      const mEl  = $(id + '-metric');
      mEl.textContent = metric;
      card.querySelector('.det-sub').textContent = sub;
      card.querySelector('.det-state').textContent = fire ? 'FIRED' : 'ok';
      card.classList.toggle('fire', !!fire);
      // v2: also drive the corner pill so detector state is glanceable
      const pill = card.querySelector('.det-pill');
      if (pill) pill.textContent = fire ? 'fired' : 'ok';
    };
    if (ev.ddd) apply('ddd',
      `${fmt(ev.ddd.ks_max_p, 3)} · ${fmt(ev.ddd.mmd, 3)}`,
      'KS p-max · MMD²', ev.ddd.triggered);
    if (ev.dpd) apply('dpd',
      `${fmt(ev.dpd.if_rate, 3)} · ${fmt(ev.dpd.mahal_max, 2)}`,
      'IF rate · Mahal max', ev.dpd.triggered);
    if (ev.cdd) apply('cdd',
      `${fmt(ev.cdd.ph_stat, 2)} · ${fmt(ev.cdd.perf_drop, 3)}`,
      'PH stat · perf drop', ev.cdd.triggered);
    if (ev.cpd) apply('cpd',
      `${fmt(ev.cpd.shadow_divergence, 3)}`,
      'shadow divergence', ev.cpd.triggered);
  }

  // ── logs / model history ───────────────────────────────────────────
  function appendLog(target, stepText, sev, reasons, tail) {
    const box = $(target);
    if (!box) { console.warn('appendLog: missing element', target); return; }
    // Drop the empty-state placeholder on first real entry so it doesn't
    // sit above the genuine log rows.
    const placeholder = box.querySelector('.log-empty');
    if (placeholder) placeholder.remove();
    const el = document.createElement('div');
    el.className = 'log-entry';
    el.innerHTML =
      `<span class="step">${stepText}</span>` +
      `<span class="sev sev-${sev}">${sev}</span>` +
      `<span class="text">${reasons.join(', ')}</span>` +
      (tail ? `<span class="tail" style="color:var(--fg-dim);margin-left:auto;font-size:10px;">${tail}</span>` : '');
    box.appendChild(el);
    while (box.children.length > 50) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
  }

  function renderModelHistory(versions) {
    const tl = $('model-timeline');
    tl.innerHTML = '';
    for (const v of versions) {
      const chip = document.createElement('div');
      const src = (v.source || '').toLowerCase();
      const srcClass = src.includes('local') ? 'local'
                     : src.includes('cloud') ? 'cloud'
                     : src.includes('mtp-e') || src.includes('external') ? 'external'
                     : 'initial';
      chip.className = 'ver-chip ' + srcClass;
      chip.title = `source=${v.source} acc=${(v.accuracy*100).toFixed(1)}%`;
      chip.textContent = `v${v.version} · ${srcClass}`;
      tl.appendChild(chip);
    }
    // Update MLIN/MLIO cards with the latest version
    const latest = versions[versions.length - 1];
    if (latest) {
      $('slot-mlin').querySelector('.slot-ver').textContent = 'v' + latest.version;
    }
    // v2: badge in the section head shows total model count
    const cnt = $('model-count');
    if (cnt) cnt.textContent = versions.length + ' versions';
  }

  // ── Closed-loop RAN ────────────────────────────────────────────────
  function fmtRanCell(key, val) {
    if (val === null || val === undefined || Number.isNaN(val)) return '—';
    switch (key) {
      case 'tx_power_offset_db':
      case 'interference_offset_db':
        return `${val >= 0 ? '+' : ''}${Number(val).toFixed(1)} dB`;
      case 'sched_priority_boost':
        return `${Number(val).toFixed(1)} ms`;
      case 'mcs_robustness':
        return Number(val).toFixed(2);
      case 'serving_cell_id':
        return `#${val}`;
      case 'ue_distance_m':
        return `${Math.round(Number(val))} m`;
      default:
        return String(val);
    }
  }

  function applyRanState(rs) {
    if (!rs) return;
    for (const key of Object.keys(ranCells)) {
      const el = ranCells[key];
      if (!el) continue;
      const v   = rs[key];
      const txt = fmtRanCell(key, v);
      if (el.textContent !== txt) {
        el.textContent = txt;
        // Flash the cell briefly if value moved (skip first-paint when prev is undefined)
        if (ranPrev[key] !== undefined && ranPrev[key] !== v) {
          const card = el.parentElement;
          card.classList.remove('changed');
          // Force reflow so consecutive updates retrigger the transition
          void card.offsetWidth;
          card.classList.add('changed');
          setTimeout(() => card.classList.remove('changed'), 700);
        }
        ranPrev[key] = v;
      }
    }
    renderActiveEffects(rs.active_effects || []);
  }

  function renderActiveEffects(effects) {
    if (!ranEffectsEl) return;
    if (!effects.length) {
      ranEffectsEl.innerHTML = '<span class="ran-effects-empty">none</span>';
      return;
    }
    ranEffectsEl.innerHTML = '';
    for (const e of effects) {
      const chip = document.createElement('span');
      chip.className = 'ran-effect-chip act-' + (e.type || '');
      const eta = (e.expires_in_s != null && e.expires_in_s !== Infinity)
        ? `<span class="eff-eta">${e.expires_in_s.toFixed(0)}s</span>`
        : '<span class="eff-eta">∞</span>';
      const dlt = (e.delta != null) ? Number(e.delta).toFixed(1) : '';
      chip.innerHTML = `${e.type || '?'} ${dlt ? `Δ${dlt}` : ''} ${eta}`;
      ranEffectsEl.appendChild(chip);
    }
  }

  function applyActuator(a) {
    if (!a || !ranActuatorEl) return;
    const on = !!a.enabled;
    ranActuatorEl.textContent = on ? 'ON' : 'OFF';
    ranActuatorEl.classList.toggle('ran-actuator-on',  on);
    ranActuatorEl.classList.toggle('ran-actuator-off', !on);
    if (ranFireCountEl) {
      ranFireCountEl.textContent =
        `${a.fire_count ?? 0} / ${a.suppressed_count ?? 0}`;
    }
  }

  function appendRanActionLog(action, step) {
    if (!ranActionLog || !action) return;
    const el = document.createElement('div');
    el.className = 'log-entry';
    const tagCls = 'sev act-' + (action.type || '');
    const tail   = `Δ=${Number(action.delta || 0).toFixed(2)} · ` +
                   `${(action.duration_s || 0).toFixed(0)}s`;
    el.innerHTML =
      `<span class="step">s${step ?? action.issued_at_step ?? ''}</span>` +
      `<span class="${tagCls}">${(action.type || '').toUpperCase()}</span>` +
      `<span class="text" title="${(action.reason || '').replace(/"/g,'&quot;')}">` +
        `${action.reason || ''}` +
      `</span>` +
      `<span class="tail" style="color:var(--fg-dim);margin-left:auto;font-size:10px;">${tail}</span>`;
    ranActionLog.appendChild(el);
    while (ranActionLog.children.length > 50) ranActionLog.removeChild(ranActionLog.firstChild);
    ranActionLog.scrollTop = ranActionLog.scrollHeight;
  }

  function flashRanPanel() {
    if (!ranPanel) return;
    ranPanel.classList.remove('flash');
    void ranPanel.offsetWidth;
    ranPanel.classList.add('flash');
    setTimeout(() => ranPanel.classList.remove('flash'), 950);
  }

  function setLiveModeUI(isLive) {
    // Mode badge + panel disabled-state
    if (ranModeBadge) {
      ranModeBadge.textContent = isLive ? '● LIVE' : 'CSV REPLAY';
      ranModeBadge.classList.toggle('mode-live', isLive);
      ranModeBadge.classList.toggle('mode-csv', !isLive);
    }
    if (ranPanel) {
      ranPanel.classList.toggle('ran-panel-disabled', !isLive);
    }
    // Actuator toggle is meaningful only in live mode
    const actToggle = $('actuator-toggle');
    if (actToggle) actToggle.disabled = !isLive;
  }

  // ── WebSocket handling ────────────────────────────────────────────
  let ws = null;
  let reconnectTimer = null;

  // Single decoder reused per frame — TextDecoder allocations are non-trivial
  // and we receive >30 frames/sec.
  const wsDecoder = new TextDecoder('utf-8');

  function connect() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${scheme}://${location.host}/ws`);
    // Server sends binary frames (orjson bytes) to skip an encode pass —
    // request ArrayBuffer so we can decode synchronously instead of dealing
    // with a Blob promise per message.
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      footerStatus.textContent = 'connected';
      clearTimeout(reconnectTimer);
    };
    ws.onclose = () => {
      footerStatus.textContent = 'disconnected — retrying…';
      reconnectTimer = setTimeout(connect, 1000);
    };
    ws.onmessage = (e) => {
      try {
        // Both legacy (text) and new (binary) framing handled.
        const text = (typeof e.data === 'string')
          ? e.data
          : wsDecoder.decode(e.data);
        handle(JSON.parse(text));
      } catch (err) { console.error('ws parse', err); }
    };
  }

  function handle(msg) {
    switch (msg.type) {
      case 'status':
        applyStatus(msg);
        break;
      case 'init':
        footerStatus.textContent = `engine ready · ${msg.rows} rows · features=[${(msg.features||[]).join(', ')}]`;
        break;
      case 'started':
        setStatus('Running', 'running');
        showBoot(false);
        break;
      case 'stopped':
        setStatus('Stopped', '');
        showBoot(false);
        break;
      case 'reset':
        // Clear all buffers + hero baselines so trend chips re-anchor
        for (const k of ['steps','rsrp','sinr','tput','delay','accSteps','accVals','recent']) {
          buffers[k].length = 0;
        }
        for (const k of Object.keys(heroBaseline)) heroBaseline[k] = null;
        for (const k of Object.keys(heroVals))    if (heroVals[k])   heroVals[k].textContent = '—';
        for (const k of Object.keys(heroTrends))  if (heroTrends[k]) { heroTrends[k].textContent = ''; heroTrends[k].className = 'hero-trend'; }
        break;
      case 'samples':
        // First samples after a Boot-Start arrived → dismiss the overlay
        if (msg.items.length) showBoot(false);
        for (const s of msg.items) pushSample(s);
        if (msg.items.length) {
          const last = msg.items[msg.items.length - 1];
          fsStep.textContent = last.step;
        }
        break;
      case 'detector':
        onDetector(msg);
        break;
      case 'mtout':
        appendLog('mtout-log', 's' + msg.step, msg.severity || 'MEDIUM',
                  msg.reasons || [], '');
        break;
      case 'retrain_start':
        appendLog('retrain-log', 's' + msg.step, 'MEDIUM', ['retrain started'], '');
        break;
      case 'retrain_done': {
        // Badge:
        //   DEPLOYED    — green, new model is live
        //   SUCCESS     — trained but not deployed for non-NDT reasons
        //   SKIPPED     — neutral (warmup guard, not a failure)
        //   FAILED      — genuine failure (NDT reject / training error)
        let tag;
        if (msg.deployed)                       tag = 'DEPLOYED';
        else if (msg.status === 'SUCCESS')      tag = 'MEDIUM';
        else if (msg.status === 'SKIPPED')      tag = 'SKIPPED';
        else                                    tag = 'FAILED';
        // Show ndt scores when available so the user can distinguish
        // "NDT rejected a worse candidate" (legitimate) from "pipeline broken".
        const fmt3 = (v) => (typeof v === 'number' ? v.toFixed(3) : null);
        const ndtBits = [];
        const gtStr = fmt3(msg.ndt_gt);
        const psStr = fmt3(msg.ndt_pseudo);
        if (gtStr !== null) ndtBits.push(`gt=${gtStr}`);
        if (psStr !== null) ndtBits.push(`ps=${psStr}`);
        const ndtTxt = msg.ndt_passed === null || msg.ndt_passed === undefined
          ? '—'
          : (msg.ndt_passed ? '✓' : '✗');
        const tail = [
          msg.variant || '—',
          `${msg.duration_s?.toFixed(2) ?? '—'}s`,
          `ndt=${ndtTxt}`,
          ...ndtBits,
        ].join(' · ');
        appendLog('retrain-log', 's' + msg.step, tag,
                  [msg.status, msg.message || ''].filter(Boolean), tail);
        break;
      }
      case 'ndt_dual':
        applyNdtDual(msg);
        break;
      case 'retrain_marker':
        flashAifSlot(msg.variant);
        break;
      case 'reference_refit':
        appendLog('retrain-log', 's' + msg.step, 'LOW',
                  ['reference refit'],
                  `n=${msg.n_samples ?? '—'}`);
        break;
      case 'closed_loop_start':
        appendLog('retrain-log', 's' + msg.step, 'LOW',
                  ['closed-loop decay start'],
                  `dur=${msg.duration_s?.toFixed(2) ?? '—'}s`);
        break;
      case 'closed_loop_end':
        appendLog('retrain-log', 's' + msg.step, 'LOW',
                  ['closed-loop decay end'], '');
        break;
      case 'model_history':
        renderModelHistory(msg.versions || []);
        break;
      case 'ran_action':
        // Embedded ran_state is fresh — paint cells immediately
        if (msg.ran_state) applyRanState(msg.ran_state);
        if (msg.action)    appendRanActionLog(msg.action, msg.step);
        flashRanPanel();
        break;
      case 'error':
        footerStatus.textContent = 'engine error: ' + msg.message;
        statusBadge.textContent = 'Error';
        statusBadge.className = 'status error';
        break;
    }
  }

  // ── NDT dual-score panel ────────────────────────────────────────────
  function applyNdtDual(msg) {
    const fmtPct = v => (v === null || v === undefined || Number.isNaN(v))
      ? '—' : (Number(v) * 100).toFixed(1) + '%';
    const fmtDelta = v => (v === null || v === undefined || Number.isNaN(v))
      ? '—' : ((v >= 0 ? '+' : '') + (Number(v) * 100).toFixed(2) + 'pp');
    const setCls = (el, delta) => {
      if (delta === null || delta === undefined || Number.isNaN(delta)) {
        el.classList.remove('ok', 'bad');
      } else if (delta >= 0) {
        el.classList.add('ok'); el.classList.remove('bad');
      } else {
        el.classList.add('bad'); el.classList.remove('ok');
      }
    };

    const psBase = msg.pseudo_base, psCand = msg.pseudo_cand;
    const gtBase = msg.gt_base,     gtCand = msg.gt_cand;

    $('ndt-ps-base').textContent = fmtPct(psBase);
    $('ndt-ps-cand').textContent = fmtPct(psCand);
    $('ndt-gt-base').textContent = fmtPct(gtBase);
    $('ndt-gt-cand').textContent = fmtPct(gtCand);

    const psDelta = (psBase != null && psCand != null) ? (psCand - psBase) : null;
    const gtDelta = (gtBase != null && gtCand != null) ? (gtCand - gtBase) : null;

    const psDeltaEl = $('ndt-ps-delta');
    const gtDeltaEl = $('ndt-gt-delta');
    psDeltaEl.textContent = fmtDelta(psDelta);
    gtDeltaEl.textContent = fmtDelta(gtDelta);
    setCls(psDeltaEl, psDelta);
    setCls(gtDeltaEl, gtDelta);

    // Bias = pseudo_delta − gt_delta (how much pseudo-label over-estimates)
    const biasEl = $('ndt-bias');
    if (psDelta != null && gtDelta != null) {
      const bias = psDelta - gtDelta;
      biasEl.textContent = (bias >= 0 ? '+' : '') + (bias * 100).toFixed(2) + 'pp';
      biasEl.classList.toggle('bad', Math.abs(bias) > 0.05);
    } else {
      biasEl.textContent = '—';
      biasEl.classList.remove('bad');
    }
  }

  // ── AIF slot flash on retrain deploy ────────────────────────────────
  function flashAifSlot(variant) {
    const card = $('slot-mlin');
    if (!card) return;
    card.classList.remove('flash-local', 'flash-external', 'flash-cloud');
    let cls = 'flash-local';
    if (variant === 'MTP-E' || variant === 'EXTERNAL') cls = 'flash-external';
    else if (variant === 'MTP-C' || variant === 'CLOUD') cls = 'flash-cloud';
    // Force reflow so consecutive retrains re-trigger the animation
    void card.offsetWidth;
    card.classList.add(cls);
    setTimeout(() => card.classList.remove(cls), 1500);
  }

  function applyStatus(s) {
    const inj = s.injection || {};
    fsInj.textContent = summariseInjection(inj);
    fsStep.textContent = s.step ?? 0;
    if (s.running) {
      setStatus(s.paused ? 'Paused' : 'Running', s.paused ? 'paused' : 'running');
      // Engine already up — dismiss any leftover boot overlay (e.g. after
      // a page refresh while a previous session was still running).
      showBoot(false);
    } else {
      setStatus('Idle', '');
    }
    // sync rate slider
    if (s.rate_hz) {
      $('rate-slider').value = s.rate_hz;
      $('rate-label').textContent = Number(s.rate_hz).toFixed(0) + ' Hz';
    }
    // sync injection controls
    syncInject(inj);

    // ── Closed-loop RAN ──────────────────────────────────────────────
    const isLive = !!s.live_mode;
    setLiveModeUI(isLive);
    const liveTog = $('live-mode-toggle');
    if (liveTog && liveTog.checked !== isLive) liveTog.checked = isLive;
    if (s.ran_state) applyRanState(s.ran_state);
    if (s.actuator) {
      applyActuator(s.actuator);
      const actTog = $('actuator-toggle');
      if (actTog && actTog.checked !== !!s.actuator.enabled) {
        actTog.checked = !!s.actuator.enabled;
      }
    }
    // Start/stop the live-mode status poller — keeps ran_state cells fresh
    // between ran_action events (which are sparse if no detector firings).
    setLivePoller(isLive);
  }

  function summariseInjection(inj) {
    const bits = [];
    if (inj.sinr_bias_db)  bits.push(`SINR${inj.sinr_bias_db>0?'+':''}${inj.sinr_bias_db}dB`);
    if (inj.rsrp_bias_db)  bits.push(`RSRP${inj.rsrp_bias_db>0?'+':''}${inj.rsrp_bias_db}dB`);
    if (inj.delay_bias_ms) bits.push(`Δt+${inj.delay_bias_ms}ms`);
    if (inj.tput_scale !== 1) bits.push(`tput×${inj.tput_scale}`);
    if (inj.noise_scale > 1)  bits.push(`noise×${inj.noise_scale.toFixed(1)}`);
    if (inj.poison_mode)   bits.push('POISON');
    return bits.length ? bits.join(' · ') : 'none';
  }

  function syncInject(inj) {
    if (inj.sinr_bias_db  !== undefined) { $('sinr-slider').value = inj.sinr_bias_db;   $('sinr-val').textContent  = inj.sinr_bias_db.toFixed(1) + ' dB'; }
    if (inj.rsrp_bias_db  !== undefined) { $('rsrp-slider').value = inj.rsrp_bias_db;   $('rsrp-val').textContent  = inj.rsrp_bias_db.toFixed(1) + ' dB'; }
    if (inj.delay_bias_ms !== undefined) { $('delay-slider').value = inj.delay_bias_ms; $('delay-val').textContent = inj.delay_bias_ms.toFixed(1) + ' ms'; }
    if (inj.tput_scale    !== undefined) { $('tput-slider').value = inj.tput_scale;     $('tput-val').textContent  = inj.tput_scale.toFixed(2) + '×'; }
    if (inj.noise_scale   !== undefined) { $('noise-slider').value = inj.noise_scale;   $('noise-val').textContent = inj.noise_scale.toFixed(1) + '×'; }
    if (inj.poison_mode   !== undefined) { $('poison-toggle').checked = !!inj.poison_mode; }
    // Mode controls (variant selector, closed-loop, golden NDT)
    if (inj.preferred_variant !== undefined) {
      const v = inj.preferred_variant || 'AUTO';
      const sel = $('variant-select');
      if (sel) sel.value = v;
      const lbl = $('variant-val');
      if (lbl) lbl.textContent = v;
    }
    if (inj.closed_loop_enabled !== undefined) {
      const el = $('closed-loop-toggle');
      if (el) el.checked = !!inj.closed_loop_enabled;
    }
    if (inj.use_golden_ndt !== undefined) {
      const el = $('golden-ndt-toggle');
      if (el) el.checked = !!inj.use_golden_ndt;
    }
  }

  // ── REST helpers ───────────────────────────────────────────────────
  async function post(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: body ? JSON.stringify(body) : '{}',
    });
    return r.json();
  }

  // ── button handlers ────────────────────────────────────────────────
  // v2: Start triggers a 5-30 s synchronous _build_pipeline server-side
  // (CSV load + RTP/MLIN init + AIF compose). Show an immediate "Booting…"
  // status pill + boot overlay so the user gets feedback rather than
  // perceiving the page as frozen. Both are dismissed by the
  // 'started' / first 'samples' event coming back over the WS.
  $('btn-start').onclick = async () => {
    setStatus('Booting…', 'booting');
    showBoot(true,
      'loading Simu5G corpus · fitting detector reference · training initial MLIN');
    try {
      await post('/api/control', {action: 'start'});
    } catch (err) {
      setStatus('Error', 'error');
      showBoot(false);
      footerStatus.textContent = 'start failed: ' + err;
    }
  };
  $('btn-pause').onclick         = () => post('/api/control', {action: 'pause'});
  $('btn-resume').onclick        = () => post('/api/control', {action: 'resume'});
  $('btn-stop').onclick          = () => post('/api/control', {action: 'stop'});
  $('btn-reset').onclick         = () => post('/api/control', {action: 'reset'});
  $('btn-force-retrain').onclick = () => post('/api/control', {action: 'force_retrain'});

  // Rate slider (debounced ~5 Hz)
  const rateSlider = $('rate-slider');
  const rateLabel  = $('rate-label');
  let rateTimer = null;
  rateSlider.oninput = () => {
    rateLabel.textContent = rateSlider.value + ' Hz';
    clearTimeout(rateTimer);
    rateTimer = setTimeout(() => post('/api/rate', {rate_hz: Number(rateSlider.value)}), 150);
  };

  // Injection sliders / toggle
  function bindInject(sliderId, labelId, key, unit, fmtFn) {
    const s = $(sliderId), l = $(labelId);
    let t = null;
    s.oninput = () => {
      const v = Number(s.value);
      l.textContent = (fmtFn ? fmtFn(v) : v.toFixed(1)) + unit;
      clearTimeout(t);
      t = setTimeout(() => post('/api/inject', {[key]: v}), 120);
    };
  }
  bindInject('sinr-slider',  'sinr-val',  'sinr_bias_db',  ' dB');
  bindInject('rsrp-slider',  'rsrp-val',  'rsrp_bias_db',  ' dB');
  bindInject('delay-slider', 'delay-val', 'delay_bias_ms', ' ms');
  bindInject('tput-slider',  'tput-val',  'tput_scale',    '×', v => v.toFixed(2));
  bindInject('noise-slider', 'noise-val', 'noise_scale',   '×');
  $('poison-toggle').onchange = (e) => post('/api/inject', {poison_mode: e.target.checked});

  // Mode controls → /api/mode
  const variantSel = $('variant-select');
  if (variantSel) {
    variantSel.onchange = (e) => {
      const v = e.target.value;
      $('variant-val').textContent = v;
      post('/api/mode', {preferred_variant: v});
    };
  }
  const closedLoopToggle = $('closed-loop-toggle');
  if (closedLoopToggle) {
    closedLoopToggle.onchange = (e) =>
      post('/api/mode', {closed_loop_enabled: e.target.checked});
  }
  const goldenToggle = $('golden-ndt-toggle');
  if (goldenToggle) {
    goldenToggle.onchange = (e) =>
      post('/api/mode', {use_golden_ndt: e.target.checked});
  }

  // ── RAN source + actuator toggles ─────────────────────────────────
  // Live-mode rebuilds the engine server-side, so we must clear local
  // buffers (matching the server's 'reset' event semantics) and re-fetch
  // status to repaint the new mode.
  const liveModeToggle = $('live-mode-toggle');
  if (liveModeToggle) {
    liveModeToggle.onchange = async (e) => {
      const live = !!e.target.checked;
      const actChecked = !!$('actuator-toggle')?.checked;
      // Clear charts immediately — old samples don't belong to the new source
      for (const k of ['steps','rsrp','sinr','tput','delay','accSteps','accVals','recent']) {
        buffers[k].length = 0;
      }
      // Clear the action log too — it belongs to the previous engine instance
      if (ranActionLog) ranActionLog.innerHTML = '';
      // Reset prev-state tracker so the new engine's first paint doesn't flash
      for (const k of Object.keys(ranPrev)) delete ranPrev[k];
      try {
        await post('/api/source', {
          mode: live ? 'live' : 'csv',
          actuator_enabled: live ? actChecked : false,
        });
      } catch (err) {
        console.error('source switch failed', err);
      }
      // Re-pull status so applyStatus can repaint everything with the new mode
      try {
        const st = await (await fetch('/api/status')).json();
        applyStatus(st);
      } catch (err) { /* engine may need a moment to settle */ }
    };
  }

  const actuatorToggle = $('actuator-toggle');
  if (actuatorToggle) {
    actuatorToggle.onchange = (e) =>
      post('/api/mode', {actuator_enabled: !!e.target.checked});
  }

  // ── Live-mode status poller ──────────────────────────────────────
  // ran_state changes (effect expiry, mobility) won't ride a ran_action
  // unless a new action fires. A 2 s poll keeps the panel honest with
  // negligible cost (one HTTP GET, no WS chatter).
  let liveStatusTimer = null;
  function setLivePoller(on) {
    if (on && liveStatusTimer === null) {
      liveStatusTimer = setInterval(async () => {
        try {
          const st = await (await fetch('/api/status')).json();
          if (st.ran_state) applyRanState(st.ran_state);
          if (st.actuator)  applyActuator(st.actuator);
        } catch (_) { /* engine bouncing — next tick will recover */ }
      }, 2000);
    } else if (!on && liveStatusTimer !== null) {
      clearInterval(liveStatusTimer);
      liveStatusTimer = null;
    }
  }

  // Presets
  const presets = {
    'normal':        {sinr_bias_db: 0,   rsrp_bias_db: 0,  delay_bias_ms: 0,  tput_scale: 1,  noise_scale: 1,    poison_mode: false},
    'mild-drift':    {sinr_bias_db: -5,  rsrp_bias_db: -8, delay_bias_ms: 5,  tput_scale: 0.7, noise_scale: 1.5, poison_mode: false},
    'heavy-drift':   {sinr_bias_db: -15, rsrp_bias_db: -20, delay_bias_ms: 20, tput_scale: 0.3, noise_scale: 2.5, poison_mode: false},
    'latency-storm': {sinr_bias_db: 0,   rsrp_bias_db: 0,  delay_bias_ms: 45, tput_scale: 1,  noise_scale: 1,    poison_mode: false},
    'poison':        {sinr_bias_db: -8,  rsrp_bias_db: -10, delay_bias_ms: 15, tput_scale: 0.6, noise_scale: 1.8, poison_mode: true},
  };
  document.querySelectorAll('button[data-preset]').forEach(b => {
    b.onclick = async () => {
      const p = presets[b.dataset.preset];
      syncInject(p);
      await post('/api/inject', p);
    };
  });

  // Initial fetch of status + open WS
  (async () => {
    try {
      const st = await (await fetch('/api/status')).json();
      applyStatus(st);
    } catch (e) { /* engine may not be built yet — that's fine */ }
    connect();
  })();
})();
