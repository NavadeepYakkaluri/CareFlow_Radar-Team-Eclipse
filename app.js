// CareFlow Radar Application Logic — Clinical Hospital Operations Platform
let appData = null;

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

  const targetPane = document.getElementById(tabId);
  if (targetPane) targetPane.style.display = 'block';

  const targetBtn = document.querySelector(`[data-tab="${tabId}"]`);
  if (targetBtn) targetBtn.classList.add('active');
}

// Theme Switcher Logic (Clinical Day Shift vs ICU Night Monitor)
function initThemeToggle() {
  const currentTheme = localStorage.getItem('careflow_theme') || 'light';
  document.documentElement.setAttribute('data-theme', currentTheme);
  updateThemeButtonUI(currentTheme);

  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const active = document.documentElement.getAttribute('data-theme') || 'light';
      const nextTheme = active === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', nextTheme);
      localStorage.setItem('careflow_theme', nextTheme);
      updateThemeButtonUI(nextTheme);
      if (appData && appData.hourly_patterns) {
        renderHourlyChart(appData.hourly_patterns);
      }
    });
  }
}

function updateThemeButtonUI(theme) {
  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (!toggleBtn) return;
  if (theme === 'dark') {
    toggleBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      <span>Clinical Day Shift</span>
    `;
    toggleBtn.title = "Switch to Clinical Day Shift theme";
  } else {
    toggleBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <span>ICU Night Monitor</span>
    `;
    toggleBtn.title = "Switch to ICU Night Monitor theme";
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  initThemeToggle();

  // Tab Event Listeners
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  try {
    const res = await fetch('model_results.json');
    appData = await res.json();
    initApp();
  } catch (err) {
    console.error('Failed to load model_results.json:', err);
  }
});

function initApp() {
  if (!appData) return;

  renderDepartmentCards(appData.department_kpis);
  renderHourlyChart(appData.hourly_patterns);
  renderInvestigationQueue(appData.investigation_queue);
  renderDiagnosticTAT(appData.diagnostic_tat_monitor);
  renderMLBenchmarks(appData.model_performance);
  renderFeatureAttributions(appData.feature_attributions);
  initSimulator();
  initSearchAndFilters();
}

// 1. Department Cards
function renderDepartmentCards(depts) {
  const container = document.getElementById('department-cards-container');
  if (!container) return;

  container.innerHTML = depts.map(d => {
    const isDelayed = d.avg_wait_minutes > 55;
    const loadPercent = Math.min(100, Math.round((d.avg_wait_minutes / 75) * 100));
    const fillColor = isDelayed ? 'var(--status-critical)' : (d.avg_wait_minutes > 48 ? 'var(--status-warning)' : 'var(--status-success)');

    return `
      <div class="dept-card" onclick="filterByDept('${d.department_name}')">
        <div class="dept-card-top">
          <div>
            <div class="dept-name">${d.department_name}</div>
            <div class="dept-type">${d.service_type} &bull; Ward ${d.department_id}</div>
          </div>
          <span class="badge ${isDelayed ? 'badge-amber' : 'badge-cyan'}">
            ${isDelayed ? '⚠ High Bottleneck' : '✔ Normal Flow'}
          </span>
        </div>

        <div class="dept-stats">
          <div class="stat-item">
            <div class="stat-label">Avg Wait Time</div>
            <div class="stat-val ${isDelayed ? 'delayed' : ''}">${d.avg_wait_minutes}m</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Staff / Beds</div>
            <div class="stat-val">${d.staff_count} / ${d.bed_count}</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Post-Service Lag</div>
            <div class="stat-val" style="color:var(--accent-cyan); font-weight:700;">${d.post_service_lag_avg}m</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Severe Delay %</div>
            <div class="stat-val" style="color:var(--text-muted);">${d.severe_delay_rate}%</div>
          </div>
        </div>

        <div class="progress-container">
          <div class="progress-info">
            <span>Bed/Queue Congestion</span>
            <span>${loadPercent}%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: ${loadPercent}%; background: ${fillColor};"></div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// 2. 24-Hour Time-of-Day Hourly Demand SVG Chart
function renderHourlyChart(hourly) {
  const container = document.getElementById('hourly-chart-container');
  if (!container || !hourly.length) return;

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const width = 1100;
  const height = 220;
  const padding = { top: 25, right: 30, bottom: 35, left: 45 };

  const maxWait = Math.max(...hourly.map(h => h.avg_wait)) + 10;
  const maxVol = Math.max(...hourly.map(h => h.patient_volume));

  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const xStep = chartW / (hourly.length - 1);

  // Points for wait time line
  const waitPoints = hourly.map((h, i) => {
    const x = padding.left + i * xStep;
    const y = padding.top + chartH - (h.avg_wait / maxWait) * chartH;
    return `${x},${y}`;
  }).join(' ');

  // Bar elements for volume
  const bars = hourly.map((h, i) => {
    const x = padding.left + i * xStep - 10;
    const barH = (h.patient_volume / maxVol) * (chartH * 0.7);
    const y = padding.top + chartH - barH;
    const barFill = isDark ? 'rgba(56, 189, 248, 0.25)' : 'rgba(2, 132, 199, 0.22)';
    const barStroke = isDark ? 'rgba(56, 189, 248, 0.6)' : 'rgba(2, 132, 199, 0.5)';
    return `
      <rect x="${x}" y="${y}" width="20" height="${barH}" fill="${barFill}" stroke="${barStroke}" stroke-width="1" rx="3">
        <title>Hour ${h.hour}:00 — Volume: ${h.patient_volume} patients, Avg Wait: ${h.avg_wait.toFixed(1)}m</title>
      </rect>
    `;
  }).join('');

  // Axis labels
  const xLabels = hourly.filter((_, i) => i % 2 === 0).map((h, i) => {
    const x = padding.left + (h.hour) * xStep;
    return `<text x="${x}" y="${height - 10}" class="chart-label" text-anchor="middle">${h.hour}:00</text>`;
  }).join('');

  const gridStroke = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)';
  const lineStroke = isDark ? '#f43f5e' : '#dc2626';
  const legendTextColor = isDark ? '#f8fafc' : '#0f2942';

  container.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}">
      <!-- Grid lines -->
      <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="${gridStroke}" stroke-dasharray="4" />
      <line x1="${padding.left}" y1="${padding.top + chartH/2}" x2="${width - padding.right}" y2="${padding.top + chartH/2}" stroke="${gridStroke}" stroke-dasharray="4" />
      <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="${gridStroke}" />

      <!-- Volume bars -->
      ${bars}

      <!-- Wait line -->
      <polyline points="${waitPoints}" fill="none" stroke="${lineStroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />

      <!-- X-axis labels -->
      ${xLabels}

      <!-- Legend -->
      <circle cx="${padding.left + 15}" cy="12" r="4" fill="${lineStroke}" />
      <text x="${padding.left + 25}" y="15" fill="${legendTextColor}" font-size="11" font-weight="600" font-family="var(--font-sans)">Avg Wait (Minutes)</text>

      <rect x="${padding.left + 180}" y="7" width="12" height="10" fill="rgba(2, 132, 199, 0.5)" rx="2" />
      <text x="${padding.left + 200}" y="15" fill="${legendTextColor}" font-size="11" font-weight="600" font-family="var(--font-sans)">Hourly Arrival Volume</text>
    </svg>
  `;
}

// 3. "Investigate First" Action Queue Table
function renderInvestigationQueue(items, filter = 'all') {
  const tbody = document.getElementById('investigation-table-body');
  if (!tbody) return;

  const filtered = items.filter(item => {
    if (filter === 'critical') return item.impact_score >= 7.0;
    if (filter === 'high') return item.impact_score >= 5.0 && item.impact_score < 7.0;
    return true;
  });

  tbody.innerHTML = filtered.map((item, idx) => {
    const isCrit = item.impact_score >= 7.0;
    const isHigh = item.impact_score >= 5.0 && !isCrit;
    const tagClass = isCrit ? 'tag-critical' : (isHigh ? 'tag-high' : 'tag-medium');
    const tagText = isCrit ? 'CRITICAL IMPACT' : (isHigh ? 'HIGH IMPACT' : 'MODERATE');

    return `
      <tr>
        <td>
          <span class="badge-tag ${tagClass}">#${idx + 1} &bull; ${tagText}</span>
        </td>
        <td>
          <strong style="color:var(--text-heading); font-size:0.92rem;">${item.encounter_id}</strong><br>
          <span style="font-size:0.75rem; color:var(--text-dim);">${item.patient_id} (Age ${item.age})</span>
        </td>
        <td>
          <strong style="color:var(--text-heading);">${item.department_name}</strong><br>
          <span style="font-size:0.75rem; color:var(--text-dim);">${item.visit_type}</span>
        </td>
        <td>
          <span class="badge-tag ${item.triage_level === 'Low' ? 'tag-low' : (item.triage_level === 'Critical' ? 'tag-critical' : 'tag-high')}">
            ${item.triage_level}
          </span>
        </td>
        <td><strong style="color:var(--accent-cyan); font-size:0.95rem;">${item.predicted_wait_min}m</strong></td>
        <td>
          <strong style="color:${item.delay_probability > 0.5 ? 'var(--status-critical)' : 'var(--status-success)'};">
            ${Math.round(item.delay_probability * 100)}%
          </strong>
        </td>
        <td><strong style="font-size:1.05rem; color:var(--status-warning);">${item.impact_score}</strong></td>
        <td style="max-width:220px; font-size:0.8rem; color:var(--text-muted); line-height:1.4;">${item.root_cause}</td>
        <td style="max-width:280px; font-size:0.82rem; color:var(--accent-cyan); font-weight:600; line-height:1.4;">
          ${item.recommended_action}
        </td>
      </tr>
    `;
  }).join('');
}

// 4. Diagnostic TAT Table
function renderDiagnosticTAT(tats) {
  const tbody = document.getElementById('diagnostic-tat-body');
  if (!tbody) return;

  tbody.innerHTML = tats.map(t => {
    const isBreach = t.sla_breach_rate > 20;
    return `
      <tr>
        <td><strong style="color:var(--text-heading);">${t.test_type}</strong></td>
        <td style="font-weight:600; color:var(--text-main);">${t.total_count}</td>
        <td><strong style="color:var(--accent-cyan);">${t.overall_avg_tat}m</strong></td>
        <td><span style="color:var(--status-critical); font-weight:700;">${t.urgent_avg_tat}m</span></td>
        <td><span style="color:var(--text-muted);">${t.routine_avg_tat}m</span></td>
        <td>
          <span class="badge-tag ${isBreach ? 'tag-critical' : 'tag-low'}">
            ${t.sla_breach_rate}% Breached
          </span>
        </td>
      </tr>
    `;
  }).join('');
}

// 5. ML Benchmarks
function renderMLBenchmarks(perf) {
  const tbody = document.getElementById('ml-benchmark-body');
  if (!tbody) return;

  const rows = [
    { name: 'Global Mean Baseline', mae: perf.baseline_global_mean.mae, rmse: perf.baseline_global_mean.rmse, r2: perf.baseline_global_mean.r2, status: 'Baseline' },
    { name: 'Triage-Only Heuristic', mae: perf.baseline_triage_only.mae, rmse: perf.baseline_triage_only.rmse, r2: perf.baseline_triage_only.r2, status: 'Clinical Rule' },
    { name: 'From-Scratch Ridge Regressor', mae: perf.ridge_regressor_from_scratch.mae, rmse: perf.ridge_regressor_from_scratch.rmse, r2: perf.ridge_regressor_from_scratch.r2, status: '★ Champion (Scratch)' },
    { name: 'From-Scratch GBDT Ensemble', mae: perf.gbdt_ensemble_from_scratch.mae, rmse: perf.gbdt_ensemble_from_scratch.rmse, r2: perf.gbdt_ensemble_from_scratch.r2, status: 'Ensemble (Scratch)' }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong style="color:var(--text-heading);">${r.name}</strong></td>
      <td><strong style="color:${r.mae < 15 ? 'var(--status-success)' : 'var(--status-critical)'}; font-size:0.95rem;">${r.mae}m</strong></td>
      <td style="color:var(--text-muted);">${r.rmse}m</td>
      <td><strong style="color:var(--accent-cyan);">${r.r2}</strong></td>
      <td><span class="badge-tag ${r.status.includes('Champion') ? 'tag-critical' : 'tag-medium'}">${r.status}</span></td>
    </tr>
  `).join('');
}

// 6. Feature Attributions
function renderFeatureAttributions(feats) {
  const container = document.getElementById('feature-attributions-container');
  if (!container) return;

  container.innerHTML = feats.map(f => {
    const isPos = f.weight > 0;
    const cleanName = f.feature.replace(/_/g, ' ');
    return `
      <div class="feature-row">
        <div class="feat-name" title="${f.feature}">${cleanName}</div>
        <div class="feat-bar-container">
          <div class="feat-bar-fill ${isPos ? '' : 'negative'}" style="width: ${f.importance_pct * 2.2}%;"></div>
        </div>
        <div class="feat-val">${f.weight > 0 ? '+' : ''}${f.weight} (${f.importance_pct}%)</div>
      </div>
    `;
  }).join('');
}

// 7. What-If Simulator Workstation
function initSimulator() {
  const deptSelect = document.getElementById('sim-dept-select');
  const sliderStaff = document.getElementById('slider-staff');
  const sliderTat = document.getElementById('slider-tat');
  const sliderFast = document.getElementById('slider-fasttrack');

  const valStaff = document.getElementById('val-staff');
  const valTat = document.getElementById('val-tat');
  const valFast = document.getElementById('val-fasttrack');

  const currWaitEl = document.getElementById('sim-current-wait');
  const projWaitEl = document.getElementById('sim-projected-wait');
  const deltaEl = document.getElementById('sim-delta');
  const probDropEl = document.getElementById('sim-prob-drop');

  function updateSim() {
    if (!appData || !appData.department_kpis) return;
    const deptId = deptSelect.value;
    const deptObj = appData.department_kpis.find(d => d.department_id === deptId) || appData.department_kpis[0];
    const baseWait = deptObj.avg_wait_minutes;

    const addedStaff = parseInt(sliderStaff.value);
    const tatReduction = parseInt(sliderTat.value);
    const fasttrackOn = parseInt(sliderFast.value);

    valStaff.innerText = `+${addedStaff} Nurses`;
    valTat.innerText = tatReduction === 0 ? 'Standard TAT' : `-${tatReduction}% Lab TAT`;
    valFast.innerText = fasttrackOn ? 'Enabled' : 'Off';

    // Counterfactual ML adjustment:
    const staffSavings = addedStaff * 3.4;
    const tatSavings = (tatReduction * 0.28);
    const fastSavings = fasttrackOn * 7.5;

    const totalSavings = Math.min(baseWait - 15, staffSavings + tatSavings + fastSavings);
    const projectedWait = Math.max(12.0, baseWait - totalSavings);

    const deltaPct = Math.round((totalSavings / baseWait) * 100);
    const probDrop = Math.min(65, Math.round((totalSavings / baseWait) * 55));

    currWaitEl.innerText = `${baseWait.toFixed(1)}m`;
    projWaitEl.innerText = `${projectedWait.toFixed(1)}m`;
    deltaEl.innerText = `-${totalSavings.toFixed(1)} mins (${deltaPct}%)`;
    probDropEl.innerText = `-${probDrop}% Delay Risk`;
  }

  [deptSelect, sliderStaff, sliderTat, sliderFast].forEach(el => {
    if (el) el.addEventListener('input', updateSim);
  });

  updateSim();
}

function initSearchAndFilters() {
  const search = document.getElementById('dept-search');
  if (search) {
    search.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = appData.department_kpis.filter(d => 
        d.department_name.toLowerCase().includes(q) || d.service_type.toLowerCase().includes(q)
      );
      renderDepartmentCards(filtered);
    });
  }

  const priorityFilter = document.getElementById('filter-priority');
  if (priorityFilter) {
    priorityFilter.addEventListener('change', (e) => {
      renderInvestigationQueue(appData.investigation_queue, e.target.value);
    });
  }
}

function filterByDept(deptName) {
  switchTab('tab-investigate');
  const tbody = document.getElementById('investigation-table-body');
  if (!tbody || !appData) return;
  const filtered = appData.investigation_queue.filter(item => item.department_name === deptName);
  renderInvestigationQueue(filtered.length ? filtered : appData.investigation_queue);
}
