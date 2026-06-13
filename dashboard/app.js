function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value || 0);
}

function formatPercent(value) {
  return `${((value || 0) * 100).toFixed(2)}%`;
}

function formatDate(timestamp) {
  if (!timestamp) {
    return "-";
  }
  return new Date(timestamp * 1000).toLocaleTimeString();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function renderStackList(containerId, entries, formatter, colorPicker) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  if (!entries.length) {
    container.innerHTML = '<p class="empty-state">No data available.</p>';
    return;
  }

  const maxValue = Math.max(...entries.map(([, value]) => Math.abs(value)), 1);
  for (const [label, value, extra] of entries) {
    const row = document.createElement("div");
    row.className = "stack-row";
    const width = `${(Math.abs(value) / maxValue) * 100}%`;
    row.innerHTML = `
      <div class="stack-label">
        <strong>${label}</strong>
        <small>${extra || formatter(value)}</small>
        <div class="bar-track">
          <div class="bar-fill ${colorPicker(value)}" style="width:${width}"></div>
        </div>
      </div>
      <strong>${formatter(value)}</strong>
    `;
    container.appendChild(row);
  }
}

function renderEquityChart(points) {
  renderLineChart("equityChart", points);
}

function renderLineChart(containerId, points) {
  const chart = document.getElementById(containerId);
  chart.innerHTML = "";
  if (!points || points.length < 2) {
    return;
  }

  const width = 920;
  const height = 280;
  const padding = 24;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;

  const grid = document.createElementNS("http://www.w3.org/2000/svg", "g");
  for (let index = 0; index < 4; index += 1) {
    const y = padding + (index / 3) * (height - padding * 2);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(padding));
    line.setAttribute("y1", String(y));
    line.setAttribute("x2", String(width - padding));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", "#d6cebf");
    line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
  }

  const pointsString = points
    .map((point, index) => {
      const x = padding + (index / (points.length - 1)) * (width - padding * 2);
      const y = height - padding - ((point - min) / span) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  polyline.setAttribute("points", pointsString);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#177e89");
  polyline.setAttribute("stroke-width", "3");
  polyline.setAttribute("stroke-linecap", "round");
  polyline.setAttribute("stroke-linejoin", "round");

  chart.appendChild(grid);
  chart.appendChild(polyline);
}

function renderRlGenEquityCurve(status) {
  const history = status && status.history ? status.history : [];
  const points = [50000];
  for (const item of history) {
    points.push(points[points.length - 1] + (Number(item.avg_net_pnl) || 0));
  }
  renderLineChart("rlGenEquityChart", points);
}

function renderTrades(trades) {
  const body = document.getElementById("tradesTable");
  body.innerHTML = "";

  const recent = [...trades].reverse().slice(0, 12);
  for (const trade of recent) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${trade.timestamp}</td>
      <td>${trade.strategy_name}</td>
      <td>${trade.side}</td>
      <td>${trade.quantity}</td>
      <td>${trade.exit_reason}</td>
      <td class="${trade.net_pnl >= 0 ? "pnl-positive" : "pnl-negative"}">${formatCurrency(trade.net_pnl)}</td>
    `;
    body.appendChild(row);
  }
}

function renderDecisions(decisions) {
  const body = document.getElementById("decisionsTable");
  body.innerHTML = "";

  const recent = [...decisions].reverse().slice(0, 14);
  for (const decision of recent) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${decision.timestamp}</td>
      <td><span class="status-pill status-${decision.status}">${decision.status}</span></td>
      <td>${decision.regime}</td>
      <td>${decision.selected_strategy || "-"}</td>
      <td>${decision.veto_reason || (decision.notes && decision.notes[0]) || "-"}</td>
    `;
    body.appendChild(row);
  }
}

function renderTaskButtons(activeTaskId) {
  document.querySelectorAll(".task-button").forEach((button) => {
    const running = button.dataset.task === activeTaskId;
    button.classList.toggle("running", running);
    button.disabled = Boolean(activeTaskId);
  });
}

function renderActiveTask(taskState) {
  const active = taskState.active_task || taskState.last_task;
  const consolePanel = document.getElementById("taskConsole");

  if (!active) {
    document.getElementById("activeTaskName").textContent = "No task running";
    document.getElementById("activeTaskStatus").textContent = "idle";
    document.getElementById("activeTaskCode").textContent = "-";
    document.getElementById("activeTaskStarted").textContent = "-";
    consolePanel.textContent = "Waiting for task output...";
    renderTaskButtons(null);
    return;
  }

  document.getElementById("activeTaskName").textContent = active.label;
  document.getElementById("activeTaskStatus").textContent = active.status;
  document.getElementById("activeTaskCode").textContent =
    active.return_code === null || active.return_code === undefined ? "-" : String(active.return_code);
  document.getElementById("activeTaskStarted").textContent = formatDate(active.started_at);
  consolePanel.textContent = active.log_lines.length ? active.log_lines.join("\n") : "No output yet.";
  consolePanel.scrollTop = consolePanel.scrollHeight;
  renderTaskButtons(taskState.active_task ? taskState.active_task.task_id : null);
}

function renderModels(models) {
  renderStackList(
    "modelStatus",
    models.map((model) => [
      model.name,
      model.size_bytes || 0,
      model.exists ? `${model.size_bytes} bytes` : "artifact missing",
    ]),
    (value) => `${value} B`,
    (value) => (value > 0 ? "blue" : "amber"),
  );
}

function renderRlGenStatus(status) {
  const entries = [];
  const best = status && status.best_candidate ? status.best_candidate : null;
  if (status && status.status) {
    entries.push(["status", 1, status.status]);
  }
  if (status && status.generation !== undefined) {
    entries.push(["generation", Number(status.generation) || 0, `gen ${status.generation}`]);
  }
  if (status && status.jax_backend) {
    entries.push(["jax_backend", 1, String(status.jax_backend)]);
  }
  if (status && status.prefilter_backend) {
    entries.push(["prefilter", 1, String(status.prefilter_backend)]);
  }
  if (status && status.jax_prefilter_enabled !== undefined) {
    entries.push([
      "jax_prefilter",
      status.jax_prefilter_enabled ? 1 : -1,
      status.jax_prefilter_enabled ? "enabled" : "disabled",
    ]);
  }
  if (status && status.jax_shortlist_size !== undefined) {
    entries.push([
      "jax_shortlist",
      Number(status.jax_shortlist_size) || 0,
      `top ${status.jax_shortlist_size}`,
    ]);
  }
  if (status && status.numpy_shortlist_size !== undefined && Number(status.numpy_shortlist_size) > 0) {
    entries.push([
      "numpy_shortlist",
      Number(status.numpy_shortlist_size) || 0,
      `top ${status.numpy_shortlist_size}`,
    ]);
  }
  if (status && status.jax_error) {
    entries.push(["jax_error", -1, String(status.jax_error)]);
  }
  if (status && status.unsupported_rocm_target) {
    entries.push([
      "rocm_target",
      -1,
      `unsupported ${status.unsupported_rocm_target}`,
    ]);
  }
  if (status && status.population_size !== undefined) {
    entries.push(["population", Number(status.population_size) || 0, `${status.population_size} candidates`]);
  }
  if (status && status.target_trades_per_candidate !== undefined) {
    entries.push([
      "min_trades",
      Number(status.target_trades_per_candidate) || 0,
      `${status.target_trades_per_candidate} trades target`,
    ]);
  }
  if (best) {
    entries.push(["fitness", Number(best.fitness) || 0, `fitness ${best.fitness}`]);
    entries.push(["avg_net_pnl", Number(best.avg_net_pnl) || 0, formatCurrency(best.avg_net_pnl)]);
    entries.push(["avg_drawdown", Number(best.avg_drawdown) || 0, formatCurrency(best.avg_drawdown)]);
    entries.push(["avg_trades", Number(best.avg_trade_count) || 0, `${best.avg_trade_count} trades`]);
  }
  renderStackList(
    "rlGenStatus",
    entries,
    (value) => String(value),
    (value) => (value >= 0 ? "green" : "red"),
  );
}

function updateSummary(summary, config) {
  document.getElementById("tradeCount").textContent = String(summary.trade_count);
  document.getElementById("winRate").textContent = formatPercent(summary.win_rate);
  document.getElementById("netPnl").textContent = formatCurrency(summary.net_pnl);
  document.getElementById("maxDrawdown").textContent = formatCurrency(summary.max_drawdown);
  document.getElementById("profitFactor").textContent = Number.isFinite(summary.profit_factor)
    ? summary.profit_factor.toFixed(2)
    : "inf";
  document.getElementById("endingEquity").textContent = formatCurrency(summary.ending_equity);
  document.getElementById("approvedCount").textContent = String(summary.approved_count);
  document.getElementById("blockedCount").textContent = String(summary.blocked_count);
  document.getElementById("skippedCount").textContent = String(summary.skipped_count);

  document.getElementById("runtimeMode").textContent = config.environment || "paper";
  document.getElementById("primaryMarket").textContent = config.primary_market || "NQ";
  document.getElementById("datasetWindow").textContent = `Last ${config.max_rows || 0} bars`;
  document.getElementById("lastRefresh").textContent = new Date().toLocaleTimeString();
}

function renderDashboard(payload) {
  updateSummary(payload.summary, payload.config);
  renderEquityChart(payload.summary.equity_curve);
  renderTrades(payload.trades || []);
  renderDecisions(payload.decisions || []);
  renderActiveTask(payload.tasks || {});
  renderModels(payload.models || []);
  renderRlGenStatus(payload.rl_gen_status || {});
  renderRlGenEquityCurve(payload.rl_gen_status || {});

  renderStackList(
    "strategyBreakdown",
    Object.entries(payload.strategy_breakdown || {})
      .map(([name, stats]) => [name, stats.net_pnl, `${stats.count} trades`])
      .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1])),
    (value) => formatCurrency(value),
    (value) => (value >= 0 ? "green" : "red"),
  );

  renderStackList(
    "regimeBreakdown",
    Object.entries(payload.regime_breakdown || {})
      .map(([name, stats]) => [name, stats.net_pnl, `${stats.count} trades`])
      .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1])),
    (value) => formatCurrency(value),
    (value) => (value >= 0 ? "green" : "red"),
  );

  renderStackList(
    "hourBreakdown",
    Object.entries(payload.hour_breakdown || {})
      .map(([name, stats]) => [name, stats.net_pnl, `${stats.count} trades`])
      .sort((left, right) => left[0].localeCompare(right[0])),
    (value) => formatCurrency(value),
    (value) => (value >= 0 ? "green" : "red"),
  );

  renderStackList(
    "vetoBreakdown",
    Object.entries(payload.veto_breakdown || {}).sort((left, right) => right[1] - left[1]),
    (value) => String(value),
    () => "amber",
  );

  renderStackList(
    "patternBreakdown",
    Object.entries(payload.patterns || {}).sort((left, right) => right[1] - left[1]),
    (value) => String(value),
    () => "red",
  );

  renderStackList(
    "decisionStrategyBreakdown",
    Object.entries(payload.decision_strategy_breakdown || {})
      .map(([name, stats]) => [
        name,
        stats.approved - stats.blocked,
        `approved=${stats.approved} blocked=${stats.blocked} skipped=${stats.skipped}`,
      ])
      .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1])),
    (value) => String(value),
    (value) => (value >= 0 ? "green" : "red"),
  );
}

async function loadDashboard() {
  try {
    const payload = await fetchJson("/api/dashboard");
    renderDashboard(payload);
  } catch (error) {
    document.getElementById("taskConsole").textContent = error.message;
  }
}

async function startTask(taskId) {
  try {
    await fetchJson(`/api/tasks/${taskId}/start`, { method: "POST" });
    await loadDashboard();
  } catch (error) {
    alert(error.message);
  }
}

async function stopTask() {
  try {
    await fetchJson("/api/tasks/stop", { method: "POST" });
    await loadDashboard();
  } catch (error) {
    alert(error.message);
  }
}

document.querySelectorAll(".task-button").forEach((button) => {
  button.addEventListener("click", () => startTask(button.dataset.task));
});

document.getElementById("refreshButton").addEventListener("click", loadDashboard);
document.getElementById("stopTaskButton").addEventListener("click", stopTask);

loadDashboard();
setInterval(loadDashboard, 2000);
