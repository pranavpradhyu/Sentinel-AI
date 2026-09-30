/* SentinelAI dashboard controller */
const $ = (s) => document.querySelector(s);
const api = (p) => fetch(p).then((r) => r.json());

const SEV = { Normal: "clear", Probe: "med", DoS: "high", R2L: "high", U2R: "high" };
const severityOf = (r) => (r.zero_day ? "zero" : SEV[r.verdict] || "med");
const label = (r) =>
  r.zero_day ? "Suspected zero-day" : r.verdict === "Normal" ? "Benign" : r.verdict;

const state = { engine: "isolation_forest", sensitivity: 1.0 };
const qs = () => `engine=${state.engine}&sensitivity=${state.sensitivity}`;

let META = null;
let streaming = true;
let timer = null;
const recent = [];

/* ---------------------------------------------------- boot */
async function boot() {
  try {
    META = await api("/api/meta");
  } catch {
    $("#liveLabel").textContent = "engine offline";
    return;
  }
  renderTopStats();
  renderMetrics();
  renderImportance();
  renderEngineCompare();
  renderMatrix();
  $("#liveLabel").textContent = "live";
  startClock();
  startStream();
  wireControls();
  buildPlayground();
}

function renderTopStats() {
  const m = META.metrics;
  $("#topStats").innerHTML = [
    ["accuracy", (m.accuracy * 100).toFixed(0) + "%"],
    ["attack AUC", m.attack_roc_auc.toFixed(2)],
    ["detection", (m.detection_rate * 100).toFixed(0) + "%"],
  ].map(([l, n]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
}

function renderMetrics() {
  const m = META.metrics;
  const cells = [
    [(m.detection_rate * 100).toFixed(0) + "%", "attacks caught"],
    [(m.false_alarm_rate * 100).toFixed(1) + "%", "false alarms"],
    [m.attack_roc_auc.toFixed(3), "attack ROC-AUC"],
    [(m.combined_anomaly_catch_rate * 100).toFixed(0) + "%", "zero-days (both engines)"],
  ];
  $("#metrics").innerHTML = cells
    .map(([n, l]) => `<div class="metric"><div class="n">${n}</div><div class="l">${l}</div></div>`)
    .join("");
}

function renderImportance() {
  const feats = META.top_features.slice(0, 8);
  const max = Math.max(...feats.map((f) => f.importance));
  $("#importanceBars").innerHTML = feats
    .map(
      (f) => `<div class="bar-row">
        <span class="name" title="${f.feature}">${f.feature}</span>
        <div class="bar-track"><div class="bar-fill" data-w="${(f.importance / max) * 100}"></div></div>
        <span class="val">${(f.importance * 100).toFixed(0)}</span>
      </div>`
    )
    .join("");
  requestAnimationFrame(() =>
    document.querySelectorAll(".bar-fill").forEach((b) => (b.style.width = b.dataset.w + "%"))
  );
}

function renderEngineCompare() {
  const e = META.anomaly_engines;
  const order = [
    ["isolation_forest", "Isolation Forest", "var(--signal)"],
    ["autoencoder", "Autoencoder (deep)", "var(--zero)"],
  ];
  $("#engineCompare").innerHTML =
    order
      .map(([k, name, col]) => {
        const d = e[k];
        return `<div class="eng-row">
        <div class="eng-head"><span>${name} <span class="k">${d.kind}</span></span>
          <span class="pct">${(d.catch_rate * 100).toFixed(0)}%</span></div>
        <div class="eng-track"><div class="eng-fill" style="background:${col}" data-w="${d.catch_rate * 100}"></div></div>
      </div>`;
      })
      .join("") +
    `<p class="explain-title" style="margin-top:12px">Combined, they flag
      <b class="sv-zero">${(META.metrics.combined_anomaly_catch_rate * 100).toFixed(0)}%</b>
      of attacks with no attack labels in training — the two paradigms catch different intrusions.</p>`;
  requestAnimationFrame(() =>
    document.querySelectorAll(".eng-fill").forEach((b) => (b.style.width = b.dataset.w + "%"))
  );
}

function renderMatrix() {
  const cm = META.confusion_matrix;
  const cls = META.classes.map((c) => c.slice(0, 4));
  const rowMax = cm.map((row) => Math.max(...row));
  let html = `<div class="mrow"><span class="axis"></span>${cls
    .map((c) => `<span class="axis">${c}</span>`)
    .join("")}</div>`;
  cm.forEach((row, i) => {
    html += `<div class="mrow"><span class="axis">${cls[i]}</span>`;
    row.forEach((v, j) => {
      const t = v / (rowMax[i] || 1);
      const bg = i === j ? `rgba(88,229,196,${0.12 + t * 0.5})` : `rgba(255,93,108,${0.05 + t * 0.55})`;
      html += `<span class="cell" style="background:${bg}" title="${META.classes[i]} predicted as ${META.classes[j]}: ${v}">${v}</span>`;
    });
    html += `</div>`;
  });
  $("#matrix").innerHTML = html;
}

/* ---------------------------------------------------- live stream */
function startStream() {
  clearInterval(timer);
  timer = setInterval(tick, 2200);
  tick();
}
async function tick() {
  if (!streaming) return;
  let data;
  try {
    data = await api(`/api/stream?n=3&${qs()}`);
  } catch {
    return;
  }
  data.feed.forEach((r, i) => setTimeout(() => addRow(r), i * 300));
}

function addRow(r) {
  const sev = severityOf(r);
  const feed = $("#feed");
  const row = document.createElement("div");
  row.className = "flow-row";
  row.dataset.id = r.id;
  const tag =
    r.verdict === "Normal" && !r.zero_day
      ? ""
      : `<span class="tag sv-${sev} bd-${sev}">${(r.threat_score * 100).toFixed(0)}% threat</span>`;
  row.innerHTML = `
    <span class="sev bg-${sev}"></span>
    <span class="verdict sv-${sev}">${label(r)}</span>
    <span class="meta">${r.protocol}/${r.service} · ${fmtBytes(r.src_bytes)}</span>
    ${tag}
    <span class="conf">${(r.confidence * 100).toFixed(0)}%</span>`;
  row.addEventListener("click", () => inspect(r.id));
  feed.prepend(row);
  while (feed.children.length > 14) feed.lastChild.remove();

  recent.unshift(r.zero_day ? Math.max(r.threat_score, 0.85) : r.threat_score);
  if (recent.length > 12) recent.pop();
  updateGauge();
}

function updateGauge() {
  const avg = recent.reduce((a, b) => a + b, 0) / (recent.length || 1);
  const idx = Math.round(avg * 100);
  const C = 351.9;
  const el = $("#gaugeVal");
  el.style.strokeDashoffset = C - (C * idx) / 100;
  $("#gaugeNum").textContent = idx;

  let sev = "clear", stt = "Systems nominal", desc = "Live traffic looks benign. Nothing needs your attention.";
  if (idx >= 66) { sev = "high"; stt = "Active threats"; desc = "The engine is classifying multiple malicious flows in the live feed."; }
  else if (idx >= 33) { sev = "med"; stt = "Elevated activity"; desc = "Some flows are scoring as suspicious. Worth a look."; }
  if (recent.slice(0, 3).some((v) => v >= 0.85) && idx < 66) {
    sev = "zero"; stt = "Anomaly seen"; desc = "A flow the classifier called benign looks abnormal — a possible novel attack.";
  }
  el.style.stroke = `var(--${sev})`;
  const st = $("#stateText");
  st.textContent = stt;
  st.className = "state sv-" + sev;
  $("#stateDesc").textContent = desc;
  const meter = $("#meter");
  meter.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const s = document.createElement("span");
    s.className = "seg" + (i < idx / 10 ? " bg-" + sev : "");
    meter.appendChild(s);
  }
}

/* ---------------------------------------------------- shared: contributions */
function contribHTML(explanation) {
  const maxc = Math.max(...explanation.map((e) => Math.abs(e.contribution))) || 1;
  return explanation
    .map((e) => {
      const w = (Math.abs(e.contribution) / maxc) * 50;
      const pos = e.contribution > 0;
      const style = pos
        ? `left:50%;width:${w}%;background:var(--high)`
        : `right:50%;width:${w}%;background:var(--signal)`;
      return `<div class="contrib-row">
        <span class="cf">${e.feature} <span class="cv">= ${fmtVal(e.value)}</span></span>
        <div class="contrib-bar-wrap"><div class="mid"></div><div class="contrib-bar" style="${style}"></div></div>
      </div>`;
    })
    .join("");
}

/* ---------------------------------------------------- inspector */
async function inspect(id) {
  switchTab("inspect");
  $("#inspectEmpty").style.display = "none";
  const body = $("#inspectBody");
  body.style.display = "block";
  body.innerHTML = `<div style="padding:30px;text-align:center"><span class="spinner"></span></div>`;
  let r;
  try {
    r = await api(`/api/flow/${id}?${qs()}`);
  } catch {
    body.innerHTML = `<div class="inspector-empty">Couldn't reach the engine for that flow.</div>`;
    return;
  }
  const sev = severityOf(r);
  const truth =
    r.ground_truth && r.ground_truth !== r.verdict
      ? `<span class="why">labelled ${r.ground_truth} in the test set</span>`
      : `<span class="why">${META.family_desc[r.verdict] || ""}</span>`;
  body.innerHTML = `
    <div class="verdict-banner">
      <span class="sev bg-${sev}" style="width:14px;height:14px"></span>
      <div><div class="big sv-${sev}">${label(r)}</div>${truth}</div>
      <div class="num"><div class="p sv-${sev}">${(r.confidence * 100).toFixed(0)}%</div><div class="pl">confidence</div></div>
    </div>
    <p class="explain-title">Why this verdict — features pushing the score up (red) or down toward benign (teal):</p>
    <div class="contrib">${contribHTML(r.explanation)}</div>`;
  animateContribs();
}

function animateContribs() {
  requestAnimationFrame(() =>
    document.querySelectorAll(".contrib-bar").forEach((b) => {
      const w = b.style.width;
      b.style.width = "0";
      requestAnimationFrame(() => (b.style.width = w));
    })
  );
}

/* ---------------------------------------------------- playground */
const SERVICES = ["http", "private", "ftp", "ftp_data", "smtp", "domain", "telnet", "ssh", "pop_3", "finger", "other", "ecr_i"];
const FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "S1", "OTH"];
const CTRLS = [
  { k: "protocol_type", t: "select", label: "Protocol", opts: ["tcp", "udp", "icmp"] },
  { k: "service", t: "select", label: "Service (destination)", opts: SERVICES },
  { k: "flag", t: "select", label: "Connection state", opts: FLAGS },
  { k: "src_bytes", t: "range", label: "Bytes sent", min: 0, max: 10000, step: 10 },
  { k: "dst_bytes", t: "range", label: "Bytes received", min: 0, max: 10000, step: 10 },
  { k: "count", t: "range", label: "Connections to host (2s)", min: 0, max: 511, step: 1 },
  { k: "srv_count", t: "range", label: "Connections to service (2s)", min: 0, max: 511, step: 1 },
  { k: "serror_rate", t: "range", label: "SYN-error rate", min: 0, max: 1, step: 0.05 },
  { k: "same_srv_rate", t: "range", label: "Same-service rate", min: 0, max: 1, step: 0.05 },
  { k: "diff_srv_rate", t: "range", label: "Different-service rate", min: 0, max: 1, step: 0.05 },
  { k: "dst_host_count", t: "range", label: "Host history count", min: 0, max: 255, step: 1 },
  { k: "dst_host_srv_count", t: "range", label: "Host service count", min: 0, max: 255, step: 1 },
];
const PRESETS = {
  normal: { protocol_type: "tcp", service: "http", flag: "SF", src_bytes: 320, dst_bytes: 4200, count: 6, srv_count: 6, serror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, dst_host_count: 80, dst_host_srv_count: 60 },
  scan: { protocol_type: "tcp", service: "private", flag: "S0", src_bytes: 0, dst_bytes: 0, count: 34, srv_count: 2, serror_rate: 1, same_srv_rate: 0.1, diff_srv_rate: 0.9, dst_host_count: 210, dst_host_srv_count: 4 },
  flood: { protocol_type: "tcp", service: "http", flag: "S0", src_bytes: 0, dst_bytes: 0, count: 500, srv_count: 500, serror_rate: 1, same_srv_rate: 1, diff_srv_rate: 0, dst_host_count: 255, dst_host_srv_count: 255 },
  r2l: { protocol_type: "tcp", service: "ftp", flag: "SF", src_bytes: 130, dst_bytes: 45, count: 2, srv_count: 2, serror_rate: 0, same_srv_rate: 1, diff_srv_rate: 0, dst_host_count: 22, dst_host_srv_count: 10 },
};
let play = { ...PRESETS.normal };
let playTimer = null;

function buildPlayground() {
  const wrap = $("#playControls");
  wrap.innerHTML = CTRLS.map((c) => {
    if (c.t === "select") {
      return `<div class="ctrl"><div class="row"><label>${c.label}</label></div>
        <select data-k="${c.k}">${c.opts.map((o) => `<option value="${o}">${o}</option>`).join("")}</select></div>`;
    }
    return `<div class="ctrl"><div class="row"><label>${c.label}</label><span class="cval" id="cv-${c.k}"></span></div>
      <input type="range" data-k="${c.k}" min="${c.min}" max="${c.max}" step="${c.step}" /></div>`;
  }).join("");
  wrap.querySelectorAll("[data-k]").forEach((el) => {
    el.addEventListener("input", () => {
      const k = el.dataset.k;
      play[k] = el.type === "range" ? Number(el.value) : el.value;
      const cv = $("#cv-" + k);
      if (cv) cv.textContent = play[k];
      queuePlay();
    });
  });
  syncPlayControls();
  classifyPlay();
}

function syncPlayControls() {
  CTRLS.forEach((c) => {
    const el = $(`#playControls [data-k="${c.k}"]`);
    if (!el) return;
    el.value = play[c.k];
    const cv = $("#cv-" + c.k);
    if (cv) cv.textContent = play[c.k];
  });
}

function queuePlay() {
  clearTimeout(playTimer);
  playTimer = setTimeout(classifyPlay, 220);
}

async function classifyPlay() {
  let r;
  try {
    r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engine: state.engine, sensitivity: state.sensitivity, flow: play }),
    }).then((x) => x.json());
  } catch {
    return;
  }
  const sev = severityOf(r);
  const probs = META.classes
    .map((c) => {
      const p = r.probabilities[c] || 0;
      const s = SEV[c] || "med";
      return `<div class="prob-bar"><span class="pl">${c}</span>
        <div class="pt"><div class="pf bg-${s}" style="width:${p * 100}%"></div></div>
        <span class="pv">${(p * 100).toFixed(0)}</span></div>`;
    })
    .join("");
  $("#playOut").innerHTML = `
    <div class="play-verdict">
      <div class="pv-top"><span class="sev bg-${sev}" style="width:13px;height:13px"></span>
        <span class="big sv-${sev}">${label(r)}</span>
        <span class="pv-conf sv-${sev}">${(r.confidence * 100).toFixed(0)}%</span></div>
      <div class="prob-bars">${probs}</div>
      <div class="anom-pair">
        ${ring("Isolation Forest", r.iso_score)}
        ${ring("Autoencoder", r.deep_score)}
      </div>
      <p class="explain-title" style="margin-top:14px">Top drivers of this verdict:</p>
      <div class="contrib">${contribHTML(r.explanation.slice(0, 6))}</div>
    </div>`;
  animateContribs();
}

function ring(name, score) {
  const C = 175.9, off = C - C * score;
  const col = score > 0.7 ? "var(--zero)" : score > 0.4 ? "var(--med)" : "var(--signal)";
  return `<div class="anom-cell"><div class="lab">${name}</div>
    <div class="ring"><svg width="66" height="66" viewBox="0 0 66 66">
      <circle cx="33" cy="33" r="28" fill="none" stroke="var(--line)" stroke-width="6"/>
      <circle cx="33" cy="33" r="28" fill="none" stroke="${col}" stroke-width="6" stroke-linecap="round"
        stroke-dasharray="${C}" stroke-dashoffset="${off}"/></svg>
      <div class="n" style="color:${col}">${(score * 100).toFixed(0)}</div></div></div>`;
}

/* ---------------------------------------------------- upload / pcap */
let uploadSource = "csv";
function wireUpload() {
  const drop = $("#drop"), input = $("#fileInput");
  $("#sourceSeg").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      uploadSource = b.dataset.src;
      $("#sourceSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      const pcap = uploadSource === "pcap";
      $("#dropTitle").textContent = pcap ? "Drop a .pcap capture, or tap to choose" : "Drop a CSV of flows, or tap to choose";
      $("#dropHint").textContent = pcap
        ? "Raw packets → reassembled into flows on the server → scored"
        : "NSL-KDD format · first 5,000 flows scored · nothing leaves the request";
      input.setAttribute("accept", pcap ? ".pcap,.pcapng,.cap" : ".csv,.txt");
      $("#batchResult").innerHTML = "";
    })
  );
  ["dragenter", "dragover"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && handleFile(ev.dataTransfer.files[0]));
  input.addEventListener("change", () => input.files[0] && handleFile(input.files[0]));
}

async function handleFile(file) {
  const out = $("#batchResult");
  const pcap = uploadSource === "pcap";
  out.innerHTML = `<div style="padding:24px;text-align:center"><span class="spinner"></span> ${pcap ? "Reassembling flows from" : "Scoring"} ${file.name}…</div>`;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("engine", state.engine);
  fd.append("sensitivity", state.sensitivity);
  let data;
  try {
    data = await fetch(pcap ? "/api/pcap" : "/api/upload", { method: "POST", body: fd }).then((r) => r.json());
  } catch {
    out.innerHTML = `<div class="inspector-empty">Upload failed — check the file and try again.</div>`;
    return;
  }
  if (data.error) {
    out.innerHTML = `<div class="inspector-empty">${data.error}</div>`;
    return;
  }
  const cards = [
    [data.total, pcap ? "flows reassembled" : "flows scored"],
    [data.attacks, "attacks"],
    [data.zero_days, "zero-day flags"],
    [((data.attacks / data.total) * 100).toFixed(0) + "%", "malicious"],
  ];
  const rows = data.rows
    .map((r) => {
      const sev = severityOf(r);
      return `<tr><td>${r.row}</td><td class="sv-${sev}">${r.zero_day ? "zero-day" : r.verdict}</td>
        <td>${(r.confidence * 100).toFixed(0)}%</td><td>${(r.threat_score * 100).toFixed(0)}%</td></tr>`;
    })
    .join("");
  out.innerHTML = `
    <div class="result-summary">${cards
      .map(([n, l]) => `<div class="rs"><div class="n">${n}</div><div class="l">${l}</div></div>`)
      .join("")}</div>
    <div class="table-scroll"><table class="result-table">
      <thead><tr><th>#</th><th>verdict</th><th>confidence</th><th>threat</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    ${data.total > data.rows.length ? `<p class="explain-title">Showing first ${data.rows.length} of ${data.total} rows.</p>` : ""}`;
}

/* ---------------------------------------------------- ui plumbing */
function wireControls() {
  $("#toggleStream").addEventListener("click", () => {
    streaming = !streaming;
    $("#toggleStream").textContent = streaming ? "⏸ Pause" : "▶ Resume";
    $("#toggleStream").classList.toggle("on", streaming);
  });
  $("#stepBtn").addEventListener("click", async () => {
    const d = await api(`/api/stream?n=1&${qs()}`);
    d.feed.forEach(addRow);
  });
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchTab(t.dataset.pane)));

  $("#engineSeg").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      state.engine = b.dataset.engine;
      $("#engineSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      classifyPlay();
    }));
  $("#sensRange").addEventListener("input", (e) => {
    state.sensitivity = Number(e.target.value);
    $("#sensVal").textContent = state.sensitivity.toFixed(1) + "×";
    queuePlay();
  });

  $("#presets").querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      play = { ...PRESETS[b.dataset.preset] };
      $("#presets").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      syncPlayControls();
      classifyPlay();
    }));

  wireUpload();
}
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.pane === name));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === "pane-" + name));
}

function startClock() {
  const upd = () => ($("#clock").textContent = new Date().toISOString().slice(11, 19) + " UTC");
  upd();
  setInterval(upd, 1000);
}

/* ---------------------------------------------------- helpers */
function fmtBytes(b) {
  b = Number(b) || 0;
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}
function fmtVal(v) {
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(2);
  return v;
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/service-worker.js").catch(() => {}));
}
boot();
