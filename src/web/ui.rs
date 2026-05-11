//! Embedded SPA. Shows the node graph + tasks + transcripts + files.

pub const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>bureau-rs</title>
<style>
  :root {
    --bg: #0f1115; --bg2: #161922; --bg3: #1d2230;
    --fg: #d8dde6; --muted: #7a8499;
    --accent: #5aa9e6; --good: #5eb988; --warn: #d6a14a; --bad: #d96a6a;
    --running: #a36ae6;
    --border: #283044;
    --mono: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace;
  }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--fg); font-family: var(--mono); font-size: 13px; }
  #app { display: grid; grid-template-rows: 1fr auto; height: 100vh; min-height: 0; }
  /* Resizable columns: a flex row of [panel | splitter | panel | splitter | panel].
     Splitters drag-resize the adjacent panels via JS. min-width: 0 lets each
     panel actually shrink below its content's natural width (long paths!). */
  #panels { display: flex; flex-direction: row; min-height: 0; min-width: 0; }
  .panel { display: flex; flex-direction: column; min-height: 0; min-width: 0;
           overflow: hidden; }
  .panel.col-1 { width: 320px; flex: 0 0 auto; border-right: 1px solid var(--border); }
  .panel.col-2 { flex: 1 1 auto; border-right: 1px solid var(--border); }
  .panel.col-3 { width: 360px; flex: 0 0 auto; }
  .splitter { width: 4px; cursor: col-resize; background: transparent; flex: 0 0 auto; }
  .splitter:hover, .splitter.dragging { background: var(--accent); }
  .panel-h { padding: 8px 12px; background: var(--bg2); border-bottom: 1px solid var(--border); font-weight: 600; flex: 0 0 auto; }
  /* Panel body lays its sections out in a column. Each open section gets a
     share of the height; closed sections collapse to their summary. The
     SCROLL happens inside each section's body, not on the panel. */
  .panel-body { display: flex; flex-direction: column; padding: 6px; flex: 1 1 auto;
                min-height: 0; min-width: 0; gap: 6px; overflow: hidden; }
  #status { background: var(--bg2); border-top: 1px solid var(--border); padding: 6px 12px; display: flex; gap: 16px; align-items: center; font-size: 12px; }
  /* Color semantics — ONE meaning per color:
     - blue  (.entity) = an entity-type label (e.g. "node", "task"). Static.
     - grey  (.status.queued)   = not started / pending.
     - purple-throbbing (.status.running) = in progress.
     - green (.status.ok)       = done / succeeded.
     - red   (.status.failed)   = errored / unresolved failure.
     - amber (.status.retrying) = transient, will retry.
     Phase names (spec/iface/tests/impl/debug/opt) carry NO color of their
     own — they're plain text. The phase's STATUS gets the color. */
  .entity { display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-size: 11px; background: #233a55; color: #b9d9ff; }
  .phase  { display: inline-block; padding: 1px 4px; font-size: 11px;
            color: var(--muted); }
  .status { display: inline-block; padding: 1px 6px; border-radius: 3px;
            font-size: 11px; }
  .status.queued, .status.pending, .status.not_started, .status.skipped {
    background: #25272f; color: #888;
  }
  .status.running, .status.in_progress {
    background: #3a2f55; color: #d4b8ff; animation: pulse 1.2s infinite;
  }
  .status.ok, .status.done, .status.resolved {
    background: #1f3a2a; color: #98e3b3;
  }
  .status.failed, .status.permanent, .status.error {
    background: #3a1f1f; color: #f0a0a0;
  }
  .status.retrying {
    background: #4a3f1f; color: #f0d59e; animation: pulse 1.6s infinite;
  }
  .status.resolved { opacity: 0.65; }
  /* A combined phase+status pill: phase name in muted text, background
     tinted by status. Used for per-stage badges in the node + task UI. */
  .phase-pill { display: inline-block; padding: 1px 6px; border-radius: 3px;
                font-size: 11px; }
  .phase-pill.queued, .phase-pill.pending, .phase-pill.not_started,
  .phase-pill.skipped { background: #25272f; color: #888; }
  .phase-pill.running, .phase-pill.in_progress {
    background: #3a2f55; color: #d4b8ff; animation: pulse 1.2s infinite;
  }
  .phase-pill.ok, .phase-pill.done { background: #1f3a2a; color: #98e3b3; }
  .phase-pill.failed { background: #3a1f1f; color: #f0a0a0; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
  .node { padding: 6px 8px; border-radius: 4px; margin-bottom: 4px; }
  .node:hover { background: var(--bg2); }
  .node.selected { background: var(--bg3); }
  .node .name { font-weight: 600; display: flex; align-items: center; gap: 4px; }
  .node .stages { display: flex; gap: 3px; margin-top: 4px; flex-wrap: wrap; }
  .tree-chevron { display: inline-block; width: 12px; text-align: center;
                  cursor: pointer; color: var(--muted); user-select: none; }
  .tree-chevron:hover { color: var(--accent); }
  .tree-chevron-spacer { display: inline-block; width: 12px; text-align: center;
                         color: var(--border); user-select: none; }
  .phase-pill.clickable { cursor: pointer; }
  .phase-pill.clickable:hover { outline: 1px solid var(--accent); }
  .children { padding-left: 14px; border-left: 1px solid var(--border); margin-left: 4px; }
  .task { padding: 4px 6px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .task:hover { background: var(--bg2); }
  .task.selected { background: var(--bg3); }
  .transcript .entry { border-bottom: 1px dashed var(--border); padding: 6px 4px; }
  /* Border-left distinguishes who is speaking: model entries get a
     stronger accent stripe; bureau entries get a muted one. No
     per-cycle-role color anywhere. */
  .transcript .entry.speaker-model  { border-left: 2px solid var(--accent); padding-left: 6px; }
  .transcript .entry.speaker-bureau { border-left: 2px solid #3a4055; padding-left: 6px; }
  /* Speaker pill (model | bureau) — the only colored badge in the
     transcript header. Blue = "this is a label", same family as the
     entity badges in the left panel. */
  .speaker { display: inline-block; padding: 0 6px; border-radius: 3px; font-size: 10px;
             font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
  .speaker.model  { background: #233a55; color: #b9d9ff; }
  .speaker.bureau { background: #2a2f3d; color: #aab2c5; }
  /* Cycle role (writer/critic/reviser/judge) is plain text. */
  .role { display: inline-block; padding: 0 4px; font-size: 10px; color: var(--muted);
          letter-spacing: 0.04em; text-transform: uppercase; }
  code.tool-name { background: var(--bg2); padding: 0 4px; border-radius: 3px;
                   font-size: 11px; color: var(--fg); }
  .transcript .entry .h { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); cursor: pointer; }
  .transcript .entry pre { white-space: pre-wrap; font-size: 11px; margin: 4px 0 0; word-break: break-all; }
  .transcript .entry.system pre { color: #c8b89e; }
  .transcript .entry.assistant_text pre { color: #d8e8f0; }
  .transcript .entry.tool_call pre { color: #a0c8a0; }
  .transcript .entry.tool_result pre { color: #c8a8e0; }
  .transcript .entry.tool_result.failed pre { color: var(--bad); }
  .transcript .entry.tool_definitions .defs { display: block; }
  .transcript .entry.tool_definitions.collapsed .defs { display: none; }
  .transcript .entry.tool_definitions .tool-def { margin: 4px 0; padding: 4px 6px; background: var(--bg2); border-radius: 3px; }
  .transcript .entry.tool_definitions .tool-def .tool-name { font-weight: 600; color: #b0d4e8; }
  .transcript .entry.tool_definitions .tool-def .tool-desc { white-space: pre-wrap; font-size: 11px; color: #a8b8c8; margin-top: 2px; }
  .collapsed pre { display: none; }
  pre.preview { background: var(--bg2); padding: 8px; border-radius: 4px; white-space: pre-wrap; font-size: 12px; word-break: break-all; margin: 0; }
  /* Each section is a flex container itself: a sticky-ish summary plus a
     scrollable body. When .open it gets `flex: 1` and shares the panel
     vertically with its open siblings; when not .open it collapses to
     just the summary line. We do NOT use <details>/<summary> for this —
     the native disclosure widget has unreliable interaction with flex
     layout (summary doesn't always behave as a flex item, breaking the
     height chain to .sec-body). Plain divs + a toggled `open` class are
     predictable. */
  .section { border: 1px solid var(--border); border-radius: 4px;
             background: var(--bg); display: flex; flex-direction: column;
             min-height: 0; flex: 0 0 auto; overflow: hidden; }
  .section.open { flex: 1 1 0; min-height: 80px; }
  .sec-summary { font-weight: 600; padding: 6px 8px; cursor: pointer;
                 flex: 0 0 auto; user-select: none;
                 display: flex; align-items: center; gap: 6px; }
  .sec-summary::before { content: "▸"; color: var(--muted); display: inline-block;
                         transition: transform 0.1s; }
  .section.open > .sec-summary::before { transform: rotate(90deg); }
  .sec-body { padding: 4px 8px; overflow: auto; flex: 1 1 auto; min-height: 0; }
  .section:not(.open) .sec-body { display: none; }
  .file-tree li { list-style: none; padding: 1px 0; }
  .file-tree a { color: var(--fg); cursor: pointer; }
  .file-tree a:hover { color: var(--accent); }
  .issue { padding: 6px 8px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .issue:hover { background: var(--bg2); }
  .issue .imsg { font-size: 12px; margin-top: 2px; }
  .issue.permanent .imsg { color: var(--bad); }
  .issue.retrying  .imsg { color: var(--warn); }
  .issue.resolved  .imsg { color: var(--muted); text-decoration: line-through; }
  .issue.resolved  { opacity: 0.65; }
  button { background: var(--bg3); color: var(--fg); border: 1px solid var(--border); padding: 3px 8px; border-radius: 3px; cursor: pointer; font: inherit; font-size: 11px; }
  button:hover { background: var(--bg2); }
  button.reset-btn { padding: 0 6px; margin-left: 4px; font-size: 10px; }
  button.reset-btn:hover { background: var(--bad); border-color: var(--bad); }
  .muted { color: var(--muted); }
</style>
</head>
<body>
<div id="app">
<div id="panels">
  <div class="panel col-1">
    <div class="panel-h">Graph + Tasks <span class="muted" id="counts"></span></div>
    <div class="panel-body">
      <div class="section open">
        <div class="sec-summary">Node graph</div>
        <div class="sec-body" id="graph-tree"></div>
      </div>
      <div class="section open">
        <div class="sec-summary">Recent tasks</div>
        <div class="sec-body" id="task-list"></div>
      </div>
      <div class="section">
        <div class="sec-summary">Issues <span class="muted" id="issues-count"></span></div>
        <div class="sec-body" id="issues-list"></div>
      </div>
    </div>
  </div>
  <div class="splitter" data-resize="col-1" data-edge="right"></div>
  <div class="panel col-2">
    <div class="panel-h">Transcript <span class="muted" id="task-title"></span></div>
    <div class="panel-body">
      <div class="section open">
        <div class="sec-summary">Turn-by-turn transcript</div>
        <div class="sec-body transcript" id="transcript">
          <div class="muted">Select a task on the left to view its transcript.</div>
        </div>
      </div>
    </div>
  </div>
  <div class="splitter" data-resize="col-3" data-edge="left"></div>
  <div class="panel col-3">
    <div class="panel-h">Files</div>
    <div class="panel-body">
      <div class="section open">
        <div class="sec-summary">
          Workdir <span class="muted" id="files-count"></span>
          <select id="files-source" style="margin-left:auto; font-size:11px;">
            <option value="main">main</option>
          </select>
        </div>
        <div class="sec-body"><ul class="file-tree" id="file-tree"></ul></div>
      </div>
      <div class="section">
        <div class="sec-summary">Git log <span class="muted" id="git-count"></span></div>
        <div class="sec-body" id="git-log" style="font-size: 11px;"></div>
      </div>
      <div class="section">
        <div class="sec-summary">Preview <span class="muted" id="preview-h"></span></div>
        <div class="sec-body"><pre class="preview" id="preview"></pre></div>
      </div>
    </div>
  </div>
</div>
<div id="status">
  <span><b>Sched:</b> <span id="sched">—</span></span>
  <span><b>Nodes:</b> <span id="node-count">0</span></span>
  <span><b>Tasks:</b> <span id="task-count">0</span></span>
  <span class="muted" id="status-issues"></span>
  <span style="flex:1"></span>
  <span class="muted">Tokens: <span id="tokens">0/0</span></span>
  <span class="muted">USD: <span id="cost">$0.00</span></span>
  <button onclick="api('/api/pause','POST')">Pause</button>
  <button onclick="api('/api/resume','POST')">Resume</button>
  <button onclick="api('/api/checkpoint','POST').then(r => alert('saved: '+(r.path||'')))">Checkpoint</button>
  <button onclick="api('/api/stop','POST')">Stop</button>
</div>
</div>
<script>
let state = null, issues = [], selectedTaskId = null, collapsed = {};
// Per-node-id: are this node's children hidden in the tree view? Default
// false (children shown). Persists across re-renders within a session.
let nodeCollapsed = {};

// ---- Section open/close ----
// Each .section toggles its `open` class when its summary is clicked.
// When open, the section participates in the panel-body's flex layout
// and shares vertical space with its open siblings; when closed, it
// collapses to just the summary line. State persists per-section to
// localStorage (the `id` of the .sec-body inside is the persistence
// key — every section already has a unique inner id).
function initSections() {
  for (const sec of document.querySelectorAll('.section')) {
    const body = sec.querySelector('.sec-body');
    const key = body && body.id ? 'sec-open-' + body.id : null;
    if (key) {
      const stored = localStorage.getItem(key);
      if (stored === '1') sec.classList.add('open');
      else if (stored === '0') sec.classList.remove('open');
    }
    const summary = sec.querySelector('.sec-summary');
    if (summary) {
      summary.addEventListener('click', () => {
        sec.classList.toggle('open');
        if (key) localStorage.setItem(key, sec.classList.contains('open') ? '1' : '0');
      });
    }
  }
}

// ---- Resizable columns ----
// Each .splitter resizes ONE adjacent panel — the one with a fixed width
// (col-1 and col-3); the middle col-2 has flex: 1 and just reflows. The
// `data-resize` attribute names the panel; `data-edge` says which edge of
// that panel the splitter sits on, which controls the drag direction:
//   - edge=right  (col-1's right edge): drag right → panel grows.
//   - edge=left   (col-3's left edge):  drag right → panel shrinks.
// Width persists to localStorage so user preferences stick across reloads.
function initSplitters() {
  for (const sp of document.querySelectorAll('.splitter')) {
    const target = sp.dataset.resize;
    const edge = sp.dataset.edge || 'right';
    const panel = target ? document.querySelector('.panel.' + target) : null;
    if (!panel) continue;
    const stored = localStorage.getItem('panel-w-' + target);
    if (stored) panel.style.width = stored;
    sp.addEventListener('mousedown', e => {
      e.preventDefault();
      sp.classList.add('dragging');
      const startX = e.clientX;
      const startW = panel.getBoundingClientRect().width;
      function onMove(ev) {
        const dx = ev.clientX - startX;
        const w = edge === 'right'
          ? Math.max(120, startW + dx) // splitter on panel's right; right-drag grows panel
          : Math.max(120, startW - dx); // splitter on panel's left; right-drag shrinks panel
        panel.style.width = w + 'px';
      }
      function onUp() {
        sp.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        localStorage.setItem('panel-w-' + target, panel.style.width);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }
}

async function api(path, method='GET', body=null) {
  const r = await fetch(path, {
    method,
    headers: body ? {'Content-Type':'application/json'} : {},
    body: body ? JSON.stringify(body) : null,
  });
  if (r.headers.get('content-type')?.includes('application/json')) return r.json();
  return r.text();
}

function fmt(n) { return Number(n).toLocaleString(); }
function escapeHtml(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function formatJsonish(s) { try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s; } }

async function load() {
  state = await api('/api/state');
  render();
  refreshFiles(); refreshGit(); refreshIssues();
}

// Indexes rebuilt once per render() so the recursive node/stage walk
// doesn't go quadratic. Without these, `findTaskFor` was O(all-tasks)
// per stage-pill (×6 pills × N nodes per render) and the children
// lookup was O(all-nodes) per parent. Under load (many nodes, thousands
// of tasks) those dominated each render's CPU.
let _taskIndex = null;       // Map "<node_id>|<stage>" -> latest task
let _childIndex = null;      // Map parent_id -> [child_node, ...]

function rebuildIndexes() {
  _taskIndex = new Map();
  for (const t of Object.values(state.tasks || {})) {
    const key = t.node_id + '|' + t.stage;
    const cur = _taskIndex.get(key);
    if (!cur || (t.started_at || '') > (cur.started_at || '')) {
      _taskIndex.set(key, t);
    }
  }
  _childIndex = new Map();
  for (const n of Object.values(state.graph?.nodes || {})) {
    if (n.parent != null) {
      let bucket = _childIndex.get(n.parent);
      if (!bucket) { bucket = []; _childIndex.set(n.parent, bucket); }
      bucket.push(n);
    }
  }
}

// Per-section dirty flags. Without these every SSE event re-rendered
// the WHOLE page including the node graph — and rebuilding the graph
// DOM from scratch restarted the throb animation on every in-progress
// pill, which the user perceived as flickering. We now mark only the
// affected sections dirty on each event and the next rAF tick only
// repaints those.
const dirty = { header: true, graph: true, tasks: true, transcript: true };
function markDirty(what) { dirty[what] = true; }
function markAllDirty() {
  dirty.header = true; dirty.graph = true; dirty.tasks = true; dirty.transcript = true;
}

function render() {
  if (!state) return;
  rebuildIndexes();
  if (dirty.header) {
    document.getElementById('sched').textContent = state.scheduler;
    const nodeCount = Object.keys(state.graph?.nodes || {}).length;
    const taskCount = Object.keys(state.tasks || {}).length;
    document.getElementById('node-count').textContent = nodeCount;
    document.getElementById('task-count').textContent = taskCount;
    document.getElementById('counts').textContent = `${nodeCount} nodes, ${taskCount} tasks`;
    document.getElementById('tokens').textContent =
      `${fmt(state.total_cost?.input_tokens||0)} in / ${fmt(state.total_cost?.output_tokens||0)} out`;
    document.getElementById('cost').textContent = `$${(state.estimated_cost_usd||0).toFixed(4)}`;
    dirty.header = false;
  }
  if (dirty.graph) { renderGraph(); dirty.graph = false; }
  if (dirty.tasks) { renderTasks(); dirty.tasks = false; }
  if (dirty.transcript) {
    if (selectedTaskId) renderTranscript();
    dirty.transcript = false;
  }
}

function renderGraph() {
  const tree = document.getElementById('graph-tree');
  tree.innerHTML = '';
  if (!state.graph || !state.graph.root) {
    tree.innerHTML = '<div class="muted">Graph not yet bootstrapped.</div>';
    return;
  }
  renderNodeRecursive(tree, state.graph.root);
}

// Find the task that ran for (node_id, stage). If none yet, returns null.
// Picks the most recently-started one (later attempts override earlier).
function findTaskFor(nodeId, stage) {
  if (!_taskIndex) return null;
  return _taskIndex.get(nodeId + '|' + stage) || null;
}

function renderNodeRecursive(parent, nodeId) {
  const n = state.graph.nodes[nodeId];
  if (!n) return;
  const children = (_childIndex && _childIndex.get(nodeId)) || [];
  const isCollapsed = !!nodeCollapsed[nodeId];
  const div = document.createElement('div');
  div.className = 'node';
  const stages = ['spec','iface','tests','impl','debug','opt'];
  // Each stage is shown as a phase-pill: the phase NAME is plain text;
  // its background color is driven by the stage's STATUS (grey / purple
  // throbbing / green / red). No per-phase color. Click jumps to that
  // stage's task transcript when one exists.
  const stageBadges = stages.map(s => {
    const st = n.stages[s] || 'not_started';
    const clickable = (st === 'in_progress' || st === 'done' || st === 'failed');
    return `<span class="phase-pill ${st}${clickable ? ' clickable' : ''}" `
         + `data-node-id="${n.id}" data-stage="${s}" `
         + `title="${s}: ${st}${clickable ? ' (click to view transcript)' : ''}">${s}</span>`;
  }).join(' ');
  // Chevron is a clickable disclosure widget for the children subtree.
  // Only shown when there ARE children.
  const chevron = children.length
    ? `<span class="tree-chevron" data-node-id="${n.id}" title="${isCollapsed ? 'Expand' : 'Collapse'} subtree">${isCollapsed ? '▸' : '▾'}</span>`
    : `<span class="tree-chevron-spacer">·</span>`;
  div.innerHTML = `
    <div class="name">
      ${chevron}
      <span class="entity">node</span> ${escapeHtml(n.name)}${n.crate_boundary ? ' <span class="muted">(crate)</span>' : ''}
      <button class="reset-btn" data-node-id="${n.id}" title="Reset all stages of this node and re-run">↻</button>
    </div>
    <div class="muted" style="font-size:10px;">${escapeHtml(n.description.slice(0, 100))}</div>
    <div class="stages">${stageBadges}</div>`;
  div.querySelector('.reset-btn').onclick = (e) => {
    e.stopPropagation();
    if (!confirm(`Reset node '${n.name}' and all dependents back to NotStarted? This will re-run their stages.`)) return;
    api('/api/reset_node', 'POST', { node_id: n.id, cascade: true })
      .then(r => {
        const names = (r && r.reset) ? r.reset.join(', ') : '';
        console.log('reset', names);
      })
      .catch(err => alert('reset failed: ' + err));
  };
  // Click chevron → toggle subtree collapse.
  const chev = div.querySelector('.tree-chevron');
  if (chev) {
    chev.onclick = (e) => {
      e.stopPropagation();
      nodeCollapsed[nodeId] = !nodeCollapsed[nodeId];
      render();
    };
  }
  // Click a phase-pill in a runnable state → jump to that task's transcript.
  for (const pill of div.querySelectorAll('.phase-pill.clickable')) {
    pill.onclick = (e) => {
      e.stopPropagation();
      const t = findTaskFor(pill.dataset.nodeId, pill.dataset.stage);
      if (t) { selectTask(t.id); }
    };
  }
  parent.appendChild(div);
  // Children (unless collapsed by user).
  if (children.length && !isCollapsed) {
    const ch = document.createElement('div');
    ch.className = 'children';
    parent.appendChild(ch);
    for (const c of children) renderNodeRecursive(ch, c.id);
  }
}

function renderTasks() {
  const el = document.getElementById('task-list');
  el.innerHTML = '';
  // Dedupe by (node_name, stage): retries and integrator passes create
  // fresh task UUIDs for the same (node, stage), which made the list
  // look like the same row appearing 3-4 times. We keep only the
  // most-recently-started one per (node, stage) and show retry count
  // separately. The user can still click into older attempts from
  // the issues list or the graph's phase pills.
  const grouped = new Map(); // "node|stage" -> { latest, count, totalCost }
  for (const t of Object.values(state.tasks || {})) {
    const key = t.node_name + '|' + t.stage;
    let g = grouped.get(key);
    if (!g) { g = { latest: t, count: 0, totalIn: 0, totalOut: 0 }; grouped.set(key, g); }
    g.count++;
    g.totalIn += (t.cost?.input_tokens || 0);
    g.totalOut += (t.cost?.output_tokens || 0);
    if ((t.started_at || '') > (g.latest.started_at || '')) g.latest = t;
  }
  // Sort newest-first by started_at.
  const rows = [...grouped.values()].sort(
    (a, b) => (b.latest.started_at || '').localeCompare(a.latest.started_at || '')
  );
  for (const g of rows.slice(0, 30)) {
    const t = g.latest;
    const div = document.createElement('div');
    div.className = 'task' + (t.id === selectedTaskId ? ' selected' : '');
    const tokens = g.totalIn + g.totalOut;
    const attemptsTag = g.count > 1 ? ` <span class="muted">×${g.count}</span>` : '';
    div.innerHTML = `
      <span class="entity">task</span>
      <strong>${escapeHtml(t.node_name)}</strong>${attemptsTag}
      <span class="phase">${t.stage}</span>
      <span class="status ${t.status}">${t.status}</span>
      <span class="muted">${fmt(tokens)} tok</span>`;
    div.onclick = () => { selectTask(t.id); };
    el.appendChild(div);
  }
}

// Which side of the framework/model boundary an entry came from.
// Mirror of TranscriptEntry::speaker() in src/tools.rs.
function speakerFor(kind) {
  const t = kind?.type;
  return (t === 'assistant_text' || t === 'tool_call') ? 'model' : 'bureau';
}

// Friendly entry-kind label for the header.
function kindLabel(kind) {
  switch (kind?.type) {
    case 'system':           return 'system prompt';
    case 'user_prompt':      return 'user prompt';
    case 'assistant_text':   return 'assistant text';
    case 'tool_definitions': return 'tool definitions';
    case 'tool_call':        return 'tool call';
    case 'tool_result':      return 'tool result';
    case 'note':             return 'note';
    case 'error':            return 'error';
    default:                 return kind?.type || 'note';
  }
}

// Select a task and lazily fetch its transcript (which /api/state no
// longer ships, to keep the polled state-payload small). The SSE
// `transcript_appended` handler keeps the in-memory copy live after the
// initial fetch.
async function selectTask(id, after) {
  selectedTaskId = id;
  // Render once immediately so the selection highlight and task header
  // update without waiting on the fetch.
  render();
  if (id) {
    try {
      const transcript = await api('/api/task_transcript?id=' + encodeURIComponent(id));
      const t = state.tasks?.[id];
      if (t && Array.isArray(transcript)) {
        t.transcript = transcript;
        render();
      }
    } catch (e) {
      console.error('fetch task transcript failed', e);
    }
  }
  if (typeof after === 'function') after();
}

function renderTranscript() {
  const el = document.getElementById('transcript');
  const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
  const prevTop = el.scrollTop;
  const t = state.tasks?.[selectedTaskId];
  if (!t) { el.innerHTML = '<div class="muted">Task not found.</div>'; return; }
  document.getElementById('task-title').textContent =
    `${t.node_name} · ${t.stage} · ${t.status}`;
  el.innerHTML = '';
  (t.transcript || []).forEach((e, i) => {
    const cls = (e.kind && e.kind.type) || 'note';
    const speaker = speakerFor(e.kind);    // 'bureau' or 'model'
    const role = e.role || null;           // 'writer' / 'critic' / ... or null
    const failed = e.kind?.type === 'tool_result' && e.kind?.ok === false;
    const div = document.createElement('div');
    div.className = 'entry ' + cls + ' speaker-' + speaker + (failed ? ' failed' : '');
    // Header: [speaker] [role] · entry-kind · tool-name? · ✓/✗?
    let header = `<span class="speaker ${speaker}">${speaker}</span>`;
    if (role) header += ` <span class="role">${role}</span>`;
    header += ` <span class="muted">${kindLabel(e.kind)}</span>`;
    if (e.kind?.type === 'tool_call' || e.kind?.type === 'tool_result') {
      header += ` <code class="tool-name">${escapeHtml(e.kind.tool)}</code>`;
    }
    if (e.kind?.type === 'tool_result') {
      header += e.kind.ok
        ? ' <span class="status ok">ok</span>'
        : ' <span class="status failed">failed</span>';
    }
    if (e.kind?.type === 'tool_definitions') {
      const n = (e.kind.tools || []).length;
      header += ` <span class="muted">(${n} tool${n === 1 ? '' : 's'})</span>`;
    }
    let body = '';
    let bodyHtml = '';
    if (e.kind?.type === 'tool_call') body = formatJsonish(e.content);
    else if (e.kind?.type === 'tool_result') {
      if (!e.kind.ok && e.kind.error) body = e.kind.error;
      else if (e.kind.output) body = formatJsonish(e.kind.output);
    } else if (e.kind?.type === 'tool_definitions') {
      bodyHtml = (e.kind.tools || []).map(td => `
        <div class="tool-def">
          <div class="tool-name">${escapeHtml(td.name)}</div>
          <div class="tool-desc">${escapeHtml(td.description)}</div>
        </div>`).join('');
    } else if (typeof e.content === 'string') body = e.content;
    const key = selectedTaskId + '-' + i;
    const isCollapsed = collapsed[key] === undefined
      ? (cls === 'system' || cls === 'tool_definitions')
      : collapsed[key];
    div.className += isCollapsed ? ' collapsed' : '';
    const inner = bodyHtml
      ? `<div class="defs">${bodyHtml}</div>`
      : (body ? `<pre>${escapeHtml(body)}</pre>` : '');
    div.innerHTML = `
      <div class="h" data-entry-key="${key}">
        <span>${header}</span>
        <span>${e.timestamp.replace('T',' ').replace('Z','')}</span>
      </div>
      ${inner}`;
    div.querySelector('.h').onclick = () => {
      // Toggle the class directly instead of doing a full transcript
      // re-render. For 500+ entries that re-render was visibly janky.
      const nowCollapsed = !div.classList.contains('collapsed');
      div.classList.toggle('collapsed', nowCollapsed);
      collapsed[key] = nowCollapsed;
      // Periodic sweep: if `collapsed` has grown large, drop entries
      // whose task is no longer selected (they can't be referenced
      // again until the user re-selects that task, at which point
      // their default state is fine).
      const keyCount = Object.keys(collapsed).length;
      if (keyCount > CLIENT_COLLAPSED_CAP) {
        const prefix = selectedTaskId + '-';
        for (const k of Object.keys(collapsed)) {
          if (!k.startsWith(prefix)) delete collapsed[k];
        }
      }
    };
    el.appendChild(div);
  });
  el.scrollTop = wasAtBottom ? el.scrollHeight : prevTop;
}

// Currently-selected file source for the Files panel: "main" or a
// task UUID matching an active worktree.
let filesSource = 'main';

async function refreshFiles() {
  // Refresh worktree list and dropdown options first.
  const worktrees = await api('/api/worktrees').catch(() => []);
  const sel = document.getElementById('files-source');
  if (sel) {
    const desired = filesSource;
    const opts = ['<option value="main">main</option>'];
    for (const wt of worktrees) {
      const short = wt.task_id.slice(0, 8);
      opts.push(`<option value="${wt.task_id}">WIP · ${short} (${escapeHtml(wt.branch)})</option>`);
    }
    // Only re-set HTML when the option set changed, to preserve focus.
    const newHtml = opts.join('');
    if (sel.innerHTML !== newHtml) {
      sel.innerHTML = newHtml;
    }
    // Pick the desired source if it's still in the list; else fall back to main.
    if ([...sel.options].some(o => o.value === desired)) {
      sel.value = desired;
    } else {
      sel.value = 'main';
      filesSource = 'main';
    }
    sel.onchange = () => { filesSource = sel.value; refreshFiles(); };
  }
  const url = filesSource === 'main'
    ? '/api/files'
    : '/api/files?worktree=' + encodeURIComponent(filesSource);
  const files = await api(url).catch(() => []);
  const el = document.getElementById('file-tree');
  el.innerHTML = '';
  for (const f of files) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.textContent = f.path;
    a.onclick = () => openFile(f.path);
    li.appendChild(a);
    el.appendChild(li);
  }
  const label = filesSource === 'main'
    ? `(${files.length} on main)`
    : `(${files.length} WIP)`;
  document.getElementById('files-count').textContent = label;
}

async function refreshGit() {
  try {
    const log = await api('/api/gitlog');
    document.getElementById('git-log').innerHTML = log.slice(0, 50).map(c =>
      `<div>${c.sha.slice(0,7)} ${escapeHtml(c.message)}</div>`).join('');
    document.getElementById('git-count').textContent = `(${log.length})`;
  } catch {}
}

async function refreshIssues() {
  issues = await api('/api/issues') || [];
  const el = document.getElementById('issues-list');
  el.innerHTML = '';
  // Order: permanent (red) first, then retrying (amber), then resolved
  // (faded/struck-through). Within each group, newest first.
  const order = { permanent: 0, retrying: 1, resolved: 2 };
  issues.sort((a, b) =>
    (order[a.status] ?? 3) - (order[b.status] ?? 3)
    || b.timestamp.localeCompare(a.timestamp)
  );
  for (const it of issues) {
    const div = document.createElement('div');
    div.className = 'issue ' + (it.status || 'permanent');
    div.innerHTML = `
      <div>
        <span class="status ${it.status}">${it.status || 'permanent'}</span>
        <span class="phase">${it.stage}</span>
        <strong>${escapeHtml(it.node_name)}</strong>
        ${it.tool ? `<span class="muted">${escapeHtml(it.tool)}</span>` : ''}
      </div>
      <div class="imsg">${escapeHtml(it.message)}</div>`;
    div.onclick = () => {
      selectTask(it.task_id, () => {
        const target = document.querySelector(`[data-entry-key="${selectedTaskId}-${it.entry_index}"]`);
        if (target) target.scrollIntoView({behavior:'smooth', block:'center'});
      });
    };
    el.appendChild(div);
  }
  // Count visible (= unresolved) issues for the panel/status counters;
  // resolved issues are noise.
  const open = issues.filter(it => it.status !== 'resolved').length;
  const total = issues.length;
  document.getElementById('issues-count').textContent =
    total ? `(${open} open${total > open ? `, ${total - open} resolved` : ''})` : '';
  document.getElementById('status-issues').textContent =
    open ? `Issues: ${open}` : '';
}

async function openFile(path) {
  // Fetch from whatever source the Files panel is currently showing.
  let url = '/api/file?path=' + encodeURIComponent(path);
  if (filesSource && filesSource !== 'main') {
    url += '&worktree=' + encodeURIComponent(filesSource);
  }
  const t = await api(url);
  const tag = filesSource === 'main' ? '' : `  ·  WIP (${filesSource.slice(0,8)})`;
  document.getElementById('preview-h').textContent = path + tag;
  document.getElementById('preview').textContent = t;
}

// Maximum entries to retain on the SELECTED task's transcript in browser
// memory. The server caps its copy at 500 (configurable); after the
// initial fetch the client extends via SSE without bound, so we cap
// independently. On overflow we drop the oldest half (the head, since
// the most recent activity is what the operator usually wants to see).
const CLIENT_TRANSCRIPT_CAP = 1500;
// Cap state.history; the engine emits a HistoryAppended for every
// `note()` call, which gets noisy under high concurrency.
const CLIENT_HISTORY_CAP = 500;
// Cap `collapsed` keys. Toggling expanders never compacts this on its
// own; we sweep when it grows past this size.
const CLIENT_COLLAPSED_CAP = 5000;

function applyEvent(ev) {
  if (!state) return;
  switch (ev.type) {
    case 'scheduler_state_changed':
      state.scheduler = ev.state;
      markDirty('header');
      break;
    case 'task_created':
      state.tasks[ev.task.id] = ev.task;
      markDirty('header'); markDirty('tasks'); markDirty('graph');
      break;
    case 'task_status_changed':
      if (state.tasks[ev.id]) state.tasks[ev.id].status = ev.status;
      markDirty('tasks'); markDirty('graph');
      break;
    case 'task_updated':
      state.tasks[ev.task.id] = ev.task;
      markDirty('tasks'); markDirty('graph');
      break;
    case 'transcript_appended':
      // Only retain the transcript for the SELECTED task in browser
      // memory. Unselected tasks' transcripts can be fetched fresh
      // via /api/task_transcript if/when the user clicks them — keeping
      // them all in memory grows the heap without bound during long
      // runs (the source of the client-side memory pressure).
      if (ev.task_id === selectedTaskId) {
        const t = state.tasks?.[ev.task_id];
        if (t) {
          t.transcript = t.transcript || [];
          t.transcript.push(ev.entry);
          // Cap selected-task transcript: drop oldest half on overflow.
          if (t.transcript.length > CLIENT_TRANSCRIPT_CAP) {
            t.transcript.splice(0, t.transcript.length - CLIENT_TRANSCRIPT_CAP / 2);
          }
          markDirty('transcript');
        }
      }
      break;
    case 'task_cost':
      const tc = state.tasks?.[ev.task_id];
      if (tc) tc.cost = ev.cost;
      state.total_cost = ev.total;
      state.estimated_cost_usd = ev.estimated_usd;
      markDirty('header'); markDirty('tasks');
      break;
    case 'history_appended':
      state.history = state.history || [];
      state.history.push(ev.entry);
      if (state.history.length > CLIENT_HISTORY_CAP) {
        state.history.splice(0, state.history.length - CLIENT_HISTORY_CAP);
      }
      break;
    case 'file_changed':
      scheduleRefresh();
      break;
    case 'node_changed':
      // Graph nodes update on the periodic state poll; setting graph
      // dirty here would just cause animation restarts without new
      // info.
      break;
  }
  scheduleRender();
}

// Coalesce render calls via requestAnimationFrame. With high engine
// concurrency the SSE stream can deliver dozens of events per frame;
// calling render() for each was burning the main thread on redundant
// DOM rebuilds and was the dominant lockup symptom under load.
let _renderPending = false;
function scheduleRender() {
  if (_renderPending) return;
  _renderPending = true;
  requestAnimationFrame(() => {
    _renderPending = false;
    render();
  });
}

let _refTimer;
function scheduleRefresh() {
  clearTimeout(_refTimer);
  _refTimer = setTimeout(() => { refreshFiles(); refreshIssues(); }, 250);
}

let _es = null;
function connectSse() {
  // Close any existing connection BEFORE creating a new one. Previously
  // the order was inverted (schedule reconnect, then close), so under
  // flaky network conditions multiple EventSource instances could
  // accumulate, each leaking its handler closures.
  if (_es) { try { _es.close(); } catch (_) {} _es = null; }
  const es = new EventSource('/api/events');
  _es = es;
  es.onmessage = e => { try { applyEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  es.onerror = () => {
    try { es.close(); } catch (_) {}
    if (_es === es) _es = null;
    setTimeout(connectSse, 2000);
  };
}

initSections(); initSplitters(); load(); connectSse();
setInterval(refreshFiles, 4000);
setInterval(refreshGit, 5000);
setInterval(refreshIssues, 4000);
// Periodic state sync. /api/state ships a SLIM snapshot with each
// task's transcript empty (the wire payload is dominated by transcripts
// otherwise — see snapshot_slim in src/state.rs). Naively replacing
// `state` would clobber the in-memory transcript of the selected task
// on every poll; we preserve it here so the transcript view doesn't
// flash empty every 8 seconds.
setInterval(async () => {
  const prev = selectedTaskId && state
    ? state.tasks?.[selectedTaskId]?.transcript
    : null;
  state = await api('/api/state');
  if (prev && selectedTaskId && state.tasks?.[selectedTaskId]) {
    state.tasks[selectedTaskId].transcript = prev;
  }
  markAllDirty();
  render();
}, 8000);
</script>
</body>
</html>
"#;
