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
  #app { display: grid; grid-template-rows: 1fr auto; height: 100vh; }
  #panels { display: grid; grid-template-columns: 360px 1fr 360px; min-height: 0; }
  .panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; min-height: 0; }
  .panel:last-child { border-right: none; }
  .panel-h { padding: 8px 12px; background: var(--bg2); border-bottom: 1px solid var(--border); font-weight: 600; }
  .panel-body { overflow: auto; padding: 8px; flex: 1; min-height: 0; }
  #status { background: var(--bg2); border-top: 1px solid var(--border); padding: 6px 12px; display: flex; gap: 16px; align-items: center; font-size: 12px; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
  .badge.spec { background: #233a55; color: #b9d9ff; }
  .badge.iface { background: #2a3f2a; color: #aef0ae; }
  .badge.tests { background: #4a3f1f; color: #f0d59e; }
  .badge.impl { background: #503355; color: #d4a8e0; }
  .badge.debug { background: #4a2c2c; color: #f0a8a8; }
  .badge.opt { background: #2c4a4a; color: #a0e0e0; }
  .badge.pending, .badge.not_started { background: #2a2f3d; color: #aab2c5; }
  .badge.running, .badge.in_progress { background: #3a2f55; color: #d4b8ff; animation: pulse 1.2s infinite; }
  .badge.done { background: #1f3a2a; color: #98e3b3; }
  .badge.failed { background: #3a1f1f; color: #f0a0a0; }
  .badge.skipped { background: #2a2a2a; color: #888; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
  .node { padding: 6px 8px; border-radius: 4px; cursor: pointer; margin-bottom: 4px; }
  .node:hover { background: var(--bg2); }
  .node.selected { background: var(--bg3); }
  .node .name { font-weight: 600; }
  .node .stages { display: flex; gap: 3px; margin-top: 4px; flex-wrap: wrap; }
  .children { padding-left: 14px; border-left: 1px solid var(--border); margin-left: 4px; }
  .task { padding: 4px 6px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .task:hover { background: var(--bg2); }
  .task.selected { background: var(--bg3); }
  .transcript .entry { border-bottom: 1px dashed var(--border); padding: 6px 4px; }
  .transcript .entry.role-actor { border-left: 2px solid #5aa9e6; padding-left: 6px; }
  .transcript .entry.role-critic { border-left: 2px solid #d6a14a; padding-left: 6px; }
  .transcript .entry.role-reviser { border-left: 2px solid #5eb988; padding-left: 6px; }
  .transcript .entry.role-judge { border-left: 2px solid #a36ae6; padding-left: 6px; }
  .role-badge { display: inline-block; padding: 0 5px; border-radius: 3px; font-size: 10px; margin-right: 4px; }
  .role-badge.actor { background: #233a55; color: #b9d9ff; }
  .role-badge.critic { background: #4a3f1f; color: #f0d59e; }
  .role-badge.reviser { background: #1f3a2a; color: #98e3b3; }
  .role-badge.judge { background: #503355; color: #d4a8e0; }
  .transcript .entry .h { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); cursor: pointer; }
  .transcript .entry pre { white-space: pre-wrap; font-size: 11px; margin: 4px 0 0; word-break: break-all; }
  .transcript .entry.system pre { color: #c8b89e; }
  .transcript .entry.assistant_text pre { color: #d8e8f0; }
  .transcript .entry.tool_call pre { color: #a0c8a0; }
  .transcript .entry.tool_result pre { color: #c8a8e0; }
  .transcript .entry.tool_result.failed pre { color: var(--bad); }
  .collapsed pre { display: none; }
  pre.preview { background: var(--bg2); padding: 8px; border-radius: 4px; white-space: pre-wrap; font-size: 12px; max-height: 50vh; overflow: auto; word-break: break-all; }
  details.section { margin-bottom: 8px; border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; background: var(--bg); }
  details.section > summary { font-weight: 600; padding: 4px 0; cursor: pointer; }
  .file-tree li { list-style: none; padding: 1px 0; }
  .file-tree a { color: var(--fg); cursor: pointer; }
  .file-tree a:hover { color: var(--accent); }
  .issue { padding: 6px 8px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .issue:hover { background: var(--bg2); }
  .issue .imsg { color: var(--bad); font-size: 12px; margin-top: 2px; }
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
  <div class="panel">
    <div class="panel-h">Graph + Tasks <span class="muted" id="counts"></span></div>
    <div class="panel-body">
      <details class="section" open>
        <summary>Node graph</summary>
        <div id="graph-tree"></div>
      </details>
      <details class="section" open>
        <summary>Recent tasks</summary>
        <div id="task-list"></div>
      </details>
      <details class="section">
        <summary>Issues <span class="muted" id="issues-count"></span></summary>
        <div id="issues-list"></div>
      </details>
    </div>
  </div>
  <div class="panel">
    <div class="panel-h">Transcript <span class="muted" id="task-title"></span></div>
    <div class="panel-body transcript" id="transcript">
      <div class="muted">Select a task on the left to view its transcript.</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-h">Files</div>
    <div class="panel-body">
      <details class="section" open>
        <summary>Workdir <span class="muted" id="files-count"></span></summary>
        <ul class="file-tree" id="file-tree"></ul>
      </details>
      <details class="section" open>
        <summary>Git log <span class="muted" id="git-count"></span></summary>
        <div id="git-log" style="font-size: 11px;"></div>
      </details>
      <details class="section" open>
        <summary>Preview <span class="muted" id="preview-h"></span></summary>
        <pre class="preview" id="preview"></pre>
      </details>
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

function render() {
  if (!state) return;
  document.getElementById('sched').textContent = state.scheduler;
  const nodeCount = Object.keys(state.graph?.nodes || {}).length;
  const taskCount = Object.keys(state.tasks || {}).length;
  document.getElementById('node-count').textContent = nodeCount;
  document.getElementById('task-count').textContent = taskCount;
  document.getElementById('counts').textContent = `${nodeCount} nodes, ${taskCount} tasks`;
  document.getElementById('tokens').textContent =
    `${fmt(state.total_cost?.input_tokens||0)} in / ${fmt(state.total_cost?.output_tokens||0)} out`;
  document.getElementById('cost').textContent = `$${(state.estimated_cost_usd||0).toFixed(4)}`;
  renderGraph();
  renderTasks();
  if (selectedTaskId) renderTranscript();
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

function renderNodeRecursive(parent, nodeId) {
  const n = state.graph.nodes[nodeId];
  if (!n) return;
  const div = document.createElement('div');
  div.className = 'node';
  const stages = ['spec','iface','tests','impl','debug','opt'];
  const stageBadges = stages.map(s => {
    const st = n.stages[s] || 'not_started';
    return `<span class="badge ${st}" title="${s}: ${st}">${s}</span>`;
  }).join('');
  div.innerHTML = `
    <div class="name">
      <span class="badge spec">node</span> ${escapeHtml(n.name)}${n.crate_boundary ? ' <span class="muted">(crate)</span>' : ''}
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
  parent.appendChild(div);
  // Children
  const children = Object.values(state.graph.nodes).filter(c => c.parent === nodeId);
  if (children.length) {
    const ch = document.createElement('div');
    ch.className = 'children';
    parent.appendChild(ch);
    for (const c of children) renderNodeRecursive(ch, c.id);
  }
}

function renderTasks() {
  const el = document.getElementById('task-list');
  el.innerHTML = '';
  const tasks = Object.values(state.tasks || {}).slice().reverse(); // newest first
  for (const t of tasks.slice(0, 30)) {
    const div = document.createElement('div');
    div.className = 'task' + (t.id === selectedTaskId ? ' selected' : '');
    div.innerHTML = `
      <span class="badge ${t.stage}">${t.stage}</span>
      <span class="badge ${t.status}">${t.status}</span>
      <strong>${escapeHtml(t.node_name)}</strong>
      <span class="muted">${(t.cost?.input_tokens||0)+(t.cost?.output_tokens||0)} tok</span>`;
    div.onclick = () => { selectedTaskId = t.id; render(); };
    el.appendChild(div);
  }
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
    const role = e.role || 'actor';
    const failed = e.kind?.type === 'tool_result' && e.kind?.ok === false;
    const div = document.createElement('div');
    div.className = 'entry ' + cls + ' role-' + role + (failed ? ' failed' : '');
    let header = cls;
    let okBadge = '';
    if (e.kind?.type === 'tool_call') header += ` · ${e.kind.tool}`;
    if (e.kind?.type === 'tool_result') {
      header += ` · ${e.kind.tool}`;
      okBadge = e.kind.ok ? '✓' : '✗';
    }
    let body = '';
    if (e.kind?.type === 'tool_call') body = formatJsonish(e.content);
    else if (e.kind?.type === 'tool_result') {
      if (!e.kind.ok && e.kind.error) body = e.kind.error;
      else if (e.kind.output) body = formatJsonish(e.kind.output);
    } else if (typeof e.content === 'string') body = e.content;
    const key = selectedTaskId + '-' + i;
    const isCollapsed = collapsed[key] === undefined ? (cls === 'system') : collapsed[key];
    div.className += isCollapsed ? ' collapsed' : '';
    div.innerHTML = `
      <div class="h" data-entry-key="${key}">
        <span><span class="role-badge ${role}">${role}</span>${header} ${okBadge}</span>
        <span>${e.timestamp.replace('T',' ').replace('Z','')}</span>
      </div>
      ${body ? `<pre>${escapeHtml(body)}</pre>` : ''}`;
    div.querySelector('.h').onclick = () => {
      collapsed[key] = !isCollapsed;
      renderTranscript();
    };
    el.appendChild(div);
  });
  el.scrollTop = wasAtBottom ? el.scrollHeight : prevTop;
}

async function refreshFiles() {
  const files = await api('/api/files');
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
  document.getElementById('files-count').textContent = `(${files.length})`;
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
  for (const it of issues) {
    const div = document.createElement('div');
    div.className = 'issue';
    div.innerHTML = `
      <div><span class="badge ${it.stage}">${it.stage}</span> <strong>${escapeHtml(it.node_name)}</strong>
           ${it.tool ? `<span class="muted">${it.tool}</span>` : ''}</div>
      <div class="imsg">${escapeHtml(it.message)}</div>`;
    div.onclick = () => {
      selectedTaskId = it.task_id;
      render();
      setTimeout(() => {
        const target = document.querySelector(`[data-entry-key="${selectedTaskId}-${it.entry_index}"]`);
        if (target) target.scrollIntoView({behavior:'smooth', block:'center'});
      }, 0);
    };
    el.appendChild(div);
  }
  document.getElementById('issues-count').textContent = issues.length ? `(${issues.length})` : '';
  document.getElementById('status-issues').textContent = issues.length ? `Issues: ${issues.length}` : '';
}

async function openFile(path) {
  const t = await api('/api/file?path=' + encodeURIComponent(path));
  document.getElementById('preview-h').textContent = path;
  document.getElementById('preview').textContent = t;
}

function applyEvent(ev) {
  if (!state) return;
  switch (ev.type) {
    case 'scheduler_state_changed': state.scheduler = ev.state; break;
    case 'task_created':
      state.tasks[ev.task.id] = ev.task;
      break;
    case 'task_status_changed':
      if (state.tasks[ev.id]) state.tasks[ev.id].status = ev.status;
      break;
    case 'task_updated':
      state.tasks[ev.task.id] = ev.task;
      break;
    case 'transcript_appended':
      const t = state.tasks?.[ev.task_id];
      if (t) { t.transcript = t.transcript || []; ev.entry.role = ev.role; t.transcript.push(ev.entry); }
      break;
    case 'task_cost':
      const tc = state.tasks?.[ev.task_id];
      if (tc) tc.cost = ev.cost;
      state.total_cost = ev.total;
      state.estimated_cost_usd = ev.estimated_usd;
      break;
    case 'history_appended':
      state.history = state.history || []; state.history.push(ev.entry);
      break;
    case 'file_changed':
      scheduleRefresh();
      break;
    case 'node_changed':
      // Defer to periodic state poll for now.
      break;
  }
  render();
}

let _refTimer;
function scheduleRefresh() {
  clearTimeout(_refTimer);
  _refTimer = setTimeout(() => { refreshFiles(); refreshIssues(); }, 250);
}

function connectSse() {
  const es = new EventSource('/api/events');
  es.onmessage = e => { try { applyEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  es.onerror = () => { setTimeout(connectSse, 2000); es.close(); };
}

load(); connectSse();
setInterval(refreshFiles, 4000);
setInterval(refreshGit, 5000);
setInterval(refreshIssues, 4000);
setInterval(async () => { state = await api('/api/state'); render(); }, 8000);
</script>
</body>
</html>
"#;
