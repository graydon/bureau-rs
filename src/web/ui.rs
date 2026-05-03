//! Embedded single-page UI. Served from GET /.

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
    --running: #a36ae6; --skipped: #6b7280;
    --border: #283044;
    --mono: ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
  }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--fg); font-family: var(--mono); font-size: 13px; }
  #app { display: grid; grid-template-rows: 1fr auto; height: 100vh; }
  #panels { display: grid; grid-template-columns: 360px 1fr 380px; min-height: 0; }
  .panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; min-height: 0; }
  .panel:last-child { border-right: none; }
  .panel-h { padding: 8px 12px; background: var(--bg2); border-bottom: 1px solid var(--border); font-weight: 600; display: flex; justify-content: space-between; align-items: center; }
  .panel-body { overflow: auto; padding: 8px; flex: 1; min-height: 0; }
  .tabs { display: flex; gap: 0; background: var(--bg2); border-bottom: 1px solid var(--border); }
  .tab { padding: 6px 12px; cursor: pointer; border-right: 1px solid var(--border); font-size: 12px; }
  .tab.active { background: var(--bg); color: var(--accent); border-bottom: 2px solid var(--accent); }
  .tab .count { color: var(--muted); margin-left: 4px; font-size: 10px; }
  .tab-body { display: none; flex: 1; min-height: 0; flex-direction: column; }
  .tab-body.active { display: flex; }
  details.section { margin-bottom: 8px; border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; background: var(--bg); }
  details.section > summary { font-weight: 600; padding: 4px 0; }
  details.section > summary:hover { color: var(--accent); }
  details.section[open] > summary { border-bottom: 1px solid var(--border); margin-bottom: 6px; }
  #status { background: var(--bg2); border-top: 1px solid var(--border); padding: 6px 12px; display: flex; gap: 16px; align-items: center; font-size: 12px; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
  .badge.spec { background: #233a55; color: #b9d9ff; }
  .badge.interface { background: #2a3f2a; color: #aef0ae; }
  .badge.test { background: #4a3f1f; color: #f0d59e; }
  .badge.impl { background: #503355; color: #d4a8e0; }
  .badge.debug { background: #4a2c2c; color: #f0a8a8; }
  .badge.opt { background: #2c4a4a; color: #a0e0e0; }
  .badge.pending { background: #2a2f3d; color: #aab2c5; }
  .badge.running { background: #3a2f55; color: #d4b8ff; animation: pulse 1.2s infinite; }
  .badge.done { background: #1f3a2a; color: #98e3b3; }
  .badge.failed { background: #3a1f1f; color: #f0a0a0; }
  .badge.skipped { background: #2a2a2a; color: #888; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
  .task { padding: 4px 6px; border-radius: 4px; cursor: pointer; }
  .task:hover { background: var(--bg2); }
  .task.selected { background: var(--bg3); }
  .task .top { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .task .num { color: var(--muted); font-size: 11px; min-width: 38px; }
  .task .desc { font-size: 12px; margin-top: 3px; color: var(--fg); white-space: pre-wrap; word-break: break-word; }
  .task .meta { font-size: 10px; color: var(--muted); display: flex; gap: 8px; margin-top: 3px; flex-wrap: wrap; }
  .children { padding-left: 14px; border-left: 1px solid var(--border); margin-left: 4px; }
  .file-tree { font-size: 12px; padding: 0; margin: 0; }
  .file-tree li { list-style: none; padding: 1px 0; }
  .file-tree li.dir { color: var(--muted); }
  .file-tree a { color: var(--fg); cursor: pointer; text-decoration: none; }
  .file-tree a:hover { color: var(--accent); }
  .file-tree .size { color: var(--muted); font-size: 10px; margin-left: 6px; }
  pre.preview { background: var(--bg2); padding: 8px; border-radius: 4px; white-space: pre-wrap; font-size: 12px; overflow: auto; max-height: 60vh; word-break: break-all; }
  .transcript .entry { border-bottom: 1px dashed var(--border); padding: 6px 4px; transition: background-color 0.5s; }
  .transcript .entry.flash { background: rgba(217, 106, 106, 0.18); }
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
  .transcript .entry.user_prompt pre { color: #b8c8d8; }
  .transcript .entry.assistant_text pre { color: #d8e8f0; }
  .transcript .entry.tool_call pre { color: #a0c8a0; }
  .transcript .entry.tool_result pre { color: #c8a8e0; }
  .transcript .entry.tool_result.failed pre { color: var(--bad); }
  .transcript .entry.error pre { color: var(--bad); }
  .transcript .entry .h .ok { color: var(--good); }
  .transcript .entry .h .fail { color: var(--bad); }
  .collapsed pre { display: none; }
  .issue { padding: 6px 8px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .issue:hover { background: var(--bg2); }
  .issue .ih { display: flex; gap: 8px; align-items: center; font-size: 11px; }
  .issue .imsg { font-size: 12px; color: var(--bad); margin-top: 2px; }
  .issue .iargs { font-size: 11px; color: var(--muted); margin-top: 2px; word-break: break-all; max-height: 60px; overflow: hidden; text-overflow: ellipsis; }
  button { background: var(--bg3); color: var(--fg); border: 1px solid var(--border); padding: 3px 8px; border-radius: 3px; cursor: pointer; font-family: inherit; font-size: 11px; }
  button:hover { background: var(--bg2); }
  .row { display: flex; gap: 6px; align-items: center; }
  .grow { flex: 1; }
  .muted { color: var(--muted); }
  .mono { font-family: var(--mono); }
  .pill { background: var(--bg3); padding: 1px 6px; border-radius: 8px; font-size: 10px; }
  details summary { cursor: pointer; padding: 4px 0; color: var(--muted); }
  .errcount { color: var(--bad); }
</style>
</head>
<body>
<div id="app">
  <div id="panels">
    <div class="panel">
      <div class="tabs">
        <div class="tab active" data-tab="tasks">Tasks <span class="count" id="tab-tasks-count"></span></div>
        <div class="tab" data-tab="issues">Issues <span class="count errcount" id="tab-issues-count"></span></div>
      </div>
      <div class="tab-body active" id="tab-tasks">
        <div class="panel-body" id="task-tree"></div>
      </div>
      <div class="tab-body" id="tab-issues">
        <div class="panel-body" id="issues-list"></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-h">
        <span>Transcript</span>
        <span class="muted" id="task-title"></span>
      </div>
      <div class="panel-body transcript" id="transcript">
        <div class="muted">Select a task on the left to view its transcript.</div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-h"><span>Files & Git</span><span class="muted" id="rightcount"></span></div>
      <div class="panel-body" id="rightcol">
        <details class="section" open>
          <summary>Workdir <span class="muted" id="files-count"></span></summary>
          <ul class="file-tree" id="file-tree"></ul>
        </details>
        <details class="section" open id="task-files-section">
          <summary>Task worktree <span class="muted" id="task-files-h"></span></summary>
          <ul class="file-tree" id="task-file-tree"></ul>
        </details>
        <details class="section" open>
          <summary>Git log <span class="muted" id="git-count"></span></summary>
          <div id="git-log" style="font-size: 11px;"></div>
        </details>
        <details class="section" open>
          <summary>Preview <span class="muted" id="file-preview-h"></span></summary>
          <pre class="preview" id="file-preview"></pre>
        </details>
      </div>
    </div>
  </div>
  <div id="status">
    <span><b>Phase:</b> <span id="phase">—</span></span>
    <span><b>Sched:</b> <span id="sched">—</span></span>
    <span class="grow"></span>
    <span class="muted">Tasks: <span id="counts">0/0/0/0/0</span></span>
    <span class="muted errcount" id="status-issues"></span>
    <span class="muted">Tokens: <span id="tokens">0/0</span></span>
    <span class="muted">USD: <span id="cost">$0.00</span></span>
    <button onclick="api('/api/pause','POST')">Pause</button>
    <button onclick="api('/api/resume','POST')">Resume</button>
    <button onclick="api('/api/checkpoint','POST').then(r => alert('saved: ' + (r.path||'')))">Checkpoint</button>
    <button onclick="api('/api/stop','POST')">Stop</button>
  </div>
</div>
<script>
let state = null;
let issues = [];
let selectedTaskId = null;
let collapsed = {}; // entry-key -> bool

async function api(path, method='GET', body=null) {
  const r = await fetch(path, {
    method,
    headers: body ? {'Content-Type':'application/json'} : {},
    body: body ? JSON.stringify(body) : null,
  });
  if (r.headers.get('content-type')?.includes('application/json')) return r.json();
  return r.text();
}

function setupTabs() {
  // Left panel only (Tasks / Issues). Right panel uses always-visible <details>
  // sections instead.
  document.querySelectorAll('.tab').forEach(tab => {
    tab.onclick = () => {
      const group = tab.parentElement;
      const bodies = group.parentElement;
      group.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      bodies.querySelectorAll(':scope > .tab-body').forEach(b => b.classList.remove('active'));
      const target = tab.dataset.tab;
      const body = document.getElementById('tab-' + target);
      if (body) body.classList.add('active');
      if (target === 'issues') refreshIssues();
    };
  });
}

async function load() {
  state = await api('/api/state');
  computeNumbering();
  render();
  refreshFiles();
  refreshGitLog();
  refreshIssues();
}

function fmt(n) { return Number(n).toLocaleString(); }

// Compute dotted-path numbers like "1.2.1" for each task in the graph.
let taskNum = {}; // task_id -> "1.2.1"
function computeNumbering() {
  taskNum = {};
  if (!state || !state.graph) return;
  const visit = (id, prefix) => {
    taskNum[id] = prefix;
    const t = state.graph.tasks[id];
    if (!t) return;
    (t.subtasks || []).forEach((cid, i) => visit(cid, prefix + '.' + (i+1)));
  };
  (state.graph.roots || []).forEach((rid, i) => visit(rid, String(i+1)));
}

function render() {
  if (!state) return;
  document.getElementById('phase').textContent = state.phase;
  document.getElementById('sched').textContent = state.scheduler;
  const tasks = state.graph?.tasks ? Object.values(state.graph.tasks) : [];
  const c = tasks.reduce((acc, t) => { acc[t.status] = (acc[t.status] || 0) + 1; return acc; }, {});
  document.getElementById('counts').textContent =
    `${c.running||0}r / ${c.pending||0}p / ${c.done||0}d / ${c.failed||0}f / ${c.skipped||0}s`;
  document.getElementById('tab-tasks-count').textContent = tasks.length || '';
  document.getElementById('tab-issues-count').textContent = issues.length ? issues.length : '';
  document.getElementById('status-issues').textContent = issues.length ? `Issues: ${issues.length}` : '';
  const tot = state.total_cost || {};
  document.getElementById('tokens').textContent = `${fmt(tot.input_tokens||0)} in / ${fmt(tot.output_tokens||0)} out`;
  document.getElementById('cost').textContent = `$${(state.estimated_cost_usd||0).toFixed(4)}`;
  renderTasks();
  if (selectedTaskId) renderTranscript();
}

function renderTasks() {
  const root = document.getElementById('task-tree');
  root.innerHTML = '';
  if (!state.graph) return;
  const roots = state.graph.roots || [];
  for (const id of roots) renderTaskNode(root, id);
}

function renderTaskNode(parent, id) {
  const t = state.graph.tasks[id];
  if (!t) return;
  const div = document.createElement('div');
  div.className = 'task' + (id === selectedTaskId ? ' selected' : '');
  const phaseBadge = `<span class="badge ${t.phase}">${t.phase}</span>`;
  const statusBadge = `<span class="badge ${t.status}">${t.status}</span>`;
  const tokens = (t.cost?.input_tokens || 0) + (t.cost?.output_tokens || 0);
  const elapsed = t.started_at
    ? `${Math.round((((t.finished_at ? Date.parse(t.finished_at) : Date.now()) - Date.parse(t.started_at))/1000))}s`
    : '';
  const errCount = countTaskErrors(t);
  div.innerHTML = `
    <div class="top">
      <span class="num">${taskNum[id] || ''}</span>
      ${phaseBadge}${statusBadge}
      <span class="muted">${(t.model||'').replace('claude-','').replace('anthropic/','').replace('qwen/','')}</span>
      ${errCount ? `<span class="pill errcount">${errCount} err</span>` : ''}
      ${t.worktree ? `<span title="${escapeHtml(t.worktree)}">⎇</span>` : ''}
    </div>
    <div class="desc">${escapeHtml(t.description)}</div>
    <div class="meta">
      <span>${fmt(tokens)} tok</span>
      <span>${elapsed}</span>
      <span>d=${t.depth || 0}</span>
    </div>`;
  div.onclick = (e) => { e.stopPropagation(); selectedTaskId = id; render(); refreshTaskFiles(); };
  parent.appendChild(div);
  if (t.subtasks && t.subtasks.length) {
    const ch = document.createElement('div');
    ch.className = 'children';
    parent.appendChild(ch);
    for (const cid of t.subtasks) renderTaskNode(ch, cid);
  }
}

function countTaskErrors(t) {
  if (!t.transcript) return 0;
  return t.transcript.filter(e =>
    e.kind?.type === 'error' ||
    (e.kind?.type === 'tool_result' && e.kind?.ok === false)
  ).length;
}

function renderTranscript() {
  const el = document.getElementById('transcript');
  const t = state.graph?.tasks?.[selectedTaskId];
  document.getElementById('task-title').textContent = t
    ? `${taskNum[selectedTaskId] || ''} · ${t.description.slice(0, 80)}`
    : '';
  // Preserve scroll position. Only auto-scroll to bottom if the user was
  // already pinned there before this re-render.
  const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;
  const prevScroll = el.scrollTop;
  if (!t) { el.innerHTML = '<div class="muted">Task not found.</div>'; return; }
  el.innerHTML = '';

  // Synthetic header: phase tools catalog (collapsible, collapsed by default)
  const toolsKey = selectedTaskId + '-tools';
  if (collapsed[toolsKey] === undefined) collapsed[toolsKey] = true;
  const toolsCollapsed = collapsed[toolsKey];
  const toolsDiv = document.createElement('div');
  toolsDiv.className = 'entry tools' + (toolsCollapsed ? ' collapsed' : '');
  const phaseTools = phaseToolsCache[t.phase] || [];
  const toolsBody = phaseTools.map(tt =>
    `<b>${escapeHtml(tt.name)}</b><br><span class="muted">${escapeHtml(tt.description)}</span>` +
    `<pre style="margin: 4px 0 8px;">${escapeHtml(JSON.stringify(tt.parameters, null, 2))}</pre>`
  ).join('');
  toolsDiv.innerHTML = `
    <div class="h" data-key="${toolsKey}">
      <span>tools available · ${phaseTools.length}</span>
      <span class="muted">(static catalog sent to LLM each turn)</span>
    </div>
    <pre>${toolsBody || 'loading…'}</pre>`;
  toolsDiv.querySelector('.h').onclick = () => {
    collapsed[toolsKey] = !toolsCollapsed;
    renderTranscript();
  };
  el.appendChild(toolsDiv);
  if (!phaseTools.length) ensurePhaseTools(t.phase);

  (t.transcript || []).forEach((e, i) => {
    const cls = (e.kind && e.kind.type) || 'note';
    const failed = e.kind?.type === 'tool_result' && e.kind?.ok === false;
    const role = e.role || 'actor';
    const div = document.createElement('div');
    div.className = 'entry ' + cls + ' role-' + role + (failed ? ' failed' : '');
    const roleBadge = `<span class="role-badge ${role}">${role}</span>`;
    let header = cls;
    let okBadge = '';
    if (e.kind?.type === 'tool_call') header += ` · ${e.kind.tool}`;
    if (e.kind?.type === 'tool_result') {
      header += ` · ${e.kind.tool}`;
      okBadge = e.kind.ok ? '<span class="ok">✓</span>' : '<span class="fail">✗</span>';
    }
    let body = '';
    if (e.kind?.type === 'tool_call') {
      body = formatJsonish(e.kind.args);
    } else if (e.kind?.type === 'tool_result') {
      if (!e.kind.ok && e.kind.error) body = e.kind.error;
      else if (e.kind.output) body = formatJsonish(e.kind.output);
    } else if (typeof e.content === 'string') {
      body = e.content;
    }
    const key = selectedTaskId + '-' + i;
    const isCollapsed = collapsed[key] === undefined ? defaultCollapsed(cls) : collapsed[key];
    div.className += isCollapsed ? ' collapsed' : '';
    div.innerHTML = `
      <div class="h" data-key="${key}" data-entry-key="${selectedTaskId}-${i}">
        <span>${roleBadge}${header} ${okBadge}</span>
        <span>${e.timestamp.replace('T',' ').replace('Z','')}</span>
      </div>
      ${body ? `<pre>${escapeHtml(body)}</pre>` : ''}`;
    div.querySelector('.h').onclick = () => {
      collapsed[key] = !isCollapsed;
      renderTranscript();
    };
    el.appendChild(div);
  });
  if (wasAtBottom) {
    el.scrollTop = el.scrollHeight;
  } else {
    el.scrollTop = prevScroll;
  }
}

function formatJsonish(s) {
  if (!s) return '';
  try {
    const v = JSON.parse(s);
    return JSON.stringify(v, null, 2);
  } catch {
    return s;
  }
}

const phaseToolsCache = {};
async function ensurePhaseTools(phase) {
  if (phaseToolsCache[phase]) return;
  try {
    const resp = await api('/api/phase_info?phase=' + encodeURIComponent(phase));
    if (resp && resp.tools) {
      phaseToolsCache[phase] = resp.tools;
      // Re-render only if the same task is still selected
      if (selectedTaskId && state.graph?.tasks?.[selectedTaskId]?.phase === phase) {
        renderTranscript();
      }
    }
  } catch (e) {}
}

function defaultCollapsed(kind) {
  // Collapse the big system prompts by default; expand the rest.
  return kind === 'system';
}

async function refreshFiles() {
  const files = await api('/api/files');
  const el = document.getElementById('file-tree');
  el.innerHTML = '';
  let count = 0;
  for (const f of files) {
    if (f.is_dir) continue;
    count += 1;
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.textContent = f.path;
    a.onclick = () => openFile(f.path);
    li.appendChild(a);
    const sp = document.createElement('span');
    sp.className = 'size';
    sp.textContent = fmt(f.size) + ' b';
    li.appendChild(sp);
    el.appendChild(li);
  }
  const cnt = document.getElementById('files-count');
  if (cnt) cnt.textContent = `(${count})`;
}

async function refreshTaskFiles() {
  const el = document.getElementById('task-file-tree');
  const h = document.getElementById('task-files-h');
  el.innerHTML = '';
  if (!selectedTaskId) { if (h) h.textContent = '(no task selected)'; return; }
  const r = await api('/api/task_files?id=' + selectedTaskId);
  if (!r || !r.files) { if (h) h.textContent = '(no worktree)'; return; }
  if (h) h.textContent = `(${r.files.length})`;
  for (const f of r.files) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.textContent = f.path;
    a.onclick = () => openTaskFile(selectedTaskId, f.path);
    li.appendChild(a);
    const sp = document.createElement('span');
    sp.className = 'size';
    sp.textContent = fmt(f.size) + ' b';
    li.appendChild(sp);
    el.appendChild(li);
  }
}

async function refreshIssues() {
  issues = await api('/api/issues') || [];
  const el = document.getElementById('issues-list');
  el.innerHTML = '';
  for (const it of issues) {
    const div = document.createElement('div');
    div.className = 'issue';
    const num = taskNum[it.task_id] || '?';
    div.innerHTML = `
      <div class="ih">
        <span class="muted">${num}</span>
        <span class="badge ${it.phase}">${it.phase}</span>
        <span class="pill">${it.kind}</span>
        ${it.tool ? `<span class="muted">${it.tool}</span>` : ''}
        <span class="grow"></span>
        <span class="muted">${it.timestamp.replace('T',' ').replace('Z','')}</span>
      </div>
      <div class="imsg">${escapeHtml(it.message || '')}</div>
      ${it.args ? `<div class="iargs">${escapeHtml(it.args)}</div>` : ''}`;
    div.onclick = () => {
      selectedTaskId = it.task_id;
      pendingScrollEntryIdx = it.entry_index;
      // Switch left panel back to tasks tab so the user sees the task
      document.querySelector('.tab[data-tab="tasks"]').click();
      render();
      refreshTaskFiles();
      // Scroll happens after render() flushes DOM.
      setTimeout(() => scrollToEntry(it.entry_index), 0);
    };
    el.appendChild(div);
  }
  // update counts
  document.getElementById('tab-issues-count').textContent = issues.length || '';
  document.getElementById('status-issues').textContent = issues.length ? `Issues: ${issues.length}` : '';
}

let pendingScrollEntryIdx = null;
function scrollToEntry(idx) {
  if (idx == null) return;
  const target = document.querySelector(`[data-entry-key="${selectedTaskId}-${idx}"]`);
  if (!target) return;
  // Expand the entry if collapsed so the user sees the content.
  const div = target.closest('.entry');
  if (div) {
    div.classList.remove('collapsed');
    collapsed[`${selectedTaskId}-${idx}`] = false;
  }
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  // Briefly highlight.
  if (div) {
    div.classList.add('flash');
    setTimeout(() => div.classList.remove('flash'), 1500);
  }
  pendingScrollEntryIdx = null;
}

async function refreshGitLog() {
  try {
    const log = await api('/api/gitlog');
    const el = document.getElementById('git-log');
    el.innerHTML = log.slice(0, 50).map(c =>
      `<div><a onclick="showDiff('${c.sha}')" style="cursor:pointer; color: var(--accent)">${c.sha.slice(0,7)}</a> ${escapeHtml(c.message)}</div>`
    ).join('');
    const cnt = document.getElementById('git-count');
    if (cnt) cnt.textContent = `(${log.length})`;
  } catch (e) {}
}

async function showDiff(sha) {
  const text = await api('/api/gitdiff?hash=' + encodeURIComponent(sha));
  document.getElementById('file-preview-h').textContent = `diff ${sha.slice(0,7)}`;
  document.getElementById('file-preview').textContent = text;
}

async function openFile(path) {
  const text = await api('/api/file?path=' + encodeURIComponent(path));
  document.getElementById('file-preview-h').textContent = path + ' (workdir)';
  document.getElementById('file-preview').textContent = text;
}

async function openTaskFile(tid, path) {
  const text = await api('/api/task_file?id=' + tid + '&path=' + encodeURIComponent(path));
  document.getElementById('file-preview-h').textContent = path + ' (worktree of ' + (taskNum[tid]||'?') + ')';
  document.getElementById('file-preview').textContent = text;
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function applyEvent(ev) {
  if (!state) return;
  switch (ev.type) {
    case 'phase_changed': state.phase = ev.phase; break;
    case 'scheduler_state_changed': state.scheduler = ev.state; break;
    case 'task_created':
      state.graph.tasks[ev.task.id] = ev.task;
      if (!ev.task.parent) state.graph.roots.push(ev.task.id);
      else {
        const p = state.graph.tasks[ev.task.parent];
        if (p && !p.subtasks.includes(ev.task.id)) p.subtasks.push(ev.task.id);
      }
      computeNumbering();
      break;
    case 'task_status_changed':
      if (state.graph.tasks[ev.id]) state.graph.tasks[ev.id].status = ev.status;
      break;
    case 'task_updated':
      state.graph.tasks[ev.task.id] = ev.task;
      break;
    case 'transcript_appended':
      const tt = state.graph.tasks[ev.task_id];
      if (tt) {
        tt.transcript = tt.transcript || [];
        tt.transcript.push(ev.entry);
        // If a tool failed, refresh issues so the count goes up
        if (ev.entry.kind?.type === 'tool_result' && ev.entry.kind?.ok === false) {
          scheduleIssuesRefresh();
        }
        if (ev.entry.kind?.type === 'error') scheduleIssuesRefresh();
      }
      break;
    case 'task_cost':
      const tc = state.graph.tasks[ev.task_id];
      if (tc) tc.cost = ev.cost;
      state.total_cost = ev.total;
      state.estimated_cost_usd = ev.estimated_usd;
      break;
    case 'history_appended':
      state.history = state.history || []; state.history.push(ev.entry);
      break;
    case 'file_changed':
      scheduleFileRefresh();
      if (selectedTaskId) refreshTaskFiles();
      break;
  }
  render();
}

let _fileRefreshTimer, _issuesRefreshTimer;
function scheduleFileRefresh() {
  clearTimeout(_fileRefreshTimer);
  _fileRefreshTimer = setTimeout(refreshFiles, 250);
}
function scheduleIssuesRefresh() {
  clearTimeout(_issuesRefreshTimer);
  _issuesRefreshTimer = setTimeout(refreshIssues, 400);
}

function connectSse() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    try { applyEvent(JSON.parse(e.data)); } catch (err) { console.error(err); }
  };
  es.onerror = () => { setTimeout(connectSse, 2000); es.close(); };
}

setupTabs();
load();
connectSse();
// Periodic fallback refresh in case SSE events are missed/lagged.
setInterval(refreshGitLog, 4000);
setInterval(refreshFiles, 4000);
setInterval(refreshIssues, 4000);
setInterval(async () => { state = await api('/api/state'); computeNumbering(); render(); }, 8000);
</script>
</body>
</html>
"#;
