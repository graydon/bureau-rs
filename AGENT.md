# `bureau-rs` — Specification

A Rust rewrite of [bureau](https://github.com/graydon/bureau), a hierarchical multi-phase
agent orchestrator for generating Rust software. The Python original used Anthropic's
agent SDK and vendor coding agents. This rewrite calls LLM APIs directly via `rig`, owns
its own agent loop, and is specialized entirely for generating Rust programs.

---

## Goals and Non-Goals

**Goals:**
- Generate Rust programs via a rigid top-down multi-phase waterfall process
- Call LLM APIs directly (no shelling out to Claude Code, Codex, etc.)
- Parallel task execution within phases, with read/write interference analysis
- Git worktrees for parallel isolation, with custom merge drivers
- Rich web UI showing live task tree, streaming transcripts, cost, file tree
- Single-machine, single-process (tokio async runtime)
- Provider-agnostic via `rig` (Anthropic, OpenAI, etc. swappable)

**Non-goals:**
- General-purpose (Python, JS, etc.) code generation
- Distributed execution
- Replicating vendor agent features (computer use, semantic search, etc.)

---

## Phases

Phases execute sequentially. Each phase is a complete top-down decomposition and
execution pass over the project. Phase order:

1. **Spec** — produce structured specification documents
2. **Interface** — produce Rust type signatures, module declarations, trait definitions
   (no function bodies; stubs use `todo!()`)
3. **Test** — produce test modules against the interface (tests will fail until impl)
4. **Implementation** — fill in function bodies to make tests pass
5. **Debug** — targeted fixup of `cargo check` / `cargo test` failures
6. **Optimization** — targeted performance improvements (body-only edits)

Phase transition gate:
- **Spec→Interface:** spec artifact is well-formed (schema validated)
- **Interface→Test:** `cargo check` passes
- **Test→Implementation:** `cargo check` passes (tests compile, expected to fail)
- **Implementation→Debug:** `cargo test` passes OR max retry exceeded
- **Debug→Optimization:** `cargo test` passes
- **Done:** `cargo test` passes

On gate failure: identify tasks responsible for failing items (via write-set tracking),
retry those tasks. After configurable max retries, run a serial fixup pass (a single
agent with full visibility does sequential `cargo check`-driven repairs).

---

## Task Model

### Task Structure

```
Task {
    id: Uuid,
    phase: Phase,
    description: String,
    read_files: Vec<PathBuf>,   // declared at decomposition time
    write_files: Vec<PathBuf>,  // declared at decomposition time
    subtasks: Vec<Task>,        // populated during execution
    status: TaskStatus,
    agent_transcript: Vec<Message>,
    cost: TokenUsage,
    worktree: Option<PathBuf>,
}
```

### Task Execution Model

Each task node:
1. Receives injected context: its declared read-files (full content), phase instructions,
   and any locked interface artifacts from prior phases
2. Does work at its level (writes files via tools)
3. Emits a list of subtasks, each with declared read/write file sets
4. Subtasks are scheduled by the orchestrator subject to interference analysis

This interleaving of "do work, then emit subtasks" (rather than pure upfront
decomposition) is intentional — the node has real context when subdividing.

### Parallelism

Within a phase, tasks execute in parallel subject to:
- **Multiple readers allowed concurrently** for any file
- **Writers serialize** — a write to file F blocks all other readers and writers of F
- Implemented as a per-file `tokio::sync::RwLock` (or equivalent reader-writer set)

If a task discovers mid-execution that it needs to write a file not in its declared
write-set, it must either fail-and-redecompose or request a write-set extension
(re-checking interference). Prefer fail-and-redecompose for simplicity.

### Git Worktrees

Each parallel task gets its own git worktree. On completion, the orchestrator merges
back to the main tree. Conflicts are handled by custom merge drivers:

- **`mod`-declaration merge driver:** understands `mod foo;` in `lib.rs` / `main.rs`
  and deduplicates declarations rather than conflicting
- **`Cargo.toml` merge driver:** merges `[dependencies]` sections by union; panics if
  same key has different versions (dependencies must be declared in Interface phase and
  not modified thereafter)

Dependencies must be declared during Interface phase. Later phases cannot add new crate
dependencies.

---

## Tool Surface Per Phase

Tools are the only way agents write artifacts. Each phase has a restricted tool set.

### Spec Phase
- `write_spec_section(section: &str, content: &str)`
- `read_spec_section(section: &str) -> String`
- `list_spec_sections() -> Vec<String>`
- `emit_subtasks(tasks: Vec<SubtaskDecl>)`

### Interface Phase
- `read_spec_section(section: &str) -> String`
- `list_spec_sections() -> Vec<String>`
- `write_file(path: &Path, content: &str)` — must produce valid Rust (syn-checked)
- `read_file(path: &Path) -> String`
- `list_files(dir: &Path) -> Vec<PathBuf>`
- `emit_subtasks(tasks: Vec<SubtaskDecl>)`

No function bodies. The orchestrator enforces this by post-processing: any function with
a body that isn't `todo!()` or `unimplemented!()` gets its body replaced and a warning
logged.

### Test Phase
- `read_file(path: &Path) -> String` — interface files only (enforced by read-set)
- `read_spec_section(section: &str) -> String`
- `write_file(path: &Path, content: &str)` — test files only (enforced by write-set)
- `list_files(dir: &Path) -> Vec<PathBuf>`
- `emit_subtasks(tasks: Vec<SubtaskDecl>)`

Cannot write to interface files. Enforced by write-set intersection check.

### Implementation Phase
- `read_file(path: &Path) -> String` — interface + test files
- `write_file(path: &Path, content: &str)` — impl files only; cannot modify signatures
- `list_files(dir: &Path) -> Vec<PathBuf>`
- `emit_subtasks(tasks: Vec<SubtaskDecl>)`

The orchestrator post-processes impl-phase writes: if a public signature in a declared
interface file was changed, reject the write and return an error to the agent.

### Debug Phase
- `read_file(path: &Path) -> String`
- `read_compiler_error(error_id: &str) -> CompilerError` — structured cargo JSON output
- `list_compiler_errors() -> Vec<CompilerErrorSummary>`
- `replace_fn_body(path: &Path, fn_name: &str, new_body: &str)`
- `write_file(path: &Path, content: &str)` — narrow: only files in write-set
- No `emit_subtasks` — debug phase is sequential fixup, no further decomposition

### Optimization Phase
- Same as Debug but without compiler error tools; with optional perf annotation tools
- No interface modifications

---

## File Conventions

Small files are strongly preferred. The orchestrator enforces:
- **Max file size:** configurable, default ~150 lines. Tasks that produce files exceeding
  this are asked to split them.
- **Max spec section size:** configurable, default ~300 lines.

Rationale: small files allow LLMs to navigate by filename effectively, limit per-task
context, and reduce merge conflicts.

File layout enforced by orchestrator:
```
<workdir>/
  spec/               # spec phase outputs (markdown sections)
  src/                # Rust source (managed by interface/impl phases)
  tests/              # integration test files (managed by test phase)
  Cargo.toml          # managed; dependency changes only in Interface phase
```

---

## LLM Integration

Use `rig` (`rig-core` crate) for:
- Unified provider interface (Anthropic, OpenAI, etc.)
- Tool/function-call dispatch
- Streaming responses
- Token usage tracking

Each agent invocation is a fresh context (no conversation history carried across tasks).
Context injected per-task:
- Phase-specific system prompt
- Read-file contents (full, not chunked)
- Locked interface artifacts from prior phases (signatures only, not bodies)
- Task description and constraints

Model routing: configurable per phase. Heavier models for Spec/Interface (decisions are
locked), lighter for Debug/Optimization (mechanical fixes).

Structured output from agents (subtask declarations, etc.) via rig's tool-call mechanism
rather than parsing free text.

---

## Scheduler

```
Orchestrator {
    phase: Phase,
    task_graph: TaskGraph,          // DAG of tasks with read/write sets
    file_locks: HashMap<PathBuf, RwLock<()>>,
    worktrees: WorktreePool,
    llm_client: rig::Client,
    state: Arc<Mutex<OrchestratorState>>,  // shared with web UI
}
```

Main loop per phase:
1. Run root decomposition task (sequential, produces top-level task list)
2. Feed tasks into scheduler
3. Scheduler acquires read/write locks, spawns tokio tasks
4. Each task may emit subtasks; scheduler feeds them back in
5. On task completion: release locks, merge worktree, update state
6. When task queue empty: run phase gate (`cargo check` or `cargo test`)
7. On gate failure: identify responsible tasks by write-set, retry up to N times
8. After max retries: run serial fixup pass
9. Advance to next phase

---

## Web UI

Single-page app served by the orchestrator (axum). SSE for live updates.

### Endpoints

```
GET  /                        — SPA (single HTML file, self-contained)
GET  /api/state               — full OrchestratorState as JSON
GET  /api/events              — SSE stream of state deltas
GET  /api/files               — file tree as JSON
GET  /api/file?path=...       — file content
GET  /api/gitlog              — git log --oneline
GET  /api/gitdiff?hash=...    — diff for a commit
GET  /api/task_transcript?id= — full agent transcript for a task
POST /api/skip                — skip a running task: {"task_id": "..."}
POST /api/retry               — retry a failed task: {"task_id": "..."}
POST /api/pause               — pause scheduler
POST /api/resume              — resume scheduler
POST /api/checkpoint          — save full state to JSON on disk
POST /api/stop                — graceful shutdown
```

### UI Panels

**Task Tree** (left panel)
- Hierarchical tree of tasks, expand/collapse per node
- Per-task: phase badge, status (pending/running/done/failed/skipped), model name,
  token cost (input/output/cache), elapsed time
- Color-coded by status
- Click task → open transcript panel
- Skip / Retry buttons per task (only shown when applicable)
- Worktree indicator badge

**Transcript** (center panel)
- Live streaming of agent input and output for selected task
- Input section (system prompt + injected context) collapsible
- Output streams in as tokens arrive (SSE)
- Tool calls shown inline with arguments and results
- Cost ticker updates live

**File Tree** (right panel, top)
- Evolving tree of `<workdir>/src/` and `<workdir>/spec/`
- Click file → preview panel below
- Files highlighted when being written by a running task
- File size indicator (warn if approaching max)

**File Preview** (right panel, bottom)
- Syntax-highlighted Rust or Markdown
- Read-only

**Status Bar** (bottom)
- Current phase
- Tasks: N running / M pending / K done / J failed
- Total cost (USD estimate, updated live)
- Tokens: input / output / cache
- Pause / Resume / Stop buttons
- Burn rate (tokens/min)

**Git Log** (collapsible drawer)
- `git log --oneline` of workdir
- Click commit → inline diff view

### UI Implementation Notes

- Self-contained single HTML file served from axum (no npm, no build step)
- Use vanilla JS or a CDN-loaded minimal framework (e.g. Alpine.js from cdnjs)
- SSE events carry JSON state patches; UI applies them incrementally
- Syntax highlighting via highlight.js from cdnjs
- Mobile-tolerant but desktop-primary layout

---

## Configuration

Config directory (passed as CLI arg) contains:

```
problem.md          — problem statement / top-level spec seed
phases.toml         — phase configuration (which phases, model per phase, etc.)
prompts/            — per-phase system prompt overrides (optional)
  spec.md
  interface.md
  test.md
  impl.md
  debug.md
  opt.md
```

`phases.toml` example:
```toml
[phases.spec]
model = "claude-opus-4-6"       # rig provider model string
max_tokens = 8192
max_retries = 2

[phases.interface]
model = "claude-opus-4-6"
max_tokens = 8192
max_retries = 3

[phases.test]
model = "claude-sonnet-4-6"
max_tokens = 4096
max_retries = 2

[phases.impl]
model = "claude-sonnet-4-6"
max_tokens = 8192
max_retries = 3

[phases.debug]
model = "claude-haiku-4-5"
max_tokens = 4096
max_retries = 5

[phases.opt]
model = "claude-haiku-4-5"
max_tokens = 4096
max_retries = 2

[limits]
max_file_lines = 150
max_spec_section_lines = 300
max_parallel_tasks = 8
cost_cap_usd = 50.0             # pause and prompt user if exceeded
```

---

## CLI

```
bureau-rs <config-dir> <work-dir> [options]

Options:
  --port <N>           Web UI port (default: 8765)
  --resume <file>      Resume from checkpoint JSON
  --phase <phase>      Start from a specific phase (skip earlier phases)
  --dry-run            Decompose and show task graph, don't execute
  --no-ui              Don't start web server
```

---

## Checkpointing

Full orchestrator state serializable to JSON:
- Task graph (all tasks, statuses, transcripts, costs)
- File write history (which task wrote which file, at which commit)
- Current phase
- Git HEAD at each phase boundary

`POST /api/checkpoint` writes `checkpoint-<timestamp>.json` to workdir.
`--resume <file>` reloads state and continues from where it left off.

---

## Crate Structure

```
bureau-rs/
  Cargo.toml
  src/
    main.rs           — CLI entry, config loading, orchestrator init
    config.rs         — Config / phases.toml parsing
    phase.rs          — Phase enum, phase gate logic
    task.rs           — Task, TaskGraph, TaskStatus types
    scheduler.rs      — Async scheduler, lock management, worktree dispatch
    agent.rs          — LLM agent invocation via rig, tool dispatch
    tools.rs          — Tool implementations (file read/write, cargo check, etc.)
    merge.rs          — Custom git merge drivers (mod declarations, Cargo.toml)
    artifact.rs       — Artifact model (file index, syn-based validation)
    web/
      mod.rs          — axum router, SSE state broadcasting
      state.rs        — OrchestratorState, SSE event types
      ui.rs           — Embedded HTML/JS (include_str!)
    checkpoint.rs     — Serialization of full state
```

---

## Key Dependencies

```toml
[dependencies]
rig-core = "*"              # LLM provider abstraction + tool calling
tokio = { version = "*", features = ["full"] }
axum = "*"                  # web UI server
tower = "*"
serde = { version = "*", features = ["derive"] }
serde_json = "*"
syn = { version = "*", features = ["full", "extra-traits"] }
git2 = "*"                  # git worktree management
uuid = { version = "*", features = ["v4"] }
anyhow = "*"
tracing = "*"
tracing-subscriber = "*"
tokio-stream = "*"          # SSE
```

---

## Design Decisions and Rationale

**Why not ra_ap_* for semantic analysis?**
Phase-boundary `cargo check` is sufficient for a waterfall model. `ra_ap_*` APIs are
unstable and would add maintenance burden. `syn` handles structural validation and edits
within phases.

**Why file-level rather than item-level read/write sets?**
Files are the natural git merge unit. Small enforced file sizes make file-level
granularity approximately as fine as item-level in practice, while being simpler to
implement and reason about.

**Why serial debug/fixup phase at the end?**
Parallel agents fixing compiler errors can interfere with each other in hard-to-detect
ways (fixing the same error differently, introducing new errors in shared files). A
single serial agent with full visibility and a tight `cargo check` loop is more reliable
for convergence.

**Why no vendor coding agents?**
- Lock-in and pricing volatility
- Inability to compose into custom orchestration topology
- Opinionated tool sets not suited to this structured workflow
- Direct API access gives full control over context, tool surface, and agent loop

**Why small files?**
- LLM navigates by filename; good names + small files ≈ semantic index
- Limits per-task context size naturally
- Reduces merge conflicts in parallel worktrees
- Forces modular decomposition at the Rust level too

---

## Differences from bureau (Python original)

| Feature | bureau (Python) | bureau-rs |
|---|---|---|
| Language | Python | Rust |
| LLM access | Anthropic agent SDK | rig (direct API) |
| Agent loop | SDK-managed | Custom |
| Target language | General | Rust only |
| Tool surface | Vendor agent tools | Custom restricted per-phase |
| Artifact model | Files only | Files + syn validation |
| Merge | Basic git | Custom mod/Cargo merge drivers |
| Parallelism | File-level r/w sets | File-level r/w sets (same) |
| Phase gate | Ad hoc | Explicit cargo check/test |
| Checkpoint | JSON | JSON (same) |
| Web UI | SSE + vanilla JS | SSE + vanilla JS (same concept) |
