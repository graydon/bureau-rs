//! The tool surface exposed to the LLM in the new node-stage engine.
//!
//! Three families:
//!
//! 1. **Slot-fillers** — `submit_spec`, `submit_public`, `submit_private`,
//!    `submit_tests`. Each writes a single, well-defined slot of the
//!    currently-active node. The framework re-renders the on-disk source
//!    tree after every successful submit. The model never names file paths.
//!
//! 2. **Graph mutators** — `decompose`. Adds children and/or self-deps to
//!    the current node. Cycle-checked, name-validated, dep-validated.
//!
//! 3. **Diagnostics** — `cargo_check`, `cargo_test`, `cargo_test_no_run`,
//!    `cargo_clippy`. Read-only; let the model iterate within a turn.
//!
//! 4. **Verdict** — `submit_verdict`. Judge-only.
//!
//! 5. **Quickfix file-edit tools** — `read_file`, `write_file`,
//!    `write_file_range`, `apply_patch`. `read_file` is registered on
//!    every role (it's read-only). The write/patch trio is only
//!    registered for the QuickFixer role and is scoped to slots owned
//!    by the current stage.

use crate::graph::{Node, NodeGraph, NodeId, Stage};
#[cfg(test)]
use crate::graph::StageState;
use crate::node_validate::{self, ValidateError};
use crate::render::{self, Layout};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ToolFailure {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("validation: {0}")]
    Validate(#[from] ValidateError),
    #[error(
        "wrong stage: tool `{tool}` is not available in the current stage `{stage}`. \
         The model is calling tools outside its role's allowed set."
    )]
    WrongStage { tool: String, stage: String },
    #[error(
        "loop detected: you have called the `{tool}` tool {count} times in a row with \
         the same arguments. Stop repeating; either move on or end your turn."
    )]
    Loop { tool: String, count: usize },
    #[error("graph: {0}")]
    Graph(#[from] crate::graph::GraphError),
    #[error("subtask validation: {0}")]
    Subtask(String),
    #[error("file too large: {0} lines (limit {1})")]
    FileTooLarge(usize, usize),
    #[error("{0}")]
    Other(String),
}

// --------------------------------------------------------------------------
// Per-task context.
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeVerdict {
    pub satisfactory: bool,
    pub reason: String,
}

/// Shared scratch passed through every tool call inside a single agent
/// invocation. The orchestrator constructs this when it spawns a stage's
/// actor / critic / reviser / judge call, and reads its mutated state out
/// after the call returns.
pub struct TaskCtx {
    pub task_id: Uuid,
    /// The node, stage, and cycle role this task is advancing. The role
    /// is stamped onto every transcript entry the ctx records.
    pub node_id: NodeId,
    pub stage: Stage,
    pub role: Role,
    /// Workdir on disk. The graph lives here as `.bureau/graph.json` +
    /// `.bureau/nodes/*.json` and is the source of truth — tools that
    /// need the graph load it fresh from disk on each call via
    /// `graph::load(&ctx.workdir)`, mutate, and persist via
    /// `render::render_graph` (which `graph::save`s as a side effect).
    /// Rig dispatches tool calls with `concurrency = 1` so within a
    /// task they're serialized at the agent layer; no in-process lock
    /// needed.
    pub workdir: PathBuf,
    pub layout: Layout,
    pub max_file_lines: usize,
    pub max_spec_section_lines: usize,
    /// Hard cap on the total number of nodes the graph may hold. The
    /// decompose tool refuses to exceed it.
    pub max_nodes: usize,
    /// Hard cap on the depth of the node tree (root is depth 0).
    pub max_node_depth: usize,
    /// All mutable per-task state, behind one mutex. Rig serializes tool
    /// calls within a task (concurrency=1), so the lock is uncontended —
    /// it exists only to give `&mut` semantics through a `&self` rig
    /// `Tool::call`. Access via the `with_state` / `take_*` / `drain_*`
    /// methods; do not lock directly from outside.
    state: Mutex<TaskCtxState>,
    /// Held for the duration of any cargo invocation so parallel tasks
    /// don't trample each other's `target/` dir / lock files.
    pub cargo_lock: Arc<tokio::sync::Mutex<()>>,
    /// Optional handle to the engine's live `StateHandle`. When set,
    /// tools push transcript entries into the canonical
    /// `s.tasks[task_id].transcript` **and** broadcast a
    /// [`UiEvent::TranscriptAppended`] in the same operation — see
    /// [`Self::live_append_transcript`]. The streaming LLM driver also
    /// uses it to push [`UiEvent::AssistantChunk`] deltas. `None` for
    /// mock-driver tests that don't have a `StateHandle` available.
    live: Option<crate::state::StateHandle>,
}

/// All mutable accumulator state for a `TaskCtx`. Lives behind a single
/// `Mutex` on the ctx — collapses what used to be five separate per-field
/// `Mutex`es into one cohesive value.
#[derive(Default)]
struct TaskCtxState {
    /// Loop detection — same args three times in a row triggers an error.
    recent_calls: VecDeque<(String, u64)>,
    /// Filled by `submit_verdict` (judge stage only).
    verdict: Option<JudgeVerdict>,
    /// Filled by `submit_critique` (critic stage only). The reviser and
    /// fast-path detection both read this instead of fuzzy-parsing prose.
    critique: Option<Critique>,
    /// Transcript callback for recording tool calls / results.
    transcript: Vec<TranscriptEntry>,
    /// File changes queued for the orchestrator to broadcast over SSE.
    fs_events: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub timestamp: DateTime<Utc>,
    pub kind: TranscriptKind,
    pub content: String,
    /// The cycle role active when this entry was produced
    /// (writer/critic/reviser/judge). Stored on the entry itself so it
    /// survives serialization to disk / over `/api/state` — the UI can
    /// no longer rely on a separate SSE-only channel.
    #[serde(default)]
    pub role: Option<Role>,
}

impl TranscriptEntry {
    /// True if this entry was produced by the LLM (assistant text or
    /// tool call); false for framework-produced entries (system/user
    /// prompts, tool results, notes, errors). Used by the UI to pick
    /// the entry's CSS class.
    pub fn is_model(&self) -> bool {
        matches!(
            self.kind,
            TranscriptKind::AssistantText | TranscriptKind::ToolCall { .. }
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptKind {
    System,
    UserPrompt,
    AssistantText,
    /// Snapshot of the tool definitions sent to the model alongside the
    /// system prompt. Recorded once per (stage, role) invocation so the UI
    /// can show exactly what the model was told the tools do.
    ToolDefinitions {
        tools: Vec<ToolDefSnapshot>,
    },
    ToolCall {
        tool: String,
    },
    ToolResult {
        tool: String,
        ok: bool,
        #[serde(default)]
        error: Option<String>,
        #[serde(default)]
        output: Option<String>,
    },
    Note,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefSnapshot {
    pub name: String,
    pub description: String,
}

const LOOP_BREAK_THRESHOLD: usize = 3;
const LOOP_WINDOW: usize = 8;

impl TaskCtx {
    pub fn prompt_limits(&self) -> PromptLimits {
        PromptLimits {
            max_file_lines: self.max_file_lines,
            max_spec_section_lines: self.max_spec_section_lines,
        }
    }

    pub fn new(
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        role: Role,
        workdir: PathBuf,
        layout: Layout,
        max_file_lines: usize,
        max_spec_section_lines: usize,
        max_nodes: usize,
        max_node_depth: usize,
        cargo_lock: Arc<tokio::sync::Mutex<()>>,
    ) -> Self {
        Self {
            task_id,
            node_id,
            stage,
            role,
            workdir,
            layout,
            max_file_lines,
            max_spec_section_lines,
            max_nodes,
            max_node_depth,
            state: Mutex::new(TaskCtxState::default()),
            cargo_lock,
            live: None,
        }
    }

    /// Attach the engine's `StateHandle`. With this set:
    /// - tools live-write transcript entries to the canonical
    ///   `s.tasks[task_id].transcript` AND broadcast over SSE
    ///   (see [`Self::live_append_transcript`])
    /// - the streaming LLM driver pushes [`UiEvent::AssistantChunk`]
    ///   text deltas
    ///
    /// Without it, tools still record entries into `ctx`'s internal
    /// transcript buffer (for the engine to drain at end-of-call), but
    /// nothing reaches the UI mid-stage — used by unit tests and the
    /// mock driver.
    pub fn with_state(mut self, handle: crate::state::StateHandle) -> Self {
        self.live = Some(handle);
        self
    }

    /// Push a UI event onto the live broadcast channel. Dropped (no
    /// receivers) and disconnected (channel closed) errors are both
    /// silently ignored — emit is best-effort. Most code should prefer
    /// [`Self::live_append_transcript`] for transcript entries so that
    /// `/api/task_transcript` reflects the same view as live SSE.
    pub fn emit(&self, ev: crate::state::UiEvent) {
        if let Some(h) = &self.live {
            h.emit(ev);
        }
    }

    /// Live-append a transcript entry. Writes the entry into
    /// `s.tasks[task_id].transcript` (so `/api/task_transcript` is
    /// always current) AND broadcasts a [`UiEvent::TranscriptAppended`]
    /// (so connected SSE clients with the task selected see it
    /// instantly). This is the way tool calls / tool results / etc.
    /// reach the UI mid-stage — without it they'd buffer in
    /// `ctx.state.transcript` until the engine drains at end-of-call.
    pub fn live_append_transcript(&self, entry: TranscriptEntry) {
        let task_id = self.task_id;
        if let Some(h) = &self.live {
            let cloned = entry.clone();
            h.write(|s| {
                if let Some(t) = s.tasks.get_mut(&task_id) {
                    t.transcript.push(cloned);
                }
            });
            h.emit(crate::state::UiEvent::TranscriptAppended { task_id, entry });
        }
    }

    /// Apply a per-turn token usage delta to canonical state AND emit
    /// a [`UiEvent::TaskCost`] event. The streaming LLM driver calls
    /// this after each turn's `Final` marker so the token counter
    /// ticks live during a long multi-turn call.
    ///
    /// We mutate state (not just emit a view) so that:
    /// 1. The periodic `/api/state` poll picks up the running total
    ///    instead of resetting the UI to the engine's stale "at last
    ///    completed call" value.
    /// 2. Other engine code reading `s.total_cost` (e.g. the cost
    ///    cap) sees the latest figure mid-stage instead of being
    ///    blind to in-flight burn.
    ///
    /// To avoid double-counting at end of `run_role`, the engine no
    /// longer adds `DriveResponse.usage` to state — see the comment
    /// at the end of `run_role`. The streaming driver returns
    /// `DriveResponse.usage = 0` because all of it was applied
    /// incrementally. (The mock driver doesn't stream, so for tests
    /// the engine still applies its returned usage; this is gated on
    /// `DriveResponse.applied_via_streaming`.)
    pub fn live_apply_partial_cost(&self, delta: &crate::state::TokenUsage) {
        let Some(h) = &self.live else { return };
        let task_id = self.task_id;
        let (task_cost, total, estimated_usd) = h.write(|s| {
            if let Some(t) = s.tasks.get_mut(&task_id) {
                t.cost.add(delta);
            }
            s.total_cost.add(delta);
            let task_cost = s
                .tasks
                .get(&task_id)
                .map(|t| t.cost.clone())
                .unwrap_or_default();
            (task_cost, s.total_cost.clone(), s.estimated_cost_usd)
        });
        h.emit(crate::state::UiEvent::TaskCost {
            task_id,
            cost: task_cost,
            total,
            estimated_usd,
        });
    }

    /// Inspectors for the orchestrator. These run between tool calls (with
    /// rig not holding the ctx), so there is no lock contention in
    /// practice; each method just takes the inner mutex briefly.

    /// Clone the current transcript without consuming it. Used to inspect
    /// in-flight state during the forced-retry loop.
    pub fn snapshot_transcript(&self) -> Vec<TranscriptEntry> {
        self.state.lock().transcript.clone()
    }

    /// Drain the transcript — caller takes ownership and the ctx's copy
    /// resets to empty. Called once per role invocation when the
    /// orchestrator merges per-tool entries into the engine task.
    pub fn drain_transcript(&self) -> Vec<TranscriptEntry> {
        std::mem::take(&mut self.state.lock().transcript)
    }

    /// Take the verdict out of the ctx — judge tools set it; orchestrator
    /// takes it after the judge call.
    pub fn take_verdict(&self) -> Option<JudgeVerdict> {
        self.state.lock().verdict.take()
    }

    /// Take the critique out of the ctx — critic tools set it; orchestrator
    /// takes it after the critic call.
    pub fn take_critique(&self) -> Option<Critique> {
        self.state.lock().critique.take()
    }

    /// Drain the queue of file changes from this task's tools. The
    /// orchestrator broadcasts them over SSE.
    pub fn drain_fs_events(&self) -> Vec<PathBuf> {
        std::mem::take(&mut self.state.lock().fs_events)
    }

    /// Set the judge's verdict. Used by `submit_verdict`.
    pub(crate) fn set_verdict(&self, v: JudgeVerdict) {
        self.state.lock().verdict = Some(v);
    }

    /// Set the critic's structured critique. Used by `submit_critique`.
    pub(crate) fn set_critique(&self, c: Critique) {
        self.state.lock().critique = Some(c);
    }

    /// Load the graph from this task's workdir. Tools call this whenever
    /// they need to read or mutate the graph; on mutation they then
    /// call `render_after_write_with(&graph)` to persist + re-render.
    fn load_graph(&self) -> Result<NodeGraph, ToolFailure> {
        crate::graph::load(&self.workdir, self.layout)
            .map_err(|e| ToolFailure::Other(format!("load graph: {e}")))
    }

    fn record_call_check_loop<T: Serialize>(
        &self,
        name: &str,
        args: &T,
    ) -> Result<(), ToolFailure> {
        let s = serde_json::to_string(args).unwrap_or_default();
        let entry = TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::ToolCall {
                tool: name.to_string(),
            },
            content: s.clone(),
            role: Some(self.role),
        };
        // Live-append to canonical state + broadcast. Doing it here
        // (before pushing into ctx.state.transcript) means:
        //   1. `/api/task_transcript` sees this entry immediately
        //      (otherwise a mid-stage click would return a transcript
        //      missing all the in-flight tool calls).
        //   2. Connected SSE clients receive a TranscriptAppended
        //      event right now — no waiting for the engine to drain
        //      ctx.state.transcript at end-of-run_role.
        tracing::debug!(task_id = %self.task_id, tool = %name, "tool_call live emit");
        self.live_append_transcript(entry.clone());
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        s.hash(&mut hasher);
        let h = hasher.finish();
        let mut st = self.state.lock();
        st.transcript.push(entry);
        st.recent_calls.push_back((name.to_string(), h));
        while st.recent_calls.len() > LOOP_WINDOW {
            st.recent_calls.pop_front();
        }
        let consecutive = st
            .recent_calls
            .iter()
            .rev()
            .take_while(|(n, hh)| n == name && *hh == h)
            .count();
        if consecutive >= LOOP_BREAK_THRESHOLD {
            return Err(ToolFailure::Loop {
                tool: name.to_string(),
                count: consecutive,
            });
        }
        Ok(())
    }

    fn finish<T: Serialize>(
        &self,
        name: &str,
        r: Result<T, ToolFailure>,
    ) -> Result<T, ToolFailure> {
        let entry = match &r {
            Ok(v) => TranscriptEntry {
                timestamp: Utc::now(),
                kind: TranscriptKind::ToolResult {
                    tool: name.to_string(),
                    ok: true,
                    error: None,
                    output: serde_json::to_string(v).ok(),
                },
                content: String::new(),
                role: Some(self.role),
            },
            Err(e) => TranscriptEntry {
                timestamp: Utc::now(),
                kind: TranscriptKind::ToolResult {
                    tool: name.to_string(),
                    ok: false,
                    error: Some(format!("{e}")),
                    output: None,
                },
                content: String::new(),
                role: Some(self.role),
            },
        };
        // Live-append to canonical state + broadcast. See
        // record_call_check_loop for the rationale.
        tracing::debug!(
            task_id = %self.task_id,
            tool = %name,
            ok = r.is_ok(),
            "tool_result live emit"
        );
        self.live_append_transcript(entry.clone());
        self.state.lock().transcript.push(entry);
        r
    }

    fn require_stage(&self, tool: &'static str, allowed: &[Stage]) -> Result<(), ToolFailure> {
        if allowed.iter().any(|s| *s == self.stage) {
            return Ok(());
        }
        Err(ToolFailure::WrongStage {
            tool: tool.to_string(),
            stage: self.stage.to_string(),
        })
    }

    /// Re-render the workspace from the given graph state, persisting
    /// `.bureau/graph.json` + `.bureau/nodes/*.json` (via `graph::save`
    /// inside `render_graph`) and tracking which files changed for the
    /// SSE event stream. Tools that mutate the graph in memory call
    /// this with the mutated graph; tools that only read don't.
    ///
    /// Does NOT emit [`UiEvent::GraphTopologyChanged`] — that signal
    /// is reserved for the small set of tools that actually mutate
    /// topology (architect, decompose, root seed). Firing it on every
    /// content-only `submit_*` would have the client refetching
    /// `/api/graph` dozens of times per second under normal write
    /// load, which was overwhelming the BroadcastStream consumer.
    fn render_after_write(&self, graph: &NodeGraph) -> Result<(), ToolFailure> {
        let report = render::render_graph(&self.workdir, graph, self.layout)
            .map_err(|e| ToolFailure::Other(format!("re-render failed: {e}")))?;
        self.state.lock().fs_events.extend(report.files_written);
        Ok(())
    }

    /// Like [`Self::render_after_write`] but also fires
    /// [`UiEvent::GraphTopologyChanged`]. Call this from tools that
    /// added / removed nodes or dep edges (architect's
    /// `submit_architecture`, decompose's child-add path) so the UI
    /// refetches `/api/graph` and repaints the tree. The other
    /// `submit_*` tools just fill slot content and use the plain
    /// version.
    fn render_after_topology_change(&self, graph: &NodeGraph) -> Result<(), ToolFailure> {
        self.render_after_write(graph)?;
        self.emit(crate::state::UiEvent::GraphTopologyChanged);
        Ok(())
    }
}

// --------------------------------------------------------------------------
// submit_spec  (one composite tool — the spec stage's whole submission)
// --------------------------------------------------------------------------
//
// The spec stage doesn't iterate on cargo errors, so there's no value in
// breaking the writer's output across multiple tool calls (each one would
// be a separate API roundtrip shipping the full transcript). Instead the
// writer makes ONE `submit_spec` call carrying the whole submission:
// public spec content, optional private notes, optional children to
// create, optional dep edges to add. The schema dynamically hides the
// `children` field when the decomposition cap is exhausted, which gives
// the same enforcement-by-absence as filtering tools out of the catalog.

#[derive(Deserialize, Serialize, Debug)]
pub struct SubmitSpecArgs {
    /// Public spec markdown — REQUIRED. What dependents and downstream
    /// stages see.
    pub public: String,
    /// Private spec markdown — OPTIONAL. Implementation notes / design
    /// rationale only this node's later writers will see.
    #[serde(default)]
    pub private: Option<String>,
    /// Existing node names that THIS node should depend on. Adds dep
    /// edges from the current node; does NOT create nodes (the architect
    /// stage already laid out the tree). The dep edge is local to this
    /// node — other nodes' state is not mutated.
    #[serde(default)]
    pub deps: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct SubmitSpecOk {
    pub public_bytes: u64,
    pub public_lines: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_bytes: Option<u64>,
    pub deps_added: Vec<NodeIdRef>,
}

pub struct SubmitSpecTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitSpecTool {
    const NAME: &'static str = "submit_spec";
    type Error = ToolFailure;
    type Args = SubmitSpecArgs;
    type Output = SubmitSpecOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let max_spec = self.ctx.max_spec_section_lines;
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "public": {
                        "type": "string",
                        "description": format!(
                            "Public spec markdown — REQUIRED, ≤{max_spec} lines. \
                             Audience: dependents and downstream stages."
                        ),
                    },
                    "private": {
                        "type": "string",
                        "description": format!(
                            "Private spec markdown — OPTIONAL, ≤{max_spec} lines. \
                             Audience: only this node's own iface/impl writers."
                        ),
                    },
                    "deps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "OPTIONAL. Names of existing graph nodes that \
                                        THIS node should depend on. Local mutation \
                                        only — other nodes are not affected."
                    }
                },
                "required": ["public"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<SubmitSpecOk, _>(e));
        }
        let r = submit_spec_apply(&self.ctx, args);
        self.ctx.finish(Self::NAME, r)
    }
}

/// Apply the spec submission. Validates fully before mutating.
fn submit_spec_apply(
    ctx: &Arc<TaskCtx>,
    args: SubmitSpecArgs,
) -> Result<SubmitSpecOk, ToolFailure> {
    ctx.require_stage(SubmitSpecTool::NAME, &[Stage::Spec])?;

    // ---- Validate sizes (public required, private optional) ----
    let public_lines = args.public.lines().count();
    if public_lines > ctx.max_spec_section_lines {
        return Err(ToolFailure::FileTooLarge(
            public_lines,
            ctx.max_spec_section_lines,
        ));
    }
    if args.public.trim().is_empty() {
        return Err(ToolFailure::Subtask(
            "submit_spec: `public` must be non-empty markdown describing what \
             this node does"
                .into(),
        ));
    }
    let priv_lines = args.private.as_deref().map(|s| s.lines().count());
    if let Some(pl) = priv_lines {
        if pl > ctx.max_spec_section_lines {
            return Err(ToolFailure::FileTooLarge(pl, ctx.max_spec_section_lines));
        }
    }

    let mut g = ctx.load_graph()?;
    let self_id = ctx.node_id;
    let self_name = g.get(self_id).map(|n| n.name.clone()).unwrap_or_default();

    // ---- Validate `deps` (resolve names to ids, check self-dep) ----
    let mut deps_resolved: Vec<NodeId> = Vec::new();
    for name in &args.deps {
        if name == &self_name {
            return Err(ToolFailure::Subtask(format!(
                "deps: '{name}' is THIS node — a node cannot depend on itself"
            )));
        }
        let id = g.find_by_name(name).map(|n| n.id).ok_or_else(|| {
            ToolFailure::Subtask(format!("deps: no existing node named '{name}'"))
        })?;
        deps_resolved.push(id);
    }

    // ---- Apply: write spec content, add dep edges ----
    let public_bytes = args.public.len() as u64;
    let private_bytes = args.private.as_ref().map(|p| p.len() as u64);
    {
        let n = g
            .get_mut(self_id)
            .ok_or_else(|| ToolFailure::Other(format!("node {self_id} missing")))?;
        n.spec_public_md = Some(args.public);
        if let Some(p) = args.private {
            n.spec_private_md = Some(p);
        }
        n.updated_at = Utc::now();
    }
    // Add dep edges. Existing edges are no-ops (add_dep dedupes). We
    // used to cascade-reset dependents here when a new dep was added;
    // that mutated nodes outside the current task's scope (the "task X
    // mutating node Y" anti-pattern). Removed: adding a dep doesn't
    // change THIS node's public surface, so dependents continue to
    // compile against the same iface. If a spec change DOES alter the
    // public surface, that's caught at this node's own iface re-run.
    let mut deps_added = Vec::new();
    for to in &deps_resolved {
        g.add_dep(self_id, *to)?;
        let n = g.get(*to).unwrap();
        deps_added.push(NodeIdRef {
            id: n.id.to_string(),
            name: n.name.clone(),
        });
    }

    // Spec can both fill content (always) and add dep edges (sometimes).
    // If deps were added, topology changed — fire the topology-changed
    // signal so the UI refetches /api/graph. Otherwise just content.
    if deps_added.is_empty() {
        ctx.render_after_write(&g)?;
    } else {
        ctx.render_after_topology_change(&g)?;
    }
    Ok(SubmitSpecOk {
        public_bytes,
        public_lines,
        private_bytes,
        deps_added,
    })
}

// --------------------------------------------------------------------------
// submit_public / submit_private / submit_tests
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug)]
pub struct SubmitRustArgs {
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct SubmitRustOk {
    pub bytes: u64,
    pub lines: usize,
    /// True when the new content equals what was already there.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_change: bool,
}

/// Which `*.rs` slot a submit_* tool writes to.
///
/// The submit_public / submit_private / submit_tests tools share a single
/// validation-then-write pipeline; they differ only in (a) which `Stage`s
/// are allowed to call them, (b) which slot on the `Node` they update,
/// and (c) which validator runs. This enum keeps those three knobs in one
/// table; the three `Tool` impls delegate to `submit_rust_slot` below.
#[derive(Clone, Copy)]
enum SubmitSlot {
    Public,
    Private,
    Tests,
}

impl SubmitSlot {
    fn allowed_stages(self) -> &'static [Stage] {
        match self {
            SubmitSlot::Public => &[Stage::Iface],
            SubmitSlot::Private => &[Stage::Iface, Stage::Impl, Stage::Debug],
            SubmitSlot::Tests => &[Stage::Tests, Stage::Debug],
        }
    }

    fn validate(self, content: &str, node: &crate::graph::Node, g: &crate::graph::NodeGraph)
        -> Result<(), ValidateError>
    {
        match self {
            // public.rs has its own narrow validator (forbids impl blocks etc.);
            // private and tests share the import-scope validator.
            SubmitSlot::Public => node_validate::validate_public(content),
            SubmitSlot::Private | SubmitSlot::Tests => {
                node_validate::validate_private(content, node, g)
            }
        }
    }

    /// Read/write the slot on the node. Returns the previous value so the
    /// caller can detect no-change.
    fn slot_mut(self, n: &mut crate::graph::Node) -> &mut Option<String> {
        match self {
            SubmitSlot::Public => &mut n.public_rs,
            SubmitSlot::Private => &mut n.private_rs,
            SubmitSlot::Tests => &mut n.tests_rs,
        }
    }
}

/// Shared body of submit_public / submit_private / submit_tests. Loads the
/// graph, runs the slot-appropriate validator, writes if changed, re-renders.
fn submit_rust_slot(
    ctx: &TaskCtx,
    slot: SubmitSlot,
    content: &str,
    tool_name: &'static str,
) -> Result<SubmitRustOk, ToolFailure> {
    ctx.require_stage(tool_name, slot.allowed_stages())?;
    let lines = content.lines().count();
    if lines > ctx.max_file_lines {
        return Err(ToolFailure::FileTooLarge(lines, ctx.max_file_lines));
    }
    let mut g = ctx.load_graph()?;
    let n = g
        .get(ctx.node_id)
        .ok_or_else(|| ToolFailure::Other(format!("node {} missing", ctx.node_id)))?;
    slot.validate(content, n, &g)?;
    let n = g.get_mut(ctx.node_id).unwrap();
    let cur = slot.slot_mut(n);
    let no_change = cur.as_deref() == Some(content);
    if !no_change {
        *cur = Some(content.to_string());
        n.updated_at = Utc::now();
        ctx.render_after_write(&g)?;
    }
    Ok(SubmitRustOk {
        bytes: content.len() as u64,
        lines,
        no_change,
    })
}

/// One macro-style declaration per public-facing submit_* tool. Each tool
/// type carries the rig `NAME` constant (rig requires it const) and a
/// `SubmitSlot` discriminator; the actual work lives in `submit_rust_slot`.
macro_rules! submit_rust_tool {
    ($ty:ident, $name:literal, $slot:expr) => {
        pub struct $ty {
            pub ctx: Arc<TaskCtx>,
        }

        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = ToolFailure;
            type Args = SubmitRustArgs;
            type Output = SubmitRustOk;

            async fn definition(&self, _prompt: String) -> ToolDefinition {
                ToolDefinition {
                    name: Self::NAME.to_string(),
                    description: tool_description(Self::NAME, self.ctx.prompt_limits()),
                    parameters: json!({
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"]
                    }),
                }
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
                    return self.ctx.finish(Self::NAME, Err::<SubmitRustOk, _>(e));
                }
                let r = submit_rust_slot(&self.ctx, $slot, &args.content, Self::NAME);
                self.ctx.finish(Self::NAME, r)
            }
        }
    };
}

submit_rust_tool!(SubmitPublicTool, "submit_public", SubmitSlot::Public);
submit_rust_tool!(SubmitPrivateTool, "submit_private", SubmitSlot::Private);
submit_rust_tool!(SubmitTestsTool, "submit_tests", SubmitSlot::Tests);

#[derive(Serialize, Debug, Clone)]
pub struct NodeIdRef {
    pub id: String,
    pub name: String,
}

// --------------------------------------------------------------------------
// submit_architecture  (architect stage — builds the whole tree in one shot)
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ArchNode {
    /// snake_case Rust ident; unique among siblings.
    pub name: String,
    /// One-sentence description. Read by every dependent's prompt.
    pub description: String,
    #[serde(default)]
    pub crate_boundary: bool,
    /// Names of OTHER nodes anywhere in the tree this node depends on.
    /// References resolve after the full tree is built.
    #[serde(default)]
    pub deps: Vec<String>,
    /// Recursive children.
    #[serde(default)]
    pub children: Vec<ArchNode>,
}

// External crate deps are defined in `graph::ExternalCrateDep` — the
// same type the renderer reads to populate `[workspace.dependencies]`.
// The architect submits values of that type directly; no conversion.
pub use crate::graph::ExternalCrateDep;

#[derive(Deserialize, Serialize, Debug)]
pub struct SubmitArchitectureArgs {
    /// Top-level children of the workspace root node, recursively
    /// describing the entire tree.
    pub children: Vec<ArchNode>,
    /// Anticipated external Cargo dependencies the project will need.
    /// Stored on the root node for later use; not enforced.
    #[serde(default)]
    pub external_deps: Vec<ExternalCrateDep>,
}

#[derive(Serialize, Debug)]
pub struct SubmitArchitectureOk {
    pub nodes_created: usize,
    pub deps_added: usize,
    pub external_deps: usize,
}

pub struct SubmitArchitectureTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitArchitectureTool {
    const NAME: &'static str = "submit_architecture";
    type Error = ToolFailure;
    type Args = SubmitArchitectureArgs;
    type Output = SubmitArchitectureOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "children": {
                        "type": "array",
                        "description": "Top-level children of the workspace root, recursively describing the whole tree.",
                        "items": arch_node_schema(),
                    },
                    "external_deps": {
                        "type": "array",
                        "description": "Anticipated external Cargo dependencies. Optional.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "reason": {"type": "string"},
                                "features": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["name"]
                        }
                    }
                },
                "required": ["children"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<SubmitArchitectureOk, _>(e));
        }
        let r = submit_architecture_apply(&self.ctx, args);
        self.ctx.finish(Self::NAME, r)
    }
}

/// Recursive JSON schema fragment describing one `ArchNode`. We define
/// it once here and reference it from `submit_architecture` and (in
/// principle) anywhere else that wants to describe the recursive shape.
fn arch_node_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "snake_case Rust ident; unique among siblings."},
            "description": {"type": "string", "description": "One short sentence describing this node's purpose."},
            "crate_boundary": {
                "type": "boolean",
                "default": false,
                "description": "true ONLY at major top-level subsystems that need their own Cargo crate. Most should be false."
            },
            "deps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of OTHER nodes (anywhere in the tree) this node depends on."
            },
            "children": {
                "type": "array",
                "items": {"$ref": "#/definitions/ArchNode"},
                "description": "Recursive children of this node. Most leaves have an empty list."
            }
        },
        "required": ["name", "description"]
    })
}

fn submit_architecture_apply(
    ctx: &Arc<TaskCtx>,
    args: SubmitArchitectureArgs,
) -> Result<SubmitArchitectureOk, ToolFailure> {
    ctx.require_stage(SubmitArchitectureTool::NAME, &[Stage::Architect])?;

    // 1. Flatten the recursive children into a list of (name, parent_name)
    //    plus a list of dep edges. Validate names + structure.
    struct Pending<'a> {
        decl: &'a ArchNode,
        parent_path: Vec<String>,
    }
    let mut flat: Vec<(Vec<String>, &ArchNode)> = Vec::new();
    let mut stack: Vec<Pending> = args
        .children
        .iter()
        .map(|c| Pending {
            decl: c,
            parent_path: Vec::new(),
        })
        .collect();
    while let Some(p) = stack.pop() {
        flat.push((p.parent_path.clone(), p.decl));
        for child in &p.decl.children {
            let mut sub = p.parent_path.clone();
            sub.push(p.decl.name.clone());
            stack.push(Pending {
                decl: child,
                parent_path: sub,
            });
        }
    }
    // (Empty children list is valid — a single-crate project with no
    // sub-modules is a legitimate output. The post-stage check in the
    // engine only requires that the architect ran and called the tool;
    // it doesn't require a non-trivial tree.)

    // 2. Validate names: each must be a valid snake_case Rust ident, and
    //    NAMES MUST BE GLOBALLY UNIQUE so deps can be resolved by name
    //    without ambiguity.
    let mut name_seen: HashSet<String> = HashSet::new();
    for (_path, decl) in &flat {
        if !is_valid_ident(&decl.name) {
            return Err(ToolFailure::Subtask(format!(
                "submit_architecture: node name '{}' is not a valid Rust identifier",
                decl.name
            )));
        }
        if !name_seen.insert(decl.name.clone()) {
            return Err(ToolFailure::Subtask(format!(
                "submit_architecture: node name '{}' is used more than once. Architect-stage \
                 names must be globally unique so deps can be resolved unambiguously.",
                decl.name
            )));
        }
    }

    // 3. Apply: build NodeIds, insert into graph as children of root.
    //    We have to build top-down so parents exist before their kids.
    flat.sort_by_key(|(path, _)| path.len());

    let mut g = ctx.load_graph()?;
    let root_id = g
        .root
        .ok_or_else(|| ToolFailure::Other("graph has no root".into()))?;
    let root_name = g.get(root_id).unwrap().name.clone();
    if name_seen.contains(&root_name) {
        return Err(ToolFailure::Subtask(format!(
            "submit_architecture: node name '{root_name}' clashes with the workspace \
             root's name. Pick a different name for that child."
        )));
    }
    // Map from name → NodeId for resolving deps later.
    let mut by_name: std::collections::HashMap<String, NodeId> =
        std::collections::HashMap::new();
    by_name.insert(root_name.clone(), root_id);

    let mut nodes_created = 0usize;
    for (parent_path, decl) in &flat {
        let parent_id = if parent_path.is_empty() {
            root_id
        } else {
            // The immediate parent is the last segment of parent_path.
            let p = parent_path.last().unwrap();
            *by_name.get(p).ok_or_else(|| {
                ToolFailure::Subtask(format!(
                    "submit_architecture: internal — parent '{p}' not yet created"
                ))
            })?
        };
        let mut node = Node::new(&decl.name, &decl.description);
        node.crate_boundary = decl.crate_boundary;
        let new_id = g.add_child(parent_id, node)?;
        by_name.insert(decl.name.clone(), new_id);
        nodes_created += 1;
    }

    // 4. Apply dep edges (after all nodes exist). These cross subtrees.
    let mut deps_added = 0usize;
    for (_path, decl) in &flat {
        let from = *by_name.get(&decl.name).expect("just created");
        for dep_name in &decl.deps {
            if dep_name == &decl.name {
                return Err(ToolFailure::Subtask(format!(
                    "submit_architecture: node '{}' lists itself in its own `deps`",
                    decl.name
                )));
            }
            let to = *by_name.get(dep_name).ok_or_else(|| {
                ToolFailure::Subtask(format!(
                    "submit_architecture: node '{}' deps on unknown node '{}'. \
                     All dep names must reference nodes declared elsewhere in this same call.",
                    decl.name, dep_name
                ))
            })?;
            g.add_dep(from, to)?;
            deps_added += 1;
        }
    }

    // 5. External crate deps: store as structured data on the root node.
    //    Renderer reads `root.external_crate_deps` and writes them into
    //    `[workspace.dependencies]` (workspace layout) or `[dependencies]`
    //    (single crate). Without this every node that referenced an
    //    external crate would fail to compile because the crate isn't
    //    in any Cargo.toml.
    let external_deps = args.external_deps.len();
    if external_deps > 0 {
        g.get_mut(root_id).unwrap().external_crate_deps = args.external_deps;
    }

    // Architect built the whole tree → topology definitively changed.
    ctx.render_after_topology_change(&g)?;
    Ok(SubmitArchitectureOk {
        nodes_created,
        deps_added,
        external_deps,
    })
}

/// Resolve a model-supplied `package` arg to a REAL workspace member's
/// crate name. Returns `None` to skip the `-p` flag entirely (single-crate
/// mode), `Some(name)` to use a valid workspace package. If the model
/// passed something invalid (a module name, a typo'd crate name), we fall
/// back to the containing crate of the current node — that's almost
/// always what they meant.
fn resolve_cargo_package(ctx: &TaskCtx, requested: Option<&str>) -> Option<String> {
    let g = ctx.load_graph().ok()?;
    // In single-crate layout, cargo doesn't need `-p`; the workdir IS
    // the one and only package.
    if matches!(ctx.layout, crate::render::Layout::SingleCrate) {
        return None;
    }
    // Workspace mode: collect the names of crate-boundary nodes (these
    // are the workspace members).
    let mut members: std::collections::HashSet<String> = std::collections::HashSet::new();
    for n in g.iter() {
        if n.crate_boundary {
            members.insert(n.name.clone());
        }
    }
    // 1. If the model supplied a name and it's a real member, use it.
    if let Some(p) = requested {
        if members.contains(p) {
            return Some(p.to_string());
        }
        // Else: the model handed us a non-package name (typically a
        // module path inside a crate). Fall through to the default —
        // log a hint so the operator can spot misuse but don't fail
        // the call (the model would just retry uselessly).
        tracing::debug!(
            "cargo `package` arg `{p}` is not a workspace member; \
             falling back to current node's crate"
        );
    }
    // 2. Default: the crate that owns the current node.
    if let Some(node) = g.get(ctx.node_id) {
        let crate_id = crate::render::containing_crate(&g, node);
        if let Some(crate_node) = g.get(crate_id) {
            if members.contains(&crate_node.name) {
                return Some(crate_node.name.clone());
            }
            // Crate-boundary node exists but isn't in members? (root,
            // perhaps — root is technically a crate boundary in
            // workspace mode). Still pass its name; cargo will accept
            // it if it's a real package, otherwise omit.
            return Some(crate_node.name.clone());
        }
    }
    None
}

fn is_valid_ident(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// --------------------------------------------------------------------------
// submit_critique
// --------------------------------------------------------------------------
//
// The critic emits a structured list of issues via this tool instead of
// writing free-form prose for the engine to fuzzy-match on. Empty
// `issues` list = nothing to fix = engine skips reviser + judge for
// the round. Each issue carries a `description` (REQUIRED) and an
// optional `location` (path:line as the model best knows it) and
// `severity` ("error" | "warning" | "nit"). The reviser sees the
// rendered list as its critique context.

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CritiqueIssue {
    /// Short concrete description of the problem. The reviser reads
    /// these one-by-one as its task list; this is the *content*, not
    /// just a label.
    pub description: String,
    /// Optional `file:line` (or just `file`) the issue points at.
    #[serde(default)]
    pub location: Option<String>,
    /// Optional severity: "error", "warning", or "nit". Defaults to
    /// "warning" if absent.
    #[serde(default)]
    pub severity: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Critique {
    pub issues: Vec<CritiqueIssue>,
}

impl Critique {
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Render the issues list as a markdown bullet list for the reviser's
    /// critique-cycle context.
    pub fn render(&self) -> String {
        if self.issues.is_empty() {
            return "(critic reported no issues)".to_string();
        }
        let mut s = String::new();
        for (i, issue) in self.issues.iter().enumerate() {
            let sev = issue.severity.as_deref().unwrap_or("warning");
            let loc = issue
                .location
                .as_deref()
                .map(|l| format!(" ({l})"))
                .unwrap_or_default();
            s.push_str(&format!(
                "{n}. [{sev}]{loc} {desc}\n",
                n = i + 1,
                desc = issue.description.trim()
            ));
        }
        s
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SubmitCritiqueArgs {
    /// Zero or more concrete issues with the writer's output. Empty
    /// list = nothing to fix = the framework will skip reviser + judge
    /// for the round.
    #[serde(default)]
    pub issues: Vec<CritiqueIssue>,
}

#[derive(Serialize, Debug)]
pub struct SubmitCritiqueOk {
    pub recorded: bool,
    pub issue_count: usize,
}

pub struct SubmitCritiqueTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitCritiqueTool {
    const NAME: &'static str = "submit_critique";
    type Error = ToolFailure;
    type Args = SubmitCritiqueArgs;
    type Output = SubmitCritiqueOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "location": {"type": "string"},
                                "severity": {
                                    "type": "string",
                                    "enum": ["error", "warning", "nit"]
                                }
                            },
                            "required": ["description"]
                        }
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<SubmitCritiqueOk, _>(e));
        }
        let issue_count = args.issues.len();
        let r: Result<SubmitCritiqueOk, ToolFailure> = {
            self.ctx.set_critique(Critique { issues: args.issues });
            Ok(SubmitCritiqueOk {
                recorded: true,
                issue_count,
            })
        };
        self.ctx.finish(Self::NAME, r)
    }
}

// --------------------------------------------------------------------------
// submit_verdict
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug)]
pub struct SubmitVerdictArgs {
    pub satisfactory: bool,
    #[serde(default)]
    pub reason: String,
}

#[derive(Serialize, Debug)]
pub struct SubmitVerdictOk {
    pub recorded: bool,
}

pub struct SubmitVerdictTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitVerdictTool {
    const NAME: &'static str = "submit_verdict";
    type Error = ToolFailure;
    type Args = SubmitVerdictArgs;
    type Output = SubmitVerdictOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "satisfactory": {"type": "boolean"},
                    "reason": {"type": "string"}
                },
                "required": ["satisfactory"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<SubmitVerdictOk, _>(e));
        }
        let r: Result<SubmitVerdictOk, ToolFailure> = {
            self.ctx.set_verdict(JudgeVerdict {
                satisfactory: args.satisfactory,
                reason: args.reason,
            });
            Ok(SubmitVerdictOk { recorded: true })
        };
        self.ctx.finish(Self::NAME, r)
    }
}

// --------------------------------------------------------------------------
// cargo_check / cargo_test / cargo_test_no_run / cargo_clippy
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug)]
pub struct CargoArgs {
    #[serde(default)]
    pub package: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CargoTestArgs {
    #[serde(default)]
    pub package: Option<String>,
    #[serde(default)]
    pub test_filter: Option<String>,
    #[serde(default)]
    pub test_filters: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct CargoErrorBrief {
    pub id: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub message: String,
}

#[derive(Serialize, Debug)]
pub struct CargoOk {
    pub passed: bool,
    pub errors: Vec<CargoErrorBrief>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    pub total_errors: usize,
    pub stderr_tail: String,
    pub elapsed_ms: u64,
    pub command: String,
}

const MAX_ERRORS_RETURNED: usize = 8;
const MAX_STDERR_TAIL_BYTES: usize = 2048;
const MAX_ERROR_MESSAGE_BYTES: usize = 1200;

async fn run_cargo(
    ctx: &TaskCtx,
    kind: crate::gate::GateKind,
    package: Option<&str>,
    test_filters: &[String],
) -> Result<CargoOk, ToolFailure> {
    let start = std::time::Instant::now();
    let mut args: Vec<String> = kind.args().iter().map(|s| s.to_string()).collect();
    // Resolve `-p <name>` to a REAL workspace member. Models often pass
    // module names (not crates) here; cargo would reject those with a
    // confusing error. We map invalid names to the current node's
    // containing crate, and skip -p entirely in single-crate mode where
    // cargo doesn't need it.
    if let Some(p) = resolve_cargo_package(ctx, package) {
        args.push("-p".to_string());
        args.push(p);
    }
    if !test_filters.is_empty()
        && matches!(
            kind,
            crate::gate::GateKind::Test | crate::gate::GateKind::TestNoRun
        )
    {
        args.push("--".to_string());
        for f in test_filters {
            args.push(f.clone());
        }
    }
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(&args)
        .current_dir(&ctx.workdir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env("CARGO_TERM_COLOR", "never");
    let _guard = ctx.cargo_lock.lock().await;
    let output = cmd
        .output()
        .await
        .map_err(|e| ToolFailure::Other(format!("spawning cargo failed: {e}")))?;
    drop(_guard);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let outcome =
        crate::gate::parse_cargo_output(&stdout, &stderr, output.status.success(), kind);
    let total_errors = outcome.errors.len();
    let truncated = total_errors > MAX_ERRORS_RETURNED;
    let errors = outcome
        .errors
        .into_iter()
        .take(MAX_ERRORS_RETURNED)
        .map(|e| CargoErrorBrief {
            id: e.id,
            file: e.file.map(|p| p.display().to_string()),
            line: e.line,
            message: truncate_bytes(&e.message, MAX_ERROR_MESSAGE_BYTES),
        })
        .collect();
    let stderr_tail = truncate_bytes(&tail_lines(&stderr, 30), MAX_STDERR_TAIL_BYTES);
    Ok(CargoOk {
        passed: outcome.passed,
        errors,
        truncated,
        total_errors,
        stderr_tail,
        elapsed_ms: start.elapsed().as_millis() as u64,
        command: format!("cargo {}", args.join(" ")),
    })
}

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut cut = max;
        while !s.is_char_boundary(cut) && cut > 0 {
            cut -= 1;
        }
        let mut out = s[..cut].to_string();
        out.push_str(&format!("\n…[{} bytes omitted]…", s.len() - cut));
        out
    }
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

pub struct CargoCheckTool {
    pub ctx: Arc<TaskCtx>,
}
impl Tool for CargoCheckTool {
    const NAME: &'static str = "cargo_check";
    type Error = ToolFailure;
    type Args = CargoArgs;
    type Output = CargoOk;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {"package": {"type": "string"}}
            }),
        }
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => {
                run_cargo(&self.ctx, crate::gate::GateKind::Check, args.package.as_deref(), &[])
                    .await
            }
        };
        self.ctx.finish(Self::NAME, r)
    }
}

pub struct CargoTestTool {
    pub ctx: Arc<TaskCtx>,
}
impl Tool for CargoTestTool {
    const NAME: &'static str = "cargo_test";
    type Error = ToolFailure;
    type Args = CargoTestArgs;
    type Output = CargoOk;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "test_filter": {"type": "string"},
                    "test_filters": {"type": "array", "items": {"type": "string"}}
                }
            }),
        }
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let filters = collect_filters(&args);
        let r = match self.ctx.record_call_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => {
                run_cargo(
                    &self.ctx,
                    crate::gate::GateKind::Test,
                    args.package.as_deref(),
                    &filters,
                )
                .await
            }
        };
        self.ctx.finish(Self::NAME, r)
    }
}

pub struct CargoTestNoRunTool {
    pub ctx: Arc<TaskCtx>,
}
impl Tool for CargoTestNoRunTool {
    const NAME: &'static str = "cargo_test_no_run";
    type Error = ToolFailure;
    type Args = CargoTestArgs;
    type Output = CargoOk;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "package": {"type": "string"},
                    "test_filter": {"type": "string"},
                    "test_filters": {"type": "array", "items": {"type": "string"}}
                }
            }),
        }
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let filters = collect_filters(&args);
        let r = match self.ctx.record_call_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => {
                run_cargo(
                    &self.ctx,
                    crate::gate::GateKind::TestNoRun,
                    args.package.as_deref(),
                    &filters,
                )
                .await
            }
        };
        self.ctx.finish(Self::NAME, r)
    }
}

pub struct CargoClippyTool {
    pub ctx: Arc<TaskCtx>,
}
impl Tool for CargoClippyTool {
    const NAME: &'static str = "cargo_clippy";
    type Error = ToolFailure;
    type Args = CargoArgs;
    type Output = CargoOk;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {"package": {"type": "string"}}
            }),
        }
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => run_clippy(&self.ctx, args.package.as_deref()).await,
        };
        self.ctx.finish(Self::NAME, r)
    }
}

async fn run_clippy(ctx: &TaskCtx, package: Option<&str>) -> Result<CargoOk, ToolFailure> {
    let start = std::time::Instant::now();
    let mut args: Vec<String> = vec![
        "clippy".to_string(),
        "--message-format=json".to_string(),
        "--no-deps".to_string(),
    ];
    if let Some(p) = resolve_cargo_package(ctx, package) {
        args.push("-p".to_string());
        args.push(p);
    }
    args.push("--".to_string());
    args.push("-D".to_string());
    args.push("warnings".to_string());
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(&args)
        .current_dir(&ctx.workdir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env("CARGO_TERM_COLOR", "never");
    let _guard = ctx.cargo_lock.lock().await;
    let output = cmd
        .output()
        .await
        .map_err(|e| ToolFailure::Other(format!("spawning cargo clippy failed: {e}")))?;
    drop(_guard);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let outcome = crate::gate::parse_cargo_output(
        &stdout,
        &stderr,
        output.status.success(),
        crate::gate::GateKind::Check,
    );
    let total_errors = outcome.errors.len();
    let truncated = total_errors > MAX_ERRORS_RETURNED;
    let errors = outcome
        .errors
        .into_iter()
        .take(MAX_ERRORS_RETURNED)
        .map(|e| CargoErrorBrief {
            id: e.id,
            file: e.file.map(|p| p.display().to_string()),
            line: e.line,
            message: truncate_bytes(&e.message, MAX_ERROR_MESSAGE_BYTES),
        })
        .collect();
    let stderr_tail = truncate_bytes(&tail_lines(&stderr, 30), MAX_STDERR_TAIL_BYTES);
    Ok(CargoOk {
        passed: outcome.passed,
        errors,
        truncated,
        total_errors,
        stderr_tail,
        elapsed_ms: start.elapsed().as_millis() as u64,
        command: format!("cargo {}", args.join(" ")),
    })
}

fn collect_filters(args: &CargoTestArgs) -> Vec<String> {
    if !args.test_filters.is_empty() {
        args.test_filters.clone()
    } else if let Some(f) = &args.test_filter {
        vec![f.clone()]
    } else {
        Vec::new()
    }
}

// --------------------------------------------------------------------------
// File-editing tools — used in the quickfix loop to let the writer iterate
// on compile / test failures by reading and editing the files it just
// authored, rather than escalating to the critic/reviser cycle for what
// are usually mechanical fixes.
//
// These tools operate via the graph slot for managed files: read_file
// reads from disk; write_file / write_file_range / apply_patch resolve
// the target path to a node-slot (public.rs / private.rs / tests.rs /
// spec markdown), update that slot, and re-render — same code path as
// the submit_* tools, so the model can't accidentally desync the
// rendered tree from the graph.
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug)]
pub struct ReadFileArgs {
    /// Path relative to the workdir root. No absolute paths, no `..`.
    pub path: String,
    /// 1-based line number to start reading at. Optional; defaults to 1.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// 1-based inclusive line number to stop reading at. Optional;
    /// defaults to end of file.
    #[serde(default)]
    pub end_line: Option<usize>,
}

#[derive(Serialize, Debug)]
pub struct ReadFileOk {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub total_lines: usize,
    pub content: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

pub struct ReadFileTool {
    pub ctx: Arc<TaskCtx>,
}

const READ_FILE_MAX_BYTES: usize = 64 * 1024;

impl Tool for ReadFileTool {
    const NAME: &'static str = "read_file";
    type Error = ToolFailure;
    type Args = ReadFileArgs;
    type Output = ReadFileOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1}
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<ReadFileOk, _>(e));
        }
        let r: Result<ReadFileOk, ToolFailure> = (|| {
            let abs = scoped_path(&self.ctx.workdir, &args.path)?;
            let rel = std::path::PathBuf::from(&args.path);
            // Scope: only allow reads of files this node has a legitimate
            // need for. Specifically: own slots, declared deps' public
            // surface, ancestor specs/public, and framework-rendered
            // metadata (Cargo.toml, mod.rs, lib.rs). Anything else
            // (sibling private internals, unrelated nodes' tests, etc.)
            // is rejected — read access has to be scoped or models can
            // peek at things they shouldn't reason about.
            {
                let g = self.ctx.load_graph()?;
                if !is_readable_by_node(&g, self.ctx.node_id, &rel, self.ctx.layout) {
                    let hint = readable_paths_hint(&g, self.ctx.node_id, self.ctx.layout);
                    return Err(ToolFailure::Other(format!(
                        "read_file: path '{}' is not in this node's read scope. Readable \
                         paths are: this node's own slots; ancestor specs (public.md / \
                         private.md); any node's public surface (public.rs / \
                         spec/<path>/public.md). Framework files (mod.rs, lib.rs, \
                         Cargo.toml) are NOT readable — they're auto-generated and carry \
                         no design info.{hint}",
                        args.path
                    )));
                }
            }
            if !abs.exists() {
                // Policy-allowed but absent on disk. Most common cause:
                // the model guessed a slot that hasn't been authored
                // yet (e.g. asking for sibling's public.rs before that
                // node finished its iface stage), or a path the
                // framework doesn't render at all. Tell the model
                // exactly that — the surrounding "Files you can read"
                // context lists the paths that ARE present.
                return Err(ToolFailure::Other(format!(
                    "read_file: '{}' does not exist on disk yet. The path is policy-allowed \
                     but the slot hasn't been authored (or the framework doesn't render this \
                     file). Pick a path from the `Files you can read` context section that \
                     names a real file.",
                    args.path
                )));
            }
            let content = std::fs::read_to_string(&abs).map_err(|e| {
                ToolFailure::Other(format!("read {}: {e}", args.path))
            })?;
            let total = content.lines().count();
            let start = args.start_line.unwrap_or(1).max(1);
            let end = args.end_line.unwrap_or(total).min(total);
            let slice: String = if start > total {
                String::new()
            } else {
                content
                    .lines()
                    .skip(start - 1)
                    .take(end.saturating_sub(start) + 1)
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let truncated = slice.len() > READ_FILE_MAX_BYTES;
            let slice = if truncated {
                let mut cut = READ_FILE_MAX_BYTES;
                while cut > 0 && !slice.is_char_boundary(cut) {
                    cut -= 1;
                }
                slice[..cut].to_string()
            } else {
                slice
            };
            Ok(ReadFileOk {
                path: args.path,
                start_line: start,
                end_line: end,
                total_lines: total,
                content: slice,
                truncated,
            })
        })();
        self.ctx.finish(Self::NAME, r)
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct WriteFileOk {
    pub path: String,
    pub bytes: u64,
    pub lines: usize,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_change: bool,
}

pub struct WriteFileTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for WriteFileTool {
    const NAME: &'static str = "write_file";
    type Error = ToolFailure;
    type Args = WriteFileArgs;
    type Output = WriteFileOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<WriteFileOk, _>(e));
        }
        let r = apply_slot_edit(&self.ctx, &args.path, args.content.clone(), Self::NAME);
        self.ctx.finish(Self::NAME, r.map(|(bytes, lines, no_change)| WriteFileOk {
            path: args.path,
            bytes,
            lines,
            no_change,
        }))
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct WriteFileRangeArgs {
    pub path: String,
    /// 1-based inclusive start line of the range to replace.
    pub start_line: usize,
    /// 1-based inclusive end line of the range to replace.
    pub end_line: usize,
    /// Replacement content (may be multi-line, may be empty to delete).
    pub content: String,
}

pub struct WriteFileRangeTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for WriteFileRangeTool {
    const NAME: &'static str = "write_file_range";
    type Error = ToolFailure;
    type Args = WriteFileRangeArgs;
    type Output = WriteFileOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "content": {"type": "string"}
                },
                "required": ["path", "start_line", "end_line", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<WriteFileOk, _>(e));
        }
        let r: Result<WriteFileOk, ToolFailure> = (|| {
            let current = read_slot_content(&self.ctx, &args.path)?;
            let lines: Vec<&str> = current.lines().collect();
            if args.start_line == 0 || args.end_line < args.start_line {
                return Err(ToolFailure::Other(format!(
                    "write_file_range: invalid range [{},{}] (must be 1-based, end>=start)",
                    args.start_line, args.end_line
                )));
            }
            let total = lines.len();
            let end = args.end_line.min(total);
            let start = args.start_line.min(total + 1);
            let mut new_content = String::new();
            for line in &lines[..start - 1] {
                new_content.push_str(line);
                new_content.push('\n');
            }
            if !args.content.is_empty() {
                new_content.push_str(&args.content);
                if !args.content.ends_with('\n') {
                    new_content.push('\n');
                }
            }
            for line in &lines[end..] {
                new_content.push_str(line);
                new_content.push('\n');
            }
            let (bytes, line_count, no_change) =
                apply_slot_edit(&self.ctx, &args.path, new_content, Self::NAME)?;
            Ok(WriteFileOk {
                path: args.path.clone(),
                bytes,
                lines: line_count,
                no_change,
            })
        })();
        self.ctx.finish(Self::NAME, r)
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ApplyPatchArgs {
    /// A unified-diff patch (`--- a/file\n+++ b/file\n@@ ...`) or a
    /// markdown code block containing one. Multi-file patches are
    /// supported; mpatch detects the format automatically and applies
    /// each hunk with fuzzy matching.
    pub patch: String,
}

#[derive(Serialize, Debug)]
pub struct ApplyPatchOk {
    pub files_changed: Vec<String>,
    pub hunks_applied: usize,
    pub hunks_failed: usize,
}

pub struct ApplyPatchTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for ApplyPatchTool {
    const NAME: &'static str = "apply_patch";
    type Error = ToolFailure;
    type Args = ApplyPatchArgs;
    type Output = ApplyPatchOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: tool_description(Self::NAME, self.ctx.prompt_limits()),
            parameters: json!({
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<ApplyPatchOk, _>(e));
        }
        let r = apply_patch_inner(&self.ctx, &args.patch);
        self.ctx.finish(Self::NAME, r)
    }
}

fn apply_patch_inner(ctx: &Arc<TaskCtx>, patch_text: &str) -> Result<ApplyPatchOk, ToolFailure> {
    let patches = mpatch::parse_auto(patch_text).map_err(|e| {
        ToolFailure::Other(format!("apply_patch: parse failed: {e}"))
    })?;
    if patches.is_empty() {
        return Err(ToolFailure::Other(
            "apply_patch: parsed zero patches (no recognizable diff in input)".into(),
        ));
    }
    let opts = mpatch::ApplyOptions::new();
    let mut files_changed: Vec<String> = Vec::new();
    let mut hunks_applied = 0usize;
    let mut hunks_failed = 0usize;
    for patch in &patches {
        // The Patch struct exposes its target file via `Display`'d header
        // or via fields. We use the most reliable: serialize the patch
        // and extract the `+++ b/<path>` line.
        let header_text = format!("{patch}");
        let path = extract_target_path(&header_text).ok_or_else(|| {
            ToolFailure::Other(
                "apply_patch: could not determine target path from patch header".into(),
            )
        })?;
        let current = read_slot_content(ctx, &path)?;
        // Reconstruct a single-patch string and apply via patch_content_str.
        let single = header_text;
        match mpatch::patch_content_str(&single, Some(&current), &opts) {
            Ok(new_content) => {
                let (_b, _l, no_change) =
                    apply_slot_edit(ctx, &path, new_content, ApplyPatchTool::NAME)?;
                if !no_change {
                    files_changed.push(path);
                    hunks_applied += patch.hunks.len();
                }
            }
            Err(e) => {
                hunks_failed += patch.hunks.len();
                tracing::warn!("apply_patch: hunk(s) failed on {path}: {e}");
            }
        }
    }
    if files_changed.is_empty() && hunks_failed > 0 {
        return Err(ToolFailure::Other(format!(
            "apply_patch: all {hunks_failed} hunk(s) failed to apply (no fuzzy match)"
        )));
    }
    Ok(ApplyPatchOk {
        files_changed,
        hunks_applied,
        hunks_failed,
    })
}

fn extract_target_path(patch_text: &str) -> Option<String> {
    for line in patch_text.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let p = rest.trim();
            let p = p.strip_prefix("b/").unwrap_or(p);
            // Strip a trailing tab+timestamp if present.
            let p = p.split('\t').next().unwrap_or(p);
            if !p.is_empty() && p != "/dev/null" {
                return Some(p.to_string());
            }
        }
    }
    None
}

/// Resolve `path` to its graph slot, fetch the current slot content (or
/// the on-disk rendered fallback for the slot's default placeholder).
fn read_slot_content(ctx: &TaskCtx, path: &str) -> Result<String, ToolFailure> {
    let rel = std::path::PathBuf::from(path);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(ToolFailure::Other(format!(
            "path '{path}' must be a relative path with no parent-dir traversal"
        )));
    }
    let g = ctx.load_graph()?;
    let resolved = render::resolve_path_to_slot(&g, &rel, ctx.layout);
    if let Some((node_id, slot)) = resolved {
        let n = g
            .get(node_id)
            .ok_or_else(|| ToolFailure::Other(format!("node {node_id} missing")))?;
        let s = match slot {
            render::NodeSlot::PublicRs => n.public_rs.clone().unwrap_or_default(),
            render::NodeSlot::PrivateRs => n.private_rs.clone().unwrap_or_default(),
            render::NodeSlot::TestsRs => n.tests_rs.clone().unwrap_or_default(),
            render::NodeSlot::SpecPublicMd => n.spec_public_md.clone().unwrap_or_default(),
            render::NodeSlot::SpecPrivateMd => n.spec_private_md.clone().unwrap_or_default(),
        };
        return Ok(s);
    }
    drop(g);
    // Fall through: read from disk (may be a managed file that hasn't
    // been authored yet — its rendered placeholder is on disk).
    let abs = ctx.workdir.join(&rel);
    std::fs::read_to_string(&abs).map_err(|e| {
        ToolFailure::Other(format!("read {path}: {e}"))
    })
}

/// Apply an edit to the slot identified by `path`. Validates the content
/// using the same per-slot validator as the corresponding `submit_*`
/// tool, then updates the graph and re-renders. Returns
/// `(bytes, lines, no_change)`.
/// Build a short, concrete hint listing a few real, readable paths
/// for the current node. Models frequently call `read_file` with
/// reasonable INTENT but the WRONG path (workspace layout confusion
/// is common: writing `src/foo.rs` when the file is at
/// `crates/<crate>/src/foo/public.rs`). Showing actual readable paths
/// in the error message cuts down on the trial-and-error storm.
fn readable_paths_hint(g: &NodeGraph, self_id: NodeId, layout: render::Layout) -> String {
    let mut samples: Vec<String> = Vec::new();
    if let Some(self_node) = g.get(self_id) {
        let src = render::node_src_dir(g, self_node, layout);
        let spec = render::node_spec_dir(g, self_node);
        samples.push(format!("{}/public.rs", src.display()));
        samples.push(format!("{}/private.rs", src.display()));
        samples.push(format!("{}/public.md", spec.display()));
    }
    // Also show one or two sibling/ancestor public.rs paths so the
    // model has a template for cross-node reads.
    let mut others: Vec<String> = Vec::new();
    for n in g.iter().take(8) {
        if n.id == self_id {
            continue;
        }
        let src = render::node_src_dir(g, n, layout);
        others.push(format!("{}/public.rs", src.display()));
        if others.len() >= 3 {
            break;
        }
    }
    if !others.is_empty() {
        samples.extend(others);
    }
    if samples.is_empty() {
        String::new()
    } else {
        format!("\nExamples of readable paths: {}", samples.join(", "))
    }
}

/// Decide whether the current node has a legitimate reason to read
/// the file at `rel`. The model's reading needs are bounded:
///   - own node's slots: always
///   - any node's PUBLIC surface (`public.rs`, `spec/public.md`):
///     allowed everywhere — these are conceptually the project's
///     visible API
///   - ancestors: also their `private.md` (descendants benefit from
///     ancestor design context)
/// Denied:
///   - other nodes' `private.rs`, `tests.rs`, `private.md`: these
///     are internals or test code that other nodes shouldn't reason
///     about
///   - framework-rendered files (`mod.rs`, `lib.rs`, `Cargo.toml`):
///     these carry no design info — just `pub mod foo;` lines the
///     framework writes itself. Previously allowed; the model burned
///     calls reading them and frequently hit "file does not exist"
///     because not every crate layout produces lib.rs (we render
///     `mod.rs` as the entry point).
pub(crate) fn is_readable_by_node(
    g: &NodeGraph,
    self_id: NodeId,
    rel: &std::path::Path,
    layout: render::Layout,
) -> bool {
    let Some((target_id, slot)) = render::resolve_path_to_slot(g, rel, layout) else {
        return false;
    };
    if target_id == self_id {
        return true;
    }
    // Public surface is readable globally — any node's `public.rs` and
    // `spec/public.md` are conceptually the project's interface.
    if matches!(
        slot,
        render::NodeSlot::PublicRs | render::NodeSlot::SpecPublicMd
    ) {
        return true;
    }
    // For private slots (`private.rs`, `tests.rs`, `spec/private.md`),
    // only ancestors are readable.
    let ancestors = g.ancestors(self_id, false);
    ancestors.contains(&target_id)
}

fn apply_slot_edit(
    ctx: &Arc<TaskCtx>,
    path: &str,
    content: String,
    tool_name: &'static str,
) -> Result<(u64, usize, bool), ToolFailure> {
    let rel = std::path::PathBuf::from(path);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(ToolFailure::Other(format!(
            "path '{path}' must be a relative path with no parent-dir traversal"
        )));
    }
    let line_count = content.lines().count();
    if line_count > ctx.max_file_lines {
        return Err(ToolFailure::FileTooLarge(line_count, ctx.max_file_lines));
    }
    let resolved = {
        let g = ctx.load_graph()?;
        render::resolve_path_to_slot(&g, &rel, ctx.layout)
    };
    let (node_id, slot) = resolved.ok_or_else(|| {
        ToolFailure::Other(format!(
            "{tool_name}: path '{path}' is not a managed slot (allowed slots: \
             <src>/public.rs, <src>/private.rs, <src>/tests.rs, \
             <spec>/public.md, <spec>/private.md). Auto-generated files \
             (mod.rs, lib.rs, Cargo.toml) cannot be edited."
        ))
    })?;
    if node_id != ctx.node_id {
        return Err(ToolFailure::Other(format!(
            "{tool_name}: path '{path}' belongs to another node — you can \
             only edit files in your own node's slots"
        )));
    }
    // Stage-ownership check. The model used to be able to edit any of
    // its node's slots from any stage — so a quickfix during the iface
    // stage could rewrite `tests.rs`. `files_owned_by_stage(iface)`
    // doesn't include tests.rs, so the change wouldn't land on main,
    // but it would still be in the graph slot, and a downstream task
    // rendering from the graph would pick up the unaudited content.
    // Now we restrict edits to exactly the slots this stage owns
    // (matching the submit_* tools' `require_stage` checks).
    {
        let allowed = slots_owned_by_stage(ctx.stage);
        if !allowed.iter().any(|s| *s == slot) {
            return Err(ToolFailure::Other(format!(
                "{tool_name}: slot {:?} is not writable in stage `{}` — only \
                 {:?} are. Use a different path or wait for the appropriate stage.",
                slot, ctx.stage, allowed
            )));
        }
    }
    // Load once: validate against the loaded graph, then mutate +
    // render through the same in-memory copy.
    let mut g = ctx.load_graph()?;
    let n = g
        .get(node_id)
        .ok_or_else(|| ToolFailure::Other(format!("node {node_id} missing")))?;
    match slot {
        render::NodeSlot::PublicRs => {
            node_validate::validate_public(&content)?;
        }
        render::NodeSlot::PrivateRs | render::NodeSlot::TestsRs => {
            node_validate::validate_private(&content, n, &g)?;
        }
        render::NodeSlot::SpecPublicMd | render::NodeSlot::SpecPrivateMd => {
            // Spec is freeform markdown; no content validator beyond
            // the line cap we already enforced above.
        }
    }
    let n = g
        .get_mut(node_id)
        .ok_or_else(|| ToolFailure::Other(format!("node {node_id} missing")))?;
    let cur: Option<&String> = match slot {
        render::NodeSlot::PublicRs => n.public_rs.as_ref(),
        render::NodeSlot::PrivateRs => n.private_rs.as_ref(),
        render::NodeSlot::TestsRs => n.tests_rs.as_ref(),
        render::NodeSlot::SpecPublicMd => n.spec_public_md.as_ref(),
        render::NodeSlot::SpecPrivateMd => n.spec_private_md.as_ref(),
    };
    let no_change = cur.map(|s| s == &content).unwrap_or(false);
    if !no_change {
        match slot {
            render::NodeSlot::PublicRs => n.public_rs = Some(content.clone()),
            render::NodeSlot::PrivateRs => n.private_rs = Some(content.clone()),
            render::NodeSlot::TestsRs => n.tests_rs = Some(content.clone()),
            render::NodeSlot::SpecPublicMd => n.spec_public_md = Some(content.clone()),
            render::NodeSlot::SpecPrivateMd => n.spec_private_md = Some(content.clone()),
        }
        n.updated_at = Utc::now();
        ctx.render_after_write(&g)?;
    }
    Ok((content.len() as u64, line_count, no_change))
}

/// Resolve `path` against `workdir`, rejecting absolute paths and any
/// component that would escape the workdir.
fn scoped_path(workdir: &std::path::Path, path: &str) -> Result<std::path::PathBuf, ToolFailure> {
    let rel = std::path::PathBuf::from(path);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(ToolFailure::Other(format!(
            "path '{path}' must be a relative path with no parent-dir traversal"
        )));
    }
    Ok(workdir.join(&rel))
}

// --------------------------------------------------------------------------
// Tool catalogs by stage and role.
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Writer,
    Critic,
    Reviser,
    Judge,
    /// Inner-loop role that runs after writer/reviser whenever the cargo
    /// gate fails. Sees the failing compiler/test output and gets
    /// read_file/write_file/apply_patch tools to iterate on fixes
    /// directly, instead of escalating to the critic for what's usually
    /// a mechanical fix-it-up cycle.
    QuickFixer,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Writer => "writer",
            Role::Critic => "critic",
            Role::Reviser => "reviser",
            Role::Judge => "judge",
            Role::QuickFixer => "quickfixer",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tool name catalog for a given (stage, role). Used by the engine to
/// register the right tools and by the UI to render "tools available".
/// Knobs the prompt builder needs from `Limits` (just the line caps, so
/// callers don't need to plumb the whole struct around).
#[derive(Debug, Clone, Copy)]
pub struct PromptLimits {
    pub max_file_lines: usize,
    pub max_spec_section_lines: usize,
}

/// Description string sent to the model for each tool. Single source of
/// truth — every `Tool::definition()` impl pulls from here, and
/// `tool_definitions_for` exposes the same strings to the engine for
/// transcript recording. Limits are interpolated from the runtime config
/// so the model is told the actual hard caps, not a guess.
pub fn tool_description(name: &str, limits: PromptLimits) -> String {
    let max_file = limits.max_file_lines;
    let max_spec = limits.max_spec_section_lines;
    match name {
        "submit_architecture" => format!(
            "Build the WHOLE project tree in ONE call. The Architect stage runs once, \
            on the root node, before any per-node stages. You produce the SKELETON: \
            crates, modules, parent-child relationships, cross-node dependency edges, \
            anticipated external Cargo deps. You do NOT write specs, traits, or code \
            here — those come later, run per-node, on the graph you produce.\n\
            \n\
            Fields:\n\
            - `children` (REQUIRED) — top-level children of the workspace root, \
              recursively describing the whole tree. Each node has:\n\
                · `name` — snake_case Rust ident, GLOBALLY unique across the whole \
                  tree so deps can be resolved unambiguously by name.\n\
                · `description` — one short sentence on what the node is for.\n\
                · `crate_boundary` (default false) — set true ONLY at major top-level \
                  subsystem boundaries that need their own Cargo crate. Most should \
                  be false: children become modules within their parent's crate.\n\
                · `deps` — names of OTHER nodes (anywhere in the tree) this node \
                  depends on. Resolved after the full tree is built; cycle-checked \
                  at the node AND crate level.\n\
                · `children` — recursive children. Most leaves have an empty list.\n\
            - `external_deps` (optional) — anticipated crates.io dependencies the \
              project will need, with a one-sentence reason. Stored for downstream \
              stages; not a binding contract.\n\
            \n\
            Sizing: the framework caps total node count and depth (visible in the \
            Decomposition budget section). Aim shallower-and-broader rather than \
            deeper-and-narrower. Each leaf module ≈ one Rust file (per-file cap \
            {max_file} lines)."
        ),

        "submit_spec" => format!(
            "Submit THIS node's spec — the spec stage's whole writer output in ONE call. \
            Required: `public`. Optional: `private`, `deps`. Call once per spec stage.\n\
            \n\
            The spec is a SPECIFICATION DOCUMENT describing the software, NOT a literate \
            Rust file. Specs talk about data shapes, ownership, concurrency, error model, \
            invariants, I/O surfaces — at the level of REQUIREMENTS and ARCHITECTURE. \
            The iface stage is what writes Rust traits and signatures; do not preempt \
            that here. THIS STAGE CANNOT CREATE NEW NODES — the architect already laid \
            out the tree. The most you can do to the graph is add new dep edges via the \
            `deps` field if you discover one was missing.\n\
            \n\
            Fields:\n\
            - `public` (REQUIRED, ≤{max_spec} lines) — the INTERFACE specification: \
              what dependents observe and rely on. Suggested headings: `## What it \
              does`, `## Public surface`, `## Invariants and guarantees`, `## Out of \
              scope`. Avoid the word `Goal` — describe behaviour, not aspiration. Only \
              externally-observable concepts go here; internal types/backends/helpers \
              go in `private`.\n\
            - `private` (optional, ≤{max_spec} lines) — the IMPLEMENTATION \
              specification: HOW the node is built internally — backends, internal \
              data structures, concurrency, algorithmic notes, tradeoffs considered. \
              Audience: this node's own future iface/impl writers. Other nodes never \
              see this. NOT a changelog of your edits.\n\
            - `deps` (optional) — names of existing graph nodes that THIS node depends \
              on. Adds dep edges to the graph; cycle-checked at the node AND crate \
              level. This is a LOCAL mutation — only this node's deps list changes; \
              dependents are not touched.\n\
            \n\
            (Per-file cap for code is {max_file} lines, for context.)"
        ),

        "submit_public" => format!(
            "Author `public.rs` — the node's public API surface. This file must \
            DEFINE each public item; it must NOT re-export anything. \
            ALLOWED: `pub trait Foo {{ fn bar(...) -> ...; }}` (signatures only, \
            NO method bodies, NO default impls); `pub struct/enum/type/const/static` \
            declarations defined here; non-pub `use super::private::Inner` to refer \
            to a private type in a public type position (e.g. \
            `pub struct Wrapper(super::private::Inner);`); doc comments. \
            FORBIDDEN: ANY `pub use` — including `pub use super::private::Foo`, \
            `pub use private::Foo`, `pub use self::*`, `pub use std::*`, and \
            `pub use crate::other_node::Foo`. The smuggle pattern of defining a \
            type in `private.rs` and re-exporting it here is rejected; move the \
            DEFINITION into `public.rs`. Also forbidden: `mod` (the framework \
            auto-generates module scaffolding); `impl` blocks; `fn` outside trait \
            decls; `extern crate`; macro invocations. To rename a foreign type, \
            use `pub type Alias = <foreign>;`. Hard cap: {max_file} lines."
        ),

        "submit_private" => format!(
            "Author `private.rs` — the node's hidden implementation. \
            Module-path rules: \
            - `use super::public::*;` to reference your OWN public types (NEVER \
              `use crate::TypeName`). \
            - `use crate::<dep_name>::...` for declared deps in single-crate mode; \
              `use <dep_crate>::...` for cross-crate deps in workspace mode. The \
              `import as` line in each Dependency context section is authoritative. \
            - The first segment after `crate::` MUST resolve to a declared dep, an \
              ancestor, an own child, or this node itself; otherwise the framework \
              rejects the submission before invoking cargo. \
            Hard cap: {max_file} lines."
        ),

        "submit_tests" => format!(
            "Author `tests.rs` — `#[test]` functions exercising this node's public \
            surface. Use `use super::public::*;` to import the surface (NOT \
            `use crate::TypeName`). Same `use crate::<X>::...` rule as `submit_private`: \
            X must be a declared dep / ancestor / own child. Hard cap: {max_file} lines."
        ),

        "submit_verdict" => "Record the judge's verdict. Pass `satisfactory: true` if the reviser \
addressed every critic point (or there were no points), or \
`satisfactory: false` with a concrete `reason` quoting the unaddressed \
point(s). When in doubt: `satisfactory: true`. Call exactly once."
            .to_string(),

        "submit_critique" => "Record the critic's structured list of issues. Each issue has a \
required `description` (a concrete, actionable problem with the writer's output — \
not a restatement of what's good) and optional `location` (file:line if known) and \
`severity` (`error` | `warning` | `nit`). Pass an EMPTY `issues` list if the writer's \
output has no actionable problems — the framework will skip the reviser and judge \
for this round. Call exactly once. Do NOT include cosmetic praise, style preferences \
that aren't in the spec/style guide, or rephrasings of fine output."
            .to_string(),

        "cargo_check" => "Run `cargo check` on the workspace and return structured diagnostics \
(file:line + message). Use mid-task to verify what you wrote compiles \
before finishing. Capped at ~8 errors + 2KB stderr per call. Optional \
`package` narrows to a single crate."
            .to_string(),

        "cargo_test" => "Run `cargo test --no-fail-fast` and return compile errors plus runtime \
test failures. `test_filter` / `test_filters` narrow to substring-matched \
tests. Optional `package` narrows to a single crate. Cheap relative to \
LLM tokens — use it freely."
            .to_string(),

        "cargo_test_no_run" => "Run `cargo test --no-run` to verify tests COMPILE without running \
them. `test_filter` / `test_filters` narrow scope; optional `package` \
narrows to one crate."
            .to_string(),

        "cargo_clippy" => "Run `cargo clippy` and return lint warnings. Optional `package` narrows \
to one crate. Returns warnings up to a small cap."
            .to_string(),

        "read_file" => format!(
            "Read a file from the workspace. `path` is relative to the workdir \
            root (no absolute paths, no `..`). Optional `start_line` / `end_line` \
            (1-based, inclusive) narrow the slice; default reads the whole file. \
            Returns up to {kb} KB of content; longer reads are truncated. Use \
            this in the quickfix loop to inspect surrounding code when fixing \
            compile / test errors.",
            kb = READ_FILE_MAX_BYTES / 1024
        ),

        "write_file" => format!(
            "Replace the whole content of a managed file. `path` must point at \
            one of THIS node's slots: `<src>/public.rs`, `<src>/private.rs`, \
            `<src>/tests.rs`, `<spec>/public.md`, `<spec>/private.md`. \
            Auto-generated files (`mod.rs`, `lib.rs`, `Cargo.toml`) cannot be \
            edited — those are framework-rendered. Hard cap: {max_file} lines. \
            Validation runs (same as `submit_*`); on success the graph slot is \
            updated and the workspace is re-rendered."
        ),

        "write_file_range" => format!(
            "Replace lines `[start_line, end_line]` (1-based, inclusive) in a \
            managed file with `content`. Same scope and slot rules as \
            `write_file`. Use this for localized edits — e.g. fixing one \
            function's body — without re-sending the whole file. To DELETE \
            a range, pass empty `content`. To INSERT before a line, pass \
            `start_line == end_line == line_number_to_insert_at` and put the \
            inserted lines + the original line in `content`. After-edit cap \
            still applies: total {max_file} lines."
        ),

        "apply_patch" => "Apply a unified diff (or markdown code block containing one) to \
managed files. `patch` may contain multiple files; mpatch detects the \
format automatically and uses fuzzy matching so slightly stale context \
still applies. Same slot rules as `write_file`: only THIS node's slots \
can be edited. Returns the list of files changed and a per-hunk \
applied/failed count."
            .to_string(),

        _ => "(no description registered)".to_string(),
    }
}

/// (name, description) pairs for every tool registered for the given
/// (stage, role). Used by the engine to record what the model was told
/// the tools do, so the UI can show it.
pub fn tool_definitions_for(
    stage: Stage,
    role: Role,
    limits: PromptLimits,
) -> Vec<ToolDefSnapshot> {
    tool_names_for(stage, role)
        .into_iter()
        .map(|name| ToolDefSnapshot {
            name: name.to_string(),
            description: tool_description(name, limits),
        })
        .collect()
}

/// Tool catalog for one (stage, role) invocation.
///
/// **For every non-quickfix call, this returns the same constant list.**
/// The set is identical across all stages and roles so the tool-list
/// portion of the LLM request — which providers include in the prompt
/// cache key — is byte-identical across the entire run. That lets the
/// cross-stage / cross-node / cross-role caching wins from #80, #84,
/// #85 actually land at the provider side instead of being defeated by
/// a per-stage tool list.
///
/// Per-stage runtime gating still works: each tool's `call()` does its
/// own `require_stage` check and rejects inappropriate calls with a
/// clear error. The user prompt's role block tells the model which
/// tools to actually call at this (stage, role). The model occasionally
/// picks a wrong one and burns one turn on the rejection — bounded
/// cost, single cache hit on the tool list pays for itself many times
/// over.
///
/// Quickfix is its own mode (different tools entirely — read/write/patch
/// for direct file edits) and keeps a separate list.
pub fn tool_names_for(stage: Stage, role: Role) -> Vec<&'static str> {
    if matches!(role, Role::QuickFixer) {
        return quickfix_tools_for(stage);
    }
    unified_tool_names()
}

/// The constant tool list every non-quickfix (stage, role) sees. The
/// only deliberate omission is the quickfix-only set (write_file,
/// write_file_range, apply_patch) — those bypass the submit_* validation
/// pipeline and only make sense inside the quickfix inner loop.
pub fn unified_tool_names() -> Vec<&'static str> {
    vec![
        ReadFileTool::NAME,
        SubmitArchitectureTool::NAME,
        SubmitSpecTool::NAME,
        SubmitPublicTool::NAME,
        SubmitPrivateTool::NAME,
        SubmitTestsTool::NAME,
        SubmitCritiqueTool::NAME,
        SubmitVerdictTool::NAME,
        CargoCheckTool::NAME,
        CargoTestTool::NAME,
        CargoTestNoRunTool::NAME,
        CargoClippyTool::NAME,
    ]
}

/// Which tools the framework will ACCEPT at the given (stage, role).
/// Other tools in the unified catalog are listed but their `call()` will
/// reject with a "wrong stage/role" error. Surfaced in the user prompt
/// so the model knows what's actually going to work at this turn.
///
/// For QuickFixer, the tool list is already the quickfix-specific set
/// (read/write/patch + the stage's gate tool) — every tool in it is
/// eligible by construction, so we just return that list.
pub fn tools_accepted_at(stage: Stage, role: Role) -> Vec<&'static str> {
    use Role::*;
    use Stage::*;
    if matches!(role, QuickFixer) {
        return quickfix_tools_for(stage);
    }
    // read_file is universal — every non-quickfix role can inspect files.
    let mut v: Vec<&'static str> = vec![ReadFileTool::NAME];
    match (stage, role) {
        (Architect, Writer) => v.push(SubmitArchitectureTool::NAME),
        (Architect, _) => {} // single-shot stage, non-writer roles produce nothing.

        (Spec, Writer) | (Spec, Reviser) => v.push(SubmitSpecTool::NAME),
        (Spec, Critic) => v.push(SubmitCritiqueTool::NAME),
        (Spec, Judge) => v.push(SubmitVerdictTool::NAME),

        (Iface, Writer) | (Iface, Reviser) => {
            v.push(SubmitPublicTool::NAME);
            v.push(SubmitPrivateTool::NAME);
            v.push(CargoCheckTool::NAME);
        }
        (Iface, Critic) => {
            v.push(CargoCheckTool::NAME);
            v.push(SubmitCritiqueTool::NAME);
        }
        (Iface, Judge) => {
            v.push(CargoCheckTool::NAME);
            v.push(SubmitVerdictTool::NAME);
        }

        (Tests, Writer) | (Tests, Reviser) => {
            v.push(SubmitTestsTool::NAME);
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestNoRunTool::NAME);
        }
        (Tests, Critic) => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestNoRunTool::NAME);
            v.push(SubmitCritiqueTool::NAME);
        }
        (Tests, Judge) => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestNoRunTool::NAME);
            v.push(SubmitVerdictTool::NAME);
        }

        (Impl, Writer) | (Impl, Reviser) => {
            v.push(SubmitPrivateTool::NAME);
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestTool::NAME);
            v.push(CargoClippyTool::NAME);
        }
        (Impl, Critic) => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestTool::NAME);
            v.push(CargoClippyTool::NAME);
            v.push(SubmitCritiqueTool::NAME);
        }
        (Impl, Judge) => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestTool::NAME);
            v.push(SubmitVerdictTool::NAME);
        }

        (Debug, Writer) | (Debug, Reviser) => {
            v.push(SubmitPrivateTool::NAME);
            v.push(SubmitTestsTool::NAME);
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestTool::NAME);
            v.push(CargoClippyTool::NAME);
        }
        (Debug, Critic) => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestTool::NAME);
            v.push(SubmitCritiqueTool::NAME);
        }
        (Debug, Judge) => {
            v.push(CargoTestTool::NAME);
            v.push(SubmitVerdictTool::NAME);
        }

        (_, QuickFixer) => unreachable!("quickfix handled at top of fn"),
    }
    v
}

/// Instantiate the rig `Tool` impl that corresponds to a tool name from
/// `tool_names_for`. Single source of truth for "name → impl" — both the
/// production LLM driver and the scripted mock driver route through this
/// to attach / invoke tools by their catalog name.
///
/// A missing arm is a bug: `tool_names_for` and `instantiate_tool` must
/// cover the same set of names.
pub fn instantiate_tool(name: &str, ctx: Arc<TaskCtx>) -> Box<dyn rig::tool::ToolDyn> {
    match name {
        n if n == ReadFileTool::NAME => Box::new(ReadFileTool { ctx }),
        n if n == SubmitArchitectureTool::NAME => Box::new(SubmitArchitectureTool { ctx }),
        n if n == SubmitSpecTool::NAME => Box::new(SubmitSpecTool { ctx }),
        n if n == SubmitPublicTool::NAME => Box::new(SubmitPublicTool { ctx }),
        n if n == SubmitPrivateTool::NAME => Box::new(SubmitPrivateTool { ctx }),
        n if n == SubmitTestsTool::NAME => Box::new(SubmitTestsTool { ctx }),
        n if n == SubmitCritiqueTool::NAME => Box::new(SubmitCritiqueTool { ctx }),
        n if n == SubmitVerdictTool::NAME => Box::new(SubmitVerdictTool { ctx }),
        n if n == CargoCheckTool::NAME => Box::new(CargoCheckTool { ctx }),
        n if n == CargoTestTool::NAME => Box::new(CargoTestTool { ctx }),
        n if n == CargoTestNoRunTool::NAME => Box::new(CargoTestNoRunTool { ctx }),
        n if n == CargoClippyTool::NAME => Box::new(CargoClippyTool { ctx }),
        n if n == WriteFileTool::NAME => Box::new(WriteFileTool { ctx }),
        n if n == WriteFileRangeTool::NAME => Box::new(WriteFileRangeTool { ctx }),
        n if n == ApplyPatchTool::NAME => Box::new(ApplyPatchTool { ctx }),
        other => panic!(
            "instantiate_tool: no rig Tool impl registered for `{other}` — \
             tool_names_for and instantiate_tool are out of sync"
        ),
    }
}

/// Tool set for the quickfix inner loop. Same shape for every stage:
/// read/write/patch on the node's slots, plus the gate's diagnostic
/// tool so the model can re-check after each edit without having to
/// guess whether the fix worked.
/// Slots a given stage is permitted to write to. Used by the quickfix
/// file-edit tools (`write_file`, `apply_patch`, etc.) to reject edits
/// outside the stage's scope — without this check, an Iface-stage
/// quickfix could overwrite `tests.rs` and the change would never go
/// through the gate.
pub(crate) fn slots_owned_by_stage(stage: Stage) -> &'static [crate::render::NodeSlot] {
    use crate::render::NodeSlot::*;
    match stage {
        Stage::Architect => &[],
        Stage::Spec => &[SpecPublicMd, SpecPrivateMd],
        Stage::Iface => &[PublicRs, PrivateRs],
        Stage::Tests => &[TestsRs],
        Stage::Impl => &[PrivateRs],
        Stage::Debug => &[PrivateRs, TestsRs],
    }
}

fn quickfix_tools_for(stage: Stage) -> Vec<&'static str> {
    use Stage::*;
    let mut v = vec![
        ReadFileTool::NAME,
        WriteFileTool::NAME,
        WriteFileRangeTool::NAME,
        ApplyPatchTool::NAME,
    ];
    match stage {
        Architect | Spec => {
            // No cargo gate at these stages — quickfix shouldn't be
            // triggered for them. Return the editing tools anyway so
            // the catalog is total; callers gate on whether to invoke.
        }
        Iface => v.push(CargoCheckTool::NAME),
        Tests => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestNoRunTool::NAME);
        }
        Impl | Debug => {
            v.push(CargoCheckTool::NAME);
            v.push(CargoTestTool::NAME);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;
    use crate::render::Layout;

    /// Test fixture. Builds a fresh single-crate workdir with a root
    /// node, persists it via `render::render_graph` (which writes the
    /// `.bureau/` graph state), and hands back the TempDir + root id +
    /// a fresh TaskCtx pointing at it. Tests that need to read or
    /// mutate the graph go through `graph::load`/`graph::save` against
    /// `ctx.workdir` — the same path the tools use.
    fn fixture(stage: Stage) -> (tempfile::TempDir, NodeId, Arc<TaskCtx>) {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "the app")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            stage,
            Role::Writer,
            workdir,
            Layout::SingleCrate,
            300,
            500,
            64,
            5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        (tmp, root, ctx)
    }

    /// Build a fixture with the StateHandle wired up so live-emit
    /// paths can be exercised. The state's task table has a single
    /// task entry for the ctx's task_id so live_append_transcript has
    /// somewhere to write to.
    fn fixture_with_state(stage: Stage) -> (tempfile::TempDir, Arc<TaskCtx>, crate::state::StateHandle) {
        use crate::state::{EngineState, StateHandle};
        let (tmp, _root, ctx_no_state) = fixture(stage);
        let task_id = ctx_no_state.task_id;
        let workdir = ctx_no_state.workdir.clone();
        let state = StateHandle::new(EngineState::new(
            workdir.clone(),
            workdir.clone(),
            "app".into(),
        ));
        // Pre-create a task entry so live_append_transcript finds it.
        state.write(|s| {
            s.tasks.insert(
                task_id,
                crate::state::EngineTask {
                    id: task_id,
                    node_id: ctx_no_state.node_id,
                    node_name: "app".into(),
                    stage,
                    status: crate::state::TaskStatus::Running,
                    model: "mock".into(),
                    transcript: Vec::new(),
                    cost: crate::state::TokenUsage::default(),
                    started_at: None,
                    finished_at: None,
                    error: None,
                    final_verdict: None,
                    retries: 0,
                },
            );
        });
        let ctx = Arc::new(
            TaskCtx::new(
                task_id,
                ctx_no_state.node_id,
                stage,
                Role::Writer,
                workdir,
                Layout::SingleCrate,
                300,
                500,
                64,
                5,
                Arc::new(tokio::sync::Mutex::new(())),
            )
            .with_state(state.clone()),
        );
        (tmp, ctx, state)
    }

    #[tokio::test]
    async fn tool_call_live_emits_event_and_appends_to_canonical_state() {
        // The streaming-visibility contract: when a tool runs, the
        // tool_call AND tool_result entries must be visible BEFORE
        // the engine drains ctx at end-of-run_role. Without this, the
        // UI sees a multi-minute blank gap during tool-heavy stages
        // and `/api/task_transcript` returns a stale view if the user
        // clicks the task mid-stage.
        let (_tmp, ctx, state) = fixture_with_state(Stage::Spec);
        let task_id = ctx.task_id;
        let mut rx = state.subscribe();

        let tool = SubmitSpecTool { ctx };
        tool.call(SubmitSpecArgs {
            public: "# Spec\n\nDoes the thing.".into(),
            private: None,
            deps: vec![],
        })
        .await
        .unwrap();

        // Canonical state has BOTH the tool_call AND tool_result
        // entries — already, no engine drain needed.
        let entries = state.read(|s| s.tasks.get(&task_id).unwrap().transcript.clone());
        let kinds: Vec<_> = entries
            .iter()
            .map(|e| match &e.kind {
                TranscriptKind::ToolCall { .. } => "tool_call",
                TranscriptKind::ToolResult { .. } => "tool_result",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["tool_call", "tool_result"],
            "expected tool_call followed by tool_result in canonical state"
        );

        // SSE subscribers see them as separate TranscriptAppended
        // events (in the right order).
        let mut got_kinds: Vec<&'static str> = Vec::new();
        // Drain whatever's already queued — both events should be
        // sitting there because `tool.call` is fully synchronous from
        // our point of view (no awaits inside the lifecycle except
        // ones rig might insert, which still complete before we get
        // here).
        loop {
            match rx.try_recv() {
                Ok(crate::state::UiEvent::TranscriptAppended { entry, .. }) => {
                    got_kinds.push(match &entry.kind {
                        TranscriptKind::ToolCall { .. } => "tool_call",
                        TranscriptKind::ToolResult { .. } => "tool_result",
                        _ => "other",
                    });
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert_eq!(got_kinds, vec!["tool_call", "tool_result"]);
    }

    #[tokio::test]
    async fn submit_spec_public_persists_to_node_and_disk() {
        let (tmp, root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# Spec\n\nDoes the thing.".into(),
                private: None,
                deps: vec![],
            })
            .await
            .unwrap();
        assert!(r.public_lines >= 2);
        assert_eq!(
            crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap().get(root).unwrap().spec_public_md.as_deref(),
            Some("# Spec\n\nDoes the thing.")
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("spec/app/public.md")).unwrap();
        assert!(on_disk.contains("Does the thing"));
    }

    #[tokio::test]
    async fn submit_spec_private_writes_to_separate_slot_and_file() {
        let (tmp, root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        tool.call(SubmitSpecArgs {
            public: "# Spec\n\nDoes the thing.".into(),
            private: Some("# Notes\n\nWhy I chose option B.".into()),
            deps: vec![],
        })
        .await
        .unwrap();
        assert_eq!(
            crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap().get(root).unwrap().spec_private_md.as_deref(),
            Some("# Notes\n\nWhy I chose option B.")
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("spec/app/private.md")).unwrap();
        assert!(on_disk.contains("option B"));
    }

    #[tokio::test]
    async fn submit_spec_rejected_outside_spec_stage() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitSpecTool { ctx };
        let err = tool
            .call(SubmitSpecArgs {
                public: "x".into(),
                private: None,
                deps: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::WrongStage { .. }));
    }

    #[tokio::test]
    async fn submit_spec_adding_dep_does_not_mutate_other_nodes() {
        // Adding a dep edge from THIS node to another is a local
        // mutation: only this node's `deps` field changes. We
        // intentionally do NOT cascade-reset dependents — adding a dep
        // doesn't alter this node's public surface, so dependents
        // continue to compile against the same iface they already
        // tested against. If a spec change DOES alter the public
        // surface, that's caught at this node's own iface re-run.
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let util = g.add_child(root, Node::new("util", "shared")).unwrap();
        let core = g.add_child(root, Node::new("core", "main logic")).unwrap();
        let leaf = g.add_child(core, Node::new("leaf", "depends on core")).unwrap();
        g.add_dep(leaf, core).unwrap();
        // Pretend leaf, core, util all have iface/impl Done.
        for id in [util, core, leaf] {
            let n = g.get_mut(id).unwrap();
            n.stages.spec = StageState::Done;
            n.stages.iface = StageState::Done;
            n.stages.impl_ = StageState::Done;
        }
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        // Run a spec stage on `core` that adds a dep on `util`.
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            core,
            Stage::Spec,
            Role::Writer,
            workdir,
            Layout::SingleCrate,
            300,
            500,
            64,
            5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# core\n\nNow uses util.".into(),
                private: None,
                deps: vec!["util".into()],
            })
            .await
            .unwrap();
        assert_eq!(r.deps_added.len(), 1);
        let g = crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap();
        // leaf's iface MUST stay Done — we don't mutate dependents.
        assert_eq!(g.get(leaf).unwrap().stages.iface, StageState::Done);
        assert_eq!(g.get(leaf).unwrap().stages.impl_, StageState::Done);
        // util's stages also untouched.
        assert_eq!(g.get(util).unwrap().stages.iface, StageState::Done);
        // core has the new dep.
        assert!(g.get(core).unwrap().deps.contains(&util));
    }

    #[tokio::test]
    async fn submit_architecture_builds_a_full_tree_in_one_call() {
        // The architect tool: one shot, builds the whole graph.
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "the app")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            Stage::Architect,
            Role::Writer,
            workdir,
            Layout::SingleCrate,
            300,
            500,
            64,
            5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitArchitectureTool { ctx };
        let r = tool
            .call(SubmitArchitectureArgs {
                children: vec![
                    ArchNode {
                        name: "util".into(),
                        description: "shared utilities".into(),
                        crate_boundary: false,
                        deps: vec![],
                        children: vec![],
                    },
                    ArchNode {
                        name: "core".into(),
                        description: "core logic".into(),
                        crate_boundary: true,
                        deps: vec!["util".into()],
                        children: vec![ArchNode {
                            name: "engine".into(),
                            description: "inner engine".into(),
                            crate_boundary: false,
                            deps: vec![],
                            children: vec![],
                        }],
                    },
                ],
                external_deps: vec![],
            })
            .await
            .unwrap();
        assert_eq!(r.nodes_created, 3); // util, core, engine
        assert_eq!(r.deps_added, 1); // core -> util
        let g = crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap();
        assert_eq!(g.len(), 4); // root + 3
        let core = g.find_by_name("core").unwrap();
        let util = g.find_by_name("util").unwrap();
        assert_eq!(core.deps, vec![util.id]);
        assert!(core.crate_boundary);
        // engine is a child of core
        let engine = g.find_by_name("engine").unwrap();
        assert_eq!(engine.parent, Some(core.id));
    }

    #[tokio::test]
    async fn submit_architecture_rejects_duplicate_names() {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            Stage::Architect,
            Role::Writer,
            workdir,
            Layout::SingleCrate,
            300, 500, 64, 5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitArchitectureTool { ctx };
        let err = tool
            .call(SubmitArchitectureArgs {
                children: vec![
                    ArchNode {
                        name: "x".into(),
                        description: "first".into(),
                        crate_boundary: false,
                        deps: vec![],
                        children: vec![ArchNode {
                            name: "x".into(),
                            description: "duplicate".into(),
                            crate_boundary: false,
                            deps: vec![],
                            children: vec![],
                        }],
                    },
                ],
                external_deps: vec![],
            })
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("globally unique"), "got: {msg}");
    }

    #[tokio::test]
    async fn submit_architecture_rejects_unknown_dep() {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            Stage::Architect,
            Role::Writer,
            workdir,
            Layout::SingleCrate,
            300, 500, 64, 5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitArchitectureTool { ctx };
        let err = tool
            .call(SubmitArchitectureArgs {
                children: vec![ArchNode {
                    name: "a".into(),
                    description: "deps on missing".into(),
                    crate_boundary: false,
                    deps: vec!["nonexistent".into()],
                    children: vec![],
                }],
                external_deps: vec![],
            })
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown node"), "got: {msg}");
    }

    #[tokio::test]
    async fn submit_public_validates_and_persists() {
        let (tmp, root, ctx) = fixture(Stage::Iface);
        let tool = SubmitPublicTool { ctx };
        let r = tool
            .call(SubmitRustArgs {
                content: "pub trait T { fn f(&self); }\n".into(),
            })
            .await
            .unwrap();
        assert!(!r.no_change);
        assert!(crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap().get(root).unwrap().public_rs.is_some());
    }

    #[tokio::test]
    async fn submit_public_rejects_impl_block() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitPublicTool { ctx };
        let err = tool
            .call(SubmitRustArgs {
                content: "pub struct X; impl X { pub fn n() -> Self { X } }".into(),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::Validate(_)));
    }

    #[tokio::test]
    async fn submit_public_rejects_private_smuggle() {
        // `pub use super::private::Inner` defines the real type in private
        // and re-exports it as the public surface. The validator rejects
        // it so the smuggle pattern can never reach the cargo gate.
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitPublicTool { ctx };
        let err = tool
            .call(SubmitRustArgs {
                content: "pub use super::private::Inner;\n".into(),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::Validate(_)));
    }

    #[tokio::test]
    async fn submit_public_idempotent_no_change() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitPublicTool { ctx };
        let body = "pub trait T { fn f(&self); }\n";
        let _ = tool
            .call(SubmitRustArgs { content: body.into() })
            .await
            .unwrap();
        let r2 = tool
            .call(SubmitRustArgs { content: body.into() })
            .await
            .unwrap();
        assert!(r2.no_change);
    }

    // (Tests for spec-stage decomposition removed — `submit_spec` no
    // longer accepts `children`. Decomposition is now exclusively the
    // architect stage's job; see the architect tests above.)

    #[tokio::test]
    async fn submit_verdict_records_into_ctx() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitVerdictTool { ctx: ctx.clone() };
        let _ = tool
            .call(SubmitVerdictArgs {
                satisfactory: false,
                reason: "missing thing".into(),
            })
            .await
            .unwrap();
        let v = ctx.take_verdict().unwrap();
        assert!(!v.satisfactory);
        assert_eq!(v.reason, "missing thing");
    }

    #[tokio::test]
    async fn loop_detection_triggers_after_three_identical_calls() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let body = "pub trait T { fn f(&self); }\n";
        let tool = SubmitPublicTool { ctx };
        let _ = tool
            .call(SubmitRustArgs { content: body.into() })
            .await
            .unwrap();
        let _ = tool
            .call(SubmitRustArgs { content: body.into() })
            .await
            .unwrap();
        let r = tool
            .call(SubmitRustArgs { content: body.into() })
            .await
            .unwrap_err();
        assert!(matches!(r, ToolFailure::Loop { .. }));
    }

    #[test]
    fn unified_tool_list_is_byte_identical_across_stages_and_roles() {
        // The whole point of unification: every non-quickfix call sees
        // the same tool catalog (so the API tool-schemas portion of the
        // request is byte-stable for prompt caching). Per-stage gating
        // happens at runtime via `require_stage` checks inside each
        // tool's `call()`.
        let baseline = tool_names_for(Stage::Spec, Role::Writer);
        for stage in Stage::ALL {
            for role in [Role::Writer, Role::Critic, Role::Reviser, Role::Judge] {
                assert_eq!(
                    tool_names_for(stage, role),
                    baseline,
                    "tool list at ({stage}, {role:?}) must equal the spec/writer baseline",
                );
            }
        }
        // Quickfix is the exception — different mode, different tools.
        assert_ne!(tool_names_for(Stage::Iface, Role::QuickFixer), baseline);
        // The composite tool replaces the separate public/private/decompose
        // trio — those legacy names must NOT appear anywhere.
        assert!(!baseline.contains(&"submit_spec_public"));
        assert!(!baseline.contains(&"submit_spec_private"));
        assert!(!baseline.contains(&"decompose"));
    }

    #[test]
    fn tools_accepted_at_filters_to_per_stage_role_subset() {
        // The per-(stage, role) "accepted" list is what the prompt's
        // Tools-eligibility block tells the model to call. It's a strict
        // subset of the unified catalog.
        let unified = unified_tool_names();
        for stage in Stage::ALL {
            for role in [Role::Writer, Role::Critic, Role::Reviser, Role::Judge] {
                let accepted = tools_accepted_at(stage, role);
                for name in &accepted {
                    assert!(
                        unified.contains(name),
                        "accepted tool {name} at ({stage}, {role:?}) not in unified catalog"
                    );
                }
            }
        }
        // Spot-checks of the per-(stage, role) accepted sets.
        assert!(tools_accepted_at(Stage::Spec, Role::Writer).contains(&"submit_spec"));
        assert!(tools_accepted_at(Stage::Iface, Role::Writer).contains(&"submit_public"));
        assert!(tools_accepted_at(Stage::Tests, Role::Writer).contains(&"submit_tests"));
        assert!(tools_accepted_at(Stage::Impl, Role::Writer).contains(&"submit_private"));
        assert!(tools_accepted_at(Stage::Impl, Role::Judge).contains(&"submit_verdict"));
        // Critic doesn't get submit_verdict; judge doesn't get submit_critique.
        assert!(!tools_accepted_at(Stage::Impl, Role::Critic).contains(&"submit_verdict"));
        assert!(!tools_accepted_at(Stage::Impl, Role::Judge).contains(&"submit_critique"));
        // Every critic in a content-producing stage gets submit_critique.
        for stage in [
            Stage::Spec,
            Stage::Iface,
            Stage::Tests,
            Stage::Impl,
            Stage::Debug,
        ] {
            assert!(
                tools_accepted_at(stage, Role::Critic).contains(&"submit_critique"),
                "stage {stage} critic should accept submit_critique"
            );
        }
        // read_file is universal — present at every (stage, role).
        for stage in Stage::ALL {
            for role in [Role::Writer, Role::Critic, Role::Reviser, Role::Judge] {
                assert!(
                    tools_accepted_at(stage, role).contains(&"read_file"),
                    "read_file should be universally accepted: missing at ({stage}, {role:?})"
                );
            }
        }
    }

    #[test]
    fn every_registered_tool_has_a_real_description() {
        let limits = PromptLimits {
            max_file_lines: 300,
            max_spec_section_lines: 400,
        };
        for stage in Stage::ALL {
            for role in [Role::Writer, Role::Critic, Role::Reviser, Role::Judge] {
                for name in tool_names_for(stage, role) {
                    let d = tool_description(name, limits);
                    assert_ne!(
                        d, "(no description registered)",
                        "tool '{name}' has no description registered (stage={stage}, role={role:?})"
                    );
                    assert!(d.len() > 30, "tool '{name}' description is suspiciously short");
                }
            }
        }
    }

    #[test]
    fn tool_descriptions_interpolate_limits_from_config() {
        let limits = PromptLimits {
            max_file_lines: 1234,
            max_spec_section_lines: 5678,
        };
        let pub_d = tool_description("submit_public", limits);
        assert!(pub_d.contains("1234"), "submit_public should mention max_file_lines: {pub_d}");
        let spec_d = tool_description("submit_spec", limits);
        assert!(spec_d.contains("5678"), "submit_spec should mention max_spec_section_lines: {spec_d}");
        // cargo_check is size-independent — should not contain a stale hardcoded number.
        let chk_d = tool_description("cargo_check", limits);
        assert!(!chk_d.contains("1234") && !chk_d.contains("5678"));
    }

    #[test]
    fn tool_definitions_for_returns_name_description_pairs() {
        let limits = PromptLimits {
            max_file_lines: 300,
            max_spec_section_lines: 400,
        };
        let defs = tool_definitions_for(Stage::Iface, Role::Writer, limits);
        assert!(!defs.is_empty());
        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"submit_public"));
        assert!(names.contains(&"submit_private"));
        let pub_def = defs.iter().find(|d| d.name == "submit_public").unwrap();
        // The description should explicitly forbid `mod` so the model knows.
        assert!(
            pub_def.description.contains("FORBIDDEN") && pub_def.description.contains("mod"),
            "submit_public description should explicitly forbid `mod`: {}",
            pub_def.description
        );
    }

    #[test]
    fn tool_definitions_serialize_via_transcript_kind() {
        // The whole point of the new variant: roundtrip through serde so the
        // UI sees the tool list.
        let kind = TranscriptKind::ToolDefinitions {
            tools: vec![ToolDefSnapshot {
                name: "submit_public".into(),
                description: "...".into(),
            }],
        };
        let s = serde_json::to_string(&kind).unwrap();
        assert!(s.contains("\"type\":\"tool_definitions\""));
        let _back: TranscriptKind = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn transcript_entry_role_round_trips_via_serde() {
        // The "all entries look like ACTOR" UI bug was caused by role
        // living only on the SSE event, not on the entry — so initial
        // /api/state load lost it. Pin that role survives serialization.
        let e = TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::AssistantText,
            content: "hi".into(),
            role: Some(Role::Critic),
        };
        let s = serde_json::to_string(&e).unwrap();
        let back: TranscriptEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(back.role, Some(Role::Critic));
        // Backward compat: an entry serialized without `role` deserializes
        // with role = None (existing checkpoints don't lose data).
        let legacy = r#"{"timestamp":"2026-05-04T00:00:00Z","kind":{"type":"system"},"content":"x"}"#;
        let parsed: TranscriptEntry = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.role, None);
    }

    #[test]
    fn is_model_classifies_model_vs_bureau_correctly() {
        let cases = [
            (TranscriptKind::System, false),
            (TranscriptKind::UserPrompt, false),
            (TranscriptKind::ToolDefinitions { tools: vec![] }, false),
            (
                TranscriptKind::ToolResult {
                    tool: "x".into(),
                    ok: true,
                    error: None,
                    output: None,
                },
                false,
            ),
            (TranscriptKind::Note, false),
            (TranscriptKind::Error, false),
            (TranscriptKind::AssistantText, true),
            (TranscriptKind::ToolCall { tool: "x".into() }, true),
        ];
        for (kind, expected) in cases {
            let e = TranscriptEntry {
                timestamp: Utc::now(),
                kind,
                content: String::new(),
                role: None,
            };
            assert_eq!(e.is_model(), expected, "wrong is_model for {:?}", e.kind);
        }
    }

    // ---- New tool tests: read_file / write_file / write_file_range / apply_patch ----

    #[tokio::test]
    async fn write_file_replaces_managed_slot_and_renders() {
        let (tmp, root, ctx) = fixture(Stage::Iface);
        let tool = WriteFileTool { ctx };
        // public.rs lives at src/public.rs for the root node in single-crate mode.
        let r = tool
            .call(WriteFileArgs {
                path: "src/public.rs".into(),
                content: "pub trait Foo {}\n".into(),
            })
            .await
            .unwrap();
        assert!(!r.no_change);
        assert_eq!(
            crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap().get(root).unwrap().public_rs.as_deref(),
            Some("pub trait Foo {}\n")
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("src/public.rs")).unwrap();
        assert!(on_disk.contains("pub trait Foo"));
    }

    #[tokio::test]
    async fn write_file_no_change_when_content_matches() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = WriteFileTool { ctx };
        tool.call(WriteFileArgs {
            path: "src/public.rs".into(),
            content: "pub trait Foo {}\n".into(),
        })
        .await
        .unwrap();
        let r = tool
            .call(WriteFileArgs {
                path: "src/public.rs".into(),
                content: "pub trait Foo {}\n".into(),
            })
            .await
            .unwrap();
        assert!(r.no_change);
    }

    #[tokio::test]
    async fn write_file_rejects_unmanaged_path() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = WriteFileTool { ctx };
        let err = tool
            .call(WriteFileArgs {
                path: "Cargo.toml".into(),
                content: "[package]\n".into(),
            })
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not a managed slot"), "got: {msg}");
    }

    #[tokio::test]
    async fn write_file_rejects_other_nodes_slots() {
        // Add a sibling child whose public.rs belongs to it, not root.
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let mut g = crate::graph::load(&ctx.workdir, ctx.layout).unwrap();
        let _child = g
            .add_child(ctx.node_id, Node::new("helper", "helper"))
            .unwrap();
        crate::render::render_graph(&ctx.workdir, &g, Layout::SingleCrate).unwrap();
        let tool = WriteFileTool { ctx };
        let err = tool
            .call(WriteFileArgs {
                path: "src/helper/public.rs".into(),
                content: "pub trait X {}\n".into(),
            })
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("another node"), "got: {msg}");
    }

    #[tokio::test]
    async fn read_file_slices_by_line_range() {
        let (tmp, _root, ctx) = fixture(Stage::Iface);
        std::fs::write(
            tmp.path().join("src/public.rs"),
            "line1\nline2\nline3\nline4\nline5\n",
        )
        .unwrap();
        let tool = ReadFileTool { ctx };
        let r = tool
            .call(ReadFileArgs {
                path: "src/public.rs".into(),
                start_line: Some(2),
                end_line: Some(4),
            })
            .await
            .unwrap();
        assert_eq!(r.content, "line2\nline3\nline4");
        assert_eq!(r.start_line, 2);
        assert_eq!(r.end_line, 4);
        assert_eq!(r.total_lines, 5);
    }

    #[tokio::test]
    async fn read_file_rejects_parent_dir_traversal() {
        let (_tmp, _root, ctx) = fixture(Stage::Iface);
        let tool = ReadFileTool { ctx };
        let err = tool
            .call(ReadFileArgs {
                path: "../escape".into(),
                start_line: None,
                end_line: None,
            })
            .await
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("parent-dir"), "got: {msg}");
    }

    #[tokio::test]
    async fn write_file_range_inserts_between_existing_lines() {
        let (tmp, root, ctx) = fixture(Stage::Iface);
        // Seed public.rs with three lines.
        let mut g = crate::graph::load(&ctx.workdir, ctx.layout).unwrap();
        g.get_mut(root).unwrap().public_rs =
            Some("pub trait A {}\npub trait B {}\npub trait C {}\n".into());
        render::render_graph(&ctx.workdir, &g, Layout::SingleCrate).unwrap();
        let tool = WriteFileRangeTool { ctx };
        // Replace line 2 ("pub trait B {}") with two new lines.
        tool.call(WriteFileRangeArgs {
            path: "src/public.rs".into(),
            start_line: 2,
            end_line: 2,
            content: "pub trait B {}\npub trait B_inserted {}\n".into(),
        })
        .await
        .unwrap();
        let final_content = crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap().get(root).unwrap().public_rs.clone().unwrap();
        assert_eq!(
            final_content,
            "pub trait A {}\npub trait B {}\npub trait B_inserted {}\npub trait C {}\n"
        );
    }

    #[tokio::test]
    async fn apply_patch_applies_unified_diff() {
        let (tmp, root, ctx) = fixture(Stage::Iface);
        let mut g = crate::graph::load(&ctx.workdir, ctx.layout).unwrap();
        g.get_mut(root).unwrap().public_rs = Some("pub trait Old {}\n".into());
        render::render_graph(&ctx.workdir, &g, Layout::SingleCrate).unwrap();
        let patch = "--- a/src/public.rs\n+++ b/src/public.rs\n@@ -1 +1 @@\n-pub trait Old {}\n+pub trait New {}\n";
        let tool = ApplyPatchTool { ctx };
        let r = tool
            .call(ApplyPatchArgs {
                patch: patch.into(),
            })
            .await
            .unwrap();
        assert_eq!(r.files_changed, vec!["src/public.rs".to_string()]);
        let after = crate::graph::load(tmp.path(), Layout::SingleCrate).unwrap().get(root).unwrap().public_rs.clone().unwrap();
        assert!(after.contains("New"));
        assert!(!after.contains("Old"));
    }

    #[tokio::test]
    async fn critique_render_empty_says_no_issues() {
        let c = Critique { issues: vec![] };
        assert!(c.is_clean());
        assert!(c.render().contains("no issues"));
    }

    #[tokio::test]
    async fn critique_render_lists_issues_with_severity() {
        let c = Critique {
            issues: vec![
                CritiqueIssue {
                    description: "missing edge case".into(),
                    location: Some("src/foo.rs:42".into()),
                    severity: Some("error".into()),
                },
                CritiqueIssue {
                    description: "typo".into(),
                    location: None,
                    severity: None,
                },
            ],
        };
        assert!(!c.is_clean());
        let r = c.render();
        assert!(r.contains("missing edge case"));
        assert!(r.contains("src/foo.rs:42"));
        assert!(r.contains("[error]"));
        assert!(r.contains("[warning]"));
    }

    #[tokio::test]
    async fn submit_critique_sets_ctx_critique() {
        let (_tmp, _root, ctx) = fixture(Stage::Spec);
        let tool = SubmitCritiqueTool { ctx: ctx.clone() };
        tool.call(SubmitCritiqueArgs {
            issues: vec![CritiqueIssue {
                description: "vague".into(),
                location: None,
                severity: None,
            }],
        })
        .await
        .unwrap();
        let stored = ctx.take_critique();
        assert!(stored.is_some());
        assert_eq!(stored.unwrap().issues.len(), 1);
    }
}
