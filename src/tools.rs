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
//! All other tools from the old harness (`read_file`, `write_file`,
//! `list_files`, `replace_fn_body`, `list_compiler_errors`, `emit_subtasks`,
//! the spec-section trio) are gone. Reads are pre-loaded into the prompt
//! by `node_context`; writes go through these slot-fillers.

use crate::graph::{Node, NodeGraph, NodeId, Stage};
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
    /// Live shared graph (per-task tools mutate this directly via the
    /// orchestrator's lock; we go through the shared Arc to avoid copying
    /// the whole graph each call).
    pub graph: Arc<Mutex<NodeGraph>>,
    /// Workdir on disk; we re-render after each successful submit.
    pub workdir: PathBuf,
    pub layout: Layout,
    pub max_file_lines: usize,
    pub max_spec_section_lines: usize,
    /// Hard cap on the total number of nodes the graph may hold. The
    /// decompose tool refuses to exceed it.
    pub max_nodes: usize,
    /// Hard cap on the depth of the node tree (root is depth 0).
    pub max_node_depth: usize,
    /// Loop detection — same args three times in a row triggers an error.
    pub recent_calls: Mutex<VecDeque<(String, u64)>>,
    /// Filled by `submit_verdict` (judge stage only).
    pub verdict: Mutex<Option<JudgeVerdict>>,
    /// Transcript callback for recording tool calls / results.
    pub transcript: Mutex<Vec<TranscriptEntry>>,
    /// File changes queued for the orchestrator to broadcast over SSE.
    pub fs_events: Mutex<Vec<PathBuf>>,
    /// Held for the duration of any cargo invocation so parallel tasks
    /// don't trample each other's `target/` dir / lock files.
    pub cargo_lock: Arc<tokio::sync::Mutex<()>>,
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
    /// Who is "speaking" in this entry — the bureau (engine, framework,
    /// tool result) or the model. Useful for transcript UI.
    pub fn speaker(&self) -> Speaker {
        match &self.kind {
            TranscriptKind::AssistantText | TranscriptKind::ToolCall { .. } => Speaker::Model,
            _ => Speaker::Bureau,
        }
    }
}

/// Which side of the actor/framework boundary an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    /// The framework — system prompt, user prompt, tool definitions, tool
    /// results, notes, errors.
    Bureau,
    /// The LLM — its assistant text and its tool calls.
    Model,
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
        graph: Arc<Mutex<NodeGraph>>,
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
            graph,
            workdir,
            layout,
            max_file_lines,
            max_spec_section_lines,
            max_nodes,
            max_node_depth,
            recent_calls: Mutex::new(VecDeque::new()),
            verdict: Mutex::new(None),
            transcript: Mutex::new(Vec::new()),
            fs_events: Mutex::new(Vec::new()),
            cargo_lock,
        }
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
        self.transcript.lock().push(entry);
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        s.hash(&mut hasher);
        let h = hasher.finish();
        let mut recent = self.recent_calls.lock();
        recent.push_back((name.to_string(), h));
        while recent.len() > LOOP_WINDOW {
            recent.pop_front();
        }
        let consecutive = recent
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
        self.transcript.lock().push(entry);
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

    fn render_after_write(&self) -> Result<(), ToolFailure> {
        let graph = self.graph.lock();
        let report = render::render_graph(&self.workdir, &graph, self.layout)
            .map_err(|e| ToolFailure::Other(format!("re-render failed: {e}")))?;
        let mut events = self.fs_events.lock();
        events.extend(report.files_written.into_iter());
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
    /// New child nodes to create under THIS node. Omit (or empty) for
    /// a leaf. The schema hides this field entirely when the framework
    /// has no room for more children.
    #[serde(default)]
    pub children: Vec<ChildDecl>,
    /// Existing node names that THIS node should depend on. Adds dep
    /// edges from the current node; does not create nodes.
    #[serde(default)]
    pub deps: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct SubmitSpecOk {
    pub public_bytes: u64,
    pub public_lines: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_bytes: Option<u64>,
    pub children_created: Vec<NodeIdRef>,
    pub deps_added: Vec<NodeIdRef>,
}

pub struct SubmitSpecTool {
    pub ctx: Arc<TaskCtx>,
}

impl SubmitSpecTool {
    /// Whether this stage may decompose right now. False once the graph
    /// has hit the node-count cap or this node is at the depth cap.
    fn decomposition_allowed(&self) -> bool {
        let g = self.ctx.graph.lock();
        if g.len() >= self.ctx.max_nodes {
            return false;
        }
        let depth = g.ancestors(self.ctx.node_id, true).len().saturating_sub(1);
        if depth + 1 > self.ctx.max_node_depth {
            return false;
        }
        true
    }
}

impl Tool for SubmitSpecTool {
    const NAME: &'static str = "submit_spec";
    type Error = ToolFailure;
    type Args = SubmitSpecArgs;
    type Output = SubmitSpecOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        // Build the schema with `children` only when decomposition is
        // allowed. Model can't add children it can't see in the schema.
        let mut props = serde_json::Map::new();
        props.insert("public".into(), json!({
            "type": "string",
            "description": format!(
                "Public spec markdown — REQUIRED. Cap {} lines. Audience: \
                 dependents and downstream stages.",
                self.ctx.max_spec_section_lines
            ),
        }));
        props.insert("private".into(), json!({
            "type": "string",
            "description": format!(
                "Private spec markdown — OPTIONAL. Cap {} lines. Audience: \
                 only this node's own iface/impl writers. Implementation \
                 notes, design rationale, alternatives considered.",
                self.ctx.max_spec_section_lines
            ),
        }));
        if self.decomposition_allowed() {
            props.insert("children".into(), json!({
                "type": "array",
                "description": "OPTIONAL. New child nodes to create under THIS \
                                node. Omit or [] for a leaf. Most nodes are leaves.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "snake_case Rust ident"},
                        "description": {"type": "string"},
                        "deps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names this child depends on (existing nodes or earlier siblings in this call)."
                        },
                        "crate_boundary": {
                            "type": "boolean",
                            "default": false,
                            "description": "true ONLY at major subsystem boundaries that need their own Cargo crate. Default false."
                        }
                    },
                    "required": ["name", "description"]
                }
            }));
        }
        props.insert("deps".into(), json!({
            "type": "array",
            "items": {"type": "string"},
            "description": "OPTIONAL. Names of existing graph nodes that THIS node should depend on."
        }));
        // Description: switch the children-related blurb based on whether
        // the schema even includes the field. Don't tell the model about
        // a knob it doesn't have.
        let limits = self.ctx.prompt_limits();
        let description = if self.decomposition_allowed() {
            tool_description(Self::NAME, limits)
        } else {
            submit_spec_description_leaf_only(limits)
        };
        ToolDefinition {
            name: Self::NAME.to_string(),
            description,
            parameters: json!({
                "type": "object",
                "properties": props,
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

/// `submit_spec` description used when the decomposition cap is exhausted
/// — the schema for this turn does NOT include `children`, so the prose
/// shouldn't mention it as an option either. The model is told plainly
/// that this node will be a leaf.
fn submit_spec_description_leaf_only(limits: PromptLimits) -> String {
    let max_spec = limits.max_spec_section_lines;
    let max_file = limits.max_file_lines;
    format!(
        "Submit THIS node's spec — the spec stage's whole writer output in ONE call. \
        The decomposition cap is exhausted for this turn (either the node-count cap \
        or the depth cap), so this node MUST be a LEAF — no children. The schema \
        accordingly does not offer a `children` field.\n\
        \n\
        Fields:\n\
        - `public` (REQUIRED, ≤{max_spec} lines) — markdown describing what this \
          node DOES, exposes, and guarantees. Audience: dependents and downstream \
          stages. Suggested sections: `## Goal`, `## API`, `## Invariants`, \
          `## Out of scope`.\n\
        - `private` (optional, ≤{max_spec} lines) — implementation notes / \
          rationale for this node's own future iface/impl writers. Skip if there's \
          nothing worth recording.\n\
        - `deps` (optional) — names of existing graph nodes that THIS node should \
          depend on. Cycle-checked at submit time.\n\
        \n\
        (Per-file cap for code is {max_file} lines — describe an API the iface \
        stage can fit in that.)"
    )
}

/// Apply the composite `submit_spec` submission. Validates fully before
/// mutating so a failure mid-way doesn't leave the graph half-changed.
fn submit_spec_apply(
    ctx: &Arc<TaskCtx>,
    args: SubmitSpecArgs,
) -> Result<SubmitSpecOk, ToolFailure> {
    ctx.require_stage(SubmitSpecTool::NAME, &[Stage::Spec])?;

    // ---- 1. Validate sizes (public required, private optional) ----
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
             this node does".into(),
        ));
    }
    let priv_lines = args.private.as_deref().map(|s| s.lines().count());
    if let Some(pl) = priv_lines {
        if pl > ctx.max_spec_section_lines {
            return Err(ToolFailure::FileTooLarge(pl, ctx.max_spec_section_lines));
        }
    }

    let mut g = ctx.graph.lock();
    let parent_id = ctx.node_id;
    let parent_name = g
        .get(parent_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    // ---- 2. Validate decomposition cap (only matters if children present) ----
    let cur_nodes = g.len();
    let new_count = args.children.len();
    let remaining = ctx.max_nodes.saturating_sub(cur_nodes);
    if new_count > remaining {
        let msg = if remaining == 0 {
            format!(
                "ABANDON CHILDREN — DO NOT RETRY with children. The node-count \
                 cap ({}) is fully exhausted; this graph cannot accept ANY new \
                 children. Resubmit `submit_spec` WITHOUT the `children` field, \
                 keeping `public` and the rest. Treat this node as a leaf.",
                ctx.max_nodes
            )
        } else {
            format!(
                "submit_spec rejected — you asked for {new_count} children but \
                 only {remaining} slot(s) remain (cap {}). Either resubmit with \
                 AT MOST {remaining} children (drop the less-essential ones), \
                 OR resubmit without `children` and treat this node as a leaf.",
                ctx.max_nodes
            )
        };
        return Err(ToolFailure::Subtask(msg));
    }
    let parent_depth = g.ancestors(parent_id, true).len().saturating_sub(1);
    let child_depth = parent_depth + 1;
    if !args.children.is_empty() && child_depth > ctx.max_node_depth {
        return Err(ToolFailure::Subtask(format!(
            "ABANDON CHILDREN — DO NOT RETRY with children. Children of this \
             node would land at depth {child_depth}, past the depth cap of {}. \
             Resubmit `submit_spec` WITHOUT `children` — this node must be a \
             leaf.",
            ctx.max_node_depth
        )));
    }

    // ---- 3. Validate `deps` (current-node-to-existing edges) ----
    let mut deps_resolved: Vec<NodeId> = Vec::new();
    for name in &args.deps {
        if name == &parent_name {
            return Err(ToolFailure::Subtask(format!(
                "deps: '{name}' is THIS node — a node cannot depend on itself"
            )));
        }
        let id = g.find_by_name(name).map(|n| n.id).ok_or_else(|| {
            ToolFailure::Subtask(format!("deps: no existing node named '{name}'"))
        })?;
        deps_resolved.push(id);
    }

    // ---- 4. Validate children + classify their per-child deps ----
    #[derive(Clone)]
    enum DepRef {
        Existing(NodeId),
        Sibling(String),
    }
    let new_names: HashSet<&str> = args.children.iter().map(|c| c.name.as_str()).collect();
    if new_names.len() != args.children.len() {
        return Err(ToolFailure::Subtask(
            "duplicate child names in this submit_spec call".into(),
        ));
    }
    let mut planned: Vec<(ChildDecl, Vec<DepRef>)> = Vec::new();
    let mut created_so_far: HashSet<String> = HashSet::new();
    for child in &args.children {
        if !is_valid_ident(&child.name) {
            return Err(ToolFailure::Subtask(format!(
                "child name '{}' is not a valid Rust identifier",
                child.name
            )));
        }
        let mut deps: Vec<DepRef> = Vec::new();
        for dep_name in &child.deps {
            if dep_name == &child.name {
                return Err(ToolFailure::Subtask(format!(
                    "child '{name}' lists itself in its own `deps` — that's a \
                     self-loop. To declare an edge FROM the current parent node \
                     TO this child or any other existing node, use the top-level \
                     `deps` field of submit_spec instead.",
                    name = child.name
                )));
            }
            if created_so_far.contains(dep_name) {
                deps.push(DepRef::Sibling(dep_name.clone()));
            } else if new_names.contains(dep_name.as_str()) {
                return Err(ToolFailure::Subtask(format!(
                    "child '{}' references later sibling '{}'; reorder so the \
                     dep comes first",
                    child.name, dep_name
                )));
            } else {
                let id = g.find_by_name(dep_name).map(|n| n.id).ok_or_else(|| {
                    ToolFailure::Subtask(format!(
                        "child '{}': no existing node named '{}'",
                        child.name, dep_name
                    ))
                })?;
                deps.push(DepRef::Existing(id));
            }
        }
        planned.push((child.clone(), deps));
        created_so_far.insert(child.name.clone());
    }

    // ---- 5. Apply (validation passed; mutate) ----
    let public_bytes = args.public.len() as u64;
    let private_bytes = args.private.as_ref().map(|p| p.len() as u64);
    {
        let n = g.get_mut(parent_id).ok_or_else(|| {
            ToolFailure::Other(format!("node {} missing", parent_id))
        })?;
        n.spec_public_md = Some(args.public);
        if let Some(p) = args.private {
            n.spec_private_md = Some(p);
        }
        n.updated_at = Utc::now();
    }
    let mut deps_added = Vec::new();
    for to in deps_resolved {
        g.add_dep(parent_id, to)?;
        let n = g.get(to).unwrap();
        deps_added.push(NodeIdRef {
            id: n.id.to_string(),
            name: n.name.clone(),
        });
    }
    let mut name_to_id: std::collections::HashMap<String, NodeId> =
        std::collections::HashMap::new();
    let mut children_created = Vec::new();
    for (child, dep_refs) in planned {
        let mut node = Node::new(&child.name, &child.description);
        node.crate_boundary = child.crate_boundary;
        let new_id = g.add_child(parent_id, node)?;
        name_to_id.insert(child.name.clone(), new_id);
        for dep_ref in dep_refs {
            let dep_id = match dep_ref {
                DepRef::Existing(id) => id,
                DepRef::Sibling(name) => *name_to_id.get(&name).ok_or_else(|| {
                    ToolFailure::Subtask(format!(
                        "internal: sibling '{name}' should have been created already"
                    ))
                })?,
            };
            g.add_dep(new_id, dep_id)?;
        }
        children_created.push(NodeIdRef {
            id: new_id.to_string(),
            name: child.name,
        });
    }

    drop(g);
    ctx.render_after_write()?;
    Ok(SubmitSpecOk {
        public_bytes,
        public_lines,
        private_bytes,
        children_created,
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

pub struct SubmitPublicTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitPublicTool {
    const NAME: &'static str = "submit_public";
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
        let r: Result<SubmitRustOk, ToolFailure> = (|| {
            self.ctx.require_stage(Self::NAME, &[Stage::Iface])?;
            let lines = args.content.lines().count();
            if lines > self.ctx.max_file_lines {
                return Err(ToolFailure::FileTooLarge(lines, self.ctx.max_file_lines));
            }
            // Validation: enforce the public.rs constraint set.
            node_validate::validate_public(&args.content)?;
            let mut no_change = false;
            {
                let mut g = self.ctx.graph.lock();
                let n = g.get_mut(self.ctx.node_id).ok_or_else(|| {
                    ToolFailure::Other(format!("node {} missing", self.ctx.node_id))
                })?;
                if n.public_rs.as_deref() == Some(args.content.as_str()) {
                    no_change = true;
                } else {
                    n.public_rs = Some(args.content.clone());
                    n.updated_at = Utc::now();
                }
            }
            if !no_change {
                self.ctx.render_after_write()?;
            }
            Ok(SubmitRustOk {
                bytes: args.content.len() as u64,
                lines,
                no_change,
            })
        })();
        self.ctx.finish(Self::NAME, r)
    }
}

pub struct SubmitPrivateTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitPrivateTool {
    const NAME: &'static str = "submit_private";
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
        let r: Result<SubmitRustOk, ToolFailure> = (|| {
            self.ctx
                .require_stage(Self::NAME, &[Stage::Iface, Stage::Impl, Stage::Debug])?;
            let lines = args.content.lines().count();
            if lines > self.ctx.max_file_lines {
                return Err(ToolFailure::FileTooLarge(lines, self.ctx.max_file_lines));
            }
            // Need a snapshot of the graph for validation.
            let validated = {
                let g = self.ctx.graph.lock();
                let n = g
                    .get(self.ctx.node_id)
                    .ok_or_else(|| {
                        ToolFailure::Other(format!("node {} missing", self.ctx.node_id))
                    })?
                    .clone();
                node_validate::validate_private(&args.content, &n, &g)?;
                n
            };
            let _ = validated;
            let mut no_change = false;
            {
                let mut g = self.ctx.graph.lock();
                let n = g.get_mut(self.ctx.node_id).unwrap();
                if n.private_rs.as_deref() == Some(args.content.as_str()) {
                    no_change = true;
                } else {
                    n.private_rs = Some(args.content.clone());
                    n.updated_at = Utc::now();
                }
            }
            if !no_change {
                self.ctx.render_after_write()?;
            }
            Ok(SubmitRustOk {
                bytes: args.content.len() as u64,
                lines,
                no_change,
            })
        })();
        self.ctx.finish(Self::NAME, r)
    }
}

pub struct SubmitTestsTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for SubmitTestsTool {
    const NAME: &'static str = "submit_tests";
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
        let r: Result<SubmitRustOk, ToolFailure> = (|| {
            self.ctx
                .require_stage(Self::NAME, &[Stage::Tests, Stage::Debug])?;
            let lines = args.content.lines().count();
            if lines > self.ctx.max_file_lines {
                return Err(ToolFailure::FileTooLarge(lines, self.ctx.max_file_lines));
            }
            // Tests can use anything private would; same validator.
            {
                let g = self.ctx.graph.lock();
                let n = g.get(self.ctx.node_id).ok_or_else(|| {
                    ToolFailure::Other(format!("node {} missing", self.ctx.node_id))
                })?;
                node_validate::validate_private(&args.content, n, &g)?;
            }
            let mut no_change = false;
            {
                let mut g = self.ctx.graph.lock();
                let n = g.get_mut(self.ctx.node_id).unwrap();
                if n.tests_rs.as_deref() == Some(args.content.as_str()) {
                    no_change = true;
                } else {
                    n.tests_rs = Some(args.content.clone());
                    n.updated_at = Utc::now();
                }
            }
            if !no_change {
                self.ctx.render_after_write()?;
            }
            Ok(SubmitRustOk {
                bytes: args.content.len() as u64,
                lines,
                no_change,
            })
        })();
        self.ctx.finish(Self::NAME, r)
    }
}

// --------------------------------------------------------------------------
// Shared types used by submit_spec's child-creation field.
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ChildDecl {
    pub name: String,
    pub description: String,
    /// References to existing node names (or earlier siblings in the same
    /// `submit_spec` call) that this child will depend on.
    #[serde(default)]
    pub deps: Vec<String>,
    /// If true, this child is a separate Cargo crate (workspace mode only).
    #[serde(default)]
    pub crate_boundary: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct NodeIdRef {
    pub id: String,
    pub name: String,
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
            *self.ctx.verdict.lock() = Some(JudgeVerdict {
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
    if let Some(p) = package {
        args.push("-p".to_string());
        args.push(p.to_string());
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
    if let Some(p) = package {
        args.push("-p".to_string());
        args.push(p.to_string());
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
// Tool catalogs by stage and role.
// --------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Writer,
    Critic,
    Reviser,
    Judge,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Writer => "writer",
            Role::Critic => "critic",
            Role::Reviser => "reviser",
            Role::Judge => "judge",
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
        "submit_spec" => format!(
            "Submit THIS node's spec — the spec stage's whole writer output in ONE call. \
            Required: `public`. Optional: `private`, `children`, `deps`. Call once per \
            spec stage.\n\
            \n\
            The spec is a SPECIFICATION DOCUMENT describing the software, NOT a literate \
            Rust file. Specs talk about data shapes, ownership, concurrency, error model, \
            invariants, I/O surfaces — at the level of REQUIREMENTS and ARCHITECTURE. \
            The iface stage is what writes Rust traits and signatures; do not preempt \
            that here.\n\
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
              see this. NOT a changelog of your edits — describe the software's \
              internals, not the document's history.\n\
            - `children` (optional; schema only includes this field when the \
              decomposition cap allows) — list of child nodes to create under THIS \
              node. Most nodes should be LEAVES. Each child:\n\
                · `name` — snake_case Rust ident (not CamelCase — that's a type).\n\
                · `description` — one short sentence.\n\
                · `deps` — names this child depends on (existing graph nodes or \
                  earlier siblings in this same call).\n\
                · `crate_boundary` (default false) — set true ONLY at major top-level \
                  subsystem boundaries. Most children leave this false; they become \
                  modules within the parent's crate.\n\
            - `deps` (optional) — names of existing graph nodes that THIS node depends \
              on. Cycle-checked at submit time at BOTH the node and crate level.\n\
            \n\
            (Per-file cap for code is {max_file} lines, for context.)"
        ),

        "submit_public" => format!(
            "Author `public.rs` — the node's public API surface. \
            ALLOWED items: `pub trait Foo {{ fn bar(...) -> ...; }}` (signatures only, \
            NO method bodies, NO default impls); `pub struct/enum/type/const/static` \
            declarations; `use super::private::ConcreteType` re-aliases; doc comments. \
            FORBIDDEN: `mod` (the framework auto-generates the module scaffolding — do \
            not write your own `mod` blocks); `impl` blocks of any kind; `fn` outside \
            trait declarations; `extern crate`; macro invocations; `pub use crate::*` \
            cross-node re-exports. Hard cap: {max_file} lines."
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

pub fn tool_names_for(stage: Stage, role: Role) -> Vec<&'static str> {
    use Role::*;
    use Stage::*;
    match (stage, role) {
        (Spec, Writer) | (Spec, Reviser) => {
            vec![
                SubmitSpecTool::NAME,
            ]
        }
        (Spec, Critic) => vec![],
        (Spec, Judge) => vec![SubmitVerdictTool::NAME],

        (Iface, Writer) | (Iface, Reviser) => vec![
            SubmitPublicTool::NAME,
            SubmitPrivateTool::NAME, // initial scaffold; impl stage will refine
            CargoCheckTool::NAME,
        ],
        (Iface, Critic) => vec![CargoCheckTool::NAME],
        (Iface, Judge) => vec![CargoCheckTool::NAME, SubmitVerdictTool::NAME],

        (Tests, Writer) | (Tests, Reviser) => vec![
            SubmitTestsTool::NAME,
            CargoCheckTool::NAME,
            CargoTestNoRunTool::NAME,
        ],
        (Tests, Critic) => vec![CargoCheckTool::NAME, CargoTestNoRunTool::NAME],
        (Tests, Judge) => vec![
            CargoCheckTool::NAME,
            CargoTestNoRunTool::NAME,
            SubmitVerdictTool::NAME,
        ],

        (Impl, Writer) | (Impl, Reviser) => vec![
            SubmitPrivateTool::NAME,
            CargoCheckTool::NAME,
            CargoTestTool::NAME,
            CargoClippyTool::NAME,
        ],
        (Impl, Critic) => vec![CargoCheckTool::NAME, CargoTestTool::NAME, CargoClippyTool::NAME],
        (Impl, Judge) => vec![
            CargoCheckTool::NAME,
            CargoTestTool::NAME,
            SubmitVerdictTool::NAME,
        ],

        (Debug, Writer) | (Debug, Reviser) => vec![
            SubmitPrivateTool::NAME,
            SubmitTestsTool::NAME,
            CargoCheckTool::NAME,
            CargoTestTool::NAME,
            CargoClippyTool::NAME,
        ],
        (Debug, Critic) => vec![CargoCheckTool::NAME, CargoTestTool::NAME],
        (Debug, Judge) => vec![CargoTestTool::NAME, SubmitVerdictTool::NAME],

        (Opt, Writer) | (Opt, Reviser) => vec![
            SubmitPrivateTool::NAME,
            CargoTestTool::NAME,
            CargoClippyTool::NAME,
        ],
        (Opt, Critic) => vec![CargoTestTool::NAME, CargoClippyTool::NAME],
        (Opt, Judge) => vec![CargoTestTool::NAME, SubmitVerdictTool::NAME],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;
    use crate::render::Layout;

    fn fixture(stage: Stage) -> (tempfile::TempDir, Arc<Mutex<NodeGraph>>, NodeId, Arc<TaskCtx>) {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "the app")).unwrap();
        // Initial render so the workdir is set up enough for cargo (we
        // don't actually run cargo in unit tests).
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            stage,
            Role::Writer,
            graph.clone(),
            workdir,
            Layout::SingleCrate,
            300,
            500,
            64,
            5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        (tmp, graph, root, ctx)
    }

    #[tokio::test]
    async fn submit_spec_public_persists_to_node_and_disk() {
        let (tmp, graph, root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# Spec\n\nDoes the thing.".into(),
                private: None,
                children: vec![],
                deps: vec![],
            })
            .await
            .unwrap();
        assert!(r.public_lines >= 2);
        assert_eq!(
            graph.lock().get(root).unwrap().spec_public_md.as_deref(),
            Some("# Spec\n\nDoes the thing.")
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("spec/app/public.md")).unwrap();
        assert!(on_disk.contains("Does the thing"));
    }

    #[tokio::test]
    async fn submit_spec_private_writes_to_separate_slot_and_file() {
        let (tmp, graph, root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        tool.call(SubmitSpecArgs {
            public: "# Spec\n\nDoes the thing.".into(),
            private: Some("# Notes\n\nWhy I chose option B.".into()),
            children: vec![],
            deps: vec![],
        })
        .await
        .unwrap();
        assert_eq!(
            graph.lock().get(root).unwrap().spec_private_md.as_deref(),
            Some("# Notes\n\nWhy I chose option B.")
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("spec/app/private.md")).unwrap();
        assert!(on_disk.contains("option B"));
    }

    #[tokio::test]
    async fn submit_spec_rejected_outside_spec_stage() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitSpecTool { ctx };
        let err = tool
            .call(SubmitSpecArgs {
                public: "x".into(),
                private: None,
                children: vec![],
                deps: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::WrongStage { .. }));
    }

    #[tokio::test]
    async fn submit_spec_creates_children_in_one_call() {
        // The whole point of the composite tool: spec, private, children,
        // deps all in ONE roundtrip.
        let (_tmp, graph, _root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# umbrella\n\ncoordinates two children".into(),
                private: None,
                children: vec![
                    ChildDecl {
                        name: "alpha".into(),
                        description: "first".into(),
                        deps: vec![],
                        crate_boundary: false,
                    },
                    ChildDecl {
                        name: "beta".into(),
                        description: "second; depends on alpha".into(),
                        deps: vec!["alpha".into()],
                        crate_boundary: false,
                    },
                ],
                deps: vec![],
            })
            .await
            .unwrap();
        assert_eq!(r.children_created.len(), 2);
        let g = graph.lock();
        assert_eq!(g.len(), 3); // root + alpha + beta
        let beta = g.find_by_name("beta").unwrap();
        let alpha = g.find_by_name("alpha").unwrap();
        assert_eq!(beta.deps, vec![alpha.id]);
    }

    #[tokio::test]
    async fn submit_spec_schema_hides_children_when_at_cap() {
        // When the cap is exhausted, the schema must not contain the
        // `children` field — the model literally can't ask for one.
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            Stage::Spec,
            Role::Writer,
            graph.clone(),
            workdir,
            Layout::SingleCrate,
            300,
            500,
            1, // max_nodes — already at cap (1 root)
            5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitSpecTool { ctx };
        let def = tool.definition(String::new()).await;
        let props = def
            .parameters
            .get("properties")
            .and_then(|v| v.as_object())
            .unwrap();
        assert!(props.contains_key("public"), "public is required, must always appear");
        assert!(props.contains_key("private"), "private is always optional");
        assert!(props.contains_key("deps"), "deps doesn't add nodes, always allowed");
        assert!(
            !props.contains_key("children"),
            "children must be HIDDEN at cap — got: {props:?}"
        );
    }

    #[tokio::test]
    async fn submit_public_validates_and_persists() {
        let (_tmp, graph, root, ctx) = fixture(Stage::Iface);
        let tool = SubmitPublicTool { ctx };
        let r = tool
            .call(SubmitRustArgs {
                content: "pub trait T { fn f(&self); }\n".into(),
            })
            .await
            .unwrap();
        assert!(!r.no_change);
        assert!(graph.lock().get(root).unwrap().public_rs.is_some());
    }

    #[tokio::test]
    async fn submit_public_rejects_impl_block() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Iface);
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
    async fn submit_public_idempotent_no_change() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Iface);
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

    #[tokio::test]
    async fn submit_spec_creates_children_with_deps() {
        let (_tmp, graph, root, ctx) = fixture(Stage::Spec);
        // Pre-existing utility node so a child can declare dep on it.
        graph
            .lock()
            .add_child(root, Node::new("util", "shared"))
            .unwrap();
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# umbrella\n\nFrobs widgets via util".into(),
                private: None,
                children: vec![ChildDecl {
                    name: "frob".into(),
                    description: "frobs widgets".into(),
                    deps: vec!["util".into()],
                    crate_boundary: false,
                }],
                deps: vec![],
            })
            .await
            .unwrap();
        assert_eq!(r.children_created.len(), 1);
        let g = graph.lock();
        let frob = g.find_by_name("frob").unwrap();
        assert_eq!(frob.deps.len(), 1);
        assert_eq!(g.get(frob.deps[0]).unwrap().name, "util");
    }

    #[tokio::test]
    async fn submit_spec_unknown_child_dep_fails_atomically() {
        let (_tmp, graph, _root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        let err = tool
            .call(SubmitSpecArgs {
                public: "# x".into(),
                private: None,
                children: vec![ChildDecl {
                    name: "x".into(),
                    description: "".into(),
                    deps: vec!["nonexistent".into()],
                    crate_boundary: false,
                }],
                deps: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::Subtask(_)));
        // Atomic: no node created, public spec NOT written.
        assert_eq!(graph.lock().len(), 1);
        assert_eq!(graph.lock().iter().next().unwrap().spec_public_md, None);
    }

    #[tokio::test]
    async fn submit_spec_invalid_child_name_rejected() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        let err = tool
            .call(SubmitSpecArgs {
                public: "# x".into(),
                private: None,
                children: vec![ChildDecl {
                    name: "1bad".into(),
                    description: "".into(),
                    deps: vec![],
                    crate_boundary: false,
                }],
                deps: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::Subtask(_)));
    }

    #[tokio::test]
    async fn submit_spec_rejects_when_max_nodes_would_be_exceeded() {
        // fixture() seeds 1 root node. Build a ctx with max_nodes=2: one
        // child is OK, two children at once is rejected atomically.
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "the app")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            root,
            Stage::Spec,
            Role::Writer,
            graph.clone(),
            workdir,
            Layout::SingleCrate,
            300,
            500,
            2, // max_nodes — only one more child fits
            5,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# x".into(),
                private: None,
                children: vec![
                    ChildDecl {
                        name: "a".into(),
                        description: "".into(),
                        deps: vec![],
                        crate_boundary: false,
                    },
                    ChildDecl {
                        name: "b".into(),
                        description: "".into(),
                        deps: vec![],
                        crate_boundary: false,
                    },
                ],
                deps: vec![],
            })
            .await;
        let err = r.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ABANDON") || msg.contains("Either"), "got: {msg}");
        assert!(msg.contains("cap"), "should mention the cap: {msg}");
        assert_eq!(graph.lock().len(), 1);
    }

    #[tokio::test]
    async fn submit_spec_rejects_when_max_node_depth_would_be_exceeded() {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let mid = g.add_child(root, Node::new("mid", "")).unwrap();
        render::render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
        let graph = Arc::new(Mutex::new(g));
        let ctx = Arc::new(TaskCtx::new(
            Uuid::new_v4(),
            mid,
            Stage::Spec,
            Role::Writer,
            graph.clone(),
            workdir,
            Layout::SingleCrate,
            300,
            500,
            64,
            1, // max_node_depth
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                public: "# mid".into(),
                private: None,
                children: vec![ChildDecl {
                    name: "deep".into(),
                    description: "".into(),
                    deps: vec![],
                    crate_boundary: false,
                }],
                deps: vec![],
            })
            .await;
        let err = r.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ABANDON"), "should tell model to abandon: {msg}");
        assert!(msg.contains("depth cap"), "should mention depth cap: {msg}");
        assert_eq!(graph.lock().len(), 2);
    }

    #[tokio::test]
    async fn submit_verdict_records_into_ctx() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitVerdictTool { ctx: ctx.clone() };
        let _ = tool
            .call(SubmitVerdictArgs {
                satisfactory: false,
                reason: "missing thing".into(),
            })
            .await
            .unwrap();
        let v = ctx.verdict.lock().clone().unwrap();
        assert!(!v.satisfactory);
        assert_eq!(v.reason, "missing thing");
    }

    #[tokio::test]
    async fn loop_detection_triggers_after_three_identical_calls() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Iface);
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
    fn tool_catalogs_per_stage_role() {
        // Spot-check
        assert!(tool_names_for(Stage::Spec, Role::Writer).contains(&"submit_spec"));
        // The composite tool replaces the separate public/private/decompose
        // trio — those names should NOT appear anywhere.
        for stage in Stage::ALL {
            for role in [Role::Writer, Role::Critic, Role::Reviser, Role::Judge] {
                let names = tool_names_for(stage, role);
                assert!(!names.contains(&"submit_spec_public"));
                assert!(!names.contains(&"submit_spec_private"));
                assert!(!names.contains(&"decompose"));
            }
        }
        assert!(tool_names_for(Stage::Iface, Role::Writer).contains(&"submit_public"));
        assert!(tool_names_for(Stage::Tests, Role::Writer).contains(&"submit_tests"));
        assert!(tool_names_for(Stage::Impl, Role::Writer).contains(&"submit_private"));
        assert!(tool_names_for(Stage::Impl, Role::Judge).contains(&"submit_verdict"));
        // Critic gets diagnostics in coding stages but no verdict tool.
        assert!(!tool_names_for(Stage::Impl, Role::Critic).contains(&"submit_verdict"));
        // Spec critic has no tools (it just reads the inlined context).
        assert!(tool_names_for(Stage::Spec, Role::Critic).is_empty());
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
    fn speaker_classifies_model_vs_bureau_correctly() {
        let cases = [
            (TranscriptKind::System, Speaker::Bureau),
            (TranscriptKind::UserPrompt, Speaker::Bureau),
            (
                TranscriptKind::ToolDefinitions { tools: vec![] },
                Speaker::Bureau,
            ),
            (
                TranscriptKind::ToolResult {
                    tool: "x".into(),
                    ok: true,
                    error: None,
                    output: None,
                },
                Speaker::Bureau,
            ),
            (TranscriptKind::Note, Speaker::Bureau),
            (TranscriptKind::Error, Speaker::Bureau),
            (TranscriptKind::AssistantText, Speaker::Model),
            (TranscriptKind::ToolCall { tool: "x".into() }, Speaker::Model),
        ];
        for (kind, expected) in cases {
            let e = TranscriptEntry {
                timestamp: Utc::now(),
                kind,
                content: String::new(),
                role: None,
            };
            assert_eq!(e.speaker(), expected, "wrong speaker for {:?}", e.kind);
        }
    }
}
