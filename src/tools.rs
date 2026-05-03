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
    /// The node and stage this task is advancing.
    pub node_id: NodeId,
    pub stage: Stage,
    /// Live shared graph (per-task tools mutate this directly via the
    /// orchestrator's lock; we go through the shared Arc to avoid copying
    /// the whole graph each call).
    pub graph: Arc<Mutex<NodeGraph>>,
    /// Workdir on disk; we re-render after each successful submit.
    pub workdir: PathBuf,
    pub layout: Layout,
    pub max_file_lines: usize,
    pub max_spec_section_lines: usize,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptKind {
    System,
    UserPrompt,
    AssistantText,
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

const LOOP_BREAK_THRESHOLD: usize = 3;
const LOOP_WINDOW: usize = 8;

impl TaskCtx {
    pub fn new(
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        graph: Arc<Mutex<NodeGraph>>,
        workdir: PathBuf,
        layout: Layout,
        max_file_lines: usize,
        max_spec_section_lines: usize,
        cargo_lock: Arc<tokio::sync::Mutex<()>>,
    ) -> Self {
        Self {
            task_id,
            node_id,
            stage,
            graph,
            workdir,
            layout,
            max_file_lines,
            max_spec_section_lines,
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
// submit_spec
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug)]
pub struct SubmitSpecArgs {
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct SubmitSpecOk {
    pub bytes: u64,
    pub lines: usize,
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
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Author the markdown spec for this node. The framework persists it as \
                 spec/<node-path>/spec.md. Only available in the spec stage. Replaces any \
                 prior content."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<SubmitSpecOk, _>(e));
        }
        let r: Result<SubmitSpecOk, ToolFailure> = (|| {
            self.ctx.require_stage(Self::NAME, &[Stage::Spec])?;
            let lines = args.content.lines().count();
            if lines > self.ctx.max_spec_section_lines {
                return Err(ToolFailure::FileTooLarge(
                    lines,
                    self.ctx.max_spec_section_lines,
                ));
            }
            let bytes = args.content.len() as u64;
            {
                let mut g = self.ctx.graph.lock();
                let n = g.get_mut(self.ctx.node_id).ok_or_else(|| {
                    ToolFailure::Other(format!("node {} missing", self.ctx.node_id))
                })?;
                n.spec_md = Some(args.content);
                n.updated_at = Utc::now();
            }
            self.ctx.render_after_write()?;
            Ok(SubmitSpecOk { bytes, lines })
        })();
        self.ctx.finish(Self::NAME, r)
    }
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
            description:
                "Author the public surface (`public.rs`) for this node. Allowed: `pub trait` \
                 declarations (signatures only — no default bodies), `pub struct/enum/type` \
                 declarations, `pub const/static`, `use super::private::...` imports, doc \
                 comments. Forbidden: `impl` blocks, free functions, inline `mod`, \
                 `pub use crate::*` cross-node re-exports. Available only in the iface stage."
                    .into(),
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
            description:
                "Author the private internals (`private.rs`) for this node. Anything is \
                 allowed that compiles, except: `use crate::<X>::...` paths must reference \
                 a declared dep / ancestor / own child / self of this node. The framework \
                 catches this at submit time before invoking cargo. Available in iface \
                 (initial scaffold), impl, and debug stages."
                    .into(),
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
            description:
                "Author the tests (`tests.rs`) for this node. Lives inside `#[cfg(test)]` so \
                 its imports are unrestricted relative to private — but `use crate::<X>::...` \
                 paths must still reference declared deps / ancestors. Available in tests \
                 (initial author) and debug stages."
                    .into(),
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
// decompose
// --------------------------------------------------------------------------

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ChildDecl {
    pub name: String,
    pub description: String,
    /// References to existing node names that this child will depend on.
    #[serde(default)]
    pub deps: Vec<String>,
    /// If true, this child is a separate Cargo crate (workspace mode only).
    #[serde(default)]
    pub crate_boundary: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DecomposeArgs {
    /// New child nodes to create under the current node.
    #[serde(default)]
    pub children: Vec<ChildDecl>,
    /// Add dep edges from the current node to these existing nodes.
    #[serde(default)]
    pub add_self_deps: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct DecomposeOk {
    pub created: Vec<NodeIdRef>,
    pub self_deps_added: Vec<NodeIdRef>,
}

#[derive(Serialize, Debug, Clone)]
pub struct NodeIdRef {
    pub id: String,
    pub name: String,
}

pub struct DecomposeTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for DecomposeTool {
    const NAME: &'static str = "decompose";
    type Error = ToolFailure;
    type Args = DecomposeArgs;
    type Output = DecomposeOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Decompose this node into children and/or declare dep edges to existing \
                 nodes. Each child has a `name` (snake_case Rust ident), a `description`, \
                 and an optional `deps` list referencing existing nodes by name. Use \
                 `add_self_deps` to add edges from the CURRENT node to existing nodes \
                 without creating children. Cycle-checked. Available only in the spec \
                 stage."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "children": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "deps": {"type": "array", "items": {"type": "string"}},
                                "crate_boundary": {"type": "boolean"}
                            },
                            "required": ["name", "description"]
                        }
                    },
                    "add_self_deps": {"type": "array", "items": {"type": "string"}}
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<DecomposeOk, _>(e));
        }
        let r: Result<DecomposeOk, ToolFailure> = (|| {
            self.ctx.require_stage(Self::NAME, &[Stage::Spec])?;
            let mut g = self.ctx.graph.lock();
            let parent_id = self.ctx.node_id;

            // Plan first, mutate second — so any error in planning aborts
            // before we touch the graph.
            // Step 1a: resolve add_self_deps against the EXISTING graph.
            let self_deps_resolved: Vec<NodeId> = args
                .add_self_deps
                .iter()
                .map(|name| {
                    g.find_by_name(name).map(|n| n.id).ok_or_else(|| {
                        ToolFailure::Subtask(format!(
                            "add_self_deps: no existing node named '{name}'"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Step 1b: validate children + classify each child-dep as either
            // an existing-graph reference (resolved now) or a forward
            // reference to a yet-to-be-created sibling (resolved at apply
            // time). Names of children to be created can only be referenced
            // by LATER siblings in the same call (no forward refs).
            #[derive(Clone)]
            enum DepRef {
                Existing(NodeId),
                Sibling(String),
            }
            let new_names: HashSet<&str> =
                args.children.iter().map(|c| c.name.as_str()).collect();
            if new_names.len() != args.children.len() {
                return Err(ToolFailure::Subtask(
                    "duplicate child names in this decompose call".into(),
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
                            "child '{}' cannot depend on itself",
                            child.name
                        )));
                    }
                    if created_so_far.contains(dep_name) {
                        // Earlier sibling in this same call.
                        deps.push(DepRef::Sibling(dep_name.clone()));
                    } else if new_names.contains(dep_name.as_str()) {
                        // Sibling that comes LATER in this call — forward
                        // reference, which the apply order can't satisfy.
                        return Err(ToolFailure::Subtask(format!(
                            "child '{}' references later sibling '{}'; reorder so the dep comes first",
                            child.name, dep_name
                        )));
                    } else {
                        // Must already exist in the graph.
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

            // Step 2: apply.
            let mut self_deps_added = Vec::new();
            for to in self_deps_resolved {
                g.add_dep(parent_id, to)?;
                let n = g.get(to).unwrap();
                self_deps_added.push(NodeIdRef {
                    id: n.id.to_string(),
                    name: n.name.clone(),
                });
            }
            let mut name_to_id: std::collections::HashMap<String, NodeId> =
                std::collections::HashMap::new();
            let mut created = Vec::new();
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
                created.push(NodeIdRef {
                    id: new_id.to_string(),
                    name: child.name,
                });
            }

            drop(g);
            self.ctx.render_after_write()?;
            Ok(DecomposeOk {
                created,
                self_deps_added,
            })
        })();
        self.ctx.finish(Self::NAME, r)
    }
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
            description:
                "Record judge verdict. Pass satisfactory=true if the reviser addressed the \
                 critique; satisfactory=false with a concrete reason otherwise."
                    .into(),
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
            description:
                "Run `cargo check` and get structured diagnostics. Use mid-task to verify what \
                 you wrote compiles. Capped at 8 errors + 2KB stderr."
                    .into(),
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
            description:
                "Run `cargo test --no-fail-fast`. Returns compile errors AND runtime test \
                 failures. `test_filter` / `test_filters` narrow to substring-matched tests."
                    .into(),
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
            description:
                "Run `cargo test --no-run` to verify tests COMPILE without running them. \
                 Useful in the tests stage where bodies are still stubs."
                    .into(),
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
            description:
                "Run `cargo clippy -- -D warnings` and get structured lint diagnostics."
                    .into(),
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
    Actor,
    Critic,
    Reviser,
    Judge,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Actor => "actor",
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
pub fn tool_names_for(stage: Stage, role: Role) -> Vec<&'static str> {
    use Role::*;
    use Stage::*;
    match (stage, role) {
        (Spec, Actor) | (Spec, Reviser) => {
            vec![SubmitSpecTool::NAME, DecomposeTool::NAME]
        }
        (Spec, Critic) => vec![],
        (Spec, Judge) => vec![SubmitVerdictTool::NAME],

        (Iface, Actor) | (Iface, Reviser) => vec![
            SubmitPublicTool::NAME,
            SubmitPrivateTool::NAME, // initial scaffold; impl stage will refine
            CargoCheckTool::NAME,
        ],
        (Iface, Critic) => vec![CargoCheckTool::NAME],
        (Iface, Judge) => vec![CargoCheckTool::NAME, SubmitVerdictTool::NAME],

        (Tests, Actor) | (Tests, Reviser) => vec![
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

        (Impl, Actor) | (Impl, Reviser) => vec![
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

        (Debug, Actor) | (Debug, Reviser) => vec![
            SubmitPrivateTool::NAME,
            SubmitTestsTool::NAME,
            CargoCheckTool::NAME,
            CargoTestTool::NAME,
            CargoClippyTool::NAME,
        ],
        (Debug, Critic) => vec![CargoCheckTool::NAME, CargoTestTool::NAME],
        (Debug, Judge) => vec![CargoTestTool::NAME, SubmitVerdictTool::NAME],

        (Opt, Actor) | (Opt, Reviser) => vec![
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
            graph.clone(),
            workdir,
            Layout::SingleCrate,
            300,
            500,
            Arc::new(tokio::sync::Mutex::new(())),
        ));
        (tmp, graph, root, ctx)
    }

    #[tokio::test]
    async fn submit_spec_persists_to_node_and_disk() {
        let (tmp, graph, root, ctx) = fixture(Stage::Spec);
        let tool = SubmitSpecTool { ctx };
        let r = tool
            .call(SubmitSpecArgs {
                content: "# Spec\n\nDoes the thing.".into(),
            })
            .await
            .unwrap();
        assert!(r.lines >= 2);
        assert_eq!(
            graph.lock().get(root).unwrap().spec_md.as_deref(),
            Some("# Spec\n\nDoes the thing.")
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("spec/app/spec.md")).unwrap();
        assert!(on_disk.contains("Does the thing"));
    }

    #[tokio::test]
    async fn submit_spec_rejected_outside_spec_stage() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Iface);
        let tool = SubmitSpecTool { ctx };
        let err = tool
            .call(SubmitSpecArgs { content: "x".into() })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::WrongStage { .. }));
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
    async fn decompose_creates_children_with_deps() {
        let (_tmp, graph, root, ctx) = fixture(Stage::Spec);
        // Pre-existing utility node so a child can declare dep on it.
        graph
            .lock()
            .add_child(root, Node::new("util", "shared"))
            .unwrap();
        let tool = DecomposeTool { ctx };
        let r = tool
            .call(DecomposeArgs {
                children: vec![ChildDecl {
                    name: "frob".into(),
                    description: "frobs widgets".into(),
                    deps: vec!["util".into()],
                    crate_boundary: false,
                }],
                add_self_deps: vec![],
            })
            .await
            .unwrap();
        assert_eq!(r.created.len(), 1);
        let g = graph.lock();
        let frob = g.find_by_name("frob").unwrap();
        assert_eq!(frob.deps.len(), 1);
        assert_eq!(g.get(frob.deps[0]).unwrap().name, "util");
    }

    #[tokio::test]
    async fn decompose_unknown_dep_fails_atomically() {
        let (_tmp, graph, _root, ctx) = fixture(Stage::Spec);
        let tool = DecomposeTool { ctx };
        let err = tool
            .call(DecomposeArgs {
                children: vec![ChildDecl {
                    name: "x".into(),
                    description: "".into(),
                    deps: vec!["nonexistent".into()],
                    crate_boundary: false,
                }],
                add_self_deps: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::Subtask(_)));
        // Ensure no node was created.
        assert_eq!(graph.lock().len(), 1);
    }

    #[tokio::test]
    async fn decompose_invalid_child_name_rejected() {
        let (_tmp, _g, _root, ctx) = fixture(Stage::Spec);
        let tool = DecomposeTool { ctx };
        let err = tool
            .call(DecomposeArgs {
                children: vec![ChildDecl {
                    name: "1bad".into(),
                    description: "".into(),
                    deps: vec![],
                    crate_boundary: false,
                }],
                add_self_deps: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolFailure::Subtask(_)));
    }

    #[tokio::test]
    async fn decompose_two_children_can_dep_on_each_other_in_order() {
        // child 'a' is created first; child 'b' (created after) can dep on 'a'.
        let (_tmp, graph, _root, ctx) = fixture(Stage::Spec);
        let tool = DecomposeTool { ctx };
        let r = tool
            .call(DecomposeArgs {
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
                        deps: vec!["a".into()],
                        crate_boundary: false,
                    },
                ],
                add_self_deps: vec![],
            })
            .await
            .unwrap();
        assert_eq!(r.created.len(), 2);
        let g = graph.lock();
        let b = g.find_by_name("b").unwrap();
        assert_eq!(g.get(b.deps[0]).unwrap().name, "a");
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
        assert!(tool_names_for(Stage::Spec, Role::Actor).contains(&"submit_spec"));
        assert!(tool_names_for(Stage::Spec, Role::Actor).contains(&"decompose"));
        assert!(tool_names_for(Stage::Iface, Role::Actor).contains(&"submit_public"));
        assert!(tool_names_for(Stage::Tests, Role::Actor).contains(&"submit_tests"));
        assert!(tool_names_for(Stage::Impl, Role::Actor).contains(&"submit_private"));
        assert!(tool_names_for(Stage::Impl, Role::Judge).contains(&"submit_verdict"));
        // Critic gets diagnostics in coding stages but no verdict tool.
        assert!(!tool_names_for(Stage::Impl, Role::Critic).contains(&"submit_verdict"));
        // Spec critic has no tools (it just reads the inlined context).
        assert!(tool_names_for(Stage::Spec, Role::Critic).is_empty());
    }
}
