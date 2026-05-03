//! Tool implementations for agent invocation. Tools are typed wrappers around
//! filesystem and orchestrator operations, dispatched by `rig`.
//!
//! Each tool carries an `Arc<TaskCtx>` so the orchestrator can:
//! - enforce read/write set policies declared at task creation time
//! - capture file writes as commit material in the worktree
//! - collect subtask emissions
//! - record transcript entries

use crate::artifact;
use crate::phase::Phase;
use crate::state::{StateHandle, UiEvent};
use crate::task::{Role, SubtaskDecl, TranscriptEntry, TranscriptKind};
use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// Static description of one tool that gets sent to the LLM in the tools
/// catalog of every API call. Used by the UI to display "what we are putting
/// in front of the model" without having to inspect a live agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Render the static tool catalog for a given phase. The returned definitions
/// match what `run_agent_for_phase` registers with `rig::AgentBuilder`. The
/// `prompt` argument is unused by all our tools' definition() impls, so we
/// pass empty.
pub async fn phase_tools(phase: Phase) -> Vec<ToolInfo> {
    use rig::tool::Tool as _;
    // Construct a minimal TaskCtx-less tool by leveraging that all our
    // `definition` methods ignore `&self`. We can't do that directly
    // (Tool::definition takes &self), so we instead hand-roll the
    // descriptions here by calling each tool's `definition` with a dummy
    // ctx. To avoid that gymnastics we just hard-code the descriptions
    // by introspecting the same JSON the definition() returns.
    fn td(name: &str, description: &str, parameters: serde_json::Value) -> ToolInfo {
        ToolInfo {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }
    let write_file = td(
        WriteFileTool::NAME,
        "Write a file relative to the workdir. Content is fully replaced. \
         Phase-specific: in Interface phase, function bodies are auto-stubbed \
         to todo!(). Test phase: only files under tests/. Impl phase: must \
         not change public signatures.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "path relative to workdir"},
                "content": {"type": "string", "description": "full file content"},
            },
            "required": ["path", "content"]
        }),
    );
    let read_file = td(
        ReadFileTool::NAME,
        "Read a file relative to the workdir.",
        serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }),
    );
    let list_files = td(
        ListFilesTool::NAME,
        "List files in the workdir or a subdirectory (recursive).",
        serde_json::json!({
            "type": "object",
            "properties": {"dir": {"type": "string"}}
        }),
    );
    let emit = td(
        EmitSubtasksTool::NAME,
        "Emit child tasks for parallel execution. Each task declares its \
         read/write file sets. Subject to depth + count caps.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "read_files": {"type": "array", "items": {"type": "string"}},
                            "write_files": {"type": "array", "items": {"type": "string"}},
                            "spec_sections": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["description"]
                    }
                }
            },
            "required": ["tasks"]
        }),
    );
    let replace_body = td(
        ReplaceFnBodyTool::NAME,
        "Replace the body of a named function in a Rust file (signature unchanged).",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "fn_name": {"type": "string"},
                "new_body": {"type": "string"}
            },
            "required": ["path", "fn_name", "new_body"]
        }),
    );
    let list_errors = td(
        ListCompilerErrorsTool::NAME,
        "List current cargo check / cargo test compiler errors.",
        serde_json::json!({"type": "object", "properties": {}}),
    );
    let read_error = td(
        ReadCompilerErrorTool::NAME,
        "Read a single compiler error in detail.",
        serde_json::json!({
            "type": "object",
            "properties": {"error_id": {"type": "string"}},
            "required": ["error_id"]
        }),
    );
    let _submit_verdict = td(
        SubmitVerdictTool::NAME,
        "Record judge verdict (satisfactory + reason). Only available to the judge role.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "satisfactory": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["satisfactory"]
        }),
    );
    let cargo_check = td(
        CargoCheckTool::NAME,
        "Run `cargo check` and get structured diagnostics. Use mid-task to verify what you wrote compiles.",
        serde_json::json!({
            "type": "object",
            "properties": {"package": {"type": "string"}}
        }),
    );
    let cargo_test = td(
        CargoTestTool::NAME,
        "Run `cargo test --no-fail-fast`. Returns compile errors and runtime test failures.",
        serde_json::json!({
            "type": "object",
            "properties": {"package": {"type": "string"}}
        }),
    );
    let cargo_test_no_run = td(
        CargoTestNoRunTool::NAME,
        "Run `cargo test --no-run` to verify tests compile without running them.",
        serde_json::json!({
            "type": "object",
            "properties": {"package": {"type": "string"}}
        }),
    );
    let cargo_clippy = td(
        CargoClippyTool::NAME,
        "Run `cargo clippy -- -D warnings` and get structured lint diagnostics.",
        serde_json::json!({
            "type": "object",
            "properties": {"package": {"type": "string"}}
        }),
    );

    match phase {
        Phase::Spec => vec![write_file, read_file, list_files, emit],
        Phase::Interface => vec![
            write_file,
            read_file,
            list_files,
            cargo_check,
            emit,
        ],
        Phase::Test => vec![
            write_file,
            read_file,
            list_files,
            cargo_check,
            cargo_test_no_run,
            emit,
        ],
        Phase::Impl => vec![
            write_file,
            read_file,
            list_files,
            cargo_check,
            cargo_test,
            cargo_clippy,
            emit,
        ],
        Phase::Debug => vec![
            write_file,
            read_file,
            list_files,
            replace_body,
            list_errors,
            read_error,
            cargo_check,
            cargo_test,
            cargo_clippy,
        ],
        Phase::Opt => vec![
            write_file,
            read_file,
            list_files,
            replace_body,
            cargo_test,
            cargo_clippy,
        ],
    }
}

#[derive(Debug, Error)]
pub enum ToolFailure {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("path '{0}' is outside the workdir")]
    OutsideWorkdir(PathBuf),
    #[error("path '{path}' is not in the declared write-set ({declared}); declare a wider write_files in your subtask, or work on a path you already declared")]
    WriteNotAllowed {
        path: PathBuf,
        declared: String,
    },
    #[error("path '{path}' is not in the declared read-set ({declared})")]
    ReadNotAllowed {
        path: PathBuf,
        declared: String,
    },
    #[error("write target '{0}' is forbidden in {1} phase")]
    PhaseForbidden(PathBuf, Phase),
    #[error("invalid utf-8 path")]
    InvalidPath,
    #[error("file too large: {0} lines (max {1})")]
    FileTooLarge(usize, usize),
    #[error("rust syntax error: {0}")]
    Syntax(String),
    #[error("function '{0}' not found")]
    FnNotFound(String),
    #[error("toml: {0}")]
    Toml(String),
    #[error("forbidden in {phase} phase: {reason}")]
    Forbidden { phase: Phase, reason: String },
    #[error(
        "loop detected: you have called the `{tool}` tool {count} times in a row with the same \
         arguments. The previous call(s) succeeded; there is nothing more to do for that input. \
         Stop repeating this call. If you have more files to write, call write_file with a \
         DIFFERENT path. If you are done, end your message with a brief summary instead of \
         calling more tools."
    )]
    Loop { tool: String, count: usize },
    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct CompilerError {
    pub id: String,
    pub file: Option<PathBuf>,
    pub line: Option<u32>,
    pub message: String,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeVerdict {
    pub satisfactory: bool,
    pub reason: String,
}

/// Per-task context shared by all tool instances bound to a task.
/// Tool methods take `&self`, so all mutable bookkeeping lives behind a Mutex.
///
/// When the critique cycle is enabled, multiple roles (actor, critic, reviser,
/// judge) each construct their own `TaskCtx` for one agent invocation, but
/// share the cumulative scratch (`written`, `emitted_subtasks`, `verdict`)
/// via cloned `Arc`s so the orchestrator sees the union of their effects.
pub struct TaskCtx {
    pub task_id: Uuid,
    pub phase: Phase,
    /// Which role is currently driving the agent. Stamped onto every
    /// transcript entry recorded through this ctx.
    pub role: Role,
    /// Workdir root for filesystem operations (typically the task's worktree).
    pub workdir: PathBuf,
    /// Allowed read paths (empty = unrestricted within workdir).
    pub read_set: HashSet<PathBuf>,
    /// Allowed write paths.
    pub write_set: HashSet<PathBuf>,
    /// Files written by ANY role during this task's execution.
    pub written: Arc<Mutex<HashSet<PathBuf>>>,
    /// Subtasks emitted via emit_subtasks tool (actor only).
    pub emitted_subtasks: Arc<Mutex<Vec<SubtaskDecl>>>,
    /// Judge's verdict, if any.
    pub verdict: Arc<Mutex<Option<JudgeVerdict>>>,
    pub compiler_errors: Vec<CompilerError>,
    pub max_file_lines: usize,
    pub max_spec_section_lines: usize,
    /// Current task's depth in the task tree.
    pub depth: u32,
    /// Hard cap on subtask depth. emit_subtasks fails when depth >= cap.
    pub max_subtask_depth: u32,
    /// Sliding window of recent (tool_name, args_hash) pairs used to detect
    /// the model repeating the same call.
    pub recent_calls: Arc<Mutex<VecDeque<(String, u64)>>>,
    pub state: StateHandle,
}

/// How many consecutive identical (tool, args) calls before the harness
/// breaks the loop with an error.
pub const LOOP_BREAK_THRESHOLD: usize = 3;
/// How much history we keep for the loop detector.
pub const LOOP_WINDOW: usize = 8;

impl TaskCtx {
    fn record(&self, kind: TranscriptKind) {
        let entry = TranscriptEntry {
            timestamp: Utc::now(),
            kind: kind.clone(),
            content: String::new(),
            role: self.role,
        };
        self.state.write(|s| {
            if let Some(t) = s.graph.get_mut(self.task_id) {
                t.transcript.push(entry.clone());
            }
        });
        self.state.emit(UiEvent::TranscriptAppended {
            task_id: self.task_id,
            entry,
        });
    }

    /// Serialize `args`, record the tool_call entry, and update the sliding
    /// window used for loop detection. Returns `Err(...)` if the same tool
    /// call has been issued LOOP_BREAK_THRESHOLD times in a row, which means
    /// the model is stuck and the harness should break the loop.
    pub fn record_call_and_check_loop<T: Serialize>(
        &self,
        name: &str,
        args: &T,
    ) -> Result<(), ToolFailure> {
        let s = serde_json::to_string(args).unwrap_or_default();
        self.record_call(name, &s);
        let h = {
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            s.hash(&mut hasher);
            hasher.finish()
        };
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

    fn record_call(&self, name: &str, args: &str) {
        let kind = TranscriptKind::ToolCall {
            tool: name.to_string(),
            args: args.to_string(),
        };
        self.record(kind);
    }

    fn record_result(&self, name: &str, ok: bool) {
        self.record_result_with(name, ok, None, None);
    }

    fn record_result_with(
        &self,
        name: &str,
        ok: bool,
        error: Option<String>,
        output: Option<String>,
    ) {
        let kind = TranscriptKind::ToolResult {
            tool: name.to_string(),
            ok,
            error,
            output,
        };
        self.record(kind);
    }

    fn finish<T: Serialize>(
        &self,
        name: &str,
        r: Result<T, ToolFailure>,
    ) -> Result<T, ToolFailure> {
        match &r {
            Ok(v) => {
                let out = serde_json::to_string(v).ok();
                self.record_result_with(name, true, None, out);
            }
            Err(e) => self.record_result_with(name, false, Some(format!("{e}")), None),
        }
        r
    }

    fn check_inside_workdir(&self, p: &Path) -> Result<PathBuf, ToolFailure> {
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.workdir.join(p)
        };
        let canonical_root = std::fs::canonicalize(&self.workdir)
            .unwrap_or_else(|_| self.workdir.clone());
        // Don't require canonical for new files, just normalize.
        let normalized = normalize_path(&abs);
        if !normalized.starts_with(&canonical_root) && !normalized.starts_with(&self.workdir) {
            return Err(ToolFailure::OutsideWorkdir(p.to_path_buf()));
        }
        Ok(normalized)
    }

    fn rel_path(&self, abs: &Path) -> PathBuf {
        abs.strip_prefix(&self.workdir)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| abs.to_path_buf())
    }

    fn check_read(&self, rel: &Path) -> Result<(), ToolFailure> {
        if self.read_set.is_empty() {
            return Ok(());
        }
        if path_in_set(rel, &self.read_set) || path_in_set(rel, &self.write_set) {
            return Ok(());
        }
        Err(ToolFailure::ReadNotAllowed {
            path: rel.to_path_buf(),
            declared: format_set(&self.read_set),
        })
    }

    fn check_write(&self, rel: &Path) -> Result<(), ToolFailure> {
        if path_in_set(rel, &self.write_set) {
            return Ok(());
        }
        Err(ToolFailure::WriteNotAllowed {
            path: rel.to_path_buf(),
            declared: if self.write_set.is_empty() {
                "<empty — task declared no writable paths>".to_string()
            } else {
                format_set(&self.write_set)
            },
        })
    }

    fn record_written(&self, rel: PathBuf) {
        self.written.lock().insert(rel);
    }
}

/// Match `rel` against an access set with two kinds of entries:
///  - exact path: `spec/types.md` matches only that one path
///  - directory prefix (entry whose string ends in `/`): `spec/` matches
///    `spec/types.md`, `spec/sub/x.md`, etc., but not `spec` alone.
pub fn path_in_set(rel: &Path, set: &HashSet<PathBuf>) -> bool {
    if set.contains(rel) {
        return true;
    }
    let rel_s = rel.to_string_lossy();
    for entry in set {
        let s = entry.to_string_lossy();
        if let Some(prefix) = s.strip_suffix('/') {
            // dir-prefix match: rel must start with `prefix/`.
            let with_slash = format!("{prefix}/");
            if rel_s.starts_with(&with_slash) {
                return true;
            }
        }
    }
    false
}

fn format_set(s: &HashSet<PathBuf>) -> String {
    let mut v: Vec<String> = s.iter().map(|p| p.display().to_string()).collect();
    v.sort();
    v.join(", ")
}

fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// =====================================================================
// write_file
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct WriteFileOk {
    pub bytes: u64,
    pub warnings: Vec<String>,
    /// True when the file already contained byte-identical content and the
    /// write was a no-op. Surfaced to the model so it knows not to repeat.
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
        let extra = match self.ctx.phase {
            Phase::Interface => " Function bodies, if any, must be `todo!()`; the orchestrator will normalize otherwise.",
            Phase::Test => " Only test files (under tests/) are allowed.",
            Phase::Impl => " Only implementation files; do not modify public signatures from interface files.",
            _ => "",
        };
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Write a file relative to the workdir. Content is fully replaced.{extra}"
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "path relative to workdir"},
                    "content": {"type": "string", "description": "full file content"},
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => do_write_file(&self.ctx, args).await,
        };
        self.ctx.finish(Self::NAME, r)
    }
}

async fn do_write_file(ctx: &TaskCtx, args: WriteFileArgs) -> Result<WriteFileOk, ToolFailure> {
    let rel = PathBuf::from(&args.path);
    let abs = ctx.check_inside_workdir(&rel)?;
    let rel_norm = ctx.rel_path(&abs);
    ctx.check_write(&rel_norm)?;

    let mut warnings = Vec::new();
    let mut content = args.content;

    // Phase-specific guards
    let kind = crate::paths::classify(&rel_norm);
    let is_spec_md = kind == crate::paths::PathKind::Spec;
    // Phase guards based on path kind. Each phase produces only its own
    // artifact type; later phases must not edit earlier phases' artifacts.
    // Both single-crate and workspace layouts are accepted via the
    // classifier — it pattern-matches `**/src/**/*.rs`, `**/Cargo.toml`,
    // `**/tests/**/*.rs`, and Rust internal-test conventions.
    match (ctx.phase, kind) {
        (Phase::Spec, crate::paths::PathKind::Spec) => {}
        (Phase::Spec, _) => return Err(ToolFailure::PhaseForbidden(rel_norm, ctx.phase)),

        (Phase::Interface, crate::paths::PathKind::RustSource)
        | (Phase::Interface, crate::paths::PathKind::CargoToml) => {}
        (Phase::Interface, _) => {
            return Err(ToolFailure::PhaseForbidden(rel_norm, ctx.phase));
        }

        (Phase::Test, crate::paths::PathKind::RustTest)
        | (Phase::Test, crate::paths::PathKind::CargoToml) => {}
        (Phase::Test, _) => return Err(ToolFailure::PhaseForbidden(rel_norm, ctx.phase)),

        (Phase::Impl, crate::paths::PathKind::RustSource)
        | (Phase::Impl, crate::paths::PathKind::CargoToml) => {}
        (Phase::Impl, _) => return Err(ToolFailure::PhaseForbidden(rel_norm, ctx.phase)),

        // Debug fixes whatever's broken: source, tests, Cargo.toml.
        (Phase::Debug, crate::paths::PathKind::RustSource)
        | (Phase::Debug, crate::paths::PathKind::RustTest)
        | (Phase::Debug, crate::paths::PathKind::CargoToml) => {}
        (Phase::Debug, _) => return Err(ToolFailure::PhaseForbidden(rel_norm, ctx.phase)),

        (Phase::Opt, crate::paths::PathKind::RustSource) => {}
        (Phase::Opt, _) => return Err(ToolFailure::PhaseForbidden(rel_norm, ctx.phase)),
    }

    // Rust files: validate syntax and (in Interface phase) stub bodies
    if artifact::is_rust_file(&rel_norm) {
        if ctx.phase == Phase::Interface {
            match artifact::stub_function_bodies(&content) {
                Ok((stubbed, w)) => {
                    content = stubbed;
                    warnings.extend(w);
                }
                Err(e) => return Err(ToolFailure::Syntax(format!("{e:#}"))),
            }
        } else if let Err(e) = artifact::validate_rust(&rel_norm, &content) {
            return Err(ToolFailure::Syntax(format!("{e:#}")));
        }
    }

    // File-size guard. Spec markdown sections get a more generous limit than
    // Rust source files because prose isn't bounded the same way.
    let line_count = content.lines().count();
    let limit = if is_spec_md {
        ctx.max_spec_section_lines
    } else {
        ctx.max_file_lines
    };
    if line_count > limit {
        return Err(ToolFailure::FileTooLarge(line_count, limit));
    }

    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = content.len() as u64;
    let no_change = match std::fs::read_to_string(&abs) {
        Ok(existing) => existing == content,
        Err(_) => false,
    };
    if !no_change {
        std::fs::write(&abs, content.as_bytes())?;
        ctx.record_written(rel_norm.clone());
        ctx.state.emit(UiEvent::FileChanged { path: rel_norm });
    } else {
        warnings.push(format!(
            "no_change: file already contained byte-identical content; skipping write"
        ));
    }

    Ok(WriteFileOk {
        bytes,
        warnings,
        no_change,
    })
}

// =====================================================================
// read_file
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct ReadFileArgs {
    pub path: String,
}

#[derive(Serialize, Debug)]
pub struct ReadFileOk {
    pub content: String,
    pub lines: usize,
}

pub struct ReadFileTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for ReadFileTool {
    const NAME: &'static str = "read_file";
    type Error = ToolFailure;
    type Args = ReadFileArgs;
    type Output = ReadFileOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Read a file relative to the workdir.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => do_read_file(&self.ctx, args).await,
        };
        self.ctx.finish(Self::NAME, r)
    }
}

async fn do_read_file(ctx: &TaskCtx, args: ReadFileArgs) -> Result<ReadFileOk, ToolFailure> {
    let rel = PathBuf::from(&args.path);
    let abs = ctx.check_inside_workdir(&rel)?;
    let rel_norm = ctx.rel_path(&abs);
    ctx.check_read(&rel_norm)?;
    if !abs.exists() {
        return Err(ToolFailure::Other(format!(
            "file '{}' does not exist (workdir is empty or this file has not been written yet)",
            rel_norm.display()
        )));
    }
    let content = std::fs::read_to_string(&abs).map_err(|e| {
        ToolFailure::Other(format!("could not read '{}': {}", rel_norm.display(), e))
    })?;
    let lines = content.lines().count();
    Ok(ReadFileOk { content, lines })
}

// =====================================================================
// list_files
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct ListFilesArgs {
    #[serde(default)]
    pub dir: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct ListFilesOk {
    /// Flat list of files relative to workdir, sorted.
    pub files: Vec<String>,
    /// Pretty tree rendering of the same set, suitable for direct inclusion
    /// in a model's context. Provided alongside `files` so the model has
    /// both forms.
    pub tree: String,
}

pub struct ListFilesTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for ListFilesTool {
    const NAME: &'static str = "list_files";
    type Error = ToolFailure;
    type Args = ListFilesArgs;
    type Output = ListFilesOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Recursively list every file under `dir` (or the whole workdir if \
                `dir` is omitted). Returns BOTH a flat list of relative paths AND a tree \
                rendering — call this once with no `dir` and you have the full layout; do \
                NOT call repeatedly for each subdirectory."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dir": {
                        "type": "string",
                        "description": "optional subdirectory to limit the listing to; usually leave unset"
                    }
                },
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => do_list_files(&self.ctx, args).await,
        };
        self.ctx.finish(Self::NAME, r)
    }
}

async fn do_list_files(ctx: &TaskCtx, args: ListFilesArgs) -> Result<ListFilesOk, ToolFailure> {
    let base = match args.dir {
        Some(d) => ctx.check_inside_workdir(Path::new(&d))?,
        None => ctx.workdir.clone(),
    };
    let mut out = Vec::new();
    if !base.exists() {
        return Ok(ListFilesOk {
            files: out,
            tree: "(empty)".to_string(),
        });
    }
    for entry in walkdir::WalkDir::new(&base)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let p = entry.path();
            let rel = ctx.rel_path(p);
            // Skip orchestrator bookkeeping. The filter must be applied to
            // the path RELATIVE TO workdir, not to `p` (absolute), because
            // the workdir itself can be nested under `.bureau/worktrees/...`
            // — checking absolute components would exclude every file.
            if rel.components().any(|c| {
                let s = c.as_os_str().to_string_lossy();
                s == "target" || s == ".git" || s == ".bureau"
            }) {
                continue;
            }
            out.push(rel.to_string_lossy().to_string());
        }
    }
    out.sort();
    let tree = render_tree(&out);
    Ok(ListFilesOk { files: out, tree })
}

/// Render a list of relative paths as an ASCII tree. Files only — directories
/// are inferred from path prefixes. Output looks like:
///
///   .
///   ├── Cargo.toml
///   ├── spec/
///   │   ├── problem.md
///   │   └── types.md
///   └── src/
///       ├── greeter.rs
///       └── lib.rs
pub fn render_tree(paths: &[String]) -> String {
    if paths.is_empty() {
        return "(empty)".to_string();
    }
    // Build a tree of {name -> (Option<Self>, is_dir)}. Use BTreeMap for
    // stable alphabetical order.
    use std::collections::BTreeMap;
    #[derive(Default)]
    struct Node {
        children: BTreeMap<String, Node>,
        is_file: bool,
    }
    let mut root = Node::default();
    for p in paths {
        let mut cur = &mut root;
        let parts: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
        let last = parts.len().saturating_sub(1);
        for (i, part) in parts.iter().enumerate() {
            let is_last = i == last;
            let entry = cur.children.entry(part.to_string()).or_default();
            if is_last {
                entry.is_file = true;
            }
            cur = entry;
        }
    }

    fn render(node: &Node, prefix: &str, out: &mut String) {
        let n = node.children.len();
        for (i, (name, child)) in node.children.iter().enumerate() {
            let last = i + 1 == n;
            let connector = if last { "└── " } else { "├── " };
            let label = if child.is_file && child.children.is_empty() {
                name.clone()
            } else {
                format!("{name}/")
            };
            out.push_str(prefix);
            out.push_str(connector);
            out.push_str(&label);
            out.push('\n');
            let next_prefix = format!("{prefix}{}", if last { "    " } else { "│   " });
            render(child, &next_prefix, out);
        }
    }

    let mut out = String::from(".\n");
    render(&root, "", &mut out);
    out
}

// =====================================================================
// emit_subtasks
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct EmitSubtasksArgs {
    pub tasks: Vec<SubtaskDecl>,
}

#[derive(Serialize, Debug)]
pub struct EmitSubtasksOk {
    pub accepted: usize,
}

pub struct EmitSubtasksTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for EmitSubtasksTool {
    const NAME: &'static str = "emit_subtasks";
    type Error = ToolFailure;
    type Args = EmitSubtasksArgs;
    type Output = EmitSubtasksOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Emit child tasks. Each task declares its read/write file sets and \
                description. Subtasks are scheduled by the orchestrator subject to interference \
                analysis.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "read_files": {"type": "array", "items": {"type": "string"}},
                                "write_files": {"type": "array", "items": {"type": "string"}},
                                "spec_sections": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["description"]
                        }
                    }
                },
                "required": ["tasks"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<EmitSubtasksOk, _>(e));
        }
        let r: Result<_, ToolFailure> = (|| {
            if !self.ctx.phase.allows_subtasks() {
                return Err(ToolFailure::Forbidden {
                    phase: self.ctx.phase,
                    reason: "subtask emission disabled in this phase".into(),
                });
            }
            if self.ctx.depth >= self.ctx.max_subtask_depth {
                return Err(ToolFailure::Forbidden {
                    phase: self.ctx.phase,
                    reason: format!(
                        "depth cap reached (depth={}, cap={}). Do the work yourself.",
                        self.ctx.depth, self.ctx.max_subtask_depth
                    ),
                });
            }
            // Validate each subtask declaration. For non-spec phases, the
            // child cannot do anything without a non-empty write_files: file
            // writes are gated on it. Reject up front rather than silently
            // creating zombie tasks.
            for (i, t) in args.tasks.iter().enumerate() {
                if t.description.trim().is_empty() {
                    return Err(ToolFailure::Other(format!(
                        "subtask #{i} has an empty description"
                    )));
                }
                if t.write_files.is_empty() {
                    return Err(ToolFailure::Other(format!(
                        "subtask #{i} ('{}') has empty write_files; every subtask must declare \
                         at least one path it intends to write. For a {} subtask the typical \
                         targets are: {}. Use a directory prefix like 'spec/' if the subtask \
                         needs to write multiple files in the same area.",
                        truncate_desc(&t.description, 60),
                        self.ctx.phase,
                        suggested_writes_for(self.ctx.phase),
                    )));
                }
                // Reject paths that are obviously wrong for the phase.
                for w in &t.write_files {
                    if !subtask_write_fits_phase(w, self.ctx.phase) {
                        let s = w.to_string_lossy();
                        return Err(ToolFailure::Other(format!(
                            "subtask #{i} declares write '{s}' which doesn't fit the {} phase",
                            self.ctx.phase
                        )));
                    }
                }
            }
            let n = args.tasks.len();
            self.ctx.emitted_subtasks.lock().extend(args.tasks);
            Ok(EmitSubtasksOk { accepted: n })
        })();
        self.ctx.finish(Self::NAME, r)
    }
}

/// Whether a declared subtask write path fits the given phase.
///
/// Accepts both exact paths (`crates/foo/src/lib.rs`) and directory-prefix
/// entries (`crates/foo/src/`). For prefix entries we look at the segments
/// between the trailing slash and the workdir root — `spec/` should produce
/// only specs; `src/`, `crates/`, etc. allow any path under them; `tests/`
/// must produce test files; and so on.
pub fn subtask_write_fits_phase(p: &std::path::PathBuf, phase: Phase) -> bool {
    let s = p.to_string_lossy();
    let is_prefix = s.ends_with('/');
    if is_prefix {
        // Directory-prefix entries: classify by what KINDS of files would
        // be produced under that prefix, then check phase-fitness on each.
        let segs: Vec<&str> = s.trim_end_matches('/').split('/').collect();
        return prefix_fits_phase(&segs, phase);
    }
    let kind = crate::paths::classify(p);
    match (phase, kind) {
        (Phase::Spec, crate::paths::PathKind::Spec) => true,
        (Phase::Interface, crate::paths::PathKind::RustSource) => true,
        (Phase::Interface, crate::paths::PathKind::CargoToml) => true,
        (Phase::Test, crate::paths::PathKind::RustTest) => true,
        (Phase::Test, crate::paths::PathKind::CargoToml) => true,
        (Phase::Impl, crate::paths::PathKind::RustSource) => true,
        (Phase::Impl, crate::paths::PathKind::CargoToml) => true,
        (Phase::Debug, crate::paths::PathKind::RustSource)
        | (Phase::Debug, crate::paths::PathKind::RustTest)
        | (Phase::Debug, crate::paths::PathKind::CargoToml) => true,
        (Phase::Opt, crate::paths::PathKind::RustSource) => true,
        _ => false,
    }
}

fn prefix_fits_phase(segs: &[&str], phase: Phase) -> bool {
    // Empty prefix doesn't make sense; reject.
    if segs.is_empty() {
        return false;
    }
    // The semantics: a directory prefix authorizes everything under it. We
    // ask "is there at least one well-formed file kind that's plausible
    // under this prefix and that fits the phase?"
    //
    // We approximate by looking for known structural segments:
    //   - any `src` segment in the prefix → contents are RustSource files
    //   - any `tests` or `test` segment   → contents are RustTest files
    //   - any `spec` segment              → contents are Spec files
    //   - `crates`, `crates/<name>`, or just the workspace root: any kind
    let has = |name: &str| segs.iter().any(|s| *s == name);
    let last_named = |name: &str| segs.last().map(|s| *s == name).unwrap_or(false);
    if has("spec") {
        return matches!(phase, Phase::Spec);
    }
    if has("tests") || (has("test") && has("src")) || last_named("test") {
        // tests dir, src/.../tests, src/test — all test prefixes
        return matches!(phase, Phase::Test | Phase::Debug);
    }
    if has("src") {
        return matches!(
            phase,
            Phase::Interface | Phase::Impl | Phase::Debug | Phase::Opt | Phase::Test
        );
        // (Test is allowed because internal tests live under src/.)
    }
    // Generic prefixes like `crates/` or `crates/foo/` — allow for any
    // phase that produces Rust-related artifacts (i.e. anything except
    // Spec). The per-file phase guard is the actual gate at write time.
    !matches!(phase, Phase::Spec)
}

fn truncate_desc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

pub fn suggested_writes_for(phase: Phase) -> &'static str {
    match phase {
        Phase::Spec => "spec/<section>.md",
        Phase::Interface => {
            "src/<mod>.rs, src/lib.rs, Cargo.toml (single crate); or crates/<name>/src/<mod>.rs, crates/<name>/Cargo.toml (workspace)"
        }
        Phase::Test => {
            "tests/<name>.rs (integration), src/<mod>/tests.rs (internal), or crates/<name>/tests/<name>.rs"
        }
        Phase::Impl => "src/<mod>.rs (single crate) or crates/<name>/src/<mod>.rs (workspace)",
        Phase::Debug => "any src/ or tests/ file under the workdir or a member crate",
        Phase::Opt => "any src/ file under the workdir or a member crate",
    }
}

// =====================================================================
// replace_fn_body
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct ReplaceFnBodyArgs {
    pub path: String,
    pub fn_name: String,
    pub new_body: String,
}

#[derive(Serialize, Debug)]
pub struct ReplaceFnBodyOk {
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_change: bool,
}

pub struct ReplaceFnBodyTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for ReplaceFnBodyTool {
    const NAME: &'static str = "replace_fn_body";
    type Error = ToolFailure;
    type Args = ReplaceFnBodyArgs;
    type Output = ReplaceFnBodyOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Replace the body of a named function in a Rust file. The signature is left unchanged.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "fn_name": {"type": "string"},
                    "new_body": {"type": "string", "description": "block contents (without surrounding braces)"}
                },
                "required": ["path", "fn_name", "new_body"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => do_replace_fn_body(&self.ctx, args).await,
        };
        self.ctx.finish(Self::NAME, r)
    }
}

async fn do_replace_fn_body(
    ctx: &TaskCtx,
    args: ReplaceFnBodyArgs,
) -> Result<ReplaceFnBodyOk, ToolFailure> {
    let rel = PathBuf::from(&args.path);
    let abs = ctx.check_inside_workdir(&rel)?;
    let rel_norm = ctx.rel_path(&abs);
    ctx.check_write(&rel_norm)?;
    let original = std::fs::read_to_string(&abs)?;
    let new_content =
        artifact::replace_fn_body(&original, &args.fn_name, &args.new_body)
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    ToolFailure::FnNotFound(args.fn_name.clone())
                } else {
                    ToolFailure::Syntax(format!("{e:#}"))
                }
            })?;
    let no_change = new_content == original;
    if !no_change {
        std::fs::write(&abs, new_content.as_bytes())?;
        ctx.record_written(rel_norm.clone());
        ctx.state.emit(UiEvent::FileChanged { path: rel_norm });
    }
    Ok(ReplaceFnBodyOk {
        bytes: new_content.len() as u64,
        no_change,
    })
}

// =====================================================================
// list_compiler_errors / read_compiler_error
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct ListCompilerErrArgs {}

#[derive(Serialize, Debug)]
pub struct CompilerErrSummary {
    pub id: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub message: String,
}

#[derive(Serialize, Debug)]
pub struct ListCompilerErrOk {
    pub errors: Vec<CompilerErrSummary>,
}

pub struct ListCompilerErrorsTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for ListCompilerErrorsTool {
    const NAME: &'static str = "list_compiler_errors";
    type Error = ToolFailure;
    type Args = ListCompilerErrArgs;
    type Output = ListCompilerErrOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List current cargo check / cargo test compiler errors.".into(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<ListCompilerErrOk, _>(e));
        }
        let errors = self
            .ctx
            .compiler_errors
            .iter()
            .map(|e| CompilerErrSummary {
                id: e.id.clone(),
                file: e.file.as_ref().map(|f| f.display().to_string()),
                line: e.line,
                message: first_line(&e.message),
            })
            .collect();
        self.ctx.record_result(Self::NAME, true);
        Ok(ListCompilerErrOk { errors })
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ReadCompilerErrArgs {
    pub error_id: String,
}

#[derive(Serialize, Debug)]
pub struct ReadCompilerErrOk {
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub raw: serde_json::Value,
}

pub struct ReadCompilerErrorTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for ReadCompilerErrorTool {
    const NAME: &'static str = "read_compiler_error";
    type Error = ToolFailure;
    type Args = ReadCompilerErrArgs;
    type Output = ReadCompilerErrOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Read a single compiler error in detail.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"error_id": {"type": "string"}},
                "required": ["error_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            return self.ctx.finish(Self::NAME, Err::<ReadCompilerErrOk, _>(e));
        }
        let r: Result<_, ToolFailure> = self
            .ctx
            .compiler_errors
            .iter()
            .find(|e| e.id == args.error_id)
            .ok_or_else(|| ToolFailure::Other(format!("error id '{}' not found", args.error_id)))
            .map(|e| ReadCompilerErrOk {
                message: e.message.clone(),
                file: e.file.as_ref().map(|f| f.display().to_string()),
                line: e.line,
                raw: e.raw.clone(),
            });
        self.ctx.finish(Self::NAME, r)
    }
}

// =====================================================================
// submit_verdict (judge role only)
// =====================================================================

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
            description: "Record your verdict on whether the task is now \
                satisfactorily completed. Pass satisfactory=true to accept \
                the work, or false with a reason to trigger another \
                critique→revise round."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "satisfactory": {"type": "boolean"},
                    "reason": {"type": "string", "description": "free-text rationale; required when satisfactory=false"}
                },
                "required": ["satisfactory"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Err(e) = self.ctx.record_call_and_check_loop(Self::NAME, &args) {
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

// =====================================================================
// cargo_check / cargo_test / cargo_test_no_run (live diagnostics)
// =====================================================================

#[derive(Deserialize, Serialize, Debug)]
pub struct CargoArgs {
    /// Optional crate / package name (workspace member). If unset, cargo
    /// runs against the whole workspace.
    #[serde(default)]
    pub package: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CargoTestArgs {
    /// Optional crate / package name (workspace member).
    #[serde(default)]
    pub package: Option<String>,
    /// Optional test-name filter — passed as the positional arg to cargo
    /// test, which libtest treats as a substring match against test names.
    /// Use this to iterate on a single failing test rather than re-running
    /// the whole suite. Examples: `tests::add_handles_negative`, `add::`,
    /// `it_works`.
    #[serde(default)]
    pub test_filter: Option<String>,
    /// Optional list of explicit test name filters; cargo will run any
    /// test whose name matches ANY of the filters as a substring. Mutually
    /// preferred over `test_filter` when supplied.
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
    /// First N errors (capped to avoid swamping the model context).
    pub errors: Vec<CargoErrorBrief>,
    /// Total error count if it exceeded the cap.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    pub total_errors: usize,
    /// Last few lines of stderr — useful when the failure is non-structured
    /// (linker errors, build script panics, etc.).
    pub stderr_tail: String,
    pub elapsed_ms: u64,
    pub command: String,
}

const MAX_ERRORS_RETURNED: usize = 8;
const MAX_STDERR_TAIL_BYTES: usize = 2048;
const MAX_ERROR_MESSAGE_BYTES: usize = 1200;

async fn run_cargo_gate(
    ctx: &TaskCtx,
    kind: crate::gate::GateKind,
    package: Option<&str>,
) -> Result<CargoOk, ToolFailure> {
    run_cargo_gate_filtered(ctx, kind, package, &[]).await
}

/// Like `run_cargo_gate`, but for `cargo test` invocations we additionally
/// pass `-- <test_filters...>` after the cargo args. libtest treats each
/// positional arg as a substring filter and runs any test whose name
/// matches any of them.
async fn run_cargo_gate_filtered(
    ctx: &TaskCtx,
    kind: crate::gate::GateKind,
    package: Option<&str>,
    test_filters: &[String],
) -> Result<CargoOk, ToolFailure> {
    let start = std::time::Instant::now();
    // Build args. We piggyback on `gate::GateKind::args` for the base, then
    // append `-p <package>` and (for test kinds) `-- <filters...>`.
    let mut args: Vec<String> = kind.args().iter().map(|s| s.to_string()).collect();
    if let Some(p) = package {
        args.push("-p".to_string());
        args.push(p.to_string());
    }
    if !test_filters.is_empty()
        && matches!(kind, crate::gate::GateKind::Test | crate::gate::GateKind::TestNoRun)
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
    let output = cmd
        .output()
        .await
        .map_err(|e| ToolFailure::Other(format!("spawning cargo failed: {e}")))?;
    // Re-parse via the shared run_gate logic by calling it directly.
    // Simpler: replicate the result shape ourselves so we control the
    // returned bytes count.
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
    let command = format!("cargo {}", args.join(" "));
    Ok(CargoOk {
        passed: outcome.passed,
        errors,
        truncated,
        total_errors,
        stderr_tail,
        elapsed_ms: start.elapsed().as_millis() as u64,
        command,
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Run `cargo check` on the workdir (or a specific package via `package`) \
                and return structured diagnostics. Use this BEFORE finishing your turn to verify \
                what you wrote actually compiles. Returns at most 8 errors plus a stderr tail."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "optional workspace member name"}
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => {
                run_cargo_gate(
                    &self.ctx,
                    crate::gate::GateKind::Check,
                    args.package.as_deref(),
                )
                .await
            }
        };
        self.ctx.finish(Self::NAME, r)
    }
}

pub struct CargoTestTool {
    pub ctx: Arc<TaskCtx>,
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

impl Tool for CargoTestTool {
    const NAME: &'static str = "cargo_test";
    type Error = ToolFailure;
    type Args = CargoTestArgs;
    type Output = CargoOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Run `cargo test --no-fail-fast` and return structured diagnostics \
                (compile errors and runtime test failures). To iterate on a single failing \
                test, pass `test_filter` (substring match) or `test_filters` (multiple \
                substrings). Without filters, runs the whole suite."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "optional workspace member"},
                    "test_filter": {"type": "string", "description": "single substring filter passed to libtest"},
                    "test_filters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "multiple substring filters; takes precedence over test_filter"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let filters = collect_filters(&args);
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => {
                run_cargo_gate_filtered(
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Run `cargo test --no-run` to verify that test files COMPILE without \
                running them. Useful in the Test phase where bodies are still stubs and tests \
                will fail at runtime, but must at least compile. `test_filter` / `test_filters` \
                limit which tests are compiled."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "optional workspace member"},
                    "test_filter": {"type": "string"},
                    "test_filters": {"type": "array", "items": {"type": "string"}}
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let filters = collect_filters(&args);
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
            Err(e) => Err(e),
            Ok(()) => {
                run_cargo_gate_filtered(
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

// ---- cargo_clippy ----

pub struct CargoClippyTool {
    pub ctx: Arc<TaskCtx>,
}

impl Tool for CargoClippyTool {
    const NAME: &'static str = "cargo_clippy";
    type Error = ToolFailure;
    type Args = CargoArgs;
    type Output = CargoOk;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Run `cargo clippy` and return structured diagnostics. Clippy reports \
                cargo's `compiler-message` errors AND its own lint warnings/errors. Use this \
                to catch idiomatic Rust issues that the type checker won't flag. May be slow \
                on first invocation if clippy isn't cached."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "optional workspace member"}
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let r = match self.ctx.record_call_and_check_loop(Self::NAME, &args) {
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
    // Treat clippy warnings as errors so they become first-class citizens
    // of the gate-style result and the model sees them.
    args.push("--".to_string());
    args.push("-D".to_string());
    args.push("warnings".to_string());

    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(&args)
        .current_dir(&ctx.workdir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env("CARGO_TERM_COLOR", "never");
    let output = cmd
        .output()
        .await
        .map_err(|e| ToolFailure::Other(format!("spawning cargo clippy failed: {e}")))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    // Clippy emits the same compiler-message JSON shape as cargo check, so
    // the existing parser handles it. Use Check-shaped parsing.
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
