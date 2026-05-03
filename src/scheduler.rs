//! Async task scheduler with per-file read/write locks. Multiple readers may
//! hold a shared lock concurrently; a writer takes an exclusive lock that
//! blocks all readers and writers of the same path.

use crate::agent::AgentRunner;
use crate::config::Config;
use crate::gate::{GateKind, GateOutcome, run_gate};
use crate::phase::Phase;
use crate::state::{SchedulerState, StateHandle, UiEvent};
use crate::task::{Task, TaskStatus, TranscriptEntry, TranscriptKind};
use crate::tools::CompilerError;
use crate::worktree::WorktreePool;
use anyhow::{Result, anyhow};
use chrono::Utc;
use parking_lot::Mutex as PMutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Notify, Semaphore};
use uuid::Uuid;

/// One element of a per-file lock manager. Allows multiple readers or a
/// single writer.
#[derive(Debug, Default)]
struct FileLockSlot {
    readers: u32,
    writer: bool,
}

/// Internal per-file lock manager. Tasks acquire locks for their declared
/// read/write sets atomically (all-or-nothing) via `try_acquire` to avoid
/// deadlocks between competing tasks.
pub struct FileLockManager {
    inner: PMutex<HashMap<PathBuf, FileLockSlot>>,
    notify: Notify,
}

impl FileLockManager {
    pub fn new() -> Self {
        Self {
            inner: PMutex::new(HashMap::new()),
            notify: Notify::new(),
        }
    }

    fn try_acquire(&self, reads: &[PathBuf], writes: &[PathBuf]) -> bool {
        let mut g = self.inner.lock();
        // First, check if all locks are available.
        for w in writes {
            let slot = g.entry(w.clone()).or_default();
            if slot.writer || slot.readers > 0 {
                return false;
            }
        }
        for r in reads {
            if writes.contains(r) {
                continue;
            }
            let slot = g.entry(r.clone()).or_default();
            if slot.writer {
                return false;
            }
        }
        // Acquire all.
        for w in writes {
            let slot = g.entry(w.clone()).or_default();
            slot.writer = true;
        }
        for r in reads {
            if writes.contains(r) {
                continue;
            }
            let slot = g.entry(r.clone()).or_default();
            slot.readers += 1;
        }
        true
    }

    fn release(&self, reads: &[PathBuf], writes: &[PathBuf]) {
        {
            let mut g = self.inner.lock();
            for w in writes {
                if let Some(slot) = g.get_mut(w) {
                    slot.writer = false;
                }
            }
            for r in reads {
                if writes.contains(r) {
                    continue;
                }
                if let Some(slot) = g.get_mut(r) {
                    if slot.readers > 0 {
                        slot.readers -= 1;
                    }
                }
            }
        }
        self.notify.notify_waiters();
    }

    /// Acquire all of `reads` and `writes`; await on the wait set until they
    /// become available.
    pub async fn acquire(&self, reads: Vec<PathBuf>, writes: Vec<PathBuf>) -> LockGuard<'_> {
        loop {
            if self.try_acquire(&reads, &writes) {
                break;
            }
            self.notify.notified().await;
        }
        LockGuard {
            mgr: self,
            reads,
            writes,
            released: false,
        }
    }
}

pub struct LockGuard<'a> {
    mgr: &'a FileLockManager,
    reads: Vec<PathBuf>,
    writes: Vec<PathBuf>,
    released: bool,
}

impl<'a> Drop for LockGuard<'a> {
    fn drop(&mut self) {
        if !self.released {
            self.mgr.release(&self.reads, &self.writes);
        }
    }
}

/// Top-level orchestrator.
pub struct Orchestrator {
    pub config: Arc<Config>,
    pub state: StateHandle,
    pub agent: Arc<AgentRunner>,
    pub workspace: Arc<crate::worktree::Workspace>,
    pub pool: Arc<WorktreePool>,
    pub locks: Arc<FileLockManager>,
    pub semaphore: Arc<Semaphore>,
}

impl Orchestrator {
    pub fn new(
        config: Arc<Config>,
        state: StateHandle,
        workspace: Arc<crate::worktree::Workspace>,
    ) -> Result<Self> {
        let agent = Arc::new(AgentRunner::new(config.clone(), state.clone())?);
        let pool = Arc::new(WorktreePool::new(workspace.clone())?);
        let semaphore = Arc::new(Semaphore::new(config.limits.max_parallel_tasks));
        Ok(Self {
            config,
            state,
            agent,
            workspace,
            pool,
            locks: Arc::new(FileLockManager::new()),
            semaphore,
        })
    }

    /// Drive the entire pipeline from `start` through Done.
    pub async fn run(self: Arc<Self>, start: Phase) -> Result<()> {
        self.state.write(|s| {
            s.scheduler = SchedulerState::Running;
        });
        self.state
            .emit(UiEvent::SchedulerStateChanged { state: SchedulerState::Running });

        let mut current = start;
        loop {
            self.state.write(|s| {
                s.phase = current;
                s.note(format!("entering phase {}", current));
            });
            self.state.emit(UiEvent::PhaseChanged { phase: current });

            // Run the phase, retrying on gate failure up to max_retries.
            let phase_cfg = self.config.phase_config(current);
            let max_attempts = phase_cfg.max_retries.max(1);
            let mut passed = false;
            let mut last_gate: Option<GateOutcome> = None;
            for attempt in 1..=max_attempts {
                self.state.write(|s| {
                    s.note(format!(
                        "phase {} attempt {}/{}",
                        current, attempt, max_attempts
                    ))
                });
                // Mark any prior-attempt failed tasks in this phase as
                // Skipped so they don't poison the next attempt's gate
                // check.
                self.state.write(|s| {
                    for t in s.graph.tasks.values_mut() {
                        if t.phase == current && t.status == TaskStatus::Failed {
                            t.status = TaskStatus::Skipped;
                        }
                    }
                });
                self.clone().run_phase(current).await?;
                let gate = self.run_phase_gate(current).await?;
                if gate.passed {
                    passed = true;
                    last_gate = Some(gate);
                    break;
                }
                // Surface the actual error messages so the user (and the
                // next phase attempt's transcripts) can see WHY the gate
                // failed. Previously we only said "<N> errors" which made
                // legitimate retries look mysterious.
                let summary = summarize_gate_errors(&gate, 5);
                tracing::warn!(
                    "phase {} attempt {}/{} did not pass: {} error(s):\n{}",
                    current,
                    attempt,
                    max_attempts,
                    gate.errors.len(),
                    summary
                );
                self.state.write(|s| {
                    s.note(format!(
                        "phase {} attempt {}/{} did not pass ({} errors):\n{}",
                        current,
                        attempt,
                        max_attempts,
                        gate.errors.len(),
                        summary
                    ))
                });
                // Append errors to the orchestrator history so they show up
                // in the Issues panel as gate-attributed problems, with a
                // synthetic task_id so they group separately from per-task
                // tool failures.
                last_gate = Some(gate);
            }

            // Commit only if the workdir actually changed (skips empty commits).
            self.workspace
                .commit_if_dirty(&format!("phase {} complete", current))
                .await
                .ok();

            self.auto_checkpoint(&format!("after-{}", current));

            if !passed {
                let n_err = last_gate.as_ref().map(|g| g.errors.len()).unwrap_or(0);
                self.state.write(|s| {
                    s.scheduler = SchedulerState::Stopped;
                    s.note(format!(
                        "halting pipeline: phase {} failed gate after {} attempts ({} errors)",
                        current, max_attempts, n_err
                    ));
                });
                self.state.emit(UiEvent::SchedulerStateChanged {
                    state: SchedulerState::Stopped,
                });
                return Err(anyhow!(
                    "phase {} failed gate after {} attempts ({} errors); halting pipeline",
                    current,
                    max_attempts,
                    n_err
                ));
            }

            match current.next() {
                Some(next) => {
                    current = next;
                }
                None => {
                    self.state.write(|s| {
                        s.scheduler = SchedulerState::Done;
                        s.note("all phases complete");
                    });
                    self.state.emit(UiEvent::SchedulerStateChanged {
                        state: SchedulerState::Done,
                    });
                    self.auto_checkpoint("done");
                    break;
                }
            }
        }
        Ok(())
    }

    fn auto_checkpoint(&self, label: &str) {
        let dir = self.workspace.root.join(".bureau").join("checkpoints");
        let snap = self.state.snapshot();
        match crate::checkpoint::save(&snap, &dir) {
            Ok(path) => {
                tracing::info!(
                    "checkpoint saved: {} ({label})",
                    path.display()
                );
                let _ = crate::checkpoint::save_latest(&snap, &dir);
            }
            Err(e) => tracing::warn!("checkpoint failed ({label}): {e:#}"),
        }
    }

    async fn run_phase(self: Arc<Self>, phase: Phase) -> Result<()> {
        // Seed root task: root decomposition for the phase
        let root = make_root_task(phase, &self.config);
        let root_id = self.state.write(|s| s.graph.insert_root(root.clone()));
        self.state.emit(UiEvent::TaskCreated { task: root.clone() });

        // Sequential queue: run in waves. We process everything currently
        // pending, awaiting their completion before checking for newly
        // emitted subtasks. Tasks within a wave run concurrently subject to
        // file-lock interference and the global semaphore.
        let mut queue: Vec<Uuid> = vec![root_id];
        while let Some(_) = queue.first() {
            let mut handles = Vec::new();
            for tid in std::mem::take(&mut queue) {
                let this = self.clone();
                let h = tokio::spawn(async move { this.run_one_task(tid).await });
                handles.push(h);
            }
            let mut new_subtasks: Vec<Uuid> = Vec::new();
            for h in handles {
                match h.await {
                    Ok(Ok(ids)) => new_subtasks.extend(ids),
                    Ok(Err(e)) => {
                        tracing::error!("task error: {e:#}");
                    }
                    Err(je) => {
                        tracing::error!("task join error: {je}");
                    }
                }
            }
            queue.extend(new_subtasks);
        }
        Ok(())
    }

    /// Run a single task: allocate worktree, acquire locks, invoke agent,
    /// merge writes back, release locks, store emitted subtasks. Returns
    /// IDs of any newly created child tasks.
    async fn run_one_task(self: Arc<Self>, task_id: Uuid) -> Result<Vec<Uuid>> {
        let _permit = self.semaphore.clone().acquire_owned().await?;
        // Snapshot the task
        let task = self
            .state
            .read(|s| s.graph.get(task_id).cloned())
            .ok_or_else(|| anyhow!("task {task_id} not found"))?;

        // Dependency check passed for declared sets.
        let reads = task.read_files.clone();
        let writes = task.write_files.clone();
        let _guard = self.locks.acquire(reads.clone(), writes.clone()).await;

        // Allocate worktree (a copy of the workdir)
        let wt = self.pool.allocate()?;

        self.state.write(|s| {
            if let Some(t) = s.graph.get_mut(task_id) {
                t.status = TaskStatus::Running;
                t.started_at = Some(Utc::now());
                t.worktree = Some(wt.path.clone());
                t.model = Some(self.config.phase_config(t.phase).model.clone());
                s.running_tasks.insert(task_id);
            }
        });
        self.state.emit(UiEvent::TaskStatusChanged {
            id: task_id,
            status: TaskStatus::Running,
        });

        // Compiler errors are only relevant in Debug
        let compiler_errors: Vec<CompilerError> = if task.phase == Phase::Debug {
            let outcome = run_gate(&self.workspace.root, GateKind::Test).await?;
            outcome.errors
        } else {
            Vec::new()
        };

        let outcome = match self
            .agent
            .run_task(&task, wt.path.clone(), compiler_errors)
            .await
        {
            Ok(o) => o,
            Err(e) => {
                let msg = format!("{:#}", e);
                self.state.write(|s| {
                    if let Some(t) = s.graph.get_mut(task_id) {
                        t.status = TaskStatus::Failed;
                        t.finished_at = Some(Utc::now());
                        t.error = Some(msg.clone());
                        let entry = TranscriptEntry {
                            timestamp: Utc::now(),
                            kind: TranscriptKind::Error,
                            content: msg.clone(),
                            role: Default::default(),
                        };
                        t.transcript.push(entry.clone());
                        s.running_tasks.remove(&task_id);
                    }
                });
                self.state.emit(UiEvent::TaskStatusChanged {
                    id: task_id,
                    status: TaskStatus::Failed,
                });
                let _ = self.pool.release(wt);
                return Err(anyhow!("agent failed: {msg}"));
            }
        };

        // Merge writes back and commit
        let merged = match self.pool.merge_back(&wt, &outcome.written_files).await {
            Ok(m) => {
                tracing::info!(
                    task = %task_id,
                    phase = %task.phase,
                    files = m.len(),
                    "merged {} files back to workdir",
                    m.len()
                );
                m
            }
            Err(e) => {
                tracing::error!(task = %task_id, "merge_back failed: {e:#}");
                self.state
                    .write(|s| s.note(format!("merge_back failed for task {task_id}: {e:#}")));
                Vec::new()
            }
        };
        match self
            .workspace
            .commit_if_dirty(&format!(
                "[{}] {} (task {})",
                task.phase,
                truncate(&task.description, 60),
                task_id
            ))
            .await
        {
            Ok(Some(_)) => {}
            Ok(None) => tracing::debug!(task = %task_id, "no changes to commit"),
            Err(e) => tracing::warn!(task = %task_id, "commit failed: {e:#}"),
        }
        for p in &merged {
            self.state.emit(UiEvent::FileChanged { path: p.clone() });
        }

        // Insert emitted subtasks into the graph, subject to depth + count caps.
        let mut new_ids = Vec::new();
        let depth_cap = self.config.limits.max_subtask_depth;
        let task_count_cap = self.config.limits.max_tasks_per_phase;
        let child_depth = task.depth + 1;
        let phase_task_count = self
            .state
            .read(|s| s.graph.iter().filter(|t| t.phase == task.phase).count());
        let dropped_for_depth = child_depth > depth_cap;
        let mut accepted = 0usize;
        let mut dropped_for_count = 0usize;
        if !outcome.emitted_subtasks.is_empty() && dropped_for_depth {
            self.state.write(|s| {
                s.note(format!(
                    "task {task_id}: dropping {} subtasks (depth {} exceeds cap {})",
                    outcome.emitted_subtasks.len(),
                    child_depth,
                    depth_cap
                ))
            });
        }
        for decl in outcome.emitted_subtasks {
            if dropped_for_depth {
                break;
            }
            if phase_task_count + accepted >= task_count_cap {
                dropped_for_count += 1;
                continue;
            }
            let child = Task::from_decl(task_id, task.phase, child_depth, decl);
            let cid = child.id;
            let task_clone = self.state.write(|s| {
                s.graph.insert_child(task_id, child.clone());
                s.graph.get(cid).cloned().unwrap_or(child)
            });
            self.state.emit(UiEvent::TaskCreated { task: task_clone });
            new_ids.push(cid);
            accepted += 1;
        }
        if dropped_for_count > 0 {
            self.state.write(|s| {
                s.note(format!(
                    "task {task_id}: dropped {dropped_for_count} subtasks (per-phase cap {task_count_cap} reached)"
                ))
            });
        }

        // If the critique cycle ran and the final judge verdict was
        // unsatisfactory, mark the task as Failed so the phase gate refuses
        // to advance.
        let unsatisfactory = outcome
            .final_verdict
            .as_ref()
            .map(|v| !v.satisfactory)
            .unwrap_or(false);
        let final_status = if unsatisfactory {
            TaskStatus::Failed
        } else {
            TaskStatus::Done
        };
        let verdict_msg = outcome
            .final_verdict
            .as_ref()
            .map(|v| format!("judge verdict: {}", truncate(&v.reason, 200)));
        self.state.write(|s| {
            if let Some(t) = s.graph.get_mut(task_id) {
                t.status = final_status;
                t.finished_at = Some(Utc::now());
                if unsatisfactory {
                    t.error = Some(
                        verdict_msg
                            .clone()
                            .unwrap_or_else(|| "judge rejected".to_string()),
                    );
                }
                s.running_tasks.remove(&task_id);
            }
        });
        self.state.emit(UiEvent::TaskStatusChanged {
            id: task_id,
            status: final_status,
        });
        if let Some(msg) = verdict_msg {
            self.state.write(|s| {
                s.note(format!(
                    "task {task_id} {}: {msg}",
                    if unsatisfactory {
                        "rejected by judge"
                    } else {
                        "approved by judge"
                    }
                ))
            });
        }

        let _ = self.pool.release(wt);
        Ok(new_ids)
    }

    async fn run_phase_gate(&self, phase: Phase) -> Result<GateOutcome> {
        // Cross-cutting check: any task in the current phase ended Failed
        // (typically because the judge rejected it after the critique cycle).
        // That alone fails the gate.
        let failed_tasks: Vec<(uuid::Uuid, String)> = self.state.read(|s| {
            s.graph
                .iter()
                .filter(|t| t.phase == phase && t.status == TaskStatus::Failed)
                .map(|t| {
                    (
                        t.id,
                        t.error.clone().unwrap_or_else(|| "task failed".into()),
                    )
                })
                .collect()
        });
        if !failed_tasks.is_empty() {
            let errors: Vec<crate::tools::CompilerError> = failed_tasks
                .into_iter()
                .map(|(tid, msg)| crate::tools::CompilerError {
                    id: format!("task-{}", tid.simple()),
                    file: None,
                    line: None,
                    message: format!("task {tid}: {msg}"),
                    raw: serde_json::json!({"reason": "task-failed"}),
                })
                .collect();
            return Ok(GateOutcome {
                passed: false,
                errors,
                stdout: String::new(),
                stderr: String::new(),
            });
        }
        match phase {
            Phase::Spec => {
                // Spec phase passes only if at least one spec/*.md file exists.
                let spec_dir = self.workspace.root.join("spec");
                let any_section = spec_dir
                    .read_dir()
                    .ok()
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok())
                    .any(|e| {
                        e.path().extension().and_then(|x| x.to_str()) == Some("md")
                    });
                if any_section {
                    return Ok(GateOutcome::empty_ok());
                }
                return Ok(GateOutcome {
                    passed: false,
                    errors: vec![crate::tools::CompilerError {
                        id: "spec-empty".into(),
                        file: None,
                        line: None,
                        message: "spec phase produced no spec/*.md files".into(),
                        raw: serde_json::json!({"reason": "spec-empty"}),
                    }],
                    stdout: String::new(),
                    stderr: String::new(),
                });
            }
            _ => {}
        }
        let kind = match phase {
            Phase::Spec => unreachable!(),
            Phase::Interface => GateKind::Check,
            Phase::Test => GateKind::TestNoRun,
            Phase::Impl => GateKind::Test,
            Phase::Debug => GateKind::Test,
            Phase::Opt => GateKind::Test,
        };
        if !self.workspace.root.join("Cargo.toml").exists() {
            return Ok(GateOutcome {
                passed: false,
                errors: vec![crate::tools::CompilerError {
                    id: "no-cargo-toml".into(),
                    file: None,
                    line: None,
                    message: format!(
                        "phase {} has no Cargo.toml in workdir; nothing to check",
                        phase
                    ),
                    raw: serde_json::json!({"reason": "no-cargo-toml"}),
                }],
                stdout: String::new(),
                stderr: String::new(),
            });
        }
        // Phases after Interface require at least one .rs file; otherwise
        // there's nothing to test/optimize.
        if matches!(
            phase,
            Phase::Test | Phase::Impl | Phase::Debug | Phase::Opt
        ) {
            let any_rs = walkdir::WalkDir::new(&self.workspace.root)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
                .any(|e| {
                    let p = e.path();
                    if p.components().any(|c| {
                        let s = c.as_os_str().to_string_lossy();
                        s == ".git" || s == "target" || s == ".bureau"
                    }) {
                        return false;
                    }
                    p.extension().and_then(|x| x.to_str()) == Some("rs")
                });
            if !any_rs {
                return Ok(GateOutcome {
                    passed: false,
                    errors: vec![crate::tools::CompilerError {
                        id: "no-rs-files".into(),
                        file: None,
                        line: None,
                        message: format!("phase {} produced no .rs files", phase),
                        raw: serde_json::json!({"reason": "no-rs-files"}),
                    }],
                    stdout: String::new(),
                    stderr: String::new(),
                });
            }
        }
        run_gate(&self.workspace.root, kind).await
    }
}

fn make_root_task(phase: Phase, config: &Config) -> Task {
    let description = match phase {
        Phase::Spec => "Produce a structured specification for the problem.".to_string(),
        Phase::Interface => "Produce Rust types, signatures, and trait stubs.".to_string(),
        Phase::Test => "Produce integration tests against the interface.".to_string(),
        Phase::Impl => "Fill in function bodies to make tests pass.".to_string(),
        Phase::Debug => "Fix any remaining compiler errors and test failures.".to_string(),
        Phase::Opt => "Make targeted performance improvements.".to_string(),
    };
    let mut t = Task::new(phase, description);
    t.model = Some(config.phase_config(phase).model.clone());
    // Root task gets a directory-prefix write_set broad enough to cover both
    // single-crate and workspace layouts. The phase guard at write time is
    // the actual filter (e.g. Test phase only accepts test-shaped paths
    // even when its write_set includes `src/`). Subtasks are expected to
    // narrow these down to specific files or member crates.
    let workspace = matches!(
        config.layout.kind,
        crate::config::WorkspaceLayout::Workspace
    );
    match phase {
        Phase::Spec => {
            t.write_files.push(PathBuf::from("spec/"));
        }
        Phase::Interface => {
            t.write_files.push(PathBuf::from("src/"));
            t.write_files.push(PathBuf::from("Cargo.toml"));
            if workspace {
                t.write_files.push(PathBuf::from("crates/"));
            }
        }
        Phase::Test => {
            t.write_files.push(PathBuf::from("tests/"));
            // Internal tests live under src/.
            t.write_files.push(PathBuf::from("src/"));
            if workspace {
                t.write_files.push(PathBuf::from("crates/"));
            }
        }
        Phase::Impl => {
            t.write_files.push(PathBuf::from("src/"));
            if workspace {
                t.write_files.push(PathBuf::from("crates/"));
            }
        }
        Phase::Debug => {
            t.write_files.push(PathBuf::from("src/"));
            t.write_files.push(PathBuf::from("tests/"));
            if workspace {
                t.write_files.push(PathBuf::from("crates/"));
            }
        }
        Phase::Opt => {
            t.write_files.push(PathBuf::from("src/"));
            if workspace {
                t.write_files.push(PathBuf::from("crates/"));
            }
        }
    }
    t
}

/// Render a gate outcome's errors as a short, human-readable bullet list.
/// Caps the per-error length and the count so a 200-error cargo run
/// doesn't blow up the history pane.
fn summarize_gate_errors(gate: &GateOutcome, max_errors: usize) -> String {
    if gate.errors.is_empty() {
        return "(no specific errors recorded)".to_string();
    }
    let mut out = String::new();
    let n = gate.errors.len();
    for e in gate.errors.iter().take(max_errors) {
        let loc = match (&e.file, e.line) {
            (Some(f), Some(l)) => format!("{}:{}", f.display(), l),
            (Some(f), None) => f.display().to_string(),
            _ => e.id.clone(),
        };
        // First line of the message keeps things compact.
        let first = e.message.lines().next().unwrap_or("").trim();
        out.push_str(&format!("  - [{loc}] {}\n", truncate(first, 200)));
    }
    if n > max_errors {
        out.push_str(&format!("  - ... and {} more\n", n - max_errors));
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}
