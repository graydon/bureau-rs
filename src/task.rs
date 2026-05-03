use crate::phase::Phase;
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

impl TaskStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskStatus::Done | TaskStatus::Failed | TaskStatus::Skipped
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

impl TokenUsage {
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }

    pub fn cost_usd(&self, input_per_mtok: f64, output_per_mtok: f64) -> f64 {
        let billable_input =
            self.input_tokens.saturating_sub(self.cached_input_tokens) as f64
                + self.cache_creation_input_tokens as f64 * 1.25;
        let cached = self.cached_input_tokens as f64 * 0.1;
        let input_cost = (billable_input + cached) * input_per_mtok / 1_000_000.0;
        let output_cost = self.output_tokens as f64 * output_per_mtok / 1_000_000.0;
        input_cost + output_cost
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Actor,
    Critic,
    Reviser,
    Judge,
}

impl Default for Role {
    fn default() -> Self {
        Role::Actor
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub timestamp: DateTime<Utc>,
    pub kind: TranscriptKind,
    pub content: String,
    /// Which role produced this entry. Defaults to Actor for backward
    /// compatibility with checkpoints predating the critique cycle.
    #[serde(default)]
    pub role: Role,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptKind {
    System,
    UserPrompt,
    AssistantText,
    ToolCall {
        tool: String,
        args: String,
    },
    ToolResult {
        tool: String,
        ok: bool,
        #[serde(default)]
        error: Option<String>,
        /// JSON-stringified successful Output payload returned to the LLM.
        #[serde(default)]
        output: Option<String>,
    },
    Note,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskDecl {
    pub description: String,
    #[serde(default)]
    pub read_files: Vec<PathBuf>,
    #[serde(default)]
    pub write_files: Vec<PathBuf>,
    #[serde(default)]
    pub spec_sections: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub parent: Option<Uuid>,
    pub phase: Phase,
    #[serde(default)]
    pub depth: u32,
    pub description: String,
    #[serde(default)]
    pub read_files: Vec<PathBuf>,
    #[serde(default)]
    pub write_files: Vec<PathBuf>,
    #[serde(default)]
    pub spec_sections: Vec<String>,
    #[serde(default)]
    pub subtasks: Vec<Uuid>,
    pub status: TaskStatus,
    #[serde(default)]
    pub transcript: Vec<TranscriptEntry>,
    #[serde(default)]
    pub cost: TokenUsage,
    pub worktree: Option<PathBuf>,
    pub model: Option<String>,
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub retries: u32,
    pub error: Option<String>,
}

impl Task {
    pub fn new(phase: Phase, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            parent: None,
            phase,
            depth: 0,
            description: description.into(),
            read_files: Vec::new(),
            write_files: Vec::new(),
            spec_sections: Vec::new(),
            subtasks: Vec::new(),
            status: TaskStatus::Pending,
            transcript: Vec::new(),
            cost: TokenUsage::default(),
            worktree: None,
            model: None,
            started_at: None,
            finished_at: None,
            retries: 0,
            error: None,
        }
    }

    pub fn from_decl(parent: Uuid, phase: Phase, depth: u32, decl: SubtaskDecl) -> Self {
        let mut t = Self::new(phase, decl.description);
        t.parent = Some(parent);
        t.depth = depth;
        t.read_files = decl.read_files;
        t.write_files = decl.write_files;
        t.spec_sections = decl.spec_sections;
        t
    }

    pub fn elapsed_ms(&self) -> Option<i64> {
        let start = self.started_at?;
        let end = self.finished_at.unwrap_or_else(Utc::now);
        Some((end - start).num_milliseconds())
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TaskGraph {
    pub roots: Vec<Uuid>,
    pub tasks: IndexMap<Uuid, Task>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_root(&mut self, task: Task) -> Uuid {
        let id = task.id;
        self.roots.push(id);
        self.tasks.insert(id, task);
        id
    }

    pub fn insert_child(&mut self, parent: Uuid, mut task: Task) -> Uuid {
        task.parent = Some(parent);
        let id = task.id;
        if let Some(p) = self.tasks.get_mut(&parent) {
            p.subtasks.push(id);
        }
        self.tasks.insert(id, task);
        id
    }

    pub fn get(&self, id: Uuid) -> Option<&Task> {
        self.tasks.get(&id)
    }

    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut Task> {
        self.tasks.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Task> {
        self.tasks.values()
    }

    pub fn count_by_status(&self) -> StatusCounts {
        let mut c = StatusCounts::default();
        for t in self.tasks.values() {
            match t.status {
                TaskStatus::Pending => c.pending += 1,
                TaskStatus::Running => c.running += 1,
                TaskStatus::Done => c.done += 1,
                TaskStatus::Failed => c.failed += 1,
                TaskStatus::Skipped => c.skipped += 1,
            }
        }
        c
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct StatusCounts {
    pub pending: u32,
    pub running: u32,
    pub done: u32,
    pub failed: u32,
    pub skipped: u32,
}
