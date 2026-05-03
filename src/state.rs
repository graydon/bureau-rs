//! Runtime state shared between engine and web UI.

use crate::graph::{NodeGraph, NodeId, Stage};
use crate::tools::{JudgeVerdict, Role, TranscriptEntry};
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerState {
    Idle,
    Running,
    Paused,
    Done,
    Stopped,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

/// One unit of orchestrator work — a (node, stage) advancement that runs
/// through the actor → critic → reviser → judge cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineTask {
    pub id: Uuid,
    pub node_id: NodeId,
    pub node_name: String,
    pub stage: Stage,
    pub status: TaskStatus,
    pub model: String,
    #[serde(default)]
    pub transcript: Vec<TranscriptEntry>,
    #[serde(default)]
    pub cost: TokenUsage,
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub final_verdict: Option<JudgeVerdict>,
    #[serde(default)]
    pub retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub at: DateTime<Utc>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    pub workdir: PathBuf,
    pub config_dir: PathBuf,
    pub project_name: String,
    pub graph: NodeGraph,
    pub tasks: IndexMap<Uuid, EngineTask>,
    pub scheduler: SchedulerState,
    pub total_cost: TokenUsage,
    pub estimated_cost_usd: f64,
    pub started_at: DateTime<Utc>,
    pub last_event_at: DateTime<Utc>,
    pub history: Vec<HistoryEntry>,
}

impl EngineState {
    pub fn new(workdir: PathBuf, config_dir: PathBuf, project_name: String) -> Self {
        let now = Utc::now();
        Self {
            workdir,
            config_dir,
            project_name,
            graph: NodeGraph::new(),
            tasks: IndexMap::new(),
            scheduler: SchedulerState::Idle,
            total_cost: TokenUsage::default(),
            estimated_cost_usd: 0.0,
            started_at: now,
            last_event_at: now,
            history: Vec::new(),
        }
    }

    pub fn note(&mut self, msg: impl Into<String>) {
        self.history.push(HistoryEntry {
            at: Utc::now(),
            message: msg.into(),
        });
        if self.history.len() > 1000 {
            self.history.drain(..self.history.len() - 1000);
        }
        self.last_event_at = Utc::now();
    }
}

/// Event broadcast over SSE to UI clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiEvent {
    SchedulerStateChanged {
        state: SchedulerState,
    },
    NodeChanged {
        id: NodeId,
    },
    TaskCreated {
        task: EngineTask,
    },
    TaskStatusChanged {
        id: Uuid,
        status: TaskStatus,
    },
    TaskUpdated {
        task: EngineTask,
    },
    TranscriptAppended {
        task_id: Uuid,
        entry: TranscriptEntry,
        role: Role,
    },
    TaskCost {
        task_id: Uuid,
        cost: TokenUsage,
        total: TokenUsage,
        estimated_usd: f64,
    },
    HistoryAppended {
        entry: HistoryEntry,
    },
    FileChanged {
        path: PathBuf,
    },
    Heartbeat {
        at: DateTime<Utc>,
    },
}

#[derive(Clone)]
pub struct StateHandle {
    inner: Arc<Mutex<EngineState>>,
    tx: broadcast::Sender<UiEvent>,
}

impl StateHandle {
    pub fn new(state: EngineState) -> Self {
        let (tx, _rx) = broadcast::channel(16384);
        Self {
            inner: Arc::new(Mutex::new(state)),
            tx,
        }
    }

    pub fn read<R>(&self, f: impl FnOnce(&EngineState) -> R) -> R {
        f(&self.inner.lock())
    }

    pub fn write<R>(&self, f: impl FnOnce(&mut EngineState) -> R) -> R {
        let mut g = self.inner.lock();
        let r = f(&mut g);
        g.last_event_at = Utc::now();
        r
    }

    pub fn snapshot(&self) -> EngineState {
        self.inner.lock().clone()
    }

    pub fn replace(&self, new: EngineState) {
        *self.inner.lock() = new;
    }

    pub fn emit(&self, ev: UiEvent) {
        let _ = self.tx.send(ev);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<UiEvent> {
        self.tx.subscribe()
    }
}
