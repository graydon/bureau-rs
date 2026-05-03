use crate::phase::Phase;
use crate::task::{StatusCounts, Task, TaskGraph, TaskStatus, TokenUsage, TranscriptEntry};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    Stopped,
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorState {
    pub phase: Phase,
    pub scheduler: SchedulerState,
    pub graph: TaskGraph,
    pub total_cost: TokenUsage,
    pub started_at: DateTime<Utc>,
    pub last_event_at: DateTime<Utc>,
    pub workdir: PathBuf,
    pub config_dir: PathBuf,
    pub running_tasks: HashSet<Uuid>,
    pub history: Vec<HistoryEntry>,
    pub estimated_cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub at: DateTime<Utc>,
    pub message: String,
    pub phase: Phase,
}

impl OrchestratorState {
    pub fn new(workdir: PathBuf, config_dir: PathBuf) -> Self {
        let now = Utc::now();
        Self {
            phase: Phase::Spec,
            scheduler: SchedulerState::Idle,
            graph: TaskGraph::new(),
            total_cost: TokenUsage::default(),
            started_at: now,
            last_event_at: now,
            workdir,
            config_dir,
            running_tasks: HashSet::new(),
            history: Vec::new(),
            estimated_cost_usd: 0.0,
        }
    }

    pub fn counts(&self) -> StatusCounts {
        self.graph.count_by_status()
    }

    pub fn note(&mut self, msg: impl Into<String>) {
        let entry = HistoryEntry {
            at: Utc::now(),
            message: msg.into(),
            phase: self.phase,
        };
        self.history.push(entry);
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
    PhaseChanged {
        phase: Phase,
    },
    SchedulerStateChanged {
        state: SchedulerState,
    },
    TaskCreated {
        task: Task,
    },
    TaskStatusChanged {
        id: Uuid,
        status: TaskStatus,
    },
    TaskUpdated {
        task: Task,
    },
    TranscriptAppended {
        task_id: Uuid,
        entry: TranscriptEntry,
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

/// Shared, mutex-protected handle to the orchestrator state plus an SSE broadcast.
#[derive(Clone)]
pub struct StateHandle {
    inner: Arc<parking_lot::Mutex<OrchestratorState>>,
    tx: broadcast::Sender<UiEvent>,
}

impl StateHandle {
    pub fn new(state: OrchestratorState) -> Self {
        // Generously sized so a slow web client doesn't drop events; the
        // server-side broadcaster otherwise lags-then-drops oldest messages.
        let (tx, _rx) = broadcast::channel(16384);
        Self {
            inner: Arc::new(parking_lot::Mutex::new(state)),
            tx,
        }
    }

    pub fn read<R>(&self, f: impl FnOnce(&OrchestratorState) -> R) -> R {
        let g = self.inner.lock();
        f(&g)
    }

    pub fn write<R>(&self, f: impl FnOnce(&mut OrchestratorState) -> R) -> R {
        let mut g = self.inner.lock();
        let r = f(&mut g);
        g.last_event_at = Utc::now();
        r
    }

    pub fn snapshot(&self) -> OrchestratorState {
        self.inner.lock().clone()
    }

    pub fn replace(&self, new_state: OrchestratorState) {
        *self.inner.lock() = new_state;
    }

    pub fn emit(&self, ev: UiEvent) {
        let _ = self.tx.send(ev);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<UiEvent> {
        self.tx.subscribe()
    }
}
