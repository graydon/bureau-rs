//! Runtime state shared between engine and web UI.

use crate::graph::{NodeGraph, NodeId, Stage};
use crate::tools::{JudgeVerdict, TranscriptEntry};
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

/// Bound a task's transcript at `cap` entries. The opening entries
/// (system prompt, tool definitions, first user prompt) and recent
/// entries are both useful for debugging, so we keep both ends and drop
/// the middle. A single elision marker stays in place so the UI can show
/// "[N entries elided]" if it cares.
pub fn cap_transcript(task: &mut EngineTask, cap: usize) {
    if cap == 0 || task.transcript.len() <= cap {
        return;
    }
    let head_keep = (cap / 4).max(50).min(task.transcript.len());
    let tail_target = cap.saturating_sub(head_keep).max(1);
    let total = task.transcript.len();
    let tail_start = total.saturating_sub(tail_target);
    if tail_start <= head_keep {
        // Nothing to drop after enforcing both ends — leave alone.
        return;
    }
    let dropped = tail_start - head_keep;
    // Replace the dropped span with one synthetic Note describing the elision.
    let marker = TranscriptEntry {
        timestamp: chrono::Utc::now(),
        kind: crate::tools::TranscriptKind::Note,
        content: format!(
            "[{dropped} transcript entries elided to bound per-task memory]"
        ),
        role: None,
    };
    let tail: Vec<TranscriptEntry> = task.transcript.split_off(tail_start);
    task.transcript.truncate(head_keep);
    task.transcript.push(marker);
    task.transcript.extend(tail);
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
        // Capacity sized for short bursts under high concurrency, not for
        // long-running slow consumers. The broadcast channel is
        // drop-oldest-on-overflow; under sustained backpressure a slow
        // consumer will see gaps but each gap costs the consumer one
        // re-fetch, not unbounded memory. Larger capacities here cost
        // every slow consumer N × sizeof(UiEvent), which gets expensive
        // because transcript_appended events can carry multi-KB entries.
        let (tx, _rx) = broadcast::channel(1024);
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

    /// Like `snapshot` but produces an `EngineState` with each task's
    /// transcript *omitted* (empty `Vec`). Used by the web layer's
    /// polled `/api/state` route — transcripts are the bulk of state
    /// memory and shipping them on every poll dominates both the time
    /// spent holding the inner lock (which blocks engine writes) and
    /// the JSON payload size. Clients that need a specific task's
    /// transcript fetch it on demand via `/api/task_transcript`.
    pub fn snapshot_slim(&self) -> EngineState {
        let s = self.inner.lock();
        // Build a fresh EngineState that copies every field EXCEPT the
        // per-task transcripts — those are replaced with empty Vecs
        // without ever cloning the originals.
        let mut tasks = IndexMap::with_capacity(s.tasks.len());
        for (id, t) in s.tasks.iter() {
            tasks.insert(
                *id,
                EngineTask {
                    id: t.id,
                    node_id: t.node_id,
                    node_name: t.node_name.clone(),
                    stage: t.stage,
                    status: t.status,
                    model: t.model.clone(),
                    transcript: Vec::new(),
                    cost: t.cost.clone(),
                    started_at: t.started_at,
                    finished_at: t.finished_at,
                    error: t.error.clone(),
                    final_verdict: t.final_verdict.clone(),
                    retries: t.retries,
                },
            );
        }
        EngineState {
            workdir: s.workdir.clone(),
            config_dir: s.config_dir.clone(),
            project_name: s.project_name.clone(),
            graph: s.graph.clone(),
            tasks,
            scheduler: s.scheduler,
            total_cost: s.total_cost.clone(),
            estimated_cost_usd: s.estimated_cost_usd,
            started_at: s.started_at,
            last_event_at: s.last_event_at,
            history: s.history.clone(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Stage;
    use crate::tools::{Role, TranscriptKind};
    use uuid::Uuid;

    fn make_entry(i: usize) -> TranscriptEntry {
        TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::Note,
            content: format!("entry-{i}"),
            role: Some(Role::Writer),
        }
    }

    fn make_task(n: usize) -> EngineTask {
        let mut t = EngineTask {
            id: Uuid::new_v4(),
            node_id: crate::graph::NodeId::new(),
            node_name: "n".into(),
            stage: Stage::Spec,
            status: TaskStatus::Running,
            model: "mock".into(),
            transcript: Vec::new(),
            cost: TokenUsage::default(),
            started_at: Some(Utc::now()),
            finished_at: None,
            error: None,
            final_verdict: None,
            retries: 0,
        };
        for i in 0..n {
            t.transcript.push(make_entry(i));
        }
        t
    }

    #[test]
    fn cap_transcript_below_cap_is_noop() {
        let mut t = make_task(100);
        cap_transcript(&mut t, 500);
        assert_eq!(t.transcript.len(), 100);
        assert_eq!(t.transcript[0].content, "entry-0");
        assert_eq!(t.transcript[99].content, "entry-99");
    }

    #[test]
    fn cap_transcript_zero_disables_capping() {
        let mut t = make_task(10_000);
        cap_transcript(&mut t, 0);
        assert_eq!(t.transcript.len(), 10_000);
    }

    #[test]
    fn cap_transcript_preserves_head_and_tail_with_marker() {
        let mut t = make_task(2000);
        cap_transcript(&mut t, 500);
        // The result should be roughly head + marker + tail.
        // head = cap/4 = 125 (or at least 50). tail = cap - head = 375.
        assert!(t.transcript.len() < 2000, "should have been capped");
        // First entry preserved.
        assert_eq!(t.transcript[0].content, "entry-0");
        // Last entry preserved.
        let last = t.transcript.last().unwrap();
        assert_eq!(last.content, "entry-1999");
        // An elision marker appears in the middle.
        let elided = t
            .transcript
            .iter()
            .any(|e| matches!(e.kind, TranscriptKind::Note) && e.content.contains("elided"));
        assert!(elided, "elision marker not found");
    }

    #[test]
    fn snapshot_slim_omits_transcripts_but_keeps_everything_else() {
        let workdir = std::path::PathBuf::from("/tmp/no");
        let mut state = EngineState::new(workdir.clone(), workdir.clone(), "p".into());
        let task = make_task(50);
        let tid = task.id;
        state.tasks.insert(tid, task);
        state.total_cost = TokenUsage {
            input_tokens: 99,
            ..Default::default()
        };
        let handle = StateHandle::new(state);
        let slim = handle.snapshot_slim();
        assert!(slim.tasks.get(&tid).unwrap().transcript.is_empty());
        // Cost / model / status preserved.
        assert_eq!(slim.total_cost.input_tokens, 99);
        assert_eq!(slim.tasks.get(&tid).unwrap().model, "mock");
        assert_eq!(slim.tasks.get(&tid).unwrap().status, TaskStatus::Running);
        // Original (non-slim) still has the transcript.
        let full = handle.snapshot();
        assert_eq!(full.tasks.get(&tid).unwrap().transcript.len(), 50);
    }
}
