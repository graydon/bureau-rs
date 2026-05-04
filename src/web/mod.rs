//! Web UI: axum router + SSE event broadcast + embedded SPA.

pub mod ui;

use crate::checkpoint;
use crate::graph::NodeId;
use crate::state::{SchedulerState, StateHandle, TaskStatus, UiEvent};
use anyhow::Result;
use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub state: StateHandle,
    pub workdir: PathBuf,
    /// The engine's authoritative graph. Web mutations (e.g. reset_node)
    /// must hit this so the running engine sees them immediately.
    pub graph: std::sync::Arc<parking_lot::Mutex<crate::graph::NodeGraph>>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(ui_index))
        .route("/api/state", get(api_state))
        .route("/api/events", get(api_events))
        .route("/api/files", get(api_files))
        .route("/api/file", get(api_file))
        .route("/api/gitlog", get(api_gitlog))
        .route("/api/task_transcript", get(api_task_transcript))
        .route("/api/issues", get(api_issues))
        .route("/api/checkpoint", post(api_checkpoint))
        .route("/api/pause", post(api_pause))
        .route("/api/resume", post(api_resume))
        .route("/api/stop", post(api_stop))
        .route("/api/reset_node", post(api_reset_node))
        .with_state(state)
}

pub async fn serve(state: AppState, port: u16) -> Result<()> {
    let app = router(state);
    let addr = format!("0.0.0.0:{port}").parse::<std::net::SocketAddr>()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("web UI: http://{addr}");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn ui_index() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        ui::INDEX_HTML,
    )
}

async fn api_state(State(s): State<AppState>) -> Json<crate::state::EngineState> {
    Json(s.state.snapshot())
}

async fn api_events(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = s.state.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| match res {
        Ok(ev) => match serde_json::to_string(&ev) {
            Ok(json) => Some(Ok(Event::default().data(json))),
            Err(_) => None,
        },
        Err(_) => None,
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Serialize)]
struct FileEntry {
    path: String,
    size: u64,
}

async fn api_files(State(s): State<AppState>) -> Json<Vec<FileEntry>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&s.workdir)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let rel = entry
            .path()
            .strip_prefix(&s.workdir)
            .ok()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| entry.path().to_path_buf());
        if rel.components().any(|c| {
            let c = c.as_os_str().to_string_lossy();
            c == ".git" || c == "target" || c == ".bureau"
        }) {
            continue;
        }
        let size = entry.metadata().ok().map(|m| m.len()).unwrap_or(0);
        out.push(FileEntry {
            path: rel.to_string_lossy().to_string(),
            size,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Json(out)
}

#[derive(Deserialize)]
struct PathQ {
    path: String,
}

async fn api_file(State(s): State<AppState>, Query(q): Query<PathQ>) -> Response {
    let rel = PathBuf::from(&q.path);
    if rel.is_absolute() || rel.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return (StatusCode::BAD_REQUEST, "invalid path").into_response();
    }
    let abs = s.workdir.join(&rel);
    match std::fs::read_to_string(&abs) {
        Ok(c) => c.into_response(),
        Err(e) => (StatusCode::NOT_FOUND, format!("{e}")).into_response(),
    }
}

#[derive(Serialize)]
struct CommitEntry {
    sha: String,
    message: String,
}

async fn api_gitlog(State(s): State<AppState>) -> Json<Vec<CommitEntry>> {
    Json(read_gitlog(&s.workdir).unwrap_or_default())
}

fn read_gitlog(dir: &Path) -> Result<Vec<CommitEntry>> {
    let repo = git2::Repository::open(dir)?;
    let mut walk = repo.revwalk()?;
    walk.push_head().ok();
    walk.set_sorting(git2::Sort::TIME)?;
    let mut out = Vec::new();
    for oid in walk.take(200) {
        let oid = oid?;
        let c = repo.find_commit(oid)?;
        out.push(CommitEntry {
            sha: oid.to_string(),
            message: c.summary().unwrap_or("").to_string(),
        });
    }
    Ok(out)
}

#[derive(Deserialize)]
struct TaskIdQ {
    id: Uuid,
}

async fn api_task_transcript(State(s): State<AppState>, Query(q): Query<TaskIdQ>) -> Response {
    match s.state.read(|st| st.tasks.get(&q.id).cloned()) {
        Some(t) => Json(t.transcript).into_response(),
        None => (StatusCode::NOT_FOUND, "task not found").into_response(),
    }
}

#[derive(Serialize)]
struct Issue {
    task_id: Uuid,
    node_id: NodeId,
    node_name: String,
    stage: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    kind: &'static str,
    tool: Option<String>,
    message: String,
    args: Option<String>,
    entry_index: usize,
    /// Lifecycle of this failure:
    /// - `"resolved"`: a later same-tool call succeeded after this one.
    /// - `"retrying"`: still unresolved, but the owning task is in
    ///   progress so the engine may yet retry.
    /// - `"permanent"`: still unresolved and the owning task has finished.
    status: &'static str,
}

async fn api_issues(State(s): State<AppState>) -> Json<Vec<Issue>> {
    let mut out = Vec::new();
    s.state.read(|st| {
        for (tid, t) in st.tasks.iter() {
            let entries = &t.transcript;
            // Pre-compute, per tool name, the index of the latest
            // SUCCESSFUL result. A failure is "resolved" iff a later
            // success exists for the same tool.
            let mut latest_success: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for (i, e) in entries.iter().enumerate() {
                if let crate::tools::TranscriptKind::ToolResult {
                    tool, ok: true, ..
                } = &e.kind
                {
                    latest_success.insert(tool.clone(), i);
                }
            }
            let task_in_progress = matches!(t.status, TaskStatus::Pending | TaskStatus::Running);

            for (i, e) in entries.iter().enumerate() {
                match &e.kind {
                    crate::tools::TranscriptKind::ToolResult {
                        tool, ok: false, error, ..
                    } => {
                        let resolved = latest_success
                            .get(tool)
                            .map(|s| *s > i)
                            .unwrap_or(false);
                        let status = if resolved {
                            "resolved"
                        } else if task_in_progress {
                            "retrying"
                        } else {
                            "permanent"
                        };
                        let args = entries[..i].iter().rev().find_map(|p| match &p.kind {
                            crate::tools::TranscriptKind::ToolCall { tool: t2 } if t2 == tool => {
                                Some(p.content.clone())
                            }
                            _ => None,
                        });
                        out.push(Issue {
                            task_id: *tid,
                            node_id: t.node_id,
                            node_name: t.node_name.clone(),
                            stage: t.stage.to_string(),
                            timestamp: e.timestamp,
                            kind: "tool_failure",
                            tool: Some(tool.clone()),
                            message: error.clone().unwrap_or_default(),
                            args,
                            entry_index: i,
                            status,
                        });
                    }
                    crate::tools::TranscriptKind::Error => {
                        let status = if task_in_progress { "retrying" } else { "permanent" };
                        out.push(Issue {
                            task_id: *tid,
                            node_id: t.node_id,
                            node_name: t.node_name.clone(),
                            stage: t.stage.to_string(),
                            timestamp: e.timestamp,
                            kind: "task_error",
                            tool: None,
                            message: e.content.clone(),
                            args: None,
                            entry_index: i,
                            status,
                        });
                    }
                    _ => {}
                }
            }
        }
    });
    out.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Json(out)
}

async fn api_checkpoint(State(s): State<AppState>) -> Response {
    let dir = s.workdir.join(".bureau").join("checkpoints");
    let snap = s.state.snapshot();
    match checkpoint::save(&snap, &dir) {
        Ok(p) => Json(serde_json::json!({"path": p.display().to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response(),
    }
}

async fn api_pause(State(s): State<AppState>) -> Response {
    s.state.write(|st| st.scheduler = SchedulerState::Paused);
    s.state.emit(UiEvent::SchedulerStateChanged {
        state: SchedulerState::Paused,
    });
    StatusCode::OK.into_response()
}

async fn api_resume(State(s): State<AppState>) -> Response {
    s.state.write(|st| st.scheduler = SchedulerState::Running);
    s.state.emit(UiEvent::SchedulerStateChanged {
        state: SchedulerState::Running,
    });
    StatusCode::OK.into_response()
}

#[derive(Deserialize)]
struct ResetNodeBody {
    node_id: crate::graph::NodeId,
    /// If true (default), also reset every node that transitively depends
    /// on this node — their work is now stale.
    #[serde(default = "default_true")]
    cascade: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct ResetNodeOk {
    reset: Vec<String>,
}

async fn api_reset_node(State(s): State<AppState>, Json(body): Json<ResetNodeBody>) -> Response {
    use crate::graph::{Stage, StageState};
    let mut reset_names: Vec<String> = Vec::new();
    {
        let mut g = s.graph.lock();
        // Collect targets: the named node plus (if cascading) every node
        // whose transitive deps include the named node.
        let mut targets: std::collections::HashSet<crate::graph::NodeId> =
            std::collections::HashSet::new();
        targets.insert(body.node_id);
        if body.cascade {
            let ids: Vec<_> = g.nodes.keys().copied().collect();
            for id in ids {
                if id != body.node_id && g.dep_reaches(id, body.node_id) {
                    targets.insert(id);
                }
            }
        }
        for id in &targets {
            if let Some(n) = g.get_mut(*id) {
                for stage in Stage::ALL {
                    n.stages.set(stage, StageState::NotStarted);
                }
                reset_names.push(n.name.clone());
            }
        }
    }
    // Sync the change into the EngineState so the UI reflects it
    // immediately rather than waiting for the engine's next sync.
    let snap = s.graph.lock().clone();
    let msg = format!(
        "reset {} node(s) from web UI: {}",
        reset_names.len(),
        reset_names.join(", ")
    );
    s.state.write(|st| {
        st.graph = snap;
        st.note(msg);
    });
    Json(ResetNodeOk { reset: reset_names }).into_response()
}

async fn api_stop(State(s): State<AppState>) -> Response {
    s.state.write(|st| st.scheduler = SchedulerState::Stopped);
    s.state.emit(UiEvent::SchedulerStateChanged {
        state: SchedulerState::Stopped,
    });
    StatusCode::OK.into_response()
}

// Suppress unused warning when TaskStatus isn't directly referenced.
fn _force_taskstatus(_: TaskStatus) {}
