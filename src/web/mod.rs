//! Web UI: axum router, SSE state broadcasting, embedded SPA.

pub mod ui;

use crate::checkpoint;
use crate::state::{SchedulerState, StateHandle, UiEvent};
use crate::task::TaskStatus;
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
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub state: StateHandle,
    pub workdir: PathBuf,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(ui_index))
        .route("/api/state", get(api_state))
        .route("/api/events", get(api_events))
        .route("/api/files", get(api_files))
        .route("/api/file", get(api_file))
        .route("/api/gitlog", get(api_gitlog))
        .route("/api/gitdiff", get(api_gitdiff))
        .route("/api/task_transcript", get(api_task_transcript))
        .route("/api/task_files", get(api_task_files))
        .route("/api/task_file", get(api_task_file))
        .route("/api/issues", get(api_issues))
        .route("/api/phase_info", get(api_phase_info))
        .route("/api/skip", post(api_skip))
        .route("/api/retry", post(api_retry))
        .route("/api/pause", post(api_pause))
        .route("/api/resume", post(api_resume))
        .route("/api/checkpoint", post(api_checkpoint))
        .route("/api/stop", post(api_stop))
        .with_state(state)
}

pub async fn serve(state: AppState, port: u16) -> Result<()> {
    let app = router(state);
    let addr = format!("0.0.0.0:{port}").parse::<std::net::SocketAddr>()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("web UI listening on http://{}", addr);
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

async fn api_state(State(s): State<AppState>) -> Json<crate::state::OrchestratorState> {
    Json(s.state.snapshot())
}

async fn api_events(State(s): State<AppState>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
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
    is_dir: bool,
}

async fn api_files(State(s): State<AppState>) -> Json<Vec<FileEntry>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&s.workdir)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let rel = entry
            .path()
            .strip_prefix(&s.workdir)
            .ok()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| entry.path().to_path_buf());
        // Filter on relative path so a workspace whose own path contains
        // `.bureau` doesn't hide everything.
        if rel.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == ".git" || s == "target" || s == ".bureau"
        }) {
            continue;
        }
        let size = entry
            .metadata()
            .ok()
            .map(|m| m.len())
            .unwrap_or(0);
        out.push(FileEntry {
            path: rel.to_string_lossy().to_string(),
            size,
            is_dir: entry.file_type().is_dir(),
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
        Ok(content) => content.into_response(),
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
struct HashQ {
    hash: String,
}

async fn api_gitdiff(State(s): State<AppState>, Query(q): Query<HashQ>) -> Response {
    match git_diff(&s.workdir, &q.hash) {
        Ok(diff) => diff.into_response(),
        Err(e) => (StatusCode::NOT_FOUND, format!("{e}")).into_response(),
    }
}

fn git_diff(dir: &Path, hash: &str) -> Result<String> {
    let repo = git2::Repository::open(dir)?;
    let oid = git2::Oid::from_str(hash)?;
    let commit = repo.find_commit(oid)?;
    let tree = commit.tree()?;
    let parent = commit.parent(0).ok();
    let parent_tree = parent.as_ref().and_then(|p| p.tree().ok());
    let diff = repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), None)?;
    let mut buf = String::new();
    diff.print(git2::DiffFormat::Patch, |_d, _h, line| {
        let prefix = match line.origin() {
            '+' | '-' | ' ' => format!("{}", line.origin()),
            _ => String::new(),
        };
        buf.push_str(&prefix);
        buf.push_str(std::str::from_utf8(line.content()).unwrap_or(""));
        true
    })?;
    Ok(buf)
}

#[derive(Deserialize)]
struct TaskIdQ {
    id: Uuid,
}

async fn api_task_transcript(State(s): State<AppState>, Query(q): Query<TaskIdQ>) -> Response {
    match s.state.read(|st| st.graph.get(q.id).cloned()) {
        Some(t) => Json(t.transcript).into_response(),
        None => (StatusCode::NOT_FOUND, "task not found").into_response(),
    }
}

#[derive(Serialize)]
struct TaskFileEntry {
    path: String,
    size: u64,
    location: &'static str,
}

#[derive(Serialize)]
struct TaskFilesResp {
    worktree_root: Option<String>,
    files: Vec<TaskFileEntry>,
}

async fn api_task_files(State(s): State<AppState>, Query(q): Query<TaskIdQ>) -> Response {
    let task = match s.state.read(|st| st.graph.get(q.id).cloned()) {
        Some(t) => t,
        None => return (StatusCode::NOT_FOUND, "task not found").into_response(),
    };
    let mut files = Vec::new();
    if let Some(wt) = &task.worktree {
        if wt.exists() {
            for e in walkdir::WalkDir::new(wt)
                .min_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if !e.file_type().is_file() {
                    continue;
                }
                let rel_path = e.path().strip_prefix(wt).unwrap_or(e.path());
                // Worktrees themselves live under `.bureau/worktrees/...`, so
                // filter relative-to-worktree, not absolute path components.
                if rel_path.components().any(|c| {
                    let s = c.as_os_str().to_string_lossy();
                    s == ".git" || s == "target" || s == ".bureau"
                }) {
                    continue;
                }
                let rel = rel_path.to_string_lossy().to_string();
                let size = e.metadata().ok().map(|m| m.len()).unwrap_or(0);
                files.push(TaskFileEntry {
                    path: rel,
                    size,
                    location: "worktree",
                });
            }
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Json(TaskFilesResp {
        worktree_root: task.worktree.as_ref().map(|p| p.display().to_string()),
        files,
    })
    .into_response()
}

#[derive(Deserialize)]
struct TaskFileQ {
    id: Uuid,
    path: String,
}

async fn api_task_file(State(s): State<AppState>, Query(q): Query<TaskFileQ>) -> Response {
    let task = match s.state.read(|st| st.graph.get(q.id).cloned()) {
        Some(t) => t,
        None => return (StatusCode::NOT_FOUND, "task not found").into_response(),
    };
    let wt = match &task.worktree {
        Some(p) => p.clone(),
        None => return (StatusCode::NOT_FOUND, "task has no worktree").into_response(),
    };
    let rel = PathBuf::from(&q.path);
    if rel.is_absolute() || rel.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return (StatusCode::BAD_REQUEST, "invalid path").into_response();
    }
    let abs = wt.join(&rel);
    match std::fs::read_to_string(&abs) {
        Ok(c) => c.into_response(),
        Err(e) => (StatusCode::NOT_FOUND, format!("{e}")).into_response(),
    }
}

#[derive(Serialize)]
struct Issue {
    task_id: Uuid,
    task_description: String,
    phase: crate::phase::Phase,
    timestamp: chrono::DateTime<chrono::Utc>,
    kind: &'static str,
    tool: Option<String>,
    message: String,
    args: Option<String>,
    /// Index of the originating entry within the task's transcript, so the
    /// UI can scroll directly to it.
    entry_index: usize,
}

#[derive(Deserialize)]
struct PhaseQ {
    phase: String,
}

#[derive(Serialize)]
struct PhaseInfoResp {
    phase: String,
    tools: Vec<crate::tools::ToolInfo>,
}

async fn api_phase_info(Query(q): Query<PhaseQ>) -> Response {
    let phase = match crate::phase::Phase::parse(&q.phase) {
        Some(p) => p,
        None => return (StatusCode::BAD_REQUEST, "unknown phase").into_response(),
    };
    let tools = crate::tools::phase_tools(phase).await;
    Json(PhaseInfoResp {
        phase: phase.to_string(),
        tools,
    })
    .into_response()
}

async fn api_issues(State(s): State<AppState>) -> Json<Vec<Issue>> {
    let mut out = Vec::new();
    s.state.read(|st| {
        for (tid, t) in st.graph.tasks.iter() {
            // Pair tool_call args with the immediately-following tool_result
            // so we can attribute errors back to the call that triggered them.
            let entries = &t.transcript;
            for (i, e) in entries.iter().enumerate() {
                match &e.kind {
                    crate::task::TranscriptKind::ToolResult { tool, ok: false, error, .. } => {
                        let args = entries[..i].iter().rev().find_map(|p| match &p.kind {
                            crate::task::TranscriptKind::ToolCall { tool: t2, args }
                                if t2 == tool =>
                            {
                                Some(args.clone())
                            }
                            _ => None,
                        });
                        out.push(Issue {
                            task_id: *tid,
                            task_description: t.description.clone(),
                            phase: t.phase,
                            timestamp: e.timestamp,
                            kind: "tool_failure",
                            tool: Some(tool.clone()),
                            message: error.clone().unwrap_or_default(),
                            args,
                            entry_index: i,
                        });
                    }
                    crate::task::TranscriptKind::Error => {
                        out.push(Issue {
                            task_id: *tid,
                            task_description: t.description.clone(),
                            phase: t.phase,
                            timestamp: e.timestamp,
                            kind: "task_error",
                            tool: None,
                            message: e.content.clone(),
                            args: None,
                            entry_index: i,
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

#[derive(Deserialize)]
struct TaskIdBody {
    task_id: Uuid,
}

async fn api_skip(State(s): State<AppState>, Json(b): Json<TaskIdBody>) -> Response {
    s.state.write(|st| {
        if let Some(t) = st.graph.get_mut(b.task_id) {
            if !t.status.is_terminal() {
                t.status = TaskStatus::Skipped;
                t.finished_at = Some(chrono::Utc::now());
            }
        }
    });
    s.state.emit(UiEvent::TaskStatusChanged {
        id: b.task_id,
        status: TaskStatus::Skipped,
    });
    StatusCode::OK.into_response()
}

async fn api_retry(State(s): State<AppState>, Json(b): Json<TaskIdBody>) -> Response {
    s.state.write(|st| {
        if let Some(t) = st.graph.get_mut(b.task_id) {
            if matches!(t.status, TaskStatus::Failed) {
                t.status = TaskStatus::Pending;
                t.started_at = None;
                t.finished_at = None;
                t.error = None;
                t.retries += 1;
            }
        }
    });
    s.state.emit(UiEvent::TaskStatusChanged {
        id: b.task_id,
        status: TaskStatus::Pending,
    });
    StatusCode::OK.into_response()
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

async fn api_checkpoint(State(s): State<AppState>) -> Response {
    let dir = s.workdir.join(".bureau").join("checkpoints");
    let snap = s.state.snapshot();
    match checkpoint::save(&snap, &dir) {
        Ok(p) => Json(serde_json::json!({"path": p.display().to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")).into_response(),
    }
}

async fn api_stop(State(s): State<AppState>) -> Response {
    s.state.write(|st| st.scheduler = SchedulerState::Stopped);
    s.state.emit(UiEvent::SchedulerStateChanged {
        state: SchedulerState::Stopped,
    });
    StatusCode::OK.into_response()
}

