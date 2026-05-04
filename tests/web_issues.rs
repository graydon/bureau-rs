//! Integration test for `/api/issues`. Pins the resolved/retrying/permanent
//! lifecycle the UI uses to differentiate transient from permanent failures.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use bureau_rs::graph::{Node, NodeGraph, Stage};
use bureau_rs::state::{EngineState, EngineTask, StateHandle, TaskStatus, TokenUsage};
use bureau_rs::tools::{TranscriptEntry, TranscriptKind};
use bureau_rs::web::{router, AppState};
use chrono::Utc;
use parking_lot::Mutex;
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

fn entry(kind: TranscriptKind, content: &str) -> TranscriptEntry {
    TranscriptEntry {
        timestamp: Utc::now(),
        kind,
        content: content.into(),
        role: None,
    }
}

async fn fetch_issues(workdir: std::path::PathBuf, tasks: Vec<EngineTask>) -> serde_json::Value {
    let mut graph = NodeGraph::new();
    let root_id = graph.insert_root(Node::new("p", "umbrella")).unwrap();
    let _ = root_id;
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "p".into(),
    ));
    state.write(|st| {
        st.graph = graph.clone();
        for t in &tasks {
            st.tasks.insert(t.id, t.clone());
        }
    });
    let r = router(AppState {
        state,
        workdir,
        graph: Arc::new(Mutex::new(graph)),
        worktrees: None,
    });
    let resp = r
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/issues")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

fn task_with(node_name: &str, status: TaskStatus, transcript: Vec<TranscriptEntry>) -> EngineTask {
    let g_node_id = bureau_rs::graph::NodeId::new();
    EngineTask {
        id: Uuid::new_v4(),
        node_id: g_node_id,
        node_name: node_name.into(),
        stage: Stage::Spec,
        status,
        model: "mock".into(),
        transcript,
        cost: TokenUsage::default(),
        started_at: Some(Utc::now()),
        finished_at: None,
        error: None,
        final_verdict: None,
        retries: 0,
    }
}

#[tokio::test]
async fn issue_status_resolved_when_retry_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let transcript = vec![
        entry(
            TranscriptKind::ToolCall { tool: "decompose".into() },
            "{\"children\":[{\"name\":\"x\",\"deps\":[\"x\"]}]}",
        ),
        entry(
            TranscriptKind::ToolResult {
                tool: "decompose".into(),
                ok: false,
                error: Some("self-loop".into()),
                output: None,
            },
            "",
        ),
        // Retry, this time successful.
        entry(
            TranscriptKind::ToolCall { tool: "decompose".into() },
            "{\"children\":[{\"name\":\"x\",\"deps\":[]}]}",
        ),
        entry(
            TranscriptKind::ToolResult {
                tool: "decompose".into(),
                ok: true,
                error: None,
                output: Some("{\"created\":[\"x\"]}".into()),
            },
            "",
        ),
    ];
    let task = task_with("p", TaskStatus::Done, transcript);
    let issues = fetch_issues(tmp.path().to_path_buf(), vec![task]).await;
    let arr = issues.as_array().unwrap();
    assert_eq!(arr.len(), 1, "one failure (the retry success isn't an issue)");
    assert_eq!(arr[0]["status"], "resolved");
    assert_eq!(arr[0]["tool"], "decompose");
}

#[tokio::test]
async fn issue_status_retrying_when_task_still_in_progress() {
    let tmp = tempfile::tempdir().unwrap();
    let transcript = vec![
        entry(
            TranscriptKind::ToolCall { tool: "submit_public".into() },
            "{\"content\":\"mod foo;\"}",
        ),
        entry(
            TranscriptKind::ToolResult {
                tool: "submit_public".into(),
                ok: false,
                error: Some("forbidden mod".into()),
                output: None,
            },
            "",
        ),
    ];
    let task = task_with("p", TaskStatus::Running, transcript);
    let issues = fetch_issues(tmp.path().to_path_buf(), vec![task]).await;
    let arr = issues.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["status"], "retrying");
}

#[tokio::test]
async fn issue_status_permanent_when_task_done_without_retry() {
    let tmp = tempfile::tempdir().unwrap();
    let transcript = vec![
        entry(
            TranscriptKind::ToolCall { tool: "submit_public".into() },
            "{\"content\":\"mod foo;\"}",
        ),
        entry(
            TranscriptKind::ToolResult {
                tool: "submit_public".into(),
                ok: false,
                error: Some("forbidden mod".into()),
                output: None,
            },
            "",
        ),
    ];
    let task = task_with("p", TaskStatus::Failed, transcript);
    let issues = fetch_issues(tmp.path().to_path_buf(), vec![task]).await;
    let arr = issues.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["status"], "permanent");
}
