//! Integration test for the `/api/reset_node` web endpoint. Builds a small
//! graph in memory, fires a reset request through the axum router, and
//! verifies that the named node and its transitive dependents are reset.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bureau_rs::graph::{Node, NodeGraph, Stage, StageState};
use bureau_rs::state::{EngineState, StateHandle};
use bureau_rs::web::{AppState, router};
use parking_lot::Mutex;
use std::sync::Arc;
use tower::ServiceExt;

fn done_all(g: &mut NodeGraph) {
    for n in g.nodes.values_mut() {
        for s in Stage::ALL {
            n.stages.set(s, StageState::Done);
        }
    }
}

#[tokio::test]
async fn reset_node_cascades_through_dependents() {
    // Build root with two children: lib (no deps) and app (deps on lib).
    let mut graph = NodeGraph::new();
    let root_id = graph.insert_root(Node::new("proj", "umbrella")).unwrap();
    let lib_id = graph
        .add_child(root_id, Node::new("lib", "library"))
        .unwrap();
    let app_id = graph
        .add_child(root_id, Node::new("app", "application"))
        .unwrap();
    graph.add_dep(app_id, lib_id).unwrap(); // app -> lib
    done_all(&mut graph);

    let workdir = tempfile::tempdir().unwrap();
    let state = StateHandle::new(EngineState::new(
        workdir.path().to_path_buf(),
        workdir.path().to_path_buf(),
        "proj".into(),
    ));
    state.write(|st| st.graph = graph.clone());

    let graph_mutex = Arc::new(Mutex::new(graph));
    let app = AppState {
        state: state.clone(),
        workdir: workdir.path().to_path_buf(),
        graph: graph_mutex.clone(),
    };
    let r = router(app);

    let body = serde_json::to_vec(&serde_json::json!({
        "node_id": lib_id,
        "cascade": true,
    }))
    .unwrap();
    let resp = r
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/reset_node")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Engine's authoritative graph should reflect the reset.
    let g = graph_mutex.lock();
    let lib_n = g.get(lib_id).unwrap();
    let app_n = g.get(app_id).unwrap();
    let root_n = g.get(root_id).unwrap();
    for s in Stage::ALL {
        assert_eq!(
            lib_n.stages.get(s),
            StageState::NotStarted,
            "lib stage {s:?}"
        );
        assert_eq!(
            app_n.stages.get(s),
            StageState::NotStarted,
            "app (cascaded) stage {s:?}"
        );
        assert_eq!(
            root_n.stages.get(s),
            StageState::Done,
            "root must be untouched (lib is not its dep)"
        );
    }

    // EngineState snapshot must also reflect the reset (so the UI sees it
    // immediately without waiting for the engine's next sync).
    let snap = state.snapshot();
    let lib_snap = snap.graph.get(lib_id).unwrap();
    assert_eq!(lib_snap.stages.get(Stage::Spec), StageState::NotStarted);
}

#[tokio::test]
async fn reset_node_without_cascade_only_resets_target() {
    let mut graph = NodeGraph::new();
    let root_id = graph.insert_root(Node::new("proj", "umbrella")).unwrap();
    let lib_id = graph
        .add_child(root_id, Node::new("lib", "library"))
        .unwrap();
    let app_id = graph
        .add_child(root_id, Node::new("app", "application"))
        .unwrap();
    graph.add_dep(app_id, lib_id).unwrap();
    done_all(&mut graph);

    let workdir = tempfile::tempdir().unwrap();
    let state = StateHandle::new(EngineState::new(
        workdir.path().to_path_buf(),
        workdir.path().to_path_buf(),
        "proj".into(),
    ));
    state.write(|st| st.graph = graph.clone());

    let graph_mutex = Arc::new(Mutex::new(graph));
    let r = router(AppState {
        state: state.clone(),
        workdir: workdir.path().to_path_buf(),
        graph: graph_mutex.clone(),
    });

    let body = serde_json::to_vec(&serde_json::json!({
        "node_id": lib_id,
        "cascade": false,
    }))
    .unwrap();
    let resp = r
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/reset_node")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let g = graph_mutex.lock();
    assert_eq!(g.get(lib_id).unwrap().stages.spec, StageState::NotStarted);
    // app is a dependent of lib but cascade=false, so it stays Done.
    assert_eq!(g.get(app_id).unwrap().stages.spec, StageState::Done);
}
