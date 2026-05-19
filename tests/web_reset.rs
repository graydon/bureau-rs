//! Integration test for the `/api/reset_node` web endpoint. Builds a small
//! graph on disk via `graph::save`, fires a reset request through the
//! axum router, and verifies that the named node and its transitive
//! dependents are reset (in the on-disk graph and the UI snapshot).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bureau_rs::graph::{self, Node, NodeGraph, Stage, StageState};
use bureau_rs::state::{EngineState, StateHandle};
use bureau_rs::web::{AppState, router};
use bureau_rs::worktree::{Workspace, WorktreePool};
use std::sync::Arc;
use tower::ServiceExt;

/// Set up a git-initialized workdir with the graph committed on main.
/// The reset endpoint takes `main_lock` and commits its mutation, so
/// tests need a real Workspace+WorktreePool rather than just an
/// untracked directory.
fn make_workspace_with_graph(g: &NodeGraph) -> (tempfile::TempDir, Arc<WorktreePool>) {
    let tmp = tempfile::tempdir().unwrap();
    let ws = Workspace::init(tmp.path()).unwrap();
    graph::save(tmp.path(), g).unwrap();
    ws.commit_main("seed").unwrap();
    let pool = Arc::new(WorktreePool::new(ws).unwrap());
    (tmp, pool)
}

fn done_all(g: &mut NodeGraph) {
    for n in g.nodes.values_mut() {
        for s in Stage::ALL {
            n.stages.set(s, StageState::Done);
        }
    }
}

#[tokio::test]
async fn reset_node_cascades_through_dependents() {
    let mut g = NodeGraph::new();
    let root_id = g.insert_root(Node::new("proj", "umbrella")).unwrap();
    let lib_id = g
        .add_child(root_id, Node::new("lib", "library"))
        .unwrap();
    let app_id = g
        .add_child(root_id, Node::new("app", "application"))
        .unwrap();
    g.add_dep(app_id, lib_id).unwrap();
    done_all(&mut g);

    let (workdir, pool) = make_workspace_with_graph(&g);
    let state = StateHandle::new(EngineState::new(
        workdir.path().to_path_buf(),
        workdir.path().to_path_buf(),
        "proj".into(),
    ));

    let app = AppState {
        state: state.clone(),
        workdir: workdir.path().to_path_buf(),
        layout: bureau_rs::render::Layout::SingleCrate,
        worktrees: Some(pool),
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

    // On-disk graph reflects the reset.
    let g2 = graph::load(workdir.path(), bureau_rs::render::Layout::SingleCrate).unwrap();
    let lib_n = g2.get(lib_id).unwrap();
    let app_n = g2.get(app_id).unwrap();
    let root_n = g2.get(root_id).unwrap();
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

    // The in-memory state no longer mirrors the graph; the on-disk
    // graph is the only source of truth (already asserted above).
}

#[tokio::test]
async fn reset_node_without_cascade_only_resets_target() {
    let mut g = NodeGraph::new();
    let root_id = g.insert_root(Node::new("proj", "umbrella")).unwrap();
    let lib_id = g
        .add_child(root_id, Node::new("lib", "library"))
        .unwrap();
    let app_id = g
        .add_child(root_id, Node::new("app", "application"))
        .unwrap();
    g.add_dep(app_id, lib_id).unwrap();
    done_all(&mut g);

    let (workdir, pool) = make_workspace_with_graph(&g);
    let state = StateHandle::new(EngineState::new(
        workdir.path().to_path_buf(),
        workdir.path().to_path_buf(),
        "proj".into(),
    ));

    let r = router(AppState {
        state: state.clone(),
        workdir: workdir.path().to_path_buf(),
        layout: bureau_rs::render::Layout::SingleCrate,
        worktrees: Some(pool),
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

    let g2 = graph::load(workdir.path(), bureau_rs::render::Layout::SingleCrate).unwrap();
    assert_eq!(g2.get(lib_id).unwrap().stages.spec, StageState::NotStarted);
    assert_eq!(g2.get(app_id).unwrap().stages.spec, StageState::Done);
}

#[tokio::test]
async fn reset_node_wipes_authored_slot_files_to_placeholders() {
    // The bug this prevents: operator clicks "reset" in the UI, the
    // engine resets stage states but the on-disk `public.rs` /
    // `tests.rs` / etc. still contain the prior authored content. The
    // next run sees stages NotStarted but the cargo gate compiles the
    // orphan content and the writer agent gets confused (same failure
    // mode as the crashed-restart bug we fixed for engine startup).
    //
    // Build a small graph with authored content on the target node,
    // render it to disk, fire /api/reset_node, then verify the slot
    // files on disk are placeholder content (= "not yet authored").
    use bureau_rs::placeholders;

    let mut g = NodeGraph::new();
    let root_id = g.insert_root(Node::new("proj", "umbrella")).unwrap();
    let lib_id = g
        .add_child(root_id, Node::new("lib", "library"))
        .unwrap();
    // Author content on `lib` so the renderer writes real files.
    {
        let n = g.get_mut(lib_id).unwrap();
        n.spec_public_md = Some("# lib spec\n".into());
        n.public_rs = Some("pub trait L { fn f(&self); }\n".into());
        n.private_rs =
            Some("use super::public::*; pub struct LL; impl L for LL { fn f(&self) { todo!() } }\n".into())
                ;
        n.tests_rs = Some("#[test] fn t() {}\n".into());
        n.stages.spec = StageState::Done;
        n.stages.iface = StageState::Done;
        n.stages.tests = StageState::Done;
        n.stages.impl_ = StageState::Done;
    }

    let tmp = tempfile::tempdir().unwrap();
    let ws = Workspace::init(tmp.path()).unwrap();
    graph::save(tmp.path(), &g).unwrap();
    bureau_rs::render::render_graph(tmp.path(), &g, bureau_rs::render::Layout::SingleCrate)
        .unwrap();
    ws.commit_main("seed with authored content").unwrap();
    let pool = Arc::new(WorktreePool::new(ws).unwrap());

    // Sanity: the slot files on disk contain the authored content (NOT
    // placeholders) before the reset.
    let src_dir = tmp.path().join("src/lib");
    let public_rs_before = std::fs::read_to_string(src_dir.join("public.rs")).unwrap();
    let tests_rs_before = std::fs::read_to_string(src_dir.join("tests.rs")).unwrap();
    assert!(!placeholders::is_placeholder_public_rs(&public_rs_before));
    assert!(!placeholders::is_placeholder_tests_rs(&tests_rs_before));

    let state = StateHandle::new(EngineState::new(
        tmp.path().to_path_buf(),
        tmp.path().to_path_buf(),
        "proj".into(),
    ));
    let r = router(AppState {
        state,
        workdir: tmp.path().to_path_buf(),
        layout: bureau_rs::render::Layout::SingleCrate,
        worktrees: Some(pool),
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

    // After the reset: the slot files on disk MUST be placeholder
    // content. If they still contain the prior authored Rust, the bug
    // is back.
    let public_rs_after = std::fs::read_to_string(src_dir.join("public.rs")).unwrap();
    let private_rs_after = std::fs::read_to_string(src_dir.join("private.rs")).unwrap();
    let tests_rs_after = std::fs::read_to_string(src_dir.join("tests.rs")).unwrap();
    assert!(
        placeholders::is_placeholder_public_rs(&public_rs_after),
        "public.rs not wiped: {public_rs_after:?}"
    );
    assert!(
        placeholders::is_placeholder_private_rs(&private_rs_after),
        "private.rs not wiped: {private_rs_after:?}"
    );
    assert!(
        placeholders::is_placeholder_tests_rs(&tests_rs_after),
        "tests.rs not wiped: {tests_rs_after:?}"
    );
    // Spec markdown also wiped. Path includes the root crate name in
    // `node_spec_dir` (`spec/<root>/<...>`).
    let spec_md_after =
        std::fs::read_to_string(tmp.path().join("spec/proj/lib/public.md")).unwrap();
    assert!(
        placeholders::is_placeholder_public_md(&spec_md_after),
        "spec/proj/lib/public.md not wiped: {spec_md_after:?}"
    );
}
