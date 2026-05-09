//! Integration tests that drive the rendered workspace through actual
//! `cargo` invocations. The point is to catch regressions where the
//! framework writes a graph that LOOKS valid but doesn't compile — most
//! famously the workspace `-p` bug where the model passed module names
//! (not crate names) and cargo errored confusingly.
//!
//! These tests are slow-ish (one cargo invocation each) but cheap
//! relative to e2e LLM runs. They build a graph in code, render it,
//! invoke cargo via the framework's tool surface, and verify the
//! outcome.

use bureau_rs::graph::{Node, NodeGraph, Stage, StageState};
use bureau_rs::render::{Layout, render_graph};
use bureau_rs::tools::{
    CargoCheckTool, Role, SubmitPrivateTool, SubmitPublicTool, SubmitRustArgs, TaskCtx,
};
use parking_lot::Mutex;
use rig::tool::Tool;
use std::sync::Arc;
use uuid::Uuid;

fn ctx_for(
    workdir: std::path::PathBuf,
    layout: Layout,
    graph: Arc<Mutex<NodeGraph>>,
    node_id: bureau_rs::graph::NodeId,
    stage: Stage,
) -> Arc<TaskCtx> {
    Arc::new(TaskCtx::new(
        Uuid::new_v4(),
        node_id,
        stage,
        Role::Writer,
        graph,
        workdir,
        layout,
        300,
        500,
        64,
        5,
        Arc::new(tokio::sync::Mutex::new(())),
    ))
}

/// Build a single-crate workspace, render it, run cargo_check via the
/// framework's tool. The minimum-viable end-to-end smoke test.
#[tokio::test]
async fn single_crate_cargo_check_succeeds_on_default_scaffold() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut g = NodeGraph::new();
    let mut root = Node::new("hello", "tiny project");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
    let graph = Arc::new(Mutex::new(g));
    // Need iface stage (cargo_check is registered there) and content
    // that compiles. Author public.rs with a trivial trait.
    let ctx = ctx_for(workdir.clone(), Layout::SingleCrate, graph.clone(), root_id, Stage::Iface);
    let pub_tool = SubmitPublicTool { ctx: ctx.clone() };
    pub_tool
        .call(SubmitRustArgs {
            content: "pub trait Hello { fn say(&self) -> &str; }\npub struct H;\n".into(),
        })
        .await
        .unwrap();
    let priv_tool = SubmitPrivateTool { ctx: ctx.clone() };
    priv_tool
        .call(SubmitRustArgs {
            content: "use super::public::*;\nimpl Hello for super::H { fn say(&self) -> &str { todo!() } }\n".into(),
        })
        .await
        .unwrap();
    // cargo_check from the framework — should succeed.
    let check = CargoCheckTool { ctx };
    let out = check
        .call(bureau_rs::tools::CargoArgs { package: None })
        .await
        .expect("cargo_check tool should not itself error");
    assert!(
        out.passed,
        "single-crate scaffold should pass cargo check; errors: {:?}",
        out.errors
    );
}

/// Workspace mode with two member crates (`util` and `core`). Run
/// cargo_check from `core`'s perspective. Pin that the framework's
/// package-name resolver scopes correctly to a real workspace member.
#[tokio::test]
async fn workspace_cargo_check_resolves_package_to_real_member() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut g = NodeGraph::new();
    let mut root = Node::new("ws", "workspace");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    let mut util_node = Node::new("util", "shared util");
    util_node.crate_boundary = true;
    let _util_id = g.add_child(root_id, util_node).unwrap();
    let mut core_node = Node::new("core", "main logic");
    core_node.crate_boundary = true;
    let core_id = g.add_child(root_id, core_node).unwrap();
    render_graph(&workdir, &g, Layout::Workspace).unwrap();
    let graph = Arc::new(Mutex::new(g));
    // Run cargo_check; the model passes a BAD `package` arg (a module
    // name that doesn't exist as a workspace member). The framework
    // should resolve it to the current node's containing crate (`core`)
    // and run cargo successfully.
    let ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
        graph.clone(),
        core_id,
        Stage::Iface,
    );
    let check = CargoCheckTool { ctx };
    let out = check
        .call(bureau_rs::tools::CargoArgs {
            package: Some("not_a_real_module".into()),
        })
        .await
        .expect("cargo_check tool should not itself error on bad package");
    assert!(
        out.passed,
        "workspace cargo_check should succeed via package fallback; errors: {:?}",
        out.errors
    );
}

/// Same workspace but the model passes a valid workspace member name.
/// Pin that the resolver respects the model's choice when valid.
#[tokio::test]
async fn workspace_cargo_check_uses_valid_package_arg_verbatim() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut g = NodeGraph::new();
    let mut root = Node::new("ws", "");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    let mut util_node = Node::new("util", "");
    util_node.crate_boundary = true;
    let util_id = g.add_child(root_id, util_node).unwrap();
    let mut core_node = Node::new("core", "");
    core_node.crate_boundary = true;
    let _core_id = g.add_child(root_id, core_node).unwrap();
    render_graph(&workdir, &g, Layout::Workspace).unwrap();
    let graph = Arc::new(Mutex::new(g));
    let ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
        graph.clone(),
        util_id,
        Stage::Iface,
    );
    let check = CargoCheckTool { ctx };
    let out = check
        .call(bureau_rs::tools::CargoArgs {
            package: Some("util".into()),
        })
        .await
        .unwrap();
    assert!(
        out.passed,
        "explicit valid -p util should succeed; errors: {:?}",
        out.errors
    );
}

/// Regression: workspace mode with crate-boundary children must not
/// declare them as `pub mod` in the parent's mod.rs. Doing so makes
/// rustc look for `<name>/mod.rs` inside the parent's src/ tree and
/// fail with E0583 — the bug the user actually hit.
#[tokio::test]
async fn workspace_root_mod_rs_does_not_declare_crate_children_as_modules() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut g = NodeGraph::new();
    let mut root = Node::new("ws", "");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    let mut auth = Node::new("auth", "");
    auth.crate_boundary = true; // separate crate
    let _auth_id = g.add_child(root_id, auth).unwrap();
    let mut helper = Node::new("helper", ""); // module of root
    helper.crate_boundary = false;
    let _helper_id = g.add_child(root_id, helper).unwrap();
    render_graph(&workdir, &g, Layout::Workspace).unwrap();
    // Root's mod.rs lives at src/mod.rs in the root crate. It should
    // declare `pub mod helper;` (same crate) but NOT `pub mod auth;`
    // (separate crate).
    let root_mod = std::fs::read_to_string(workdir.join("src/mod.rs")).unwrap();
    assert!(
        root_mod.contains("pub mod helper;"),
        "in-crate child should be declared as a module: {root_mod}"
    );
    assert!(
        !root_mod.contains("pub mod auth;"),
        "crate-boundary child should NOT be declared as a module of the parent crate (would trigger E0583): {root_mod}"
    );
    // Cargo check the workspace — should pass cleanly.
    let graph = Arc::new(Mutex::new(g));
    let ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
        graph.clone(),
        root_id,
        Stage::Iface,
    );
    let check = CargoCheckTool { ctx };
    let out = check
        .call(bureau_rs::tools::CargoArgs { package: None })
        .await
        .unwrap();
    assert!(
        out.passed,
        "workspace with crate-boundary children should compile; errors: {:?}",
        out.errors
    );
}

/// Workspace with a cross-crate dep edge. Render it, verify the per-crate
/// `Cargo.toml` carries the path dependency, and that cargo can compile
/// both crates together.
#[tokio::test]
async fn workspace_with_cross_crate_dep_compiles() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut g = NodeGraph::new();
    let mut root = Node::new("ws", "");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    let mut util_node = Node::new("util", "");
    util_node.crate_boundary = true;
    let util_id = g.add_child(root_id, util_node).unwrap();
    let mut core_node = Node::new("core", "");
    core_node.crate_boundary = true;
    let core_id = g.add_child(root_id, core_node).unwrap();
    g.add_dep(core_id, util_id).unwrap();
    render_graph(&workdir, &g, Layout::Workspace).unwrap();
    // Verify util's Cargo.toml exists and core's manifest references util.
    let core_cargo = std::fs::read_to_string(workdir.join("crates/core/Cargo.toml")).unwrap();
    assert!(
        core_cargo.contains("util") && core_cargo.contains("path"),
        "core/Cargo.toml should declare a path dep on util:\n{core_cargo}"
    );
    // Author tiny content in util that core can call.
    let graph = Arc::new(Mutex::new(g));
    let util_ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
        graph.clone(),
        util_id,
        Stage::Iface,
    );
    SubmitPublicTool { ctx: util_ctx.clone() }
        .call(SubmitRustArgs {
            content: "pub trait Answer { fn answer(&self) -> u32; }\npub struct A;\n".into(),
        })
        .await
        .unwrap();
    SubmitPrivateTool { ctx: util_ctx }
        .call(SubmitRustArgs {
            content: "use super::public::*;\nimpl Answer for super::A { fn answer(&self) -> u32 { 42 } }\n".into(),
        })
        .await
        .unwrap();
    // Mark util's iface Done so core's iface can declare a dep on it.
    graph.lock().get_mut(util_id).unwrap().stages.iface = StageState::Done;
    let core_ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
        graph.clone(),
        core_id,
        Stage::Iface,
    );
    SubmitPublicTool { ctx: core_ctx.clone() }
        .call(SubmitRustArgs {
            content: "pub trait Core { fn run(&self) -> u32; }\npub struct CoreImpl;\n".into(),
        })
        .await
        .unwrap();
    SubmitPrivateTool { ctx: core_ctx.clone() }
        .call(SubmitRustArgs {
            content: "use super::public::*;\nuse util::Answer;\nimpl Core for super::CoreImpl { fn run(&self) -> u32 { <util::A as Answer>::answer(&util::A) } }\n".into(),
        })
        .await
        .unwrap();
    // Run cargo_check on core. Should pass.
    let check = CargoCheckTool { ctx: core_ctx };
    let out = check
        .call(bureau_rs::tools::CargoArgs {
            package: Some("core".into()),
        })
        .await
        .unwrap();
    assert!(
        out.passed,
        "core+util workspace should compile; errors: {:?}",
        out.errors
    );
}

/// Regression: two parallel worktrees both rendering the full graph
/// in their own scratch dirs should both successfully land their
/// per-task changes on main without conflict. The previous code did a
/// three-way merge and tripped on "both branches modified the same
/// file with different content" (because each worktree's full-tree
/// render included other nodes' content at slightly different graph
/// snapshots). The fix: only land each task's OWNED files via
/// `apply_to_main`.
#[tokio::test]
async fn parallel_tasks_land_without_merge_conflicts() {
    use bureau_rs::worktree::{Workspace, WorktreePool};
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    // Build a tiny workspace: root + two leaf modules.
    let mut g = NodeGraph::new();
    let mut root = Node::new("p", "");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    let alpha_id = g.add_child(root_id, Node::new("alpha", "")).unwrap();
    let beta_id = g.add_child(root_id, Node::new("beta", "")).unwrap();
    render_graph(&workdir, &g, Layout::SingleCrate).unwrap();
    let workspace = Workspace::init(&workdir).unwrap();
    workspace.commit_main("scaffold: initial render").unwrap();
    let pool = Arc::new(WorktreePool::new(workspace.clone()).unwrap());
    let graph = Arc::new(Mutex::new(g));
    // Allocate two worktrees; in each, write spec content for the
    // task's own node (the bug used to surface here because each
    // worktree's full-tree render captured the OTHER node's not-yet-
    // landed spec content from the shared graph).
    let wta = pool.allocate(Uuid::new_v4()).await.unwrap();
    let wtb = pool.allocate(Uuid::new_v4()).await.unwrap();
    // Task A: spec for alpha. Task B: spec for beta. Apply both via
    // their own ctx, which renders to their own worktree.
    let ctx_a = ctx_for(
        wta.path.clone(),
        Layout::SingleCrate,
        graph.clone(),
        alpha_id,
        Stage::Spec,
    );
    let ctx_b = ctx_for(
        wtb.path.clone(),
        Layout::SingleCrate,
        graph.clone(),
        beta_id,
        Stage::Spec,
    );
    bureau_rs::tools::SubmitSpecTool { ctx: ctx_a }
        .call(bureau_rs::tools::SubmitSpecArgs {
            public: "# alpha\n\nFirst.".into(),
            private: None,
            deps: vec![],
        })
        .await
        .unwrap();
    bureau_rs::tools::SubmitSpecTool { ctx: ctx_b }
        .call(bureau_rs::tools::SubmitSpecArgs {
            public: "# beta\n\nSecond.".into(),
            private: None,
            deps: vec![],
        })
        .await
        .unwrap();
    // Re-render canonical state into both worktrees (mirrors what the
    // engine does just before landing). Both worktrees now contain
    // BOTH specs (full-tree render reads the shared graph).
    {
        let g_lock = graph.lock();
        render_graph(&wta.path, &g_lock, Layout::SingleCrate).unwrap();
        render_graph(&wtb.path, &g_lock, Layout::SingleCrate).unwrap();
    }
    // Land just A's owned files first. Then B's. Old code: three-way
    // merge would trip because both branches added beta's spec.md
    // (with the same content here, but in real runs the content
    // differs per snapshot). New code: only A's own files land.
    let alpha_owned = {
        let g_lock = graph.lock();
        let n = g_lock.get(alpha_id).unwrap();
        bureau_rs::render::files_owned_by_stage(&g_lock, n, Stage::Spec, Layout::SingleCrate)
    };
    pool.apply_to_main(wta, &alpha_owned, "spec: alpha")
        .await
        .expect("first apply should succeed");
    let beta_owned = {
        let g_lock = graph.lock();
        let n = g_lock.get(beta_id).unwrap();
        bureau_rs::render::files_owned_by_stage(&g_lock, n, Stage::Spec, Layout::SingleCrate)
    };
    pool.apply_to_main(wtb, &beta_owned, "spec: beta")
        .await
        .expect("second apply should succeed without conflict");
    // Both specs should be on main.
    let alpha_md =
        std::fs::read_to_string(workdir.join("spec/p/alpha/public.md")).unwrap();
    assert!(alpha_md.contains("First."));
    let beta_md =
        std::fs::read_to_string(workdir.join("spec/p/beta/public.md")).unwrap();
    assert!(beta_md.contains("Second."));
}
