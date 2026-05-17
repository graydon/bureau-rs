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
use rig::tool::Tool;
use std::sync::Arc;
use uuid::Uuid;

fn ctx_for(
    workdir: std::path::PathBuf,
    layout: Layout,
    node_id: bureau_rs::graph::NodeId,
    stage: Stage,
) -> Arc<TaskCtx> {
    Arc::new(TaskCtx::new(
        Uuid::new_v4(),
        node_id,
        stage,
        Role::Writer,
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
    // Need iface stage (cargo_check is registered there) and content
    // that compiles. Author public.rs with a trivial trait.
    let ctx = ctx_for(workdir.clone(), Layout::SingleCrate, root_id, Stage::Iface);
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
    // Run cargo_check; the model passes a BAD `package` arg (a module
    // name that doesn't exist as a workspace member). The framework
    // should resolve it to the current node's containing crate (`core`)
    // and run cargo successfully.
    let ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
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
    let ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
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
    let ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
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
    let util_ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
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
    let mut g = bureau_rs::graph::load(&workdir).unwrap();
    g.get_mut(util_id).unwrap().stages.iface = StageState::Done;
    render_graph(&workdir, &g, Layout::Workspace).unwrap();
    let core_ctx = ctx_for(
        workdir.clone(),
        Layout::Workspace,
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

/// Regression: two parallel worktrees both authoring different nodes'
/// spec content should both land successfully on main. Under the new
/// rebase + ff-merge model the second-lander rebases its branch atop
/// the first-lander's tip (which brings in the first task's
/// `.bureau/nodes/alpha.json` cleanly since the second task didn't
/// touch it) and re-runs the gate before fast-forwarding.
#[tokio::test]
async fn parallel_tasks_land_without_merge_conflicts() {
    use bureau_rs::worktree::{Workspace, WorktreePool};
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
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
    // Each worktree gets its own per-task graph state, loaded fresh
    // from disk at allocate time.
    let wta = pool.allocate(Uuid::new_v4()).await.unwrap();
    let wtb = pool.allocate(Uuid::new_v4()).await.unwrap();
    let ctx_a = ctx_for(wta.path.clone(), Layout::SingleCrate, alpha_id, Stage::Spec);
    let ctx_b = ctx_for(wtb.path.clone(), Layout::SingleCrate, beta_id, Stage::Spec);
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
    // Land A first: commit, rebase (no-op — no other landings since
    // allocate), ff-merge.
    pool.commit_in_worktree(&wta, "spec: alpha").unwrap();
    workspace
        .rebase_branch_onto_main(&wta.path, &wta.branch)
        .unwrap();
    workspace.fast_forward_main(&wta.branch).unwrap();
    pool.abandon(wta).await.unwrap();
    // Land B: it was allocated from PRE-A main. Rebase brings in A's
    // changes, then ff-merge.
    pool.commit_in_worktree(&wtb, "spec: beta").unwrap();
    workspace
        .rebase_branch_onto_main(&wtb.path, &wtb.branch)
        .unwrap();
    workspace.fast_forward_main(&wtb.branch).unwrap();
    pool.abandon(wtb).await.unwrap();
    let alpha_md =
        std::fs::read_to_string(workdir.join("spec/p/alpha/public.md")).unwrap();
    assert!(alpha_md.contains("First."));
    let beta_md =
        std::fs::read_to_string(workdir.join("spec/p/beta/public.md")).unwrap();
    assert!(beta_md.contains("Second."));
}

/// Regression: the iface-stage gate is `cargo check` on the worktree.
/// Without `--workspace` cargo only compiles the root package, so a
/// member crate (under `crates/<name>/`) can have unresolved imports
/// and the gate still reports `passed = true`. The framework would
/// mark the stage Done and ff-merge broken code onto main.
#[tokio::test]
async fn cargo_check_gate_catches_broken_member_crate() {
    use bureau_rs::gate::{GateKind, run_gate};

    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    // Workspace with one member crate `child`.
    let mut g = NodeGraph::new();
    let mut root = Node::new("root", "");
    root.crate_boundary = true;
    let root_id = g.insert_root(root).unwrap();
    let mut child = Node::new("child", "");
    child.crate_boundary = true;
    // Make the child's private.rs reference a crate (`nonexistent_crate_xyz`)
    // that's not in any Cargo.toml — this is exactly the failure mode
    // the user hit when the model wrote `use hmac::Hmac;` without
    // declaring hmac as an external dep.
    child.private_rs =
        Some("use nonexistent_crate_xyz::Whatever;\npub fn x() -> Whatever { todo!() }\n".into());
    let _child_id = g.add_child(root_id, child).unwrap();
    render_graph(&workdir, &g, Layout::Workspace).unwrap();
    // The root crate compiles fine in isolation. The bug was that the
    // gate also reported passed for the workspace because it didn't
    // pass `--workspace`. With the fix, it should fail.
    let outcome = run_gate(&workdir, GateKind::Check).await.unwrap();
    assert!(
        !outcome.passed,
        "cargo check gate must catch broken member crate; outcome:\n{:#?}",
        outcome.errors
    );
    assert!(
        !outcome.errors.is_empty(),
        "expected at least one cargo error; got passed={}",
        outcome.passed
    );
}
