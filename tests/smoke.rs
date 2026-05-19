//! End-to-end smoke tests for the new node-stage engine. The legacy
//! tests/tools_test.rs was deleted along with the old phase-based
//! orchestrator.

use bureau_rs::graph::{Node, NodeGraph, Stage, StageState};
use bureau_rs::node_context;
use bureau_rs::node_validate;
use bureau_rs::render::{Layout, render_graph};
use std::path::PathBuf;

#[test]
fn empty_graph_renders_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let g = NodeGraph::new();
    let report = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
    assert!(report.files_written.is_empty());
}

#[test]
fn single_node_renders_full_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let mut g = NodeGraph::new();
    let _root = g
        .insert_root(Node::new("app", "the application"))
        .unwrap();
    render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
    assert!(tmp.path().join("Cargo.toml").exists());
    assert!(tmp.path().join("src/mod.rs").exists());
    assert!(tmp.path().join("src/public.rs").exists());
    assert!(tmp.path().join("src/private.rs").exists());
    assert!(tmp.path().join("src/tests.rs").exists());
    assert!(tmp.path().join("spec/app/public.md").exists());
}

#[test]
fn workspace_render_with_member_crate() {
    let tmp = tempfile::tempdir().unwrap();
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("ws", "workspace root")).unwrap();
    let mut server = Node::new("server", "the server crate");
    server.crate_boundary = true;
    let _server = g.add_child(root, server).unwrap();
    render_graph(tmp.path(), &g, Layout::Workspace).unwrap();
    let cargo = std::fs::read_to_string(tmp.path().join("Cargo.toml")).unwrap();
    assert!(cargo.contains("[workspace]"));
    assert!(cargo.contains("\"crates/server\""));
    assert!(tmp.path().join("crates/server/src/mod.rs").exists());
}

#[test]
fn graph_serializes_round_trip() {
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let a = g.add_child(root, Node::new("a", "")).unwrap();
    let b = g.add_child(root, Node::new("b", "")).unwrap();
    g.add_dep(a, b).unwrap();
    g.get_mut(a).unwrap().stages.spec = StageState::Done;
    let json = serde_json::to_string(&g).unwrap();
    let g2: NodeGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(g2.len(), 3);
    assert_eq!(g2.get(a).unwrap().deps, vec![b]);
    assert_eq!(g2.get(a).unwrap().stages.spec, StageState::Done);
}

#[test]
fn dep_cycle_rejected() {
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let a = g.add_child(root, Node::new("a", "")).unwrap();
    let b = g.add_child(root, Node::new("b", "")).unwrap();
    let c = g.add_child(root, Node::new("c", "")).unwrap();
    g.add_dep(a, b).unwrap();
    g.add_dep(b, c).unwrap();
    let err = g.add_dep(c, a).unwrap_err();
    assert!(matches!(err, bureau_rs::graph::GraphError::WouldCycle { .. }));
}

#[test]
fn public_validator_accepts_well_formed_iface() {
    let src = r#"
pub trait Frob {
    fn shape(&self) -> i32;
}

pub struct Frobber(super::private::FrobInner);

pub enum Color { Red, Green }
"#;
    node_validate::validate_public(src).unwrap();
}

#[test]
fn public_validator_rejects_impl_block() {
    let src = "pub struct X; impl X { pub fn n() -> Self { X } }";
    assert!(node_validate::validate_public(src).is_err());
}

#[test]
fn private_validator_rejects_undeclared_dep() {
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let a = g.add_child(root, Node::new("a", "")).unwrap();
    let _b = g.add_child(root, Node::new("b", "")).unwrap();
    // a doesn't declare a dep on b
    let src = "use crate::b::Stuff;";
    let err = node_validate::validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
    assert!(matches!(err, node_validate::ValidateError::PrivateUndeclaredDep { .. }));
}

#[test]
fn context_for_iface_inlines_dep_public_rs() {
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let mut errs = Node::new("errors", "shared error types");
    errs.public_rs = Some("pub enum Err { NotFound }\n".into());
    let errs_id = g.add_child(root, errs).unwrap();
    let user_id = g.add_child(root, Node::new("user", "uses errors")).unwrap();
    g.add_dep(user_id, errs_id).unwrap();
    let bundle = node_context::build_for_iface(&g, user_id, bureau_rs::render::Layout::SingleCrate);
    let md = bundle.to_markdown();
    assert!(md.contains("Dependency `errors`"));
    assert!(md.contains("pub enum Err"));
}

#[test]
fn module_path_for_workspace_node() {
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let frontend = g.add_child(root, Node::new("frontend", "")).unwrap();
    let router = g.add_child(frontend, Node::new("router", "")).unwrap();
    assert_eq!(node_context::module_path(&g, root), "crate");
    assert_eq!(node_context::module_path(&g, frontend), "crate::frontend");
    assert_eq!(node_context::module_path(&g, router), "crate::frontend::router");
}

#[test]
fn render_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let _ = g.add_child(root, Node::new("a", "")).unwrap();
    render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
    let r2 = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
    assert!(r2.files_written.is_empty());
}

#[test]
fn topological_order_respects_dep_graph() {
    let mut g = NodeGraph::new();
    let root = g.insert_root(Node::new("app", "")).unwrap();
    let a = g.add_child(root, Node::new("a", "")).unwrap();
    let b = g.add_child(root, Node::new("b", "")).unwrap();
    g.add_dep(a, b).unwrap();
    let order = g.topo_order().unwrap();
    let pos_a = order.iter().position(|x| *x == a).unwrap();
    let pos_b = order.iter().position(|x| *x == b).unwrap();
    assert!(pos_b < pos_a, "b must come before a in topo order");
}

#[test]
fn stage_state_round_trip_via_serde() {
    use bureau_rs::graph::NodeStages;
    let mut s = NodeStages::default();
    for st in Stage::ALL {
        s.set(st, StageState::Done);
    }
    let j = serde_json::to_string(&s).unwrap();
    let s2: NodeStages = serde_json::from_str(&j).unwrap();
    for st in Stage::ALL {
        assert_eq!(s2.get(st), StageState::Done);
    }
}

#[test]
fn render_writes_authored_content() {
    let tmp = tempfile::tempdir().unwrap();
    let mut g = NodeGraph::new();
    let mut root = Node::new("app", "");
    root.public_rs = Some("pub trait App {}\n".into());
    root.spec_public_md = Some("# App spec\n\nReal content.\n".into());
    let _id = g.insert_root(root).unwrap();
    render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
    let public_rs = std::fs::read_to_string(tmp.path().join("src/public.rs")).unwrap();
    assert!(public_rs.contains("pub trait App"));
    let spec_md = std::fs::read_to_string(tmp.path().join("spec/app/public.md")).unwrap();
    assert!(spec_md.contains("Real content"));
}

// Marker so reorganization is obvious in test output.
#[test]
fn _smoke_module_loads() {
    // Force usage of a few public APIs to catch link errors early.
    let mut g = NodeGraph::new();
    let _ = g.insert_root(Node::new("app", ""));
    let _: PathBuf = bureau_rs::render::node_spec_dir(&g, g.iter().next().unwrap());
}
