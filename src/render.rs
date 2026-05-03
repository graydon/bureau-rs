//! Render a `NodeGraph` onto the filesystem.
//!
//! For each node in the graph this produces:
//!
//! - `<workdir>/src/<path>/public.rs`   — public surface (model-authored)
//! - `<workdir>/src/<path>/private.rs`  — hidden internals (model-authored)
//! - `<workdir>/src/<path>/tests.rs`    — tests (model-authored)
//! - `<workdir>/src/<path>/mod.rs`      — framework-rendered glue
//! - `<workdir>/spec/<path>/spec.md`    — markdown spec (model-authored)
//!
//! For nodes marked `crate_boundary = true`, additionally:
//!
//! - `<workdir>/<crate_root>/Cargo.toml` (a per-crate manifest)
//! - root-level `Cargo.toml` declares the workspace.
//!
//! `mod.rs` is the only file the framework writes substantively. Everything
//! else is the model's content if present, or a tiny placeholder if not yet
//! written. `mod.rs` declares the three required submodules and re-exports
//! `public::*`. It also declares `pub mod <child>;` for each child node so
//! cross-cutting deps can reach it.
//!
//! Rendering is **idempotent**: re-rendering a node with the same content
//! and the same children produces the same files. We never touch
//! model-authored content; if `node.public_rs` is `Some(s)`, we write `s`.
//! If it's `None`, we write a minimal placeholder so `cargo check` still
//! works incrementally.

use crate::graph::{Node, NodeGraph, NodeId};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Layout choice: single crate at the workdir root, or workspace where each
/// `crate_boundary` node becomes its own crate under `crates/<name>/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    SingleCrate,
    Workspace,
}

/// What the renderer wrote, returned for logging/UI purposes.
#[derive(Debug, Clone, Default)]
pub struct RenderReport {
    pub files_written: Vec<PathBuf>,
    pub files_unchanged: Vec<PathBuf>,
}

/// Render the entire graph to `workdir`. Existing files are overwritten only
/// if their content would change.
pub fn render_graph(workdir: &Path, graph: &NodeGraph, layout: Layout) -> Result<RenderReport> {
    let Some(root_id) = graph.root else {
        // Nothing to render; that's OK.
        return Ok(RenderReport::default());
    };

    let mut report = RenderReport::default();

    // 1. Workspace root Cargo.toml (workspace mode) or single Cargo.toml.
    write_root_manifest(workdir, graph, root_id, layout, &mut report)?;

    // 2. Per-node files.
    for node in graph.iter() {
        render_node(workdir, graph, node, layout, &mut report)?;
    }

    Ok(report)
}

/// On-disk path of a node's source directory, relative to `workdir`.
pub fn node_src_dir(graph: &NodeGraph, node: &Node, layout: Layout) -> PathBuf {
    let Some(path) = graph.name_path(node.id) else {
        return PathBuf::new();
    };
    // The root's "name" is the package name (single-crate) or the workspace
    // name; either way the root's source goes at `src/` (single crate) or
    // `crates/<root>/src/` is wrong because the root isn't a crate child;
    // it lives at the top in workspace mode too — but with workspace mode,
    // additional crate-boundary descendants live under `crates/<name>/`.
    //
    // Layout rules (v1):
    //   - Single crate: every node is a module under `src/<path>/` where
    //     <path> is the chain below the root.
    //   - Workspace: the root contains a workspace Cargo.toml plus its own
    //     src tree as the "umbrella" crate (named after the root). Nodes
    //     marked `crate_boundary` (other than root) become independent
    //     crates at `crates/<name>/src/...` with their own descendant
    //     modules nested under them.
    match layout {
        Layout::SingleCrate => {
            let mut p = PathBuf::from("src");
            for seg in path.iter().skip(1) {
                p.push(seg);
            }
            p
        }
        Layout::Workspace => {
            // Find the deepest crate-boundary ancestor (or the node itself).
            let mut chain_ids: Vec<NodeId> = Vec::new();
            let mut current: Option<NodeId> = Some(node.id);
            while let Some(c) = current {
                chain_ids.push(c);
                current = graph.get(c).and_then(|n| n.parent);
            }
            chain_ids.reverse();
            // Walk the chain and split into "[outer] / crates/<name> / [inner]".
            // The root counts as a crate.
            let mut p = PathBuf::new();
            let mut inside_crate = false;
            for (i, id) in chain_ids.iter().enumerate() {
                let n = graph.get(*id).expect("node in chain");
                if i == 0 {
                    // Root: lives at workdir root with `src/`.
                    inside_crate = true;
                    p.push("src");
                    continue;
                }
                if n.crate_boundary {
                    // Member crate.
                    p = PathBuf::from("crates").join(&n.name).join("src");
                    inside_crate = true;
                } else if inside_crate {
                    p.push(&n.name);
                }
            }
            p
        }
    }
}

/// The nearest crate-boundary ancestor of `node`, inclusive. Always returns
/// some node (the root is always a crate boundary).
pub fn containing_crate(graph: &NodeGraph, node: &Node) -> NodeId {
    let mut cur = Some(node.id);
    while let Some(id) = cur {
        let n = graph.get(id).expect("walk");
        if n.crate_boundary {
            return id;
        }
        cur = n.parent;
    }
    // Should be unreachable.
    node.id
}

/// Compute the set of *other* crates that nodes within `crate_id` collectively
/// depend on. Used to render `[dependencies]` in the per-crate Cargo.toml.
pub fn cross_crate_dep_targets(graph: &NodeGraph, crate_id: NodeId) -> Vec<NodeId> {
    use std::collections::HashSet;
    let mut targets: HashSet<NodeId> = HashSet::new();
    for n in graph.iter() {
        if containing_crate(graph, n) != crate_id {
            continue;
        }
        for d in &n.deps {
            let Some(dep) = graph.get(*d) else { continue };
            let dep_crate = containing_crate(graph, dep);
            if dep_crate != crate_id {
                targets.insert(dep_crate);
            }
        }
    }
    let mut v: Vec<NodeId> = targets.into_iter().collect();
    // Stable order by name for deterministic Cargo.toml.
    v.sort_by_key(|id| graph.get(*id).map(|n| n.name.clone()).unwrap_or_default());
    v
}

/// Containing crate-root directory (the directory holding Cargo.toml) for
/// `node`. For the root node in single-crate mode this is the workdir.
pub fn node_crate_root(graph: &NodeGraph, node: &Node, layout: Layout) -> PathBuf {
    match layout {
        Layout::SingleCrate => PathBuf::new(),
        Layout::Workspace => {
            // Walk up to the nearest crate boundary.
            let mut current = Some(node.id);
            while let Some(id) = current {
                let n = graph.get(id).expect("walk");
                if n.crate_boundary {
                    if n.parent.is_none() {
                        return PathBuf::new(); // root crate at workdir root
                    } else {
                        return PathBuf::from("crates").join(&n.name);
                    }
                }
                current = n.parent;
            }
            PathBuf::new()
        }
    }
}

/// Spec markdown lives in a separate tree under `<workdir>/spec/<name-path>/`.
pub fn node_spec_dir(graph: &NodeGraph, node: &Node) -> PathBuf {
    let Some(path) = graph.name_path(node.id) else {
        return PathBuf::new();
    };
    let mut p = PathBuf::from("spec");
    for seg in path {
        p.push(seg);
    }
    p
}

fn render_node(
    workdir: &Path,
    graph: &NodeGraph,
    node: &Node,
    layout: Layout,
    report: &mut RenderReport,
) -> Result<()> {
    // Per-crate Cargo.toml for non-root crate boundaries (workspace mode).
    if matches!(layout, Layout::Workspace) && node.crate_boundary && node.parent.is_some() {
        let crate_dir = workdir.join("crates").join(&node.name);
        let manifest = crate_dir.join("Cargo.toml");
        let deps = render_cross_crate_deps(graph, node.id);
        let content = format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
             [lib]\npath = \"src/mod.rs\"\n{deps}",
            node.name
        );
        write_if_changed(&manifest, &content, report)?;
    }

    // Source directory.
    let src_dir = workdir.join(node_src_dir(graph, node, layout));
    let public_path = src_dir.join("public.rs");
    let private_path = src_dir.join("private.rs");
    let tests_path = src_dir.join("tests.rs");
    let mod_path = src_dir.join("mod.rs");

    // public.rs / private.rs / tests.rs: write model content if any, else
    // a minimal placeholder (no items) so the module compiles incrementally.
    write_if_changed(
        &public_path,
        node.public_rs
            .as_deref()
            .unwrap_or("// public surface — not yet authored\n"),
        report,
    )?;
    write_if_changed(
        &private_path,
        node.private_rs
            .as_deref()
            .unwrap_or("// private internals — not yet authored\n"),
        report,
    )?;
    write_if_changed(
        &tests_path,
        node.tests_rs
            .as_deref()
            .unwrap_or("// tests — not yet authored\n"),
        report,
    )?;

    // mod.rs: framework-rendered.
    let mod_content = render_mod_rs(graph, node);
    write_if_changed(&mod_path, &mod_content, report)?;

    // Spec markdown lives under <workdir>/spec/<name-path>/spec.md.
    let spec_dir = workdir.join(node_spec_dir(graph, node));
    let spec_path = spec_dir.join("spec.md");
    let spec_content = node
        .spec_md
        .clone()
        .unwrap_or_else(|| format!("# {}\n\n{}\n\n*spec not yet authored*\n", node.name, node.description));
    write_if_changed(&spec_path, &spec_content, report)?;

    Ok(())
}

/// Render `mod.rs` for a node. Always:
///
/// ```ignore
/// mod public;
/// mod private;
/// #[cfg(test)] mod tests;
/// pub use public::*;
/// ```
///
/// Plus `pub mod <child>;` for each child node.
fn render_mod_rs(graph: &NodeGraph, node: &Node) -> String {
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED by bureau-rs. Do not edit by hand.\n");
    s.push_str("// This file declares the node's submodules and re-exports its public surface.\n\n");
    s.push_str("mod public;\n");
    s.push_str("mod private;\n");
    s.push_str("#[cfg(test)]\nmod tests;\n");
    s.push_str("pub use public::*;\n");
    let children = graph.children_of(node.id);
    if !children.is_empty() {
        s.push('\n');
        s.push_str("// Children (sub-decompositions).\n");
        for child in children {
            // Children are `pub mod` so cross-cutting deps from elsewhere in
            // the graph can reach them via `crate::a::b::child`. Their own
            // visibility is governed by what they put in their public.rs.
            s.push_str(&format!("pub mod {};\n", child.name));
        }
    }
    s
}

fn render_cross_crate_deps(graph: &NodeGraph, crate_id: NodeId) -> String {
    let targets = cross_crate_dep_targets(graph, crate_id);
    if targets.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n[dependencies]\n");
    for tid in targets {
        let Some(t) = graph.get(tid) else { continue };
        // For now, every member crate lives at `crates/<name>` and the root
        // crate lives at the workspace root. Path-deps are relative to the
        // dependent's own Cargo.toml.
        let path_str = if t.parent.is_none() {
            // Depending on the workspace root crate from a member crate at
            // `crates/<name>`: relative path is `../..`.
            "../..".to_string()
        } else {
            // Depending on another member crate `crates/<other>`: relative
            // path from `crates/<this>` is `../<other>`.
            format!("../{}", t.name)
        };
        out.push_str(&format!("{} = {{ path = \"{}\" }}\n", t.name, path_str));
    }
    out
}

fn write_root_manifest(
    workdir: &Path,
    graph: &NodeGraph,
    root_id: NodeId,
    layout: Layout,
    report: &mut RenderReport,
) -> Result<()> {
    let root = graph.get(root_id).expect("root in graph");
    let manifest = workdir.join("Cargo.toml");
    let content = match layout {
        Layout::SingleCrate => format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/mod.rs\"\n",
            root.name
        ),
        Layout::Workspace => {
            let mut members: Vec<String> = vec![".".to_string()];
            for n in graph.iter() {
                if n.crate_boundary && n.parent.is_some() {
                    members.push(format!("crates/{}", n.name));
                }
            }
            members.sort();
            let members_list = members
                .iter()
                .map(|m| format!("    \"{}\"", m))
                .collect::<Vec<_>>()
                .join(",\n");
            // Root crate may need to depend on member crates if its
            // descendants declare deps that cross into them. Render those
            // here as path = "crates/<name>".
            let root_deps_section = {
                let targets = cross_crate_dep_targets(graph, root_id);
                if targets.is_empty() {
                    String::new()
                } else {
                    let mut s = String::from("\n[dependencies]\n");
                    for tid in targets {
                        if let Some(t) = graph.get(tid) {
                            // Member crates live at crates/<name>.
                            let path = if t.parent.is_none() {
                                ".".to_string()
                            } else {
                                format!("crates/{}", t.name)
                            };
                            s.push_str(&format!(
                                "{} = {{ path = \"{}\" }}\n",
                                t.name, path
                            ));
                        }
                    }
                    s
                }
            };
            format!(
                "[workspace]\nresolver = \"2\"\nmembers = [\n{}\n]\n\n[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/mod.rs\"\n{root_deps_section}",
                members_list, root.name
            )
        }
    };
    write_if_changed(&manifest, &content, report)
}

fn write_if_changed(path: &Path, content: &str, report: &mut RenderReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating dir {}", parent.display()))?;
    }
    let needs_write = match std::fs::read_to_string(path) {
        Ok(existing) => existing != content,
        Err(_) => true,
    };
    if needs_write {
        std::fs::write(path, content)
            .with_context(|| format!("writing {}", path.display()))?;
        report.files_written.push(path.to_path_buf());
    } else {
        report.files_unchanged.push(path.to_path_buf());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;

    fn graph_with_two_children() -> (NodeGraph, NodeId, NodeId, NodeId) {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "the app")).unwrap();
        let a = g.add_child(root, Node::new("a", "node A")).unwrap();
        let b = g.add_child(root, Node::new("b", "node B")).unwrap();
        (g, root, a, b)
    }

    #[test]
    fn renders_single_crate_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let (g, root, a, b) = graph_with_two_children();
        let report = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        // Root files
        assert!(tmp.path().join("Cargo.toml").exists());
        assert!(tmp.path().join("src/mod.rs").exists());
        assert!(tmp.path().join("src/public.rs").exists());
        assert!(tmp.path().join("src/private.rs").exists());
        assert!(tmp.path().join("src/tests.rs").exists());
        // Children
        assert!(tmp.path().join("src/a/mod.rs").exists());
        assert!(tmp.path().join("src/a/public.rs").exists());
        assert!(tmp.path().join("src/b/mod.rs").exists());
        // Specs in the parallel tree
        assert!(tmp.path().join("spec/app/spec.md").exists());
        assert!(tmp.path().join("spec/app/a/spec.md").exists());
        assert!(tmp.path().join("spec/app/b/spec.md").exists());
        assert!(!report.files_written.is_empty());
        // Root mod.rs lists children as pub mods
        let root_mod = std::fs::read_to_string(tmp.path().join("src/mod.rs")).unwrap();
        assert!(root_mod.contains("pub mod a;"));
        assert!(root_mod.contains("pub mod b;"));
        assert!(root_mod.contains("pub use public::*;"));
        // Cargo.toml has the right package name
        let cargo = std::fs::read_to_string(tmp.path().join("Cargo.toml")).unwrap();
        assert!(cargo.contains("name = \"app\""));
        let _ = (root, a, b);
    }

    #[test]
    fn renders_workspace_layout_with_member_crate() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "umbrella")).unwrap();
        let mut server_node = Node::new("server", "the server crate");
        server_node.crate_boundary = true;
        let server = g.add_child(root, server_node).unwrap();
        let _handler = g
            .add_child(server, Node::new("handler", "request handler"))
            .unwrap();

        render_graph(tmp.path(), &g, Layout::Workspace).unwrap();
        // Workspace Cargo.toml at root with members.
        let root_cargo = std::fs::read_to_string(tmp.path().join("Cargo.toml")).unwrap();
        assert!(root_cargo.contains("[workspace]"));
        assert!(root_cargo.contains("\"crates/server\""));
        // Member crate has its own Cargo.toml + src tree.
        assert!(tmp
            .path()
            .join("crates/server/Cargo.toml")
            .exists());
        assert!(tmp
            .path()
            .join("crates/server/src/mod.rs")
            .exists());
        assert!(tmp
            .path()
            .join("crates/server/src/public.rs")
            .exists());
        // The handler is a module under server's src tree (NOT a separate crate).
        assert!(tmp
            .path()
            .join("crates/server/src/handler/mod.rs")
            .exists());
        // Server's mod.rs declares `pub mod handler;`.
        let server_mod =
            std::fs::read_to_string(tmp.path().join("crates/server/src/mod.rs")).unwrap();
        assert!(server_mod.contains("pub mod handler;"));
    }

    #[test]
    fn renders_idempotently() {
        let tmp = tempfile::tempdir().unwrap();
        let (g, _, _, _) = graph_with_two_children();
        let r1 = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let written_first = r1.files_written.len();
        let r2 = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        assert_eq!(r2.files_written.len(), 0, "second render writes nothing");
        assert!(r2.files_unchanged.len() == written_first);
    }

    #[test]
    fn writes_authored_content_when_present() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "the app");
        root.public_rs = Some("pub trait App {}\n".to_string());
        root.private_rs = Some("// nothing yet\n".to_string());
        root.tests_rs = Some("#[test] fn ok() {}\n".to_string());
        root.spec_md = Some("# App spec\n\nThe app does things.\n".to_string());
        let _root_id = g.insert_root(root).unwrap();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let pub_ = std::fs::read_to_string(tmp.path().join("src/public.rs")).unwrap();
        assert!(pub_.contains("pub trait App"));
        let spec = std::fs::read_to_string(tmp.path().join("spec/app/spec.md")).unwrap();
        assert!(spec.contains("App spec"));
    }

    #[test]
    fn placeholder_for_unauthored_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (g, _, _, _) = graph_with_two_children();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let pub_ = std::fs::read_to_string(tmp.path().join("src/public.rs")).unwrap();
        assert!(pub_.contains("not yet authored"));
    }

    #[test]
    fn renders_empty_graph_safely() {
        let tmp = tempfile::tempdir().unwrap();
        let g = NodeGraph::new();
        let report = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        assert!(report.files_written.is_empty());
        assert!(report.files_unchanged.is_empty());
    }

    #[test]
    fn deeply_nested_module_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let b = g.add_child(a, Node::new("b", "")).unwrap();
        let _c = g.add_child(b, Node::new("c", "")).unwrap();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        assert!(tmp.path().join("src/a/b/c/mod.rs").exists());
        assert!(tmp.path().join("spec/app/a/b/c/spec.md").exists());
    }

    #[test]
    fn workspace_renders_cross_crate_path_deps() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("ws", "umbrella")).unwrap();
        // Two member crates; `app` depends on `errors`.
        let mut errors_node = Node::new("errors", "shared error types");
        errors_node.crate_boundary = true;
        let errors = g.add_child(root, errors_node).unwrap();
        let mut app_node = Node::new("app", "main app crate");
        app_node.crate_boundary = true;
        let app = g.add_child(root, app_node).unwrap();
        g.add_dep(app, errors).unwrap();
        render_graph(tmp.path(), &g, Layout::Workspace).unwrap();
        // app's manifest should declare `errors = { path = "../errors" }`.
        let app_cargo =
            std::fs::read_to_string(tmp.path().join("crates/app/Cargo.toml")).unwrap();
        assert!(
            app_cargo.contains("[dependencies]"),
            "expected [dependencies] in app's Cargo.toml; got:\n{app_cargo}"
        );
        assert!(
            app_cargo.contains("errors = { path = \"../errors\" }"),
            "expected path-dep on errors; got:\n{app_cargo}"
        );
        // errors's manifest should NOT have [dependencies] (no outgoing deps).
        let err_cargo =
            std::fs::read_to_string(tmp.path().join("crates/errors/Cargo.toml")).unwrap();
        assert!(
            !err_cargo.contains("[dependencies]"),
            "errors crate has no deps; got:\n{err_cargo}"
        );
    }

    #[test]
    fn workspace_root_dep_on_member_renders_as_crates_path() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("ws", "umbrella")).unwrap();
        let mut helper_node = Node::new("helper", "");
        helper_node.crate_boundary = true;
        let helper = g.add_child(root, helper_node).unwrap();
        // A non-crate child of the root depends on the helper crate.
        let inner = g.add_child(root, Node::new("inner", "")).unwrap();
        g.add_dep(inner, helper).unwrap();
        render_graph(tmp.path(), &g, Layout::Workspace).unwrap();
        // root's Cargo.toml should declare helper = { path = "crates/helper" }.
        let root_cargo = std::fs::read_to_string(tmp.path().join("Cargo.toml")).unwrap();
        assert!(
            root_cargo.contains("helper = { path = \"crates/helper\" }"),
            "expected root crate dep; got:\n{root_cargo}"
        );
    }

    #[test]
    fn workspace_member_with_grandchildren() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("ws", "")).unwrap();
        let mut server = Node::new("server", "");
        server.crate_boundary = true;
        let server = g.add_child(root, server).unwrap();
        let handler = g.add_child(server, Node::new("handler", "")).unwrap();
        let _route = g.add_child(handler, Node::new("route", "")).unwrap();
        render_graph(tmp.path(), &g, Layout::Workspace).unwrap();
        assert!(tmp
            .path()
            .join("crates/server/src/handler/route/mod.rs")
            .exists());
        assert!(tmp.path().join("spec/ws/server/handler/route/spec.md").exists());
    }

    #[test]
    fn re_render_updates_changed_content() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        // Now author public.rs and re-render.
        g.get_mut(root).unwrap().public_rs = Some("pub trait T {}\n".to_string());
        let report = render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        // public.rs should be re-written, but other files unchanged.
        let written: Vec<&Path> =
            report.files_written.iter().map(|p| p.as_path()).collect();
        assert!(written.iter().any(|p| p.ends_with("src/public.rs")));
        let pub_ = std::fs::read_to_string(tmp.path().join("src/public.rs")).unwrap();
        assert!(pub_.contains("pub trait T"));
    }

    #[test]
    fn root_mod_does_not_list_unrelated_children() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let _aa = g.add_child(a, Node::new("aa", "")).unwrap();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let root_mod = std::fs::read_to_string(tmp.path().join("src/mod.rs")).unwrap();
        // Root mod should declare `pub mod a;` but NOT `pub mod aa;` (that
        // belongs in `src/a/mod.rs`).
        assert!(root_mod.contains("pub mod a;"));
        assert!(!root_mod.contains("pub mod aa;"));
        let a_mod = std::fs::read_to_string(tmp.path().join("src/a/mod.rs")).unwrap();
        assert!(a_mod.contains("pub mod aa;"));
    }
}
