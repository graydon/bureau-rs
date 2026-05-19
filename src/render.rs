//! Render a `NodeGraph` onto the filesystem.
//!
//! For each node in the graph this produces:
//!
//! - `<workdir>/src/<path>/public.rs`   — public surface (model-authored)
//! - `<workdir>/src/<path>/private.rs`  — hidden internals (model-authored)
//! - `<workdir>/src/<path>/tests.rs`    — tests (model-authored)
//! - `<workdir>/src/<path>/mod.rs`      — framework-rendered glue
//! - `<workdir>/spec/<path>/public.md`    — markdown spec (model-authored)
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
/// if their content would change. Also persists the graph as JSON under
/// `.bureau/` so the worktree's branch carries the graph state as files —
/// see `graph::save`.
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

    // 3. Persist the graph as JSON under .bureau/ — that's how the worktree
    //    branch carries the graph state.
    crate::graph::save(workdir, graph)?;

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

/// Files in the workdir that THIS stage on THIS node legitimately owns
/// — i.e. the framework expects only this task to write to these paths
/// during its lifetime. Used by the worktree merge to apply only this
/// task's diff to main, avoiding the "full-tree render in worktree
/// conflicts with concurrent tasks" problem.
///
/// Architect runs on root only and writes nothing per-node; its
/// "owned files" are the entire workspace structure — handled separately
/// (the architect just commits everything via `commit_main` directly).
///
/// Spec stage: own node's `spec/<path>/public.md` and `private.md`. Plus
/// the containing crate's `Cargo.toml` (deps may have been added).
///
/// Iface / impl / debug / opt: own node's source files in `src/...`,
/// plus its containing crate's `Cargo.toml`.
/// Which graph slot on a node corresponds to a given file path.
///
/// The quickfix tools (`write_file`, `apply_patch`, etc.) operate on file
/// paths, but for files the framework manages (per-node `public.rs`,
/// `private.rs`, `tests.rs`, `spec/<node>/public.md`, `private.md`) the
/// real source of truth is the graph. Editing the file on disk directly
/// would be reverted by the next `render_after_write`. So instead the
/// tools map path → slot, update the slot, and re-render.
///
/// Returns `None` for unmanaged paths (the model shouldn't edit those —
/// generated Cargo.toml, mod.rs, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSlot {
    PublicRs,
    PrivateRs,
    TestsRs,
    SpecPublicMd,
    SpecPrivateMd,
}

/// Map a workspace-relative path back to a (node, slot) pair. Walks every
/// node in the graph looking for one whose src_dir / spec_dir is a parent
/// of `rel` and whose file name matches one of the managed filenames.
///
/// Returns `None` if the path isn't a managed slot — auto-rendered files
/// (`mod.rs`, `lib.rs`, `Cargo.toml`) and arbitrary paths both yield None.
pub fn resolve_path_to_slot(
    graph: &NodeGraph,
    rel: &Path,
    layout: Layout,
) -> Option<(NodeId, NodeSlot)> {
    let filename = rel.file_name().and_then(|s| s.to_str())?;
    let parent = rel.parent()?;
    for n in graph.iter() {
        let src = node_src_dir(graph, n, layout);
        if parent == src {
            match filename {
                "public.rs" => return Some((n.id, NodeSlot::PublicRs)),
                "private.rs" => return Some((n.id, NodeSlot::PrivateRs)),
                "tests.rs" => return Some((n.id, NodeSlot::TestsRs)),
                _ => {}
            }
        }
        let spec = node_spec_dir(graph, n);
        if parent == spec {
            match filename {
                "public.md" => return Some((n.id, NodeSlot::SpecPublicMd)),
                "private.md" => return Some((n.id, NodeSlot::SpecPrivateMd)),
                _ => {}
            }
        }
    }
    None
}

pub fn files_owned_by_stage(
    graph: &NodeGraph,
    node: &Node,
    stage: crate::graph::Stage,
    layout: Layout,
) -> Vec<PathBuf> {
    use crate::graph::Stage;
    let mut paths = Vec::new();
    let src_dir = node_src_dir(graph, node, layout);
    let spec_dir = node_spec_dir(graph, node);
    match stage {
        Stage::Architect => {
            // Architect modifies the whole structure; the engine handles
            // its commit directly. Return empty so apply_to_main is a
            // no-op for architect (engine takes a different path).
        }
        Stage::Spec => {
            paths.push(spec_dir.join("public.md"));
            paths.push(spec_dir.join("private.md"));
            // Cargo.toml of the containing crate may have new path deps.
            paths.push(node_crate_root(graph, node, layout).join("Cargo.toml"));
            // Top-level workspace Cargo.toml (single-crate mode at root).
            paths.push(PathBuf::from("Cargo.toml"));
        }
        Stage::Iface => {
            paths.push(src_dir.join("public.rs"));
            paths.push(src_dir.join("private.rs"));
            paths.push(node_crate_root(graph, node, layout).join("Cargo.toml"));
            paths.push(PathBuf::from("Cargo.toml"));
        }
        Stage::Tests => {
            paths.push(src_dir.join("tests.rs"));
        }
        Stage::Impl | Stage::Debug => {
            paths.push(src_dir.join("private.rs"));
            paths.push(src_dir.join("tests.rs")); // debug stage may rewrite tests
        }
    }
    paths
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
    // Placeholder strings are STATIC (no node-specific interpolation) so
    // `graph::load` can detect them via exact-match and treat the slot as
    // unauthored — see `placeholders.rs`.
    write_if_changed(
        &public_path,
        node.public_rs
            .as_deref()
            .unwrap_or(crate::placeholders::PUBLIC_RS),
        report,
    )?;
    write_if_changed(
        &private_path,
        node.private_rs
            .as_deref()
            .unwrap_or(crate::placeholders::PRIVATE_RS),
        report,
    )?;
    write_if_changed(
        &tests_path,
        node.tests_rs
            .as_deref()
            .unwrap_or(crate::placeholders::TESTS_RS),
        report,
    )?;

    // mod.rs: framework-rendered. In workspace mode we pass the layout
    // so children that are SEPARATE crates aren't declared as `pub mod`
    // (they live in `crates/<name>/`, not as a submodule of this node's
    // crate; declaring them as `pub mod` would make rustc look for
    // `<name>/mod.rs` inside this crate's src/ tree and fail).
    let mod_content = render_mod_rs(graph, node, layout);
    write_if_changed(&mod_path, &mod_content, report)?;

    // Spec markdown lives under <workdir>/spec/<name-path>/, split into
    // public.md (audience: dependents and downstream stages) and
    // private.md (audience: this node's own writer/reviser, for design
    // notes and rationale).
    //
    // public.md uses a static placeholder when un-authored (so load can
    // detect it). The node's name + description live in the surrounding
    // prompt context, not duplicated into a placeholder body.
    let spec_dir = workdir.join(node_spec_dir(graph, node));
    let public_md_path = spec_dir.join("public.md");
    write_if_changed(
        &public_md_path,
        node.spec_public_md
            .as_deref()
            .unwrap_or(crate::placeholders::PUBLIC_MD),
        report,
    )?;
    // private.md is written only when authored — its absence is the
    // signal that no private notes exist for this node.
    if let Some(priv_md) = &node.spec_private_md {
        let private_md_path = spec_dir.join("private.md");
        write_if_changed(&private_md_path, priv_md, report)?;
    }

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
/// Plus `pub mod <child>;` for each child node THAT LIVES IN THE SAME
/// CRATE. In workspace mode, crate-boundary children are separate Cargo
/// packages — declaring them as submodules here would make rustc look
/// for source files inside this crate's tree and error out (`E0583`:
/// file not found for module). Cross-crate references go through
/// `<crate_name>::...` instead, not `crate::<name>`.
fn render_mod_rs(graph: &NodeGraph, node: &Node, layout: Layout) -> String {
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED by bureau-rs. Do not edit by hand.\n");
    s.push_str("// This file declares the node's submodules and re-exports its public surface.\n\n");
    s.push_str("mod public;\n");
    s.push_str("mod private;\n");
    s.push_str("#[cfg(test)]\nmod tests;\n");
    s.push_str("pub use public::*;\n");
    let children = graph.children_of(node.id);
    // Only declare children that are part of THIS node's crate. Children
    // that are crate-boundaries become separate Cargo packages and are
    // wired via path deps in the parent crate's Cargo.toml (in workspace
    // mode); they are NOT submodules of this node's crate.
    let same_crate_children: Vec<&Node> = children
        .into_iter()
        .filter(|c| match layout {
            Layout::SingleCrate => true,
            Layout::Workspace => !c.crate_boundary,
        })
        .collect();
    if !same_crate_children.is_empty() {
        s.push('\n');
        s.push_str("// Children (sub-decompositions in the same crate).\n");
        for child in same_crate_children {
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
    // Member crates also inherit all workspace-level external crates so
    // any node in the member can `use foo::...` without having to know
    // which Cargo.toml holds the dep.
    let external = graph
        .root
        .and_then(|rid| graph.get(rid))
        .map(|r| r.external_crate_deps.as_slice())
        .unwrap_or(&[]);
    if targets.is_empty() && external.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n[dependencies]\n");
    for tid in targets {
        let Some(t) = graph.get(tid) else { continue };
        // For now, every member crate lives at `crates/<name>` and the root
        // crate lives at the workspace root. Path-deps are relative to the
        // dependent's own Cargo.toml.
        let path_str = if t.parent.is_none() {
            "../..".to_string()
        } else {
            format!("../{}", t.name)
        };
        out.push_str(&format!("{} = {{ path = \"{}\" }}\n", t.name, path_str));
    }
    for d in external {
        out.push_str(&format!("{} = {{ workspace = true }}\n", d.name));
    }
    out
}

/// Render external crates.io deps as `name = "version"` or the
/// expanded table form when features are set. Used both in the
/// workspace root's `[workspace.dependencies]` (workspace layout) and
/// the single crate's `[dependencies]` (single-crate layout).
fn render_external_dep_lines(deps: &[crate::graph::ExternalCrateDep]) -> String {
    let mut out = String::new();
    for d in deps {
        let version = d.version.as_deref().unwrap_or("*");
        if d.features.is_empty() {
            out.push_str(&format!("{} = \"{}\"\n", d.name, version));
        } else {
            let feats = d
                .features
                .iter()
                .map(|f| format!("\"{}\"", f))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "{} = {{ version = \"{}\", features = [{}] }}\n",
                d.name, version, feats
            ));
        }
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
    let ext_deps = render_external_dep_lines(&root.external_crate_deps);
    let content = match layout {
        Layout::SingleCrate => {
            let mut s = format!(
                "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/mod.rs\"\n",
                root.name
            );
            if !ext_deps.is_empty() {
                s.push_str("\n[dependencies]\n");
                s.push_str(&ext_deps);
            }
            s
        }
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
            // [workspace.dependencies] holds external crates.io deps
            // declared by the architect. Member crates pull each in via
            // `name.workspace = true` (see `render_cross_crate_deps`).
            let workspace_deps_section = if ext_deps.is_empty() {
                String::new()
            } else {
                format!("\n[workspace.dependencies]\n{}", ext_deps)
            };
            // Root crate may need to depend on member crates if its
            // descendants declare deps that cross into them. Plus
            // workspace-inherited external deps for the root's own use.
            let root_deps_section = {
                let targets = cross_crate_dep_targets(graph, root_id);
                let inherited = root
                    .external_crate_deps
                    .iter()
                    .map(|d| format!("{} = {{ workspace = true }}\n", d.name))
                    .collect::<String>();
                if targets.is_empty() && inherited.is_empty() {
                    String::new()
                } else {
                    let mut s = String::from("\n[dependencies]\n");
                    for tid in targets {
                        if let Some(t) = graph.get(tid) {
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
                    s.push_str(&inherited);
                    s
                }
            };
            format!(
                "[workspace]\nresolver = \"2\"\nmembers = [\n{}\n]\n{workspace_deps_section}\n\n[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/mod.rs\"\n{root_deps_section}",
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
        assert!(tmp.path().join("spec/app/public.md").exists());
        assert!(tmp.path().join("spec/app/a/public.md").exists());
        assert!(tmp.path().join("spec/app/b/public.md").exists());
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
        root.spec_public_md = Some("# App spec\n\nThe app does things.\n".to_string());
        let _root_id = g.insert_root(root).unwrap();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let pub_ = std::fs::read_to_string(tmp.path().join("src/public.rs")).unwrap();
        assert!(pub_.contains("pub trait App"));
        let spec = std::fs::read_to_string(tmp.path().join("spec/app/public.md")).unwrap();
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
        assert!(tmp.path().join("spec/app/a/b/c/public.md").exists());
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
        assert!(tmp.path().join("spec/ws/server/handler/route/public.md").exists());
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
        assert!(root_mod.contains("pub mod a;"));
        assert!(!root_mod.contains("pub mod aa;"));
        let a_mod = std::fs::read_to_string(tmp.path().join("src/a/mod.rs")).unwrap();
        assert!(a_mod.contains("pub mod aa;"));
    }

    #[test]
    fn external_crate_deps_render_into_workspace_cargo_toml() {
        // The architect declares `hmac` and `digest` as crates.io
        // deps. They must end up in `[workspace.dependencies]` at the
        // workspace root AND every member crate's Cargo.toml must
        // inherit them via `name.workspace = true`. Before the fix
        // these deps were stored as a markdown note in
        // `spec_private_md` and never rendered into any Cargo.toml —
        // models referencing the crates produced unbuildable code.
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.crate_boundary = true;
        root.external_crate_deps = vec![
            crate::graph::ExternalCrateDep {
                name: "hmac".into(),
                version: Some("0.12".into()),
                features: vec![],
                reason: "shared HMAC".into(),
            },
            crate::graph::ExternalCrateDep {
                name: "digest".into(),
                version: None, // → "*"
                features: vec!["alloc".into()],
                reason: "trait abstraction".into(),
            },
        ];
        let root_id = g.insert_root(root).unwrap();
        let mut member = Node::new("crypto", "");
        member.crate_boundary = true;
        let _crypto_id = g.add_child(root_id, member).unwrap();
        render_graph(tmp.path(), &g, Layout::Workspace).unwrap();
        let root_cargo = std::fs::read_to_string(tmp.path().join("Cargo.toml")).unwrap();
        assert!(
            root_cargo.contains("[workspace.dependencies]"),
            "missing section:\n{root_cargo}"
        );
        assert!(root_cargo.contains("hmac = \"0.12\""));
        assert!(
            root_cargo.contains("digest = { version = \"*\", features = [\"alloc\"] }"),
            "digest with features missing:\n{root_cargo}"
        );
        // Member crate inherits both via workspace = true.
        let member_cargo =
            std::fs::read_to_string(tmp.path().join("crates/crypto/Cargo.toml")).unwrap();
        assert!(
            member_cargo.contains("hmac = { workspace = true }"),
            "member missing hmac.workspace = true:\n{member_cargo}"
        );
        assert!(
            member_cargo.contains("digest = { workspace = true }"),
            "member missing digest.workspace = true:\n{member_cargo}"
        );
    }

    #[test]
    fn external_crate_deps_render_into_single_crate_cargo_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.crate_boundary = true;
        root.external_crate_deps = vec![crate::graph::ExternalCrateDep {
            name: "serde".into(),
            version: Some("1".into()),
            features: vec!["derive".into()],
            reason: "model serialization".into(),
        }];
        let _root_id = g.insert_root(root).unwrap();
        render_graph(tmp.path(), &g, Layout::SingleCrate).unwrap();
        let cargo = std::fs::read_to_string(tmp.path().join("Cargo.toml")).unwrap();
        assert!(cargo.contains("[dependencies]"), "missing section:\n{cargo}");
        assert!(
            cargo.contains("serde = { version = \"1\", features = [\"derive\"] }"),
            "serde line missing:\n{cargo}"
        );
    }
}
