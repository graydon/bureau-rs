//! Build the prompt context for a node-stage agent invocation.
//!
//! The whole point of this module is to *preempt* the model's need to call
//! `read_file` / `list_files` by stuffing everything it could plausibly need
//! into the prompt up front. For each node-stage we know:
//!
//! - what the node's own already-authored slots look like (e.g. for the
//!   `tests` stage, the model needs to see its own `public.rs`),
//! - what the parent's public surface looks like (the API this node fits
//!   into),
//! - what every declared dep's public surface looks like (the APIs the
//!   model can call),
//! - what the spec context is (own spec + ancestor chain, for design
//!   rationale),
//! - which other nodes already exist in the graph (so the model can
//!   declare deps on them rather than reinventing them).
//!
//! The orchestrator composes these sections into a single markdown context
//! document that goes alongside the role's preamble. Reads are unnecessary;
//! the harness has already given the model what it needs.

use crate::graph::{Node, NodeGraph, NodeId, Stage};

/// One labeled chunk of the context document. Rendered as `# {title}\n\n{body}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextSection {
    pub title: String,
    pub body: String,
}

/// A composed context bundle for one node-stage invocation. Render as a
/// markdown document via `to_markdown()`.
#[derive(Debug, Clone, Default)]
pub struct ContextBundle {
    pub sections: Vec<ContextSection>,
}

impl ContextBundle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, title: impl Into<String>, body: impl Into<String>) -> &mut Self {
        self.sections.push(ContextSection {
            title: title.into(),
            body: body.into(),
        });
        self
    }

    pub fn extend_from(&mut self, other: ContextBundle) {
        self.sections.extend(other.sections);
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        for sec in &self.sections {
            s.push_str(&format!("# {}\n\n", sec.title));
            let body = sec.body.trim_end_matches('\n');
            s.push_str(body);
            s.push_str("\n\n");
        }
        s
    }

    pub fn approx_size(&self) -> usize {
        self.sections.iter().map(|s| s.title.len() + s.body.len() + 8).sum()
    }
}

/// The node's own module path within its containing crate (`crate::a::b::c`).
/// The root yields `crate`. Used as a self-reference label in prompts.
pub fn module_path(graph: &NodeGraph, node_id: NodeId) -> String {
    let Some(node) = graph.get(node_id) else {
        return "crate".to_string();
    };
    let crate_id = crate::render::containing_crate(graph, node);
    let crate_path = graph.name_path(crate_id).unwrap_or_default();
    let to_path = graph.name_path(node_id).unwrap_or_default();
    let inner: Vec<&&str> = to_path[crate_path.len()..].iter().collect();
    if inner.is_empty() {
        "crate".to_string()
    } else {
        let parts: Vec<String> = inner.iter().map(|s| s.to_string()).collect();
        format!("crate::{}", parts.join("::"))
    }
}

/// Import path for `to` as seen from `from`. Within the same crate, this is
/// `crate::<inner>`; across crate boundaries (workspace mode), it's
/// `<other_crate>::<inner-within-that-crate>`. This is what the model
/// should write in a `use` statement.
pub fn import_path(graph: &NodeGraph, from: NodeId, to: NodeId) -> String {
    let (Some(fnode), Some(tnode)) = (graph.get(from), graph.get(to)) else {
        return "crate".to_string();
    };
    let from_crate = crate::render::containing_crate(graph, fnode);
    let to_crate = crate::render::containing_crate(graph, tnode);
    let to_path = graph.name_path(to).unwrap_or_default();
    let to_crate_path = graph.name_path(to_crate).unwrap_or_default();
    let inner: Vec<String> = to_path[to_crate_path.len()..]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let root = if from_crate == to_crate {
        "crate".to_string()
    } else {
        graph.get(to_crate).map(|n| n.name.clone()).unwrap_or_default()
    };
    if inner.is_empty() {
        root
    } else {
        format!("{}::{}", root, inner.join("::"))
    }
}

/// Build the context for the **spec** stage: ancestor specs (so the design
/// thread of "why we're decomposing this way" is visible), siblings'
/// specs (for consistency with parallel decompositions), the current graph
/// overview (for decompose reuse), and the node's existing spec if any.
pub fn build_for_spec(graph: &NodeGraph, node_id: NodeId) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    bundle.push(
        "Node",
        format!(
            "**name**: `{}`\n**module path**: `{}`\n**description**: {}\n",
            node.name,
            module_path(graph, node_id),
            node.description
        ),
    );
    push_ancestor_specs(&mut bundle, graph, node_id);
    push_sibling_specs(&mut bundle, graph, node_id);
    if let Some(own) = &node.spec_md {
        bundle.push("Existing spec for this node", own.clone());
    }
    bundle.push("Existing graph", render_graph_overview(graph));
    bundle
}

/// Build the context for the **iface** stage: own spec, parent's iface
/// (the API this node fits into), each dep's iface, and ancestor specs.
pub fn build_for_iface(graph: &NodeGraph, node_id: NodeId) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    bundle.push("Node", node_header(graph, node));
    if let Some(spec) = &node.spec_md {
        bundle.push("Spec for this node", spec.clone());
    }
    push_ancestor_specs(&mut bundle, graph, node_id);
    push_parent_iface(&mut bundle, graph, node_id);
    push_dep_ifaces(&mut bundle, graph, node);
    bundle.push("Module path", module_path(graph, node_id));
    bundle
}

/// Build the context for the **tests** stage: spec, own iface (the surface
/// to test), deps' ifaces (test setup may need them), and ancestor context.
pub fn build_for_tests(graph: &NodeGraph, node_id: NodeId) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    bundle.push("Node", node_header(graph, node));
    if let Some(spec) = &node.spec_md {
        bundle.push("Spec for this node", spec.clone());
    }
    if let Some(public_rs) = &node.public_rs {
        bundle.push(
            "Public interface to test (this node's `public.rs`)",
            wrap_rust(public_rs),
        );
    }
    push_dep_ifaces(&mut bundle, graph, node);
    push_ancestor_specs_brief(&mut bundle, graph, node_id);
    bundle
}

/// Build the context for the **impl** stage: spec, own iface, own tests
/// (the goal — make these pass), deps' ifaces (the tools available).
pub fn build_for_impl(graph: &NodeGraph, node_id: NodeId) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    bundle.push("Node", node_header(graph, node));
    if let Some(spec) = &node.spec_md {
        bundle.push("Spec for this node", spec.clone());
    }
    if let Some(public_rs) = &node.public_rs {
        bundle.push(
            "Public interface (this node's `public.rs`)",
            wrap_rust(public_rs),
        );
    }
    if let Some(tests_rs) = &node.tests_rs {
        bundle.push(
            "Tests to make pass (this node's `tests.rs`)",
            wrap_rust(tests_rs),
        );
    }
    if let Some(private_rs) = &node.private_rs {
        bundle.push(
            "Existing private content (`private.rs` so far)",
            wrap_rust(private_rs),
        );
    }
    push_dep_ifaces(&mut bundle, graph, node);
    bundle
}

/// Build the context for the **debug** stage: identical to `impl` plus the
/// caller is responsible for appending the failing-test list / cargo errors.
pub fn build_for_debug(graph: &NodeGraph, node_id: NodeId) -> ContextBundle {
    let mut bundle = build_for_impl(graph, node_id);
    bundle.sections.insert(
        0,
        ContextSection {
            title: "Debug stage".to_string(),
            body: "Tests didn't pass after `impl`. Look at the failing-test \
                   list (appended below) and apply minimal fixes to make them \
                   pass without changing the public interface."
                .to_string(),
        },
    );
    bundle
}

/// Convenience entry point: pick the right builder by stage.
pub fn build_for_stage(graph: &NodeGraph, node_id: NodeId, stage: Stage) -> ContextBundle {
    match stage {
        Stage::Spec => build_for_spec(graph, node_id),
        Stage::Iface => build_for_iface(graph, node_id),
        Stage::Tests => build_for_tests(graph, node_id),
        Stage::Impl => build_for_impl(graph, node_id),
        Stage::Debug => build_for_debug(graph, node_id),
        Stage::Opt => build_for_impl(graph, node_id), // same surface as impl
    }
}

// ---- helpers ----

fn node_header(graph: &NodeGraph, node: &Node) -> String {
    format!(
        "**name**: `{}`\n**module path**: `{}`\n**description**: {}\n",
        node.name,
        module_path(graph, node.id),
        node.description
    )
}

/// Insert ancestor specs from immediate parent up to root. Each ancestor
/// gets its own section with the spec body.
fn push_ancestor_specs(bundle: &mut ContextBundle, graph: &NodeGraph, node_id: NodeId) {
    let ancestors = graph.ancestors(node_id, false);
    if ancestors.is_empty() {
        return;
    }
    // Render parent → root order so the closest context is first.
    for anc_id in ancestors {
        let Some(anc) = graph.get(anc_id) else {
            continue;
        };
        let Some(spec) = &anc.spec_md else {
            continue;
        };
        bundle.push(
            format!("Ancestor spec: `{}`", anc.name),
            spec.clone(),
        );
    }
}

/// Brief variant: only the headings of ancestor specs (saves tokens when
/// the model doesn't need full design rationale).
fn push_ancestor_specs_brief(bundle: &mut ContextBundle, graph: &NodeGraph, node_id: NodeId) {
    let ancestors = graph.ancestors(node_id, false);
    if ancestors.is_empty() {
        return;
    }
    let mut s = String::new();
    for anc_id in ancestors {
        let Some(anc) = graph.get(anc_id) else {
            continue;
        };
        let summary = match &anc.spec_md {
            Some(spec) => first_paragraph(spec),
            None => anc.description.clone(),
        };
        s.push_str(&format!("- **`{}`**: {}\n", anc.name, summary));
    }
    bundle.push("Ancestor context (brief)", s);
}

fn push_sibling_specs(bundle: &mut ContextBundle, graph: &NodeGraph, node_id: NodeId) {
    let Some(node) = graph.get(node_id) else {
        return;
    };
    let Some(parent) = node.parent else {
        return;
    };
    let mut s = String::new();
    for sib in graph.children_of(parent) {
        if sib.id == node_id {
            continue;
        }
        let summary = sib
            .spec_md
            .as_ref()
            .map(|md| first_paragraph(md))
            .unwrap_or_else(|| sib.description.clone());
        s.push_str(&format!("- **`{}`**: {}\n", sib.name, summary));
    }
    if !s.is_empty() {
        bundle.push("Sibling specs (already decided)", s);
    }
}

/// If the parent exists and has a `public.rs`, include it. This is the
/// "API this node is part of" — a node implementing one thing in a parent
/// abstraction needs to know what the parent exposes.
fn push_parent_iface(bundle: &mut ContextBundle, graph: &NodeGraph, node_id: NodeId) {
    let Some(node) = graph.get(node_id) else {
        return;
    };
    let Some(parent_id) = node.parent else {
        return;
    };
    let Some(parent) = graph.get(parent_id) else {
        return;
    };
    let Some(public_rs) = &parent.public_rs else {
        return;
    };
    bundle.push(
        format!("Parent public interface: `{}` (`public.rs`)", parent.name),
        wrap_rust(public_rs),
    );
}

/// For each declared dep, include the dep's `public.rs` (full) plus a brief
/// description so the model knows what the dep is for.
fn push_dep_ifaces(bundle: &mut ContextBundle, graph: &NodeGraph, node: &Node) {
    if node.deps.is_empty() {
        return;
    }
    for dep_id in &node.deps {
        let Some(dep) = graph.get(*dep_id) else {
            continue;
        };
        // Use the dep's import path AS SEEN FROM `node` — handles workspace
        // cross-crate cases where the path becomes `<dep_crate>::...` rather
        // than `crate::...`.
        let path = import_path(graph, node.id, dep.id);
        let title = format!(
            "Dependency `{}` (import as `{}`)",
            dep.name, path
        );
        let mut body = String::new();
        body.push_str(&format!("**description**: {}\n\n", dep.description));
        if let Some(spec) = &dep.spec_md {
            body.push_str("**spec excerpt**:\n");
            body.push_str(&first_paragraph(spec));
            body.push_str("\n\n");
        }
        body.push_str("**`public.rs`**:\n");
        if let Some(public_rs) = &dep.public_rs {
            body.push_str(&wrap_rust(public_rs));
        } else {
            body.push_str("*(not yet authored — depend at your own risk)*");
        }
        bundle.push(title, body);
    }
}

/// A short summary of every node currently in the graph. Used by the spec
/// stage's `decompose` tool so the model can choose to dep on existing
/// nodes rather than reinventing them.
pub fn render_graph_overview(graph: &NodeGraph) -> String {
    if graph.is_empty() {
        return "*(graph is empty)*".to_string();
    }
    let mut s = String::new();
    s.push_str("Existing nodes (newest last). Reference these by name in your `add_self_deps` ");
    s.push_str("or in a child's `deps` list rather than creating duplicates:\n\n");
    for n in graph.iter() {
        let path = module_path(graph, n.id);
        let crate_marker = if n.crate_boundary { " *(crate)*" } else { "" };
        s.push_str(&format!(
            "- **`{}`**{} — `{}` — {}\n",
            n.name,
            crate_marker,
            path,
            n.description
        ));
    }
    s
}

fn first_paragraph(md: &str) -> String {
    let mut out = String::new();
    for line in md.lines() {
        if line.trim().is_empty() && !out.is_empty() {
            break;
        }
        if line.starts_with('#') && out.is_empty() {
            // Skip leading heading; we want the prose paragraph after it.
            continue;
        }
        if !line.starts_with('#') {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(line.trim());
        }
    }
    if out.is_empty() {
        // Fallback: first non-empty line as-is (if it was a heading).
        for line in md.lines() {
            if !line.trim().is_empty() {
                return line.trim().trim_start_matches('#').trim().to_string();
            }
        }
    }
    out
}

fn wrap_rust(src: &str) -> String {
    format!("```rust\n{}\n```", src.trim_end_matches('\n'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;

    fn fresh() -> (NodeGraph, NodeId) {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "the application root");
        root.spec_md = Some("# App\n\nThe top-level application.\n".to_string());
        let id = g.insert_root(root).unwrap();
        (g, id)
    }

    #[test]
    fn spec_context_includes_node_header_and_existing_graph() {
        let (g, root) = fresh();
        let bundle = build_for_spec(&g, root);
        let md = bundle.to_markdown();
        assert!(md.contains("**name**: `app`"));
        assert!(md.contains("**module path**: `crate`"));
        assert!(md.contains("Existing graph"));
    }

    #[test]
    fn iface_context_inlines_parent_public_rs() {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.public_rs = Some("pub trait Routing { fn route(&self); }\n".into());
        let root_id = g.insert_root(root).unwrap();
        let child = g
            .add_child(root_id, Node::new("router", "routes requests"))
            .unwrap();
        let bundle = build_for_iface(&g, child);
        let md = bundle.to_markdown();
        assert!(md.contains("Parent public interface: `app`"));
        assert!(md.contains("pub trait Routing"));
    }

    #[test]
    fn iface_context_includes_dep_public_rs() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let mut errs = Node::new("errors", "shared error types");
        errs.public_rs = Some("pub enum Err { NotFound, Bad }\n".into());
        errs.spec_md = Some("# Errors\n\nShared error enum used everywhere.\n".into());
        let errs_id = g.add_child(root, errs).unwrap();
        let widget_id = g.add_child(root, Node::new("widget", "")).unwrap();
        g.add_dep(widget_id, errs_id).unwrap();
        let bundle = build_for_iface(&g, widget_id);
        let md = bundle.to_markdown();
        assert!(md.contains("Dependency `errors`"));
        assert!(md.contains("import as `crate::errors`"));
        assert!(md.contains("pub enum Err"));
        assert!(md.contains("Shared error enum"));
    }

    #[test]
    fn iface_context_omits_section_when_no_deps() {
        let (g, root) = fresh();
        let bundle = build_for_iface(&g, root);
        let md = bundle.to_markdown();
        assert!(!md.contains("Dependency `"));
    }

    #[test]
    fn tests_context_includes_own_public_rs() {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.public_rs = Some("pub trait App { fn run(&self); }\n".into());
        let id = g.insert_root(root).unwrap();
        let bundle = build_for_tests(&g, id);
        let md = bundle.to_markdown();
        assert!(md.contains("Public interface to test"));
        assert!(md.contains("pub trait App"));
    }

    #[test]
    fn impl_context_includes_public_tests_and_existing_private() {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.public_rs = Some("pub trait App { fn run(&self); }\n".into());
        root.tests_rs = Some("#[test] fn ok() { /* asserts */ }\n".into());
        root.private_rs = Some("// scaffolding\n".into());
        let id = g.insert_root(root).unwrap();
        let bundle = build_for_impl(&g, id);
        let md = bundle.to_markdown();
        assert!(md.contains("Public interface"));
        assert!(md.contains("Tests to make pass"));
        assert!(md.contains("Existing private content"));
    }

    #[test]
    fn debug_context_prepends_debug_section() {
        let (g, root) = fresh();
        let bundle = build_for_debug(&g, root);
        assert_eq!(bundle.sections[0].title, "Debug stage");
    }

    #[test]
    fn ancestor_specs_render_in_order() {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.spec_md = Some("# App\n\nThe whole thing.\n".into());
        let root_id = g.insert_root(root).unwrap();
        let mut frontend = Node::new("frontend", "");
        frontend.spec_md = Some("# Frontend\n\nThe frontend layer.\n".into());
        let f = g.add_child(root_id, frontend).unwrap();
        let r = g.add_child(f, Node::new("router", "")).unwrap();
        let bundle = build_for_iface(&g, r);
        let md = bundle.to_markdown();
        let frontend_pos = md.find("Ancestor spec: `frontend`").unwrap();
        let app_pos = md.find("Ancestor spec: `app`").unwrap();
        // Closest first: frontend should appear before app.
        assert!(frontend_pos < app_pos);
    }

    #[test]
    fn sibling_specs_listed_for_consistency() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let mut a = Node::new("a", "node A");
        a.spec_md = Some("# A\n\nThe A subsystem.\n".into());
        g.add_child(root, a).unwrap();
        let b = g.add_child(root, Node::new("b", "node B")).unwrap();
        let bundle = build_for_spec(&g, b);
        let md = bundle.to_markdown();
        assert!(md.contains("Sibling specs"));
        assert!(md.contains("**`a`**"));
        assert!(md.contains("The A subsystem"));
    }

    #[test]
    fn graph_overview_lists_all_nodes_with_module_paths() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "umbrella")).unwrap();
        let _a = g.add_child(root, Node::new("a", "thing A")).unwrap();
        let mut b_node = Node::new("b", "thing B (a separate crate)");
        b_node.crate_boundary = true;
        let _b = g.add_child(root, b_node).unwrap();
        let overview = render_graph_overview(&g);
        assert!(overview.contains("**`app`**"));
        assert!(overview.contains("**`a`**"));
        assert!(overview.contains("**`b`**"));
        assert!(overview.contains("(crate)"));
        assert!(overview.contains("crate::a"));
        // `b` is its own crate, so its `module_path` (own-crate-relative) is
        // just `crate`. The overview shows it that way; an outside dependent
        // would import it as `b::...` (handled by `import_path`).
    }

    #[test]
    fn graph_overview_handles_empty_graph() {
        let g = NodeGraph::new();
        let overview = render_graph_overview(&g);
        assert!(overview.contains("graph is empty"));
    }

    #[test]
    fn first_paragraph_strips_leading_heading() {
        assert_eq!(
            first_paragraph("# Title\n\nThe body of the spec.\n\nMore."),
            "The body of the spec."
        );
    }

    #[test]
    fn first_paragraph_handles_heading_only() {
        // No prose body; fall back to the heading text itself.
        assert_eq!(first_paragraph("# Just a Title\n"), "Just a Title");
    }

    #[test]
    fn first_paragraph_multi_line_paragraph_joined() {
        assert_eq!(
            first_paragraph("Lorem ipsum\ndolor sit amet."),
            "Lorem ipsum dolor sit amet."
        );
    }

    #[test]
    fn module_path_for_root_is_crate() {
        let (g, root) = fresh();
        assert_eq!(module_path(&g, root), "crate");
    }

    #[test]
    fn module_path_for_nested_child() {
        let mut g = NodeGraph::new();
        let r = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(r, Node::new("a", "")).unwrap();
        let b = g.add_child(a, Node::new("b", "")).unwrap();
        assert_eq!(module_path(&g, b), "crate::a::b");
    }

    #[test]
    fn import_path_within_same_crate() {
        let mut g = NodeGraph::new();
        let r = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(r, Node::new("a", "")).unwrap();
        let b = g.add_child(r, Node::new("b", "")).unwrap();
        // Both in the same (root) crate.
        assert_eq!(import_path(&g, a, b), "crate::b");
        assert_eq!(import_path(&g, b, a), "crate::a");
    }

    #[test]
    fn import_path_across_crate_boundary_uses_crate_name() {
        let mut g = NodeGraph::new();
        let r = g.insert_root(Node::new("app", "")).unwrap();
        let mut errors_node = Node::new("errors", "");
        errors_node.crate_boundary = true;
        let errors = g.add_child(r, errors_node).unwrap();
        let mut server_node = Node::new("server", "");
        server_node.crate_boundary = true;
        let server = g.add_child(r, server_node).unwrap();
        // server depending on errors crosses the crate boundary.
        // From server's perspective: `errors` is reached as `errors::*`,
        // not `crate::errors::*`.
        assert_eq!(import_path(&g, server, errors), "errors");
        // From errors itself: `crate` is the errors crate.
        assert_eq!(module_path(&g, errors), "crate");
    }

    #[test]
    fn import_path_for_nested_node_in_other_crate() {
        let mut g = NodeGraph::new();
        let r = g.insert_root(Node::new("app", "")).unwrap();
        let mut backend_node = Node::new("backend", "");
        backend_node.crate_boundary = true;
        let backend = g.add_child(r, backend_node).unwrap();
        let inner = g.add_child(backend, Node::new("inner", "")).unwrap();
        // From the umbrella root, inner's import path is backend::inner.
        assert_eq!(import_path(&g, r, inner), "backend::inner");
        // From inner's own perspective (within backend crate), it's
        // crate::inner.
        assert_eq!(module_path(&g, inner), "crate::inner");
    }

    #[test]
    fn build_for_stage_dispatches_correctly() {
        let (g, root) = fresh();
        for stage in Stage::ALL {
            let bundle = build_for_stage(&g, root, stage);
            assert!(!bundle.sections.is_empty(), "stage {stage} produced empty bundle");
        }
    }

    #[test]
    fn missing_node_returns_empty_bundle() {
        let g = NodeGraph::new();
        let bundle = build_for_iface(&g, NodeId::new());
        assert!(bundle.sections.is_empty());
    }

    #[test]
    fn dep_iface_warns_when_dep_has_no_public_rs() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let half_baked = g.add_child(root, Node::new("dep", "WIP")).unwrap();
        let user = g.add_child(root, Node::new("user", "")).unwrap();
        g.add_dep(user, half_baked).unwrap();
        let bundle = build_for_iface(&g, user);
        let md = bundle.to_markdown();
        assert!(md.contains("not yet authored"));
    }

    #[test]
    fn approx_size_grows_with_dep_iface() {
        // build_for_iface doesn't include the node's own public.rs (the model
        // is writing it); growth comes from added context like dep ifaces.
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let user = g.add_child(root, Node::new("user", "")).unwrap();
        let small = build_for_iface(&g, user).approx_size();
        // Add a dep with a sizeable public.rs.
        let mut dep = Node::new("dep", "shared utility");
        dep.public_rs = Some("pub trait Big { ".to_string() + &"fn f(&self); ".repeat(50) + " }");
        let dep_id = g.add_child(root, dep).unwrap();
        g.add_dep(user, dep_id).unwrap();
        let big = build_for_iface(&g, user).approx_size();
        assert!(big > small);
    }

    #[test]
    fn iface_context_for_root_omits_parent_section() {
        let (g, root) = fresh();
        let bundle = build_for_iface(&g, root);
        let md = bundle.to_markdown();
        assert!(!md.contains("Parent public interface"));
    }

    #[test]
    fn impl_context_inlines_dep_public_rs_for_implementing() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let mut util = Node::new("util", "shared utilities");
        util.public_rs = Some("pub fn id<T>(x: T) -> T { x }\n".into());
        let util_id = g.add_child(root, util).unwrap();
        let mut consumer = Node::new("consumer", "uses util");
        consumer.public_rs = Some("pub trait Consumer { fn run(&self); }\n".into());
        consumer.tests_rs = Some("// tests\n".into());
        let consumer_id = g.add_child(root, consumer).unwrap();
        g.add_dep(consumer_id, util_id).unwrap();

        let bundle = build_for_impl(&g, consumer_id);
        let md = bundle.to_markdown();
        assert!(md.contains("Dependency `util`"));
        assert!(md.contains("pub fn id"));
    }

    #[test]
    fn render_to_markdown_separates_sections_with_blank_line() {
        let mut b = ContextBundle::new();
        b.push("First", "alpha");
        b.push("Second", "beta");
        let md = b.to_markdown();
        assert!(md.contains("# First\n\nalpha\n\n# Second\n\nbeta\n\n"));
    }
}
