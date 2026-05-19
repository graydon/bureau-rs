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
use crate::render::Layout;

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

// ============================================================================
// Builders are laid out for prompt-cache prefix reuse.
//
// Provider prompt caches match a common PREFIX of the request: every byte
// shared with an earlier request is free; the first differing byte breaks
// the cache. So the bundle order matters — stable content first, variable
// content last. The "scope of stability" hierarchy:
//
//   tier 1: stable across the entire project (every node, every stage)
//           — Project mission, Style guide — these live in the SYSTEM
//           prompt now (see `engine.rs` system_prompt assembly), not
//           in the user-prompt context bundle, so they're part of the
//           truly stable prefix every provider caches.
//   tier 2: stable across all descendants of an ancestor
//           — ancestor chain, root-down
//   tier 3: stable across this node's siblings (same parent)
//           — parent's full public spec, parent's public.rs (iface)
//           — sibling list (lex-ordered, INCLUDING self so every sibling
//             sees the same content here)
//   tier 4: stable across this node's stages and roles within a stage
//           — graph overview (spec), this node's header
//   tier 5: per-stage or per-node-content
//           — dep ifaces, own spec, own already-authored slots
//   tier 6: per-turn / most volatile
//           — critique cycle context (engine appends last)
//
// Earlier code put node-specific stuff (the "Node" header, the node's own
// spec) FIRST. Two sibling nodes' requests then diverged at byte zero of
// the context_doc — no cache reuse across the tree. The order below puts
// the broadly-shared content up front and pushes the node-specific bits
// to the end.
// ============================================================================

/// Build the context for the **spec** stage. The architect already laid
/// out the whole tree — this stage just writes prose for ONE node. It
/// can't add children or change topology, so no "decomposition budget"
/// section. The model gets ancestor briefs, the parent's spec, siblings,
/// and the existing graph for context.
pub fn build_for_spec(
    graph: &NodeGraph,
    node_id: NodeId,
    layout: Layout,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    let depth = graph.ancestors(node_id, true).len().saturating_sub(1);

    // Tier 2: ancestor chain (root-down), stable across descendants.
    push_ancestor_chain_brief(&mut bundle, graph, node_id, summaries);
    push_parent_full_spec(&mut bundle, graph, node_id, summaries);

    // Tier 3: sibling specs (lex-ordered, includes self).
    push_siblings_lex(&mut bundle, graph, node_id);

    // Tier 4: graph overview (stable across this node's roles, varies
    // slowly as the graph grows).
    bundle.push("Existing graph", render_graph_overview(graph));

    // Tier 4 (this node identity) → Tier 5 (own content).
    push_this_node_header(&mut bundle, graph, node, depth);
    push_readable_files(&mut bundle, graph, node, layout);
    push_own_spec(&mut bundle, node, "Existing spec for this node");

    bundle
}

/// Build the context for the **iface** stage.
pub fn build_for_iface(
    graph: &NodeGraph,
    node_id: NodeId,
    layout: Layout,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    let depth = graph.ancestors(node_id, true).len().saturating_sub(1);

    push_ancestor_chain_brief(&mut bundle, graph, node_id, summaries);
    push_parent_full_spec(&mut bundle, graph, node_id, summaries);
    push_parent_iface(&mut bundle, graph, node_id);
    push_siblings_lex(&mut bundle, graph, node_id);
    push_dep_ifaces(&mut bundle, graph, node);

    push_this_node_header(&mut bundle, graph, node, depth);
    push_readable_files(&mut bundle, graph, node, layout);
    push_own_spec(&mut bundle, node, "Spec for this node");

    bundle
}

/// Build the context for the **tests** stage.
pub fn build_for_tests(
    graph: &NodeGraph,
    node_id: NodeId,
    layout: Layout,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    let depth = graph.ancestors(node_id, true).len().saturating_sub(1);

    push_ancestor_chain_brief(&mut bundle, graph, node_id, summaries);
    push_siblings_lex(&mut bundle, graph, node_id);
    push_dep_ifaces(&mut bundle, graph, node);

    push_this_node_header(&mut bundle, graph, node, depth);
    push_readable_files(&mut bundle, graph, node, layout);
    push_own_spec(&mut bundle, node, "Spec for this node");
    if let Some(public_rs) = &node.public_rs {
        bundle.push(
            "Public interface to test (this node's `public.rs`)",
            wrap_rust(public_rs),
        );
    }

    bundle
}

/// Build the context for the **impl** stage.
pub fn build_for_impl(
    graph: &NodeGraph,
    node_id: NodeId,
    layout: Layout,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    let depth = graph.ancestors(node_id, true).len().saturating_sub(1);

    push_ancestor_chain_brief(&mut bundle, graph, node_id, summaries);
    push_siblings_lex(&mut bundle, graph, node_id);
    push_dep_ifaces(&mut bundle, graph, node);

    push_this_node_header(&mut bundle, graph, node, depth);
    push_readable_files(&mut bundle, graph, node, layout);
    push_own_spec(&mut bundle, node, "Spec for this node");
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

    bundle
}

/// Push this node's OWN spec — both public and private parts. The writer
/// is the audience for both: the public part is the contract they're
/// honoring, the private part is their own implementation notes from
/// earlier stages.
fn push_own_spec(bundle: &mut ContextBundle, node: &Node, base_title: &str) {
    if let Some(pub_md) = &node.spec_public_md {
        bundle.push(format!("{base_title} (public)"), pub_md.clone());
    }
    if let Some(priv_md) = &node.spec_private_md {
        bundle.push(format!("{base_title} (private notes)"), priv_md.clone());
    }
}

/// Build the context for the **debug** stage: identical to `impl` plus a
/// trailing "Debug stage" note. Note goes AFTER the impl-shaped context
/// so the prefix matches impl's context up to that point (cousins
/// transitioning impl→debug share the cache).
pub fn build_for_debug(
    graph: &NodeGraph,
    node_id: NodeId,
    layout: Layout,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> ContextBundle {
    let mut bundle = build_for_impl(graph, node_id, layout, summaries);
    bundle.push(
        "Debug stage",
        "Tests didn't pass after `impl`. Look at the failing-test \
         list (appended below) and apply minimal fixes to make them \
         pass without changing the public interface.",
    );
    bundle
}

/// Convenience entry point: pick the right builder by stage. Architect
/// needs the depth/node caps for its tree-building budget; later stages
/// don't (the architect already laid out the tree, the rest just author
/// content for existing nodes).
///
/// `summaries`: optional LLM-summary cache. When set, the ancestor-chain
/// section uses the cached compact summary for each ancestor instead of
/// the first-paragraph brief. Falls back to the brief on cache miss.
pub fn build_for_stage(
    graph: &NodeGraph,
    node_id: NodeId,
    stage: Stage,
    max_nodes: usize,
    max_node_depth: usize,
    layout: Layout,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> ContextBundle {
    let _ = max_nodes;
    let _ = max_node_depth;
    match stage {
        // Architect runs on the root, before any per-node work. Its
        // context is just the project mission (already prepended by the
        // engine) plus a depth/cap budget — no ancestor specs etc.
        Stage::Architect => build_for_architect(graph, node_id, max_nodes, max_node_depth),
        Stage::Spec => build_for_spec(graph, node_id, layout, summaries),
        Stage::Iface => build_for_iface(graph, node_id, layout, summaries),
        Stage::Tests => build_for_tests(graph, node_id, layout, summaries),
        Stage::Impl => build_for_impl(graph, node_id, layout, summaries),
        Stage::Debug => build_for_debug(graph, node_id, layout, summaries),
    }
}

/// Build the architect-stage context bundle. Minimal — the architect
/// just needs the budget (so it knows how many nodes/depth it has).
/// The Project mission lives in the system prompt (see engine.rs),
/// not in the user-prompt context bundle.
pub fn build_for_architect(
    graph: &NodeGraph,
    node_id: NodeId,
    max_nodes: usize,
    max_node_depth: usize,
) -> ContextBundle {
    let mut bundle = ContextBundle::new();
    let Some(node) = graph.get(node_id) else {
        return bundle;
    };
    bundle.push(
        "Architect node",
        format!(
            "**root**: `{}`\n**description (seeded from problem.md)**: {}\n",
            node.name, node.description
        ),
    );
    bundle.push(
        "Architecture budget",
        format!(
            "You may submit up to **{}** nodes total (currently {} = just the \
             root), and the tree may go up to **{}** levels deep below the root.\n\
             \n\
             Aim shallower-and-broader. Most nodes are leaves. Only set \
             `crate_boundary: true` for major top-level subsystems — most \
             children are modules within their parent's crate.\n",
            max_nodes,
            graph.len(),
            max_node_depth
        ),
    );
    bundle
}

// ---- helpers ----

/// Header for THIS node: name, description, module path, depth (where
/// applicable). Pushed LATE in the bundle — it's specific to this node
/// so it would otherwise bust prefix caching across siblings.
///
/// `max_node_depth_or_zero` is the spec stage's depth cap (passed only
/// from `build_for_spec`); other stages pass 0 and we omit the depth
/// line.
fn push_this_node_header(
    bundle: &mut ContextBundle,
    graph: &NodeGraph,
    node: &Node,
    depth: usize,
) {
    let mut body = String::new();
    body.push_str(&format!(
        "**name**: `{}`\n**module path**: `{}`\n**depth in tree**: {}\n",
        node.name,
        module_path(graph, node.id),
        depth,
    ));
    body.push_str(&format!("**description**: {}\n", node.description));
    bundle.push("This node", body);
}

/// Push the ancestor chain root-down as a single brief list. Stable
/// across all descendants of the deepest ancestor in the list: a node
/// and its cousin share the chain up to (and including) the lowest
/// common ancestor.
///
/// We list ancestors only (self excluded). Each entry is a one-liner —
/// the parent's FULL public spec is pushed separately by
/// `push_parent_full_spec` so callers that don't want it (tests, impl,
/// debug) can omit it without losing the prefix-friendly brief list.
fn push_ancestor_chain_brief(
    bundle: &mut ContextBundle,
    graph: &NodeGraph,
    node_id: NodeId,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) {
    // `graph.ancestors(node, false)` returns parent first, then grandparent,
    // etc. We want root-down for the brief list.
    let mut ancestors = graph.ancestors(node_id, false);
    if ancestors.is_empty() {
        return;
    }
    ancestors.reverse(); // root first → parent last
    let mut s = String::new();
    for anc_id in ancestors {
        let Some(anc) = graph.get(anc_id) else {
            continue;
        };
        let summary = ancestor_brief(anc, summaries);
        s.push_str(&format!("- **`{}`**: {}\n", anc.name, summary));
    }
    if !s.is_empty() {
        bundle.push("Ancestor chain (root → parent, brief)", s);
    }
}

/// Pick the most informative compact description of `anc` for the
/// ancestor-chain brief block:
///   1. If a `summaries` cache is supplied AND has a cached entry for
///      `anc`'s public spec, use it (model-summarized, 5-10 lines).
///   2. Otherwise, fall back to the framework's first-paragraph brief.
///   3. Otherwise, the node's `description` field.
fn ancestor_brief(
    anc: &Node,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) -> String {
    if let (Some(cache), Some(spec)) = (summaries, &anc.spec_public_md) {
        if let Some(s) = cache.get(spec) {
            return s;
        }
    }
    ancestor_summary(anc)
}

/// Push the IMMEDIATE parent's public spec. Stable across siblings of
/// `node_id`. Called only by stages that need the parent's design
/// narrative (spec, iface) — not by tests/impl/debug, which work against
/// this node's own iface + tests.
///
/// Uses the LLM summary cache when one is provided; falls back to the
/// full spec markdown otherwise. The parent's full spec is often the
/// largest single chunk in the bundle (it dominates token cost for
/// sibling cousins that all share it), so summarizing here is a big win.
fn push_parent_full_spec(
    bundle: &mut ContextBundle,
    graph: &NodeGraph,
    node_id: NodeId,
    summaries: Option<&crate::spec_summary::SpecSummaryCache>,
) {
    let Some(node) = graph.get(node_id) else {
        return;
    };
    let Some(parent_id) = node.parent else {
        return;
    };
    let Some(parent) = graph.get(parent_id) else {
        return;
    };
    if let Some(pub_md) = &parent.spec_public_md {
        let body = summaries
            .and_then(|c| c.get(pub_md))
            .unwrap_or_else(|| pub_md.clone());
        bundle.push(
            format!("Parent spec (public): `{}`", parent.name),
            body,
        );
    }
}

/// One-line summary of a node for the brief-ancestor / sibling list.
/// Prefers the first paragraph of the public spec (skipping leading
/// markdown headings); falls back to the node's `description` field if
/// the spec hasn't been authored yet.
fn ancestor_summary(node: &Node) -> String {
    match &node.spec_public_md {
        Some(spec) => {
            let p = first_paragraph(spec);
            if p.is_empty() { node.description.clone() } else { p }
        }
        None => node.description.clone(),
    }
}

/// Push the parent's children as a lex-sorted list INCLUDING `node_id`
/// itself. Including self is deliberate: every sibling's bundle has the
/// same content here, so the prompt cache extends through this section
/// across all siblings. The model can identify "which sibling am I" by
/// matching the name in the `This node` header section that follows.
fn push_siblings_lex(bundle: &mut ContextBundle, graph: &NodeGraph, node_id: NodeId) {
    let Some(node) = graph.get(node_id) else {
        return;
    };
    let Some(parent) = node.parent else {
        return;
    };
    let mut sibs: Vec<&Node> = graph.children_of(parent);
    if sibs.len() <= 1 {
        // Only this node; nothing to share with siblings — skip the
        // section so a singleton child doesn't carry a wasted "list of
        // one" in its prompt.
        return;
    }
    sibs.sort_by(|a, b| a.name.cmp(&b.name));
    let mut s = String::new();
    for sib in sibs {
        s.push_str(&format!("- **`{}`**: {}\n", sib.name, ancestor_summary(sib)));
    }
    bundle.push("Siblings (this node's parent's children, lex-ordered)", s);
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
    // Parent's public surface is foreign code from this node's POV —
    // summarized signatures are all the model needs. If they want
    // the verbatim source, `read_file` is still available.
    bundle.push(
        format!("Parent public interface: `{}` (`public.rs`)", parent.name),
        wrap_rust_summary(public_rs),
    );
}

/// For each declared dep, include a summary of the dep's `public.rs`
/// plus a brief description so the model knows what the dep is for.
/// Summarized — the model rarely needs trait bodies; `read_file` is
/// available if it does.
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
        if let Some(spec) = &dep.spec_public_md {
            body.push_str("**spec excerpt**:\n");
            body.push_str(&first_paragraph(spec));
            body.push_str("\n\n");
        }
        body.push_str("**`public.rs`**:\n");
        if let Some(public_rs) = &dep.public_rs {
            body.push_str(&wrap_rust_summary(public_rs));
        } else {
            body.push_str("*(not yet authored — depend at your own risk)*");
        }
        bundle.push(title, body);
    }
}

/// List the files the model is permitted to `read_file`. Mirrors the
/// `is_readable_by_node` policy in `tools.rs` — listing them up front
/// cuts down the storm of failed `read_file` calls the model otherwise
/// makes while guessing paths.
///
/// Important: every path we list is a **real file the framework has
/// already written**. We don't list slots that haven't been authored
/// yet (e.g. a node's `private.md` is only written when the spec stage
/// has authored private notes; listing it before then would lead to a
/// "file does not exist" error on read).
///
/// We list:
///   1. this node's own slots — public.rs / private.rs / tests.rs /
///      public.md always exist after the first render; private.md
///      only if private notes have been authored.
///   2. ancestor specs (public.md always; private.md only if
///      authored) — descendants have design-context read access.
///   3. dep nodes' `public.rs` + `public.md`.
///   4. a generic mention of "any node's `public.rs` / `public.md`".
///
/// Framework-rendered files (`Cargo.toml`, `mod.rs`, `lib.rs`) are
/// EXCLUDED — they're auto-generated boilerplate the model never
/// needs to inspect. Previously we allowed them and the model burned
/// calls reading `lib.rs` files our render layout doesn't produce.
fn push_readable_files(
    bundle: &mut ContextBundle,
    graph: &NodeGraph,
    node: &Node,
    layout: Layout,
) {
    let mut s = String::new();
    s.push_str(
        "`read_file` is restricted to the files listed below. Paths are \
         **workspace-relative** (no leading `./`, no absolute paths) — \
         use them verbatim as the tool's `path` argument. Any other path \
         is rejected.\n\n",
    );

    // 1. This node's own slots — only those that the framework has
    //    actually rendered.
    let own_src = crate::render::node_src_dir(graph, node, layout);
    let own_spec = crate::render::node_spec_dir(graph, node);
    s.push_str("**This node's own slots:**\n");
    // public/private/tests.rs and public.md are always rendered (with
    // placeholders if not yet authored), so they always exist on disk.
    for f in ["public.rs", "private.rs", "tests.rs"] {
        s.push_str(&format!("- `{}/{}`\n", own_src.display(), f));
    }
    s.push_str(&format!("- `{}/public.md`\n", own_spec.display()));
    if node.spec_private_md.is_some() {
        s.push_str(&format!("- `{}/private.md`\n", own_spec.display()));
    }
    s.push('\n');

    // 2. Ancestor specs (root-down). Only list private.md when the
    //    ancestor actually authored private notes.
    let mut ancestors = graph.ancestors(node.id, false);
    ancestors.reverse(); // root first → parent last
    if !ancestors.is_empty() {
        s.push_str("**Ancestor spec docs (design context):**\n");
        for anc_id in &ancestors {
            if let Some(anc) = graph.get(*anc_id) {
                let anc_spec = crate::render::node_spec_dir(graph, anc);
                s.push_str(&format!("- `{}/public.md`\n", anc_spec.display()));
                if anc.spec_private_md.is_some() {
                    s.push_str(&format!("- `{}/private.md`\n", anc_spec.display()));
                }
            }
        }
        s.push('\n');
    }

    // 3. Dep public surfaces. public.rs is always rendered.
    if !node.deps.is_empty() {
        s.push_str("**Dep public surfaces (also inlined above; cite path for line-range reads):**\n");
        for dep_id in &node.deps {
            if let Some(dep) = graph.get(*dep_id) {
                let dep_src = crate::render::node_src_dir(graph, dep, layout);
                let dep_spec = crate::render::node_spec_dir(graph, dep);
                s.push_str(&format!(
                    "- `{}/public.rs`, `{}/public.md`\n",
                    dep_src.display(),
                    dep_spec.display()
                ));
            }
        }
        s.push('\n');
    }

    // 4. Generic patterns + explicit denylist.
    s.push_str(
        "**Generic patterns also allowed** (any node in the graph):\n\
         - any node's `public.rs` and `spec/<path>/public.md`\n\
         \n\
         **NOT readable** (don't try):\n\
         - another node's `private.rs`, `tests.rs`, or `spec/<path>/private.md` \
         (your own private slots and ancestor private slots are accessible — \
         see the lists above).\n\
         - `mod.rs`, `lib.rs`, `Cargo.toml` — framework-rendered boilerplate \
         (each crate's library entry point is `mod.rs`, not `lib.rs`). These \
         carry no design info; the contents of all sibling/child modules are \
         already inlined into your context as needed.\n",
    );

    bundle.push("Files you can read", s);
}

/// A short summary of every node currently in the graph. Surfaced in
/// the spec stage's context so the model knows what other nodes exist
/// and can pick the right names for its `deps` declarations.
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

/// Compact signature-only summary of foreign Rust (dep ifaces, parent
/// surfaces) wrapped in a code fence. Falls back to the raw source if
/// summarization yields nothing useful. Use this for ANY `.rs` that's
/// not the current node's own editable slot — those keep full content.
fn wrap_rust_summary(src: &str) -> String {
    let summarized = crate::rust_summary::summarize_rust(src);
    format!(
        "```rust\n// (signature-only summary)\n{}\n```",
        summarized.trim_end_matches('\n')
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;

    fn fresh() -> (NodeGraph, NodeId) {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "the application root");
        root.spec_public_md = Some("# App\n\nThe top-level application.\n".to_string());
        let id = g.insert_root(root).unwrap();
        (g, id)
    }

    #[test]
    fn spec_context_includes_node_header_and_existing_graph() {
        let (g, root) = fresh();
        let bundle = build_for_spec(&g, root, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        assert!(md.contains("**name**: `app`"));
        assert!(md.contains("**module path**: `crate`"));
        assert!(md.contains("Existing graph"));
    }

    #[test]
    fn spec_context_does_not_include_decomposition_budget() {
        // The architect lays out the whole tree; the spec stage just
        // writes prose for one existing node. If "Decomposition budget"
        // appears we've regressed on that separation.
        let (g, root) = fresh();
        let bundle = build_for_spec(&g, root, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        assert!(
            !md.contains("Decomposition budget"),
            "spec bundle must NOT contain a decomposition budget: {md}"
        );
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
        let bundle = build_for_iface(&g, child, Layout::SingleCrate, None);
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
        errs.spec_public_md = Some("# Errors\n\nShared error enum used everywhere.\n".into());
        let errs_id = g.add_child(root, errs).unwrap();
        let widget_id = g.add_child(root, Node::new("widget", "")).unwrap();
        g.add_dep(widget_id, errs_id).unwrap();
        let bundle = build_for_iface(&g, widget_id, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        assert!(md.contains("Dependency `errors`"));
        assert!(md.contains("import as `crate::errors`"));
        assert!(md.contains("pub enum Err"));
        assert!(md.contains("Shared error enum"));
    }

    #[test]
    fn iface_context_omits_section_when_no_deps() {
        let (g, root) = fresh();
        let bundle = build_for_iface(&g, root, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        assert!(!md.contains("Dependency `"));
    }

    #[test]
    fn tests_context_includes_own_public_rs() {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.public_rs = Some("pub trait App { fn run(&self); }\n".into());
        let id = g.insert_root(root).unwrap();
        let bundle = build_for_tests(&g, id, Layout::SingleCrate, None);
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
        let bundle = build_for_impl(&g, id, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        assert!(md.contains("Public interface"));
        assert!(md.contains("Tests to make pass"));
        assert!(md.contains("Existing private content"));
    }

    #[test]
    fn debug_context_includes_debug_section() {
        let (g, root) = fresh();
        let bundle = build_for_debug(&g, root, Layout::SingleCrate, None);
        // After the cache-friendly reorder, the Debug stage note lives at
        // the END of the bundle (so the prefix matches impl context).
        assert!(
            bundle.sections.iter().any(|s| s.title == "Debug stage"),
            "Debug stage section should be present somewhere in the bundle"
        );
    }

    #[test]
    fn ancestor_chain_lists_ancestors_root_down_and_omits_grandparent_details() {
        let mut g = NodeGraph::new();
        let mut root = Node::new("app", "");
        root.spec_public_md = Some(
            "# App\n\n\
             A short summary line.\n\n\
             ## Big section grandchildren should not see\n\n\
             Lots of details that would bloat a grandchild's prompt — \
             paragraphs about the API surface, invariants, lifecycle, etc.\n"
                .into(),
        );
        let root_id = g.insert_root(root).unwrap();
        let mut frontend = Node::new("frontend", "");
        frontend.spec_public_md = Some("# Frontend\n\nThe frontend layer.\n".into());
        let f = g.add_child(root_id, frontend).unwrap();
        let r = g.add_child(f, Node::new("router", "")).unwrap();
        let bundle = build_for_iface(&g, r, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        // Brief ancestor chain (root-down) must list both ancestors.
        assert!(
            md.contains("Ancestor chain"),
            "should have a root-down brief ancestor chain section: {md}"
        );
        assert!(md.contains("**`app`**"));
        assert!(md.contains("**`frontend`**"));
        // Immediate parent's full spec is still inlined (separate section).
        assert!(
            md.contains("Parent spec (public): `frontend`"),
            "parent's full public spec should still be inlined: {md}"
        );
        assert!(md.contains("The frontend layer."));
        // Grandparent's secondary sections must NOT be inlined.
        assert!(
            !md.contains("Big section grandchildren should not see")
                && !md.contains("Lots of details that would bloat"),
            "grandparent's full spec should NOT be inlined: {md}"
        );
        // Root-down ordering: app must appear BEFORE frontend in the chain.
        let app_pos = md.find("**`app`**").unwrap();
        let frontend_pos = md.find("**`frontend`**").unwrap();
        assert!(
            app_pos < frontend_pos,
            "ancestor chain should be root-down (app before frontend): {md}"
        );
    }

    #[test]
    fn siblings_listed_lex_ordered_including_self() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let mut a = Node::new("a", "node A");
        a.spec_public_md = Some("# A\n\nThe A subsystem.\n".into());
        g.add_child(root, a).unwrap();
        let b = g.add_child(root, Node::new("b", "node B")).unwrap();
        let bundle = build_for_spec(&g, b, Layout::SingleCrate, None);
        let md = bundle.to_markdown();
        assert!(md.contains("Siblings"));
        // Self-inclusion: every sibling sees the SAME list, so the prefix
        // cache survives across siblings. `b` should be present in `b`'s
        // own bundle too.
        assert!(md.contains("**`a`**"));
        assert!(md.contains("**`b`**"));
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
            let bundle = build_for_stage(&g, root, stage, 64, 5, Layout::SingleCrate, None);
            assert!(!bundle.sections.is_empty(), "stage {stage} produced empty bundle");
        }
    }

    #[test]
    fn missing_node_returns_empty_bundle() {
        let g = NodeGraph::new();
        let bundle = build_for_iface(&g, NodeId::new(), Layout::SingleCrate, None);
        assert!(bundle.sections.is_empty());
    }

    #[test]
    fn dep_iface_warns_when_dep_has_no_public_rs() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let half_baked = g.add_child(root, Node::new("dep", "WIP")).unwrap();
        let user = g.add_child(root, Node::new("user", "")).unwrap();
        g.add_dep(user, half_baked).unwrap();
        let bundle = build_for_iface(&g, user, Layout::SingleCrate, None);
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
        let small = build_for_iface(&g, user, Layout::SingleCrate, None).approx_size();
        // Add a dep with a sizeable public.rs.
        let mut dep = Node::new("dep", "shared utility");
        dep.public_rs = Some("pub trait Big { ".to_string() + &"fn f(&self); ".repeat(50) + " }");
        let dep_id = g.add_child(root, dep).unwrap();
        g.add_dep(user, dep_id).unwrap();
        let big = build_for_iface(&g, user, Layout::SingleCrate, None).approx_size();
        assert!(big > small);
    }

    #[test]
    fn iface_context_for_root_omits_parent_section() {
        let (g, root) = fresh();
        let bundle = build_for_iface(&g, root, Layout::SingleCrate, None);
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

        let bundle = build_for_impl(&g, consumer_id, Layout::SingleCrate, None);
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
