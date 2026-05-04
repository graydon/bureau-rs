//! Syn-based validators for the new node-shaped Rust files.
//!
//! Two distinct validators:
//!
//! - [`validate_public`] — for `public.rs`. Enforces that the file contains
//!   only declarations: `pub trait`, `pub struct/enum/type`, `pub use`
//!   from `super::private` (allowed for re-exporting concrete impl types
//!   declared in private), doc comments, module-level attributes. **No
//!   `impl` blocks. No `fn` definitions outside trait declarations. No
//!   `pub use crate::*` cross-node re-exports.** This makes the public
//!   file genuinely a public-surface declaration.
//!
//! - [`validate_private`] — for `private.rs`. The file must compile by
//!   itself (we don't actually run `cargo check` here — that's the gate).
//!   The structural check is: every `use crate::<X>::...` path's first
//!   segment after `crate::` must resolve to a declared dep of this node
//!   (or to `super` / `self` for intra-node references). This stops the
//!   model from invisibly creating cross-node dep edges the framework
//!   doesn't know about.

use crate::graph::{Node, NodeGraph};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidateError {
    #[error("public.rs: invalid Rust syntax: {0}")]
    PublicSyntax(String),
    #[error("private.rs: invalid Rust syntax: {0}")]
    PrivateSyntax(String),
    #[error(
        "public.rs: forbidden item '{kind}' at line {line}; public.rs may contain only \
         pub trait/struct/enum/type declarations and doc comments"
    )]
    PublicForbiddenItem { kind: String, line: usize },
    #[error(
        "public.rs: trait method '{trait_}::{method}' has a default body; in this codebase \
         public.rs declares signatures only — move the body to private.rs as an `impl` block"
    )]
    PublicTraitDefaultBody { trait_: String, method: String },
    #[error(
        "public.rs: cross-node `pub use` of '{path}' is not allowed; declare items directly or \
         use a `pub type Alias = crate::other::Type` rename"
    )]
    PublicForbiddenPubUse { path: String },
    #[error(
        "private.rs: `use crate::{first}::...` — '{first}' is not a declared dep of this \
         node, an ancestor, or one of its own children. If you meant to import this node's \
         OWN public types, write `use super::public::*;` instead. If you meant to depend on \
         another node, name it (snake_case) in the spec stage's `decompose` deps list."
    )]
    PrivateUndeclaredDep { first: String },
    #[error(
        "private.rs: `use crate::{first}::...` references the node's own subtree but '{first}' \
         is not a known module name; check the spelling"
    )]
    PrivateUnknownModule { first: String },
}

/// Result of a successful `validate_public`: the set of public item names
/// declared in the file (e.g. trait names, struct names). Useful for the
/// renderer's `pub use public::*;` invariant — there must be at least one
/// public item to be useful, but we don't currently enforce that.
#[derive(Debug, Clone, Default)]
pub struct PublicSurface {
    pub traits: Vec<String>,
    pub types: Vec<String>,
}

pub fn validate_public(content: &str) -> Result<PublicSurface, ValidateError> {
    let file =
        syn::parse_file(content).map_err(|e| ValidateError::PublicSyntax(format!("{e}")))?;
    let mut surface = PublicSurface::default();
    for item in &file.items {
        check_public_item(item, &mut surface)?;
    }
    Ok(surface)
}

fn check_public_item(
    item: &syn::Item,
    surface: &mut PublicSurface,
) -> Result<(), ValidateError> {
    use syn::Item;
    match item {
        Item::Trait(t) => {
            // Trait method bodies are forbidden — we want pure signatures.
            for ti in &t.items {
                if let syn::TraitItem::Fn(f) = ti {
                    if f.default.is_some() {
                        return Err(ValidateError::PublicTraitDefaultBody {
                            trait_: t.ident.to_string(),
                            method: f.sig.ident.to_string(),
                        });
                    }
                }
            }
            surface.traits.push(t.ident.to_string());
            Ok(())
        }
        Item::Struct(s) => {
            surface.types.push(s.ident.to_string());
            Ok(())
        }
        Item::Enum(e) => {
            surface.types.push(e.ident.to_string());
            Ok(())
        }
        Item::Type(t) => {
            surface.types.push(t.ident.to_string());
            Ok(())
        }
        Item::Use(u) => {
            // Allow `pub use super::private::X` and `use ...` re-exports
            // that are intra-module (super::*). Disallow `pub use crate::*`
            // cross-node re-exports, since they create implicit dep edges
            // the framework can't see.
            if matches!(u.vis, syn::Visibility::Public(_))
                && pub_use_starts_with_crate(&u.tree)
            {
                let path = render_use_tree(&u.tree);
                return Err(ValidateError::PublicForbiddenPubUse { path });
            }
            Ok(())
        }
        Item::Const(_) | Item::Static(_) => {
            // Allow public consts/statics (e.g. error codes). They're
            // declarations with values, no behavior.
            Ok(())
        }
        Item::Mod(_) => {
            // Inline mod blocks in public.rs are forbidden; modules belong
            // to children nodes.
            Err(ValidateError::PublicForbiddenItem {
                kind: "module".to_string(),
                line: 0,
            })
        }
        Item::Fn(_) => Err(ValidateError::PublicForbiddenItem {
            kind: "free function".to_string(),
            line: 0,
        }),
        Item::Impl(_) => Err(ValidateError::PublicForbiddenItem {
            kind: "impl block".to_string(),
            line: 0,
        }),
        Item::ExternCrate(_) => Err(ValidateError::PublicForbiddenItem {
            kind: "extern crate".to_string(),
            line: 0,
        }),
        Item::Macro(_) => Err(ValidateError::PublicForbiddenItem {
            kind: "macro invocation".to_string(),
            line: 0,
        }),
        _ => {
            // Be lenient about other items — Rust adds new ones, and we
            // care about the four big offenders above. The compile gate
            // catches anything else that's broken.
            Ok(())
        }
    }
}

/// Validate `private.rs` (or `tests.rs`) imports against the node's declared
/// dep set. Allows `use super::*`, `use self::*`, and `use crate::<name>::*`
/// only if `<name>` is a child of the current node, an ancestor's name, or
/// a directly-declared dep node's name. Foreign nodes are rejected.
pub fn validate_private(
    content: &str,
    node: &Node,
    graph: &NodeGraph,
) -> Result<(), ValidateError> {
    let file =
        syn::parse_file(content).map_err(|e| ValidateError::PrivateSyntax(format!("{e}")))?;
    let allowed = allowed_first_segments(node, graph);
    for item in &file.items {
        if let syn::Item::Use(u) = item {
            check_use_tree(&u.tree, &allowed)?;
        }
    }
    Ok(())
}

/// Compute the set of first-segment names allowed in `use crate::<X>::...`
/// within this node's private/test files. Includes:
/// - the node's own children (their names)
/// - the node's own siblings (their names) — adjacent reaches via crate::
///   require this; but in practice we'll restrict cross-sibling deps to
///   declared deps only, so leave siblings out.
/// - declared dep nodes' names
/// - the names of every ancestor of this node up to (but not including)
///   the root (so a node can refer to its parent module)
fn allowed_first_segments(node: &Node, graph: &NodeGraph) -> HashSet<String> {
    let mut allowed: HashSet<String> = HashSet::new();
    // The node's own name (so it can refer to itself via `crate::<self>`).
    if node.parent.is_some() {
        allowed.insert(node.name.clone());
    }
    // Children.
    for child in graph.children_of(node.id) {
        allowed.insert(child.name.clone());
    }
    // Declared deps.
    for dep_id in &node.deps {
        if let Some(dep) = graph.get(*dep_id) {
            allowed.insert(dep.name.clone());
        }
    }
    // Ancestors (excluding the root, which is `crate` itself, not a named
    // first segment). We DO allow `crate::<name>` where <name> is an
    // ancestor *path-wise* — the model can refer back up its own subtree.
    // For a node at `crate::a::b::c`, `crate::a` is OK, `crate::a::b` is OK.
    let ancestors = graph.ancestors(node.id, false);
    for anc in ancestors {
        if let Some(a) = graph.get(anc) {
            // The root has no "name" segment in `crate::*` — `crate::` is
            // the root. We add the names of intermediate ancestors only.
            if a.parent.is_some() {
                allowed.insert(a.name.clone());
            }
        }
    }
    allowed
}

fn check_use_tree(tree: &syn::UseTree, allowed: &HashSet<String>) -> Result<(), ValidateError> {
    use syn::UseTree;
    // Walk the tree; when we encounter a `Path`, look at the FIRST segment.
    // We're interested in `crate::X::...` paths.
    match tree {
        UseTree::Path(p) => {
            let first = p.ident.to_string();
            if first == "crate" {
                // The next segment under `crate::` is what we check.
                check_after_crate(&p.tree, allowed)?;
            }
            // Other roots (`super`, `self`, foreign crate names) are fine
            // for v1 — `super::private::*` is the canonical sibling-module
            // import; foreign crates the model adds to Cargo.toml are not
            // graph-tracked, but the cargo gate catches missing crates.
            Ok(())
        }
        UseTree::Group(g) => {
            for child in &g.items {
                check_use_tree(child, allowed)?;
            }
            Ok(())
        }
        // Direct `use Name;` (no path) at the file root is a foreign-crate
        // or prelude reference; harmless to us.
        _ => Ok(()),
    }
}

fn check_after_crate(tree: &syn::UseTree, allowed: &HashSet<String>) -> Result<(), ValidateError> {
    use syn::UseTree;
    match tree {
        UseTree::Path(p) => {
            let first = p.ident.to_string();
            if !allowed.contains(&first) {
                return Err(ValidateError::PrivateUndeclaredDep { first });
            }
            Ok(())
        }
        UseTree::Name(n) => {
            // `use crate::Foo;` — Foo must be in allowed.
            let first = n.ident.to_string();
            if !allowed.contains(&first) {
                return Err(ValidateError::PrivateUndeclaredDep { first });
            }
            Ok(())
        }
        UseTree::Group(g) => {
            for child in &g.items {
                check_after_crate(child, allowed)?;
            }
            Ok(())
        }
        UseTree::Glob(_) => {
            // `use crate::*;` — disallow; too unspecific to track.
            Err(ValidateError::PrivateUndeclaredDep {
                first: "<glob *>".to_string(),
            })
        }
        UseTree::Rename(r) => {
            let first = r.ident.to_string();
            if !allowed.contains(&first) {
                return Err(ValidateError::PrivateUndeclaredDep { first });
            }
            Ok(())
        }
    }
}

/// True if this UseTree's first segment is `crate`. Walks any wrapping
/// `Group` so that `pub use {crate::a::Foo, crate::b::Bar}` is also caught.
fn pub_use_starts_with_crate(tree: &syn::UseTree) -> bool {
    use syn::UseTree;
    match tree {
        UseTree::Path(p) => p.ident == "crate",
        UseTree::Name(n) => n.ident == "crate",
        UseTree::Rename(r) => r.ident == "crate",
        UseTree::Group(g) => g.items.iter().any(pub_use_starts_with_crate),
        UseTree::Glob(_) => false,
    }
}

fn render_use_tree(tree: &syn::UseTree) -> String {
    use quote::ToTokens;
    let mut tok = proc_macro2::TokenStream::new();
    tree.to_tokens(&mut tok);
    tok.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeGraph};

    fn fresh_graph() -> (NodeGraph, crate::graph::NodeId, crate::graph::NodeId) {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        (g, root, a)
    }

    // ---- validate_public ----

    #[test]
    fn public_accepts_traits_and_types() {
        let src = r#"
pub trait Frob {
    fn shape(&self) -> i32;
    fn new(x: i32) -> Self where Self: Sized;
}

pub struct Frobber(crate::a::FrobInner);

pub enum Color { Red, Green, Blue }

pub type Id = u64;
"#;
        let s = validate_public(src).unwrap();
        assert_eq!(s.traits, vec!["Frob"]);
        assert_eq!(s.types, vec!["Frobber", "Color", "Id"]);
    }

    #[test]
    fn public_rejects_impl_block() {
        let src = "pub struct X; impl X { pub fn new() -> Self { X } }";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(
            err,
            ValidateError::PublicForbiddenItem { ref kind, .. } if kind == "impl block"
        ));
    }

    #[test]
    fn public_rejects_free_function() {
        let src = "pub fn helper() -> i32 { 0 }";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenItem { .. }));
    }

    #[test]
    fn public_rejects_default_trait_body() {
        let src = "pub trait T { fn f(&self) -> i32 { 0 } }";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicTraitDefaultBody { .. }));
    }

    #[test]
    fn public_rejects_inline_module() {
        let src = "pub mod foo { pub struct X; }";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenItem { .. }));
    }

    #[test]
    fn public_rejects_pub_use_crate() {
        let src = "pub use crate::other::Type;";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenPubUse { .. }));
    }

    #[test]
    fn public_allows_use_super_private() {
        // Internal re-export from the private sibling module.
        let src = r#"
use super::private;

pub struct Wrapper(private::Inner);
"#;
        validate_public(src).unwrap();
    }

    #[test]
    fn public_allows_doc_comments_and_attrs() {
        let src = r#"
//! Module-level doc.
#![allow(unused)]

/// A trait.
pub trait T {
    /// A method.
    fn f(&self) -> i32;
}
"#;
        let s = validate_public(src).unwrap();
        assert_eq!(s.traits, vec!["T"]);
    }

    #[test]
    fn public_allows_pub_const_and_static() {
        let src = r#"
pub const MAX: usize = 100;
pub static HELLO: &str = "hi";
"#;
        validate_public(src).unwrap();
    }

    #[test]
    fn public_syntax_error_reported() {
        let src = "pub trait T { fn f( ;";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicSyntax(_)));
    }

    // ---- validate_private ----

    #[test]
    fn private_accepts_super_uses() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
use super::public::*;
pub(super) struct Inner;
"#;
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_accepts_use_of_declared_dep() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let b = g.add_child(root, Node::new("b", "")).unwrap();
        g.add_dep(a, b).unwrap();
        let src = r#"
use crate::b::Frob;
pub(super) struct Inner(Frob);
"#;
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_rejects_use_of_undeclared_dep() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let _b = g.add_child(root, Node::new("b", "")).unwrap();
        // `a` does NOT declare a dep on `b`.
        let src = "use crate::b::Frob;\npub(super) struct Inner(Frob);\n";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(matches!(
            err,
            ValidateError::PrivateUndeclaredDep { ref first } if first == "b"
        ));
    }

    #[test]
    fn private_accepts_use_of_own_child() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let _aa = g.add_child(a, Node::new("aa", "")).unwrap();
        let src = "use crate::a::aa::*;\n";
        // `crate::a::aa` from within `a`: first segment after `crate::` is
        // `a`, which is `a`'s own ancestor name. Allowed.
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_rejects_glob_under_crate() {
        let (g, _root, a) = fresh_graph();
        let src = "use crate::*;\n";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(matches!(err, ValidateError::PrivateUndeclaredDep { .. }));
    }

    #[test]
    fn private_accepts_external_crate() {
        let (g, _root, a) = fresh_graph();
        // External crates (declared in Cargo.toml) start with their crate
        // name, not `crate::`. We don't validate those — cargo will.
        let src = "use serde::Serialize;\nuse std::collections::HashMap;\n";
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_grouped_uses() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let b = g.add_child(root, Node::new("b", "")).unwrap();
        g.add_dep(a, b).unwrap();
        // Mix of allowed and disallowed in a group.
        let src = "use crate::{b::Frob, c::Other};\n";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        // 'c' is not a declared dep, should be rejected.
        assert!(matches!(
            err,
            ValidateError::PrivateUndeclaredDep { ref first } if first == "c"
        ));
    }

    #[test]
    fn private_allows_super_self_paths() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
use super::public::*;
use self::helper::Util;
mod helper {
    pub struct Util;
}
"#;
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_syntax_error_reported() {
        let (g, _root, a) = fresh_graph();
        let src = "fn broken( {";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(matches!(err, ValidateError::PrivateSyntax(_)));
    }

    #[test]
    fn private_aliased_use_validates_first_segment() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let _b = g.add_child(root, Node::new("b", "")).unwrap();
        // `use crate::b as alias` — `b` not declared as dep.
        let src = "use crate::b as bee;\n";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(matches!(err, ValidateError::PrivateUndeclaredDep { .. }));
    }
}
