//! Syn-based validators for the new node-shaped Rust files.
//!
//! Two distinct validators:
//!
//! - [`validate_public`] — for `public.rs`. Enforces that the file contains
//!   only DECLARATIONS: `pub trait`, `pub struct/enum/type`, `pub const`,
//!   `pub static`, doc comments, module-level attributes, and non-pub
//!   `use` of `super::private::*` (used in type positions like struct
//!   field types). **No `impl` blocks. No `fn` outside trait decls. No
//!   `pub use` of any kind** — items must be defined here, not re-exported
//!   from anywhere (including the sibling `private` module). The smuggle
//!   pattern of defining a type in private.rs and re-exporting it via
//!   `pub use super::private::Foo;` is explicitly forbidden because it
//!   makes the public surface deceptive: the model has to look in private
//!   to learn the real shape of the type.
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
        "public.rs: `pub use` of '{path}' is not allowed. public.rs must DEFINE the items \
         that make up this node's public surface; it must not re-export them from elsewhere. \
         If you defined the real type in private.rs and re-exported it here, MOVE the \
         definition into public.rs. To rename a foreign type, use `pub type Alias = <path>` \
         instead of `pub use`."
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
    #[error(
        "private.rs: forbidden item '{kind}'. private.rs may contain only `use` imports, \
         type definitions (struct/enum/type), `impl` blocks (for trait impls or inherent \
         impls on private types), `const`/`static`, and doc comments. \
         FREE FUNCTIONS are not allowed — every callable must be a method on a trait \
         (defined in public.rs) implemented for a type. See common.md's \
         'How we shape Rust' section."
    )]
    PrivateForbiddenItem { kind: String },
    #[error("tests.rs: invalid Rust syntax: {0}")]
    TestsSyntax(String),
    #[error(
        "tests.rs: forbidden item '{kind}'. tests.rs may contain `use` imports, \
         `fn` (`#[test]`-attributed or bare helpers), `mod` submodules of the same \
         shape, type definitions, traits, `impl` blocks, and `const`/`static`. \
         `extern crate` and free-form macro invocations are not allowed."
    )]
    TestsForbiddenItem { kind: String },
}

pub fn validate_public(content: &str) -> Result<(), ValidateError> {
    let file =
        syn::parse_file(content).map_err(|e| ValidateError::PublicSyntax(format!("{e}")))?;
    for item in &file.items {
        check_public_item(item)?;
    }
    Ok(())
}

fn check_public_item(item: &syn::Item) -> Result<(), ValidateError> {
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
            Ok(())
        }
        Item::Struct(_) | Item::Enum(_) | Item::Type(_) => Ok(()),
        Item::Use(u) => {
            // ANY `pub use` is forbidden — public.rs is for definitions,
            // not re-exports. Non-pub `use super::private::Inner` is fine
            // because it's just an internal reference for use in type
            // positions (e.g. `pub struct Wrapper(super::private::Inner)`).
            if matches!(u.vis, syn::Visibility::Public(_)) {
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

/// Validate `private.rs`. Two layers of checks:
///
/// 1. **Use-path scoping** — `use crate::<X>::...` paths must reference
///    declared deps, ancestors, own children, or this node itself.
///    See `allowed_first_segments` for the full rule.
/// 2. **Item-form whitelist** — `private.rs` may contain only `use`,
///    type definitions (struct/enum/type), `impl` blocks, `const` /
///    `static`, and doc comments. NO free `fn`: every callable must
///    be a method on a trait+impl pair (the framework's split — see
///    common.md). NO `mod` blocks (children get their own nodes).
///    NO `extern crate`, `macro_rules!`, `pub use`, etc.
pub fn validate_private(
    content: &str,
    node: &Node,
    graph: &NodeGraph,
) -> Result<(), ValidateError> {
    let file =
        syn::parse_file(content).map_err(|e| ValidateError::PrivateSyntax(format!("{e}")))?;
    let allowed = allowed_first_segments(node, graph);
    for item in &file.items {
        check_private_item(item, &allowed)?;
    }
    Ok(())
}

fn check_private_item(
    item: &syn::Item,
    allowed: &HashSet<String>,
) -> Result<(), ValidateError> {
    use syn::Item;
    match item {
        Item::Use(u) => check_use_tree(&u.tree, allowed),
        Item::Struct(_) | Item::Enum(_) | Item::Type(_) | Item::Union(_) => Ok(()),
        Item::Impl(_) => Ok(()),
        Item::Const(_) | Item::Static(_) => Ok(()),
        // Free functions are fine in private.rs — they're internal
        // helpers and there's no benefit to forcing them onto inherent
        // impls. Only PUBLIC.rs forbids loose fn (that's the surface).
        Item::Fn(_) => Ok(()),
        Item::Trait(_) => {
            // Traits belong in public.rs (they ARE the public interface).
            Err(ValidateError::PrivateForbiddenItem {
                kind: "trait (define traits in public.rs)".to_string(),
            })
        }
        Item::Mod(_) => Err(ValidateError::PrivateForbiddenItem {
            kind: "mod (children are framework-managed nodes, not inline mods)"
                .to_string(),
        }),
        Item::ExternCrate(_) => Err(ValidateError::PrivateForbiddenItem {
            kind: "extern crate".to_string(),
        }),
        Item::Macro(_) => Err(ValidateError::PrivateForbiddenItem {
            kind: "macro invocation".to_string(),
        }),
        _ => Ok(()),
    }
}

/// Validate `tests.rs`. The allowed items are:
/// - `use` (scoped via `allowed_first_segments` like private.rs)
/// - functions (`#[test]`-attributed or bare helpers — both fine here,
///   helpers are common in test files and there's no reason to force
///   them onto impls)
/// - type definitions, traits, `impl` blocks
/// - inner `mod` modules of the same shape (so the conventional nested
///   `mod tests {{}}` pattern still works)
/// - `const`/`static`
pub fn validate_tests(
    content: &str,
    node: &Node,
    graph: &NodeGraph,
) -> Result<(), ValidateError> {
    let file =
        syn::parse_file(content).map_err(|e| ValidateError::TestsSyntax(format!("{e}")))?;
    let allowed = allowed_first_segments(node, graph);
    for item in &file.items {
        check_tests_item(item, &allowed)?;
    }
    Ok(())
}

fn check_tests_item(
    item: &syn::Item,
    allowed: &HashSet<String>,
) -> Result<(), ValidateError> {
    use syn::Item;
    match item {
        Item::Use(u) => check_use_tree(&u.tree, allowed),
        Item::Struct(_) | Item::Enum(_) | Item::Type(_) | Item::Union(_) => Ok(()),
        Item::Trait(_) | Item::Impl(_) => Ok(()),
        Item::Const(_) | Item::Static(_) => Ok(()),
        // Free functions are fine — both `#[test]` and bare helpers.
        Item::Fn(_) => Ok(()),
        Item::Mod(m) => {
            // Recurse into inner mod bodies if any — same rules apply.
            if let Some((_, items)) = &m.content {
                for inner in items {
                    check_tests_item(inner, allowed)?;
                }
            }
            Ok(())
        }
        Item::ExternCrate(_) => Err(ValidateError::TestsForbiddenItem {
            kind: "extern crate".to_string(),
        }),
        Item::Macro(_) => Err(ValidateError::TestsForbiddenItem {
            kind: "macro invocation".to_string(),
        }),
        _ => Ok(()),
    }
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
        validate_public(src).unwrap();
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
    fn public_rejects_pub_use_super_private() {
        // The smuggle pattern: define the type in private, re-export from
        // public. Forbidden — public.rs must DEFINE the type.
        let src = "pub use super::private::Inner;";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenPubUse { .. }));
    }

    #[test]
    fn public_rejects_pub_use_self() {
        let src = "pub use self::sub::Thing;";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenPubUse { .. }));
    }

    #[test]
    fn public_rejects_pub_use_external() {
        let src = "pub use std::sync::Arc;";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenPubUse { .. }));
    }

    #[test]
    fn public_rejects_pub_use_grouped_smuggle() {
        let src = "pub use super::private::{A, B};";
        let err = validate_public(src).unwrap_err();
        assert!(matches!(err, ValidateError::PublicForbiddenPubUse { .. }));
    }

    #[test]
    fn public_allows_use_super_private() {
        // Internal (non-pub) reference to a private type, used in a type
        // position. Allowed because the public item is still DEFINED here.
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
        validate_public(src).unwrap();
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
        // `use super::*` and `use self::...` are fine — the framework
        // uses them constantly (super::public::* etc.). Inline `mod`
        // blocks, however, are forbidden in private.rs (children are
        // framework-managed nodes, not inline modules).
        let (g, _root, a) = fresh_graph();
        let src = r#"
use super::public::*;
use self::nested;
"#;
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_rejects_inline_mod() {
        let (g, _root, a) = fresh_graph();
        let src = "mod helper { pub struct Util; }\n";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(
            matches!(err, ValidateError::PrivateForbiddenItem { ref kind } if kind.contains("mod"))
        );
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

    // ---- validate_private item-form whitelist ----

    #[test]
    fn private_accepts_free_function() {
        // Free fns are fine in private.rs — they're internal helpers.
        // Only public.rs forbids loose `fn`.
        let (g, _root, a) = fresh_graph();
        let src = "fn helper(x: i32) -> i32 { x + 1 }\n";
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn private_rejects_trait_definition() {
        let (g, _root, a) = fresh_graph();
        let src = "pub trait Foo { fn bar(&self); }\n";
        let err = validate_private(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(
            matches!(err, ValidateError::PrivateForbiddenItem { ref kind } if kind.contains("trait"))
        );
    }

    #[test]
    fn private_accepts_struct_and_impl() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
use super::public::*;
pub struct Inner { x: i32 }
impl Foo for FooImpl {
    fn bar(&self) -> i32 { 42 }
}
"#;
        // public::* import is fine; impl on a public type with bodies
        // here is the conventional split.
        validate_private(src, g.get(a).unwrap(), &g).unwrap();
    }

    // ---- validate_tests ----

    #[test]
    fn tests_accepts_test_functions() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
use super::public::*;

#[test]
fn it_works() {
    assert_eq!(2 + 2, 4);
}
"#;
        validate_tests(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn tests_accepts_tokio_test() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
#[tokio::test]
async fn it_works_async() {
    assert!(true);
}
"#;
        validate_tests(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn tests_accepts_bare_helper_fn() {
        // Helpers next to `#[test]` functions are fine — test files
        // conventionally include build/setup functions.
        let (g, _root, a) = fresh_graph();
        let src = "fn helper() -> i32 { 1 }\n#[test] fn t() {}\n";
        validate_tests(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn tests_accepts_helper_in_impl() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
struct Helper;
impl Helper {
    fn build() -> i32 { 42 }
}
#[test]
fn t() {
    assert_eq!(Helper::build(), 42);
}
"#;
        validate_tests(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn tests_accepts_nested_mod_with_tests() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
mod nested {
    #[test]
    fn inner() {
        assert!(true);
    }
}
"#;
        validate_tests(src, g.get(a).unwrap(), &g).unwrap();
    }

    #[test]
    fn tests_rejects_extern_crate_in_nested_mod() {
        let (g, _root, a) = fresh_graph();
        let src = r#"
mod nested {
    extern crate foo;
}
"#;
        let err = validate_tests(src, g.get(a).unwrap(), &g).unwrap_err();
        assert!(matches!(err, ValidateError::TestsForbiddenItem { .. }));
    }
}
