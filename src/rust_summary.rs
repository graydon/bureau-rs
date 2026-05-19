//! Compact, signature-only summaries of Rust source for context bundles.
//!
//! When the engine ships a node's context to the LLM, it inlines the
//! relevant public surfaces of dep crates and parent ifaces. Full files
//! are wasteful: the model only needs to know *what shapes are available*
//! to call, not the bodies of trait impls or `#[inline]` decoration on
//! private helpers. Summaries cut token spend on every prompt across
//! every (node, stage) pair — the biggest single context-size lever
//! short of LLM-summarizing the prose specs (#107).
//!
//! What we KEEP:
//!  - All top-level items except `use` / `extern crate`.
//!  - Trait declarations (signatures, no default bodies).
//!  - Struct/enum/type/const/static AS-WRITTEN (they're already signatures).
//!  - Fn signatures (drop bodies → `;`).
//!  - Impl headers (drop inner items → `{ /* ... */ }`).
//!  - `#[derive(...)]` attributes (semantically API-relevant).
//!
//! What we DROP:
//!  - `use` / `extern crate` (noise in summaries; readers know imports
//!    from context).
//!  - Doc comments (`///`, `//!`, `#[doc = "..."]`).
//!  - Function bodies.
//!  - Inner impl items.
//!  - Other attributes (`#[inline]`, `#[cfg]`, etc.) — they're typically
//!    not signature-affecting from a caller's perspective.
//!
//! Falls back to the original content unchanged if the file fails to
//! parse (defensive — never returns a worse-than-input view).

use syn::{File, Item, TraitItem};
use thiserror::Error;

/// Return a compact summary of the given Rust source, or the original
/// content if parsing fails.
pub fn summarize_rust(content: &str) -> String {
    let file = match syn::parse_file(content) {
        Ok(f) => f,
        Err(_) => return content.to_string(),
    };
    let mut summarized = File {
        shebang: None,
        attrs: Vec::new(),
        items: Vec::new(),
    };
    for item in &file.items {
        if let Some(s) = summarize_item(item) {
            summarized.items.push(s);
        }
    }
    if summarized.items.is_empty() {
        // No signature-bearing items found; the file was just imports
        // and doc comments. Emit a tiny marker so the bundle reader
        // can tell the file existed but had nothing to show.
        return "// (no public-surface items)\n".to_string();
    }
    prettyplease::unparse(&summarized)
}

fn summarize_item(item: &Item) -> Option<Item> {
    match item {
        // Strip imports — bundle readers see them in path context, not here.
        Item::Use(_) | Item::ExternCrate(_) => None,

        Item::Trait(t) => {
            let mut t = t.clone();
            strip_non_derive_attrs(&mut t.attrs);
            t.items = t
                .items
                .into_iter()
                .filter_map(summarize_trait_item)
                .collect();
            Some(Item::Trait(t))
        }

        Item::Struct(s) => {
            let mut s = s.clone();
            strip_non_derive_attrs(&mut s.attrs);
            Some(Item::Struct(s))
        }
        Item::Enum(e) => {
            let mut e = e.clone();
            strip_non_derive_attrs(&mut e.attrs);
            Some(Item::Enum(e))
        }
        Item::Union(u) => {
            let mut u = u.clone();
            strip_non_derive_attrs(&mut u.attrs);
            Some(Item::Union(u))
        }
        Item::Type(t) => {
            let mut t = t.clone();
            strip_non_derive_attrs(&mut t.attrs);
            Some(Item::Type(t))
        }
        Item::Const(c) => {
            let mut c = c.clone();
            strip_non_derive_attrs(&mut c.attrs);
            Some(Item::Const(c))
        }
        Item::Static(s) => {
            let mut s = s.clone();
            strip_non_derive_attrs(&mut s.attrs);
            Some(Item::Static(s))
        }

        // Drop function bodies — replace with empty block. Result is
        // still valid Rust (so prettyplease handles it) but carries
        // zero implementation noise.
        Item::Fn(f) => {
            let mut f = f.clone();
            strip_non_derive_attrs(&mut f.attrs);
            f.block = Box::new(syn::Block {
                brace_token: Default::default(),
                stmts: Vec::new(),
            });
            Some(Item::Fn(f))
        }

        // Drop inner impl items — empty body. The trait declaration
        // (if any) is in the same file via the corresponding `trait`
        // item and already lists the method signatures.
        Item::Impl(i) => {
            let mut i = i.clone();
            strip_non_derive_attrs(&mut i.attrs);
            i.items = Vec::new();
            Some(Item::Impl(i))
        }

        // Mod blocks are rare in our codebase (the framework forbids
        // them in private.rs) but include their signature-summarized
        // contents if we see one.
        Item::Mod(m) => {
            let mut m = m.clone();
            strip_non_derive_attrs(&mut m.attrs);
            if let Some((brace, items)) = m.content {
                let summarized: Vec<Item> = items.into_iter().filter_map(|i| summarize_item(&i)).collect();
                m.content = Some((brace, summarized));
            }
            Some(Item::Mod(m))
        }

        // Macro defs, foreign mods, traitalias — keep as-is. Rare
        // enough not to merit special handling.
        other => Some(other.clone()),
    }
}

fn summarize_trait_item(item: TraitItem) -> Option<TraitItem> {
    match item {
        TraitItem::Fn(mut f) => {
            strip_non_derive_attrs(&mut f.attrs);
            // Drop default bodies. The signature alone is what callers
            // care about.
            f.default = None;
            Some(TraitItem::Fn(f))
        }
        TraitItem::Type(mut t) => {
            strip_non_derive_attrs(&mut t.attrs);
            Some(TraitItem::Type(t))
        }
        TraitItem::Const(mut c) => {
            strip_non_derive_attrs(&mut c.attrs);
            // Keep the default value — for trait consts it IS the surface.
            Some(TraitItem::Const(c))
        }
        TraitItem::Macro(m) => Some(TraitItem::Macro(m)),
        TraitItem::Verbatim(v) => Some(TraitItem::Verbatim(v)),
        _ => None,
    }
}

/// Strip every attribute except `#[derive(...)]`. Derive macros change
/// what trait impls are visible on a type — they're part of the public
/// surface a caller can reach. `#[inline]`, `#[cfg]`, `#[doc]`, etc.
/// are not signature-affecting for callers; they bloat the summary.
fn strip_non_derive_attrs(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|a| a.path().is_ident("derive"));
}

// --------------------------------------------------------------------------
// Item lookup — supporting `read_item` / `write_item` tools.
// --------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ItemLookupError {
    #[error("invalid Rust syntax: {0}")]
    Syntax(String),
    #[error(
        "no top-level item named '{name}' in the file. Items present: [{available}]"
    )]
    NotFound { name: String, available: String },
    #[error(
        "name '{name}' matches multiple top-level items: [{candidates}]. \
         Use a fully-qualified form to disambiguate \
         (for example, `Trait for Type` instead of just `Type`)."
    )]
    Ambiguous { name: String, candidates: String },
}

/// Render the named top-level item from `content` as standalone Rust text.
/// Preserves attributes (including doc comments via `#[doc = "..."]`) and
/// passes through `prettyplease` for readability.
///
/// `name` matches against the item's "canonical names" (see
/// [`canonical_names`]). For most items this is just the identifier
/// (`Foo` matches `struct Foo`, `trait Foo`, etc.). For impls it's
/// either the self-type alone (`Foo` for `impl Trait for Foo`) or the
/// full header (`Trait for Foo`). When a name is ambiguous (e.g. both
/// a trait impl and an inherent impl exist with self-type `Foo`), the
/// caller gets [`ItemLookupError::Ambiguous`] listing every candidate.
pub fn read_top_item(content: &str, name: &str) -> Result<String, ItemLookupError> {
    let file = parse_file(content)?;
    let idx = find_item_index(&file, name)?;
    let item = file.items[idx].clone();
    let singleton = File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![item],
    };
    Ok(prettyplease::unparse(&singleton))
}

/// Replace the named top-level item with `new_item_src` and return the
/// resulting full-file text. `new_item_src` must parse as ONE Rust item
/// of the same kind as the existing one (so callers can't accidentally
/// swap a `trait` for a `fn`). The returned text is `prettyplease`-formatted.
pub fn write_top_item(
    content: &str,
    name: &str,
    new_item_src: &str,
) -> Result<String, ItemLookupError> {
    let mut file = parse_file(content)?;
    let idx = find_item_index(&file, name)?;
    let new_file = parse_file(new_item_src)?;
    if new_file.items.len() != 1 {
        return Err(ItemLookupError::Syntax(format!(
            "write_item: expected exactly one top-level item, got {}",
            new_file.items.len()
        )));
    }
    let old_kind = item_kind(&file.items[idx]);
    let new_kind = item_kind(&new_file.items[0]);
    if old_kind != new_kind {
        return Err(ItemLookupError::Syntax(format!(
            "write_item: existing item is {old_kind}, replacement is {new_kind} — \
             they must be the same kind"
        )));
    }
    file.items[idx] = new_file.items.into_iter().next().unwrap();
    Ok(prettyplease::unparse(&file))
}

fn parse_file(content: &str) -> Result<File, ItemLookupError> {
    syn::parse_file(content).map_err(|e| ItemLookupError::Syntax(format!("{e}")))
}

fn find_item_index(file: &File, name: &str) -> Result<usize, ItemLookupError> {
    let mut matches: Vec<(usize, String)> = Vec::new();
    let mut all_names: Vec<String> = Vec::new();
    for (i, item) in file.items.iter().enumerate() {
        let names = canonical_names(item);
        for n in &names {
            all_names.push(n.clone());
            if n == name {
                matches.push((i, n.clone()));
                break;
            }
        }
    }
    match matches.len() {
        0 => Err(ItemLookupError::NotFound {
            name: name.to_string(),
            available: all_names.join(", "),
        }),
        1 => Ok(matches[0].0),
        _ => Err(ItemLookupError::Ambiguous {
            name: name.to_string(),
            candidates: matches
                .into_iter()
                .map(|(_, n)| n)
                .collect::<Vec<_>>()
                .join(", "),
        }),
    }
}

/// One or more identifying names for an item. For nameable items
/// (trait/struct/enum/etc.) it's the identifier. For impl blocks
/// (which have no `ident`) we return two candidates: the self-type
/// alone AND the full `Trait for SelfType` header (for trait impls).
/// Both work as lookup keys — the more-specific one disambiguates
/// when self-type alone would be ambiguous.
fn canonical_names(item: &Item) -> Vec<String> {
    use syn::Item::*;
    match item {
        Trait(t) => vec![t.ident.to_string()],
        Struct(s) => vec![s.ident.to_string()],
        Enum(e) => vec![e.ident.to_string()],
        Union(u) => vec![u.ident.to_string()],
        Type(t) => vec![t.ident.to_string()],
        Const(c) => vec![c.ident.to_string()],
        Static(s) => vec![s.ident.to_string()],
        Fn(f) => vec![f.sig.ident.to_string()],
        Impl(i) => {
            let self_ty = type_to_simple_string(&i.self_ty);
            if let Some((_, path, _)) = &i.trait_ {
                let trait_str = path_to_simple_string(path);
                vec![format!("{trait_str} for {self_ty}"), self_ty]
            } else {
                vec![self_ty]
            }
        }
        TraitAlias(t) => vec![t.ident.to_string()],
        Mod(m) => vec![m.ident.to_string()],
        _ => Vec::new(),
    }
}

fn item_kind(item: &Item) -> &'static str {
    use syn::Item::*;
    match item {
        Trait(_) => "trait",
        Struct(_) => "struct",
        Enum(_) => "enum",
        Union(_) => "union",
        Type(_) => "type",
        Const(_) => "const",
        Static(_) => "static",
        Fn(_) => "fn",
        Impl(_) => "impl",
        TraitAlias(_) => "trait alias",
        Mod(_) => "mod",
        Use(_) => "use",
        ExternCrate(_) => "extern crate",
        Macro(_) => "macro",
        _ => "unknown",
    }
}

/// Render a `syn::Type` to its simplest token-stream form (no extra
/// whitespace). Strips path-prefix segments so `super::public::Foo`
/// renders as `Foo` — that's what the model writes when it asks to
/// edit "the impl on Foo".
fn type_to_simple_string(ty: &syn::Type) -> String {
    use quote::ToTokens;
    if let syn::Type::Path(tp) = ty {
        if let Some(last) = tp.path.segments.last() {
            return last.ident.to_string();
        }
    }
    // Fallback for non-path types (refs, tuples, etc.): full tokens.
    ty.to_token_stream().to_string()
}

fn path_to_simple_string(p: &syn::Path) -> String {
    p.segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_drops_use_and_doc_and_bodies() {
        let src = r#"
//! Module-level doc comment.

use std::collections::HashMap;
use crate::other::Type;

/// A trait that does the thing.
pub trait Frob {
    /// Method doc.
    fn shape(&self) -> i32;
    fn with_default(&self) -> i32 { 42 }
}

/// Doc comment.
pub struct Frobber {
    pub size: usize,
}

pub fn helper(x: i32) -> i32 {
    let y = x + 1;
    y * 2
}

impl Frob for Frobber {
    fn shape(&self) -> i32 {
        self.size as i32
    }
    fn with_default(&self) -> i32 {
        99
    }
}
"#;
        let out = summarize_rust(src);
        // Imports stripped.
        assert!(!out.contains("use std::collections"));
        assert!(!out.contains("use crate::other"));
        // Doc comments stripped (rendered via attribute would be #[doc = ...]).
        assert!(!out.contains("#[doc"));
        assert!(!out.contains("Module-level doc"));
        assert!(!out.contains("Method doc"));
        // Signatures preserved.
        assert!(out.contains("pub trait Frob"));
        assert!(out.contains("fn shape"));
        assert!(out.contains("pub struct Frobber"));
        assert!(out.contains("pub fn helper"));
        // Bodies dropped.
        assert!(!out.contains("y * 2"));
        assert!(!out.contains("self.size as i32"));
        // Default body in trait method dropped.
        assert!(!out.contains("42"));
        // Impl shell preserved, contents dropped (empty `{}` body).
        assert!(out.contains("impl Frob for Frobber"));
        assert!(!out.contains("99"));
        // Function bodies dropped → empty block.
        assert!(out.contains("pub fn helper"));
    }

    #[test]
    fn summarize_keeps_derive_attrs() {
        let src = r#"
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub name: String,
}
"#;
        let out = summarize_rust(src);
        // Derive kept.
        assert!(out.contains("#[derive"));
        assert!(out.contains("Debug"));
        // Non-derive attribute stripped.
        assert!(!out.contains("#[serde"));
        assert!(!out.contains("snake_case"));
    }

    #[test]
    fn summarize_preserves_struct_fields() {
        // Struct fields ARE the signature — keep them.
        let src = "pub struct Point { pub x: f64, pub y: f64 }\n";
        let out = summarize_rust(src);
        assert!(out.contains("x: f64"));
        assert!(out.contains("y: f64"));
    }

    #[test]
    fn summarize_preserves_enum_variants() {
        let src = "pub enum Color { Red, Green, Blue }\n";
        let out = summarize_rust(src);
        assert!(out.contains("Red"));
        assert!(out.contains("Green"));
        assert!(out.contains("Blue"));
    }

    #[test]
    fn summarize_invalid_syntax_returns_input() {
        let src = "pub fn broken( {";
        let out = summarize_rust(src);
        assert_eq!(out, src);
    }

    #[test]
    fn summarize_empty_file_returns_marker() {
        let src = "// just a comment\n";
        let out = summarize_rust(src);
        assert!(out.contains("no public-surface items"));
    }

    #[test]
    fn summarize_keeps_pub_type_aliases() {
        let src = "pub type NodeId = u64;\n";
        let out = summarize_rust(src);
        assert!(out.contains("pub type NodeId = u64"));
    }

    #[test]
    fn summarize_keeps_pub_const() {
        let src = "pub const MAX: usize = 100;\n";
        let out = summarize_rust(src);
        assert!(out.contains("pub const MAX"));
        assert!(out.contains("100"));
    }

    // ---- item lookup ----

    #[test]
    fn read_item_finds_struct_by_name() {
        let src = r#"
pub struct Foo { pub x: i32 }
pub struct Bar { pub y: u8 }
"#;
        let out = read_top_item(src, "Foo").unwrap();
        assert!(out.contains("Foo"));
        assert!(out.contains("x: i32"));
        assert!(!out.contains("Bar"));
    }

    #[test]
    fn read_item_finds_trait_by_name() {
        let src = "pub trait Frob { fn shape(&self) -> i32; }\n";
        let out = read_top_item(src, "Frob").unwrap();
        assert!(out.contains("pub trait Frob"));
        assert!(out.contains("fn shape"));
    }

    #[test]
    fn read_item_finds_inherent_impl_by_self_type() {
        // No struct/trait collision: lookup by self-type works for an
        // inherent impl.
        let src = "impl FrobberImpl { fn build() {} }\n";
        let out = read_top_item(src, "FrobberImpl").unwrap();
        assert!(out.contains("impl FrobberImpl"));
        assert!(out.contains("fn build"));
    }

    #[test]
    fn read_item_disambiguates_impl_via_trait_for_self() {
        let src = r#"
pub struct FrobberImpl;
impl Frob for FrobberImpl {
    fn shape(&self) -> i32 { 1 }
}
"#;
        // The fully-qualified form picks the impl unambiguously.
        let out = read_top_item(src, "Frob for FrobberImpl").unwrap();
        assert!(out.contains("impl Frob for FrobberImpl"));
        // Self-type alone is ambiguous (struct + impl).
        let err = read_top_item(src, "FrobberImpl").unwrap_err();
        assert!(matches!(err, ItemLookupError::Ambiguous { .. }));
    }

    #[test]
    fn read_item_not_found_lists_available() {
        let src = "pub struct Foo;\npub struct Bar;\n";
        let err = read_top_item(src, "Quux").unwrap_err();
        match err {
            ItemLookupError::NotFound { available, .. } => {
                assert!(available.contains("Foo"));
                assert!(available.contains("Bar"));
            }
            _ => panic!("expected NotFound"),
        }
    }

    #[test]
    fn write_item_replaces_in_place() {
        let src = r#"
pub struct Foo { pub x: i32 }
pub struct Bar { pub y: u8 }
"#;
        let out = write_top_item(src, "Foo", "pub struct Foo { pub x: i64, pub z: bool }").unwrap();
        // Foo updated, Bar untouched.
        assert!(out.contains("x: i64"));
        assert!(out.contains("z: bool"));
        assert!(out.contains("Bar"));
        assert!(out.contains("y: u8"));
        // Old field gone.
        assert!(!out.contains("x: i32"));
    }

    #[test]
    fn write_item_rejects_kind_mismatch() {
        let src = "pub struct Foo;\n";
        let err = write_top_item(src, "Foo", "pub trait Foo { fn x(&self); }").unwrap_err();
        assert!(matches!(err, ItemLookupError::Syntax(ref m) if m.contains("same kind")));
    }

    #[test]
    fn write_item_rejects_multiple_items_in_replacement() {
        let src = "pub struct Foo;\n";
        let err = write_top_item(src, "Foo", "pub struct Foo; pub struct Bar;").unwrap_err();
        assert!(matches!(err, ItemLookupError::Syntax(ref m) if m.contains("exactly one")));
    }

    #[test]
    fn write_item_not_found_returns_error() {
        let src = "pub struct Foo;\n";
        let err = write_top_item(src, "Bar", "pub struct Bar;").unwrap_err();
        assert!(matches!(err, ItemLookupError::NotFound { .. }));
    }
}
