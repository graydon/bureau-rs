//! Placeholder strings the framework writes to slot files when a node
//! hasn't yet authored that slot. Centralized so `render::render_node`
//! (which writes them) and `graph::load` (which reads them and decides
//! "this slot is unauthored — leave the in-memory field as `None`")
//! agree byte-for-byte.
//!
//! The placeholders must compile as valid Rust / parse as markdown so
//! the framework's incremental cargo gate works while stages are still
//! in flight.

/// Placeholder for `public.rs` when the iface stage hasn't authored it.
pub const PUBLIC_RS: &str = "// public surface — not yet authored\n";

/// Placeholder for `private.rs` when the impl stage hasn't authored it.
pub const PRIVATE_RS: &str = "// private internals — not yet authored\n";

/// Placeholder for `tests.rs` when the tests stage hasn't authored it.
pub const TESTS_RS: &str = "// tests — not yet authored\n";

/// Placeholder for `spec/<name>/public.md` when the spec stage hasn't
/// authored it. No node-name or description interpolation — `graph::load`
/// matches this by exact bytes.
pub const PUBLIC_MD: &str = "*public spec not yet authored*\n";

/// Decide whether `content` is the placeholder for the given slot kind.
/// Used by `graph::load` to map "on-disk placeholder" → `None` when
/// populating in-memory content fields from rendered files.
pub fn is_placeholder_public_rs(content: &str) -> bool {
    content == PUBLIC_RS
}
pub fn is_placeholder_private_rs(content: &str) -> bool {
    content == PRIVATE_RS
}
pub fn is_placeholder_tests_rs(content: &str) -> bool {
    content == TESTS_RS
}
pub fn is_placeholder_public_md(content: &str) -> bool {
    content == PUBLIC_MD
}
