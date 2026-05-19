# IFACE · WRITER

Author the public surface and a stub private impl. Follow the trait+newtype split from common.md's "How we shape Rust" section — that's the canonical reference for what goes where. This preamble covers WORKFLOW (cap {max_file} lines per file).

Workflow:
1. Submit `public.rs` — trait declarations + concrete types + newtype wrappers around private repr. `impl`, `mod`, `pub use`, and free `fn` outside traits are FORBIDDEN in public.rs (see common.md for why).
2. Submit `private.rs` — one `impl Trait for Newtype` block per trait, method bodies as `todo!()`. Plus the private representation types the newtypes wrap.
3. Run `cargo_check` to verify; end with a one-line summary.

The stubs let dependents compile NOW; the next stage replaces them with real logic.

## Common pitfalls

- **`pub fn foo() -> Bar;`** (signature + semicolon) is a syntax error in Rust — there are no forward declarations. Put method signatures in a `pub trait { ... }`, NOT loose in the module.
- **`pub use super::private::Foo`** (smuggle) is rejected by the validator. Define types in `public.rs`. To rename a foreign type, use `pub type Alias = crate::other_node::Type;`.
- **Non-pub `use super::private::Inner`** in `public.rs` IS fine when you need to refer to a private type in a public type position (e.g. `pub struct Wrapper(super::private::Inner);`).

## Module-path rules in `private.rs`

- For OWN public types: `use super::public::*;` — NEVER `use crate::TypeName`.
- For a DECLARED DEP: copy the `import as ...` line from the dep's context section verbatim.
- First segment after `crate::` MUST resolve to a declared dep, an ancestor, an own child, or this node itself.

If this node has children (visible in the graph overview), it's an UMBRELLA — `public.rs` can be empty (doc comments only); children carry the real surface.
