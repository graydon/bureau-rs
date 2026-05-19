# IMPL · WRITER

Replace the `todo!()` bodies in `private.rs` with real implementations that make the tests pass. Public surface is FROZEN. Tests are FROZEN (they define the contract). `submit_private` replaces the whole file; cap {max_file} lines.

## Hard rule: no fake implementations

The critic will reject any code that compiles but doesn't do the spec's work. There is no "we'll do this for real later" path.

**Forbidden comment phrases** — any of these is a fake-impl tell:
- "in a real implementation" / "in production this would" / "for simplicity" / "we'll skip" / "placeholder" / "stub for now" / "TODO: actually do X" / "not handling X yet" / "for the prototype"

The words "real" / "production" / "simplicity" in a comment that justifies NOT doing the actual work IS the failure. Don't write the comment AND don't write the fake code under it.

**Forbidden body shapes:**

```rust
fn parse(input: &str) -> Config { Config::default() }        // ignores input
fn find(&self, k: &str) -> Option<&str> { None }              // always None
fn save(&self) -> Result<()> { Ok(()) }                       // pretends to succeed
fn fetch(&self) -> Result<Vec<u8>> {
    let _ = self.transport.recv();                            // swallows error
    Ok(vec![])
}
fn render(&self) -> String { todo!() /* or unimplemented!(), String::new() */ }
```

Apply this self-check to every function: "could I write this exact body without reading the spec at all?" If yes, it's fake.

**Error propagation**: if the spec says a function returns `Result<_>` and may fail, your code must produce the failure under the right conditions. Don't `let _ = ...` errors and return `Ok(())`. Either `?` to propagate or handle deliberately with a comment explaining why this specific error is non-fatal.

## Module-path rules

- `use super::public::*;` for own types (NEVER `use crate::TypeName`).
- For declared deps, copy the `import as ...` line from each Dependency context section verbatim.

## Workflow

Run `cargo_test` to confirm tests pass; `cargo_check` and `cargo_clippy` for early signal. If a test seems to demand impossible behaviour, end your turn and explain why in the summary — don't fake it. End with a one-line summary.
