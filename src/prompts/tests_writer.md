# TESTS · WRITER

Author `#[test]` functions exercising this node's public surface against the spec. The framework wraps your content in `#[cfg(test)] mod tests { ... }`. Cap: {max_file} lines.

Tests will COMPILE because `private.rs` has `todo!()` stubs satisfying the trait at the type level — they FAIL at runtime, which is expected. The next stage replaces the stubs and the same tests pass.

Workflow:
1. Import via `use super::public::*;` (NEVER `use crate::TypeName`).
2. Cover the spec's invariants and edge cases — see triviality rule below.
3. Run `cargo_test_no_run` to verify the file compiles.
4. End with a one-line summary.

## Hard rule: tests that pass against a "do-nothing" impl WILL BE DELETED

Self-check every test before submitting: **imagine a malicious impl where every method returns `Default::default()`, every collection is empty, every `Result` is `Ok(())`. Would your test pass?** If yes, delete it.

**Forbidden test shapes** (the critic deletes these):

```rust
// constructor-as-identity
let s = Foo { name: "x".into(), n: 42 };
assert_eq!(s.name, "x");

// default-equals-default
assert_eq!(Foo::default(), Foo::default());

// type-identity (compiler proves this)
let s: Foo = Foo::new();
assert!(matches!(s, Foo { .. }));

// setter-then-getter
s.set_n(7); assert_eq!(s.n, 7);

// round-tripping a getter
let s = Foo::with_n(5); assert_eq!(s.get_n(), 5);

// language-guarantee (compiler proves this)
fn _check_send<T: Send>() {} _check_send::<Foo>();
```

**Good tests** assert observable behavior the spec promises:

```rust
let parsed = Config::parse("[s]\nk=v")?;
assert_eq!(parsed.get("s", "k"), Some("v"));   // spec'd behavior

assert!(Config::parse("[").is_err());          // spec'd edge case

let s = Session::open();
s.close(); s.close();                          // spec says close is idempotent
```

One rich test exercising a real workflow beats a handful of one-line getter tests. If the only tests you can write for this node are constructor-as-identity, submit a near-empty `tests.rs` with a comment explaining there's no spec-relevant behaviour to test.

## Other things NOT to test

- **Other nodes' contracts**: tests for node X test X's public interface ONLY. No `tests::fixture_files_exist`, `tests::all_binary_entry_points_exist`, etc.
- **Implementation details**: don't test private internals. Test through the public surface.
- **Trivial assertions**: `assert_eq!(2 + 2, 4)`-style filler.

## Module-path rules

`use crate::<X>::...` rule same as `private.rs`: X must be a declared dep / ancestor / own child. Don't write integration tests requiring network or filesystem unless the spec calls for them.
