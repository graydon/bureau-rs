# bureau-rs roadmap

A running list of bigger directions the project could go. Items here
are NOT committed work — they're sketches to push on later. Each entry
should explain *what*, *why*, and at least *one* way it could be done.
Update when an idea sharpens; delete when it lands or is ruled out.

## Catalogue + auto-routing of items

**Current state.** Each node owns a `public.rs` (declarations only) and
a `private.rs` (the implementation). The model writes free-form Rust
into each via `submit_public` / `submit_private`. A validator rejects
forbidden constructs (free `fn` in `public.rs`, impl blocks, `mod`),
but the model is otherwise responsible for *where* each item lives.

**The problem.** Several recurring failure modes come from items being
in the wrong place:

- A trait declared in node A's `public.rs` whose only impl is for a
  type that doesn't exist outside node A — should the trait be in A,
  or should there be a "types" leaf node holding the trait that A
  imports? The model picks one and downstream nodes work around it.
- A data type used by N siblings ends up duplicated in each one, or
  defined in one with the others importing across a non-existent dep
  edge.
- A "pub struct Foo(InternalType)" newtype wrapper around a private
  type — does the wrapper go in `public.rs` (the wrapper is public)
  or `private.rs` (the inner is private)? The current validator
  forces it into `public.rs`, but the model often confuses itself
  about visibility.

**A direction.** Move the framework toward a *catalogue* of typed
items, separated from their location:

- The model submits *items* by category — `pub_type`, `pub_trait`,
  `private_type`, `private_impl`, `private_helper_fn`, `pub_newtype`,
  `pub_constant`, etc. — each with a name, definition, and dep
  references.
- The framework decides where each item *renders* based on its
  category + the node it was submitted from + the items it depends on.
- For shared items the framework can "float" the item to a common
  ancestor (or a dedicated "types" leaf) so it's defined once.
- The model never writes raw `public.rs` or `private.rs`; it always
  works at the catalogue level.

**Concrete tool surface (sketch).**

```
submit_item {
  kind:        "pub_trait" | "pub_struct" | "pub_enum"
              | "pub_newtype_of"  // wraps a private type
              | "pub_constant" | "pub_fn"      // (fn allowed if not in a trait)
              | "private_type" | "private_impl"
              | "private_helper_fn",
  name:        "Codec",
  definition:  "...",            // the Rust source
  refs:        ["Bytes", "Error"], // other items this depends on (by name)
  rationale:   "Why this item",
}
delete_item { name }
move_item { name, target_node }  // explicit override; framework picks by default
```

**Auto-routing rules (sketch).**

- A `pub_*` item used only inside one node stays in that node's
  `public.rs`.
- A `pub_*` item referenced by N nodes (where the deps form an
  unrooted set) floats to the *lowest common ancestor* node's
  `public.rs`, OR to a new `common_types` leaf at the LCA.
- A `private_impl Foo for Bar` lives in the node owning `Bar`'s
  impl (`Bar`'s node, not `Foo`'s).
- An asserted impl: `assert_impl_for(Foo, Bar)` placed in the public
  surface of `Foo`'s node so downstream nodes know `Bar: Foo`
  without depending on `Bar`'s private internals.

**Why this is hard.**

- Cycles: floating items can create dep cycles. Need a careful
  topological re-check after every move.
- Model UX: the model loses the freedom to organize as it sees fit.
  Net win if the framework's organization is *better*; net loss if
  it's *worse*.
- Renaming: when an item moves between nodes, every downstream
  reference's import path changes. The framework has to update
  imports automatically (re-render), but the model's *prompt
  context* needs to reflect the new location.

**Likely first step.** Don't replace `submit_public/private` outright;
*augment* them with a new `submit_item` track. Run both in parallel
for a stage or two, see whether the catalogue path produces cleaner
output. If yes, deprecate the free-form tools.

## Pure-data types vs traits-and-impls in interfaces

**Current state.** `public.rs` is freeform Rust with a validator
forbidding free fns / impl blocks / `mod`. So in practice the model
puts everything in `public.rs`: pub structs, pub enums, pub type
aliases, pub traits, doc comments. There's no machine-readable
distinction between "data that flows through interfaces" and "behavior
that crosses interfaces".

**Why it might matter.** Downstream stages have different needs:

- The iface stage needs to know data shapes (so dependents can
  construct/destructure).
- The tests stage benefits from knowing which methods on which traits
  are the spec's behavioural contract.
- The integrator could check that a `pub trait`'s only impl is
  somewhere it can find (e.g., `assert_impl_for` markers).

Splitting `public.rs` into `public/types.rs` + `public/traits.rs`
(plus their re-exports) would let the framework reason about each
category separately. The model would author into the appropriate
file.

**Likely first step.** Coupled with the catalogue idea above — the
`kind` field on `submit_item` already separates types from traits.

## More extensive QA phases (test-coverage, fuzz/arbitrary)

**Current state.** The per-node stages are spec → iface → tests →
impl → debug → opt. Tests are written before impl; impl makes them
pass; debug fixes regressions; opt is currently a no-op.

**The gap.** Once impl passes the existing tests, we have no signal
about how *much* of the impl those tests actually exercise. Models
write tests that are easy to pass; coverage can be dismal even when
the suite is green.

**A direction.** Add QA-flavored stages BETWEEN impl and debug, each
with its own gate:

1. **`coverage`** — run `cargo tarpaulin` (or similar) on the node.
   Surface uncovered lines/branches. Writer's job: add tests for the
   uncovered paths. Gate: coverage ≥ threshold (configurable).
2. **`fuzz`** — generate property tests with `proptest` /
   `arbitrary`. Writer reads the spec's invariants and writes
   property test bodies. Gate: properties pass.
3. **`debug`** — existing, runs only if any prior gate failed.

This makes the lifecycle: spec → iface → tests → impl → coverage →
fuzz → debug → opt.

Could also be a single "QA" stage that aggregates these into one
prompt-and-cycle, with the model picking what to focus on.

**Why this is hard.** Tarpaulin doesn't work on every platform.
Property tests pose their own model-direction challenge ("write a
property that's true for any conforming impl"). And the failure mode
where the model loosens the property to make it pass is real.

**Likely first step.** Add coverage as a non-blocking stage: run it,
print the report, but don't gate on it. After a few runs we'll know
whether the model can usefully act on the report.

## Auto-recovery from stuck scheduler

**Current state.** When a stage fails terminally (max retries
exhausted), it's marked Failed. If any downstream stage depends on
that node's iface/impl, the scheduler eventually hits "no ready
stages and not done; halting". The operator has to reset the node
manually via the UI.

**Possible enhancements.**

- After a halt, automatically reset the offending node ONCE and
  retry the pipeline. If we halt again at the same place, fail
  for real.
- Differentiate "failed because the model gave up" (auto-retry
  worthwhile) from "failed because the spec is genuinely
  impossible" (don't retry).
- Surface a stuck pipeline as a Paused state instead of an Err so
  the operator can intervene without restarting the process.

**Likely first step.** Add a `--auto-retry-on-halt N` CLI flag that
just re-runs the engine N times after each halt, with a small
backoff. Cheap to implement, hard to get wrong.

## Architecture decisions worth revisiting later

- **`cargo test` as the impl-stage gate.** Slow for big projects.
  Could use `cargo nextest` or run only the new node's tests.
- **Per-node JSON files vs single graph.json.** Currently per-node.
  Works fine for the scenarios we've hit. If conflicts on
  graph.json become frequent (architect-stage adds nodes from
  multiple sources), might need a smarter merge.
- **Rebase conflicts.** Currently abort + abandon. If conflicts on
  render-generated files (Cargo.toml, mod.rs) start showing up, we
  could auto-resolve by re-rendering from the merged graph.
