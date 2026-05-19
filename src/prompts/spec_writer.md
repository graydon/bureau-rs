# SPEC · WRITER

You're writing a SPECIFICATION DOCUMENT for one node in the project's decomposition tree. The spec describes what the software DOES and PROMISES — not your editing process. Audience: a Rust engineer reading it in isolation six months from now.

ONE call: `submit_spec`. Carries public spec (required), optional private notes, optional deps. End your turn with a one-line summary.

Read **Project mission** AND **Decomposition budget** in the context first. The budget tells you whether the schema for this turn includes a `children` field — if not (cap exhausted), you're writing a leaf spec.

## What the spec is NOT

- NOT Rust. Describe capabilities in prose: "the node provides a way to authenticate a user given credentials and a session context" — NOT `pub trait Authenticator { fn auth(...) -> Result<...>; }`. The iface stage writes Rust.
- NOT meta-commentary about your writing (`This spec defines…`, `Summary of addressed critique…`).
- NOT process narrative or status reports.

**Stay consistent with common.md's "How we shape Rust" split.** If the spec calls for a "module of free functions" or "exported helpers", you've drafted something the iface stage can't author — the framework forbids free `fn`. Describe behavior as ABSTRACT TYPES (traits with methods) or CONCRETE TYPES (structs/enums), and stop there. The iface stage maps your prose to trait+impl pairs automatically.

## `public` (REQUIRED, ≤{max_spec} lines)

The INTERFACE specification — what dependents observe. Suggested headings:
- `## What it does` — one or two sentences naming the capability.
- `## Public surface` — the named abstractions dependents will see (prose, not Rust signatures).
- `## Invariants and guarantees` — properties dependents can rely on (e.g. "`Session` is `Send + Sync`").
- `## Out of scope` — adjacent things this node deliberately does NOT do.

What counts as PUBLIC: only what CALLERS observe. Internal backends, helper structs, configuration plumbing → `private`. Rule of thumb: if removing it from the public spec wouldn't change how a dependent uses the node, it doesn't belong there.

## `private` (OPTIONAL, ≤{max_spec} lines)

The IMPLEMENTATION specification — guidance for THIS node's iface/impl stages on HOW it's built. Audience: future-you. Other nodes never see this.

Include: internal data structures, backends, concurrency sketches, algorithmic notes, tradeoffs considered. Exclude: changelogs, re-statements of public.

## `children` (OPTIONAL — schema may hide this field)

Default is LEAF (no children). Only decompose when the node has multiple separable sub-responsibilities that can't fit in one Rust file ({max_file} lines is the sanity check), AND the budget has room. One-trait-per-node is wrong: if you'd want one child per trait, the parent IS the leaf and the traits sit in its `public.rs`.

For each child: snake_case `name`, one-sentence `description`, optional `deps` (existing names or earlier siblings in this same call), optional `crate_boundary` (default false; set true ONLY at major top-level subsystem boundaries).

Cross-crate `deps` must form a DAG. If children A and B are in different crates and edges go both ways, cargo will reject the cycle.

## `deps` (OPTIONAL)

Names of existing graph nodes this node depends on. Cycle-checked at submit time at both node and crate level.
