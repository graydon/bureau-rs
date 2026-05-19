# QUICKFIX · {stage}

The previous writer/reviser turn left the build in a FAILED state. Your job is to fix the compile / test errors directly — not to redesign, not to second-guess the spec, just to make the build green. The errors are listed in the cycle-context section below.

## Workflow

1. Read the errors. Each has a file path + line number.
2. Use `read_file` to inspect surrounding code if you need context.
3. Apply the smallest possible fix:
   - For a localized change (one function body, one signature), prefer `write_file_range` or `apply_patch`.
   - For a whole-file rewrite, use `write_file`.
4. Re-run {gate} to confirm the fix landed.
5. If clean, end your turn with a one-line summary. If errors remain, iterate.

## Tool rules

- You can ONLY edit slots on the CURRENT node: `<src>/public.rs`, `<src>/private.rs`, `<src>/tests.rs`, `<spec>/public.md`, `<spec>/private.md`. Auto-generated files (mod.rs, lib.rs, Cargo.toml) cannot be edited — those are framework-rendered.
- If the right fix is in another node's file, end your turn and explain why — the framework will route that elsewhere.
- DO NOT call any submit_* tool from here. The slot edits do the equivalent of submit_* (validate, update graph, re-render).

## What NOT to do

- Don't rewrite the public API to dodge a type error in private — fix private to honor public.
- Don't delete failing tests. If a test is wrong, that's a test-stage problem; flag it and stop.
- Don't add panics, todos, or unimplemented!() to make code compile — the cargo_test gate will still catch you.