# Style guide

User-supplied tone / verbosity / convention guidance. The framework
inlines this file (if present) into every prompt context as a "Style
guide" section. The default writing-style instructions in the system
preamble defer to this file when they conflict.

## Writing style

- **Terse.** Specs and prose are matter-of-fact. No marketing language,
  no "in this section we will…" preambles, no padding.
- **Concrete nouns over abstractions.** "the smbd binary" beats "the
  service offering". "the SMB2 frame parser" beats "the protocol
  message handling subsystem".
- **No just-in-case caveats.** Don't enumerate edge cases the spec
  doesn't address. If something's out of scope, say so once, plainly,
  in the Out-of-scope section.
- **Short sentences.** Long sentences with clauses-within-clauses are
  the surest sign you're padding.
- **Skip jargon you wouldn't write on a whiteboard.** "leverage",
  "robustly", "comprehensive", "ensure that", "it is important to
  note" — all gone.
- **One paragraph per `##` heading is usually enough.**

## Code style

- **Standard rustfmt.** No bespoke formatting opinions; let the
  formatter decide.
- **No doc comments on private items unless the WHY is non-obvious.**
  Names should carry the obvious meaning.
- **Errors via `thiserror` for libraries, `anyhow` only at the binary
  boundary.** Don't propagate `anyhow::Error` through library APIs.
- **`Result<T>` with a crate-local alias.** Pick `pub type Result<T> =
  std::result::Result<T, Error>` per crate; don't fully-qualify
  `std::result::Result` in every signature.
- **Avoid `unwrap()` outside tests / deliberate "this can't fail"
  spots. `expect("...")` with a real message if the panic is genuinely
  load-bearing.**
- **No `mod foo;` in `public.rs` files.** The framework auto-generates
  module scaffolding.

## What I particularly don't want

- "World-class", "industry-standard", "robust", "comprehensive",
  "leverage", "facilitate", "in order to" (just say "to").
- Comments restating what the code does. Comments are for WHY only.
- Three-paragraph rationale sections in private specs. One paragraph
  is plenty.
- Emoji in code, comments, or specs. Not unless I explicitly ask.
