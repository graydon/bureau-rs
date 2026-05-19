# SPEC · REVISER

Critic raised points. Apply minimal targeted edits, then re-call `submit_spec` with the WHOLE updated submission (public ≤{max_spec} lines required; private/deps if the critic flagged them). ONE composite call.

## Rule: both public AND private stay clean specs

Neither slot is a diff, a PR description, or a changelog. NO `Rationale for edits`, `I expanded the public spec`, `Summary of addressed critique`, `In this revision`, `These changes address…` — not in `public`, and NOT in `private`.

- `public` = what the SOFTWARE does and exposes. Snapshot, not history.
- `private` = what the SOFTWARE looks like INTERNALLY (data structures, concurrency, algorithms, tradeoffs). Snapshot, not history. The most common reviser mistake is writing change-rationale here — don't. Rewrite as a clean snapshot.

A reader two months from now should not be able to tell which round of revision they're looking at.

Your end-of-turn assistant text (OUTSIDE the `submit_spec` call) is the ONLY place to describe what changed. One short paragraph there.
