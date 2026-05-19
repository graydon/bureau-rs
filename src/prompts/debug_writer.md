# DEBUG · WRITER

Tests are still failing after the previous stage. Look at the failing-test output (in the `Critique cycle context` section below, or run `cargo_test` yourself). Apply MINIMAL targeted fixes via `submit_private` (≤ {max_file} lines) and, only if a test was actually wrong, `submit_tests`. Don't redesign. The public surface is still frozen.