//! Path classification for the orchestrator. The pipeline produces both
//! single-crate projects (`Cargo.toml`, `src/`, `tests/` at the workdir root)
//! and Cargo workspaces with member crates living under `crates/<name>/` (or
//! anywhere else the model chooses). Phase guards need to recognize both
//! layouts uniformly, so we classify by pattern rather than by literal path
//! prefix.
//!
//! Conventions:
//! - **spec/**: workspace-wide, always at the workdir root.
//! - **Cargo.toml**: at the workdir root (single crate or workspace) AND
//!   optionally under each member crate.
//! - **Rust source**: anything under a `src/` segment that is NOT a test file.
//! - **Rust test**: anything under a `tests/` segment, OR a file inside `src/`
//!   that follows Rust internal-test naming conventions
//!   (`tests.rs`, `test.rs`, `*_tests.rs`, `*_test.rs`, or anything inside a
//!   `tests/`/`test/` subdirectory of `src/`).

use std::path::{Component, Path};

/// Classify what kind of artifact a path represents. Determined purely from
/// the path; never inspects file contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    Spec,
    CargoToml,
    RustSource,
    RustTest,
    Other,
}

pub fn classify(rel: &Path) -> PathKind {
    if is_spec(rel) {
        PathKind::Spec
    } else if is_cargo_toml(rel) {
        PathKind::CargoToml
    } else if is_rust_test(rel) {
        PathKind::RustTest
    } else if is_rust_source(rel) {
        PathKind::RustSource
    } else {
        PathKind::Other
    }
}

pub fn extension(rel: &Path) -> Option<&str> {
    rel.extension().and_then(|s| s.to_str())
}

pub fn is_rust_file(rel: &Path) -> bool {
    extension(rel) == Some("rs")
}

pub fn is_markdown(rel: &Path) -> bool {
    extension(rel) == Some("md")
}

pub fn is_spec(rel: &Path) -> bool {
    if !is_markdown(rel) {
        return false;
    }
    matches_first_segment(rel, "spec")
}

pub fn is_cargo_toml(rel: &Path) -> bool {
    rel.file_name().and_then(|s| s.to_str()) == Some("Cargo.toml")
}

/// True if `rel` lives under any `src/` directory in its path. Works for
/// both single-crate (`src/...`) and workspace (`crates/foo/src/...`) layouts.
pub fn is_in_src(rel: &Path) -> bool {
    rel.components()
        .any(|c| matches!(c, Component::Normal(s) if s == "src"))
}

/// Heuristic: true if `rel` looks like Rust test code by Rust conventions.
///
/// Recognized patterns:
/// - any path containing a `tests/` segment (covers both `tests/foo.rs`
///   integration tests and `src/foo/tests/inner.rs` modules);
/// - any path with a `test/` or `tests/` segment immediately under `src/`
///   (`src/test/...`, `src/foo/test/...`);
/// - any `.rs` file whose stem is `tests` or `test`;
/// - any `.rs` file whose stem ends in `_tests` or `_test`;
/// - `tests.rs` / `test.rs` modules inside a crate.
pub fn is_rust_test(rel: &Path) -> bool {
    if !is_rust_file(rel) {
        return false;
    }
    // Any `tests/` segment anywhere is a test file. (Note: a directory
    // literally named `tests` under `src/` is the "tests submodule" pattern;
    // a top-level `tests/` directory is integration tests. Both count.)
    if rel
        .components()
        .any(|c| matches!(c, Component::Normal(s) if s == "tests"))
    {
        return true;
    }
    // `src/test/...` and nested `*/test/...` test directories.
    if rel
        .components()
        .any(|c| matches!(c, Component::Normal(s) if s == "test"))
    {
        return true;
    }
    let stem = rel.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem == "tests" || stem == "test" {
        return true;
    }
    if stem.ends_with("_tests") || stem.ends_with("_test") {
        return true;
    }
    false
}

/// True if `rel` is a Rust source file that is NOT a test file. This is the
/// "interface / implementation" surface.
pub fn is_rust_source(rel: &Path) -> bool {
    is_rust_file(rel) && is_in_src(rel) && !is_rust_test(rel)
}

/// Whether the first non-prefix segment of `rel` equals `name`. Useful for
/// `is_spec` (must start with `spec/`).
fn matches_first_segment(rel: &Path, name: &str) -> bool {
    rel.components()
        .find_map(|c| match c {
            Component::Normal(s) => Some(s),
            _ => None,
        })
        .map(|s| s == name)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn classifies_single_crate_paths() {
        assert_eq!(classify(&p("Cargo.toml")), PathKind::CargoToml);
        assert_eq!(classify(&p("src/lib.rs")), PathKind::RustSource);
        assert_eq!(classify(&p("src/foo.rs")), PathKind::RustSource);
        assert_eq!(classify(&p("src/sub/mod.rs")), PathKind::RustSource);
        assert_eq!(classify(&p("tests/integration.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("spec/types.md")), PathKind::Spec);
        assert_eq!(classify(&p("README.md")), PathKind::Other);
    }

    #[test]
    fn classifies_internal_tests_under_src() {
        assert_eq!(classify(&p("src/tests.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/foo/tests.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/foo/test.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/test/foo.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/foo/tests/bar.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/foo/test/bar.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/foo_tests.rs")), PathKind::RustTest);
        assert_eq!(classify(&p("src/foo_test.rs")), PathKind::RustTest);
        // sanity check: ordinary src files not mistaken for tests
        assert_eq!(classify(&p("src/testing.rs")), PathKind::RustSource);
        assert_eq!(classify(&p("src/contestant.rs")), PathKind::RustSource);
    }

    #[test]
    fn classifies_workspace_paths() {
        assert_eq!(
            classify(&p("crates/foo/Cargo.toml")),
            PathKind::CargoToml
        );
        assert_eq!(
            classify(&p("crates/foo/src/lib.rs")),
            PathKind::RustSource
        );
        assert_eq!(
            classify(&p("crates/foo/src/sub/mod.rs")),
            PathKind::RustSource
        );
        assert_eq!(
            classify(&p("crates/foo/tests/integration.rs")),
            PathKind::RustTest
        );
        assert_eq!(
            classify(&p("crates/foo/src/sub/tests.rs")),
            PathKind::RustTest
        );
        assert_eq!(
            classify(&p("crates/foo/src/sub/tests/inner.rs")),
            PathKind::RustTest
        );
        // alternate layout: members at root rather than under crates/
        assert_eq!(classify(&p("foo/Cargo.toml")), PathKind::CargoToml);
        assert_eq!(classify(&p("foo/src/lib.rs")), PathKind::RustSource);
        assert_eq!(classify(&p("foo/tests/it.rs")), PathKind::RustTest);
    }

    #[test]
    fn spec_only_under_spec_dir() {
        assert_eq!(classify(&p("spec/types.md")), PathKind::Spec);
        assert_eq!(classify(&p("spec/sub/types.md")), PathKind::Spec);
        // a top-level markdown file isn't a spec
        assert_eq!(classify(&p("README.md")), PathKind::Other);
        // `crates/foo/spec/...` isn't the workspace-level spec dir
        assert_eq!(classify(&p("crates/foo/spec/x.md")), PathKind::Other);
    }

    #[test]
    fn non_rs_files_under_src_are_not_source() {
        assert_eq!(classify(&p("src/foo.toml")), PathKind::Other);
        assert_eq!(classify(&p("src/foo.txt")), PathKind::Other);
    }
}
