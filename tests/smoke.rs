use bureau_rs::artifact;
use bureau_rs::merge;
use bureau_rs::phase::Phase;

#[test]
fn parse_phase_names() {
    assert_eq!(Phase::parse("spec"), Some(Phase::Spec));
    assert_eq!(Phase::parse("Implementation"), Some(Phase::Impl));
    assert_eq!(Phase::parse("optimization"), Some(Phase::Opt));
    assert_eq!(Phase::parse("nope"), None);
}

#[test]
fn phase_next() {
    assert_eq!(Phase::Spec.next(), Some(Phase::Interface));
    assert_eq!(Phase::Opt.next(), None);
}

#[test]
fn rust_validation_rejects_garbage() {
    let bad = "fn foo( {";
    assert!(artifact::validate_rust(std::path::Path::new("x.rs"), bad).is_err());
    let good = "pub fn foo() { todo!() }";
    assert!(artifact::validate_rust(std::path::Path::new("x.rs"), good).is_ok());
}

#[tokio::test]
async fn gate_test_surfaces_runtime_test_failures() {
    use bureau_rs::gate::{GateKind, run_gate};

    // Build a real, tiny crate in a tempdir with one passing test and one
    // failing test. Run the Test gate. Assert it reports passed=false and
    // includes a non-empty `errors` list — this is the regression that
    // caused phase impl to retry forever with "0 errors".
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path();
    std::fs::write(
        workdir.join("Cargo.toml"),
        "[package]\nname = \"gatetest\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workdir.join("src")).unwrap();
    std::fs::write(
        workdir.join("src").join("lib.rs"),
        r#"pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ok_test() { assert_eq!(add(1, 1), 2); }
    #[test] fn failing_test() { assert_eq!(add(1, 1), 99); }
}
"#,
    )
    .unwrap();

    let outcome = run_gate(workdir, GateKind::Test).await.unwrap();
    assert!(!outcome.passed, "gate should fail when a test fails");
    assert!(
        !outcome.errors.is_empty(),
        "gate must report at least one error when a test fails (was: stdout={:?}, stderr={:?})",
        outcome.stdout,
        outcome.stderr
    );
    let any_failure_msg = outcome
        .errors
        .iter()
        .any(|e| e.message.contains("failing_test") || e.message.contains("failed"));
    assert!(
        any_failure_msg,
        "expected at least one error mentioning the failing test or 'failed'; got: {:#?}",
        outcome.errors
    );
}

#[test]
fn transient_agent_error_classifier() {
    use bureau_rs::agent::is_transient_agent_error;

    // The exact rig message we saw in production.
    assert!(is_transient_agent_error(
        "CompletionError: ResponseError: Response contained no message or tool call (empty)"
    ));
    // Other transient flavours.
    assert!(is_transient_agent_error("connection reset by peer"));
    assert!(is_transient_agent_error("operation timed out"));
    assert!(is_transient_agent_error("HTTP 502 Bad Gateway"));
    assert!(is_transient_agent_error("HTTP 503 Service Unavailable"));
    assert!(is_transient_agent_error("HTTP 429 Too Many Requests"));

    // Real failures that should NOT be retried.
    assert!(!is_transient_agent_error("invalid api key"));
    assert!(!is_transient_agent_error("model not found"));
    assert!(!is_transient_agent_error("malformed JSON in tool result"));
    assert!(!is_transient_agent_error(""));
}

#[test]
fn rust_syntax_errors_include_line_col_and_snippet() {
    // Multi-line source so we can verify the line is reported correctly.
    let bad = "pub fn good() { todo!() }\npub fn bad( ;\npub fn other() { todo!() }\n";
    let err = artifact::validate_rust(std::path::Path::new("x.rs"), bad).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("line 2"), "expected line 2 in error: {msg}");
    // Snippet should include the offending line
    assert!(msg.contains("pub fn bad("), "expected snippet in error: {msg}");
    // And a caret marker
    assert!(msg.contains("^"), "expected caret in error: {msg}");
}

#[test]
fn stub_function_bodies_replaces_real_bodies() {
    let src = "pub fn add(x: i32, y: i32) -> i32 { x + y }\n\
               pub fn already_stub() { todo!() }\n";
    let (out, warnings) = artifact::stub_function_bodies(src).unwrap();
    assert!(out.contains("todo!"));
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("add"));
}

#[test]
fn replace_fn_body_swaps_block() {
    let src = "pub fn add(x: i32, y: i32) -> i32 { todo!() }\n";
    let new = artifact::replace_fn_body(src, "add", "x + y").unwrap();
    assert!(new.contains("x + y"));
    assert!(!new.contains("todo!"));
}

#[test]
fn replace_fn_body_missing_fails() {
    let src = "pub fn add() { todo!() }\n";
    assert!(artifact::replace_fn_body(src, "missing", "0").is_err());
}

#[test]
fn merge_mod_decls_dedups() {
    let a = "mod alpha;\nmod beta;\n";
    let b = "mod beta;\nmod gamma;\n";
    let merged = artifact::merge_mod_declarations(a, b).unwrap();
    assert_eq!(merged.matches("mod beta").count(), 1);
    assert!(merged.contains("mod alpha"));
    assert!(merged.contains("mod gamma"));
}

#[test]
fn merge_cargo_toml_unions_deps() {
    let a = "[package]\nname=\"x\"\nversion=\"0.1.0\"\n[dependencies]\nfoo = \"1\"\n";
    let b = "[package]\nname=\"x\"\nversion=\"0.1.0\"\n[dependencies]\nbar = \"2\"\n";
    let merged = merge::merge_cargo_toml(a, b).unwrap();
    assert!(merged.contains("foo"));
    assert!(merged.contains("bar"));
}

#[test]
fn merge_cargo_toml_conflicts_on_version_mismatch() {
    let a = "[dependencies]\nfoo = \"1\"\n";
    let b = "[dependencies]\nfoo = \"2\"\n";
    assert!(merge::merge_cargo_toml(a, b).is_err());
}

#[test]
fn public_signatures_extract() {
    let src = "pub fn alpha(x: i32) -> i32 { todo!() }\nfn private() {}\npub struct S;\n";
    let sigs = artifact::PublicSignatures::from_source(src).unwrap();
    assert!(sigs.items.iter().any(|s| s.contains("alpha")));
    assert!(sigs.items.iter().any(|s| s.contains("struct S")));
    assert!(!sigs.items.iter().any(|s| s.contains("private")));
}
