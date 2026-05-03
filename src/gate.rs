//! Phase gates: cargo check / cargo test runners that produce structured
//! diagnostics for the Debug phase.

use crate::tools::CompilerError;
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

#[derive(Debug, Clone)]
pub struct GateOutcome {
    pub passed: bool,
    pub errors: Vec<CompilerError>,
    pub stdout: String,
    pub stderr: String,
}

impl GateOutcome {
    pub fn empty_ok() -> Self {
        Self {
            passed: true,
            errors: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GateKind {
    /// Run `cargo check`
    Check,
    /// Run `cargo build`
    Build,
    /// Run `cargo test`
    Test,
    /// Run `cargo test --no-run` (compile tests only)
    TestNoRun,
}

impl GateKind {
    pub fn args(self) -> &'static [&'static str] {
        match self {
            GateKind::Check => &["check", "--message-format=json"],
            GateKind::Build => &["build", "--message-format=json"],
            GateKind::Test => &["test", "--message-format=json", "--no-fail-fast"],
            GateKind::TestNoRun => &["test", "--no-run", "--message-format=json"],
        }
    }
}

/// Run a cargo command in `workdir` and parse JSON output for errors.
pub async fn run_gate(workdir: &Path, kind: GateKind) -> Result<GateOutcome> {
    let mut cmd = Command::new("cargo");
    cmd.args(kind.args())
        .current_dir(workdir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("CARGO_TERM_COLOR", "never");
    let output = cmd.output().await?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    Ok(parse_cargo_output(
        &stdout,
        &stderr,
        output.status.success(),
        kind,
    ))
}

/// Parse cargo's stdout/stderr into a GateOutcome. Pulled out of `run_gate`
/// so the in-task `cargo_check` / `cargo_test` tools can use the same
/// parsing logic without re-implementing it.
pub fn parse_cargo_output(
    stdout: &str,
    stderr: &str,
    status_success: bool,
    kind: GateKind,
) -> GateOutcome {
    let mut errors = Vec::new();
    let mut idx = 0u32;
    for line in stdout.lines() {
        let val: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if val.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(msg) = val.get("message") else { continue };
        let level = msg.get("level").and_then(|l| l.as_str()).unwrap_or("");
        if level != "error" {
            continue;
        }
        let rendered = msg
            .get("rendered")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                msg.get("message")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string()
            });
        let (file, line) = primary_span(msg);
        idx += 1;
        errors.push(CompilerError {
            id: format!("E{idx:04}"),
            file,
            line,
            message: rendered,
            raw: msg.clone(),
        });
    }

    // Surface test failures from cargo test. Sources to check, in order:
    //
    // 1. JSON `event: "failed"` records — these only appear if libtest is
    //    invoked with `--format=json`, which we don't (that needs nightly's
    //    `-Z unstable-options`). Kept for forward compatibility.
    //
    // 2. JSON `reason: "build-finished"` with `success: false` — cargo
    //    emits this when the build fails for non-message reasons (linker
    //    errors, panics in build scripts, etc.).
    //
    // 3. **Plain-text libtest output** of the form `test <name> ... FAILED`
    //    — this is what 99% of test failures produce on stable Rust. We
    //    parse the stdout line-by-line for this pattern; this is the bug
    //    that previously made `cargo test` failures invisible to the gate.
    //
    // 4. `test result: ... <N> failed` summary line — fallback for the
    //    rare case where the `... FAILED` lines were eaten somewhere.
    if matches!(kind, GateKind::Test) {
        for line in stdout.lines() {
            // (1) JSON failed event
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                if val.get("event").and_then(|e| e.as_str()) == Some("failed") {
                    let name = val
                        .get("name")
                        .and_then(|s| s.as_str())
                        .unwrap_or("<test>")
                        .to_string();
                    idx += 1;
                    errors.push(CompilerError {
                        id: format!("T{idx:04}"),
                        file: None,
                        line: None,
                        message: format!("test failed: {name}"),
                        raw: val,
                    });
                    continue;
                }
                // (2) build-finished success:false
                if val.get("reason").and_then(|r| r.as_str()) == Some("build-finished")
                    && val.get("success").and_then(|s| s.as_bool()) == Some(false)
                {
                    idx += 1;
                    errors.push(CompilerError {
                        id: format!("B{idx:04}"),
                        file: None,
                        line: None,
                        message: "build failed (see stderr for details)".to_string(),
                        raw: val,
                    });
                    continue;
                }
            }
            // (3) Plain-text libtest failure marker.
            if let Some(name) = parse_libtest_failed_line(line) {
                idx += 1;
                errors.push(CompilerError {
                    id: format!("T{idx:04}"),
                    file: None,
                    line: None,
                    message: format!("test failed: {name}"),
                    raw: serde_json::json!({"plain": line}),
                });
            }
        }
        // (4) `test result: FAILED. <p> passed; <f> failed; ...`
        // If we still have no recorded failures but the summary says some,
        // synthesize one entry per failure count so the model knows.
        if errors.is_empty() {
            for line in stdout.lines().chain(stderr.lines()) {
                if let Some(n) = parse_failure_count(line) {
                    if n > 0 {
                        idx += 1;
                        errors.push(CompilerError {
                            id: format!("S{idx:04}"),
                            file: None,
                            line: None,
                            message: format!(
                                "{n} test(s) failed (libtest summary line: '{}')",
                                line.trim()
                            ),
                            raw: serde_json::json!({"summary": line}),
                        });
                        break;
                    }
                }
            }
        }
    }

    // Final fallback: cargo exited non-zero and we couldn't extract a
    // structured cause. Synthesize an error with a stderr tail so the
    // gate failure is at least diagnosable rather than mysteriously
    // empty (the bug that caused phase-impl to retry forever with
    // "0 errors").
    if !status_success && errors.is_empty() {
        let tail = tail_of(stderr, 40);
        idx += 1;
        errors.push(CompilerError {
            id: format!("X{idx:04}"),
            file: None,
            line: None,
            message: format!(
                "cargo {} exited non-zero; could not parse a structured failure. \
                 Last stderr lines:\n{}",
                kind.label(),
                tail
            ),
            raw: serde_json::json!({"stderr_tail": tail}),
        });
    }

    let passed = status_success && errors.is_empty();
    GateOutcome {
        passed,
        errors,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
    }
}

impl GateKind {
    pub fn label(self) -> &'static str {
        match self {
            GateKind::Check => "check",
            GateKind::Build => "build",
            GateKind::Test => "test",
            GateKind::TestNoRun => "test --no-run",
        }
    }
}

/// Match a libtest plain-text failure line:
///   `test some::path::to::test_name ... FAILED`
fn parse_libtest_failed_line(line: &str) -> Option<&str> {
    let line = line.trim();
    if !line.starts_with("test ") {
        return None;
    }
    let rest = &line[5..];
    // We need the line to end with "... FAILED".
    let tail = " ... FAILED";
    let idx = rest.rfind(tail)?;
    let name = rest[..idx].trim();
    if name.is_empty() {
        return None;
    }
    Some(name)
}

/// Match libtest summary line of the form:
///   `test result: FAILED. 1 passed; 2 failed; 0 ignored; ...`
/// Returns the number of failed tests, or None if the line doesn't match.
fn parse_failure_count(line: &str) -> Option<usize> {
    let line = line.trim();
    if !line.starts_with("test result:") {
        return None;
    }
    // Look for `<N> failed`.
    let after = line.split(';').find(|seg| seg.trim().ends_with("failed"))?;
    let nstr = after.trim().split_whitespace().next()?;
    nstr.parse().ok()
}

fn tail_of(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_libtest_failed_lines() {
        assert_eq!(
            parse_libtest_failed_line("test it_works ... FAILED"),
            Some("it_works")
        );
        assert_eq!(
            parse_libtest_failed_line("test mod::sub::test_name ... FAILED"),
            Some("mod::sub::test_name")
        );
        assert_eq!(
            parse_libtest_failed_line("    test indented ... FAILED"),
            Some("indented")
        );
        // Negatives
        assert_eq!(parse_libtest_failed_line("test it_works ... ok"), None);
        assert_eq!(parse_libtest_failed_line("running 1 test"), None);
        assert_eq!(parse_libtest_failed_line(""), None);
        assert_eq!(parse_libtest_failed_line("test ..."), None);
    }

    #[test]
    fn parses_libtest_summary_failure_count() {
        assert_eq!(
            parse_failure_count(
                "test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out"
            ),
            Some(2)
        );
        assert_eq!(
            parse_failure_count("test result: ok. 3 passed; 0 failed; 0 ignored;"),
            Some(0)
        );
        assert_eq!(parse_failure_count("running 0 tests"), None);
        assert_eq!(parse_failure_count(""), None);
    }
}

fn primary_span(msg: &serde_json::Value) -> (Option<PathBuf>, Option<u32>) {
    #[derive(Deserialize)]
    struct Span {
        file_name: String,
        line_start: u32,
        is_primary: bool,
    }
    let spans = msg.get("spans").and_then(|s| s.as_array());
    if let Some(spans) = spans {
        for s in spans {
            if let Ok(span) = serde_json::from_value::<Span>(s.clone()) {
                if span.is_primary {
                    return (Some(PathBuf::from(span.file_name)), Some(span.line_start));
                }
            }
        }
        if let Some(s) = spans.first() {
            if let Ok(span) = serde_json::from_value::<Span>(s.clone()) {
                return (Some(PathBuf::from(span.file_name)), Some(span.line_start));
            }
        }
    }
    (None, None)
}
