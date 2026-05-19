//! Phase gates: cargo check / cargo test runners that produce structured
//! diagnostics for the Debug phase.

use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

/// One structured diagnostic produced by `cargo check`/`cargo test` — either
/// a compile error, a runtime test failure, or a synthetic "exit non-zero"
/// catch-all when neither matched. Surfaced both to the model (via the
/// cargo_* tools) and to the orchestrator (via gate failures).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompilerError {
    pub id: String,
    pub file: Option<PathBuf>,
    pub line: Option<u32>,
    pub message: String,
    pub raw: serde_json::Value,
}

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
        // `--workspace` is the critical flag: without it cargo only
        // compiles the root package, which means a workspace's member
        // crates (under `crates/<name>/`) are never type-checked.
        // Previously a member crate could have unresolved imports,
        // type mismatches, missing deps — anything — and the gate
        // would still report `passed: true` because the root compiled
        // fine.
        //
        // `--all-targets` extends `check`/`build` to also compile
        // tests, examples and benches: an iface-stage `cargo check`
        // that misses test-file compile errors only catches half of
        // what the model could have broken.
        match self {
            GateKind::Check => &[
                "check",
                "--workspace",
                "--all-targets",
                "--message-format=json",
            ],
            GateKind::Build => &[
                "build",
                "--workspace",
                "--all-targets",
                "--message-format=json",
            ],
            GateKind::Test => &[
                "test",
                "--workspace",
                "--message-format=json",
                "--no-fail-fast",
            ],
            GateKind::TestNoRun => &[
                "test",
                "--workspace",
                "--no-run",
                "--message-format=json",
            ],
        }
    }
}

/// Run a cargo command in `workdir` and parse JSON output for errors.
///
/// When cargo exits non-zero, the full stdout + stderr + exit code are
/// always dumped to `<workdir>/.bureau/last-gate-failure.log` for
/// post-hoc inspection — the X0001 fallback path can only fit a tail
/// in the per-call diagnostic, but the operator needs the whole story
/// when "the framework can't find the error" with empty-looking output.
pub async fn run_gate(workdir: &Path, kind: GateKind) -> Result<GateOutcome> {
    let mut cmd = Command::new("cargo");
    cmd.args(kind.args())
        .current_dir(workdir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("CARGO_TERM_COLOR", "never");
    let cmdline = format!("cargo {}", kind.args().join(" "));
    let output = cmd.output().await?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code();
    if !output.status.success() {
        // Always preserve a full dump on failure. The X0001 fallback
        // only carries a short tail; without the full log we can't
        // diagnose pathological cases (empty stderr, build-script
        // panics that print to stdout, etc.).
        let dump = format!(
            "command: {cmdline}\n\
             workdir: {workdir_disp}\n\
             exit code: {exit}\n\
             ---- stdout ({stdout_bytes} bytes) ----\n{stdout}\n\
             ---- stderr ({stderr_bytes} bytes) ----\n{stderr}\n",
            workdir_disp = workdir.display(),
            exit = exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "<signal>".to_string()),
            stdout_bytes = stdout.len(),
            stderr_bytes = stderr.len(),
        );
        write_gate_failure_log(workdir, &dump);
        // Also emit a high-signal warn so the operator sees a real
        // message in the binary's log rather than just a UI X0001.
        tracing::warn!(
            cmd = %cmdline,
            exit = ?exit_code,
            stdout_bytes = stdout.len(),
            stderr_bytes = stderr.len(),
            "cargo gate failed (full output dumped to .bureau/last-gate-failure.log)"
        );
        if !stderr.trim().is_empty() {
            tracing::warn!(stderr = %stderr.trim(), "cargo gate stderr");
        }
    }
    let mut outcome = parse_cargo_output(&stdout, &stderr, output.status.success(), kind);
    // Stash the exit code in the X0001 fallback if it fired so the
    // model can see what was actually different (signal vs exit 101,
    // etc.). We do this after parse so we don't change parse_cargo_output's
    // signature (still callable from tools.rs without exit info).
    if let Some(last) = outcome.errors.last_mut() {
        if last.id.starts_with('X') {
            if let Some(code) = exit_code {
                last.message = format!("{} (exit code {code})", last.message);
            } else {
                last.message = format!("{} (killed by signal)", last.message);
            }
        }
    }
    Ok(outcome)
}

/// Write a full diagnostic dump of a cargo failure to a stable path
/// under `.bureau/`. Single rolling file — operators expect to find
/// "the most recent failure" without browsing a directory.
fn write_gate_failure_log(workdir: &Path, content: &str) {
    let dir = workdir.join(".bureau");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(?e, "couldn't create .bureau for gate-failure log");
        return;
    }
    let path = dir.join("last-gate-failure.log");
    if let Err(e) = std::fs::write(&path, content) {
        tracing::warn!(path = %path.display(), ?e, "couldn't write gate-failure log");
    }
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
    // structured cause. Surface BOTH stderr and stdout tails — some
    // cargo workflow errors (`error: failed to parse manifest`, build
    // script panics) land on stdout because `--message-format=json`
    // muxes through stdout. Also surface any non-message JSON records
    // we saw (e.g. `build-finished` with `success: false`, or a
    // `compiler-message` at level=`warning` that flipped the build
    // anyway) — those carry the diagnostic the model needs.
    if !status_success && errors.is_empty() {
        let stderr_tail = tail_of(stderr, 60);
        let stdout_tail = tail_of(stdout, 60);
        // Pull out any rendered diagnostic strings we ignored above
        // (because they weren't level=error). Frequently a level=
        // "error: aborting due to previous errors" is here that
        // explains the failure even when stderr is empty.
        let mut ignored_diagnostics = Vec::new();
        for line in stdout.lines() {
            let val: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(msg) = val.get("message") {
                if let Some(rendered) =
                    msg.get("rendered").and_then(|s| s.as_str())
                {
                    if !rendered.is_empty() {
                        ignored_diagnostics.push(rendered.to_string());
                    }
                }
            }
        }
        let json_diag_block = if ignored_diagnostics.is_empty() {
            String::new()
        } else {
            format!(
                "JSON diagnostics (not classified as error level):\n{}\n",
                ignored_diagnostics.join("\n---\n")
            )
        };
        idx += 1;
        errors.push(CompilerError {
            id: format!("X{idx:04}"),
            file: None,
            line: None,
            message: format!(
                "cargo {} exited non-zero; could not extract a structured error \
                 message. Full output dumped to .bureau/last-gate-failure.log.\n\n\
                 {json_diag_block}\
                 ---- last stderr lines ----\n{stderr_tail}\n\
                 ---- last stdout lines ----\n{stdout_tail}",
                kind.label(),
            ),
            raw: serde_json::json!({
                "stderr_tail": stderr_tail,
                "stdout_tail": stdout_tail,
                "ignored_diagnostics": ignored_diagnostics,
            }),
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
