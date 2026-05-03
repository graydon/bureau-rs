//! Comprehensive tests of the tool surface. Each test sets up a controlled
//! TaskCtx in a tempdir and exercises one tool's success and failure paths.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bureau_rs::phase::Phase;
use bureau_rs::state::{OrchestratorState, StateHandle};
use bureau_rs::task::{Role, SubtaskDecl, Task, TranscriptKind};
use bureau_rs::tools::{
    self, CargoArgs, CargoCheckTool, CargoTestArgs, CargoTestTool, CompilerError,
    EmitSubtasksArgs, EmitSubtasksTool, ListCompilerErrArgs, ListCompilerErrorsTool,
    ListFilesArgs, ListFilesTool, ReadCompilerErrArgs, ReadCompilerErrorTool, ReadFileArgs,
    ReadFileTool, ReplaceFnBodyArgs, ReplaceFnBodyTool, SubmitVerdictArgs, SubmitVerdictTool,
    TaskCtx, ToolFailure, WriteFileArgs, WriteFileTool,
};
use parking_lot::Mutex;
use rig::tool::Tool;
use uuid::Uuid;

struct Fixture {
    _tmp: tempfile::TempDir,
    pub workdir: PathBuf,
    pub task_id: Uuid,
    pub state: StateHandle,
}

impl Fixture {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().to_path_buf();
        let mut state = OrchestratorState::new(workdir.clone(), workdir.clone());
        let mut t = Task::new(Phase::Spec, "test root");
        t.write_files = vec![PathBuf::from("dummy")];
        let task_id = state.graph.insert_root(t);
        let handle = StateHandle::new(state);
        Self {
            _tmp: tmp,
            workdir,
            task_id,
            state: handle,
        }
    }

    fn ctx(
        &self,
        phase: Phase,
        role: Role,
        read_set: Vec<&str>,
        write_set: Vec<&str>,
    ) -> Arc<TaskCtx> {
        Arc::new(TaskCtx {
            task_id: self.task_id,
            phase,
            role,
            workdir: self.workdir.clone(),
            read_set: read_set.into_iter().map(PathBuf::from).collect(),
            write_set: write_set.into_iter().map(PathBuf::from).collect(),
            written: Arc::new(Mutex::new(HashSet::new())),
            emitted_subtasks: Arc::new(Mutex::new(Vec::new())),
            verdict: Arc::new(Mutex::new(None)),
            compiler_errors: Vec::new(),
            max_file_lines: 200,
            max_spec_section_lines: 400,
            depth: 0,
            max_subtask_depth: 2,
            recent_calls: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            state: self.state.clone(),
        })
    }

    fn ctx_with_errors(
        &self,
        phase: Phase,
        role: Role,
        write_set: Vec<&str>,
        compiler_errors: Vec<CompilerError>,
    ) -> Arc<TaskCtx> {
        Arc::new(TaskCtx {
            task_id: self.task_id,
            phase,
            role,
            workdir: self.workdir.clone(),
            read_set: HashSet::new(),
            write_set: write_set.into_iter().map(PathBuf::from).collect(),
            written: Arc::new(Mutex::new(HashSet::new())),
            emitted_subtasks: Arc::new(Mutex::new(Vec::new())),
            verdict: Arc::new(Mutex::new(None)),
            compiler_errors,
            max_file_lines: 200,
            max_spec_section_lines: 400,
            depth: 0,
            max_subtask_depth: 2,
            recent_calls: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            state: self.state.clone(),
        })
    }

    fn read(&self, rel: &str) -> Option<String> {
        std::fs::read_to_string(self.workdir.join(rel)).ok()
    }

    fn write(&self, rel: &str, content: &str) {
        let p = self.workdir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, content).unwrap();
    }
}

// ============================================================================
// write_file
// ============================================================================

#[tokio::test]
async fn write_file_succeeds_in_write_set() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Interface, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn foo() { todo!() }\n".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
    let body = f.read("src/lib.rs").expect("file should exist");
    assert!(body.contains("foo"));
}

#[tokio::test]
async fn write_file_outside_write_set_fails() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Interface, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/main.rs".into(),
            content: "fn main() {}".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::WriteNotAllowed { .. })));
    assert!(f.read("src/main.rs").is_none());
}

#[tokio::test]
async fn write_file_with_empty_write_set_fails() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Interface, Role::Actor, vec![], vec![]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn foo() { todo!() }".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::WriteNotAllowed { .. })));
}

#[tokio::test]
async fn write_file_rejects_invalid_rust() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "fn foo( {".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::Syntax(_))));
}

#[tokio::test]
async fn write_file_interface_phase_stubs_bodies() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Interface, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn add(x: i32, y: i32) -> i32 { x + y }".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
    let body = f.read("src/lib.rs").unwrap();
    assert!(body.contains("todo!"), "interface phase should stub bodies; got:\n{body}");
    assert!(!body.contains("x + y"));
}

#[tokio::test]
async fn write_file_spec_path_forbidden_in_interface_phase() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Interface, Role::Actor, vec![], vec!["spec/wrong.md"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "spec/wrong.md".into(),
            content: "x".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::PhaseForbidden(_, _))));
}

#[tokio::test]
async fn write_file_interface_phase_rejects_tests_path() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(
            Phase::Interface,
            Role::Actor,
            vec![],
            vec!["tests/integration.rs"],
        ),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "tests/integration.rs".into(),
            content: "fn main() {}".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::PhaseForbidden(_, _))));
}

#[tokio::test]
async fn emit_subtasks_interface_phase_rejects_tests_writes() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Interface, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "sketch test signatures".into(),
                read_files: vec![],
                write_files: vec![PathBuf::from("tests/integration.rs")],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_err(), "interface subtask declaring tests/ should be rejected");
}

#[tokio::test]
async fn write_file_test_phase_allows_internal_tests_under_src() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(
            Phase::Test,
            Role::Actor,
            vec![],
            vec!["src/foo/tests.rs"],
        ),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/foo/tests.rs".into(),
            content: "#[test] fn t() {}".into(),
        })
        .await;
    assert!(r.is_ok(), "test phase should allow src/foo/tests.rs; got {:?}", r);
}

#[tokio::test]
async fn write_file_test_phase_allows_internal_test_dir() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(
            Phase::Test,
            Role::Actor,
            vec![],
            vec!["src/test/inner.rs"],
        ),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/test/inner.rs".into(),
            content: "#[test] fn t() {}".into(),
        })
        .await;
    assert!(r.is_ok(), "test phase should allow src/test/inner.rs; got {:?}", r);
}

#[tokio::test]
async fn write_file_workspace_paths_classified_correctly() {
    let f = Fixture::new();
    // Interface phase: writes to crates/<name>/src/lib.rs
    let tool = WriteFileTool {
        ctx: f.ctx(
            Phase::Interface,
            Role::Actor,
            vec![],
            vec!["crates/foo/src/lib.rs"],
        ),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "crates/foo/src/lib.rs".into(),
            content: "pub fn x() { todo!() }".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r);

    // Member-crate Cargo.toml in interface phase
    let tool2 = WriteFileTool {
        ctx: f.ctx(
            Phase::Interface,
            Role::Actor,
            vec![],
            vec!["crates/foo/Cargo.toml"],
        ),
    };
    let r2 = tool2
        .call(WriteFileArgs {
            path: "crates/foo/Cargo.toml".into(),
            content: "[package]\nname = \"foo\"\nversion = \"0.1.0\"\n".into(),
        })
        .await;
    assert!(r2.is_ok(), "{:?}", r2);
}

#[tokio::test]
async fn write_file_test_phase_workspace_internal_tests() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(
            Phase::Test,
            Role::Actor,
            vec![],
            vec!["crates/foo/src/sub/tests.rs"],
        ),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "crates/foo/src/sub/tests.rs".into(),
            content: "#[test] fn t() {}".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r);
}

#[tokio::test]
async fn write_file_test_phase_only_writes_tests() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Test, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() {}".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::PhaseForbidden(_, _))));
}

#[tokio::test]
async fn write_file_too_many_lines_rejected() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/big.rs"]),
    };
    let big = "fn x() {}\n".repeat(500);
    let r = tool
        .call(WriteFileArgs {
            path: "src/big.rs".into(),
            content: big,
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::FileTooLarge(_, _))));
}

#[tokio::test]
async fn write_file_records_in_written_set() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let tool = WriteFileTool { ctx: ctx.clone() };
    tool.call(WriteFileArgs {
        path: "src/lib.rs".into(),
        content: "pub fn x() {}".into(),
    })
    .await
    .unwrap();
    let w = ctx.written.lock();
    assert!(w.contains(&PathBuf::from("src/lib.rs")));
}

#[tokio::test]
async fn write_file_idempotent_no_change() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let tool = WriteFileTool { ctx: ctx.clone() };
    let body = "pub fn x() {}\n";
    let r1 = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: body.into(),
        })
        .await
        .unwrap();
    assert!(!r1.no_change);
    let r2 = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: body.into(),
        })
        .await
        .unwrap();
    assert!(r2.no_change, "second identical write should be no-op");
}

#[tokio::test]
async fn write_file_loop_detection_breaks_after_threshold() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let tool = WriteFileTool { ctx };
    let args = || WriteFileArgs {
        path: "src/lib.rs".into(),
        content: "pub fn x() {}\n".into(),
    };
    // First two are allowed, third hits the threshold.
    assert!(tool.call(args()).await.is_ok());
    assert!(tool.call(args()).await.is_ok());
    let r = tool.call(args()).await;
    match r {
        Err(ToolFailure::Loop { tool, count }) => {
            assert_eq!(tool, "write_file");
            assert!(count >= 3);
        }
        other => panic!("expected Loop error, got: {:?}", other),
    }
}

#[tokio::test]
async fn loop_detection_resets_on_different_args() {
    let f = Fixture::new();
    let ctx = f.ctx(
        Phase::Impl,
        Role::Actor,
        vec![],
        vec!["src/a.rs", "src/b.rs"],
    );
    let tool = WriteFileTool { ctx };
    // Two writes to a.rs, then b.rs, then back to a.rs — the run of dupes
    // should be broken by the b.rs write.
    assert!(tool.call(WriteFileArgs {
        path: "src/a.rs".into(),
        content: "pub fn a() {}".into(),
    }).await.is_ok());
    assert!(tool.call(WriteFileArgs {
        path: "src/a.rs".into(),
        content: "pub fn a() {}".into(),
    }).await.is_ok());
    assert!(tool.call(WriteFileArgs {
        path: "src/b.rs".into(),
        content: "pub fn b() {}".into(),
    }).await.is_ok());
    assert!(tool.call(WriteFileArgs {
        path: "src/a.rs".into(),
        content: "pub fn a() {}".into(),
    }).await.is_ok(), "should be allowed; consecutive run was broken");
}

#[tokio::test]
async fn replace_fn_body_idempotent_no_change() {
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn x() -> i32 { 42 }\n");
    let ctx = f.ctx(Phase::Debug, Role::Actor, vec![], vec!["src/lib.rs"]);
    let tool = ReplaceFnBodyTool { ctx };
    // First replacement: changes body
    let r1 = tool
        .call(ReplaceFnBodyArgs {
            path: "src/lib.rs".into(),
            fn_name: "x".into(),
            new_body: "0".into(),
        })
        .await
        .unwrap();
    assert!(!r1.no_change);
    // Second replacement to the same body: no_change
    let r2 = tool
        .call(ReplaceFnBodyArgs {
            path: "src/lib.rs".into(),
            fn_name: "x".into(),
            new_body: "0".into(),
        })
        .await
        .unwrap();
    assert!(r2.no_change);
}

// ============================================================================
// Cross-tool / cross-role visibility
// ============================================================================

#[tokio::test]
async fn list_files_sees_just_written_file_within_role() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let writer = WriteFileTool { ctx: ctx.clone() };
    writer
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() {}\n".into(),
        })
        .await
        .unwrap();
    let lister = ListFilesTool { ctx };
    let r = lister.call(ListFilesArgs { dir: None }).await.unwrap();
    assert!(
        r.files.iter().any(|s| s == "src/lib.rs"),
        "list_files (no dir) should see just-written file; got {:?}",
        r.files
    );
}

#[tokio::test]
async fn list_files_works_when_workdir_path_contains_dotbureau() {
    // Regression: in real runs the workdir is `<workspace>/.bureau/worktrees/<uuid>`,
    // so every absolute path under it contains `.bureau` as a component. The filter
    // must check the path RELATIVE to the workdir, not the absolute path.
    let outer = tempfile::tempdir().unwrap();
    let workdir = outer.path().join(".bureau").join("worktrees").join("abc");
    std::fs::create_dir_all(&workdir).unwrap();

    let mut state = OrchestratorState::new(workdir.clone(), workdir.clone());
    let mut t = Task::new(Phase::Impl, "test root");
    t.write_files = vec![PathBuf::from("src/lib.rs")];
    let task_id = state.graph.insert_root(t);
    let handle = StateHandle::new(state);

    let ctx = Arc::new(TaskCtx {
        task_id,
        phase: Phase::Impl,
        role: Role::Actor,
        workdir: workdir.clone(),
        read_set: HashSet::new(),
        write_set: [PathBuf::from("src/lib.rs")].into_iter().collect(),
        written: Arc::new(Mutex::new(HashSet::new())),
        emitted_subtasks: Arc::new(Mutex::new(Vec::new())),
        verdict: Arc::new(Mutex::new(None)),
        compiler_errors: Vec::new(),
        max_file_lines: 200,
        max_spec_section_lines: 400,
        depth: 0,
        max_subtask_depth: 2,
        recent_calls: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        state: handle,
    });

    WriteFileTool { ctx: ctx.clone() }
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() {}\n".into(),
        })
        .await
        .unwrap();

    // Both forms must work — list with no dir and list with subdir.
    let r1 = ListFilesTool { ctx: ctx.clone() }
        .call(ListFilesArgs { dir: None })
        .await
        .unwrap();
    assert!(
        r1.files.iter().any(|s| s == "src/lib.rs"),
        "list_files (no dir) under .bureau/-nested workdir returned: {:?}",
        r1.files
    );

    let r2 = ListFilesTool { ctx }
        .call(ListFilesArgs {
            dir: Some("src".into()),
        })
        .await
        .unwrap();
    assert!(
        r2.files.iter().any(|s| s == "src/lib.rs"),
        "list_files dir=src under .bureau/-nested workdir returned: {:?}",
        r2.files
    );
}

#[tokio::test]
async fn list_files_with_subdir_sees_just_written_file() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let writer = WriteFileTool { ctx: ctx.clone() };
    writer
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() {}\n".into(),
        })
        .await
        .unwrap();
    let lister = ListFilesTool { ctx };
    let r = lister
        .call(ListFilesArgs {
            dir: Some("src".into()),
        })
        .await
        .unwrap();
    assert!(r.files.iter().any(|s| s == "src/lib.rs"));
}

#[tokio::test]
async fn cross_role_visibility_actor_to_critic() {
    // Critic uses a fresh TaskCtx pointing at the same workdir; it should
    // immediately see what the actor wrote.
    let f = Fixture::new();
    let actor_ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let writer = WriteFileTool { ctx: actor_ctx };
    writer
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() -> i32 { 42 }\n".into(),
        })
        .await
        .unwrap();

    let critic_ctx = f.ctx(Phase::Impl, Role::Critic, vec![], vec![]);
    let lister = ListFilesTool { ctx: critic_ctx.clone() };
    let r = lister.call(ListFilesArgs { dir: None }).await.unwrap();
    assert!(
        r.files.iter().any(|s| s == "src/lib.rs"),
        "critic should see actor's written file in list_files; got {:?}",
        r.files
    );

    let reader = ReadFileTool { ctx: critic_ctx };
    let r2 = reader
        .call(ReadFileArgs {
            path: "src/lib.rs".into(),
        })
        .await
        .unwrap();
    assert!(r2.content.contains("pub fn x"));
    assert!(r2.content.contains("42"));
}

#[tokio::test]
async fn cross_role_visibility_actor_writes_then_reviser_overwrites() {
    let f = Fixture::new();
    let actor_ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    WriteFileTool { ctx: actor_ctx }
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() -> i32 { 0 }\n".into(),
        })
        .await
        .unwrap();
    // Reviser, fresh ctx, same workdir, same write_set
    let reviser_ctx = f.ctx(Phase::Impl, Role::Reviser, vec![], vec!["src/lib.rs"]);
    let r = ReadFileTool { ctx: reviser_ctx.clone() }
        .call(ReadFileArgs {
            path: "src/lib.rs".into(),
        })
        .await
        .unwrap();
    assert!(r.content.contains("0"));
    WriteFileTool { ctx: reviser_ctx }
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() -> i32 { 1 }\n".into(),
        })
        .await
        .unwrap();
    let final_content = std::fs::read_to_string(f.workdir.join("src/lib.rs")).unwrap();
    assert!(final_content.contains("1"));
    assert!(!final_content.contains("0"));
}

#[tokio::test]
async fn cross_role_visibility_judge_sees_revised_state() {
    let f = Fixture::new();
    let actor_ctx = f.ctx(Phase::Spec, Role::Actor, vec![], vec!["spec/types.md"]);
    WriteFileTool { ctx: actor_ctx }
        .call(WriteFileArgs {
            path: "spec/types.md".into(),
            content: "v1".into(),
        })
        .await
        .unwrap();
    let reviser_ctx = f.ctx(Phase::Spec, Role::Reviser, vec![], vec!["spec/types.md"]);
    WriteFileTool { ctx: reviser_ctx }
        .call(WriteFileArgs {
            path: "spec/types.md".into(),
            content: "v2 (revised)".into(),
        })
        .await
        .unwrap();
    let judge_ctx = f.ctx(Phase::Spec, Role::Judge, vec![], vec![]);
    let r = ReadFileTool { ctx: judge_ctx }
        .call(ReadFileArgs {
            path: "spec/types.md".into(),
        })
        .await
        .unwrap();
    assert_eq!(r.content, "v2 (revised)");
}

// ============================================================================
// read_file
// ============================================================================

#[tokio::test]
async fn read_file_succeeds_with_open_read_set() {
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn x() {}");
    let tool = ReadFileTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec![]),
    };
    let r = tool
        .call(ReadFileArgs {
            path: "src/lib.rs".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
    let out = r.unwrap();
    assert!(out.content.contains("pub fn x"));
}

#[tokio::test]
async fn read_file_missing_returns_useful_error() {
    let f = Fixture::new();
    let tool = ReadFileTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec![]),
    };
    let r = tool
        .call(ReadFileArgs {
            path: "src/nope.rs".into(),
        })
        .await;
    let e = r.err().expect("should fail");
    let msg = format!("{e}");
    assert!(msg.contains("does not exist"), "got: {msg}");
}

#[tokio::test]
async fn read_file_outside_read_set_fails() {
    let f = Fixture::new();
    f.write("src/secret.rs", "pub fn s() {}");
    let tool = ReadFileTool {
        ctx: f.ctx(
            Phase::Impl,
            Role::Actor,
            vec!["src/lib.rs"],
            vec!["src/lib.rs"],
        ),
    };
    let r = tool
        .call(ReadFileArgs {
            path: "src/secret.rs".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::ReadNotAllowed { .. })));
}

#[tokio::test]
async fn read_file_in_write_set_is_readable() {
    // A path in write_set is implicitly in read_set.
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn x() {}");
    let tool = ReadFileTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec!["other.rs"], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(ReadFileArgs {
            path: "src/lib.rs".into(),
        })
        .await;
    assert!(r.is_ok());
}

// ============================================================================
// list_files
// ============================================================================

#[tokio::test]
async fn list_files_returns_relative_paths() {
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn x() {}");
    f.write("src/sub/mod.rs", "// empty");
    f.write("Cargo.toml", "[package]\nname = \"x\"\nversion = \"0.1.0\"\n");
    let tool = ListFilesTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec![]),
    };
    let r = tool.call(ListFilesArgs { dir: None }).await.unwrap();
    let files: HashSet<&str> = r.files.iter().map(String::as_str).collect();
    assert!(files.contains("Cargo.toml"));
    assert!(files.contains("src/lib.rs"));
    assert!(files.contains("src/sub/mod.rs"));
}

#[tokio::test]
async fn list_files_filters_dotbureau_and_target() {
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn x() {}");
    f.write(".bureau/log.jsonl", "");
    f.write("target/debug/foo", "");
    let tool = ListFilesTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec![]),
    };
    let r = tool.call(ListFilesArgs { dir: None }).await.unwrap();
    let files: HashSet<&str> = r.files.iter().map(String::as_str).collect();
    assert!(files.contains("src/lib.rs"));
    assert!(!files.iter().any(|s| s.contains(".bureau")));
    assert!(!files.iter().any(|s| s.contains("target")));
}

#[tokio::test]
async fn list_files_subdirectory() {
    let f = Fixture::new();
    f.write("src/a.rs", "pub fn a() {}");
    f.write("src/b.rs", "pub fn b() {}");
    f.write("tests/t.rs", "");
    let tool = ListFilesTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec![]),
    };
    let r = tool
        .call(ListFilesArgs {
            dir: Some("src".into()),
        })
        .await
        .unwrap();
    let files: HashSet<&str> = r.files.iter().map(String::as_str).collect();
    assert!(files.contains("src/a.rs"));
    assert!(files.contains("src/b.rs"));
    assert!(!files.iter().any(|s| s.starts_with("tests/")));
}

#[tokio::test]
async fn list_files_returns_tree_rendering() {
    let f = Fixture::new();
    f.write("Cargo.toml", "[package]\nname = \"x\"\nversion = \"0.1.0\"\n");
    f.write("src/lib.rs", "pub fn x() {}");
    f.write("src/sub/nested.rs", "");
    let tool = ListFilesTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec![]),
    };
    let r = tool.call(ListFilesArgs { dir: None }).await.unwrap();
    assert!(r.tree.contains("src/"));
    assert!(r.tree.contains("lib.rs"));
    assert!(r.tree.contains("sub/"));
    assert!(r.tree.contains("nested.rs"));
    assert!(r.tree.contains("Cargo.toml"));
    // Spot-check that it has tree-drawing characters.
    assert!(r.tree.contains("├") || r.tree.contains("└"));
}

#[tokio::test]
async fn list_files_empty_workdir() {
    let f = Fixture::new();
    let tool = ListFilesTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec![]),
    };
    let r = tool.call(ListFilesArgs { dir: None }).await.unwrap();
    assert!(r.files.is_empty());
}

// ============================================================================
// Spec phase via write_file (no special spec tools)
// ============================================================================

#[tokio::test]
async fn write_file_in_spec_phase_writes_md() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec!["spec/problem.md"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "spec/problem.md".into(),
            content: "# Problem\n\nThe problem is...".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
    assert!(f.read("spec/problem.md").is_some());
}

#[tokio::test]
async fn write_file_in_spec_phase_rejects_non_spec_paths() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() {}".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::PhaseForbidden(_, _))));
}

#[tokio::test]
async fn write_file_in_spec_phase_rejects_non_md_under_spec() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec!["spec/notes.txt"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "spec/notes.txt".into(),
            content: "x".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::PhaseForbidden(_, _))));
}

#[tokio::test]
async fn write_file_spec_md_uses_spec_section_line_limit() {
    // max_file_lines=200, max_spec_section_lines=400 in fixture: a 350-line
    // markdown spec section should be accepted because the larger spec
    // section limit applies to spec/*.md files.
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec!["spec/big.md"]),
    };
    let body = "line\n".repeat(350);
    let r = tool
        .call(WriteFileArgs {
            path: "spec/big.md".into(),
            content: body,
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
}

#[tokio::test]
async fn write_file_spec_md_still_rejects_above_section_limit() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec!["spec/huge.md"]),
    };
    let body = "line\n".repeat(500);
    let r = tool
        .call(WriteFileArgs {
            path: "spec/huge.md".into(),
            content: body,
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::FileTooLarge(_, _))));
}

// ============================================================================
// Directory-prefix entries in write_set / read_set
// ============================================================================

#[tokio::test]
async fn write_set_directory_prefix_allows_descendants() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Spec, Role::Actor, vec![], vec!["spec/"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "spec/types.md".into(),
            content: "# Types".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
    let r2 = tool
        .call(WriteFileArgs {
            path: "spec/sub/nested.md".into(),
            content: "# Nested".into(),
        })
        .await;
    assert!(r2.is_ok(), "{:?}", r2.err());
}

#[tokio::test]
async fn write_set_directory_prefix_rejects_other_paths() {
    let f = Fixture::new();
    let tool = WriteFileTool {
        ctx: f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/"]),
    };
    let r = tool
        .call(WriteFileArgs {
            path: "tests/foo.rs".into(),
            content: "".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::WriteNotAllowed { .. })));
}

#[tokio::test]
async fn read_set_directory_prefix_allows_descendants() {
    let f = Fixture::new();
    f.write("spec/a.md", "A");
    f.write("spec/sub/b.md", "B");
    let tool = ReadFileTool {
        ctx: f.ctx(Phase::Interface, Role::Actor, vec!["spec/"], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(ReadFileArgs {
            path: "spec/sub/b.md".into(),
        })
        .await
        .unwrap();
    assert_eq!(r.content, "B");
}

// ============================================================================
// emit_subtasks
// ============================================================================

#[tokio::test]
async fn emit_subtasks_records_into_ctx() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Spec, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx: ctx.clone() };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "do thing".into(),
                read_files: vec![],
                write_files: vec![PathBuf::from("spec/x.md")],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_ok());
    assert_eq!(ctx.emitted_subtasks.lock().len(), 1);
}

#[tokio::test]
async fn emit_subtasks_disabled_in_debug() {
    let f = Fixture::new();
    let tool = EmitSubtasksTool {
        ctx: f.ctx(Phase::Debug, Role::Actor, vec![], vec![]),
    };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "x".into(),
                read_files: vec![],
                write_files: vec![],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::Forbidden { .. })));
}

#[tokio::test]
async fn emit_subtasks_rejects_empty_write_files_in_non_spec_phases() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "do something".into(),
                read_files: vec![],
                write_files: vec![],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_err());
    let msg = format!("{}", r.err().unwrap());
    assert!(msg.contains("write_files"), "got: {msg}");
}

#[tokio::test]
async fn emit_subtasks_now_requires_write_files_even_in_spec_phase() {
    // Since write_spec_section was removed and spec writes go through write_file
    // (which enforces write_set), spec subtasks must declare write_files like
    // any other phase.
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Spec, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "write a section".into(),
                read_files: vec![],
                write_files: vec![],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_err());
    let msg = format!("{}", r.err().unwrap());
    assert!(msg.contains("write_files"), "got: {msg}");
}

#[tokio::test]
async fn emit_subtasks_in_spec_phase_with_write_files_succeeds() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Spec, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "write the types section".into(),
                read_files: vec![],
                write_files: vec![PathBuf::from("spec/types.md")],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
}

#[tokio::test]
async fn emit_subtasks_rejects_phase_inappropriate_writes() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Test, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "write src code".into(),
                read_files: vec![],
                write_files: vec![PathBuf::from("src/lib.rs")],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_err(), "test phase should reject src/ writes");
}

#[tokio::test]
async fn emit_subtasks_rejects_empty_description() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Spec, Role::Actor, vec![], vec![]);
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "   ".into(),
                read_files: vec![],
                write_files: vec![],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(r.is_err());
}

#[tokio::test]
async fn emit_subtasks_blocked_at_depth_cap() {
    let f = Fixture::new();
    let ctx = Arc::new(TaskCtx {
        task_id: f.task_id,
        phase: Phase::Spec,
        role: Role::Actor,
        workdir: f.workdir.clone(),
        read_set: HashSet::new(),
        write_set: HashSet::new(),
        written: Arc::new(Mutex::new(HashSet::new())),
        emitted_subtasks: Arc::new(Mutex::new(Vec::new())),
        verdict: Arc::new(Mutex::new(None)),
        compiler_errors: Vec::new(),
        max_file_lines: 200,
        max_spec_section_lines: 400,
        depth: 2,
        max_subtask_depth: 2,
            recent_calls: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        state: f.state.clone(),
    });
    let tool = EmitSubtasksTool { ctx };
    let r = tool
        .call(EmitSubtasksArgs {
            tasks: vec![SubtaskDecl {
                description: "x".into(),
                read_files: vec![],
                write_files: vec![],
                spec_sections: vec![],
            }],
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::Forbidden { .. })));
}

// ============================================================================
// replace_fn_body
// ============================================================================

#[tokio::test]
async fn replace_fn_body_swaps_block() {
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn add(x: i32, y: i32) -> i32 { todo!() }\n");
    let tool = ReplaceFnBodyTool {
        ctx: f.ctx(Phase::Debug, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(ReplaceFnBodyArgs {
            path: "src/lib.rs".into(),
            fn_name: "add".into(),
            new_body: "x + y".into(),
        })
        .await;
    assert!(r.is_ok(), "{:?}", r.err());
    let body = f.read("src/lib.rs").unwrap();
    assert!(body.contains("x + y"));
    assert!(!body.contains("todo!"));
}

#[tokio::test]
async fn replace_fn_body_unknown_fn_errors() {
    let f = Fixture::new();
    f.write("src/lib.rs", "pub fn other() { todo!() }\n");
    let tool = ReplaceFnBodyTool {
        ctx: f.ctx(Phase::Debug, Role::Actor, vec![], vec!["src/lib.rs"]),
    };
    let r = tool
        .call(ReplaceFnBodyArgs {
            path: "src/lib.rs".into(),
            fn_name: "missing".into(),
            new_body: "0".into(),
        })
        .await;
    assert!(matches!(r, Err(ToolFailure::FnNotFound(_))));
}

// ============================================================================
// list_compiler_errors / read_compiler_error
// ============================================================================

fn fake_err(id: &str, msg: &str) -> CompilerError {
    CompilerError {
        id: id.into(),
        file: Some(PathBuf::from("src/lib.rs")),
        line: Some(42),
        message: msg.into(),
        raw: serde_json::json!({}),
    }
}

#[tokio::test]
async fn list_compiler_errors_summarizes() {
    let f = Fixture::new();
    let ctx = f.ctx_with_errors(
        Phase::Debug,
        Role::Actor,
        vec!["src/lib.rs"],
        vec![fake_err("E0001", "missing semicolon")],
    );
    let tool = ListCompilerErrorsTool { ctx };
    let r = tool.call(ListCompilerErrArgs {}).await.unwrap();
    assert_eq!(r.errors.len(), 1);
    assert_eq!(r.errors[0].id, "E0001");
}

#[tokio::test]
async fn read_compiler_error_by_id() {
    let f = Fixture::new();
    let ctx = f.ctx_with_errors(
        Phase::Debug,
        Role::Actor,
        vec!["src/lib.rs"],
        vec![fake_err("E0123", "first line\nsecond line")],
    );
    let tool = ReadCompilerErrorTool { ctx };
    let r = tool
        .call(ReadCompilerErrArgs {
            error_id: "E0123".into(),
        })
        .await
        .unwrap();
    assert!(r.message.contains("first line"));
    assert_eq!(r.line, Some(42));
}

#[tokio::test]
async fn read_compiler_error_unknown_errors() {
    let f = Fixture::new();
    let ctx = f.ctx_with_errors(Phase::Debug, Role::Actor, vec![], vec![]);
    let tool = ReadCompilerErrorTool { ctx };
    let r = tool
        .call(ReadCompilerErrArgs {
            error_id: "Enope".into(),
        })
        .await;
    assert!(r.is_err());
}

// ============================================================================
// submit_verdict
// ============================================================================

#[tokio::test]
async fn submit_verdict_records_into_ctx() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Spec, Role::Judge, vec![], vec![]);
    let tool = SubmitVerdictTool { ctx: ctx.clone() };
    let r = tool
        .call(SubmitVerdictArgs {
            satisfactory: false,
            reason: "missing types".into(),
        })
        .await;
    assert!(r.is_ok());
    let v = ctx.verdict.lock().clone().unwrap();
    assert!(!v.satisfactory);
    assert_eq!(v.reason, "missing types");
}

// ============================================================================
// Transcript capture: every tool records its full args as JSON.
// ============================================================================

fn last_tool_call_args(state: &StateHandle, task_id: Uuid) -> Option<String> {
    state.read(|s| {
        s.graph.get(task_id).and_then(|t| {
            t.transcript.iter().rev().find_map(|e| match &e.kind {
                TranscriptKind::ToolCall { args, .. } => Some(args.clone()),
                _ => None,
            })
        })
    })
}

#[tokio::test]
async fn write_file_records_full_args_in_transcript() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec!["src/lib.rs"]);
    let tool = WriteFileTool { ctx: ctx.clone() };
    let _ = tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn answer() -> i32 { 42 }".into(),
        })
        .await
        .unwrap();
    let recorded = last_tool_call_args(&f.state, f.task_id).expect("should have tool_call");
    // Args should serialize as JSON with full content visible (not just path).
    assert!(recorded.contains("answer"));
    assert!(recorded.contains("42"));
}

#[tokio::test]
async fn read_file_failure_records_error_in_transcript() {
    let f = Fixture::new();
    let ctx = f.ctx(Phase::Impl, Role::Actor, vec![], vec![]);
    let tool = ReadFileTool { ctx: ctx.clone() };
    let _ = tool
        .call(ReadFileArgs {
            path: "src/missing.rs".into(),
        })
        .await;
    let last_result = f.state.read(|s| {
        s.graph.get(f.task_id).and_then(|t| {
            t.transcript.iter().rev().find_map(|e| match &e.kind {
                TranscriptKind::ToolResult {
                    ok: false, error, ..
                } => Some(error.clone()),
                _ => None,
            })
        })
    });
    let err = last_result.flatten().expect("error should be recorded");
    assert!(err.contains("does not exist"), "got: {err}");
}

// ============================================================================
// Shared scratch across roles
// ============================================================================

#[tokio::test]
async fn actor_and_reviser_share_written_set() {
    let f = Fixture::new();
    let written = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
    let make = |role: Role| {
        Arc::new(TaskCtx {
            task_id: f.task_id,
            phase: Phase::Impl,
            role,
            workdir: f.workdir.clone(),
            read_set: HashSet::new(),
            write_set: [PathBuf::from("src/lib.rs"), PathBuf::from("src/main.rs")]
                .into_iter()
                .collect(),
            written: written.clone(),
            emitted_subtasks: Arc::new(Mutex::new(Vec::new())),
            verdict: Arc::new(Mutex::new(None)),
            compiler_errors: Vec::new(),
            max_file_lines: 200,
            max_spec_section_lines: 400,
            depth: 0,
            max_subtask_depth: 2,
            recent_calls: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            state: f.state.clone(),
        })
    };
    let actor_tool = WriteFileTool { ctx: make(Role::Actor) };
    actor_tool
        .call(WriteFileArgs {
            path: "src/lib.rs".into(),
            content: "pub fn x() {}".into(),
        })
        .await
        .unwrap();
    let reviser_tool = WriteFileTool { ctx: make(Role::Reviser) };
    reviser_tool
        .call(WriteFileArgs {
            path: "src/main.rs".into(),
            content: "fn main() {}".into(),
        })
        .await
        .unwrap();
    let w = written.lock();
    assert!(w.contains(Path::new("src/lib.rs")));
    assert!(w.contains(Path::new("src/main.rs")));
}

// ============================================================================
// cargo_check / cargo_test live diagnostics
// ============================================================================

#[tokio::test]
async fn cargo_check_passes_on_a_clean_crate() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    std::fs::write(
        workdir.join("Cargo.toml"),
        "[package]\nname = \"clean\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workdir.join("src")).unwrap();
    std::fs::write(workdir.join("src/lib.rs"), "pub fn ok() {}\n").unwrap();

    let ctx = ctx_at(workdir.clone(), Phase::Impl, Role::Actor, vec!["src/lib.rs"]);
    let tool = CargoCheckTool { ctx };
    let r = tool.call(CargoArgs { package: None }).await.unwrap();
    assert!(r.passed, "expected clean check; got: {:#?}", r);
    assert_eq!(r.errors.len(), 0);
    assert_eq!(r.total_errors, 0);
    assert!(r.command.contains("check"));
}

#[tokio::test]
async fn cargo_check_surfaces_syntax_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    std::fs::write(
        workdir.join("Cargo.toml"),
        "[package]\nname = \"bad\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workdir.join("src")).unwrap();
    std::fs::write(workdir.join("src/lib.rs"), "pub fn x() -> i32 { \"oops\" }\n").unwrap();

    let ctx = ctx_at(workdir.clone(), Phase::Impl, Role::Actor, vec!["src/lib.rs"]);
    let tool = CargoCheckTool { ctx };
    let r = tool.call(CargoArgs { package: None }).await.unwrap();
    assert!(!r.passed);
    assert!(!r.errors.is_empty(), "expected an error, got: {:#?}", r);
}

#[tokio::test]
async fn cargo_test_surfaces_runtime_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    std::fs::write(
        workdir.join("Cargo.toml"),
        "[package]\nname = \"rt\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workdir.join("src")).unwrap();
    std::fs::write(
        workdir.join("src/lib.rs"),
        r#"pub fn add(a: i32, b: i32) -> i32 { a + b }
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ok() { assert_eq!(add(1,1), 2); }
    #[test] fn boom() { assert_eq!(add(1,1), 99); }
}
"#,
    )
    .unwrap();

    let ctx = ctx_at(workdir.clone(), Phase::Impl, Role::Actor, vec!["src/lib.rs"]);
    let tool = CargoTestTool { ctx };
    let r = tool
        .call(CargoTestArgs {
            package: None,
            test_filter: None,
            test_filters: vec![],
        })
        .await
        .unwrap();
    assert!(!r.passed);
    assert!(
        r.errors.iter().any(|e| e.message.contains("boom") || e.message.contains("failed")),
        "expected an error mentioning the failing test; got: {:#?}",
        r
    );
}

#[tokio::test]
async fn cargo_test_with_filter_only_runs_matching_tests() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    std::fs::write(
        workdir.join("Cargo.toml"),
        "[package]\nname = \"flt\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workdir.join("src")).unwrap();
    std::fs::write(
        workdir.join("src/lib.rs"),
        r#"#[cfg(test)]
mod tests {
    #[test] fn alpha_ok() {}
    #[test] fn beta_boom() { panic!("boom"); }
}
"#,
    )
    .unwrap();

    // No filter: should fail because beta_boom panics.
    let ctx = ctx_at(workdir.clone(), Phase::Impl, Role::Actor, vec!["src/lib.rs"]);
    let r = CargoTestTool { ctx }
        .call(CargoTestArgs {
            package: None,
            test_filter: None,
            test_filters: vec![],
        })
        .await
        .unwrap();
    assert!(!r.passed);

    // Filter for alpha only: should pass (we never run beta).
    let ctx2 = ctx_at(workdir.clone(), Phase::Impl, Role::Actor, vec!["src/lib.rs"]);
    let r2 = CargoTestTool { ctx: ctx2 }
        .call(CargoTestArgs {
            package: None,
            test_filter: Some("alpha".into()),
            test_filters: vec![],
        })
        .await
        .unwrap();
    assert!(r2.passed, "expected pass when filtering to alpha only; got: {:#?}", r2);
    assert!(
        r2.command.contains("alpha"),
        "command should include the filter: {:?}",
        r2.command
    );
}

fn ctx_at(
    workdir: PathBuf,
    phase: Phase,
    role: Role,
    write_set: Vec<&str>,
) -> Arc<TaskCtx> {
    let mut state = OrchestratorState::new(workdir.clone(), workdir.clone());
    let mut t = Task::new(phase, "test root");
    t.write_files = vec![PathBuf::from("dummy")];
    let task_id = state.graph.insert_root(t);
    let handle = StateHandle::new(state);
    Arc::new(TaskCtx {
        task_id,
        phase,
        role,
        workdir,
        read_set: HashSet::new(),
        write_set: write_set.into_iter().map(PathBuf::from).collect(),
        written: Arc::new(Mutex::new(HashSet::new())),
        emitted_subtasks: Arc::new(Mutex::new(Vec::new())),
        verdict: Arc::new(Mutex::new(None)),
        compiler_errors: Vec::new(),
        max_file_lines: 200,
        max_spec_section_lines: 400,
        depth: 0,
        max_subtask_depth: 2,
        recent_calls: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        state: handle,
    })
}

// ============================================================================
// phase_tools catalog
// ============================================================================

#[tokio::test]
async fn phase_tools_catalog_per_phase_is_non_empty() {
    for p in Phase::ALL {
        let tools = tools::phase_tools(p).await;
        assert!(!tools.is_empty(), "{p} has no tools");
        for t in &tools {
            assert!(!t.name.is_empty());
            assert!(!t.description.is_empty());
        }
    }
}

#[tokio::test]
async fn spec_phase_uses_write_file_too() {
    let tools_v = tools::phase_tools(Phase::Spec).await;
    let names: HashSet<&str> = tools_v.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains("write_file"));
    assert!(names.contains("read_file"));
    assert!(names.contains("list_files"));
    assert!(names.contains("emit_subtasks"));
    assert!(!names.contains("write_spec_section"));
    assert!(!names.contains("read_spec_section"));
    assert!(!names.contains("list_spec_sections"));
}

#[tokio::test]
async fn debug_phase_has_compiler_error_tools() {
    let tools_v = tools::phase_tools(Phase::Debug).await;
    let names: HashSet<&str> = tools_v.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains("list_compiler_errors"));
    assert!(names.contains("read_compiler_error"));
    assert!(names.contains("replace_fn_body"));
}
