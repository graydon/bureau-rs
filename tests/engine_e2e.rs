//! End-to-end tests of the engine driving a small graph through stages
//! using the scripted [`MockLlmDriver`]. No network calls; the test
//! deterministically scripts what each (stage, role) "says".

use bureau_rs::config::{Config, ConfigToml, LayoutKind, Limits, ModelConfig, Provider};
use bureau_rs::engine::Engine;
use bureau_rs::graph::StageState;
use bureau_rs::mock_driver::{MockLlmDriver, ScriptedCall};
use bureau_rs::state::{EngineState, StateHandle};
use bureau_rs::tools::Role;
use bureau_rs::graph::Stage;
use std::sync::Arc;

fn make_config(workdir: std::path::PathBuf, project_name: &str) -> Arc<Config> {
    Arc::new(Config {
        config_dir: workdir.clone(),
        problem: "Build a thing.".to_string(),
        style: None,
        toml: ConfigToml {
            models: ModelConfig {
                default: "mock".into(),
                escalated: None,
                summary: None,
                architect: None,
                spec: None,
                iface: None,
                tests: None,
                impl_: None,
                debug: None,
                writer: None,
                critic: None,
                reviser: None,
                judge: None,
                max_tokens: 1024,
                temperature: 0.0,
                max_turns: 10,
            },
            limits: Limits {
                max_file_lines: 300,
                max_spec_section_lines: 400,
                max_parallel_tasks: 1,
                max_stage_retries: 0,
                tool_retry_budget: 0, // disable forced retry for predictable scripted tests
                args_display_cap: 60,
                critique_retries: 0,  // disable critique cycle for the simple cases
                max_tasks_total: 64,
                max_nodes: 32,
                max_node_depth: 4,
                cost_cap_usd: None,
                max_quickfix_iters: 0, // disable inner quickfix in scripted tests
                task_transcript_cap: 0, // unbounded for predictable assertions
            },
            provider: Provider::default(),
            layout: LayoutKind::SingleCrate,
            project_name: project_name.to_string(),
        },
    })
}

#[tokio::test]
async fn engine_injects_project_mission_into_root_node_preamble() {
    // The whole bug from the user's session: problem.md content never
    // reached the root node, so the model was guessing what to build from
    // the project name. This test pins the fix.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "samba_rs")).clone();
    config.problem = "# Problem: A Samba-equivalent server suite\n\n\
        Build a Rust workspace that reimplements SMB/CIFS file serving, \
        NetBIOS name resolution, and DCE/RPC."
        .into();
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "samba_rs".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    // Architect runs first (single-shot, empty tree is fine for this
    // test — we just need at least one drive to inspect).
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# samba_rs\n\nThe server.")],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;

    // The first call should be (Architect, Writer) on the root.
    let received = driver.received.lock().clone();
    assert!(!received.is_empty(), "engine should drive at least once");
    let (stage, role, preamble, _user_prompt) = &received[0];
    assert_eq!(*stage, Stage::Architect);
    assert_eq!(*role, Role::Writer);
    // Project mission lives in the SYSTEM prompt now (top-tier cache
    // stability: every call across every (node, stage, role) sees the
    // same bytes, so providers can reuse the cached prefix project-wide).
    assert!(
        preamble.contains("Project mission"),
        "system preamble must include the Project mission section: {preamble}"
    );
    assert!(
        preamble.contains("SMB/CIFS"),
        "system preamble must contain the actual problem.md content: {preamble}"
    );

    // Root node's description should also be derived from problem.md, not
    // the placeholder "Project root."
    let g = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    let root = g.root.unwrap();
    let desc = &g.get(root).unwrap().description;
    assert_ne!(desc, "Project root.", "root description should be derived from problem.md");
    assert!(
        desc.contains("Samba") || desc.contains("SMB"),
        "root description should reflect problem.md: {desc}"
    );
}

#[tokio::test]
async fn unresolved_tool_failure_triggers_forced_retry_until_resolved() {
    // The actor calls decompose with a self-dep (fail), then submit_spec
    // (success). With tool_retry_budget = 2, the framework should fire a
    // fresh drive() for the unresolved decompose failure. We script the
    // retry to call decompose again with VALID args; it succeeds and the
    // child is created.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "retry")).clone();
    config.toml.limits.tool_retry_budget = 2;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "retry".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    // Architect first — give the architect a single-leaf tree so we have
    // a `core` node to run spec on.
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[(
            "core",
            "math primitives",
        )])],
    );
    // Initial spec turn: bad submit_spec (empty public — atomic failure).
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::SubmitSpec {
            public: "".into(), // invalid — required to be non-empty
            private: None,
            deps: vec![],
        }],
    );
    // Forced retry script: re-call submit_spec with valid args.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::SubmitSpec {
            public: "# retry\n\nA tiny crate.".into(),
            private: None,
            deps: vec![],
        }],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;

    // Two spec-writer invocations should have happened (initial + retry).
    let received = driver.received.lock().clone();
    let spec_writer_drives: Vec<&(Stage, Role, String, String)> = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Writer)
        .collect();
    assert!(
        spec_writer_drives.len() >= 2,
        "expected ≥2 spec writer drives (initial + forced retry); got {}: {:?}",
        spec_writer_drives.len(),
        received.iter().map(|(s, r, _, _)| (s, r)).collect::<Vec<_>>()
    );
    // The second spec-writer drive should be the focused RETRY one.
    let retry_preamble = &spec_writer_drives[1].2;
    assert!(
        retry_preamble.contains("RETRY"),
        "second spec writer drive should use the focused retry preamble: {retry_preamble}"
    );
    assert!(
        retry_preamble.contains("submit_spec"),
        "retry preamble should name the failed tool: {retry_preamble}"
    );
}

#[tokio::test]
async fn retry_preamble_truncates_long_args() {
    // The forced-retry preamble must NOT echo the full failed args back
    // to the model — risk of losing the boundary on large args. Default
    // cap is 60 bytes; pin the truncation marker.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "trunc")).clone();
    config.toml.limits.tool_retry_budget = 1;
    config.toml.limits.args_display_cap = 60;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "trunc".into(),
    ));
    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    // Force a failure with a big args blob — submit_spec with more lines
    // than max_spec triggers FileTooLarge, and the args (the spec body)
    // are big enough to exercise truncation.
    let too_long = format!("# spec\n\n{}", "lorem ipsum dolor.\n".repeat(500));
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec(too_long)],
    );
    // Retry with a short spec that succeeds.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# spec\n\nshort and valid.")],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;

    // Find the spec-stage retry preamble (architect ran first).
    let received = driver.received.lock().clone();
    let spec_drives: Vec<&(Stage, Role, String, String)> = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Writer)
        .collect();
    assert!(spec_drives.len() >= 2, "should have retried spec");
    let retry_preamble = &spec_drives[1].2;
    assert!(
        retry_preamble.contains("TRUNCATED"),
        "retry preamble should mark the truncation: {retry_preamble}"
    );
    assert!(
        retry_preamble.contains("bytes total"),
        "retry preamble should report the original size"
    );
    // With a 60-byte cap, only a tiny prefix of the args should appear.
    let count_lorem = retry_preamble.matches("lorem ipsum").count();
    assert!(
        count_lorem <= 2,
        "args should be heavily truncated at 60 bytes; saw {count_lorem} lorem occurrences"
    );
}

#[tokio::test]
async fn args_display_cap_is_configurable_and_respected() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "cap")).clone();
    config.toml.limits.tool_retry_budget = 1;
    // A more generous cap — the args excerpt should be larger.
    config.toml.limits.args_display_cap = 400;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "cap".into(),
    ));
    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    let too_long = format!("# spec\n\n{}", "lorem ipsum dolor.\n".repeat(500));
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec(too_long)],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# spec\n\nshort.")],
    );
    driver.auto_approve_judges();
    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;
    let received = driver.received.lock().clone();
    let spec_drives: Vec<&(Stage, Role, String, String)> = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Writer)
        .collect();
    let retry_preamble = &spec_drives[1].2;
    let count_lorem = retry_preamble.matches("lorem ipsum").count();
    // With 400-byte cap, ~22 lorem occurrences (each is ~18 bytes); pin
    // that we get more than the 60-byte cap would have allowed (≤2).
    assert!(
        count_lorem >= 5,
        "with cap=400 we should see more args; saw {count_lorem}"
    );
}

#[tokio::test]
async fn failed_decompose_in_actor_turn_surfaces_to_critic_and_reviser() {
    // The user-reported bug: an actor turn calls `decompose`, gets a
    // validation error (e.g. self-dep), still calls submit_spec, then
    // ends. The framework used to lose the decomposition silently. This
    // test pins the new behavior: the next role's preamble explicitly
    // calls out the failed tool call so the model can retry.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "split")).clone();
    config.toml.limits.critique_retries = 1;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "split".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    // Architect lays out a single child, then spec stage on root makes
    // a BAD submit_spec call (empty `public` — atomic failure).
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[(
            "core",
            "core subsystem",
        )])],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::SubmitSpec {
            public: "".into(), // empty — fails atomically
            private: None,
            deps: vec![],
        }],
    );
    // Force a critique cycle so we get to see the critic preamble.
    driver.script(Stage::Spec, Role::Critic, vec![]);
    driver.script(Stage::Spec, Role::Reviser, vec![]);
    driver.script(
        Stage::Spec,
        Role::Judge,
        vec![ScriptedCall::verdict_ok()],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;

    let received = driver.received.lock().clone();
    // Find the critic's preamble — the next role after the actor.
    let critic = received
        .iter()
        .find(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Critic)
        .expect("critic should have run");
    // Failed-tool context now lives in the user prompt (context_doc),
    // not the preamble (system prompt) — see the cache-friendly prompt
    // layout in engine.rs.
    assert!(
        critic.3.contains("Prior turn had failed tool calls"),
        "critic user prompt must surface the failed submit_spec call: {}",
        critic.3
    );
    assert!(
        critic.3.contains("submit_spec"),
        "critic should see which tool failed: {}",
        critic.3
    );
}

#[tokio::test]
async fn engine_advances_root_through_spec_stage() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let config = make_config(workdir.clone(), "echo_it");
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "echo_it".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# echo_it\n\nA tiny echo CLI.")],
    );

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver).unwrap());
    // We don't run the full engine.run() — that'd require scripting every
    // stage. Instead drive one stage's worth of advancement and inspect.
    // We expose `advance_stage` indirectly by scripting all later stages
    // such that everything runs to completion. But that requires a
    // realistic Rust crate. So instead we use `cargo check`-friendly
    // content for iface/tests/impl, OR set max_tasks_total=1 to bail
    // after spec.
    // Approach: run with realistic scripts for every stage, and let cargo
    // gates pass naturally.
    //
    // For this test, we'll just halt after spec by exhausting the script.
    // The engine will then try to advance Iface (no script → empty actor
    // reply → public.rs not authored → cargo_check fails) and bail.
    //
    // We assert: spec stage Done, iface stage Failed (or NotStarted +
    // pipeline stopped). Either is acceptable for this minimal test;
    // we test full pipelines elsewhere.
    let _ = engine.run().await;

    let g = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    let root = g.root.unwrap();
    assert_eq!(g.get(root).unwrap().stages.spec, StageState::Done);
    assert_eq!(
        g.get(root).unwrap().spec_public_md.as_deref(),
        Some("# echo_it\n\nA tiny echo CLI.")
    );
}

#[tokio::test]
async fn engine_runs_full_pipeline_on_a_simple_root_node() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let config = make_config(workdir.clone(), "tiny");
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "tiny".into(),
    ));

    // Realistic Rust content that will pass cargo check / test.
    let public_rs = r#"
pub trait Adder: Sized {
    fn add(a: i32, b: i32) -> i32;
}

pub struct AdderImpl;
"#;
    // Iface stage writes a stub impl. Bodies are todo!() so tests can
    // COMPILE against the trait surface; impl stage replaces the todo!().
    let private_stub = r#"
use super::public::*;

impl Adder for super::AdderImpl {
    fn add(_a: i32, _b: i32) -> i32 { todo!() }
}
"#;
    let private_real = r#"
use super::public::*;

impl Adder for super::AdderImpl {
    fn add(a: i32, b: i32) -> i32 { a + b }
}
"#;
    let tests_rs = r#"
use super::public::*;
#[test] fn adds_correctly() {
    assert_eq!(<super::AdderImpl as Adder>::add(2, 3), 5);
}
"#;

    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec(
            "# tiny\n\nProvides an Adder with `add(a, b) -> i32`.",
        )],
    );
    driver.script(
        Stage::Iface,
        Role::Writer,
        vec![
            ScriptedCall::submit_public(public_rs),
            ScriptedCall::submit_private(private_stub),
        ],
    );
    driver.script(
        Stage::Tests,
        Role::Writer,
        vec![ScriptedCall::submit_tests(tests_rs)],
    );
    driver.script(
        Stage::Impl,
        Role::Writer,
        vec![ScriptedCall::submit_private(private_real)],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver).unwrap());
    let result = engine.run().await;
    let snap = state.snapshot();
    let g = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    let root = g.root.unwrap();
    let n = g.get(root).unwrap();

    // Spec, Iface, Tests, Impl all done.
    if n.stages.tests != StageState::Done || n.stages.impl_ != StageState::Done {
        eprintln!("---- diagnostic dump ----");
        for t in snap.tasks.values() {
            eprintln!(
                "task {} {} {}: status={:?} err={:?}",
                t.id, t.node_name, t.stage, t.status, t.error
            );
        }
        eprintln!("workdir: {}", workdir.display());
        eprintln!(
            "private.rs:\n{}",
            std::fs::read_to_string(workdir.join("src/private.rs")).unwrap_or_default()
        );
        eprintln!(
            "tests.rs:\n{}",
            std::fs::read_to_string(workdir.join("src/tests.rs")).unwrap_or_default()
        );
        eprintln!(
            "public.rs:\n{}",
            std::fs::read_to_string(workdir.join("src/public.rs")).unwrap_or_default()
        );
        eprintln!(
            "mod.rs:\n{}",
            std::fs::read_to_string(workdir.join("src/mod.rs")).unwrap_or_default()
        );
    }
    assert_eq!(n.stages.spec, StageState::Done, "spec should be done");
    assert_eq!(n.stages.iface, StageState::Done, "iface should be done");
    assert_eq!(n.stages.tests, StageState::Done, "tests should be done");
    assert_eq!(n.stages.impl_, StageState::Done, "impl should be done");
    assert!(result.is_ok(), "engine should complete: {result:?}");

    // Files actually exist on disk with the authored content.
    let pub_disk = std::fs::read_to_string(workdir.join("src/public.rs")).unwrap();
    assert!(pub_disk.contains("pub trait Adder"));
    let priv_disk = std::fs::read_to_string(workdir.join("src/private.rs")).unwrap();
    assert!(priv_disk.contains("impl Adder"));
}

#[tokio::test]
async fn engine_decomposes_root_then_advances_children() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let config = make_config(workdir.clone(), "split");
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "split".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    // Architect lays out: root + one child `core`.
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[(
            "core",
            "core math primitives",
        )])],
    );
    // Root spec: thin umbrella spec, no decompose (architect already did).
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec(
            "# split\n\nThe split crate. Decomposed by architect into `core`.",
        )],
    );
    // Child spec.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec(
            "# core\n\nMath primitives like Adder.",
        )],
    );
    // Iface for root (empty surface) and child (Adder trait + stub impl).
    driver.script(
        Stage::Iface,
        Role::Writer,
        vec![
            ScriptedCall::submit_public("// umbrella crate root\n"),
            ScriptedCall::submit_private("// no umbrella scaffolding\n"),
        ],
    );
    driver.script(
        Stage::Iface,
        Role::Writer,
        vec![
            ScriptedCall::submit_public(
                "pub trait Adder: Sized { fn add(a: i32, b: i32) -> i32; }\npub struct AdderImpl;\n",
            ),
            ScriptedCall::submit_private(
                "use super::public::*;\nimpl Adder for super::AdderImpl { fn add(_a: i32, _b: i32) -> i32 { todo!() } }\n",
            ),
        ],
    );
    // Tests: empty for root; real for child.
    driver.script(
        Stage::Tests,
        Role::Writer,
        vec![ScriptedCall::submit_tests("// no integration tests\n")],
    );
    driver.script(
        Stage::Tests,
        Role::Writer,
        vec![ScriptedCall::submit_tests(
            "use super::public::*;\n#[test] fn t() { assert_eq!(<super::AdderImpl as Adder>::add(1,2), 3); }\n",
        )],
    );
    // Impl runs bottom-up (children first, then parents), so script CORE
    // first, then root.
    driver.script(
        Stage::Impl,
        Role::Writer,
        vec![ScriptedCall::submit_private(
            "use super::public::*;\nimpl Adder for super::AdderImpl { fn add(a: i32, b: i32) -> i32 { a + b } }\n",
        )],
    );
    driver.script(
        Stage::Impl,
        Role::Writer,
        vec![ScriptedCall::submit_private("// no umbrella impl\n")],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver).unwrap());
    let result = engine.run().await;
    let snap = state.snapshot();
    let g = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    if !result.is_ok() {
        eprintln!("---- decompose-test diagnostic ----");
        for n in g.iter() {
            eprintln!(
                "node {} stages spec={:?} iface={:?} tests={:?} impl={:?}",
                n.name, n.stages.spec, n.stages.iface, n.stages.tests, n.stages.impl_
            );
        }
        for t in snap.tasks.values() {
            eprintln!(
                "task {} {} {}: status={:?} err={:?}",
                t.id, t.node_name, t.stage, t.status, t.error
            );
        }
        eprintln!("workdir: {}", workdir.display());
        for p in [
            "src/mod.rs",
            "src/public.rs",
            "src/private.rs",
            "src/tests.rs",
            "src/core/mod.rs",
            "src/core/public.rs",
            "src/core/private.rs",
            "src/core/tests.rs",
        ] {
            if let Ok(content) = std::fs::read_to_string(workdir.join(p)) {
                eprintln!("---- {p} ----\n{content}");
            }
        }
    }
    assert_eq!(g.len(), 2, "should have root + 1 child");
    let core = g.find_by_name("core").unwrap();
    assert_eq!(core.stages.spec, StageState::Done);
    assert_eq!(core.stages.iface, StageState::Done);
    assert_eq!(core.stages.tests, StageState::Done);
    assert_eq!(core.stages.impl_, StageState::Done);
    assert!(result.is_ok(), "engine should complete: {result:?}");
}

#[tokio::test]
async fn engine_runs_two_independent_nodes_in_parallel() {
    // Two independent leaf children with no inter-deps. With
    // max_parallel_tasks=2 the engine should advance both concurrently.
    // The mock driver doesn't actually need wall-clock parallelism to
    // verify correctness — we just confirm the engine completes both,
    // and that scheduling works under the parallel loop.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "para")).clone();
    config.toml.limits.max_parallel_tasks = 2;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "para".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    // Architect lays out two leaves under root.
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[
            ("alpha", "first"),
            ("beta", "second"),
        ])],
    );
    // Root + two child specs.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# para\n\nUmbrella with two leaves.\n")],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# alpha\n\nDoes alpha things.")],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# beta\n\nDoes beta things.")],
    );
    // Iface for root + 2 children. Order: root first (insertion), then
    // alpha, then beta.
    driver.script(
        Stage::Iface,
        Role::Writer,
        vec![
            ScriptedCall::submit_public("// umbrella\n"),
            ScriptedCall::submit_private("// no scaffolding\n"),
        ],
    );
    for _ in 0..2 {
        driver.script(
            Stage::Iface,
            Role::Writer,
            vec![
                ScriptedCall::submit_public("// leaf\n"),
                ScriptedCall::submit_private("// no scaffolding\n"),
            ],
        );
    }
    // Tests for all three (empty).
    for _ in 0..3 {
        driver.script(
            Stage::Tests,
            Role::Writer,
            vec![ScriptedCall::submit_tests("// no tests\n")],
        );
    }
    // Impl bottom-up: leaves first, then root.
    for _ in 0..3 {
        driver.script(
            Stage::Impl,
            Role::Writer,
            vec![ScriptedCall::submit_private("// nothing to do\n")],
        );
    }
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver).unwrap());
    let result = engine.run().await;
    let g = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    assert_eq!(g.len(), 3);
    for n in g.iter() {
        assert_eq!(n.stages.spec, StageState::Done, "node `{}` spec", n.name);
        assert_eq!(n.stages.iface, StageState::Done, "node `{}` iface", n.name);
        assert_eq!(n.stages.tests, StageState::Done, "node `{}` tests", n.name);
        assert_eq!(n.stages.impl_, StageState::Done, "node `{}` impl", n.name);
    }
    assert!(result.is_ok(), "engine should complete: {result:?}");
}

#[tokio::test]
async fn engine_halts_on_unsatisfactory_judge_verdict() {
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "tiny")).clone();
    config.toml.limits.critique_retries = 1; // force a critique cycle
    config.toml.limits.max_stage_retries = 1; // limit retries so the test ends
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "tiny".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    // The actor writes a spec; the judge always rejects.
    for _ in 0..8 {
        driver.script(
            Stage::Spec,
            Role::Writer,
            vec![ScriptedCall::submit_spec("# tiny\n\nspec body")],
        );
        driver.script(Stage::Spec, Role::Critic, vec![]);
        driver.script(Stage::Spec, Role::Reviser, vec![]);
        driver.script(
            Stage::Spec,
            Role::Judge,
            vec![ScriptedCall::verdict_fail("nope, never satisfied")],
        );
    }

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver).unwrap());
    let result = engine.run().await;
    let g = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    let root = g.root.unwrap();
    let n = g.get(root).unwrap();
    // Spec stage should be Failed because judge rejected every attempt.
    assert_eq!(n.stages.spec, StageState::Failed, "spec should fail");
    assert!(result.is_err(), "engine should error out");
}

#[tokio::test]
async fn critic_clean_skips_reviser_and_judge() {
    // When the critic calls submit_critique with an empty issues list,
    // the engine should skip the reviser AND judge for that round.
    // We confirm this by tallying which (stage, role) drives the model
    // actually saw — the reviser and judge for Spec should be ZERO
    // even though critique_retries=1 is set.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "tiny")).clone();
    config.toml.limits.critique_retries = 1;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "tiny".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    // Spec writer: produce content.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# tiny\n\nspec body\n")],
    );
    // Spec critic: empty issues list (clean) → engine skips reviser+judge.
    driver.script(Stage::Spec, Role::Critic, vec![ScriptedCall::critique_clean()]);
    // Iface / Tests / Impl: writer-only via auto_approve.
    for stage in [Stage::Iface, Stage::Tests, Stage::Impl] {
        driver.script(
            stage,
            Role::Writer,
            vec![match stage {
                Stage::Iface => ScriptedCall::submit_public("// empty\n"),
                Stage::Tests => ScriptedCall::submit_tests("// no tests\n"),
                Stage::Impl => ScriptedCall::submit_private("// nothing to do\n"),
                _ => unreachable!(),
            }],
        );
        driver.script(stage, Role::Critic, vec![ScriptedCall::critique_clean()]);
    }

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    engine.run().await.unwrap();

    // Count Reviser and Judge invocations for Spec stage — should be zero.
    let received = driver.received.lock();
    let spec_reviser = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Reviser)
        .count();
    let spec_judge = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Judge)
        .count();
    assert_eq!(spec_reviser, 0, "reviser should not run when critic is clean");
    assert_eq!(spec_judge, 0, "judge should not run when critic is clean");
}

#[tokio::test]
async fn critic_with_issues_runs_full_cycle() {
    // Counterpoint to the clean-critic test: if the critic raises any
    // issues, the reviser AND judge should both run.
    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    let mut config = (*make_config(workdir.clone(), "tiny")).clone();
    config.toml.limits.critique_retries = 1;
    let config = Arc::new(config);
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "tiny".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# tiny\n\nspec body\n")],
    );
    driver.script(
        Stage::Spec,
        Role::Critic,
        vec![ScriptedCall::critique_one("spec is too vague")],
    );
    driver.script(Stage::Spec, Role::Reviser, vec![]);
    driver.script(Stage::Spec, Role::Judge, vec![ScriptedCall::verdict_ok()]);
    // Other stages: keep them clean to avoid noise.
    for stage in [Stage::Iface, Stage::Tests, Stage::Impl] {
        driver.script(
            stage,
            Role::Writer,
            vec![match stage {
                Stage::Iface => ScriptedCall::submit_public("// empty\n"),
                Stage::Tests => ScriptedCall::submit_tests("// no tests\n"),
                Stage::Impl => ScriptedCall::submit_private("// nothing to do\n"),
                _ => unreachable!(),
            }],
        );
        driver.script(stage, Role::Critic, vec![ScriptedCall::critique_clean()]);
    }

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    engine.run().await.unwrap();

    let received = driver.received.lock();
    let spec_reviser = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Reviser)
        .count();
    let spec_judge = received
        .iter()
        .filter(|(s, r, _, _)| *s == Stage::Spec && *r == Role::Judge)
        .count();
    assert_eq!(spec_reviser, 1, "reviser should run when critic has issues");
    assert_eq!(spec_judge, 1, "judge should run when critic has issues");
}

#[tokio::test]
async fn restart_resets_stale_inprogress_stages() {
    // Simulates a restart in a workdir where a prior run crashed
    // mid-task: the on-disk graph has stages stuck at InProgress.
    // Without the reset sweep, `stage_is_ready` filters them out
    // forever and the pipeline hangs.
    use bureau_rs::graph::{self, NodeGraph, Node, StageState};
    use bureau_rs::worktree::Workspace;

    let tmp = tempfile::tempdir().unwrap();
    let workdir = tmp.path().to_path_buf();
    // Seed a graph with the root's Architect stuck at InProgress on disk.
    let ws = Workspace::init(&workdir).unwrap();
    let mut g = NodeGraph::new();
    let mut root = Node::new("tiny", "");
    root.crate_boundary = true;
    root.stages.architect = StageState::InProgress;
    g.insert_root(root).unwrap();
    graph::save(&workdir, &g).unwrap();
    ws.commit_main("seed: stale inprogress").unwrap();

    let config = make_config(workdir.clone(), "tiny");
    let state = StateHandle::new(EngineState::new(
        workdir.clone(),
        workdir.clone(),
        "tiny".into(),
    ));

    let driver = Arc::new(MockLlmDriver::new());
    driver.script(
        Stage::Architect,
        Role::Writer,
        vec![ScriptedCall::submit_architecture_simple(&[])],
    );
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# tiny\n\nbody\n")],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver).unwrap());
    let _ = engine.run().await;

    // Architect should have been re-run (not stuck at InProgress).
    let g2 = bureau_rs::graph::load(&workdir, bureau_rs::render::Layout::SingleCrate).unwrap();
    let root = g2.root.unwrap();
    assert_eq!(
        g2.get(root).unwrap().stages.architect,
        StageState::Done,
        "architect should have been re-driven after the stale InProgress was reset"
    );
}
