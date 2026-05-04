//! End-to-end tests of the engine driving a small graph through stages
//! using the scripted [`MockLlmDriver`]. No network calls; the test
//! deterministically scripts what each (stage, role) "says".

use bureau_rs::config::{Config, ConfigToml, LayoutKind, Limits, ModelConfig, Provider};
use bureau_rs::engine::Engine;
use bureau_rs::graph::StageState;
use bureau_rs::mock_driver::{MockLlmDriver, ScriptedCall};
use bureau_rs::state::{EngineState, StateHandle};
use bureau_rs::tools::{ChildDecl, Role};
use bureau_rs::graph::Stage;
use std::sync::Arc;

fn make_config(workdir: std::path::PathBuf, project_name: &str) -> Arc<Config> {
    Arc::new(Config {
        config_dir: workdir.clone(),
        problem: "Build a thing.".to_string(),
        toml: ConfigToml {
            models: ModelConfig {
                actor: "mock".into(),
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
    // Don't bother completing the pipeline — we just need ONE drive() call
    // to verify the preamble. Script enough to advance spec.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::submit_spec("# samba_rs\n\nThe server.")],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;

    // The first call should be (Spec, Actor) on the root.
    let received = driver.received.lock().clone();
    assert!(!received.is_empty(), "engine should drive at least once");
    let (stage, role, preamble) = &received[0];
    assert_eq!(*stage, Stage::Spec);
    assert_eq!(*role, Role::Writer);
    assert!(
        preamble.contains("Project mission"),
        "preamble must include the Project mission section: {preamble}"
    );
    assert!(
        preamble.contains("SMB/CIFS"),
        "preamble must contain the actual problem.md content: {preamble}"
    );

    // Root node's description should also be derived from problem.md, not
    // the placeholder "Project root."
    let g = state.snapshot().graph;
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
    // Initial actor turn: ONE composite submit_spec carrying a child
    // with a self-dep — the whole submission fails atomically.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::SubmitSpec {
            public: "# retry\n\nA tiny crate.".into(),
            private: None,
            children: vec![ChildDecl {
                name: "core".into(),
                description: "math".into(),
                deps: vec!["core".into()], // self-dep — fails
                crate_boundary: false,
            }],
            deps: vec![],
        }],
    );
    // Forced retry script: re-call submit_spec, this time with valid args.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![ScriptedCall::SubmitSpec {
            public: "# retry\n\nA tiny crate.".into(),
            private: None,
            children: vec![ChildDecl {
                name: "core".into(),
                description: "math".into(),
                deps: vec![],
                crate_boundary: false,
            }],
            deps: vec![],
        }],
    );
    driver.auto_approve_judges();

    let engine = Arc::new(Engine::with_driver(config, state.clone(), driver.clone()).unwrap());
    let _ = engine.run().await;

    // Two actor invocations should have happened (initial + 1 forced retry).
    let received = driver.received.lock().clone();
    let actor_count = received
        .iter()
        .filter(|(s, r, _)| *s == Stage::Spec && *r == Role::Writer)
        .count();
    assert!(
        actor_count >= 2,
        "expected ≥2 actor drives (initial + forced retry); got {actor_count}: {:?}",
        received.iter().map(|(s, r, _)| (s, r)).collect::<Vec<_>>()
    );

    // The retry preamble should have been the focused RETRY one.
    let retry_preamble = &received[1].2;
    assert!(
        retry_preamble.contains("RETRY"),
        "second actor drive should use the focused retry preamble: {retry_preamble}"
    );
    assert!(
        retry_preamble.contains("submit_spec"),
        "retry preamble should name the failed tool: {retry_preamble}"
    );

    // After the retry succeeded, the graph should have the child.
    let g = state.snapshot().graph;
    let core = g.find_by_name("core");
    assert!(core.is_some(), "core child should exist after successful retry");
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

    let received = driver.received.lock().clone();
    assert!(received.len() >= 2, "should have retried");
    let retry_preamble = &received[1].2;
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
    let retry_preamble = &received[1].2;
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
    // Spec actor: call submit_spec, then a BAD decompose (child deps on
    // itself).
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![
            ScriptedCall::submit_spec("# split\n\nA crate with subsystems."),
            ScriptedCall::decompose(vec![ChildDecl {
                name: "core".into(),
                description: "math primitives".into(),
                deps: vec!["core".into()], // self-dep — will fail
                crate_boundary: false,
            }]),
        ],
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
        .find(|(s, r, _)| *s == Stage::Spec && *r == Role::Critic)
        .expect("critic should have run");
    assert!(
        critic.2.contains("Prior turn had failed tool calls"),
        "critic preamble must surface the failed decompose call: {}",
        critic.2
    );
    assert!(
        critic.2.contains("decompose"),
        "critic should see which tool failed: {}",
        critic.2
    );
    assert!(
        critic.2.to_lowercase().contains("self-loop") || critic.2.contains("itself"),
        "critic should see WHY it failed: {}",
        critic.2
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

    let g = state.snapshot().graph;
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
    let g = &snap.graph;
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
    // Root spec: decompose into one child + write a thin spec for the umbrella.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![
            ScriptedCall::submit_spec(
                "# split\n\nThe split crate. Decomposes into `core` for math.",
            ),
            ScriptedCall::decompose(vec![ChildDecl {
                name: "core".into(),
                description: "core math primitives".into(),
                deps: vec![],
                crate_boundary: false,
            }]),
        ],
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
    let g = &snap.graph;
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
    // Root spec decomposes into two siblings.
    driver.script(
        Stage::Spec,
        Role::Writer,
        vec![
            ScriptedCall::submit_spec("# para\n\nUmbrella with two leaves.\n"),
            ScriptedCall::decompose(vec![
                ChildDecl {
                    name: "alpha".into(),
                    description: "first".into(),
                    deps: vec![],
                    crate_boundary: false,
                },
                ChildDecl {
                    name: "beta".into(),
                    description: "second".into(),
                    deps: vec![],
                    crate_boundary: false,
                },
            ]),
        ],
    );
    // Two child spec calls (one each).
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
    let g = state.snapshot().graph;
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
    let g = state.snapshot().graph;
    let root = g.root.unwrap();
    let n = g.get(root).unwrap();
    // Spec stage should be Failed because judge rejected every attempt.
    assert_eq!(n.stages.spec, StageState::Failed, "spec should fail");
    assert!(result.is_err(), "engine should error out");
}
