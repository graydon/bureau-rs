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
                critique_retries: 0, // disable critique cycle for the simple cases
                max_tasks_total: 64,
                cost_cap_usd: None,
            },
            provider: Provider::default(),
            layout: LayoutKind::SingleCrate,
            project_name: project_name.to_string(),
        },
    })
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
        Role::Actor,
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
        g.get(root).unwrap().spec_md.as_deref(),
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
        Role::Actor,
        vec![ScriptedCall::submit_spec(
            "# tiny\n\nProvides an Adder with `add(a, b) -> i32`.",
        )],
    );
    driver.script(
        Stage::Iface,
        Role::Actor,
        vec![
            ScriptedCall::submit_public(public_rs),
            ScriptedCall::submit_private(private_stub),
        ],
    );
    driver.script(
        Stage::Tests,
        Role::Actor,
        vec![ScriptedCall::submit_tests(tests_rs)],
    );
    driver.script(
        Stage::Impl,
        Role::Actor,
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
        Role::Actor,
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
        Role::Actor,
        vec![ScriptedCall::submit_spec(
            "# core\n\nMath primitives like Adder.",
        )],
    );
    // Iface for root (empty surface) and child (Adder trait + stub impl).
    driver.script(
        Stage::Iface,
        Role::Actor,
        vec![
            ScriptedCall::submit_public("// umbrella crate root\n"),
            ScriptedCall::submit_private("// no umbrella scaffolding\n"),
        ],
    );
    driver.script(
        Stage::Iface,
        Role::Actor,
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
        Role::Actor,
        vec![ScriptedCall::submit_tests("// no integration tests\n")],
    );
    driver.script(
        Stage::Tests,
        Role::Actor,
        vec![ScriptedCall::submit_tests(
            "use super::public::*;\n#[test] fn t() { assert_eq!(<super::AdderImpl as Adder>::add(1,2), 3); }\n",
        )],
    );
    // Impl runs bottom-up (children first, then parents), so script CORE
    // first, then root.
    driver.script(
        Stage::Impl,
        Role::Actor,
        vec![ScriptedCall::submit_private(
            "use super::public::*;\nimpl Adder for super::AdderImpl { fn add(a: i32, b: i32) -> i32 { a + b } }\n",
        )],
    );
    driver.script(
        Stage::Impl,
        Role::Actor,
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
        Role::Actor,
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
        Role::Actor,
        vec![ScriptedCall::submit_spec("# alpha\n\nDoes alpha things.")],
    );
    driver.script(
        Stage::Spec,
        Role::Actor,
        vec![ScriptedCall::submit_spec("# beta\n\nDoes beta things.")],
    );
    // Iface for root + 2 children. Order: root first (insertion), then
    // alpha, then beta.
    driver.script(
        Stage::Iface,
        Role::Actor,
        vec![
            ScriptedCall::submit_public("// umbrella\n"),
            ScriptedCall::submit_private("// no scaffolding\n"),
        ],
    );
    for _ in 0..2 {
        driver.script(
            Stage::Iface,
            Role::Actor,
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
            Role::Actor,
            vec![ScriptedCall::submit_tests("// no tests\n")],
        );
    }
    // Impl bottom-up: leaves first, then root.
    for _ in 0..3 {
        driver.script(
            Stage::Impl,
            Role::Actor,
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
            Role::Actor,
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
