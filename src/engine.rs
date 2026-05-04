//! The per-node-stage engine.
//!
//! The previous architecture had global "phases" (spec → interface → tests
//! → impl → debug → opt) that ran across the entire workspace serially. The
//! new architecture decomposes work into NODES (units of abstraction) and
//! runs each node through its own STAGE lifecycle (spec → iface → tests →
//! impl → [debug] → [opt]) in dep-aware topological order. Independent
//! nodes can advance in parallel; dependent nodes wait for their deps to
//! reach a sufficient stage.
//!
//! This module is responsible for:
//!
//! 1. Picking the next ready (node, stage) to advance.
//! 2. Running that pair through the writer → critic → reviser → judge cycle.
//! 3. Re-rendering the on-disk source tree after substantive writes.
//! 4. Running cargo gates after iface / tests / impl stages.
//! 5. Re-attempting on gate failure (within `max_stage_retries`).
//! 6. Failing the pipeline if a stage exhausts its budget.

use crate::config::Config;
use crate::graph::{Node, NodeGraph, NodeId, Stage, StageState};
use crate::node_context;
use crate::render::{self, Layout};
use crate::state::{
    EngineState, EngineTask, HistoryEntry, SchedulerState, StateHandle, TaskStatus, TokenUsage,
    UiEvent,
};
use crate::tools::{
    CargoCheckTool, CargoClippyTool, CargoTestNoRunTool, CargoTestTool,
    JudgeVerdict, Role, SubmitPrivateTool, SubmitPublicTool, SubmitSpecTool, SubmitTestsTool,
    SubmitVerdictTool, TaskCtx, TranscriptEntry, TranscriptKind,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::Utc;
use parking_lot::Mutex;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openrouter;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

/// Inputs to a single agent invocation. Bundled so we can swap rig out for
/// a fake driver in tests.
#[derive(Debug, Clone)]
pub struct DriveParams {
    pub model: String,
    pub preamble: String,
    pub user_prompt: String,
    pub stage: Stage,
    pub role: Role,
    pub max_tokens: u64,
    pub temperature: f64,
    pub max_turns: usize,
}

/// Output of a single agent invocation. Mirrors the relevant fields of
/// `rig::agent::PromptResponse` without leaking the rig type into our
/// public surface.
#[derive(Debug, Clone, Default)]
pub struct DriveResponse {
    pub output: String,
    pub usage: TokenUsage,
}

/// Abstraction over "run an LLM agent for one (stage, role) call." The
/// production implementation wraps rig + OpenRouter; tests use a scripted
/// mock. The driver is responsible for invoking tools (registering them
/// with rig in production, or calling them directly in the mock); the
/// engine just hands it the params and the shared TaskCtx.
#[async_trait]
pub trait LlmDriver: Send + Sync {
    async fn drive(
        &self,
        params: DriveParams,
        ctx: Arc<TaskCtx>,
    ) -> Result<DriveResponse>;
}

/// Production driver backed by rig + OpenRouter.
pub struct OpenRouterDriver {
    client: openrouter::Client,
}

impl OpenRouterDriver {
    pub fn from_config(config: &Config) -> Result<Self> {
        let key_var = config
            .toml
            .provider
            .api_key_env
            .clone()
            .unwrap_or_else(|| "OPENROUTER_API_KEY".to_string());
        let key = std::env::var(&key_var)
            .map_err(|_| anyhow!("missing env var {key_var}"))?;
        let mut builder = openrouter::Client::builder().api_key(&key);
        if let Some(base) = &config.toml.provider.base_url {
            builder = builder.base_url(base);
        }
        let client = builder
            .build()
            .map_err(|e| anyhow!("openrouter client build: {e}"))?;
        Ok(Self { client })
    }
}

#[async_trait]
impl LlmDriver for OpenRouterDriver {
    async fn drive(
        &self,
        params: DriveParams,
        ctx: Arc<TaskCtx>,
    ) -> Result<DriveResponse> {
        let resp = run_rig_agent(
            &self.client,
            &params.model,
            &params.preamble,
            &params.user_prompt,
            params.stage,
            params.role,
            ctx,
            params.max_tokens,
            params.temperature,
            params.max_turns,
        )
        .await?;
        Ok(DriveResponse {
            output: resp.output,
            usage: TokenUsage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
                cached_input_tokens: resp.usage.cached_input_tokens,
                cache_creation_input_tokens: resp.usage.cache_creation_input_tokens,
            },
        })
    }
}

pub struct Engine {
    pub config: Arc<Config>,
    pub state: StateHandle,
    pub graph: Arc<Mutex<NodeGraph>>,
    pub workdir: PathBuf,
    pub layout: Layout,
    pub driver: Arc<dyn LlmDriver>,
    /// Serializes cargo invocations across parallel tasks.
    pub cargo_lock: Arc<tokio::sync::Mutex<()>>,
}

impl Engine {
    pub fn new(config: Arc<Config>, state: StateHandle) -> Result<Self> {
        let driver: Arc<dyn LlmDriver> = Arc::new(OpenRouterDriver::from_config(&config)?);
        Self::with_driver(config, state, driver)
    }

    pub fn with_driver(
        config: Arc<Config>,
        state: StateHandle,
        driver: Arc<dyn LlmDriver>,
    ) -> Result<Self> {
        let workdir = state.read(|s| s.workdir.clone());
        let layout = config.layout();
        let graph = Arc::new(Mutex::new(state.read(|s| s.graph.clone())));
        Ok(Self {
            config,
            state,
            graph,
            workdir,
            layout,
            driver,
            cargo_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        self.state.write(|s| {
            s.scheduler = SchedulerState::Running;
        });
        self.state.emit(UiEvent::SchedulerStateChanged {
            state: SchedulerState::Running,
        });

        // Bootstrap: if the graph is empty, seed it with a root node and
        // render the initial scaffold so cargo can start checking things.
        self.ensure_root_seeded()?;

        let max_parallel = self.config.toml.limits.max_parallel_tasks.max(1);
        let mut joinset: tokio::task::JoinSet<Result<()>> = tokio::task::JoinSet::new();
        let mut total_tasks = 0usize;
        let mut first_error: Option<anyhow::Error> = None;

        loop {
            self.sync_graph_to_state();

            // Cost cap
            if let Some(cap) = self.config.toml.limits.cost_cap_usd {
                let est = self.state.read(|s| s.estimated_cost_usd);
                if est >= cap {
                    self.note(format!(
                        "halting: cost cap ${cap:.2} reached at ${est:.4}"
                    ));
                    break;
                }
            }

            if total_tasks >= self.config.toml.limits.max_tasks_total {
                self.note(format!(
                    "halting: max_tasks_total ({}) reached",
                    self.config.toml.limits.max_tasks_total
                ));
                break;
            }

            // Fill the slot pool with anything ready and not already running.
            // `pick_next_ready` only returns NotStarted stages — once we
            // mark a stage InProgress, it won't be picked again.
            while joinset.len() < max_parallel {
                let Some((node_id, stage)) = self.pick_next_ready() else {
                    break;
                };
                // Mark InProgress eagerly so subsequent picks skip it.
                self.graph
                    .lock()
                    .get_mut(node_id)
                    .unwrap()
                    .stages
                    .set(stage, StageState::InProgress);
                self.sync_graph_to_state();
                let this = self.clone();
                joinset
                    .spawn(async move { this.advance_stage(node_id, stage).await });
                total_tasks += 1;
            }

            // If nothing is running and nothing was ready, we're either done
            // or stuck.
            if joinset.is_empty() {
                if self.all_done() {
                    self.note("pipeline complete");
                    self.state.write(|s| s.scheduler = SchedulerState::Done);
                    self.state.emit(UiEvent::SchedulerStateChanged {
                        state: SchedulerState::Done,
                    });
                    break;
                } else {
                    self.note("no ready stages and not done; halting (likely a stuck dep)");
                    self.state.write(|s| s.scheduler = SchedulerState::Stopped);
                    return Err(first_error.unwrap_or_else(|| {
                        anyhow!("scheduler stuck — no ready stages remain")
                    }));
                }
            }

            // Wait for one task to finish.
            let Some(joined) = joinset.join_next().await else {
                continue;
            };
            match joined {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(je) => {
                    tracing::error!("advance_stage join error: {je}");
                    if first_error.is_none() {
                        first_error = Some(anyhow!("join error: {je}"));
                    }
                }
            }
        }

        // Drain remaining tasks.
        while let Some(joined) = joinset.join_next().await {
            if let Err(je) = joined {
                tracing::error!("advance_stage join error during drain: {je}");
            }
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn ensure_root_seeded(&self) -> Result<()> {
        let mut g = self.graph.lock();
        if g.root.is_some() {
            return Ok(());
        }
        // Seed the root description from the first non-empty paragraph of
        // problem.md so the model has an immediate anchor of what it's
        // building. The full problem statement is also injected as a
        // "Project mission" section into every prompt context — see
        // `run_role`.
        let desc = problem_first_paragraph(&self.config.problem);
        let mut root = Node::new(self.config.toml.project_name.as_str(), desc);
        root.crate_boundary = true;
        let _ = g.insert_root(root)?;
        render::render_graph(&self.workdir, &g, self.layout)?;
        drop(g);
        // Initialize the git repo + initial scaffold commit so subsequent
        // stage commits have somewhere to land. If the repo already
        // exists (resumed run), this is a no-op.
        if let Err(e) = self.git_init_if_needed() {
            tracing::warn!("git init: {e:#}");
        }
        Ok(())
    }

    /// Initialize the workdir as a git repo and make an initial commit
    /// of the scaffolded files. No-op if the repo already exists.
    fn git_init_if_needed(&self) -> Result<()> {
        let dotgit = self.workdir.join(".git");
        if dotgit.exists() {
            return Ok(());
        }
        let repo = git2::Repository::init(&self.workdir)?;
        // Configure a local user so commits work even if the global git
        // config doesn't have one.
        let mut cfg = repo.config()?;
        if cfg.get_string("user.name").is_err() {
            cfg.set_str("user.name", "bureau-rs")?;
        }
        if cfg.get_string("user.email").is_err() {
            cfg.set_str("user.email", "bureau-rs@localhost")?;
        }
        // Stage everything currently rendered and make the seed commit.
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = repo.signature()?;
        repo.commit(Some("HEAD"), &sig, &sig, "scaffold", &tree, &[])?;
        Ok(())
    }

    /// Commit the workdir state after a stage completes. Each commit is
    /// keyed by node + stage so the gitlog panel shows a per-stage trail
    /// of progress. Non-fatal: if git fails (e.g. no .git dir, or
    /// nothing to commit), we just log and move on.
    fn commit_stage_done(
        &self,
        _node_id: NodeId,
        stage: Stage,
        node_name: &str,
    ) -> Result<()> {
        let repo = git2::Repository::open(&self.workdir)?;
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        // If the tree matches HEAD's tree, nothing changed — skip.
        let parent = match repo.head().and_then(|h| h.peel_to_commit()) {
            Ok(c) => Some(c),
            Err(_) => None, // unborn HEAD — first commit
        };
        if let Some(p) = &parent {
            if p.tree_id() == tree_id {
                return Ok(()); // no changes to record
            }
        }
        let sig = repo.signature().or_else(|_| {
            git2::Signature::now("bureau-rs", "bureau-rs@localhost")
        })?;
        let msg = format!("{stage}: {node_name}");
        let parents: Vec<&git2::Commit> = parent.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, &msg, &tree, &parents)?;
        Ok(())
    }

    fn sync_graph_to_state(&self) {
        let g = self.graph.lock().clone();
        self.state.write(|s| s.graph = g);
    }

    fn note(&self, msg: impl Into<String>) {
        let m: String = msg.into();
        tracing::info!("{m}");
        let entry = HistoryEntry {
            at: Utc::now(),
            message: m,
        };
        self.state.write(|s| s.history.push(entry.clone()));
        self.state.emit(UiEvent::HistoryAppended { entry });
    }

    /// Find the next (node, stage) whose dependencies are satisfied. Walks
    /// the graph in topological-by-deps order, then for each node picks the
    /// earliest stage that is `NotStarted` and whose preconditions hold.
    fn pick_next_ready(&self) -> Option<(NodeId, Stage)> {
        let g = self.graph.lock();
        let order = g.topo_order()?;
        // Walk in reverse-topo so deps come first; that means leaves first
        // at any given stage. But we want the OVERALL earliest stage that's
        // ready, which can sit anywhere. So scan all nodes for each stage
        // in order, picking the first match.
        for stage in [
            Stage::Spec,
            Stage::Iface,
            Stage::Tests,
            Stage::Impl,
            Stage::Debug,
            Stage::Opt,
        ] {
            for id in &order {
                if !stage_is_ready(&g, *id, stage) {
                    continue;
                }
                return Some((*id, stage));
            }
        }
        None
    }

    fn all_done(&self) -> bool {
        let g = self.graph.lock();
        for n in g.iter() {
            for s in [Stage::Spec, Stage::Iface, Stage::Tests, Stage::Impl] {
                if !n.stages.get(s).is_done() {
                    return false;
                }
            }
        }
        true
    }

    async fn advance_stage(self: &Arc<Self>, node_id: NodeId, stage: Stage) -> Result<()> {
        // Mark the stage InProgress immediately so concurrent picks (when we
        // add parallelism later) don't double-run.
        self.graph
            .lock()
            .get_mut(node_id)
            .unwrap()
            .stages
            .set(stage, StageState::InProgress);
        self.sync_graph_to_state();

        let max_retries = self.config.toml.limits.max_stage_retries;
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 1..=(max_retries + 1) {
            match self.run_one_attempt(node_id, stage, attempt).await {
                Ok(()) => {
                    let node_name = {
                        let mut g = self.graph.lock();
                        let n = g.get_mut(node_id).unwrap();
                        n.stages.set(stage, StageState::Done);
                        n.name.clone()
                    };
                    self.sync_graph_to_state();
                    // Snapshot the workdir into git as a node-stage commit
                    // so the gitlog panel shows progress one click at a
                    // time. Commit failures are non-fatal — we just log
                    // and continue.
                    if let Err(e) = self.commit_stage_done(node_id, stage, &node_name) {
                        tracing::warn!(
                            "git commit for {node_name} {stage}: {e:#}"
                        );
                    }
                    return Ok(());
                }
                Err(e) => {
                    self.note(format!(
                        "node `{}` stage `{}` attempt {}/{} failed: {:#}",
                        self.graph.lock().get(node_id).unwrap().name,
                        stage,
                        attempt,
                        max_retries + 1,
                        e
                    ));
                    last_err = Some(e);
                    if attempt > max_retries {
                        break;
                    }
                }
            }
        }

        // Out of attempts. Mark Failed; some stages can recover via Debug
        // stage on the same node, which the scheduler will pick up next.
        self.graph
            .lock()
            .get_mut(node_id)
            .unwrap()
            .stages
            .set(stage, StageState::Failed);
        self.sync_graph_to_state();
        let msg = format!(
            "node `{}` stage `{}` exhausted {} attempts",
            self.graph.lock().get(node_id).unwrap().name,
            stage,
            max_retries + 1
        );
        self.note(&msg);
        // For Impl stage, queue a Debug retry (handled by the scheduler
        // because Debug becomes ready when Impl is Failed). For other
        // stages, the failure is terminal — the pipeline halts.
        if stage == Stage::Impl {
            // Re-set to NotStarted? No — Failed is meaningful (Debug
            // stage's ready-check fires on Impl=Failed).
            return Ok(());
        }
        Err(last_err.unwrap_or_else(|| anyhow!("{msg}")))
    }

    /// One full attempt at a (node, stage): actor → optional critique cycle
    /// → optional cargo gate. Returns `Ok(())` if the stage is Done by the
    /// end. The cargo gate is what determines pass/fail when applicable.
    async fn run_one_attempt(
        self: &Arc<Self>,
        node_id: NodeId,
        stage: Stage,
        attempt: u32,
    ) -> Result<()> {
        let task_id = Uuid::new_v4();
        let node_name = self.graph.lock().get(node_id).unwrap().name.clone();
        let actor_model = self.config.toml.models.actor.clone();
        let task = EngineTask {
            id: task_id,
            node_id,
            node_name: node_name.clone(),
            stage,
            status: TaskStatus::Running,
            model: actor_model.clone(),
            transcript: Vec::new(),
            cost: TokenUsage::default(),
            started_at: Some(Utc::now()),
            finished_at: None,
            error: None,
            final_verdict: None,
            retries: attempt - 1,
        };
        self.state.write(|s| {
            s.tasks.insert(task_id, task.clone());
        });
        self.state.emit(UiEvent::TaskCreated { task });

        // 1. Actor
        let actor = self.run_role(task_id, node_id, stage, Role::Writer, None).await?;

        // 2. Critique cycle (optional)
        let critique_retries = self.config.toml.limits.critique_retries;
        let mut last_text = actor.text;
        let mut last_failed = actor.failed_tools;
        for round in 1..=critique_retries {
            let critique = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Critic,
                    Some(CycleExtras {
                        round,
                        prior_actor_text: Some(last_text.clone()),
                        prior_critique: None,
                        prior_revision: None,
                        prior_failed_tools: last_failed.clone(),
                    }),
                )
                .await?
                .text;
            let revision = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Reviser,
                    Some(CycleExtras {
                        round,
                        prior_actor_text: Some(last_text.clone()),
                        prior_critique: Some(critique.clone()),
                        prior_revision: None,
                        prior_failed_tools: last_failed.clone(),
                    }),
                )
                .await?;
            last_text = revision.text.clone();
            last_failed = revision.failed_tools.clone();
            // Judge
            let _judge = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Judge,
                    Some(CycleExtras {
                        round,
                        prior_actor_text: None,
                        prior_critique: Some(critique),
                        prior_revision: Some(revision.text),
                        prior_failed_tools: last_failed.clone(),
                    }),
                )
                .await?;
            let v = self.state.read(|s| {
                s.tasks.get(&task_id).and_then(|t| t.final_verdict.clone())
            });
            self.note(format!(
                "task {task_id} round {round}: verdict = {}",
                v.as_ref()
                    .map(|v| if v.satisfactory { "ok" } else { "needs work" })
                    .unwrap_or("(none)")
            ));
            if matches!(v, Some(JudgeVerdict { satisfactory: true, .. })) {
                break;
            }
        }

        // 3. Cargo gate (where applicable). The actor's tools already let the
        // model verify itself, but a final hard gate ensures the on-disk
        // state is good before we mark the stage Done.
        let gate_kind = match stage {
            Stage::Spec => None,
            Stage::Iface => Some(crate::gate::GateKind::Check),
            Stage::Tests => Some(crate::gate::GateKind::TestNoRun),
            Stage::Impl | Stage::Debug | Stage::Opt => Some(crate::gate::GateKind::Test),
        };
        if let Some(kind) = gate_kind {
            let outcome = {
                let _guard = self.cargo_lock.lock().await;
                crate::gate::run_gate(&self.workdir, kind).await?
            };
            if !outcome.passed {
                let summary = self.summarize_errors(&outcome.errors, 5);
                let msg = format!(
                    "cargo {} failed for node `{node_name}` stage `{stage}`:\n{summary}",
                    kind.label()
                );
                // Mark task Failed and bubble up the error so the
                // outer retry loop fires.
                self.state.write(|s| {
                    if let Some(t) = s.tasks.get_mut(&task_id) {
                        t.status = TaskStatus::Failed;
                        t.finished_at = Some(Utc::now());
                        t.error = Some(msg.clone());
                    }
                });
                self.state.emit(UiEvent::TaskStatusChanged {
                    id: task_id,
                    status: TaskStatus::Failed,
                });
                return Err(anyhow!(msg));
            }
        }

        // 4. Special-case: spec stage doesn't have a cargo gate, but we
        // require that node.spec_public_md is populated (otherwise the
        // writer didn't call submit_spec_public). Private spec is
        // optional.
        if stage == Stage::Spec {
            let g = self.graph.lock();
            let n = g.get(node_id).unwrap();
            if n.spec_public_md.is_none() {
                drop(g);
                let msg = format!("node `{node_name}` spec stage produced no public.md");
                self.state.write(|s| {
                    if let Some(t) = s.tasks.get_mut(&task_id) {
                        t.status = TaskStatus::Failed;
                        t.finished_at = Some(Utc::now());
                        t.error = Some(msg.clone());
                    }
                });
                self.state.emit(UiEvent::TaskStatusChanged {
                    id: task_id,
                    status: TaskStatus::Failed,
                });
                return Err(anyhow!(msg));
            }
        }

        // 5. Verdict gate: if critique cycle ran and final verdict is
        // unsatisfactory, fail.
        let final_v = self
            .state
            .read(|s| s.tasks.get(&task_id).and_then(|t| t.final_verdict.clone()));
        if let Some(v) = final_v {
            if !v.satisfactory {
                let msg = format!(
                    "judge rejected node `{node_name}` stage `{stage}`: {}",
                    v.reason
                );
                self.state.write(|s| {
                    if let Some(t) = s.tasks.get_mut(&task_id) {
                        t.status = TaskStatus::Failed;
                        t.finished_at = Some(Utc::now());
                        t.error = Some(msg.clone());
                    }
                });
                self.state.emit(UiEvent::TaskStatusChanged {
                    id: task_id,
                    status: TaskStatus::Failed,
                });
                return Err(anyhow!(msg));
            }
        }

        self.state.write(|s| {
            if let Some(t) = s.tasks.get_mut(&task_id) {
                t.status = TaskStatus::Done;
                t.finished_at = Some(Utc::now());
            }
        });
        self.state.emit(UiEvent::TaskStatusChanged {
            id: task_id,
            status: TaskStatus::Done,
        });
        Ok(())
    }

    fn summarize_errors(&self, errors: &[crate::gate::CompilerError], max: usize) -> String {
        if errors.is_empty() {
            return "(no errors recorded)".into();
        }
        let mut s = String::new();
        for e in errors.iter().take(max) {
            let loc = match (&e.file, e.line) {
                (Some(f), Some(l)) => format!("{}:{}", f.display(), l),
                (Some(f), None) => f.display().to_string(),
                _ => e.id.clone(),
            };
            let first = e.message.lines().next().unwrap_or("").trim();
            s.push_str(&format!("  - [{loc}] {first}\n"));
        }
        if errors.len() > max {
            s.push_str(&format!("  - ... and {} more\n", errors.len() - max));
        }
        s
    }

    /// Drive the LLM agent once, retrying internally on transient
    /// errors (network blips, 5xx, rate-limit) with exponential backoff.
    /// Non-transient errors propagate immediately.
    async fn drive_with_transient_retry(
        self: &Arc<Self>,
        params: DriveParams,
        ctx: Arc<TaskCtx>,
    ) -> Result<DriveResponse> {
        const MAX_TRANSIENT_RETRIES: u32 = 3;
        let mut attempt = 0u32;
        loop {
            let r = self.driver.drive(params.clone(), ctx.clone()).await;
            match r {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    let msg = format!("{:#}", e);
                    if attempt < MAX_TRANSIENT_RETRIES && is_transient(&msg) {
                        attempt += 1;
                        let backoff = 400u64 * (1 << (attempt - 1).min(3));
                        tracing::warn!(
                            "transient agent error (attempt {attempt}), retrying in {backoff}ms: {msg}"
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
    }

    async fn run_role(
        self: &Arc<Self>,
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        role: Role,
        extras: Option<CycleExtras>,
    ) -> Result<RoleOutcome> {
        let model = self.config.toml.models.for_role(role).to_string();

        // Build prompt context. Always inject the project mission as the
        // first section so every node — root or deep child — sees what
        // the overall goal is. Without this the root node has no idea what
        // it's building beyond its own name.
        let g_for_ctx = self.graph.lock().clone();
        let mut bundle = node_context::ContextBundle::new();
        bundle.push("Project mission", self.config.problem.trim().to_string());
        let inner = node_context::build_for_stage(
            &g_for_ctx,
            node_id,
            stage,
            self.config.toml.limits.max_nodes,
            self.config.toml.limits.max_node_depth,
        );
        bundle.extend_from(inner);
        if let Some(ex) = &extras {
            // Annotate with the cycle extras: append a "Critique round"
            // section.
            let mut cyc = String::new();
            cyc.push_str(&format!("Round {}\n\n", ex.round));
            if !ex.prior_failed_tools.is_empty() {
                // Surface failed tool calls FIRST and prominently — these
                // are the most important thing for the next role to act on,
                // because they represent intent that didn't land.
                cyc.push_str(
                    "## ⚠ Prior turn had failed tool calls — these were NOT applied\n\n\
                     The previous turn called these tools but they returned errors. \
                     The intent behind each call was lost. Address every entry: read \
                     the error, fix the args, and retry the call (or, if the goal \
                     was actually wrong, explain in your own output why it should be \
                     dropped).\n\n",
                );
                let args_cap = self.config.toml.limits.args_display_cap;
                for (tool, args, err) in &ex.prior_failed_tools {
                    let args_display = truncate_args_for_display(args, args_cap);
                    cyc.push_str(&format!(
                        "- **`{tool}`** — args: `{args_display}` — error: {err}\n"
                    ));
                }
                cyc.push_str("\n");
            }
            if let Some(t) = &ex.prior_actor_text {
                cyc.push_str("## Prior writer summary\n\n");
                cyc.push_str(t);
                cyc.push_str("\n\n");
            }
            if let Some(t) = &ex.prior_critique {
                cyc.push_str("## Critique\n\n");
                cyc.push_str(t);
                cyc.push_str("\n\n");
            }
            if let Some(t) = &ex.prior_revision {
                cyc.push_str("## Reviser summary\n\n");
                cyc.push_str(t);
                cyc.push_str("\n\n");
            }
            bundle.push("Critique cycle context", cyc);
        }
        drop(g_for_ctx);

        let prompt_limits = crate::tools::PromptLimits {
            max_file_lines: self.config.toml.limits.max_file_lines,
            max_spec_section_lines: self.config.toml.limits.max_spec_section_lines,
        };
        let preamble = role_preamble(stage, role, prompt_limits);
        let context_doc = bundle.to_markdown();
        let user_prompt = role_user_prompt(stage, role);
        let combined_preamble = format!("{preamble}\n\n{context_doc}");

        // Record system + tool-definitions + user prompts. Tool defs are
        // surfaced as a dedicated transcript entry so the UI can display
        // exactly what the model was told the tools do — this is part of
        // the input to every turn and is otherwise invisible.
        let now = Utc::now();
        let tool_defs = crate::tools::tool_definitions_for(stage, role, prompt_limits);
        let mut entries: Vec<(TranscriptKind, String)> = Vec::new();
        entries.push((TranscriptKind::System, combined_preamble.clone()));
        if !tool_defs.is_empty() {
            entries.push((
                TranscriptKind::ToolDefinitions {
                    tools: tool_defs.clone(),
                },
                serde_json::to_string_pretty(&tool_defs).unwrap_or_default(),
            ));
        }
        entries.push((TranscriptKind::UserPrompt, user_prompt.clone()));
        for (kind, content) in entries {
            let entry = TranscriptEntry {
                timestamp: now,
                kind,
                content,
                role: Some(role),
            };
            self.state.write(|s| {
                if let Some(t) = s.tasks.get_mut(&task_id) {
                    t.transcript.push(entry.clone());
                }
            });
            self.state.emit(UiEvent::TranscriptAppended {
                task_id,
                entry,
            });
        }

        // Construct TaskCtx and run with retry on transient errors.
        let ctx = Arc::new(TaskCtx::new(
            task_id,
            node_id,
            stage,
            role,
            self.graph.clone(),
            self.workdir.clone(),
            self.layout,
            self.config.toml.limits.max_file_lines,
            self.config.toml.limits.max_spec_section_lines,
            self.config.toml.limits.max_nodes,
            self.config.toml.limits.max_node_depth,
            self.cargo_lock.clone(),
        ));

        let params = DriveParams {
            model: model.clone(),
            preamble: combined_preamble.clone(),
            user_prompt: user_prompt.clone(),
            stage,
            role,
            max_tokens: self.config.toml.models.max_tokens,
            temperature: self.config.toml.models.temperature,
            max_turns: self.config.toml.models.max_turns,
        };
        let resp = self.drive_with_transient_retry(params, ctx.clone()).await?;

        // Aggregate usage and final-message text across the initial drive
        // and any forced-retry drives we fire below for unresolved tool
        // failures.
        let mut total_usage = resp.usage.clone();
        let mut combined_text = resp.output.clone();

        // Force-retry loop: if the model left tool calls in a FAILED state
        // and didn't fix them within its own agent loop, fire fresh
        // `drive()` invocations with a focused retry preamble until either
        // the failures clear or the budget is spent.
        let max_forced_retries = self.config.toml.limits.tool_retry_budget;
        for forced_attempt in 1..=max_forced_retries {
            let snapshot = ctx.transcript.lock().clone();
            let unresolved = collect_failed_tool_calls(&snapshot);
            if unresolved.is_empty() {
                break;
            }
            let remaining = max_forced_retries - forced_attempt;
            let retry_preamble_text = retry_preamble(
                role,
                stage,
                &unresolved,
                forced_attempt,
                remaining,
                self.config.toml.limits.args_display_cap,
            );
            let retry_user_prompt = "Resolve the failed tool calls listed in the system prompt: \
                 retry each with corrected args, or end your message with one sentence per \
                 abandoned call explaining why it isn't needed."
                .to_string();

            self.note(format!(
                "task {task_id} role {role:?}: forced retry {forced_attempt}/{max_forced_retries} for {} unresolved tool failure(s)",
                unresolved.len()
            ));

            // Record the retry's system + user prompts as transcript entries
            // so the UI shows what happened.
            let retry_now = Utc::now();
            for (kind, content) in [
                (TranscriptKind::System, retry_preamble_text.clone()),
                (TranscriptKind::UserPrompt, retry_user_prompt.clone()),
            ] {
                let entry = TranscriptEntry {
                    timestamp: retry_now,
                    kind,
                    content,
                    role: Some(role),
                };
                self.state.write(|s| {
                    if let Some(t) = s.tasks.get_mut(&task_id) {
                        t.transcript.push(entry.clone());
                    }
                });
                self.state.emit(UiEvent::TranscriptAppended {
                    task_id,
                    entry,
                });
            }

            let retry_params = DriveParams {
                model: model.clone(),
                preamble: retry_preamble_text,
                user_prompt: retry_user_prompt,
                stage,
                role,
                max_tokens: self.config.toml.models.max_tokens,
                temperature: self.config.toml.models.temperature,
                max_turns: self.config.toml.models.max_turns,
            };
            let retry_resp = self.drive_with_transient_retry(retry_params, ctx.clone()).await?;
            total_usage.add(&retry_resp.usage);
            combined_text.push_str(&format!(
                "\n\n---\n[forced retry {forced_attempt}]\n{}",
                retry_resp.output
            ));
        }

        let resp = DriveResponse {
            output: combined_text,
            usage: total_usage.clone(),
        };
        let usage = total_usage;

        // Final assistant message.
        let final_entry = TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::AssistantText,
            content: resp.output.clone(),
            role: Some(role),
        };
        // Drain ctx transcripts.
        let ctx_entries = ctx.transcript.lock().drain(..).collect::<Vec<_>>();
        // Surface failed tool calls from this turn — both as a
        // higher-severity log entry (so the operator notices) and as a
        // return value (so the cycle context tells the next role to
        // retry them).
        let failed_tools = collect_failed_tool_calls(&ctx_entries);
        for (tool, args, err) in &failed_tools {
            tracing::warn!(
                node = %node_id,
                stage = %stage,
                role = ?role,
                tool = %tool,
                args = %args,
                "tool call failed: {err}"
            );
        }
        // Drain verdict.
        let verdict = ctx.verdict.lock().take();
        // Drain fs events.
        let fs_events: Vec<PathBuf> = ctx.fs_events.lock().drain(..).collect();

        self.state.write(|s| {
            if let Some(t) = s.tasks.get_mut(&task_id) {
                t.transcript.extend(ctx_entries.iter().cloned());
                t.transcript.push(final_entry.clone());
                t.cost.add(&usage);
                if matches!(role, Role::Judge) {
                    t.final_verdict = verdict.clone();
                }
                s.total_cost.add(&usage);
                s.estimated_cost_usd = compute_total_cost(s);
            }
        });

        for entry in ctx_entries {
            self.state.emit(UiEvent::TranscriptAppended {
                task_id,
                entry,
            });
        }
        self.state.emit(UiEvent::TranscriptAppended {
            task_id,
            entry: final_entry,
        });
        for path in fs_events {
            self.state.emit(UiEvent::FileChanged { path });
        }
        self.state.emit(UiEvent::TaskCost {
            task_id,
            cost: usage.clone(),
            total: self.state.read(|s| s.total_cost.clone()),
            estimated_usd: self.state.read(|s| s.estimated_cost_usd),
        });

        // Sync the engine's working graph back to the EngineState's view.
        self.sync_graph_to_state();
        self.state.emit(UiEvent::NodeChanged { id: node_id });

        Ok(RoleOutcome {
            text: resp.output,
            failed_tools,
        })
    }
}

/// Output of a single role's turn within an writer → critic → reviser →
/// judge cycle. The text is the model's final assistant message; the
/// failed-tool list lets the next role's prompt context call out tool
/// calls that errored so the model can retry rather than silently move
/// on.
#[derive(Debug, Clone, Default)]
struct RoleOutcome {
    text: String,
    failed_tools: Vec<(String, String, String)>,
}

/// Returns true if the (node, stage) combination is ready to run right now.
/// "Ready" means:
/// - the stage's state is `NotStarted`
/// - all required preconditions hold (see below)
///
/// Stage preconditions:
/// - `Spec`: parent's `Spec` is Done (or there is no parent).
/// - `Iface`: this node's `Spec` is Done AND every dep's `Iface` is Done.
/// - `Tests`: this node's `Iface` is Done AND every dep's `Iface` is Done.
/// - `Impl`: this node's `Tests` is Done AND every dep's `Impl` is Done.
/// - `Debug`: this node's `Impl` is Failed (recovery slot).
/// - `Opt`: this node's `Impl` (or `Debug`) is Done; opt is optional and
///   only fires if explicitly configured. For now, opt is skipped — we
///   leave it `NotStarted` indefinitely so it doesn't block "all done".
fn stage_is_ready(graph: &NodeGraph, id: NodeId, stage: Stage) -> bool {
    let n = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };
    let cur = n.stages.get(stage);
    match stage {
        Stage::Opt => false, // Skip opt for now; see comment above.
        Stage::Spec => {
            if cur != StageState::NotStarted {
                return false;
            }
            match n.parent {
                None => true,
                Some(p) => graph
                    .get(p)
                    .map(|pn| pn.stages.spec.is_done())
                    .unwrap_or(false),
            }
        }
        Stage::Iface => {
            if cur != StageState::NotStarted {
                return false;
            }
            if !n.stages.spec.is_done() {
                return false;
            }
            for dep in &n.deps {
                if !graph
                    .get(*dep)
                    .map(|d| d.stages.iface.is_done())
                    .unwrap_or(false)
                {
                    return false;
                }
            }
            true
        }
        Stage::Tests => {
            if cur != StageState::NotStarted {
                return false;
            }
            if !n.stages.iface.is_done() {
                return false;
            }
            for dep in &n.deps {
                if !graph
                    .get(*dep)
                    .map(|d| d.stages.iface.is_done())
                    .unwrap_or(false)
                {
                    return false;
                }
            }
            true
        }
        Stage::Impl => {
            if cur != StageState::NotStarted {
                return false;
            }
            if !n.stages.tests.is_done() {
                return false;
            }
            for dep in &n.deps {
                if !graph
                    .get(*dep)
                    .map(|d| d.stages.impl_.is_done())
                    .unwrap_or(false)
                {
                    return false;
                }
            }
            // Parent-child as an implicit ordering: every child's impl must
            // be Done before the parent's impl runs. Otherwise the parent's
            // `cargo test` gate runs the WHOLE crate's tests, including
            // children's tests that panic at todo!() bodies. Bottom-up by
            // construction.
            for child in graph.children_of(id) {
                if !child.stages.impl_.is_done() {
                    return false;
                }
            }
            true
        }
        Stage::Debug => {
            // Debug fires only when Impl Failed; same parent-child ordering.
            if !(n.stages.impl_ == StageState::Failed && cur == StageState::NotStarted) {
                return false;
            }
            for child in graph.children_of(id) {
                if !child.stages.impl_.is_done() && child.stages.impl_ != StageState::Failed {
                    return false;
                }
            }
            true
        }
    }
}

#[derive(Debug, Clone)]
struct CycleExtras {
    round: u32,
    prior_actor_text: Option<String>,
    prior_critique: Option<String>,
    prior_revision: Option<String>,
    /// Tool calls that failed during the prior actor (or reviser) turn,
    /// surfaced to the critic and the next reviser so they're not lost.
    /// Each entry: (tool_name, args_json, error_msg).
    prior_failed_tools: Vec<(String, String, String)>,
}

/// Scan a slice of transcript entries and return UNRESOLVED tool failures —
/// for each tool name, only the LAST result is considered, so a model that
/// failed once and then retried successfully reports nothing. Each tuple
/// is `(tool, args, error)`. Args are paired by walking back to the most
/// recent matching `ToolCall`.
fn collect_failed_tool_calls(entries: &[TranscriptEntry]) -> Vec<(String, String, String)> {
    use std::collections::HashMap;
    // tool name → (transcript_index, ok, args, error)
    let mut last: HashMap<String, (usize, bool, String, String)> = HashMap::new();
    for (i, e) in entries.iter().enumerate() {
        if let TranscriptKind::ToolResult { tool, ok, error, .. } = &e.kind {
            let args = entries[..i]
                .iter()
                .rev()
                .find_map(|p| match &p.kind {
                    TranscriptKind::ToolCall { tool: t2 } if t2 == tool => Some(p.content.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            last.insert(
                tool.clone(),
                (
                    i,
                    *ok,
                    args,
                    error.clone().unwrap_or_else(|| "(no error message)".into()),
                ),
            );
        }
    }
    let mut failures: Vec<(usize, String, String, String)> = last
        .into_iter()
        .filter_map(|(tool, (idx, ok, args, err))| if ok { None } else { Some((idx, tool, args, err)) })
        .collect();
    failures.sort_by_key(|(idx, ..)| *idx);
    failures.into_iter().map(|(_, t, a, e)| (t, a, e)).collect()
}

/// Truncate an args string for display in retry/critique preambles. Keeps
/// the boundary clearly marked so the model doesn't lose track of where
/// args end and prose resumes.
fn truncate_args_for_display(args: &str, max: usize) -> String {
    if args.len() <= max {
        return args.to_string();
    }
    let mut end = max;
    while end > 0 && !args.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}…  [TRUNCATED — {} bytes total; the args you sent are not echoed back in full]",
        &args[..end],
        args.len()
    )
}

/// Build the focused system prompt for a forced-retry attempt. Lists the
/// unresolved failures with truncated args and tells the model exactly
/// what to do.
fn retry_preamble(
    role: Role,
    stage: Stage,
    failures: &[(String, String, String)],
    attempt: u32,
    remaining: u32,
    args_cap: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# RETRY · {stage} · {role:?} (attempt {attempt}, {remaining} retries remaining after this)\n\n"
    ));
    s.push_str(
        "Your previous turn left tool calls in a FAILED state. The framework will not \
         accept this stage as complete until each is either retried successfully or \
         explicitly abandoned with a reason in your final message. Process every \
         failure below.\n\n",
    );
    s.push_str(
        "For each failed call you must do ONE of:\n\
         1. **Retry** the same tool with corrected arguments. Read the error message — \
            it tells you exactly what's wrong.\n\
         2. **Abandon** the call and explain in your end-of-turn message why it's not \
            actually needed (one sentence per abandoned call).\n\n",
    );
    s.push_str(
        "Note on truncated args: the args you sent are shown only as a stub for \
         identification. For `submit_*` tools, do NOT try to reconstruct the truncated \
         text from the stub — re-derive the full content from the spec / dep ifaces / \
         tests in the context document, then submit it fresh.\n\n",
    );
    s.push_str("## Failed calls to address\n\n");
    for (i, (tool, args, err)) in failures.iter().enumerate() {
        let args_display = truncate_args_for_display(args, args_cap);
        s.push_str(&format!(
            "{}. **`{}`** — error: {}\n   args: `{}`\n\n",
            i + 1,
            tool,
            err,
            args_display
        ));
    }
    s
}

#[allow(clippy::too_many_arguments)]
async fn run_rig_agent(
    client: &openrouter::Client,
    model: &str,
    preamble: &str,
    user_prompt: &str,
    stage: Stage,
    role: Role,
    ctx: Arc<TaskCtx>,
    max_tokens: u64,
    temperature: f64,
    max_turns: usize,
) -> Result<rig::agent::PromptResponse> {
    let base = client
        .agent(model)
        .preamble(preamble)
        .max_tokens(max_tokens)
        .temperature(temperature)
        .default_max_turns(max_turns.max(2));

    // Branch on (stage, role) to register the right tool set. The catalog
    // in `tools::tool_names_for` is the source of truth for "which tools";
    // we mirror it here to actually instantiate them.
    let resp = match (stage, role) {
        (Stage::Spec, Role::Writer) | (Stage::Spec, Role::Reviser) => {
            base.tool(SubmitSpecTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Spec, Role::Critic) => {
            base.build().prompt(user_prompt).extended_details().await?
        }
        (Stage::Spec, Role::Judge) => {
            base.tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }

        (Stage::Iface, Role::Writer) | (Stage::Iface, Role::Reviser) => {
            base.tool(SubmitPublicTool { ctx: ctx.clone() })
                .tool(SubmitPrivateTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Iface, Role::Critic) => {
            base.tool(CargoCheckTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Iface, Role::Judge) => {
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }

        (Stage::Tests, Role::Writer) | (Stage::Tests, Role::Reviser) => {
            base.tool(SubmitTestsTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestNoRunTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Tests, Role::Critic) => {
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestNoRunTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Tests, Role::Judge) => {
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestNoRunTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }

        (Stage::Impl, Role::Writer) | (Stage::Impl, Role::Reviser) => {
            base.tool(SubmitPrivateTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Impl, Role::Critic) => {
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Impl, Role::Judge) => {
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }

        (Stage::Debug, Role::Writer) | (Stage::Debug, Role::Reviser) => {
            base.tool(SubmitPrivateTool { ctx: ctx.clone() })
                .tool(SubmitTestsTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Debug, Role::Critic) => {
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Debug, Role::Judge) => {
            base.tool(CargoTestTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }

        (Stage::Opt, Role::Writer) | (Stage::Opt, Role::Reviser) => {
            base.tool(SubmitPrivateTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Opt, Role::Critic) => {
            base.tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Opt, Role::Judge) => {
            base.tool(CargoTestTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
    };
    Ok(resp)
}

fn role_preamble(stage: Stage, role: Role, limits: crate::tools::PromptLimits) -> String {
    let max_file = limits.max_file_lines;
    let max_spec = limits.max_spec_section_lines;
    let common = "\
You are an expert Rust software engineer participating in a hierarchical \
decomposition pipeline. The framework owns the project structure, the file \
layout, and the dependency graph; you fill in slots through the tools \
listed for this turn — never through free-form file writes. The context \
document that follows starts with **Project mission**: read it first and \
treat it as ground truth for what's being built. Subsequent sections give \
you ancestor specs, sibling specs, dep public interfaces, and the current \
node's already-authored slots.\n\n\
# Universal rules\n\
- The tool list provided this turn is exhaustive. Call only those tools; \
  ignore patterns from other stages.\n\
- When a tool returns `no_change: true`, the file already had identical \
  content. Move on; do not re-call it.\n\
- Same tool + same args three times in a row triggers a hard error. When \
  you see that, finish with a one-line summary and stop calling tools.\n\
- All node names are **snake_case Rust identifiers**. CamelCase is for \
  Rust types, not nodes — never reference a sibling/dep as CamelCase.";

    let role_block = match (stage, role) {
        // ---- SPEC ----
        (Stage::Spec, Role::Writer) => format!(
            "# SPEC · WRITER\n\
            \n\
            You're writing a SPECIFICATION DOCUMENT for one piece of \
            software (a node in the project's decomposition tree). The \
            spec describes what the software DOES and PROMISES — it is \
            NOT a record of your own work, your own goals, or your own \
            editing process. Audience: a Rust engineer reading the spec \
            in isolation, six months from now, deciding how to use the \
            node.\n\
            \n\
            ONE call: `submit_spec`. Composite tool carrying public \
            spec (required), optional private notes, optional children, \
            optional deps. After it succeeds, end your turn with a \
            one-line summary.\n\
            \n\
            Read the **Project mission** AND the **Decomposition \
            budget** sections of the context document FIRST. The budget \
            tells you whether the schema for this turn even includes a \
            `children` field — if it doesn't (cap exhausted), you're \
            writing a leaf spec, full stop.\n\
            \n\
            ## What the spec is NOT\n\
            \n\
            It is NOT a literate-Rust artifact. Specs are ARCHITECTURE \
            and REQUIREMENTS, not code:\n\
            - DON'T write Rust traits with method signatures. Describe \
              capabilities in prose: \"the node provides a way to \
              authenticate a user given credentials and a session \
              context\" — NOT `pub trait Authenticator {{ fn auth(...) \
              -> Result<...>; }}`. The iface stage writes the Rust.\n\
            - DON'T enumerate every type and method. Name a few central \
              concepts; let the iface stage flesh them out.\n\
            - DO talk about: data shapes, ownership, concurrency, \
              error model, key invariants, security/threat model, \
              I/O surfaces, operational properties.\n\
            \n\
            Also NOT in the spec: meta-commentary about your own \
            writing (`This spec defines…`, `In this revision…`, \
            `Summary of addressed critique…`), process narrative \
            (`Next steps`, `Deliverables…`), or anything that reads as \
            a status report or PR description.\n\
            \n\
            ## `public` (REQUIRED, ≤{max_spec} lines)\n\
            \n\
            The INTERFACE specification — what dependents and downstream \
            stages observe. Think of this like a public header file's \
            documentation, but in prose. Suggested headings:\n\
            - `## What it does` — one or two sentences naming the \
              capability the node provides. (Avoid the word \"goal\" — \
              describe behaviour, not aspiration.)\n\
            - `## Public surface` — the named abstractions dependents \
              will see (e.g. \"a `Session` handle that owns the \
              underlying transport; a `Request`/`Response` pair that \
              models one round-trip\"). Prose, not Rust signatures.\n\
            - `## Invariants and guarantees` — properties dependents \
              can rely on (e.g. \"`Session` is `Send + Sync`\"; \"every \
              request is signed before transmission\").\n\
            - `## Out of scope` — adjacent things this node \
              deliberately does NOT do.\n\
            \n\
            CRITICAL — what counts as PUBLIC: only what callers of this \
            node observe. If a type is purely internal — backends the \
            user picks among, helper structs, configuration plumbing \
            that callers never instantiate — it goes in `private`, NOT \
            `public`. Rule of thumb: if removing it from the public \
            spec wouldn't change how a dependent uses the node, it \
            doesn't belong there.\n\
            \n\
            ## `private` (OPTIONAL, ≤{max_spec} lines)\n\
            \n\
            The IMPLEMENTATION specification — guidance for the iface / \
            impl stages on THIS node about HOW it's built. Audience: \
            YOU and your future selves doing the iface and impl stages \
            on this node. Other nodes never see this content.\n\
            \n\
            DO include:\n\
            - Internal data structures and their relationships.\n\
            - Backends, helpers, internal types — anything observable \
              only inside the node.\n\
            - Concurrency / threading / state-machine sketches.\n\
            - Algorithmic notes, performance considerations.\n\
            - Tradeoffs you considered, alternatives rejected.\n\
            \n\
            DO NOT include:\n\
            - A changelog of edits you made (`Rationale for edits…`, \
              `I expanded section X…`). The private spec describes the \
              SOFTWARE's internals, not the document's editing history. \
              That goes in your end-of-turn summary, OUTSIDE the \
              `submit_spec` call.\n\
            - Re-statement of the public spec.\n\
            \n\
            ## `children` (OPTIONAL — schema may hide this field)\n\
            \n\
            The DEFAULT for any node is LEAF (no children). Decompose \
            only when:\n\
            - The node truly has multiple separable sub-responsibilities \
              that can't fit in one Rust file (per-file cap {max_file} \
              lines is your sanity check), AND\n\
            - The Decomposition budget says you have room.\n\
            \n\
            Project-scale roots almost always decompose. Interior nodes \
            usually shouldn't. One-trait-per-node is wrong: if you'd \
            want one child per trait, the parent IS the leaf and the \
            traits sit in its `public.rs`.\n\
            \n\
            For each child: snake_case `name` (NOT CamelCase — that's \
            a type), one-sentence `description`, optional `deps` \
            (existing names or earlier siblings in this same call), \
            optional `crate_boundary` (default false; set true ONLY at \
            major top-level subsystem boundaries — most children should \
            leave it false and become modules within the parent's crate).\n\
            \n\
            Be careful with cross-crate `deps`: if children A and B are \
            in DIFFERENT crates and A.deps includes something in B's \
            crate while another node in B's crate depends on something \
            in A's crate, you've created a cycle that cargo will \
            reject. Keep cross-crate deps acyclic — typically arrange \
            them as a DAG with shared utilities at the bottom.\n\
            \n\
            ## `deps` (OPTIONAL)\n\
            \n\
            Names of existing graph nodes that THIS node should depend \
            on. For declaring that this node uses an existing utility \
            without creating any children. Cycle-checked at submit time \
            — both at the node level AND the crate level."
        ),
        (Stage::Spec, Role::Reviser) => format!(
            "# SPEC · REVISER\n\
            \n\
            The writer wrote the spec; the critic raised points. Apply \
            minimal targeted edits and re-call `submit_spec` with the \
            WHOLE updated submission — public (≤{max_spec} lines, \
            required), and whichever of private/children/deps the \
            critic flagged. ONE composite call.\n\
            \n\
            ## Critical: BOTH public AND private stay clean specs\n\
            \n\
            Neither slot is a diff, a PR description, or a changelog. \
            Do NOT write `Rationale for edits`, `I expanded the public \
            spec`, `Summary of addressed critique`, `In this revision`, \
            `These changes address…`, or ANY meta-narrative about your \
            editing process — not in `public`, and ALSO NOT in \
            `private`.\n\
            \n\
            Specifically:\n\
            - `public` describes what the SOFTWARE does and exposes to \
              dependents. Snapshot, not history.\n\
            - `private` describes what the SOFTWARE looks like \
              INTERNALLY (data structures, concurrency, algorithms, \
              tradeoffs). Snapshot, not history. Note: the most \
              common reviser mistake is writing change-rationale here \
              — don't do that. If the previous private content needs \
              updating, REWRITE it as a clean snapshot of the \
              implementation rationale; don't append diff notes.\n\
            \n\
            A reader two months from now should not be able to tell \
            which round of revision they're looking at, in either slot.\n\
            \n\
            Your end-of-turn assistant text (OUTSIDE the `submit_spec` \
            call) is the ONLY place where you describe what you \
            changed. One short paragraph there."
        ),
        (Stage::Spec, Role::Critic) => {
            "# SPEC · CRITIC\n\
            \n\
            Read the writer's spec. Bullet-list concrete problems: missing \
            sections, vague invariants, scope creep, decomposition that \
            doesn't match the project mission, child names that aren't \
            snake_case. If the spec is fine, output exactly \
            `No issues found.` Don't pad. Don't restate the spec."
                .to_string()
        }
        (Stage::Spec, Role::Judge) => judge_block(Stage::Spec),

        // ---- IFACE ----
        (Stage::Iface, Role::Writer) | (Stage::Iface, Role::Reviser) => format!(
            "# IFACE · WRITER\n\
            \n\
            Author the public surface and a stub private impl for this \
            node. The exact contract for each tool is in the tool list; \
            this preamble covers WORKFLOW.\n\
            \n\
            Workflow:\n\
            1. Submit `public.rs` (declarations only — see the \
               `submit_public` tool spec for what's allowed; in \
               particular `mod`, `impl`, and `fn` outside trait decls \
               are FORBIDDEN; cap {max_file} lines).\n\
            2. Submit `private.rs` containing one `impl Trait for \
               Newtype` block per trait in `public.rs`, with method \
               bodies as `todo!()`. The stubs let dependents compile \
               NOW; the next stage replaces them with real logic.\n\
            3. Run `cargo_check` to verify, then end with a one-line \
               summary.\n\
            \n\
            ## Module-path rules in `private.rs`\n\
            \n\
            - For your OWN public types: `use super::public::*;` — NEVER \
              `use crate::TypeName`.\n\
            - For a DECLARED DEP: copy the `import as ...` line from the \
              dep's context section verbatim.\n\
            - The first segment after `crate::` MUST resolve to a \
              declared dep, an ancestor, an own child, or this node \
              itself; the validator rejects anything else.\n\
            - Never invent a dep. If something you need isn't in the \
              context, mention it in your summary — don't paper over it.\n\
            \n\
            If this node has children (visible in the graph overview), \
            it's an UMBRELLA — `public.rs` can be just doc comments or \
            empty; the children carry the real surface."
        ),
        (Stage::Iface, Role::Critic) => {
            "# IFACE · CRITIC\n\
            \n\
            Use `cargo_check` to verify the iface compiles. Bullet-list \
            problems: forbidden items in `public.rs`, missing `impl` stubs \
            in `private.rs`, mismatch between trait signatures and the \
            spec's API section, undeclared dep imports. If clean, say \
            exactly `No issues found.`"
                .to_string()
        }
        (Stage::Iface, Role::Judge) => judge_block(Stage::Iface),

        // ---- TESTS ----
        (Stage::Tests, Role::Writer) | (Stage::Tests, Role::Reviser) => format!(
            "# TESTS · WRITER\n\
            \n\
            Author `#[test]` functions exercising this node's public \
            surface against the spec. The framework wraps your content \
            in a `#[cfg(test)] mod tests {{ ... }}` block. The exact \
            contract for `submit_tests` is in its tool spec.\n\
            \n\
            Workflow:\n\
            1. Import the node's public surface with `use \
               super::public::*;` (NEVER `use crate::TypeName`).\n\
            2. Cover the spec's invariants and edge cases.\n\
            3. Run `cargo_test_no_run` to verify the file compiles.\n\
            4. End with a one-line summary.\n\
            \n\
            Cap: {max_file} lines. Tests will COMPILE because \
            `private.rs` has `todo!()` stubs satisfying the trait at the \
            type level — they FAIL at runtime, which is expected. The \
            next stage replaces the stubs and the same tests pass.\n\
            \n\
            `use crate::<X>::...` rule same as `private.rs`: X must be a \
            declared dep / ancestor / own child. Don't write integration \
            tests that need network or filesystem unless the spec calls \
            for it."
        ),
        (Stage::Tests, Role::Critic) => {
            "# TESTS · CRITIC\n\
            \n\
            Use `cargo_test_no_run` to confirm tests compile. Bullet-list \
            problems: tests that don't actually exercise the spec, tests \
            that import via `crate::TypeName` instead of \
            `super::public::*`, missing edge-case coverage. If clean, \
            `No issues found.`"
                .to_string()
        }
        (Stage::Tests, Role::Judge) => judge_block(Stage::Tests),

        // ---- IMPL ----
        (Stage::Impl, Role::Writer) | (Stage::Impl, Role::Reviser) => format!(
            "# IMPL · WRITER\n\
            \n\
            Replace the `todo!()` bodies in `private.rs` with real \
            implementations that make the tests pass. The public surface \
            is FROZEN (don't touch it) and so are the tests (they define \
            the contract). `submit_private` replaces the WHOLE file; cap \
            {max_file} lines.\n\
            \n\
            Module-path rules same as iface: `use super::public::*;` for \
            own types; copy the `import as ...` line from each Dependency \
            section verbatim for declared deps; never invent a dep.\n\
            \n\
            Use `cargo_test` to confirm tests pass; `cargo_check` and \
            `cargo_clippy` for early signal. End with a one-line summary."
        ),
        (Stage::Impl, Role::Critic) => {
            "# IMPL · CRITIC\n\
            \n\
            Run `cargo_test`. Bullet-list failing tests, lints with \
            obvious correctness implications, and any `unsafe` or \
            `unwrap()` smell that the spec didn't sanction. If green, \
            `No issues found.`"
                .to_string()
        }
        (Stage::Impl, Role::Judge) => judge_block(Stage::Impl),

        // ---- DEBUG ----
        (Stage::Debug, Role::Writer) | (Stage::Debug, Role::Reviser) => format!(
            "# DEBUG · WRITER\n\
            \n\
            Tests are still failing after the previous stage. Look at \
            the failing-test output (in the `Critique cycle context` \
            section below, or run `cargo_test` yourself). Apply MINIMAL \
            targeted fixes via `submit_private` (≤ {max_file} lines) \
            and, only if a test was actually wrong, `submit_tests`. \
            Don't redesign. The public surface is still frozen."
        ),
        (Stage::Debug, Role::Critic) => {
            "# DEBUG · CRITIC\n\
            \n\
            Run `cargo_test`. Are tests green? Bullet-list anything \
            still failing or any test that was loosened to make impl \
            pass. If clean, `No issues found.`"
                .to_string()
        }
        (Stage::Debug, Role::Judge) => judge_block(Stage::Debug),

        // ---- OPT ----
        (Stage::Opt, Role::Writer) | (Stage::Opt, Role::Reviser) => format!(
            "# OPT · WRITER\n\
            \n\
            Optional polish. Improve clarity, performance, or lint \
            cleanliness in `private.rs` (≤ {max_file} lines) without \
            breaking tests. Use `cargo_test` to confirm tests still \
            pass; `cargo_clippy` for lints."
        ),
        (Stage::Opt, Role::Critic) => {
            "# OPT · CRITIC\n\
            \n\
            Run `cargo_test` and `cargo_clippy`. If anything regressed \
            or was made worse, bullet it. Otherwise `No issues found.`"
                .to_string()
        }
        (Stage::Opt, Role::Judge) => judge_block(Stage::Opt),
    };

    format!("{common}\n\n{role_block}")
}

fn judge_block(stage: Stage) -> String {
    let upper = match stage {
        Stage::Spec => "SPEC",
        Stage::Iface => "IFACE",
        Stage::Tests => "TESTS",
        Stage::Impl => "IMPL",
        Stage::Debug => "DEBUG",
        Stage::Opt => "OPT",
    };
    format!(
        "# {upper} · JUDGE\n\
        \n\
        Coherence check at the end of the writer → critic → reviser \
        cycle. Confirm the reviser addressed each critic point. You are \
        NOT a fresh reviewer and you are NOT the cargo gate (it runs \
        separately).\n\
        \n\
        For each critic bullet, decide: addressed / deferred-with-good- \
        reason / ignored. Call `submit_verdict` exactly once: \
        `satisfactory: true` if all points are addressed (or there were \
        no points); `satisfactory: false` with a concrete reason quoting \
        the unaddressed point(s). When in doubt: `satisfactory: true`."
    )
}

/// Extract a one-line description from a markdown problem statement: the
/// first non-blank, non-heading paragraph (joined to a single line, trimmed
/// to ~200 chars). Falls back to the first non-blank line.
fn problem_first_paragraph(md: &str) -> String {
    let mut buf = String::new();
    for line in md.lines() {
        let t = line.trim();
        if t.is_empty() {
            if !buf.is_empty() {
                break;
            }
            continue;
        }
        if t.starts_with('#') {
            // Skip headings until we hit prose.
            continue;
        }
        if !buf.is_empty() {
            buf.push(' ');
        }
        buf.push_str(t);
    }
    if buf.is_empty() {
        // Fallback: first non-blank line, even if it's a heading.
        for line in md.lines() {
            let t = line.trim().trim_start_matches('#').trim();
            if !t.is_empty() {
                buf = t.to_string();
                break;
            }
        }
    }
    if buf.is_empty() {
        return "Project root.".to_string();
    }
    if buf.len() > 200 {
        let mut end = 197;
        while !buf.is_char_boundary(end) {
            end -= 1;
        }
        buf.truncate(end);
        buf.push_str("...");
    }
    buf
}

fn role_user_prompt(stage: Stage, role: Role) -> String {
    match (stage, role) {
        (s, Role::Writer) => format!(
            "Do the {s} stage for this node using the slot-filler tool(s). End with a one-line \
             summary."
        ),
        (s, Role::Critic) => format!(
            "Critique the writer's {s}-stage output. Bullet list. Empty list = `No issues found.`"
        ),
        (s, Role::Reviser) => format!(
            "Address each critic point for the {s} stage. End with a one-line summary of the changes."
        ),
        (s, Role::Judge) => format!(
            "Verify the reviser addressed each critic point for the {s} stage. Call \
             submit_verdict exactly once."
        ),
    }
}

fn is_transient(msg: &str) -> bool {
    msg.contains("no message or tool call")
        || msg.contains("ResponseError")
        || msg.contains("connection reset")
        || msg.contains("connection closed")
        || msg.contains("timed out")
        || msg.contains("timeout")
        || msg.contains("temporarily unavailable")
        || msg.contains("502")
        || msg.contains("503")
        || msg.contains("504")
        || msg.contains("429")
        || msg.contains("ECONNRESET")
}

fn compute_total_cost(state: &EngineState) -> f64 {
    // Heuristic per-token pricing. The model can override via config later.
    let p_in_default = 1.0;
    let p_out_default = 3.0;
    let mut total = 0.0;
    for t in state.tasks.values() {
        let model = &t.model;
        let p_in = price_in(model).unwrap_or(p_in_default);
        let p_out = price_out(model).unwrap_or(p_out_default);
        let billable_in = t.cost.input_tokens.saturating_sub(t.cost.cached_input_tokens) as f64
            + t.cost.cache_creation_input_tokens as f64 * 1.25;
        let cached = t.cost.cached_input_tokens as f64 * 0.1;
        let inp = (billable_in + cached) * p_in / 1_000_000.0;
        let out = t.cost.output_tokens as f64 * p_out / 1_000_000.0;
        total += inp + out;
    }
    total
}

fn price_in(model: &str) -> Option<f64> {
    let m = model.to_ascii_lowercase();
    if m.contains("opus") {
        Some(15.0)
    } else if m.contains("sonnet") {
        Some(3.0)
    } else if m.contains("haiku") {
        Some(1.0)
    } else if m.contains("gpt-4o-mini") || m.contains("4o-mini") {
        Some(0.15)
    } else if m.contains("gpt-4o") {
        Some(2.5)
    } else if m.contains("gpt-5-mini") {
        Some(0.25)
    } else if m.contains("qwen3-coder") {
        Some(0.2)
    } else if m.contains("nemotron") {
        Some(0.4)
    } else if m.contains("deepseek") {
        Some(0.3)
    } else {
        None
    }
}

fn price_out(model: &str) -> Option<f64> {
    let m = model.to_ascii_lowercase();
    if m.contains("opus") {
        Some(75.0)
    } else if m.contains("sonnet") {
        Some(15.0)
    } else if m.contains("haiku") {
        Some(5.0)
    } else if m.contains("gpt-4o-mini") || m.contains("4o-mini") {
        Some(0.6)
    } else if m.contains("gpt-4o") {
        Some(10.0)
    } else if m.contains("gpt-5-mini") {
        Some(2.0)
    } else if m.contains("qwen3-coder") {
        Some(0.8)
    } else if m.contains("nemotron") {
        Some(1.6)
    } else if m.contains("deepseek") {
        Some(1.2)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;

    #[test]
    fn stage_is_ready_root_spec_starts() {
        let mut g = NodeGraph::new();
        let _root = g.insert_root(Node::new("app", "")).unwrap();
        assert!(stage_is_ready(&g, g.root.unwrap(), Stage::Spec));
        assert!(!stage_is_ready(&g, g.root.unwrap(), Stage::Iface));
    }

    #[test]
    fn child_spec_waits_on_parent_spec() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let c = g.add_child(root, Node::new("c", "")).unwrap();
        // Root spec NotStarted → child spec NOT ready.
        assert!(!stage_is_ready(&g, c, Stage::Spec));
        g.get_mut(root).unwrap().stages.spec = StageState::Done;
        assert!(stage_is_ready(&g, c, Stage::Spec));
    }

    #[test]
    fn iface_waits_on_dep_iface() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let b = g.add_child(root, Node::new("b", "")).unwrap();
        g.add_dep(a, b).unwrap();
        // Set everyone's spec Done.
        for id in [root, a, b] {
            g.get_mut(id).unwrap().stages.spec = StageState::Done;
        }
        // a's iface waits on b's iface.
        assert!(!stage_is_ready(&g, a, Stage::Iface));
        // b can start.
        assert!(stage_is_ready(&g, b, Stage::Iface));
        g.get_mut(b).unwrap().stages.iface = StageState::Done;
        assert!(stage_is_ready(&g, a, Stage::Iface));
    }

    #[test]
    fn tests_waits_on_own_iface_done() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        g.get_mut(root).unwrap().stages.spec = StageState::Done;
        assert!(!stage_is_ready(&g, root, Stage::Tests));
        g.get_mut(root).unwrap().stages.iface = StageState::Done;
        assert!(stage_is_ready(&g, root, Stage::Tests));
    }

    #[test]
    fn impl_waits_on_tests_and_dep_impls() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("a", "")).unwrap();
        let b = g.add_child(root, Node::new("b", "")).unwrap();
        g.add_dep(a, b).unwrap();
        for id in [root, a, b] {
            let s = &mut g.get_mut(id).unwrap().stages;
            s.spec = StageState::Done;
            s.iface = StageState::Done;
            s.tests = StageState::Done;
        }
        // a's impl waits on b's impl.
        assert!(!stage_is_ready(&g, a, Stage::Impl));
        assert!(stage_is_ready(&g, b, Stage::Impl));
        g.get_mut(b).unwrap().stages.impl_ = StageState::Done;
        assert!(stage_is_ready(&g, a, Stage::Impl));
    }

    #[test]
    fn debug_only_fires_on_failed_impl() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        assert!(!stage_is_ready(&g, root, Stage::Debug));
        g.get_mut(root).unwrap().stages.impl_ = StageState::Failed;
        assert!(stage_is_ready(&g, root, Stage::Debug));
    }

    #[test]
    fn opt_is_skipped_for_now() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let s = &mut g.get_mut(root).unwrap().stages;
        s.spec = StageState::Done;
        s.iface = StageState::Done;
        s.tests = StageState::Done;
        s.impl_ = StageState::Done;
        // We deliberately leave opt as NotStarted forever.
        assert!(!stage_is_ready(&g, root, Stage::Opt));
    }

    #[test]
    fn transient_classifier_recognizes_known_patterns() {
        assert!(is_transient(
            "CompletionError: ResponseError: Response contained no message or tool call (empty)"
        ));
        assert!(is_transient("HTTP 502 Bad Gateway"));
        assert!(is_transient("connection reset"));
        assert!(!is_transient("invalid api key"));
    }

    #[test]
    fn problem_first_paragraph_skips_headings() {
        let md = "# Problem: A samba-equivalent server\n\n\
                  Build a Rust workspace that reimplements a substantial subset of \
                  Samba — SMB/CIFS file serving, NetBIOS, etc.\n\n\
                  More text.";
        let p = problem_first_paragraph(md);
        assert!(p.starts_with("Build a Rust workspace"), "got: {p}");
        assert!(!p.contains('#'));
    }

    #[test]
    fn problem_first_paragraph_truncates_long_text() {
        let md = format!("Lorem ipsum {}", "dolor sit amet ".repeat(40));
        let p = problem_first_paragraph(&md);
        assert!(p.len() <= 200, "len was {}", p.len());
        assert!(p.ends_with("..."));
    }

    #[test]
    fn problem_first_paragraph_falls_back_to_heading_when_no_prose() {
        let p = problem_first_paragraph("# Just a Title\n");
        assert_eq!(p, "Just a Title");
    }

    #[test]
    fn problem_first_paragraph_handles_empty_input() {
        assert_eq!(problem_first_paragraph(""), "Project root.");
        assert_eq!(problem_first_paragraph("   \n   \n"), "Project root.");
    }

    fn test_limits() -> crate::tools::PromptLimits {
        crate::tools::PromptLimits {
            max_file_lines: 600,
            max_spec_section_lines: 800,
        }
    }

    #[test]
    fn role_preamble_iface_actor_forbids_mod_and_directs_to_super_public() {
        let p = role_preamble(Stage::Iface, Role::Writer, test_limits());
        // The preamble should call out that `mod` is forbidden somewhere
        // (the long-form "what's allowed" list now lives in the tool
        // description; the preamble has the workflow-level reminder).
        assert!(
            p.contains("`mod`") && p.to_lowercase().contains("forbidden"),
            "iface actor preamble should mention `mod` is forbidden: {p}"
        );
        assert!(p.contains("super::public"), "should direct to super::public");
        assert!(
            p.contains("snake_case"),
            "should remind about snake_case node names"
        );
    }

    #[test]
    fn iface_tool_description_carries_the_full_forbidden_list() {
        // The system prompt no longer dumps every allowed/forbidden item
        // — that detail belongs in the tool description, sent separately
        // by the rig API. Pin that the tool description still has it.
        let limits = crate::tools::PromptLimits {
            max_file_lines: 600,
            max_spec_section_lines: 800,
        };
        let d = crate::tools::tool_description("submit_public", limits);
        assert!(d.contains("FORBIDDEN") && d.contains("mod"));
        assert!(d.contains("impl"));
    }

    #[test]
    fn role_preamble_spec_actor_pushes_decompose_for_large_missions() {
        let p = role_preamble(Stage::Spec, Role::Writer, test_limits());
        assert!(
            p.contains("decompose"),
            "spec actor preamble must mention decompose"
        );
        assert!(
            p.to_lowercase().contains("snake_case"),
            "spec actor preamble must mention snake_case names"
        );
    }

    #[test]
    fn role_preamble_interpolates_limits_from_config() {
        let limits = crate::tools::PromptLimits {
            max_file_lines: 777,
            max_spec_section_lines: 999,
        };
        let iface = role_preamble(Stage::Iface, Role::Writer, limits);
        assert!(
            iface.contains("777"),
            "iface actor preamble should mention max_file_lines: {iface}"
        );
        let spec = role_preamble(Stage::Spec, Role::Writer, limits);
        assert!(
            spec.contains("999"),
            "spec actor preamble should mention max_spec_section_lines: {spec}"
        );
    }

    #[test]
    fn role_preamble_universal_rules_no_longer_dump_all_tool_names() {
        // Cross-stage tools should not appear in a stage's universal rules.
        // E.g. impl writer's preamble shouldn't mention spec / verdict tools.
        let p = role_preamble(Stage::Impl, Role::Writer, test_limits());
        assert!(
            !p.contains("submit_spec_public") && !p.contains("submit_spec_private"),
            "impl writer should not see spec tools in its preamble: {p}"
        );
        assert!(
            !p.contains("submit_verdict"),
            "impl writer should not see submit_verdict in its preamble: {p}"
        );
    }

    #[test]
    fn spec_reviser_warns_against_changelog_in_spec_body() {
        // The reviser's spec stays a clean spec — no meta-narrative.
        // This test pins the guidance.
        let p = role_preamble(Stage::Spec, Role::Reviser, test_limits());
        let lc = p.to_lowercase();
        assert!(
            lc.contains("changelog") || lc.contains("meta-narrative") || lc.contains("clean spec"),
            "spec reviser preamble must call out 'no changelog/meta-narrative': {p}"
        );
    }

    #[test]
    fn collect_failed_tool_calls_pairs_args_to_results() {
        let now = Utc::now();
        let entries = vec![
            TranscriptEntry {
                timestamp: now,
                kind: TranscriptKind::ToolCall {
                    tool: "decompose".into(),
                },
                content: "{\"children\":[{\"name\":\"x\",\"deps\":[\"x\"]}]}".into(),
                role: None,
            },
            TranscriptEntry {
                timestamp: now,
                kind: TranscriptKind::ToolResult {
                    tool: "decompose".into(),
                    ok: false,
                    error: Some("child 'x' lists itself".into()),
                    output: None,
                },
                content: String::new(),
                role: None,
            },
            // A successful call should not appear in the result.
            TranscriptEntry {
                timestamp: now,
                kind: TranscriptKind::ToolCall {
                    tool: "submit_spec".into(),
                },
                content: "{\"content\":\"# x\\n\"}".into(),
                role: None,
            },
            TranscriptEntry {
                timestamp: now,
                kind: TranscriptKind::ToolResult {
                    tool: "submit_spec".into(),
                    ok: true,
                    error: None,
                    output: Some("{\"bytes\":4}".into()),
                },
                content: String::new(),
                role: None,
            },
        ];
        let failures = collect_failed_tool_calls(&entries);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "decompose");
        assert!(failures[0].1.contains("\"children\""));
        assert!(failures[0].2.contains("lists itself"));
    }
}
