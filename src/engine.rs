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
//! 2. Running that pair through the actor → critic → reviser → judge cycle.
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
    CargoCheckTool, CargoClippyTool, CargoTestNoRunTool, CargoTestTool, DecomposeTool,
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
        let mut root = Node::new(
            self.config.toml.project_name.as_str(),
            "Project root.",
        );
        root.crate_boundary = true;
        let _ = g.insert_root(root)?;
        // Initial render so the workdir exists with a buildable scaffold.
        render::render_graph(&self.workdir, &g, self.layout)?;
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
                    self.graph
                        .lock()
                        .get_mut(node_id)
                        .unwrap()
                        .stages
                        .set(stage, StageState::Done);
                    self.sync_graph_to_state();
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
        let actor_text = self.run_role(task_id, node_id, stage, Role::Actor, None).await?;

        // 2. Critique cycle (optional)
        let critique_retries = self.config.toml.limits.critique_retries;
        let mut last_text = actor_text;
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
                    }),
                )
                .await?;
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
                    }),
                )
                .await?;
            last_text = revision.clone();
            // Judge
            let _judge_text = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Judge,
                    Some(CycleExtras {
                        round,
                        prior_actor_text: None,
                        prior_critique: Some(critique),
                        prior_revision: Some(revision),
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
        // require that node.spec_md is populated (otherwise the actor
        // didn't call submit_spec).
        if stage == Stage::Spec {
            let g = self.graph.lock();
            let n = g.get(node_id).unwrap();
            if n.spec_md.is_none() {
                drop(g);
                let msg = format!("node `{node_name}` spec stage produced no spec.md");
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

    async fn run_role(
        self: &Arc<Self>,
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        role: Role,
        extras: Option<CycleExtras>,
    ) -> Result<String> {
        let model = self.config.toml.models.for_role(role).to_string();

        // Build prompt context.
        let g_for_ctx = self.graph.lock().clone();
        let mut bundle = node_context::build_for_stage(&g_for_ctx, node_id, stage);
        if let Some(ex) = &extras {
            // Annotate with the cycle extras: append a "Critique round"
            // section.
            let mut cyc = String::new();
            cyc.push_str(&format!("Round {}\n\n", ex.round));
            if let Some(t) = &ex.prior_actor_text {
                cyc.push_str("## Prior actor summary\n\n");
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

        let preamble = role_preamble(stage, role);
        let context_doc = bundle.to_markdown();
        let user_prompt = role_user_prompt(stage, role);
        let combined_preamble = format!("{preamble}\n\n{context_doc}");

        // Record system + user prompts.
        let now = Utc::now();
        for (kind, content) in [
            (TranscriptKind::System, combined_preamble.clone()),
            (TranscriptKind::UserPrompt, user_prompt.clone()),
        ] {
            let entry = TranscriptEntry {
                timestamp: now,
                kind,
                content,
            };
            self.state.write(|s| {
                if let Some(t) = s.tasks.get_mut(&task_id) {
                    t.transcript.push(entry.clone());
                }
            });
            self.state.emit(UiEvent::TranscriptAppended {
                task_id,
                entry,
                role,
            });
        }

        // Construct TaskCtx and run with retry on transient errors.
        let ctx = Arc::new(TaskCtx::new(
            task_id,
            node_id,
            stage,
            self.graph.clone(),
            self.workdir.clone(),
            self.layout,
            self.config.toml.limits.max_file_lines,
            self.config.toml.limits.max_spec_section_lines,
            self.cargo_lock.clone(),
        ));

        const MAX_TRANSIENT_RETRIES: u32 = 3;
        let mut transient_attempt = 0u32;
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
        let resp = loop {
            let r = self.driver.drive(params.clone(), ctx.clone()).await;
            match r {
                Ok(resp) => break resp,
                Err(e) => {
                    let msg = format!("{:#}", e);
                    if transient_attempt < MAX_TRANSIENT_RETRIES && is_transient(&msg) {
                        transient_attempt += 1;
                        let backoff = 400u64 * (1 << (transient_attempt - 1).min(3));
                        tracing::warn!(
                            "transient agent error (attempt {transient_attempt}), retrying in {backoff}ms: {msg}"
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        };

        let usage = resp.usage.clone();

        // Final assistant message.
        let final_entry = TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::AssistantText,
            content: resp.output.clone(),
        };
        // Drain ctx transcripts.
        let ctx_entries = ctx.transcript.lock().drain(..).collect::<Vec<_>>();
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
                role,
            });
        }
        self.state.emit(UiEvent::TranscriptAppended {
            task_id,
            entry: final_entry,
            role,
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

        Ok(resp.output)
    }
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
        (Stage::Spec, Role::Actor) | (Stage::Spec, Role::Reviser) => {
            base.tool(SubmitSpecTool { ctx: ctx.clone() })
                .tool(DecomposeTool { ctx })
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

        (Stage::Iface, Role::Actor) | (Stage::Iface, Role::Reviser) => {
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

        (Stage::Tests, Role::Actor) | (Stage::Tests, Role::Reviser) => {
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

        (Stage::Impl, Role::Actor) | (Stage::Impl, Role::Reviser) => {
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

        (Stage::Debug, Role::Actor) | (Stage::Debug, Role::Reviser) => {
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

        (Stage::Opt, Role::Actor) | (Stage::Opt, Role::Reviser) => {
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

fn role_preamble(stage: Stage, role: Role) -> String {
    let role_block = match role {
        Role::Actor => format!(
            "# ROLE: ACTOR\n\nYou are the actor for the **{stage}** stage of one node. \
             Use the available tools to author the right slot of this node:\n\
             - **spec stage**: call `submit_spec` (and optionally `decompose` to add children).\n\
             - **iface stage**: call BOTH `submit_public` AND `submit_private` in that order. \
               public.rs holds trait declarations and newtype `pub struct` wrappers (NO `impl` \
               blocks, NO function bodies). private.rs holds `impl Trait for Newtype` blocks \
               with method bodies as `todo!()` for now — this scaffolding lets dependents \
               compile against the trait. The impl stage will replace the `todo!()` bodies.\n\
             - **tests stage**: call `submit_tests`. The tests should compile against the iface \
               (impls exist as todo!() stubs) but they will fail at runtime — that's expected.\n\
             - **impl stage**: call `submit_private` to replace the todo!() bodies with real \
               implementations that make the tests pass.\n\
             - **debug stage**: call `submit_private` and/or `submit_tests` to fix what's broken.\n\
             - **opt stage**: call `submit_private` to optimize without breaking tests.\n\n\
             Use cargo_* tools to verify before finishing. End with a brief plain-text summary."
        ),
        Role::Critic => format!(
            "# ROLE: CRITIC\n\nYou are the critic for the **{stage}** stage of one node. \
             Identify concrete problems with the actor's work that will matter for THIS \
             stage's contract. Output a short bullet list of specific, actionable concerns. \
             If the work is fine, say exactly `No issues found.`\n\n\
             You may use diagnostic tools (cargo_*) to verify claims; you may NOT modify \
             files. Don't pad. Don't comment on style if behavior is correct. Don't flag \
             omissions that are out of scope for this stage."
        ),
        Role::Reviser => format!(
            "# ROLE: REVISER\n\nYou are the reviser for the **{stage}** stage. The actor \
             produced something; the critic raised concerns. Address each critic point with \
             minimal targeted edits using the same write tools as the actor. Don't \
             redesign. End with a one-paragraph summary of what you changed."
        ),
        Role::Judge => format!(
            "# ROLE: JUDGE\n\nYou are the coherence check at the end of the actor → critic \
             → reviser cycle for the **{stage}** stage. Your job: confirm the reviser \
             addressed each point the critic raised. You are NOT a fresh reviewer; you are \
             NOT the cargo gate (that runs separately). For each critic bullet, decide: \
             addressed / deferred-with-good-reason / ignored. Call `submit_verdict` exactly \
             once with `satisfactory: true` if all points are addressed (or there were no \
             points), or `satisfactory: false` with a concrete reason quoting the unaddressed \
             point(s). When in doubt: satisfactory=true."
        ),
    };
    let common = "You are an expert Rust software engineer participating in a hierarchical \
        decomposition pipeline. The framework owns the project structure, the file layout, \
        and the dependency graph; you fill in slots through tools, not through free-form \
        file writes. The context document below contains everything you should need: ancestor \
        specs, dep public interfaces, current node files. You shouldn't need to read anything \
        from disk.\n\n# IMPORTANT\n\
        - When a tool returns `no_change: true`, the file already had identical content. Move on.\n\
        - The harness detects loops: same tool + same args three times in a row → error. \
          When you see that error, finish with a summary; don't call more tools.\n\
        - Use cargo_* diagnostic tools to verify before finishing. Cheap and immediate.";
    format!("{common}\n\n{role_block}")
}

fn role_user_prompt(stage: Stage, role: Role) -> String {
    match (stage, role) {
        (s, Role::Actor) => format!(
            "Do the {s} stage for this node using the slot-filler tool(s). End with a one-line \
             summary."
        ),
        (s, Role::Critic) => format!(
            "Critique the actor's {s}-stage output. Bullet list. Empty list = `No issues found.`"
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
}
