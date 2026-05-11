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
    ApplyPatchTool, CargoCheckTool, CargoClippyTool, CargoTestNoRunTool, CargoTestTool, Critique,
    JudgeVerdict, ReadFileTool, Role, SubmitArchitectureTool, SubmitCritiqueTool,
    SubmitPrivateTool, SubmitPublicTool, SubmitSpecTool, SubmitTestsTool, SubmitVerdictTool,
    TaskCtx, TranscriptEntry, TranscriptKind, WriteFileRangeTool, WriteFileTool,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::Utc;
use parking_lot::Mutex;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openrouter;
use std::path::{Path, PathBuf};
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
    /// Serializes cargo invocations within a single workdir. Each task
    /// runs in its own worktree (with its own `target/`), so this lock
    /// is only contended for sequential cargos within ONE task — kept
    /// for safety and to make it easy to fall back to single-workdir
    /// mode if needed.
    pub cargo_lock: Arc<tokio::sync::Mutex<()>>,
    /// The main-branch git repo at `workdir`. All per-task worktrees
    /// branch from this and merge back into it.
    pub workspace: Arc<crate::worktree::Workspace>,
    /// Pool managing per-task worktrees.
    pub worktrees: Arc<crate::worktree::WorktreePool>,
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
        let workspace = crate::worktree::Workspace::init(&workdir)?;
        let worktrees = Arc::new(crate::worktree::WorktreePool::new(workspace.clone())?);
        Ok(Self {
            config,
            state,
            graph,
            workdir,
            layout,
            driver,
            cargo_lock: Arc::new(tokio::sync::Mutex::new(())),
            workspace,
            worktrees,
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

    /// Architect's "merge to main": the architect produced the entire
    /// project structure, and there's only ONE architect task ever, so
    /// we just copy every file from the worktree onto main wholesale
    /// (excluding `.git` and bookkeeping dirs) and commit. No
    /// three-way merge needed because there's no concurrent task that
    /// could race with the architect.
    async fn land_architect_to_main(
        &self,
        wt: &crate::worktree::Worktree,
        message: &str,
    ) -> Result<()> {
        // Walk the worktree, copy everything outside .git / target /
        // .bureau onto main. Any file that already exists on main is
        // overwritten — main was empty (or just-scaffolded) when the
        // architect started.
        for entry in walkdir::WalkDir::new(&wt.path).min_depth(1) {
            let entry = entry?;
            let p = entry.path();
            let rel = match p.strip_prefix(&wt.path) {
                Ok(r) => r.to_path_buf(),
                Err(_) => continue,
            };
            if rel.components().any(|c| {
                let s = c.as_os_str().to_string_lossy();
                s == ".git" || s == "target" || s == ".bureau"
            }) {
                continue;
            }
            let dst = self.workdir.join(&rel);
            if entry.file_type().is_dir() {
                std::fs::create_dir_all(&dst).ok();
            } else if entry.file_type().is_file() {
                if let Some(parent) = dst.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::copy(p, &dst)?;
            }
        }
        let _ = self.workspace.commit_main(message)?;
        // Drop the worktree (we don't need its branch).
        let _ = self.worktrees.clone().abandon(wt.clone()).await;
        Ok(())
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
        // Commit the rendered scaffold to main so the first per-task
        // worktree branches from a non-empty tree.
        if let Err(e) = self.workspace.commit_main("scaffold: initial render") {
            tracing::warn!("scaffold commit: {e:#}");
        }
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
        // Architect is FIRST — it must run on root before anything else.
        // Then per-node stages in order.
        for stage in [
            Stage::Architect,
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
        // Architect on root must be Done; spec/iface/tests/impl on every
        // node must be Done.
        if let Some(rid) = g.root {
            if !g
                .get(rid)
                .map(|r| r.stages.architect.is_done())
                .unwrap_or(false)
            {
                return false;
            }
        }
        for n in g.iter() {
            for s in [Stage::Spec, Stage::Iface, Stage::Tests, Stage::Impl] {
                if !n.stages.get(s).is_done() {
                    return false;
                }
            }
        }
        true
    }

    /// Capture the graph slots THIS stage is allowed to write to, so
    /// `restore_stage_slots` can undo a failed stage's tentative writes.
    /// Returned snapshot is opaque; only the matching `restore_*` reads it.
    fn snapshot_stage_slots(&self, node_id: NodeId, stage: Stage) -> StageSlotSnapshot {
        let g = self.graph.lock();
        let n = g.get(node_id).cloned();
        let n = match n {
            Some(n) => n,
            None => return StageSlotSnapshot::default(),
        };
        let mut snap = StageSlotSnapshot::default();
        for slot in slots_owned_by_stage(stage) {
            match slot {
                render::NodeSlot::PublicRs => snap.public_rs = Some(n.public_rs.clone()),
                render::NodeSlot::PrivateRs => snap.private_rs = Some(n.private_rs.clone()),
                render::NodeSlot::TestsRs => snap.tests_rs = Some(n.tests_rs.clone()),
                render::NodeSlot::SpecPublicMd => {
                    snap.spec_public_md = Some(n.spec_public_md.clone())
                }
                render::NodeSlot::SpecPrivateMd => {
                    snap.spec_private_md = Some(n.spec_private_md.clone())
                }
            }
        }
        snap
    }

    fn restore_stage_slots(
        &self,
        node_id: NodeId,
        stage: Stage,
        snap: StageSlotSnapshot,
    ) {
        let mut g = self.graph.lock();
        let Some(n) = g.get_mut(node_id) else { return };
        for slot in slots_owned_by_stage(stage) {
            match slot {
                render::NodeSlot::PublicRs => {
                    if let Some(v) = snap.public_rs.clone() {
                        n.public_rs = v;
                    }
                }
                render::NodeSlot::PrivateRs => {
                    if let Some(v) = snap.private_rs.clone() {
                        n.private_rs = v;
                    }
                }
                render::NodeSlot::TestsRs => {
                    if let Some(v) = snap.tests_rs.clone() {
                        n.tests_rs = v;
                    }
                }
                render::NodeSlot::SpecPublicMd => {
                    if let Some(v) = snap.spec_public_md.clone() {
                        n.spec_public_md = v;
                    }
                }
                render::NodeSlot::SpecPrivateMd => {
                    if let Some(v) = snap.spec_private_md.clone() {
                        n.spec_private_md = v;
                    }
                }
            }
        }
    }

    /// Re-render main from the current canonical graph state, then
    /// commit if anything changed. Used after restoring graph slots
    /// to ensure main's on-disk state and the next downstream
    /// worktree's render both reflect the rolled-back content.
    fn render_and_commit_main(&self, stage: Stage, node_name: &str) -> Result<()> {
        let g = self.graph.lock();
        render::render_graph(&self.workdir, &g, self.layout)?;
        drop(g);
        let _ = self
            .workspace
            .commit_main(&format!("restore: {stage} {node_name} (rolled back failed stage)"))?;
        Ok(())
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

        // Snapshot the graph slots this stage will write to. If the stage
        // gives up after exhausting retries, we restore these slots —
        // otherwise the last-submitted-but-broken content sits in the
        // graph and any downstream task that re-renders from the graph
        // picks up that broken content. Main itself may be fine (we
        // didn't land), but the canonical graph is what renders into
        // downstream worktrees, so this matters.
        let pre_stage_slots = self.snapshot_stage_slots(node_id, stage);

        // Allocate a per-stage worktree off main HEAD. All renders, cargo
        // invocations, and tool writes for this stage's attempts go into
        // it. On stage success we merge it back; on failure we abandon.
        // Retries within a stage REUSE the same worktree so the model's
        // partial work persists across retries.
        let stage_uuid = Uuid::new_v4();
        let worktree = match self.worktrees.allocate(stage_uuid).await {
            Ok(wt) => wt,
            Err(e) => {
                tracing::error!("worktree allocation: {e:#}");
                return Err(e);
            }
        };

        let max_retries = self.config.toml.limits.max_stage_retries;
        let mut last_err: Option<anyhow::Error> = None;
        let mut succeeded = false;
        for attempt in 1..=(max_retries + 1) {
            match self
                .run_one_attempt(node_id, stage, attempt, &worktree.path)
                .await
            {
                Ok(()) => {
                    let node_name = {
                        let mut g = self.graph.lock();
                        let n = g.get_mut(node_id).unwrap();
                        n.stages.set(stage, StageState::Done);
                        n.name.clone()
                    };
                    self.sync_graph_to_state();
                    // Land THIS task's owned files onto main. We
                    // deliberately AVOID a three-way merge here: each
                    // task's worktree contains a full-tree render that
                    // includes other nodes' content (because cargo_check
                    // needs the full tree to compile), and that content
                    // can diverge between concurrent tasks (different
                    // snapshots of the shared graph). The three-way
                    // merge then trips on "both branches modified the
                    // same file with different content" even though
                    // neither task actually intended to write that file.
                    //
                    // Instead we copy ONLY the files this stage owns
                    // (per-node spec/source/Cargo.toml as appropriate)
                    // from the worktree to main, then commit on main.
                    // Concurrent tasks are serialized by the worktree
                    // pool's main_lock.
                    //
                    // Architect is special: it produces the whole
                    // workspace structure, so we commit the worktree
                    // wholesale to main (effectively the existing merge
                    // flow, but it only ever runs once).
                    let merge_msg = format!("{stage}: {node_name}");
                    let canonical_render = {
                        let g = self.graph.lock();
                        if g.topo_order().is_none() {
                            Err(anyhow!(
                                "shared graph is cyclic at merge time — refusing to land \
                                 worktree for {node_name} {stage}"
                            ))
                        } else {
                            render::render_graph(&worktree.path, &g, self.layout)
                                .map_err(|e| anyhow!("re-render before commit: {e}"))
                        }
                    };
                    match canonical_render {
                        Ok(_) => {
                            let result = if stage == Stage::Architect {
                                self.land_architect_to_main(&worktree, &merge_msg).await
                            } else {
                                let owned: Vec<std::path::PathBuf> = {
                                    let g = self.graph.lock();
                                    let n = g.get(node_id).unwrap();
                                    render::files_owned_by_stage(&g, n, stage, self.layout)
                                };
                                self.worktrees
                                    .clone()
                                    .apply_to_main(worktree.clone(), &owned, &merge_msg)
                                    .await
                            };
                            if let Err(e) = result {
                                tracing::warn!(
                                    "worktree apply-to-main {node_name} {stage}: {e:#}"
                                );
                            } else {
                                // Post-merge integrator check. Runs the
                                // stage's gate on MAIN; if main now fails
                                // (a concurrent land broke our combined
                                // state), allocate a fresh worktree and
                                // run quickfix on it, copying any fixed
                                // owned files back to main on success.
                                // If quickfix exhausts, halt.
                                if let Err(e) = self
                                    .integrator_check(node_id, stage, &node_name)
                                    .await
                                {
                                    last_err = Some(e);
                                    succeeded = false;
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "refusing to merge worktree for {node_name} {stage}: {e:#}"
                            );
                            if let Err(ae) = self
                                .worktrees
                                .clone()
                                .abandon(worktree.clone())
                                .await
                            {
                                tracing::warn!("worktree abandon: {ae:#}");
                            }
                            last_err = Some(e);
                            break;
                        }
                    }
                    succeeded = true;
                    break;
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
        if !succeeded {
            // All retries exhausted — drop the worktree without merging.
            if let Err(e) = self.worktrees.clone().abandon(worktree).await {
                tracing::warn!("worktree abandon: {e:#}");
            }
            // CRITICAL: revert the graph slots this stage may have written
            // to. The submit_* and write_file/apply_patch tools update
            // slots BEFORE the gate runs; if the stage ultimately fails,
            // we'd otherwise leave the last-submitted-but-broken content
            // in the graph. Subsequent renders (including for downstream
            // tasks branching off main) read from the graph — so without
            // this restore, downstream worktrees end up with broken
            // content even though main itself is fine. Re-render main
            // to match the restored slots.
            self.restore_stage_slots(node_id, stage, pre_stage_slots);
            if let Err(e) = self.render_and_commit_main(
                stage,
                &self.graph.lock().get(node_id).unwrap().name.clone(),
            ) {
                tracing::warn!("re-render main after slot restore: {e:#}");
            }
            self.sync_graph_to_state();
        }
        if succeeded {
            return Ok(());
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
        task_workdir: &Path,
    ) -> Result<()> {
        let task_id = Uuid::new_v4();
        let node_name = self.graph.lock().get(node_id).unwrap().name.clone();
        // Resolve the writer's model for THIS stage as the task-level
        // model name (other roles within the cycle may use different
        // models — see `run_role`).
        let writer_model = self
            .config
            .toml
            .models
            .for_stage_role(stage, Role::Writer)
            .to_string();
        let task = EngineTask {
            id: task_id,
            node_id,
            node_name: node_name.clone(),
            stage,
            status: TaskStatus::Running,
            model: writer_model.clone(),
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

        // The cargo gate for this stage. None for architect/spec.
        let gate_kind = match stage {
            Stage::Architect | Stage::Spec => None,
            Stage::Iface => Some(crate::gate::GateKind::Check),
            Stage::Tests => Some(crate::gate::GateKind::TestNoRun),
            Stage::Impl | Stage::Debug | Stage::Opt => Some(crate::gate::GateKind::Test),
        };

        // 1. Writer turn.
        let actor = self
            .run_role(task_id, node_id, stage, Role::Writer, None, task_workdir)
            .await?;

        // 2. Quickfix loop after writer — if the gate fails, give the
        // writer/quickfixer a chance to fix the errors directly via
        // read/write/patch tools BEFORE escalating to critic. Mechanical
        // compile-error fixes don't need a critique cycle.
        if gate_kind.is_some() {
            self.run_quickfix_loop(task_id, node_id, stage, gate_kind.unwrap(), task_workdir)
                .await?;
        }

        // 3. Critique cycle (optional).
        // Architect runs single-shot (no critic/reviser/judge cycle); other
        // stages use the configured critique_retries.
        let critique_retries = if stage == Stage::Architect {
            0
        } else {
            self.config.toml.limits.critique_retries
        };
        let mut last_text = actor.text;
        let mut last_failed = actor.failed_tools;
        for round in 1..=critique_retries {
            let critic_outcome = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Critic,
                    Some(CycleExtras {
                        round,
                        prior_actor_text: Some(last_text.clone()),
                        prior_failed_tools: last_failed.clone(),
                        ..Default::default()
                    }),
                    task_workdir,
                )
                .await?;
            // Critic-happy fast path: empty issue list via submit_critique
            // = nothing to fix, skip reviser+judge. If the critic didn't
            // call submit_critique at all (model misbehavior), fall
            // through and run the full cycle — better wasted work than
            // a silently-skipped review.
            let (critique_text, skip_rest) = match critic_outcome.critique {
                Some(c) if c.is_clean() => {
                    self.note(format!(
                        "task {task_id} round {round}: critic reported 0 issues — skipping reviser and judge"
                    ));
                    (c.render(), true)
                }
                Some(c) => (c.render(), false),
                None => (
                    "(critic did not call submit_critique — running reviser conservatively)"
                        .to_string(),
                    false,
                ),
            };
            if skip_rest {
                break;
            }
            let revision = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Reviser,
                    Some(CycleExtras {
                        round,
                        prior_actor_text: Some(last_text.clone()),
                        prior_critique: Some(critique_text.clone()),
                        prior_failed_tools: last_failed.clone(),
                        ..Default::default()
                    }),
                    task_workdir,
                )
                .await?;
            last_text = revision.text.clone();
            last_failed = revision.failed_tools.clone();

            // Quickfix loop after reviser too — same rationale.
            if let Some(kind) = gate_kind {
                self.run_quickfix_loop(task_id, node_id, stage, kind, task_workdir)
                    .await?;
            }

            // Judge.
            let _judge = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::Judge,
                    Some(CycleExtras {
                        round,
                        prior_critique: Some(critique_text),
                        prior_revision: Some(revision.text),
                        prior_failed_tools: last_failed.clone(),
                        ..Default::default()
                    }),
                    task_workdir,
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

        // 4. Final cargo gate (safety net). The quickfix loops above
        // should have ensured the gate already passes; if it doesn't
        // somehow, this catches it.
        if let Some(kind) = gate_kind {
            let outcome = {
                let _guard = self.cargo_lock.lock().await;
                crate::gate::run_gate(task_workdir, kind).await?
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
        // (No post-stage check for Architect — the tool's call() either
        // succeeded or failed; if it failed the writer's retry logic
        // will have surfaced it. An empty children list is a legitimate
        // output for a single-crate project with no sub-modules.)

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

    /// Post-merge integrator. Runs after each successful `apply_to_main`
    /// to verify the combined state on main still passes the stage's
    /// gate. Two concurrent tasks can each pass their own worktree's
    /// gate but break the merged result on main if their changes
    /// interact. The integrator catches this:
    ///
    /// 1. Run the gate on MAIN. If it passes, we're done.
    /// 2. If it fails, allocate a fresh worktree off main, run the
    ///    quickfix loop in it (model edits the current node's files
    ///    based on the failing diagnostics), then apply the fixed
    ///    owned files back to main.
    /// 3. If the quickfix loop can't fix it, return Err — the caller
    ///    marks the stage Failed and the pipeline halts rather than
    ///    continuing to build on a broken tree.
    async fn integrator_check(
        self: &Arc<Self>,
        node_id: NodeId,
        stage: Stage,
        node_name: &str,
    ) -> Result<()> {
        let gate_kind = match stage {
            Stage::Architect | Stage::Spec => None,
            Stage::Iface => Some(crate::gate::GateKind::Check),
            Stage::Tests => Some(crate::gate::GateKind::TestNoRun),
            Stage::Impl | Stage::Debug | Stage::Opt => Some(crate::gate::GateKind::Test),
        };
        let Some(kind) = gate_kind else {
            return Ok(());
        };
        let outcome = {
            let _guard = self.cargo_lock.lock().await;
            crate::gate::run_gate(&self.workdir, kind).await?
        };
        if outcome.passed {
            return Ok(());
        }
        let summary = self.summarize_errors(&outcome.errors, 8);
        self.note(format!(
            "integrator: post-merge gate failed for `{node_name}` {stage}:\n{summary}"
        ));
        let wt = self.worktrees.allocate(Uuid::new_v4()).await?;
        let task_id = Uuid::new_v4();
        let n = self
            .graph
            .lock()
            .get(node_id)
            .ok_or_else(|| anyhow!("node {node_id} missing"))?
            .clone();
        let task = EngineTask {
            id: task_id,
            node_id,
            node_name: n.name.clone(),
            stage,
            status: TaskStatus::Running,
            model: self
                .config
                .toml
                .models
                .for_stage_role(stage, Role::QuickFixer)
                .to_string(),
            transcript: Vec::new(),
            cost: TokenUsage::default(),
            started_at: Some(Utc::now()),
            finished_at: None,
            error: None,
            final_verdict: None,
            retries: 0,
        };
        self.state.write(|s| {
            s.tasks.insert(task_id, task.clone());
        });
        self.state.emit(UiEvent::TaskCreated { task });
        let quickfix_result = self
            .run_quickfix_loop(task_id, node_id, stage, kind, &wt.path)
            .await;
        let post_outcome = match quickfix_result {
            Ok(()) => {
                let _guard = self.cargo_lock.lock().await;
                crate::gate::run_gate(&wt.path, kind).await?
            }
            Err(e) => {
                let _ = self.worktrees.clone().abandon(wt).await;
                return Err(e);
            }
        };
        if post_outcome.passed {
            let owned: Vec<std::path::PathBuf> = {
                let g = self.graph.lock();
                let nn = g.get(node_id).unwrap();
                render::files_owned_by_stage(&g, nn, stage, self.layout)
            };
            let merge_msg = format!("integrator: {stage} {node_name}");
            self.worktrees
                .clone()
                .apply_to_main(wt.clone(), &owned, &merge_msg)
                .await?;
            let recheck = {
                let _guard = self.cargo_lock.lock().await;
                crate::gate::run_gate(&self.workdir, kind).await?
            };
            if recheck.passed {
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
                self.note(format!(
                    "integrator: post-merge gate fixed for `{node_name}` {stage}"
                ));
                Ok(())
            } else {
                let recheck_summary = self.summarize_errors(&recheck.errors, 5);
                let msg = format!(
                    "integrator: post-merge gate still broken on main after quickfix \
                     for `{node_name}` {stage}:\n{recheck_summary}"
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
                Err(anyhow!(msg))
            }
        } else {
            let post_summary = self.summarize_errors(&post_outcome.errors, 5);
            let _ = self.worktrees.clone().abandon(wt).await;
            let msg = format!(
                "integrator: quickfix exhausted for `{node_name}` {stage}; tree still broken:\n\
                 {post_summary}"
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
            Err(anyhow!(msg))
        }
    }

    /// Quickfix inner loop. Run the cargo gate; if it passes, return.
    /// Otherwise feed the failing diagnostics back to the model via the
    /// `QuickFixer` role (with read/write/patch tools), let it edit,
    /// and re-check. Repeat up to `max_quickfix_iters` times. Returns
    /// `Ok(())` whether the gate ends up passing or not — the final
    /// gate check in the caller decides whether the stage attempt
    /// failed. (We don't propagate the failure here so the critic/judge
    /// cycle can still try its own fix.)
    async fn run_quickfix_loop(
        self: &Arc<Self>,
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        kind: crate::gate::GateKind,
        task_workdir: &Path,
    ) -> Result<()> {
        let max_iters = self.config.toml.limits.max_quickfix_iters;
        if max_iters == 0 {
            return Ok(());
        }
        for iter in 1..=max_iters {
            // Run the gate. Short-circuit on pass.
            let outcome = {
                let _guard = self.cargo_lock.lock().await;
                crate::gate::run_gate(task_workdir, kind).await?
            };
            if outcome.passed {
                if iter > 1 {
                    self.note(format!(
                        "task {task_id} stage {stage}: quickfix succeeded after {} iter(s)",
                        iter - 1
                    ));
                }
                return Ok(());
            }
            // Classify failing errors by which node owns the file. If
            // EVERY error is in another node's slot, this writer can't
            // fix any of them — calling the model would just produce
            // refusals (the model correctly identifies out-of-scope and
            // ends its turn). Skip the model call, log clearly, and let
            // the outer retry/halt logic handle it.
            let classification = classify_errors_by_owner(
                &outcome.errors,
                node_id,
                &self.graph.lock(),
                self.layout,
            );
            if !classification.mine_or_unknown() {
                let upstream = classification.describe_upstream();
                self.note(format!(
                    "task {task_id} stage {stage}: gate failed entirely on upstream node(s) \
                     [{upstream}] — cannot fix from this node's scope. Skipping quickfix; \
                     the outer halt logic will surface this so those nodes can be reset."
                ));
                return Ok(());
            }
            let summary = self.summarize_errors(&outcome.errors, 12);
            // If SOME errors are upstream (not all), tell the model about
            // it so it doesn't waste turns trying to fix out-of-scope code.
            let preamble_extra = classification.upstream_note();
            self.note(format!(
                "task {task_id} stage {stage}: quickfix iter {iter}/{max_iters} — gate failed:\n{summary}"
            ));
            // Run the quickfixer.
            let _ = self
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::QuickFixer,
                    Some(CycleExtras {
                        round: iter,
                        quickfix_gate_output: Some(format!("{summary}{preamble_extra}")),
                        quickfix_iter: Some((iter, max_iters - iter)),
                        ..Default::default()
                    }),
                    task_workdir,
                )
                .await?;
        }
        // One last gate check — if it now passes, log it.
        let outcome = {
            let _guard = self.cargo_lock.lock().await;
            crate::gate::run_gate(task_workdir, kind).await?
        };
        if outcome.passed {
            self.note(format!(
                "task {task_id} stage {stage}: quickfix passed on final check"
            ));
        } else {
            self.note(format!(
                "task {task_id} stage {stage}: quickfix exhausted {max_iters} iter(s); \
                 escalating to critic"
            ));
        }
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
        task_workdir: &Path,
    ) -> Result<RoleOutcome> {
        let model = self
            .config
            .toml
            .models
            .for_stage_role(stage, role)
            .to_string();

        // Build prompt context. Always inject the project mission as the
        // first section so every node — root or deep child — sees what
        // the overall goal is. Without this the root node has no idea what
        // it's building beyond its own name. If a `style.md` is present
        // in the config dir, inject it right after as a "Style guide"
        // section the user can use to nudge tone, verbosity, etc.
        let g_for_ctx = self.graph.lock().clone();
        let mut bundle = node_context::ContextBundle::new();
        bundle.push("Project mission", self.config.problem.trim().to_string());
        if let Some(style) = &self.config.style {
            bundle.push("Style guide", style.clone());
        }
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
            if let Some((iter, remaining)) = ex.quickfix_iter {
                cyc.push_str(&format!(
                    "## Quickfix iteration {iter} ({remaining} remaining)\n\n"
                ));
            }
            if let Some(out) = &ex.quickfix_gate_output {
                cyc.push_str("## ⚠ Failing cargo gate — fix these\n\n");
                cyc.push_str("```\n");
                cyc.push_str(out);
                cyc.push_str("\n```\n\n");
            }
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
            task_workdir.to_path_buf(),
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
        // Drain critique (only set when role == Critic).
        let critique = ctx.critique.lock().take();
        // Drain fs events.
        let fs_events: Vec<PathBuf> = ctx.fs_events.lock().drain(..).collect();

        let transcript_cap = self.config.toml.limits.task_transcript_cap;
        self.state.write(|s| {
            if let Some(t) = s.tasks.get_mut(&task_id) {
                t.transcript.extend(ctx_entries.iter().cloned());
                t.transcript.push(final_entry.clone());
                t.cost.add(&usage);
                if matches!(role, Role::Judge) {
                    t.final_verdict = verdict.clone();
                }
                crate::state::cap_transcript(t, transcript_cap);
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
        // Ship the task's ACCUMULATED cost, not this role's delta. The UI
        // overwrites `task.cost` from this event, so shipping a delta
        // would make the task list display only the most recent role's
        // tokens (and 0 for any role that produced no tokens). Reading
        // the accumulated value off state requires re-locking, but it's
        // a small clone.
        let task_total_cost = self
            .state
            .read(|s| s.tasks.get(&task_id).map(|t| t.cost.clone()))
            .unwrap_or_default();
        self.state.emit(UiEvent::TaskCost {
            task_id,
            cost: task_total_cost,
            total: self.state.read(|s| s.total_cost.clone()),
            estimated_usd: self.state.read(|s| s.estimated_cost_usd),
        });

        // Sync the engine's working graph back to the EngineState's view.
        self.sync_graph_to_state();
        self.state.emit(UiEvent::NodeChanged { id: node_id });

        Ok(RoleOutcome {
            text: resp.output,
            failed_tools,
            critique,
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
    /// Set when this role is Critic and `submit_critique` was called.
    /// `None` otherwise (and treated as "critic skipped the tool call",
    /// which the cycle treats conservatively as needs-revision).
    critique: Option<Critique>,
}

/// Returns true if the (node, stage) combination is ready to run right now.
/// "Ready" means:
/// - the stage's state is `NotStarted`
/// - all required preconditions hold (see below)
///
/// Stage preconditions:
/// - `Architect`: only the ROOT node, only when its Architect is
///   NotStarted. Builds the whole tree in one shot before anything else.
/// - `Spec`: root's `Architect` is Done AND parent's `Spec` is Done
///   (or this node IS the root).
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
    // Helper: is the architect phase done? It runs on root only.
    let architect_done = match graph.root {
        Some(rid) => graph
            .get(rid)
            .map(|r| r.stages.architect.is_done())
            .unwrap_or(false),
        None => false,
    };
    match stage {
        Stage::Opt => false, // Skip opt for now; see comment above.
        Stage::Architect => {
            // Architect runs ONLY on root, ONLY once, BEFORE anything else.
            if cur != StageState::NotStarted {
                return false;
            }
            n.parent.is_none()
        }
        Stage::Spec => {
            if cur != StageState::NotStarted {
                return false;
            }
            // Every node's spec waits for the architect to lay out the tree.
            if !architect_done {
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

#[derive(Debug, Clone, Default)]
struct CycleExtras {
    round: u32,
    prior_actor_text: Option<String>,
    prior_critique: Option<String>,
    prior_revision: Option<String>,
    /// Tool calls that failed during the prior actor (or reviser) turn,
    /// surfaced to the critic and the next reviser so they're not lost.
    /// Each entry: (tool_name, args_json, error_msg).
    prior_failed_tools: Vec<(String, String, String)>,
    /// For the QuickFixer role: a rendered summary of the failing cargo
    /// gate output. Pre-formatted with one entry per error / failing
    /// test so the model doesn't have to parse cargo's JSON.
    quickfix_gate_output: Option<String>,
    /// For the QuickFixer role: which iteration we're on (1-based) and
    /// how many remain. Lets the model pace itself.
    quickfix_iter: Option<(u32, u32)>,
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
        (Stage::Architect, Role::Writer) => {
            base.tool(SubmitArchitectureTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Architect, _) => {
            // Architect runs single-shot — no critic/reviser/judge cycles.
            base.build().prompt(user_prompt).extended_details().await?
        }

        (Stage::Spec, Role::Writer) | (Stage::Spec, Role::Reviser) => {
            base.tool(SubmitSpecTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Spec, Role::Critic) => {
            base.tool(SubmitCritiqueTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
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
            base.tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(SubmitCritiqueTool { ctx })
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
                .tool(CargoTestNoRunTool { ctx: ctx.clone() })
                .tool(SubmitCritiqueTool { ctx })
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
                .tool(CargoClippyTool { ctx: ctx.clone() })
                .tool(SubmitCritiqueTool { ctx })
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
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(SubmitCritiqueTool { ctx })
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
                .tool(CargoClippyTool { ctx: ctx.clone() })
                .tool(SubmitCritiqueTool { ctx })
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

        // QuickFixer — same shape of tools regardless of stage; the gate's
        // diagnostic tool varies. The loop only fires for stages with a
        // cargo gate; Architect/Spec branches are kept for exhaustiveness.
        (Stage::Spec, Role::QuickFixer) => {
            base.build().prompt(user_prompt).extended_details().await?
        }
        (Stage::Iface, Role::QuickFixer) => {
            base.tool(ReadFileTool { ctx: ctx.clone() })
                .tool(WriteFileTool { ctx: ctx.clone() })
                .tool(WriteFileRangeTool { ctx: ctx.clone() })
                .tool(ApplyPatchTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Tests, Role::QuickFixer) => {
            base.tool(ReadFileTool { ctx: ctx.clone() })
                .tool(WriteFileTool { ctx: ctx.clone() })
                .tool(WriteFileRangeTool { ctx: ctx.clone() })
                .tool(ApplyPatchTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestNoRunTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?
        }
        (Stage::Impl, Role::QuickFixer)
        | (Stage::Debug, Role::QuickFixer)
        | (Stage::Opt, Role::QuickFixer) => {
            base.tool(ReadFileTool { ctx: ctx.clone() })
                .tool(WriteFileTool { ctx: ctx.clone() })
                .tool(WriteFileRangeTool { ctx: ctx.clone() })
                .tool(ApplyPatchTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx })
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
treat it as ground truth for what's being built. If a **Style guide** \
section follows, it carries user-supplied preferences about tone, \
verbosity, code style, and what to avoid — treat its instructions as \
overriding the defaults below where they conflict. Subsequent sections \
give you ancestor specs, sibling specs, dep public interfaces, and the \
current node's already-authored slots.\n\n\
# Universal rules\n\
- The tool list provided this turn is exhaustive. Call only those tools; \
  ignore patterns from other stages.\n\
- When a tool returns `no_change: true`, the file already had identical \
  content. Move on; do not re-call it.\n\
- Same tool + same args three times in a row triggers a hard error. When \
  you see that, finish with a one-line summary and stop calling tools.\n\
- All node names are **snake_case Rust identifiers**. CamelCase is for \
  Rust types, not nodes — never reference a sibling/dep as CamelCase.\n\
- DEFAULT WRITING STYLE (overridable by **Style guide**): be terse. \
  Specs and code should be matter-of-fact and minimal. Avoid \
  just-in-case caveats, jargon padding, marketing language, or \
  rambly prose. Short sentences. Concrete nouns. If a sentence \
  doesn't add information, delete it.";

    let role_block = match (stage, role) {
        // ---- ARCHITECT ----
        (Stage::Architect, Role::Writer) => format!(
            "# ARCHITECT · WRITER\n\
            \n\
            You are designing the WHOLE STRUCTURE of this Rust project in \
            ONE call. Read the **Project mission** above, then submit the \
            project's complete decomposition tree via `submit_architecture` \
            — exactly once. After that the per-node stages take over and \
            flesh things out; you don't need to (and shouldn't) write any \
            spec content here.\n\
            \n\
            Output: the SKELETON — crates, modules, parent-child \
            relationships, cross-node dep edges, anticipated external \
            Cargo deps. Think of it like sitting down to draft the project \
            layout: which crates exist, how they nest as modules, which \
            subsystem depends on which, where the natural seams are.\n\
            \n\
            ## Heuristics\n\
            \n\
            - Aim shallower-and-broader, not deeper-and-narrower. A healthy \
              project-scale tree might be 5–10 first-level subsystems, each \
              splitting once or twice more. Not hundreds of leaves at depth \
              5.\n\
            - One module per Rust file. Per-file cap is {max_file} lines, so \
              if a leaf can't reasonably express its surface in that, split \
              it; otherwise keep it a leaf.\n\
            - `crate_boundary` is for MAJOR top-level subsystems that \
              warrant a separate Cargo package. A handful per project. \
              Most children become modules within their parent's crate. \
              One-crate-per-leaf is wrong.\n\
            - Names are GLOBALLY unique snake_case Rust idents — they're \
              how dep edges resolve. CamelCase is for types, never nodes.\n\
            - Keep cross-crate dep edges acyclic (the framework checks \
              this at submit time at both the node and crate level). \
              Typical shape: shared utility crates at the bottom, \
              subsystems above, daemons/binaries at the top.\n\
            \n\
            ## What goes in `description`\n\
            \n\
            One short sentence per node — what it's for, in functional \
            terms. Not a spec; not implementation hints. Just enough that \
            the per-node spec writer downstream can recognize what its \
            node is supposed to be.\n\
            \n\
            End your turn with a one-line summary after the tool call \
            returns."
        ),
        (Stage::Architect, _) => "# ARCHITECT (non-writer)\n\
            \n\
            The architect stage runs single-shot — only the Writer role \
            speaks. Output nothing."
            .into(),

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
            Read the writer's spec. Identify CONCRETE problems: missing \
            sections, vague invariants, scope creep, decomposition that \
            doesn't match the project mission, child names that aren't \
            snake_case. Report via `submit_critique` exactly once. Each \
            issue's `description` should be one actionable sentence the \
            reviser can act on directly. If the spec is fine, call \
            `submit_critique` with an EMPTY `issues` list — that signals \
            the framework to skip the reviser and judge. Don't pad. \
            Don't restate the spec. Don't list cosmetic preferences."
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
            ## CRITICAL — unimplemented functions go in TRAITS, not modules\n\
            \n\
            Rust has NO concept of a \"function prototype\" or \"forward \
            declaration\". Writing `pub fn foo() -> Bar;` (signature \
            followed by a semicolon) inside a module is a SYNTAX ERROR \
            — it's not valid Rust and `cargo check` will reject it.\n\
            \n\
            If you want to declare a function whose implementation \
            lives elsewhere (or is not yet written), put it inside a \
            `pub trait`:\n\
            ```rust\n\
            pub trait Foo {{\n\
                fn bar(&self) -> Bar;          // OK — trait method\n\
            }}\n\
            ```\n\
            This is the ONLY way to express an unimplemented function \
            in Rust's public surface. Free functions in modules MUST \
            have a body — even if it's `todo!()` (but `todo!()` belongs \
            in `private.rs`, not `public.rs`).\n\
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
            Use `cargo_check` to verify the iface compiles. Identify \
            concrete problems: forbidden items in `public.rs`, missing \
            `impl` stubs in `private.rs`, mismatch between trait \
            signatures and the spec's API section, undeclared dep \
            imports. Report via `submit_critique` exactly once. Each \
            issue's `description` is one actionable sentence with a \
            `file:line` `location` if you can identify one. If clean, \
            call `submit_critique` with an EMPTY `issues` list. The \
            quickfix loop already ran for mechanical compile fixes — \
            don't re-litigate compile errors that are already gone."
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
            2. Cover the spec's invariants and edge cases — see the \
               scope and triviality rules below.\n\
            3. Run `cargo_test_no_run` to verify the file compiles.\n\
            4. End with a one-line summary.\n\
            \n\
            Cap: {max_file} lines. Tests will COMPILE because \
            `private.rs` has `todo!()` stubs satisfying the trait at the \
            type level — they FAIL at runtime, which is expected. The \
            next stage replaces the stubs and the same tests pass.\n\
            \n\
            ## What to test\n\
            \n\
            Test the FUNCTIONAL CONTRACT this node's spec promises. \
            Tests should fail if the implementation violates an \
            invariant, edge case, or behaviour described in the spec.\n\
            \n\
            ## What NOT to test (these are wasted tokens)\n\
            \n\
            - **Things the language guarantees**: don't test that a \
              constructor returns a struct of the right type, that a \
              `Vec` is empty after `clear()`, that `Default::default()` \
              produces a default value, that an enum's variants \
              destructure correctly. The compiler proves these for you.\n\
            - **Implementation details**: don't test private internals \
              you happen to know exist. Test through the public surface.\n\
            - **Other nodes' contracts**: tests for node X test X's \
              public interface ONLY. Do NOT write project-level tests \
              that depend on other nodes existing, `tests::fixture_files_exist`, \
              `tests::all_binary_entry_points_exist`, etc. — these belong \
              in dedicated integration-test nodes if at all, and they \
              break every other node's gate when they fail. Stay in scope.\n\
            - **Trivially-true assertions**: `assert_eq!(2 + 2, 4)`-style \
              filler. If a test would pass for any non-empty struct of \
              the right shape, don't write it.\n\
            \n\
            ## Module-path rules\n\
            \n\
            `use crate::<X>::...` rule same as `private.rs`: X must be a \
            declared dep / ancestor / own child. Don't write integration \
            tests that need network or filesystem unless the spec calls \
            for it."
        ),
        (Stage::Tests, Role::Critic) => {
            "# TESTS · CRITIC\n\
            \n\
            Use `cargo_test_no_run` to confirm tests compile. Identify \
            concrete problems: tests that don't actually exercise the \
            spec, tests that import via `crate::TypeName` instead of \
            `super::public::*`, missing edge-case coverage, OR tests \
            that test things the language already guarantees (e.g. \
            \"a struct's constructor returns a struct\", \"a `Vec` \
            is empty after `clear()`\" — these are wasted tokens; flag \
            them for deletion). Report via `submit_critique` exactly \
            once with an actionable `description` per issue. If clean, \
            call `submit_critique` with an EMPTY `issues` list."
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
            Run `cargo_test`. Identify concrete problems: failing tests, \
            lints with obvious correctness implications, any `unsafe` or \
            `unwrap()` smell that the spec didn't sanction. Report via \
            `submit_critique` exactly once with one actionable issue per \
            entry. If green and clean, call `submit_critique` with an \
            EMPTY `issues` list. The quickfix loop already ran for \
            mechanical fixes; don't re-litigate them."
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
            Run `cargo_test`. Identify anything still failing or any \
            test that was loosened to make impl pass. Report via \
            `submit_critique` exactly once. If clean, call \
            `submit_critique` with an EMPTY `issues` list."
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
            or was made worse, report via `submit_critique`. Otherwise \
            call `submit_critique` with an EMPTY `issues` list."
                .to_string()
        }
        (Stage::Opt, Role::Judge) => judge_block(Stage::Opt),

        // QuickFixer — same preamble for every stage; the specific gate
        // and the errors to address come from the cycle context block.
        (_, Role::QuickFixer) => quickfix_preamble(stage),
    };

    format!("{common}\n\n{role_block}")
}

fn quickfix_preamble(stage: Stage) -> String {
    let gate = match stage {
        Stage::Iface => "`cargo_check`",
        Stage::Tests => "`cargo_check` and `cargo_test_no_run`",
        Stage::Impl | Stage::Debug | Stage::Opt => "`cargo_check` and `cargo_test`",
        Stage::Architect | Stage::Spec => "(no gate)",
    };
    format!(
        "# QUICKFIX · {stage}\n\
        \n\
        The previous writer/reviser turn left the build in a FAILED state. \
        Your job is to fix the compile / test errors directly — not to \
        redesign, not to second-guess the spec, just to make the build \
        green. The errors are listed in the cycle-context section below.\n\
        \n\
        ## Workflow\n\
        \n\
        1. Read the errors. Each has a file path + line number.\n\
        2. Use `read_file` to inspect surrounding code if you need context.\n\
        3. Apply the smallest possible fix:\n\
           - For a localized change (one function body, one signature), \
             prefer `write_file_range` or `apply_patch`.\n\
           - For a whole-file rewrite, use `write_file`.\n\
        4. Re-run {gate} to confirm the fix landed.\n\
        5. If clean, end your turn with a one-line summary. If errors \
           remain, iterate.\n\
        \n\
        ## Tool rules\n\
        \n\
        - You can ONLY edit slots on the CURRENT node: `<src>/public.rs`, \
          `<src>/private.rs`, `<src>/tests.rs`, `<spec>/public.md`, \
          `<spec>/private.md`. Auto-generated files (mod.rs, lib.rs, \
          Cargo.toml) cannot be edited — those are framework-rendered.\n\
        - If the right fix is in another node's file, end your turn and \
          explain why — the framework will route that elsewhere.\n\
        - DO NOT call any submit_* tool from here. The slot edits do the \
          equivalent of submit_* (validate, update graph, re-render).\n\
        \n\
        ## What NOT to do\n\
        \n\
        - Don't rewrite the public API to dodge a type error in private — \
          fix private to honor public.\n\
        - Don't delete failing tests. If a test is wrong, that's a \
          test-stage problem; flag it and stop.\n\
        - Don't add panics, todos, or unimplemented!() to make code \
          compile — the cargo_test gate will still catch you."
    )
}

fn judge_block(stage: Stage) -> String {
    let upper = match stage {
        Stage::Architect => "ARCHITECT",
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
            "Critique the writer's {s}-stage output. Call submit_critique exactly once with a \
             concrete `issues` list (empty list = nothing to fix, the framework will skip the \
             reviser and judge)."
        ),
        (s, Role::Reviser) => format!(
            "Address each critic point for the {s} stage. End with a one-line summary of the changes."
        ),
        (s, Role::Judge) => format!(
            "Verify the reviser addressed each critic point for the {s} stage. Call \
             submit_verdict exactly once."
        ),
        (_, Role::QuickFixer) => "Fix the compile / test errors listed in the system prompt. \
             Use read_file / write_file / write_file_range / apply_patch, re-check with the \
             cargo_* tool, and stop as soon as the gate passes."
            .to_string(),
    }
}

/// Slots a given stage is permitted to write to (via submit_* or the
/// quickfix file-edit tools). The framework uses this to snapshot
/// before a stage runs (so we can roll back on stage failure) AND to
/// validate quickfix edits (a stage shouldn't touch slots outside its
/// scope — that's how unaudited content was leaking into the graph).
pub(crate) fn slots_owned_by_stage(stage: Stage) -> &'static [crate::render::NodeSlot] {
    use crate::render::NodeSlot::*;
    match stage {
        Stage::Architect => &[],
        Stage::Spec => &[SpecPublicMd, SpecPrivateMd],
        Stage::Iface => &[PublicRs, PrivateRs],
        Stage::Tests => &[TestsRs],
        Stage::Impl => &[PrivateRs],
        Stage::Debug => &[PrivateRs, TestsRs],
        Stage::Opt => &[PrivateRs],
    }
}

/// Opaque snapshot of a node's slot contents at stage start. Restored
/// verbatim if the stage gives up after exhausting retries.
#[derive(Debug, Default, Clone)]
struct StageSlotSnapshot {
    public_rs: Option<Option<String>>,
    private_rs: Option<Option<String>>,
    tests_rs: Option<Option<String>>,
    spec_public_md: Option<Option<String>>,
    spec_private_md: Option<Option<String>>,
}

/// Per-owner breakdown of a gate's failing errors. "Mine" = errors in
/// the current node's slots. "Upstream" = errors in OTHER nodes' slots
/// (those owners need to fix it; the current node can't from its
/// scope). "Unknown" = errors without a file path, or in files that
/// don't map to any node's slot (build script, top-level Cargo.toml,
/// etc.) — still potentially actionable from anywhere, so treated as
/// "mine-ish" for the purpose of deciding whether to invoke quickfix.
#[derive(Debug, Default, Clone)]
struct ErrorClassification {
    mine: Vec<crate::gate::CompilerError>,
    upstream: std::collections::HashMap<NodeId, Vec<crate::gate::CompilerError>>,
    upstream_names: std::collections::HashMap<NodeId, String>,
    unknown: Vec<crate::gate::CompilerError>,
}

impl ErrorClassification {
    fn mine_or_unknown(&self) -> bool {
        !self.mine.is_empty() || !self.unknown.is_empty()
    }
    fn describe_upstream(&self) -> String {
        let mut names: Vec<&str> = self
            .upstream_names
            .values()
            .map(|s| s.as_str())
            .collect();
        names.sort();
        names.join(", ")
    }
    /// Short narrative the quickfixer sees when SOME (but not all)
    /// errors are upstream — tells it explicitly which entries to leave
    /// alone so it doesn't burn turns trying to edit out-of-scope files.
    fn upstream_note(&self) -> String {
        if self.upstream.is_empty() {
            return String::new();
        }
        let names = self.describe_upstream();
        format!(
            "\n\n⚠ Some of the failing errors are in OTHER nodes' files (node(s): {names}) — \
             you cannot edit those from here. Only fix errors in YOUR node's slots; ignore \
             the upstream errors. The framework will route those to the right node."
        )
    }
}

fn classify_errors_by_owner(
    errors: &[crate::gate::CompilerError],
    current_node: NodeId,
    graph: &crate::graph::NodeGraph,
    layout: crate::render::Layout,
) -> ErrorClassification {
    let mut out = ErrorClassification::default();
    for e in errors {
        let Some(file) = &e.file else {
            out.unknown.push(e.clone());
            continue;
        };
        match crate::render::resolve_path_to_slot(graph, file, layout) {
            Some((nid, _)) if nid == current_node => out.mine.push(e.clone()),
            Some((nid, _)) => {
                if let Some(n) = graph.get(nid) {
                    out.upstream_names
                        .entry(nid)
                        .or_insert_with(|| n.name.clone());
                }
                out.upstream.entry(nid).or_default().push(e.clone());
            }
            None => out.unknown.push(e.clone()),
        }
    }
    out
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
    fn architect_runs_first_then_root_spec() {
        let mut g = NodeGraph::new();
        let _root = g.insert_root(Node::new("app", "")).unwrap();
        // Architect is ready on root immediately.
        assert!(stage_is_ready(&g, g.root.unwrap(), Stage::Architect));
        // Spec waits for architect.
        assert!(!stage_is_ready(&g, g.root.unwrap(), Stage::Spec));
        // Once architect is done, root's spec becomes ready.
        g.get_mut(g.root.unwrap()).unwrap().stages.architect = StageState::Done;
        assert!(stage_is_ready(&g, g.root.unwrap(), Stage::Spec));
        assert!(!stage_is_ready(&g, g.root.unwrap(), Stage::Iface));
    }

    #[test]
    fn architect_only_on_root() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let c = g.add_child(root, Node::new("c", "")).unwrap();
        // Even with NotStarted, architect on a non-root is never ready.
        assert!(!stage_is_ready(&g, c, Stage::Architect));
    }

    #[test]
    fn child_spec_waits_on_parent_spec() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let c = g.add_child(root, Node::new("c", "")).unwrap();
        // Architect must be done first.
        g.get_mut(root).unwrap().stages.architect = StageState::Done;
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

    fn mk_err(file: &str) -> crate::gate::CompilerError {
        crate::gate::CompilerError {
            id: "E0001".into(),
            file: Some(std::path::PathBuf::from(file)),
            line: Some(1),
            message: "boom".into(),
            raw: serde_json::Value::Null,
        }
    }

    #[test]
    fn classify_errors_separates_mine_from_upstream() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("alpha", "")).unwrap();
        let b = g.add_child(root, Node::new("beta", "")).unwrap();
        let errors = vec![
            mk_err("src/alpha/private.rs"),
            mk_err("src/beta/public.rs"),
            mk_err("src/beta/tests.rs"),
            mk_err("build.rs"), // unknown — no slot
        ];
        let c = classify_errors_by_owner(&errors, a, &g, crate::render::Layout::SingleCrate);
        assert_eq!(c.mine.len(), 1, "alpha's private.rs is mine");
        assert_eq!(c.upstream.get(&b).map(|v| v.len()), Some(2), "two errors in beta");
        assert_eq!(c.unknown.len(), 1, "build.rs is unknown");
        assert!(c.mine_or_unknown(), "has at least one mine/unknown");
        assert!(c.describe_upstream().contains("beta"));
    }

    #[test]
    fn classify_errors_all_upstream_short_circuits() {
        // The case the user hit in production: every error is in a
        // different node's files, so the quickfix loop should skip
        // invoking the model entirely.
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let a = g.add_child(root, Node::new("alpha", "")).unwrap();
        let b = g.add_child(root, Node::new("beta", "")).unwrap();
        let errors = vec![
            mk_err("src/beta/private.rs"),
            mk_err("src/beta/public.rs"),
        ];
        let c = classify_errors_by_owner(&errors, a, &g, crate::render::Layout::SingleCrate);
        assert!(c.mine.is_empty());
        assert!(c.unknown.is_empty());
        assert_eq!(c.upstream.get(&b).map(|v| v.len()), Some(2));
        assert!(!c.mine_or_unknown(), "no mine or unknown => skip quickfix");
    }
}
