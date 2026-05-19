//! The per-node-stage engine — orchestrator for the node decomposition
//! pipeline.
//!
//! Flow per stage:
//!
//! 1. Allocate a worktree off main HEAD (its `.bureau/nodes/*.json`
//!    becomes the per-task graph state).
//! 2. Run the writer turn in that worktree.
//! 3. Optionally run a critique cycle (critic → reviser → judge).
//! 4. Run the cargo gate; on failure, iterate quickfix turns.
//! 5. Acquire main_lock. Rebase the task branch onto current main HEAD.
//!    On rebase conflict: abort + abandon + retry.
//! 6. Re-run the gate on the rebased state (one more quickfix budget).
//! 7. Fast-forward main to the branch tip. Release lock, clean up.
//!
//! The invariant: main is always at a commit whose gate passes. The graph
//! lives on the worktree branch as `.bureau/graph.json` plus one file per
//! node under `.bureau/nodes/`; no shared in-memory copy.

use crate::config::Config;
use crate::gate::GateKind;
use crate::graph::{self, Node, NodeGraph, NodeId, Stage, StageState};
use crate::node_context;
use crate::prompts;
use crate::render::{self, Layout};
use crate::state::{
    EngineState, EngineTask, HistoryEntry, SchedulerState, StateHandle, TaskStatus, TokenUsage,
    UiEvent,
};
use crate::tools::{
    Critique, JudgeVerdict, Role, TaskCtx, TranscriptEntry, TranscriptKind,
};
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

// Re-export the driver types so existing callers (tests, main.rs) see
// `engine::OpenRouterDriver` / `engine::LlmDriver` / `engine::DriveParams`
// / `engine::DriveResponse` at the same path they used before. Cheap
// stability for the public surface during the refactor.
pub use crate::llm::{DriveParams, DriveResponse, LlmDriver, OpenRouterDriver};

pub struct Engine {
    pub config: Arc<Config>,
    pub state: StateHandle,
    pub workdir: PathBuf,
    pub layout: Layout,
    pub driver: Arc<dyn LlmDriver>,
    /// Serializes cargo invocations within a single workdir. Each task has
    /// its own worktree (with its own `target/`), so this is only ever
    /// contended for sequential cargos within ONE task — kept for safety.
    pub cargo_lock: Arc<tokio::sync::Mutex<()>>,
    pub workspace: Arc<crate::worktree::Workspace>,
    pub worktrees: Arc<crate::worktree::WorktreePool>,
    /// Live OpenRouter prices, fetched once at startup. Used by
    /// `compute_total_cost` to bill each task at real rates rather than
    /// hardcoded substring-matched approximations. Empty table means
    /// the fetch failed and `pricing::fallback_price` will be used.
    pub prices: Arc<crate::pricing::PriceTable>,
    /// If `Some(t)`, every in-flight LLM call sleeps until `t` before
    /// retrying. Set when ANY driver call surfaces a rate-limit error
    /// (HTTP 429, "insufficient credits", quota messages); cleared
    /// when a call succeeds after the deadline elapses. The scheduler
    /// state is flipped to `Paused` for the duration so the UI banner
    /// shows what's happening.
    pub rate_limit_until: Arc<parking_lot::Mutex<Option<std::time::Instant>>>,
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
        Self::with_driver_and_prices(config, state, driver, crate::pricing::PriceTable::default())
    }

    pub fn with_driver_and_prices(
        config: Arc<Config>,
        state: StateHandle,
        driver: Arc<dyn LlmDriver>,
        prices: crate::pricing::PriceTable,
    ) -> Result<Self> {
        let workdir = state.read(|s| s.workdir.clone());
        let layout = config.layout();
        let workspace = crate::worktree::Workspace::init(&workdir)?;
        let worktrees = Arc::new(crate::worktree::WorktreePool::new(workspace.clone())?);
        Ok(Self {
            config,
            state,
            workdir,
            layout,
            driver,
            cargo_lock: Arc::new(tokio::sync::Mutex::new(())),
            workspace,
            worktrees,
            prices: Arc::new(prices),
            rate_limit_until: Arc::new(parking_lot::Mutex::new(None)),
        })
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        self.state.write(|s| {
            s.scheduler = SchedulerState::Running;
        });
        self.state.emit(UiEvent::SchedulerStateChanged {
            state: SchedulerState::Running,
        });

        self.ensure_root_seeded()?;
        // After ensure_root_seeded (which creates main if needed),
        // sweep any leftover InProgress stages from a crashed prior
        // run. See `reset_stale_inprogress` for the failure mode.
        self.reset_stale_inprogress()?;

        let max_parallel = self.config.toml.limits.max_parallel_tasks.max(1);
        let mut joinset: tokio::task::JoinSet<Result<()>> = tokio::task::JoinSet::new();
        let mut total_tasks = 0usize;
        let mut first_error: Option<anyhow::Error> = None;

        // Tracks WHY we broke out of the loop, so the final return can
        // distinguish "all stages done" (Ok) from "halted on budget"
        // (Err with a descriptive message). Without this the budget
        // halt path silently returned Ok and main logged "pipeline
        // complete" — operators thought work was done when it wasn't.
        let mut halt_reason: Option<String> = None;
        loop {
            if let Some(cap) = self.config.toml.limits.cost_cap_usd {
                let est = self.state.read(|s| s.estimated_cost_usd);
                if est >= cap {
                    let msg = format!("halting: cost cap ${cap:.2} reached at ${est:.4}");
                    self.note(msg.clone());
                    halt_reason = Some(msg);
                    self.state.write(|s| s.scheduler = SchedulerState::Stopped);
                    self.state.emit(UiEvent::SchedulerStateChanged {
                        state: SchedulerState::Stopped,
                    });
                    break;
                }
            }
            if total_tasks >= self.config.toml.limits.max_tasks_total {
                let msg = format!(
                    "halting: max_tasks_total ({}) reached",
                    self.config.toml.limits.max_tasks_total
                );
                self.note(msg.clone());
                halt_reason = Some(msg);
                self.state.write(|s| s.scheduler = SchedulerState::Stopped);
                self.state.emit(UiEvent::SchedulerStateChanged {
                    state: SchedulerState::Stopped,
                });
                break;
            }

            // Fill the slot pool.
            while joinset.len() < max_parallel {
                let Some((node_id, stage)) = self.pick_next_ready() else {
                    break;
                };
                // Mark InProgress eagerly in main's graph so subsequent
                // picks skip it. Concurrent tasks see this through main's
                // .bureau/nodes/*.json on their next allocate.
                self.set_stage_on_main(node_id, stage, StageState::InProgress).await?;
                let this = self.clone();
                joinset.spawn(async move { this.advance_stage(node_id, stage).await });
                total_tasks += 1;
            }

            if joinset.is_empty() {
                if self.all_done()? {
                    self.note("pipeline complete");
                    self.state.write(|s| s.scheduler = SchedulerState::Done);
                    self.state.emit(UiEvent::SchedulerStateChanged {
                        state: SchedulerState::Done,
                    });
                    break;
                }
                let blockers = self.diagnose_stuck();
                self.note(format!(
                    "no ready stages and not done; halting. Blockers:\n{blockers}"
                ));
                self.state.write(|s| s.scheduler = SchedulerState::Stopped);
                return Err(first_error.unwrap_or_else(|| {
                    anyhow!("scheduler stuck — no ready stages remain. Blockers:\n{blockers}")
                }));
            }

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

        while let Some(joined) = joinset.join_next().await {
            if let Err(je) = joined {
                tracing::error!("advance_stage join error during drain: {je}");
            }
        }

        match (first_error, halt_reason) {
            (Some(e), _) => Err(e),
            (None, Some(reason)) => Err(anyhow!("{reason}")),
            (None, None) => Ok(()),
        }
    }

    fn ensure_root_seeded(&self) -> Result<()> {
        let mut g = graph::load(&self.workdir, self.layout)?;
        if g.root.is_some() {
            return Ok(());
        }
        let desc = prompts::problem_first_paragraph(&self.config.problem);
        let mut root = Node::new(self.config.toml.project_name.as_str(), desc);
        root.crate_boundary = true;
        let _ = g.insert_root(root)?;
        render::render_graph(&self.workdir, &g, self.layout)?;
        if let Err(e) = self.workspace.commit_main("scaffold: initial render") {
            tracing::warn!("scaffold commit: {e:#}");
        }
        // The root just appeared; tell connected UIs to refetch the
        // graph so the tree renders without waiting for the safety-net
        // resync interval.
        self.state.emit(UiEvent::GraphTopologyChanged);
        Ok(())
    }

    /// On startup, any stage left at `InProgress` on disk belongs to a
    /// task that crashed (no engine task is running it — we just
    /// rebuilt `EngineState` from scratch). Reset those stages to
    /// `NotStarted`, AND cascade-reset every later stage on the same
    /// node, AND wipe the on-disk slot files for those cascaded stages
    /// back to placeholders.
    ///
    /// Why cascade + wipe: a crash in (say) the Iface stage leaves
    /// `tests.rs` on disk from a prior run. If we only reset Iface's
    /// state, the next Iface run sees an authored `tests.rs` already on
    /// disk; the cargo gate compiles those tests against the new
    /// (incomplete) iface and the agent gets confused trying to fix
    /// tests that shouldn't even exist yet. Wiping downstream slots
    /// makes restart equivalent to a fresh start from the broken stage.
    ///
    /// Without this, restarted runs hang: the UI shows nodes "running"
    /// (because the on-disk graph says so) but no task ever picks them
    /// up because `stage_is_ready` requires `cur == NotStarted` and
    /// `InProgress` fails that check.
    fn reset_stale_inprogress(&self) -> Result<()> {
        let mut g = graph::load(&self.workdir, self.layout)?;
        // Per node, find the EARLIEST InProgress stage. The cascade
        // helper handles everything from there forward, so we don't
        // need a separate entry per stage on the same node.
        let mut targets: Vec<(crate::graph::NodeId, Stage)> = Vec::new();
        for (id, node) in &g.nodes {
            for stage in Stage::ALL {
                if node.stages.get(stage) == StageState::InProgress {
                    targets.push((*id, stage));
                    break;
                }
            }
        }
        if targets.is_empty() {
            return Ok(());
        }
        let mut summaries: Vec<String> = Vec::new();
        for (id, from) in targets {
            let name = g.get(id).map(|n| n.name.clone()).unwrap_or_default();
            let node = g.get_mut(id).expect("target was just found in graph");
            let changed = graph::reset_stage_and_cascade(node, from);
            let stages_str = changed
                .iter()
                .map(|(s, _)| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            summaries.push(format!("{name}({stages_str})"));
        }
        graph::save(&self.workdir, &g)?;
        // Re-render so cleared content slots become placeholder files
        // on disk again — leaving the prior authored content there
        // would defeat the whole point of the cascade-reset.
        render::render_graph(&self.workdir, &g, self.layout)?;
        let msg = format!(
            "restart: reset {} stale InProgress node(s) and cascaded downstream: {}",
            summaries.len(),
            summaries.join(", ")
        );
        self.note(msg.clone());
        if let Err(e) = self.workspace.commit_main(&msg) {
            tracing::warn!("commit stale-inprogress reset: {e:#}");
        }
        // Stages and slot files changed — the UI's view of the
        // tree (stage badges) is now stale. The topology itself
        // didn't change but enough did to warrant a refetch.
        self.state.emit(UiEvent::GraphTopologyChanged);
        Ok(())
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

    /// Set a node's stage state on the MAIN workdir's graph and commit
    /// the change. Used to mark stages InProgress / Done / Failed in
    /// main's view so other tasks see the new state.
    /// Update one (node, stage) in main's graph. Takes `main_lock` so
    /// it doesn't race with itself (parallel tasks completing) or with
    /// an in-flight `rebase_branch_onto_main + fast_forward_main`.
    /// Without the lock, two completions can both load main, mutate
    /// different stages, save — and the later save's per-node-file
    /// write would still be correct, BUT the topology index commit
    /// races with ff-merge's commit and either can clobber the other's
    /// pending tree.
    async fn set_stage_on_main(
        &self,
        node_id: NodeId,
        stage: Stage,
        state: StageState,
    ) -> Result<()> {
        let _guard = self.worktrees.main_lock().lock().await;
        let mut g = graph::load(&self.workdir, self.layout)?;
        let node_name = g.get(node_id).map(|n| n.name.clone()).unwrap_or_default();
        if let Some(n) = g.get_mut(node_id) {
            n.stages.set(stage, state);
        }
        graph::save(&self.workdir, &g)?;
        let verb = match state {
            StageState::NotStarted => "reset",
            StageState::InProgress => "start",
            StageState::Done => "done",
            StageState::Failed => "fail",
        };
        let _ = self
            .workspace
            .commit_main(&format!("{node_name}/{stage}: {verb}"));
        Ok(())
    }

    /// Describe what's preventing the scheduler from picking a stage.
    /// Called from the halt path so the operator can see WHICH nodes
    /// are blocked and on WHAT. Otherwise the "scheduler stuck" message
    /// gives no useful information.
    fn diagnose_stuck(&self) -> String {
        let g = match graph::load(&self.workdir, self.layout) {
            Ok(g) => g,
            Err(e) => return format!("  (could not load graph: {e:#})"),
        };
        let mut out = String::new();
        for n in g.iter() {
            for stage in [
                Stage::Architect,
                Stage::Spec,
                Stage::Iface,
                Stage::Tests,
                Stage::Impl,
                Stage::Debug,
            ] {
                let st = n.stages.get(stage);
                if st == StageState::Done {
                    continue;
                }
                if stage_is_ready(&g, n.id, stage) {
                    continue;
                }
                // Stage isn't done and isn't ready — say why.
                let reason = match st {
                    StageState::Failed => format!("{stage} is Failed (terminal)"),
                    StageState::InProgress => format!("{stage} is in flight (task didn't release?)"),
                    _ => {
                        // NotStarted but not ready — find the blocker.
                        let mut blockers: Vec<String> = Vec::new();
                        for dep in &n.deps {
                            if let Some(d) = g.get(*dep) {
                                let dep_stage_done = match stage {
                                    Stage::Impl | Stage::Debug => d.stages.impl_.is_done(),
                                    _ => d.stages.iface.is_done(),
                                };
                                if !dep_stage_done {
                                    blockers.push(format!("dep `{}`", d.name));
                                }
                            }
                        }
                        if blockers.is_empty() {
                            format!("{stage} waiting on parent/own earlier stages")
                        } else {
                            format!("{stage} waiting on {}", blockers.join(", "))
                        }
                    }
                };
                out.push_str(&format!("  - node `{}`: {reason}\n", n.name));
            }
        }
        if out.is_empty() {
            "  (no obvious blockers; this might be an empty pipeline)".into()
        } else {
            out
        }
    }

    fn pick_next_ready(&self) -> Option<(NodeId, Stage)> {
        let g = graph::load(&self.workdir, self.layout).ok()?;
        let order = g.topo_order()?;
        for stage in [
            Stage::Architect,
            Stage::Spec,
            Stage::Iface,
            Stage::Tests,
            Stage::Impl,
            Stage::Debug,
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

    fn all_done(&self) -> Result<bool> {
        let g = graph::load(&self.workdir, self.layout)?;
        if let Some(rid) = g.root {
            if !g
                .get(rid)
                .map(|r| r.stages.architect.is_done())
                .unwrap_or(false)
            {
                return Ok(false);
            }
        }
        for n in g.iter() {
            for s in [Stage::Spec, Stage::Iface, Stage::Tests, Stage::Impl] {
                if !n.stages.get(s).is_done() {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Drive a (node, stage) to completion: allocate a worktree, run the
    /// writer/critique/quickfix cycle in it, then rebase + ff-merge it to
    /// main. Retries up to `max_stage_retries` on failure.
    async fn advance_stage(self: Arc<Self>, node_id: NodeId, stage: Stage) -> Result<()> {
        let max_retries = self.config.toml.limits.max_stage_retries;
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 1..=(max_retries + 1) {
            let wt = match self.worktrees.allocate(Uuid::new_v4()).await {
                Ok(wt) => wt,
                Err(e) => {
                    tracing::error!("worktree allocation: {e:#}");
                    return Err(e);
                }
            };
            match self
                .clone()
                .run_attempt_in_worktree(node_id, stage, &wt, attempt)
                .await
            {
                Ok(()) => {
                    self.set_stage_on_main(node_id, stage, StageState::Done).await?;
                    self.state.emit(UiEvent::NodeChanged { id: node_id });
                    return Ok(());
                }
                Err(e) => {
                    self.note(format!(
                        "node `{}` stage `{}` attempt {}/{} failed: {e:#}",
                        self.node_name(node_id).unwrap_or_default(),
                        stage,
                        attempt,
                        max_retries + 1
                    ));
                    // run_attempt_in_worktree owns wt cleanup on its err
                    // path; just record the error.
                    last_err = Some(e);
                }
            }
        }
        self.set_stage_on_main(node_id, stage, StageState::Failed).await?;
        self.state.emit(UiEvent::NodeChanged { id: node_id });
        // Impl failures get picked up by the Debug stage; don't bubble.
        if stage == Stage::Impl {
            return Ok(());
        }
        Err(last_err.unwrap_or_else(|| anyhow!("stage {stage} exhausted retries")))
    }

    /// One attempt at advancing a (node, stage) inside `wt`. On success:
    /// rebases, re-gates, ff-merges, and cleans up. On failure: abandons
    /// the worktree and bubbles the error to the caller for retry.
    async fn run_attempt_in_worktree(
        self: Arc<Self>,
        node_id: NodeId,
        stage: Stage,
        wt: &crate::worktree::Worktree,
        attempt: u32,
    ) -> Result<()> {
        let inner = self
            .clone()
            .attempt_inner(node_id, stage, wt, attempt)
            .await;
        if let Err(e) = inner {
            // Always abandon the worktree on failure so the next retry
            // starts fresh from main.
            if let Err(ae) = self.worktrees.abandon(wt.clone()).await {
                tracing::warn!("worktree abandon: {ae:#}");
            }
            return Err(e);
        }
        // Inner succeeded → it's already abandoned the wt as part of
        // landing. Nothing left to do.
        Ok(())
    }

    async fn attempt_inner(
        self: Arc<Self>,
        node_id: NodeId,
        stage: Stage,
        wt: &crate::worktree::Worktree,
        attempt: u32,
    ) -> Result<()> {
        let task_id = Uuid::new_v4();
        let node_name = self
            .node_name(node_id)
            .ok_or_else(|| anyhow!("node {node_id} missing"))?;
        let writer_model = self
            .config
            .toml
            .models
            .for_stage_role(stage, Role::Writer, attempt)
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

        let gate_kind = gate_kind_for(stage);

        // 1. Writer turn.
        let actor = self
            .clone()
            .run_role(task_id, node_id, stage, Role::Writer, None, &wt.path, attempt)
            .await?;

        // 2. Optional critique cycle (skip on architect — single-shot).
        let critique_retries = if stage == Stage::Architect {
            0
        } else {
            self.config.toml.limits.critique_retries
        };
        let mut last_text = actor.text;
        let mut last_failed = actor.failed_tools;
        for round in 1..=critique_retries {
            let critic = self
                .clone()
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
                    &wt.path,
                    attempt,
                )
                .await?;
            let (critique_text, skip_rest) = match critic.critique {
                Some(c) if c.is_clean() => {
                    self.note(format!(
                        "task {task_id} round {round}: critic clean — skipping reviser/judge"
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
                .clone()
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
                    &wt.path,
                    attempt,
                )
                .await?;
            last_text = revision.text.clone();
            last_failed = revision.failed_tools.clone();
            let _judge = self
                .clone()
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
                    &wt.path,
                    attempt,
                )
                .await?;
            let v = self.state.read(|s| {
                s.tasks.get(&task_id).and_then(|t| t.final_verdict.clone())
            });
            if matches!(v, Some(JudgeVerdict { satisfactory: true, .. })) {
                break;
            }
        }

        // 3. Final verdict gate.
        let final_v = self
            .state
            .read(|s| s.tasks.get(&task_id).and_then(|t| t.final_verdict.clone()));
        if let Some(v) = final_v {
            if !v.satisfactory {
                let msg = format!(
                    "judge rejected node `{node_name}` stage `{stage}`: {}",
                    v.reason
                );
                self.fail_task(task_id, msg.clone());
                return Err(anyhow!(msg));
            }
        }

        // 4. Spec stage produces no cargo content; just confirm it wrote
        //    spec_public_md.
        if stage == Stage::Spec {
            let g = graph::load(&wt.path, self.layout)?;
            let n = g.get(node_id).ok_or_else(|| anyhow!("node missing in wt graph"))?;
            if n.spec_public_md.is_none() {
                let msg = format!("node `{node_name}` spec produced no public.md");
                self.fail_task(task_id, msg.clone());
                return Err(anyhow!(msg));
            }
        }

        // 5. Pre-land gate + quickfix.
        if let Some(kind) = gate_kind {
            self.clone()
                .gate_with_quickfix(task_id, node_id, stage, kind, &wt.path, attempt)
                .await
                .with_context(|| format!("pre-land gate failed for {node_name} {stage}"))?;
        }

        // 6. Commit + land: rebase → re-gate → fast-forward.
        let landing_msg = format!("{stage}: {node_name}");
        self.worktrees
            .commit_in_worktree(wt, &landing_msg)
            .context("commit worktree before landing")?;
        let _main = self.worktrees.main_lock().lock().await;
        self.workspace
            .rebase_branch_onto_main(&wt.path, &wt.branch)
            .context("rebase onto main")?;
        if let Some(kind) = gate_kind {
            // After rebase: the worktree now sees other tasks' landed
            // content too. Re-render from the (now merged) graph + gate.
            let g = graph::load(&wt.path, self.layout)?;
            render::render_graph(&wt.path, &g, self.layout)?;
            self.worktrees
                .commit_in_worktree(wt, &format!("rerender after rebase: {node_name} {stage}"))?;
            self.clone()
                .gate_with_quickfix(task_id, node_id, stage, kind, &wt.path, attempt)
                .await
                .with_context(|| {
                    format!("post-rebase gate failed for {node_name} {stage}")
                })?;
            // Quickfix may have introduced more commits; let
            // commit_in_worktree wrap up.
            self.worktrees
                .commit_in_worktree(wt, &format!("quickfix post-rebase: {node_name} {stage}"))?;
        }
        self.workspace.fast_forward_main(&wt.branch)?;
        drop(_main);

        // Done. Tear down the worktree.
        if let Err(e) = self.worktrees.abandon(wt.clone()).await {
            tracing::warn!("worktree abandon: {e:#}");
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

    fn fail_task(&self, task_id: Uuid, msg: String) {
        self.state.write(|s| {
            if let Some(t) = s.tasks.get_mut(&task_id) {
                t.status = TaskStatus::Failed;
                t.finished_at = Some(Utc::now());
                t.error = Some(msg);
            }
        });
        self.state.emit(UiEvent::TaskStatusChanged {
            id: task_id,
            status: TaskStatus::Failed,
        });
    }

    fn node_name(&self, node_id: NodeId) -> Option<String> {
        let g = graph::load(&self.workdir, self.layout).ok()?;
        g.get(node_id).map(|n| n.name.clone())
    }

    /// Run the gate; if it fails, run quickfix turns (up to
    /// `max_quickfix_iters`), re-checking the gate after each. Returns
    /// Err if the gate is still failing after the loop.
    async fn gate_with_quickfix(
        self: Arc<Self>,
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        kind: GateKind,
        wt_path: &Path,
        attempt: u32,
    ) -> Result<()> {
        let max_iters = self.config.toml.limits.max_quickfix_iters;
        for iter in 0..=max_iters {
            let outcome = {
                let _g = self.cargo_lock.lock().await;
                crate::gate::run_gate(wt_path, kind).await?
            };
            if outcome.passed {
                if iter > 0 {
                    self.note(format!(
                        "task {task_id} stage {stage}: gate passed after {iter} quickfix iter(s)"
                    ));
                }
                return Ok(());
            }
            if iter == max_iters {
                let summary = summarize_errors(&outcome.errors, 8);
                return Err(anyhow!(
                    "cargo {} still failing after {max_iters} quickfix iter(s):\n{summary}",
                    kind.label()
                ));
            }
            let summary = summarize_errors(&outcome.errors, 12);
            self.note(format!(
                "task {task_id} stage {stage}: quickfix iter {}/{max_iters} — gate failed:\n{summary}",
                iter + 1
            ));
            let _ = self
                .clone()
                .run_role(
                    task_id,
                    node_id,
                    stage,
                    Role::QuickFixer,
                    Some(CycleExtras {
                        round: iter + 1,
                        quickfix_gate_output: Some(summary),
                        quickfix_iter: Some((iter + 1, max_iters - iter - 1)),
                        ..Default::default()
                    }),
                    wt_path,
                    attempt,
                )
                .await?;
        }
        // unreachable in practice (loop body returns)
        Ok(())
    }

    async fn drive_with_transient_retry(
        self: &Arc<Self>,
        params: DriveParams,
        ctx: Arc<TaskCtx>,
    ) -> Result<DriveResponse> {
        const MAX_TRANSIENT_RETRIES: u32 = 3;
        const RATE_LIMIT_BACKOFF: std::time::Duration = std::time::Duration::from_secs(60);
        let mut attempt = 0u32;
        let mut was_paused = false;
        loop {
            // Honor any in-flight rate-limit pause. Multiple tasks waiting
            // here all serve as periodic "probes": when one wakes up and
            // succeeds, the pause clears and the others naturally resume.
            // Snapshot the deadline out from under the lock before
            // awaiting (parking_lot MutexGuard isn't Send).
            let snapshot: Option<std::time::Instant> = *self.rate_limit_until.lock();
            if let Some(t) = snapshot {
                let now = std::time::Instant::now();
                if now < t {
                    let wait = (t - now).min(std::time::Duration::from_secs(10));
                    tokio::time::sleep(wait).await;
                    was_paused = true;
                    continue;
                }
                // Deadline elapsed — clear and probe.
                *self.rate_limit_until.lock() = None;
            }
            match self.driver.drive(params.clone(), ctx.clone()).await {
                Ok(resp) => {
                    if was_paused {
                        self.note("rate-limit pause cleared — pipeline resuming");
                        self.state.write(|s| s.scheduler = SchedulerState::Running);
                        self.state.emit(UiEvent::SchedulerStateChanged {
                            state: SchedulerState::Running,
                        });
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    let msg = format!("{:#}", e);
                    if is_rate_limited(&msg) {
                        // Set/extend the global pause, flip the scheduler
                        // to Paused so the UI shows what's happening.
                        let until = std::time::Instant::now() + RATE_LIMIT_BACKOFF;
                        {
                            let mut l = self.rate_limit_until.lock();
                            if l.map_or(true, |stored| stored < until) {
                                *l = Some(until);
                            }
                        }
                        self.state.write(|s| s.scheduler = SchedulerState::Paused);
                        self.state.emit(UiEvent::SchedulerStateChanged {
                            state: SchedulerState::Paused,
                        });
                        self.note(format!(
                            "rate-limited ({}): pausing all calls for {}s",
                            params.model,
                            RATE_LIMIT_BACKOFF.as_secs()
                        ));
                        was_paused = true;
                        continue;
                    }
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
        self: Arc<Self>,
        task_id: Uuid,
        node_id: NodeId,
        stage: Stage,
        role: Role,
        extras: Option<CycleExtras>,
        wt_path: &Path,
        attempt: u32,
    ) -> Result<RoleOutcome> {
        let model = self
            .config
            .toml
            .models
            .for_stage_role(stage, role, attempt)
            .to_string();

        // Build prompt context from the worktree's graph state.
        // Project mission + style guide are NOT pushed into the user
        // prompt bundle anymore — they're truly project-stable (same
        // bytes for every call across the whole run) so they live in
        // the system prompt below for maximum cache reuse.
        let g_for_ctx = graph::load(wt_path, self.layout).unwrap_or_default();
        let mut bundle = node_context::ContextBundle::new();
        let inner = node_context::build_for_stage(
            &g_for_ctx,
            node_id,
            stage,
            self.config.toml.limits.max_nodes,
            self.config.toml.limits.max_node_depth,
            self.layout,
        );
        bundle.extend_from(inner);
        // Cycle context (retry failures, previous critic+reviser output)
        // is NOT pushed into the bundle anymore — it's the most volatile
        // piece of state (changes per retry) and we append it at the
        // very end of the user prompt below, so the prefix through the
        // role block + role instruction can still cache across retries.
        let cycle_ctx = extras.as_ref().map(|ex| {
            build_cycle_context(ex, self.config.toml.limits.args_display_cap)
        });
        drop(g_for_ctx);

        let prompt_limits = crate::tools::PromptLimits {
            max_file_lines: self.config.toml.limits.max_file_lines,
            max_spec_section_lines: self.config.toml.limits.max_spec_section_lines,
        };
        // Cache-friendly prompt layout, nested-stable to volatile:
        //
        //   system prompt =
        //     universal_preamble()       // framework-wide rules
        //     Project mission            // project-stable (same bytes
        //                                // for every call across the
        //                                // whole run)
        //     Style guide (if any)       // project-stable
        //     (Truly stable: every call in the entire run uses this
        //     exact byte string. Cache hit across nodes, stages, roles.)
        //
        //   user prompt =
        //     <context_doc>            // node-stable bits (ancestor chain,
        //                              // siblings, this node, own slots —
        //                              // same across stages on this node
        //                              // up to first stage divergence)
        //     <role_block>             // per-(stage,role) instructions
        //                              // (cache breaks here across stages
        //                              // or roles, but everything before
        //                              // it remains shared)
        //     <role_instruction>       // tiny per-(stage,role) imperative
        //     <cycle context if any>   // per-retry; appended last so
        //                              // retries still share prefix up
        //                              // through role_instruction
        //
        // Previously the role preamble lived in the system prompt and
        // the role instruction sat at position 0 of the user prompt.
        // Both diverge per stage → no cross-stage cache reuse on the
        // same node, even though the context is otherwise identical.
        let system_prompt = {
            let mut s = String::from(prompts::universal_preamble());
            // Trim trailing newlines on the preamble before appending so
            // we get exactly one blank line between sections regardless
            // of how the .md files end.
            while s.ends_with('\n') {
                s.pop();
            }
            s.push_str("\n\n# Project mission\n\n");
            s.push_str(self.config.problem.trim());
            if let Some(style) = &self.config.style {
                s.push_str("\n\n# Style guide\n\n");
                s.push_str(style.trim());
            }
            s.push('\n');
            s
        };
        let context_doc = bundle.to_markdown();
        let role_block_str = prompts::role_block(stage, role, prompt_limits);
        let role_instruction = prompts::role_user_prompt(stage, role);
        // The provider sees the same tool list at every (stage, role)
        // call (see `tools::unified_tool_names`) — that keeps the
        // tool-schemas portion of the request byte-stable so the prompt
        // cache survives across stages. But not every tool *works* at
        // every (stage, role): the framework's `require_stage` checks
        // reject inappropriate calls at runtime. This block tells the
        // model exactly which subset is live this turn so it doesn't
        // burn turns on tools that will be rejected.
        let tools_block = build_tools_eligibility(stage, role);
        let user_prompt = match &cycle_ctx {
            Some(c) => format!(
                "{context_doc}\n---\n\n{role_block_str}\n\n{tools_block}\n\n{role_instruction}\n\n# Critique cycle context\n\n{c}",
            ),
            None => format!(
                "{context_doc}\n---\n\n{role_block_str}\n\n{tools_block}\n\n{role_instruction}",
            ),
        };

        // Record system + tool defs + user prompt entries.
        let now = Utc::now();
        let tool_defs = crate::tools::tool_definitions_for(stage, role, prompt_limits);
        let mut entries: Vec<(TranscriptKind, String)> = Vec::new();
        entries.push((TranscriptKind::System, system_prompt.clone()));
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

        // TaskCtx no longer caches the graph in memory — tools load it
        // fresh from the worktree's `.bureau/` on each call. Rig
        // serializes tool dispatch (concurrency=1) so two tool calls
        // in the same turn can't race on the on-disk state.
        let ctx = Arc::new(
            TaskCtx::new(
                task_id,
                node_id,
                stage,
                role,
                wt_path.to_path_buf(),
                self.layout,
                self.config.toml.limits.max_file_lines,
                self.config.toml.limits.max_spec_section_lines,
                self.config.toml.limits.max_nodes,
                self.config.toml.limits.max_node_depth,
                self.cargo_lock.clone(),
            )
            // Wire the live state handle: tools live-write tool_call /
            // tool_result entries into canonical state + broadcast over
            // SSE; the streaming LLM driver pushes AssistantChunk text
            // deltas. Without this, the UI sees nothing during the
            // multi-minute LLM call.
            .with_state(self.state.clone()),
        );

        let params = DriveParams {
            model: model.clone(),
            preamble: system_prompt.clone(),
            user_prompt: user_prompt.clone(),
            stage,
            role,
            max_tokens: self.config.toml.models.max_tokens,
            temperature: self.config.toml.models.temperature,
            max_turns: self.config.toml.models.max_turns,
        };
        let resp = self.drive_with_transient_retry(params, ctx.clone()).await?;

        // Forced-retry loop for unresolved tool failures.
        let mut total_usage = resp.usage.clone();
        let mut combined_text = resp.output.clone();
        let max_forced_retries = self.config.toml.limits.tool_retry_budget;
        for forced_attempt in 1..=max_forced_retries {
            let snapshot = ctx.snapshot_transcript();
            let unresolved = collect_failed_tool_calls(&snapshot);
            if unresolved.is_empty() {
                break;
            }
            let remaining = max_forced_retries - forced_attempt;
            let retry_preamble_text = prompts::retry_preamble(
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
            let retry_resp = self
                .drive_with_transient_retry(retry_params, ctx.clone())
                .await?;
            total_usage.add(&retry_resp.usage);
            combined_text.push_str(&format!(
                "\n\n---\n[forced retry {forced_attempt}]\n{}",
                retry_resp.output
            ));
        }

        let usage = total_usage;
        let final_entry = TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::AssistantText,
            content: combined_text.clone(),
            role: Some(role),
        };
        let ctx_entries = ctx.drain_transcript();
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
        let verdict = ctx.take_verdict();
        let critique = ctx.take_critique();
        let fs_events: Vec<PathBuf> = ctx.drain_fs_events();

        // Tool entries already live-appended to t.transcript by the
        // tools themselves (see `TaskCtx::live_append_transcript`) and
        // already broadcast over SSE. We only push the assistant_text
        // entry here (it's the final aggregated text and isn't
        // associated with any individual tool call). ctx_entries was
        // drained earlier just to feed `collect_failed_tool_calls`;
        // do NOT extend t.transcript with it — that would duplicate
        // everything the tools already wrote.
        //
        // For cost: the streaming driver already applied `usage` to
        // state per-turn via `live_apply_partial_cost`. Only add it
        // here if the driver returned `applied_via_streaming = false`
        // (mock driver in tests). Adding twice would double-count.
        let transcript_cap = self.config.toml.limits.task_transcript_cap;
        let applied_via_streaming = resp.applied_via_streaming;
        self.state.write(|s| {
            if let Some(t) = s.tasks.get_mut(&task_id) {
                t.transcript.push(final_entry.clone());
                if !applied_via_streaming {
                    t.cost.add(&usage);
                }
                if matches!(role, Role::Judge) {
                    t.final_verdict = verdict.clone();
                }
                crate::state::cap_transcript(t, transcript_cap);
                if !applied_via_streaming {
                    s.total_cost.add(&usage);
                }
                s.estimated_cost_usd = compute_total_cost(s, &self.prices);
            }
        });
        let _ = ctx_entries; // already used for failed-tool detection
        self.state.emit(UiEvent::TranscriptAppended {
            task_id,
            entry: final_entry,
        });
        for path in fs_events {
            self.state.emit(UiEvent::FileChanged { path });
        }
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
        self.state.emit(UiEvent::NodeChanged { id: node_id });

        Ok(RoleOutcome {
            text: combined_text,
            failed_tools,
            critique,
        })
    }
}

#[derive(Debug, Clone, Default)]
struct RoleOutcome {
    text: String,
    failed_tools: Vec<(String, String, String)>,
    critique: Option<Critique>,
}

#[derive(Debug, Clone, Default)]
struct CycleExtras {
    round: u32,
    prior_actor_text: Option<String>,
    prior_critique: Option<String>,
    prior_revision: Option<String>,
    prior_failed_tools: Vec<(String, String, String)>,
    quickfix_gate_output: Option<String>,
    quickfix_iter: Option<(u32, u32)>,
}

/// Render the "Tools eligible at this turn" block — the model sees the
/// full unified tool list in the API request (for cache stability), but
/// only a subset is actually valid at this (stage, role). This block
/// tells it which subset. Format kept short and stable so it doesn't
/// itself bust the prompt cache more than it has to.
///
/// QuickFixer is a separate mode with its own attached tool list (no
/// unified catalog applies), so we render a simpler block there.
fn build_tools_eligibility(stage: Stage, role: Role) -> String {
    let accepted = crate::tools::tools_accepted_at(stage, role);
    let accepted_str = accepted
        .iter()
        .map(|n| format!("`{n}`"))
        .collect::<Vec<_>>()
        .join(", ");
    if matches!(role, Role::QuickFixer) {
        // Quickfix has its own attached tool list — there's nothing to
        // "reject" because the unified catalog isn't in play here.
        return format!(
            "# Tools eligible at this turn\n\n\
             QUICKFIX mode for the **{stage}** stage. All attached tools \
             are eligible: {accepted_str}\n"
        );
    }
    let unified = crate::tools::unified_tool_names();
    let rejected: Vec<&'static str> = unified
        .iter()
        .copied()
        .filter(|n| !accepted.contains(n))
        .collect();
    let rejected_str = if rejected.is_empty() {
        "(none — every tool in the list is eligible this turn)".to_string()
    } else {
        rejected
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "# Tools eligible at this turn\n\n\
         The tool schemas you see attached include every tool the framework \
         knows about. Only the subset below will actually do anything at this \
         **{stage}** stage / **{role:?}** role — the rest will be rejected by \
         the framework's stage/role gate.\n\n\
         **Eligible (call these):** {accepted_str}\n\n\
         **Rejected if called (don't bother):** {rejected_str}\n"
    )
}

fn build_cycle_context(ex: &CycleExtras, args_cap: usize) -> String {
    let mut cyc = String::new();
    cyc.push_str(&format!("Round {}\n\n", ex.round));
    if let Some((iter, remaining)) = ex.quickfix_iter {
        cyc.push_str(&format!(
            "## Quickfix iteration {iter} ({remaining} remaining)\n\n"
        ));
    }
    if let Some(out) = &ex.quickfix_gate_output {
        cyc.push_str("## ⚠ Failing cargo gate — fix these\n\n```\n");
        cyc.push_str(out);
        cyc.push_str("\n```\n\n");
    }
    if !ex.prior_failed_tools.is_empty() {
        cyc.push_str(
            "## ⚠ Prior turn had failed tool calls — these were NOT applied\n\n\
             The previous turn called these tools but they returned errors. \
             The intent behind each call was lost. Address every entry: read \
             the error, fix the args, and retry the call (or, if the goal \
             was actually wrong, explain in your own output why it should be \
             dropped).\n\n",
        );
        for (tool, args, err) in &ex.prior_failed_tools {
            let args_display = prompts::truncate_args_for_display(args, args_cap);
            cyc.push_str(&format!(
                "- **`{tool}`** — args: `{args_display}` — error: {err}\n"
            ));
        }
        cyc.push('\n');
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
    cyc
}

fn gate_kind_for(stage: Stage) -> Option<GateKind> {
    match stage {
        // Architect commits the whole rendered tree; Spec can call
        // decompose which re-renders ancestor mod.rs / Cargo.toml and
        // writes placeholder Rust files for newly-added children.
        // Either can produce a tree that fails to compile (bad
        // architect tree, malformed Cargo.toml entry, duplicate
        // member names, etc.). Without a gate here, the broken state
        // lands on main and every downstream stage's critic sees
        // "pre-existing errors" it has to ignore. `Check` is cheap
        // when nothing changed (cargo's incremental compile is a
        // no-op on a clean tree) and triggers the quickfix loop when
        // it isn't.
        Stage::Architect | Stage::Spec => Some(GateKind::Check),
        Stage::Iface => Some(GateKind::Check),
        Stage::Tests => Some(GateKind::TestNoRun),
        Stage::Impl | Stage::Debug => Some(GateKind::Test),
    }
}

fn summarize_errors(errors: &[crate::gate::CompilerError], max: usize) -> String {
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

/// Scan a slice of transcript entries and return UNRESOLVED tool failures —
/// for each tool name, only the LAST result is considered.
fn collect_failed_tool_calls(entries: &[TranscriptEntry]) -> Vec<(String, String, String)> {
    use std::collections::HashMap;
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
        .filter_map(|(tool, (idx, ok, args, err))| {
            if ok { None } else { Some((idx, tool, args, err)) }
        })
        .collect();
    failures.sort_by_key(|(idx, ..)| *idx);
    failures.into_iter().map(|(_, t, a, e)| (t, a, e)).collect()
}

/// HTTP 429 / quota / daily-limit errors. These should NOT count
/// against per-call retry budgets; the engine pauses all tasks until
/// the rate-limit window passes, then resumes.
fn is_rate_limited(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("429")
        || m.contains("rate limit")
        || m.contains("rate-limit")
        || m.contains("rate_limit")
        || m.contains("too many requests")
        || m.contains("quota")
        || m.contains("insufficient credits")
        || m.contains("daily limit")
        || m.contains("usage limit")
        || m.contains("you exceeded your current quota")
}

fn is_transient(msg: &str) -> bool {
    // 429 is intentionally NOT here — it belongs to `is_rate_limited`,
    // which the caller checks first. Rate-limits get a long, shared
    // pause across the whole engine; mixing 429 into the short
    // exponential-backoff transient retry would burn the budget
    // pointlessly.
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
        || msg.contains("ECONNRESET")
}

/// Compute the total dollar cost spent so far across every task.
/// Uses live OpenRouter prices when available, falling back to
/// `pricing::fallback_price` for any model the live table doesn't know
/// about. Cache-read and cache-write tokens are billed at the model's
/// own cache rates (no more hardcoded Claude-specific 0.1x/1.25x).
fn compute_total_cost(state: &EngineState, prices: &crate::pricing::PriceTable) -> f64 {
    let mut total = 0.0;
    for t in state.tasks.values() {
        let p = prices
            .get(&t.model)
            .unwrap_or_else(|| crate::pricing::fallback_price(&t.model));
        // input_tokens is the FULL prompt token count; cached + cache-creation
        // are subsets of it. The non-cached residue is billed at full input rate.
        let cached = t.cost.cached_input_tokens;
        let creation = t.cost.cache_creation_input_tokens;
        let uncached = t
            .cost
            .input_tokens
            .saturating_sub(cached)
            .saturating_sub(creation);
        let dollars_in = uncached as f64 * p.input
            + cached as f64 * p.cache_read
            + creation as f64 * p.cache_write;
        let dollars_out = t.cost.output_tokens as f64 * p.output;
        total += (dollars_in + dollars_out) / 1_000_000.0;
    }
    total
}

/// Stage readiness check: a (node, stage) is ready iff its preconditions
/// hold AND its current state is NotStarted. See doc comments below for
/// each stage's precondition.
fn stage_is_ready(graph: &NodeGraph, id: NodeId, stage: Stage) -> bool {
    let n = match graph.get(id) {
        Some(n) => n,
        None => return false,
    };
    let cur = n.stages.get(stage);
    let architect_done = match graph.root {
        Some(rid) => graph
            .get(rid)
            .map(|r| r.stages.architect.is_done())
            .unwrap_or(false),
        None => false,
    };
    match stage {
        Stage::Architect => {
            if cur != StageState::NotStarted {
                return false;
            }
            n.parent.is_none()
        }
        Stage::Spec => {
            if cur != StageState::NotStarted {
                return false;
            }
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
            for child in graph.children_of(id) {
                if !child.stages.impl_.is_done() {
                    return false;
                }
            }
            true
        }
        Stage::Debug => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;

    #[test]
    fn architect_runs_first_then_root_spec() {
        let mut g = NodeGraph::new();
        let _root = g.insert_root(Node::new("app", "")).unwrap();
        assert!(stage_is_ready(&g, g.root.unwrap(), Stage::Architect));
        assert!(!stage_is_ready(&g, g.root.unwrap(), Stage::Spec));
        g.get_mut(g.root.unwrap()).unwrap().stages.architect = StageState::Done;
        assert!(stage_is_ready(&g, g.root.unwrap(), Stage::Spec));
        assert!(!stage_is_ready(&g, g.root.unwrap(), Stage::Iface));
    }

    #[test]
    fn architect_only_on_root() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let c = g.add_child(root, Node::new("c", "")).unwrap();
        assert!(!stage_is_ready(&g, c, Stage::Architect));
    }

    #[test]
    fn child_spec_waits_on_parent_spec() {
        let mut g = NodeGraph::new();
        let root = g.insert_root(Node::new("app", "")).unwrap();
        let c = g.add_child(root, Node::new("c", "")).unwrap();
        g.get_mut(root).unwrap().stages.architect = StageState::Done;
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
        for id in [root, a, b] {
            g.get_mut(id).unwrap().stages.spec = StageState::Done;
        }
        assert!(!stage_is_ready(&g, a, Stage::Iface));
        assert!(stage_is_ready(&g, b, Stage::Iface));
        g.get_mut(b).unwrap().stages.iface = StageState::Done;
        assert!(stage_is_ready(&g, a, Stage::Iface));
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
    fn transient_classifier_recognizes_known_patterns() {
        assert!(is_transient(
            "CompletionError: ResponseError: Response contained no message or tool call (empty)"
        ));
        assert!(is_transient("HTTP 502 Bad Gateway"));
        assert!(is_transient("connection reset"));
        assert!(!is_transient("invalid api key"));
    }

    #[test]
    fn rate_limit_classifier_catches_common_phrasings() {
        // OpenRouter and provider variants that should pause the engine.
        assert!(is_rate_limited(
            "HTTP 429 Too Many Requests: rate limit exceeded"
        ));
        assert!(is_rate_limited("Quota exceeded for this model"));
        assert!(is_rate_limited("Insufficient credits"));
        assert!(is_rate_limited("daily limit reached"));
        assert!(is_rate_limited("rate-limit hit, retry later"));
        assert!(is_rate_limited(
            "You exceeded your current quota, please check your plan"
        ));
        // Genuine errors that should NOT pause the engine.
        assert!(!is_rate_limited("HTTP 500 internal server error"));
        assert!(!is_rate_limited("compile error: type mismatch"));
        assert!(!is_rate_limited("invalid api key"));
        // Rate-limit and transient sets are independent — 429 shouldn't
        // be classified as merely transient.
        assert!(!is_transient("HTTP 429 Too Many Requests"));
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
