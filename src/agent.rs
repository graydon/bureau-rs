//! Agent invocation. Wraps `rig` with per-phase tool sets and structured
//! context injection.

use crate::config::Config;
use crate::phase::Phase;
use crate::state::{StateHandle, UiEvent};
use crate::task::{
    Role, SubtaskDecl, Task, TokenUsage, TranscriptEntry, TranscriptKind,
};
use crate::tools::{self, CompilerError, JudgeVerdict, TaskCtx};
use anyhow::{Result, anyhow};
use chrono::Utc;
use parking_lot::Mutex;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openrouter;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Result of running an agent for one task.
pub struct AgentRunOutcome {
    pub final_output: String,
    pub usage: TokenUsage,
    pub emitted_subtasks: Vec<SubtaskDecl>,
    pub written_files: HashSet<PathBuf>,
    /// Final verdict from the judge, if the critique cycle ran. None when
    /// `critique_retries == 0` for this phase.
    pub final_verdict: Option<JudgeVerdict>,
}

/// Builder/dispatcher for agent calls.
pub struct AgentRunner {
    pub config: Arc<Config>,
    pub state: StateHandle,
    pub client: openrouter::Client,
}

impl AgentRunner {
    pub fn new(config: Arc<Config>, state: StateHandle) -> Result<Self> {
        let key_var = config
            .provider
            .api_key_env
            .clone()
            .unwrap_or_else(|| "OPENROUTER_API_KEY".to_string());
        let key = std::env::var(&key_var)
            .map_err(|_| anyhow!("missing env var {} for OpenRouter API key", key_var))?;
        let mut builder = openrouter::Client::builder().api_key(&key);
        if let Some(base) = &config.provider.base_url {
            builder = builder.base_url(base);
        }
        let client = builder
            .build()
            .map_err(|e| anyhow!("openrouter client build: {}", e))?;
        Ok(Self {
            config,
            state,
            client,
        })
    }

    /// Run the agent for a single task with optional critique→revise→judge cycle.
    pub async fn run_task(
        &self,
        task: &Task,
        ctx_workdir: PathBuf,
        compiler_errors: Vec<CompilerError>,
    ) -> Result<AgentRunOutcome> {
        let phase_cfg = self.config.phase_config(task.phase);

        // Shared scratch — accumulates writes, subtasks, and the verdict
        // across every role's invocation for this task.
        let written = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
        let emitted_subtasks = Arc::new(Mutex::new(Vec::<SubtaskDecl>::new()));
        let verdict = Arc::new(Mutex::new(None::<JudgeVerdict>));

        // -------- Actor --------
        let actor_resp = self
            .run_role(
                task,
                Role::Actor,
                &ctx_workdir,
                phase_cfg,
                &compiler_errors,
                written.clone(),
                emitted_subtasks.clone(),
                verdict.clone(),
                None,
            )
            .await?;
        let mut last_text = actor_resp;

        // -------- Critique cycle (if enabled) --------
        for round in 1..=phase_cfg.critique_retries {
            // Critic
            let critique = self
                .run_role(
                    task,
                    Role::Critic,
                    &ctx_workdir,
                    phase_cfg,
                    &compiler_errors,
                    written.clone(),
                    emitted_subtasks.clone(),
                    verdict.clone(),
                    Some(CritiqueExtras {
                        round,
                        prior_actor_text: Some(last_text.clone()),
                        prior_critique: None,
                        prior_revision: None,
                    }),
                )
                .await?;

            // Reviser
            let revision = self
                .run_role(
                    task,
                    Role::Reviser,
                    &ctx_workdir,
                    phase_cfg,
                    &compiler_errors,
                    written.clone(),
                    emitted_subtasks.clone(),
                    verdict.clone(),
                    Some(CritiqueExtras {
                        round,
                        prior_actor_text: Some(last_text.clone()),
                        prior_critique: Some(critique.clone()),
                        prior_revision: None,
                    }),
                )
                .await?;
            last_text = revision.clone();

            // Judge — must call submit_verdict
            *verdict.lock() = None;
            let _judge_text = self
                .run_role(
                    task,
                    Role::Judge,
                    &ctx_workdir,
                    phase_cfg,
                    &compiler_errors,
                    written.clone(),
                    emitted_subtasks.clone(),
                    verdict.clone(),
                    Some(CritiqueExtras {
                        round,
                        prior_actor_text: None,
                        prior_critique: Some(critique),
                        prior_revision: Some(revision),
                    }),
                )
                .await?;
            let v = verdict.lock().clone();
            if let Some(v) = v {
                let note = format!(
                    "round {round}/{} verdict: {} ({})",
                    phase_cfg.critique_retries,
                    if v.satisfactory { "satisfactory" } else { "needs work" },
                    truncate(&v.reason, 200)
                );
                self.state.write(|s| s.note(note));
                if v.satisfactory {
                    break;
                }
            } else {
                self.state.write(|s| {
                    s.note(format!(
                        "round {round}: judge produced no verdict; assuming unsatisfactory and continuing"
                    ))
                });
            }
        }

        let written_set = written.lock().clone();
        let subtasks = emitted_subtasks.lock().clone();
        let final_verdict = verdict.lock().clone();
        // Total per-task cost is already accumulated via record_cost in run_role.
        let cost = self
            .state
            .read(|s| s.graph.get(task.id).map(|t| t.cost.clone()).unwrap_or_default());

        Ok(AgentRunOutcome {
            final_output: last_text,
            usage: cost,
            emitted_subtasks: subtasks,
            written_files: written_set,
            final_verdict,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_role(
        &self,
        task: &Task,
        role: Role,
        ctx_workdir: &PathBuf,
        phase_cfg: &crate::config::PhaseConfig,
        compiler_errors: &[CompilerError],
        written: Arc<Mutex<HashSet<PathBuf>>>,
        emitted_subtasks: Arc<Mutex<Vec<SubtaskDecl>>>,
        verdict: Arc<Mutex<Option<JudgeVerdict>>>,
        extras: Option<CritiqueExtras>,
    ) -> Result<String> {
        let preamble = build_role_preamble(&self.config, task, role);
        let context_doc =
            build_role_context(&self.config, task, ctx_workdir, role, extras.as_ref()).await?;
        let user_prompt = build_role_user_prompt(task, role);

        let now = Utc::now();
        for (kind, content) in [
            (TranscriptKind::System, preamble.clone()),
            (TranscriptKind::UserPrompt, user_prompt.clone()),
        ] {
            let entry = TranscriptEntry {
                timestamp: now,
                kind,
                content,
                role,
            };
            self.state.write(|s| {
                if let Some(t) = s.graph.get_mut(task.id) {
                    t.transcript.push(entry.clone());
                }
            });
            self.state.emit(UiEvent::TranscriptAppended {
                task_id: task.id,
                entry,
            });
        }

        // Per-role read/write set shaping. Critic and judge are read-only; the
        // reviser shares the actor's write_set.
        let (read_set, write_set, allow_subtasks): (HashSet<PathBuf>, HashSet<PathBuf>, bool) =
            match role {
                Role::Actor => (
                    task.read_files.iter().cloned().collect(),
                    task.write_files.iter().cloned().collect(),
                    true,
                ),
                Role::Critic | Role::Judge => {
                    (HashSet::new(), HashSet::new(), false) // empty read_set = unrestricted reads
                }
                Role::Reviser => (
                    task.read_files.iter().cloned().collect(),
                    task.write_files.iter().cloned().collect(),
                    false,
                ),
            };

        let task_ctx = Arc::new(TaskCtx {
            task_id: task.id,
            phase: task.phase,
            role,
            workdir: ctx_workdir.clone(),
            read_set,
            write_set,
            written,
            emitted_subtasks,
            verdict,
            compiler_errors: compiler_errors.to_vec(),
            max_file_lines: self.config.limits.max_file_lines,
            max_spec_section_lines: self.config.limits.max_spec_section_lines,
            depth: task.depth,
            // Disable subtask emission for non-actor roles by setting the
            // depth equal to the cap, so emit_subtasks fails fast. (Actor
            // gets the real cap.)
            max_subtask_depth: if allow_subtasks {
                self.config.limits.max_subtask_depth
            } else {
                task.depth
            },
            // Fresh loop-detection window per role invocation; a tight loop
            // by the actor shouldn't poison the reviser's state.
            recent_calls: Arc::new(parking_lot::Mutex::new(
                std::collections::VecDeque::new(),
            )),
            state: self.state.clone(),
        });

        let model = phase_cfg.model_for(role);
        // Retry the agent invocation on transient provider errors (empty
        // responses, connection drops). gpt-5-mini and similar models
        // occasionally return zero content + zero tool calls, which rig
        // surfaces as a `ResponseError` and which the orchestrator
        // previously treated as a fatal task failure.
        const MAX_TRANSIENT_RETRIES: u32 = 3;
        let mut attempt: u32 = 0;
        let resp = loop {
            let r = run_agent_for_role(
                &self.client,
                phase_cfg,
                model,
                &preamble,
                &context_doc,
                &user_prompt,
                task.phase,
                role,
                task_ctx.clone(),
            )
            .await;
            match r {
                Ok(resp) => break resp,
                Err(e) => {
                    let msg = format!("{:#}", e);
                    if attempt < MAX_TRANSIENT_RETRIES && is_transient_agent_error(&msg) {
                        attempt += 1;
                        let backoff_ms = 400u64 * (1 << (attempt - 1).min(3));
                        tracing::warn!(
                            task = %task.id,
                            role = %role,
                            attempt,
                            "transient agent error, retrying in {backoff_ms}ms: {msg}"
                        );
                        self.state.write(|s| {
                            s.note(format!(
                                "task {} {role}: transient error retry {}/{}: {}",
                                task.id,
                                attempt,
                                MAX_TRANSIENT_RETRIES,
                                truncate(&msg, 140)
                            ))
                        });
                        tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        };

        let usage = TokenUsage {
            input_tokens: resp.usage.input_tokens,
            output_tokens: resp.usage.output_tokens,
            cached_input_tokens: resp.usage.cached_input_tokens,
            cache_creation_input_tokens: resp.usage.cache_creation_input_tokens,
        };

        let final_entry = TranscriptEntry {
            timestamp: Utc::now(),
            kind: TranscriptKind::AssistantText,
            content: resp.output.clone(),
            role,
        };
        self.state.write(|s| {
            if let Some(t) = s.graph.get_mut(task.id) {
                t.transcript.push(final_entry.clone());
                t.cost.add(&usage);
                if t.model.is_none() {
                    t.model = Some(model.to_string());
                }
                s.total_cost.add(&usage);
                s.estimated_cost_usd = compute_total_cost(&self.config, s);
            }
        });
        self.state.emit(UiEvent::TranscriptAppended {
            task_id: task.id,
            entry: final_entry,
        });
        let total = self.state.read(|s| s.total_cost.clone());
        let est_usd = self.state.read(|s| s.estimated_cost_usd);
        self.state.emit(UiEvent::TaskCost {
            task_id: task.id,
            cost: usage,
            total,
            estimated_usd: est_usd,
        });

        Ok(resp.output)
    }
}

/// Extra context fed to non-actor roles: the prior actor output, critique,
/// and revision text from earlier turns of the same critique round.
#[derive(Clone)]
struct CritiqueExtras {
    round: u32,
    prior_actor_text: Option<String>,
    prior_critique: Option<String>,
    prior_revision: Option<String>,
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_agent_for_role(
    client: &openrouter::Client,
    phase_cfg: &crate::config::PhaseConfig,
    model: &str,
    preamble: &str,
    context_doc: &str,
    user_prompt: &str,
    phase: Phase,
    role: Role,
    ctx: Arc<TaskCtx>,
) -> Result<rig::agent::PromptResponse> {
    use tools::*;
    // Inline the context into the preamble. We avoid `.context(...)` because
    // rig labels each context document with an id like `static_doc_0`, which
    // models then mistake for a readable filename and try to call read_file
    // on.
    let combined_preamble = format!("{preamble}\n\n{context_doc}");
    let base = client
        .agent(model)
        .preamble(&combined_preamble)
        .max_tokens(phase_cfg.max_tokens)
        .temperature(phase_cfg.temperature)
        .default_max_turns(phase_cfg.max_turns.max(2));

    // Tool registration: actor and reviser get the phase's normal tool set;
    // critic gets read-only tools; judge gets read-only + verdict. Spec writes
    // now go through write_file (filtered to spec/*.md by phase guards), so
    // there's no longer a separate spec tool surface.
    let resp = match role {
        Role::Actor | Role::Reviser => match phase {
            Phase::Spec => {
                // No cargo tools — there's no crate yet in this phase.
                let mut a = base
                    .tool(WriteFileTool { ctx: ctx.clone() })
                    .tool(ReadFileTool { ctx: ctx.clone() })
                    .tool(ListFilesTool { ctx: ctx.clone() });
                if matches!(role, Role::Actor) {
                    a = a.tool(EmitSubtasksTool { ctx });
                }
                a.build().prompt(user_prompt).extended_details().await?
            }
            Phase::Interface => {
                let mut a = base
                    .tool(WriteFileTool { ctx: ctx.clone() })
                    .tool(ReadFileTool { ctx: ctx.clone() })
                    .tool(ListFilesTool { ctx: ctx.clone() })
                    .tool(CargoCheckTool { ctx: ctx.clone() });
                if matches!(role, Role::Actor) {
                    a = a.tool(EmitSubtasksTool { ctx });
                }
                a.build().prompt(user_prompt).extended_details().await?
            }
            Phase::Test => {
                // Test phase: bodies are still todo!() so cargo_test would
                // panic at runtime. cargo_test_no_run verifies the tests
                // compile, which is what this phase actually wants.
                let mut a = base
                    .tool(WriteFileTool { ctx: ctx.clone() })
                    .tool(ReadFileTool { ctx: ctx.clone() })
                    .tool(ListFilesTool { ctx: ctx.clone() })
                    .tool(CargoCheckTool { ctx: ctx.clone() })
                    .tool(CargoTestNoRunTool { ctx: ctx.clone() });
                if matches!(role, Role::Actor) {
                    a = a.tool(EmitSubtasksTool { ctx });
                }
                a.build().prompt(user_prompt).extended_details().await?
            }
            Phase::Impl => {
                let mut a = base
                    .tool(WriteFileTool { ctx: ctx.clone() })
                    .tool(ReadFileTool { ctx: ctx.clone() })
                    .tool(ListFilesTool { ctx: ctx.clone() })
                    .tool(CargoCheckTool { ctx: ctx.clone() })
                    .tool(CargoTestTool { ctx: ctx.clone() })
                    .tool(CargoClippyTool { ctx: ctx.clone() });
                if matches!(role, Role::Actor) {
                    a = a.tool(EmitSubtasksTool { ctx });
                }
                a.build().prompt(user_prompt).extended_details().await?
            }
            Phase::Debug => base
                .tool(WriteFileTool { ctx: ctx.clone() })
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(ReplaceFnBodyTool { ctx: ctx.clone() })
                .tool(ListCompilerErrorsTool { ctx: ctx.clone() })
                .tool(ReadCompilerErrorTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Opt => base
                .tool(WriteFileTool { ctx: ctx.clone() })
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(ReplaceFnBodyTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(CargoClippyTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
        },

        // ---- Critic: read-only across phases. Critics also get cargo_check
        // (and cargo_test in Impl/Debug/Opt) so they can verify the actor's
        // claims with real diagnostics rather than just guessing.
        Role::Critic => match phase {
            Phase::Spec => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Interface => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Test => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestNoRunTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Impl | Phase::Debug | Phase::Opt => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
        },

        // ---- Judge: read-only + submit_verdict + cargo for verification.
        Role::Judge => match phase {
            Phase::Spec => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Interface => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Test => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestNoRunTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
            Phase::Impl | Phase::Debug | Phase::Opt => base
                .tool(ReadFileTool { ctx: ctx.clone() })
                .tool(ListFilesTool { ctx: ctx.clone() })
                .tool(CargoCheckTool { ctx: ctx.clone() })
                .tool(CargoTestTool { ctx: ctx.clone() })
                .tool(SubmitVerdictTool { ctx })
                .build()
                .prompt(user_prompt)
                .extended_details()
                .await?,
        },
    };
    Ok(resp)
}

fn compute_total_cost(config: &Config, state: &crate::state::OrchestratorState) -> f64 {
    let mut total = 0.0;
    for t in state.graph.iter() {
        let pcfg = config.phase_config(t.phase);
        let p_in = pcfg.price_in_per_mtok.unwrap_or_else(|| default_price_in(&pcfg.model));
        let p_out = pcfg.price_out_per_mtok.unwrap_or_else(|| default_price_out(&pcfg.model));
        total += t.cost.cost_usd(p_in, p_out);
    }
    total
}

// Rough $/Mtok defaults for cost estimation. OpenRouter pricing varies; users
// who care about precise USD figures should set price_in/out_per_mtok in
// phases.toml. These heuristics cover common Claude / GPT / Llama / Mistral /
// Gemini / Qwen / DeepSeek model name fragments.
fn default_price_in(model: &str) -> f64 {
    let m = model.to_ascii_lowercase();
    if m.contains("opus") {
        15.0
    } else if m.contains("sonnet") {
        3.0
    } else if m.contains("haiku") {
        1.0
    } else if m.contains("gpt-4o-mini") || m.contains("4o-mini") {
        0.15
    } else if m.contains("gpt-4o") || m.contains("4o") {
        2.5
    } else if m.contains("gpt-4") {
        10.0
    } else if m.contains("gpt-3.5") {
        0.5
    } else if m.contains("gemini-2.5-pro") || m.contains("gemini-1.5-pro") {
        1.25
    } else if m.contains("gemini") {
        0.3
    } else if m.contains("llama-3.1-8b") || m.contains("llama-3-8b") {
        0.05
    } else if m.contains("llama") {
        0.2
    } else if m.contains("deepseek") {
        0.3
    } else if m.contains("qwen3-coder") || m.contains("qwen-2.5-coder") {
        0.2
    } else if m.contains("qwen") {
        0.3
    } else if m.contains("nemotron") {
        0.4
    } else if m.contains("mistral") {
        0.5
    } else {
        1.0
    }
}

fn default_price_out(model: &str) -> f64 {
    let m = model.to_ascii_lowercase();
    if m.contains("opus") {
        75.0
    } else if m.contains("sonnet") {
        15.0
    } else if m.contains("haiku") {
        5.0
    } else if m.contains("gpt-4o-mini") || m.contains("4o-mini") {
        0.6
    } else if m.contains("gpt-4o") || m.contains("4o") {
        10.0
    } else if m.contains("gpt-4") {
        30.0
    } else if m.contains("gpt-3.5") {
        1.5
    } else if m.contains("gemini-2.5-pro") || m.contains("gemini-1.5-pro") {
        5.0
    } else if m.contains("gemini") {
        1.2
    } else if m.contains("llama-3.1-8b") || m.contains("llama-3-8b") {
        0.05
    } else if m.contains("llama") {
        0.6
    } else if m.contains("deepseek") {
        1.2
    } else if m.contains("qwen3-coder") || m.contains("qwen-2.5-coder") {
        0.8
    } else if m.contains("qwen") {
        1.2
    } else if m.contains("nemotron") {
        1.6
    } else if m.contains("mistral") {
        1.5
    } else {
        3.0
    }
}

fn build_role_preamble(config: &Config, task: &Task, role: Role) -> String {
    let phase_body = if let Some(custom) = config.prompt_for(task.phase) {
        custom.to_string()
    } else {
        default_preamble(task.phase)
    };
    let access_block = build_access_block(task);
    let scope_block = phase_scope_block(task.phase);
    let layout_block = layout_block(config);
    match role {
        Role::Actor => {
            let depth_cap = config.limits.max_subtask_depth;
            let guidance = subtask_guidance(task.phase, task.depth, depth_cap);
            format!("{phase_body}\n\n{layout_block}\n\n{access_block}\n\n{scope_block}\n\n{guidance}\n\n# ROLE: ACTOR\n\nYou are the ACTOR. Do the task described above using the tools available. Stay within your declared write_files.")
        }
        Role::Critic => format!(
            "# ROLE: CRITIC\n\n\
             A prior agent (the actor) has just done initial work for this task. You are \
             the ONLY role in this cycle that gets to make subjective judgment calls — the \
             reviser will mechanically address what you raise, and the judge will only \
             check that the reviser did so. So make your concerns count.\n\n\
             # WHAT TO LOOK FOR\n\n\
             Inspect the actor's work via read_file / list_files (and any cargo_* tool you \
             have access to). Focus on issues that will MATTER for this phase's contract \
             and for downstream phases consuming this output:\n\
             - **Correctness**: actual bugs, mismatches with the spec, broken invariants.\n\
             - **Completeness**: missing pieces required by THIS phase (consult the scope \
               block — out-of-scope omissions don't count).\n\
             - **Coherence**: types/signatures that don't fit together, names that \
               contradict their content, modules that won't be reachable.\n\
             - **Phase-specific concerns**: e.g. in Interface, signatures that lock in a \
               clearly wrong shape for downstream phases. In Test, tests that miss obvious \
               cases the spec calls out.\n\n\
             # OUTPUT\n\n\
             A bullet list of 0–10 specific, actionable points. Each bullet should name \
             the file/symbol and explain the concrete issue. Examples:\n\
             - `src/parser.rs::Token` — variant `Number(i32)` should be `Number(i64)` \
                per spec/types.md (range exceeds i32).\n\
             - `tests/integration.rs` — no test exercises the empty-input case from \
                spec/invariants.md §3.\n\n\
             If everything looks good for this phase, output exactly `No issues found.` \
             and nothing else.\n\n\
             # WHAT NOT TO DO\n\n\
             - Don't flag things explicitly marked out-of-scope by the scope block below \
               (e.g. \"main() doesn't do anything\" in the Interface phase — that's by \
               design).\n\
             - Don't pad your list to look thorough. Empty critique is fine if nothing's \
               wrong.\n\
             - Don't write or modify files. You're read-only.\n\n\
             {layout_block}\n\n\
             {access_block}\n\n\
             {scope_block}\n\n\
             For reference, the phase's normal job is described below.\n\n--- BEGIN PHASE BRIEF ---\n{phase_body}\n--- END PHASE BRIEF ---"
        ),
        Role::Reviser => format!(
            "# ROLE: REVISER\n\nA prior agent (the actor) did initial work; a critic identified \
             issues. Your job is to FIX the issues raised by the critic, using the available \
             write tools. Make minimal targeted edits — do not introduce new architectural \
             changes; address each critique point. Only fix issues that are in scope for THIS \
             phase (see scope block below); ignore critiques that are about future phases. \
             Do not emit subtasks. End with a brief summary of what you changed.\n\n\
             {layout_block}\n\n\
             {access_block}\n\n\
             {scope_block}\n\n\
             For reference, the phase's normal job is described below.\n\n--- BEGIN PHASE BRIEF ---\n{phase_body}\n--- END PHASE BRIEF ---"
        ),
        Role::Judge => format!(
            "# ROLE: JUDGE\n\n\
             You are the COHERENCE CHECK at the end of an actor → critic → reviser cycle. \
             Your single job is to confirm that the reviser actually addressed what the \
             critic complained about. You are NOT a fresh code reviewer. You are NOT the \
             phase gate (cargo check / cargo test runs separately and will catch any \
             mechanical defect — that is not your concern).\n\n\
             # WHAT TO DO\n\n\
             1. Read the critic's output above (the `# Critique round N` section in your \
                context). It's a bullet list of specific concerns.\n\
             2. Read the reviser's summary in the same section. It describes what the \
                reviser changed.\n\
             3. For each critic point, decide one of:\n\
                - **Addressed** — the reviser's edits clearly fix this point.\n\
                - **Deferred with reason** — the reviser explained why this point is \
                  out of scope or doesn't apply. Acceptable if the reasoning is sound.\n\
                - **Ignored / unaddressed** — the reviser silently dropped it or claims \
                  to have fixed it without doing so. THIS is the failure case.\n\
             4. Optionally use read_file / list_files (or cargo_check etc. if available) \
                to spot-check that the reviser's claimed changes actually exist on disk \
                — useful when you suspect bullshitting.\n\n\
             # VERDICT\n\n\
             - satisfactory=true if all critic points are addressed or reasonably deferred. \
                The work doesn't need to be perfect; it just needs to be a faithful \
                response to the critique.\n\
             - satisfactory=false if one or more critic points were ignored or claimed-fixed \
                but not actually fixed. Quote the specific unaddressed point(s) as the \
                reason.\n\
             - Special case: if the critic explicitly said 'No issues found' (or similar), \
                satisfactory=true. There was nothing to address.\n\n\
             # WHAT YOU MUST NOT DO\n\n\
             - Do NOT raise new concerns the critic didn't raise. If you think the critic \
                missed something, that's information for the NEXT phase, not your verdict. \
                Inventing fresh concerns turns the cycle into infinite gold-plating.\n\
             - Do NOT fail the task because the work \"could be better designed\" or \
                \"isn't ideal\". The critic already had that opportunity.\n\
             - Do NOT use the scope block below as a basis for failing — it's reference \
                material for understanding what the phase is for, not a checklist.\n\
             - When in doubt, satisfactory=true. The gate is the safety net for mechanical \
                defects; your job is just to keep the cycle honest.\n\n\
             You MUST call submit_verdict exactly once before finishing.\n\n\
             {layout_block}\n\n\
             {access_block}\n\n\
             {scope_block}\n\n\
             For reference, the phase's normal job is described below.\n\n--- BEGIN PHASE BRIEF ---\n{phase_body}\n--- END PHASE BRIEF ---"
        ),
    }
}

fn layout_block(config: &Config) -> String {
    match config.layout.kind {
        crate::config::WorkspaceLayout::Single => {
            "# PROJECT LAYOUT: SINGLE CRATE\n\n\
             Produce a single Rust crate at the workdir root. Expected layout:\n\
             - `Cargo.toml` (workdir root)\n\
             - `src/` for source code; `src/lib.rs` and/or `src/main.rs`\n\
             - `tests/` for integration tests\n\
             - `src/<mod>/tests.rs`, `src/<mod>/test/...`, or `*_tests.rs` for \
               internal/unit tests (Test phase only)\n\
             - `spec/*.md` for the specification (Spec phase only)\n\n\
             Do NOT create a `crates/` directory or workspace structure."
                .to_string()
        }
        crate::config::WorkspaceLayout::Workspace => {
            "# PROJECT LAYOUT: CARGO WORKSPACE\n\n\
             Produce a Cargo workspace with member crates. Expected layout:\n\
             - `Cargo.toml` at the workdir root with `[workspace] members = [...]`\n\
             - `crates/<name>/Cargo.toml` for each member crate\n\
             - `crates/<name>/src/` for that crate's sources\n\
             - `crates/<name>/tests/` for that crate's integration tests\n\
             - `crates/<name>/src/<mod>/tests.rs` (or similar) for internal tests\n\
             - `spec/*.md` at the workdir root for the workspace-wide specification\n\n\
             Pick reasonable crate names from the spec. The workspace root \
             `Cargo.toml` should declare members; each member's Cargo.toml \
             declares its own dependencies. The Spec phase covers the whole \
             workspace; later phases work per-crate. When multiple crates can \
             be developed in parallel, emit subtasks scoped to specific \
             member crates with non-overlapping `crates/<name>/` write_files \
             prefixes."
                .to_string()
        }
    }
}

/// What does this phase produce, and just as importantly, what is it NOT
/// responsible for? The pipeline is a waterfall — most phases produce
/// deliberately incomplete artifacts (interface stubs, failing tests, etc.)
/// that later phases fill in. Without this block, critics and judges flag
/// the deliberate incompleteness as defects.
fn phase_scope_block(phase: Phase) -> String {
    let (in_scope, out_of_scope) = match phase {
        Phase::Spec => (
            "- Markdown specification documents under `spec/*.md`.\n\
             - Problem statement, types, modules, public APIs, invariants, errors, \
             dependencies — at the design level only.",
            "- No Rust code. No `src/` files. No `Cargo.toml`. No tests.\n\
             - It is not a defect that there is no code; this phase precedes coding.",
        ),
        Phase::Interface => (
            "- Rust type signatures, module declarations, trait definitions, struct/enum \
             definitions in `src/`.\n\
             - `Cargo.toml` with crate dependencies declared (this is the only phase that \
             may add dependencies).\n\
             - Function bodies are intentionally `todo!()` — the orchestrator auto-stubs \
             any non-stub bodies and emits a warning.",
            "- It is NOT a defect that function bodies are `todo!()`. They are SUPPOSED to \
             be `todo!()`; that is the whole point of this phase.\n\
             - It is NOT a defect that `main()` does not run, that nothing is wired up, that \
             the program produces no output, or that there is no behavior. The interface \
             phase produces signatures, not behavior.\n\
             - It is NOT a defect that there are no tests yet. Tests are written in the \
             next phase (Test).\n\
             - It is NOT a defect that there is no implementation logic. Implementation is \
             a later phase.\n\
             - DO NOT fail or critique because the code 'doesn't do anything' — that is \
             literally the contract of this phase.",
        ),
        Phase::Test => (
            "- Rust test files under `tests/` that exercise the public API from the \
             Interface phase.\n\
             - Tests SHOULD compile (`cargo test --no-run` must pass).",
            "- Tests are EXPECTED to fail at runtime — bodies are still `todo!()`. \
             It is not a defect that the tests fail; they will pass after the \
             Implementation phase.\n\
             - Do NOT modify any files in `src/`.\n\
             - Do NOT add new crate dependencies.",
        ),
        Phase::Impl => (
            "- Function-body fills in `src/` so `cargo test` passes.\n\
             - The shapes of public types and signatures from the Interface phase are \
             preserved.",
            "- Do NOT change public signatures from the Interface phase (orchestrator \
             will reject this).\n\
             - Do NOT modify test files.\n\
             - Do NOT add new crate dependencies.\n\
             - Do not redesign the architecture; trust the spec/interface.",
        ),
        Phase::Debug => (
            "- Targeted fixes to make `cargo test` pass when the Implementation phase \
             didn't quite get there.\n\
             - Minimal edits, ideally just function bodies.",
            "- Do not redesign anything. Fix narrowly.\n\
             - Do not change public signatures.",
        ),
        Phase::Opt => (
            "- Performance improvements that preserve behavior and signatures.",
            "- Do not change public signatures.\n\
             - Do not break tests; `cargo test` must still pass.\n\
             - Do not pursue micro-optimizations that bloat code without measured wins.",
        ),
    };
    format!(
        "# PHASE SCOPE: {phase}\n\n## In scope for this phase:\n{in_scope}\n\n\
         ## Explicitly out of scope (do NOT flag these as defects):\n{out_of_scope}"
    )
}

fn build_access_block(task: &Task) -> String {
    let reads = if task.read_files.is_empty() {
        "(unrestricted reads within the workdir)".to_string()
    } else {
        task.read_files
            .iter()
            .map(|p| format!("- {}", p.display()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let writes = if task.write_files.is_empty() {
        "(NONE — this task declared no writable paths; you cannot write any files. \
         If this seems wrong, the parent task that emitted you forgot to set write_files.)"
            .to_string()
    } else {
        task.write_files
            .iter()
            .map(|p| format!("- {}", p.display()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "# YOUR FILE ACCESS\n\n## You may READ:\n{reads}\n\n## You may WRITE:\n{writes}\n\n\
         These are HARD limits — write tools will return an error if you try a path \
         outside the write list. If you need a different file, do not just attempt the write \
         and hope; that will fail every time. Instead, work within the declared set, or (if \
         you're decomposing) emit subtasks with the right write_files declared on each child."
    )
}

fn subtask_guidance(phase: Phase, depth: u32, depth_cap: u32) -> String {
    let phase_word = match phase {
        Phase::Spec => "specification (markdown under spec/)",
        Phase::Interface => "Rust interface files (signatures + todo!() bodies, under src/)",
        Phase::Test => "Rust test files (under tests/)",
        Phase::Impl => "Rust implementation bodies (under src/)",
        Phase::Debug => "compiler error fixes",
        Phase::Opt => "performance improvements",
    };
    let example = match phase {
        Phase::Spec => {
            r#"  {
    "description": "Write the spec/types.md section covering core types and their invariants",
    "read_files": [],
    "write_files": ["spec/types.md"]
  }"#
        }
        Phase::Interface => {
            r#"  {
    "description": "Define the Greeter trait and its associated error type in src/greeter.rs",
    "read_files": ["spec/types.md"],
    "write_files": ["src/greeter.rs"]
  }"#
        }
        Phase::Test => {
            r#"  {
    "description": "Integration test for Greeter::greet covering the empty-name case",
    "read_files": ["src/greeter.rs"],
    "write_files": ["tests/greeter_empty_name.rs"]
  }"#
        }
        Phase::Impl => {
            r#"  {
    "description": "Implement Greeter::greet body in src/greeter.rs",
    "read_files": ["tests/greeter_empty_name.rs"],
    "write_files": ["src/greeter.rs"]
  }"#
        }
        Phase::Debug | Phase::Opt => "  (subtasks disabled in this phase)",
    };
    let write_files_rule = format!(
        "- write_files MUST be non-empty. Every file the subtask intends to create or modify \
         must be listed. The orchestrator will REJECT any subtask with empty write_files. \
         Entries can be exact paths (`spec/types.md`) or directory prefixes ending with `/` \
         (`spec/` matches anything under spec/). Suggested targets for {phase}: {}.",
        crate::tools::suggested_writes_for(phase)
    );
    if !phase.allows_subtasks() || depth >= depth_cap {
        format!(
            "# SUBTASK POLICY\n\nYou are at depth {depth} (cap {depth_cap}). emit_subtasks \
            is DISABLED for this task. Do all the {phase_word} work yourself in this turn, \
            then write a brief summary. Do not call emit_subtasks; doing so will be ignored."
        )
    } else {
        format!(
            "# SUBTASK POLICY\n\n\
            You are at depth {depth}/{depth_cap} in the task tree. Prefer doing the work \
            yourself; only emit subtasks if there are 2+ independent pieces with disjoint \
            write_files that benefit from parallelism.\n\n\
            ## REQUIRED FIELDS for every subtask declaration\n\
            - `description`: a concrete actionable instruction for THIS phase ({phase}), not \
              a high-level project goal. The current phase produces {phase_word}.\n\
            - `read_files`: list of paths the subtask will need to read. Empty is fine if it \
              writes from scratch.\n\
            {write_files_rule}\n\
            \n\
            Subtasks inherit the current phase ({phase}); do NOT try to cross into a later \
            phase by changing `description`.\n\
            \n\
            ## Example of a well-formed subtask\n\
            ```json\n\
{example}\n\
            ```\n\
            \n\
            If you can finish this task yourself in one turn by calling the write tools \
            directly, do that and skip emit_subtasks entirely."
        )
    }
}

fn default_preamble(phase: Phase) -> String {
    let common = "You are an expert Rust software engineer working as part of a multi-phase \
        agent pipeline that produces a Rust crate. Use the provided tools to read and write \
        artifacts. Final messages should be a short natural-language summary; do not paste \
        code or markdown into the final message — put it in the appropriate file via tools.\n\n\
        # IMPORTANT TOOL BEHAVIOR\n\n\
        - When write_file or replace_fn_body returns `no_change: true`, the file already \
          contained byte-identical content; the write was a no-op. Do not retry — move on \
          to a different file or end the turn.\n\
        - The harness detects loops: if you call the same tool with the exact same \
          arguments three times in a row you'll get a `Loop` error telling you to stop. \
          When you see that error, finish your turn with a summary; do not call more tools.\n\
        - Successful tool results mean the operation completed. Trust them and proceed.\n\
        - When diagnostics tools (cargo_check, cargo_test, cargo_test_no_run) are available \
          to you, USE THEM before finishing your turn. Compiling first and finding the \
          errors yourself is much cheaper than waiting for the phase gate to fail and \
          having the whole task retried. Iterate: write → cargo_check → fix → cargo_check \
          → done.";
    match phase {
        Phase::Spec => format!(
            "{common}\n\nPHASE: SPEC\n\
            Your job in this phase is ONLY to produce a structured specification as markdown \
            files under `spec/`. Use the write_file tool with paths like `spec/types.md`, \
            `spec/public_apis.md`, etc. Only paths matching `spec/*.md` are accepted in \
            this phase; do not write Rust code or files in src/, tests/, or Cargo.toml.\n\n\
            Recommended sections (one cohesive topic each, under the per-section line limit): \
            `spec/problem.md`, `spec/types.md`, `spec/modules.md`, `spec/public_apis.md`, \
            `spec/invariants.md`, `spec/errors.md`, `spec/dependencies.md`. You may add \
            others if the problem warrants it.\n\n\
            For a small problem, you can write all sections yourself in a single turn — do \
            not over-decompose."
        ),
        Phase::Interface => format!(
            "{common}\n\nPHASE: INTERFACE\n\
            Produce Rust type signatures, module declarations, and trait definitions in src/. \
            Function bodies MUST be `todo!()` — the orchestrator will overwrite any non-stub \
            body and warn. Files must be small (under the per-file line limit). Declare crate \
            dependencies in Cargo.toml — this is your only chance; later phases cannot add \
            new crate dependencies.\n\n\
            DO NOT write test files. Tests are written in the next phase (Test). Writes to \
            `tests/` will be rejected. Don't even sketch test signatures here.\n\n\
            DO NOT try to make the program 'do anything'. There is no behavior in this phase: \
            `main()` (if present) should also be a `todo!()` stub. Implementation comes later."
        ),
        Phase::Test => format!(
            "{common}\n\nPHASE: TEST\n\
            Produce test modules under tests/ that exercise the public API from the Interface \
            phase. Tests will fail until the Implementation phase fills in bodies — that's \
            expected. Do not modify any files in src/."
        ),
        Phase::Impl => format!(
            "{common}\n\nPHASE: IMPLEMENTATION\n\
            Fill in function bodies in src/ to make the tests pass. Do not change public \
            signatures from the Interface phase — the orchestrator will reject signature \
            changes. Do not modify test files. Do not add new crate dependencies."
        ),
        Phase::Debug => format!(
            "{common}\n\nPHASE: DEBUG\n\
            Fix compiler errors and failing tests. Use list_compiler_errors and \
            read_compiler_error to investigate. Make minimal targeted edits via \
            replace_fn_body and write_file."
        ),
        Phase::Opt => format!(
            "{common}\n\nPHASE: OPTIMIZATION\n\
            Make targeted performance improvements without changing public signatures or \
            test outcomes."
        ),
    }
}

async fn build_role_context(
    config: &Config,
    task: &Task,
    workdir: &std::path::Path,
    role: Role,
    extras: Option<&CritiqueExtras>,
) -> Result<String> {
    let mut s = String::new();
    s.push_str("# Task\n\n");
    s.push_str(&task.description);
    s.push_str("\n\n");

    // The judge has a narrow job (verify the cycle was coherent) and gets a
    // narrow context: critique + revision + tree. It does NOT need the full
    // problem statement, declared read-file dumps, locked interface
    // signatures, or spec section inlines — those are for the actor /
    // critic / reviser who are doing or evaluating actual work. Inflating
    // the judge's context just makes a yes/no decision more expensive and
    // tempts the model to second-guess.
    let is_judge = matches!(role, Role::Judge);

    if !is_judge {
        s.push_str("# Problem (top-level)\n\n");
        s.push_str(&config.problem);
        s.push_str("\n\n");
    }

    // Inline a tree snapshot of the workdir for non-actor roles. This stops
    // the critic / reviser / judge from spending a turn on list_files just
    // to check what's there.
    if matches!(role, Role::Critic | Role::Reviser | Role::Judge) {
        let snapshot = workdir_tree_snapshot(workdir);
        s.push_str(&format!(
            "# Workdir snapshot (live filesystem at start of this role)\n\n```\n{snapshot}```\n\n"
        ));
    }

    // Cycle outputs from prior roles. NOTE: we pass only the small assistant
    // SUMMARIES forward, not full transcripts or tool histories — the
    // critic's tool calls are private to the critic's agent loop, etc.
    //   - Critic sees: prior actor summary (one paragraph)
    //   - Reviser sees: prior actor summary + critique (one bullet list)
    //   - Judge sees: critique + reviser summary (verify reviser fixed
    //                 what critic raised; the actor summary is irrelevant
    //                 to the judge's coherence check)
    if let Some(ex) = extras {
        s.push_str(&format!("# Critique round {}\n\n", ex.round));
        if let Some(t) = &ex.prior_actor_text {
            s.push_str("## Prior actor summary\n\n");
            s.push_str(t);
            s.push_str("\n\n");
        }
        if let Some(t) = &ex.prior_critique {
            s.push_str("## Critique\n\n");
            s.push_str(t);
            s.push_str("\n\n");
        }
        if let Some(t) = &ex.prior_revision {
            s.push_str("## Reviser summary\n\n");
            s.push_str(t);
            s.push_str("\n\n");
        }
    }

    // Declared read-files: inline content for actor / critic / reviser. Skip
    // for the judge (it can read_file on demand).
    if !task.read_files.is_empty() && !is_judge {
        s.push_str("# Declared read-files\n\n");
        for p in &task.read_files {
            let abs = workdir.join(p);
            match std::fs::read_to_string(&abs) {
                Ok(content) => {
                    s.push_str(&format!("## {}\n\n```\n{}\n```\n\n", p.display(), content));
                }
                Err(_) => {
                    s.push_str(&format!("## {} (does not yet exist)\n\n", p.display()));
                }
            }
        }
    }

    if !task.write_files.is_empty() && matches!(role, Role::Actor | Role::Reviser) {
        s.push_str("# Declared write-files\n\n");
        for p in &task.write_files {
            s.push_str(&format!("- {}\n", p.display()));
        }
        s.push('\n');
    }

    if !task.spec_sections.is_empty() && !is_judge {
        s.push_str("# Declared spec sections\n\n");
        for sec in &task.spec_sections {
            let f = workdir.join("spec").join(format!("{}.md", sec));
            if let Ok(content) = std::fs::read_to_string(&f) {
                s.push_str(&format!("## spec/{}.md\n\n{}\n\n", sec, content));
            }
        }
    }

    // Locked-interface signatures are useful to actor / critic / reviser
    // doing the work, but irrelevant to the judge's coherence check.
    if !is_judge
        && matches!(
            task.phase,
            Phase::Test | Phase::Impl | Phase::Debug | Phase::Opt
        )
    {
        // Walk every Rust *source* file under workdir (single-crate
        // `src/...` and workspace `crates/<name>/src/...` both fall out of
        // the path classifier), and inline the public signatures.
        let mut header_written = false;
        for entry in walkdir::WalkDir::new(workdir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let p = entry.path();
            let Ok(rel) = p.strip_prefix(workdir) else {
                continue;
            };
            // Skip orchestrator bookkeeping (relative-path filter so
            // workdirs nested under `.bureau/` still work).
            if rel.components().any(|c| {
                let s = c.as_os_str().to_string_lossy();
                s == ".git" || s == "target" || s == ".bureau"
            }) {
                continue;
            }
            // Only Rust *source* files (not test files).
            if crate::paths::classify(rel) != crate::paths::PathKind::RustSource {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(p) else {
                continue;
            };
            let Ok(sigs) = crate::artifact::PublicSignatures::from_source(&content) else {
                continue;
            };
            if sigs.items.is_empty() {
                continue;
            }
            if !header_written {
                s.push_str("# Locked interface (public signatures)\n\n");
                header_written = true;
            }
            s.push_str(&format!("## {}\n", rel.display()));
            for it in sigs.items {
                s.push_str(&format!("- {}\n", it));
            }
            s.push('\n');
        }
    }

    Ok(s)
}

fn build_role_user_prompt(task: &Task, role: Role) -> String {
    match role {
        Role::Actor => format!(
            "Execute the task described above for phase {}. Use the appropriate tools to \
             write artifacts and (if appropriate) emit subtasks. End with a brief plain-text \
             summary of what you did.",
            task.phase
        ),
        Role::Critic => format!(
            "Critique the actor's work on this {} task. Read the relevant files and identify \
             specific concrete issues. Output a bullet list. If everything is good, say 'No \
             issues found.'",
            task.phase
        ),
        Role::Reviser => format!(
            "Address each critique point on this {} task. Use the write tools to fix the \
             issues. End with a summary of what you changed.",
            task.phase
        ),
        Role::Judge => format!(
            "Decide whether this {} task is now complete after the revision. Read the \
             relevant files, then call submit_verdict exactly once. Be strict but fair.",
            task.phase
        ),
    }
}

/// Recognize transient errors from the LLM provider that warrant a quiet
/// retry rather than failing the whole task. The error string is what
/// `format!("{:#}", e)` produces from rig's `PromptError` chain.
///
/// Categories:
/// - **Empty response**: rig surfaces `Response contained no message or tool
///   call (empty)` when the model returns content-less output. Common with
///   gpt-5-mini and similar at low temperature.
/// - **Generic ResponseError**: provider misbehavior we can't otherwise
///   classify. Cheap to retry once or twice.
/// - **Network glitches**: connection reset, timeout, DNS hiccup.
/// - **5xx provider errors**: 502, 503, 504 are typically transient.
/// - **Rate limit**: 429. Backoff helps.
pub fn is_transient_agent_error(msg: &str) -> bool {
    let m = msg;
    m.contains("no message or tool call")
        || m.contains("ResponseError")
        || m.contains("connection reset")
        || m.contains("connection closed")
        || m.contains("timed out")
        || m.contains("timeout")
        || m.contains("temporarily unavailable")
        || m.contains("502 Bad Gateway")
        || m.contains("503")
        || m.contains("504")
        || m.contains("429")
        || m.contains("ECONNRESET")
}

/// Snapshot of the workdir as a tree string. Skips `.git/`, `target/`,
/// and `.bureau/`. Used to inline filesystem state into role contexts.
fn workdir_tree_snapshot(workdir: &std::path::Path) -> String {
    let mut paths = Vec::new();
    if !workdir.exists() {
        return "(empty)\n".to_string();
    }
    for entry in walkdir::WalkDir::new(workdir)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let Ok(rel) = p.strip_prefix(workdir) else {
            continue;
        };
        // Filter on the relative path so a workdir nested under `.bureau/`
        // doesn't accidentally hide everything.
        if rel.components().any(|c| {
            let s = c.as_os_str().to_string_lossy();
            s == ".git" || s == "target" || s == ".bureau"
        }) {
            continue;
        }
        paths.push(rel.to_string_lossy().to_string());
    }
    paths.sort();
    crate::tools::render_tree(&paths)
}
