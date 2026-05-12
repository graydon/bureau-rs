//! LLM driver abstraction and the production OpenRouter implementation.
//!
//! The engine talks to LLMs through this trait so tests can swap in a
//! scripted mock. Tool registration (which rig tools to attach for a
//! given stage/role) lives here in `run_rig_agent`.

use crate::config::Config;
use crate::graph::Stage;
use crate::state::TokenUsage;
use crate::tools::{
    ApplyPatchTool, CargoCheckTool, CargoClippyTool, CargoTestNoRunTool, CargoTestTool,
    ReadFileTool, Role, SubmitArchitectureTool, SubmitCritiqueTool, SubmitPrivateTool,
    SubmitPublicTool, SubmitSpecTool, SubmitTestsTool, SubmitVerdictTool, TaskCtx,
    WriteFileRangeTool, WriteFileTool,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openrouter;
use std::sync::Arc;

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

/// Output of a single agent invocation.
#[derive(Debug, Clone, Default)]
pub struct DriveResponse {
    pub output: String,
    pub usage: TokenUsage,
}

/// Abstraction over "run an LLM agent for one (stage, role) call." The
/// production implementation wraps rig + OpenRouter; tests use a scripted
/// mock.
#[async_trait]
pub trait LlmDriver: Send + Sync {
    async fn drive(&self, params: DriveParams, ctx: Arc<TaskCtx>) -> Result<DriveResponse>;
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
    async fn drive(&self, params: DriveParams, ctx: Arc<TaskCtx>) -> Result<DriveResponse> {
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


#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_rig_agent(
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
