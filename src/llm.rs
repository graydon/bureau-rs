//! LLM driver abstraction and the production OpenRouter implementation.
//!
//! The engine talks to LLMs through this trait so tests can swap in a
//! scripted mock. Tool registration (which rig tools to attach for a
//! given stage/role) lives here in `run_rig_agent`.

use crate::config::Config;
use crate::graph::Stage;
use crate::state::TokenUsage;
use crate::tools::{self, Role, TaskCtx};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openrouter;
use rig::tool::ToolDyn;
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
    // The catalog in `tools::tool_names_for` is the source of truth for
    // which tools are attached at each (stage, role). We attach them
    // here via `instantiate_tool`; no parallel per-(stage, role) match
    // to drift.
    //
    // `tool_names_for` always lists `read_file` first (see tools.rs),
    // so its always-on registration is implicit, not duplicated.
    let tools: Vec<Box<dyn ToolDyn>> = tools::tool_names_for(stage, role)
        .into_iter()
        .map(|name| tools::instantiate_tool(name, ctx.clone()))
        .collect();

    let resp = client
        .agent(model)
        .preamble(preamble)
        .max_tokens(max_tokens)
        .temperature(temperature)
        .default_max_turns(max_turns.max(2))
        .tools(tools)
        .build()
        .prompt(user_prompt)
        .extended_details()
        .await?;
    Ok(resp)
}
