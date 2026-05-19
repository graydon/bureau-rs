//! LLM driver abstraction and the production OpenRouter implementation.
//!
//! The engine talks to LLMs through this trait so tests can swap in a
//! scripted mock. Tool registration (which rig tools to attach for a
//! given stage/role) lives here in `run_rig_agent`.
//!
//! The production driver uses rig's streaming API
//! ([`StreamingPrompt::stream_prompt`]) so the UI can display assistant
//! text as it arrives instead of waiting for the whole multi-turn call
//! to finish. Each text delta is emitted as a
//! [`UiEvent::AssistantChunk`] through the broadcast sender attached to
//! the `TaskCtx`. The final concatenated output + token usage still
//! comes back via the existing `DriveResponse` shape, so the rest of
//! the engine (transcript bookkeeping, accounting) is unchanged.

use crate::config::Config;
use crate::graph::Stage;
use crate::state::{TokenUsage, UiEvent};
use crate::tools::{self, Role, TaskCtx};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::client::CompletionClient;
use rig::providers::openrouter;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
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
    /// Token usage for this call. If `applied_via_streaming` is true,
    /// the streaming driver already added this usage to state during
    /// the call (via `TaskCtx::live_apply_partial_cost`) — the engine
    /// MUST NOT add it again or it will double-count. If false (mock
    /// driver, or any non-streaming driver), the engine adds this
    /// usage to state at end of `run_role`.
    pub usage: TokenUsage,
    /// See [`Self::usage`]. Streaming drivers set this true.
    pub applied_via_streaming: bool,
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
        run_rig_agent_streaming(
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
        .await
    }
}

/// Run the rig agent in streaming mode and forward incremental
/// assistant-text deltas to the UI as [`UiEvent::AssistantChunk`].
///
/// Multi-turn tool calls still happen inside the stream (rig dispatches
/// our `Tool::call` closures the same way as in the batch API); the
/// tool entries land in `ctx`'s transcript as before and are drained
/// by the engine after this function returns. The function's return
/// value is identical in shape to what the old `prompt()`-based path
/// produced: the final turn's assistant text and the aggregated token
/// usage across all turns. Live token-by-token output flows out the
/// side via the broadcast channel — the engine doesn't see it here.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_rig_agent_streaming(
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
) -> Result<DriveResponse> {
    let tools: Vec<Box<dyn ToolDyn>> = tools::tool_names_for(stage, role)
        .into_iter()
        .map(|name| tools::instantiate_tool(name, ctx.clone()))
        .collect();

    let agent = client
        .agent(model)
        .preamble(preamble)
        .max_tokens(max_tokens)
        .temperature(temperature)
        .default_max_turns(max_turns.max(2))
        .tools(tools)
        .build();

    let mut stream = agent.stream_prompt(user_prompt).await;

    let task_id = ctx.task_id;
    let task_role = role;
    let mut final_output = String::new();
    let mut final_usage = rig::completion::Usage::new();
    // Running per-call usage accumulated across turns of THIS call.
    // After each turn's `Final`, we add its usage and emit a partial
    // `TaskCost` so the UI's tok counter ticks during long calls
    // instead of jumping only at end of `run_role`. The canonical
    // total-of-call add happens at end of `run_role` in the engine.
    let mut running_call_usage = TokenUsage::default();

    while let Some(item) = stream.next().await {
        let item = item.map_err(|e| anyhow!("streaming agent error: {e}"))?;
        match item {
            MultiTurnStreamItem::StreamAssistantItem(content) => match content {
                StreamedAssistantContent::Text(t) => {
                    if !t.text.is_empty() {
                        tracing::debug!(
                            task_id = %task_id,
                            bytes = t.text.len(),
                            "stream text chunk → AssistantChunk emit"
                        );
                        ctx.emit(UiEvent::AssistantChunk {
                            task_id,
                            role: Some(task_role),
                            text: t.text,
                        });
                    }
                }
                StreamedAssistantContent::Final(r) => {
                    // Per-turn marker: extract this turn's token usage
                    // (cumulative across deltas within the turn) and
                    // APPLY it to state. The state update means the
                    // periodic /api/state poll picks up the live total
                    // — without this, the poll resets the displayed
                    // tokens to the stale "last completed call" value.
                    use rig::completion::GetTokenUsage;
                    if let Some(turn_usage) = r.token_usage() {
                        let delta = TokenUsage {
                            input_tokens: turn_usage.input_tokens,
                            output_tokens: turn_usage.output_tokens,
                            cached_input_tokens: turn_usage.cached_input_tokens,
                            cache_creation_input_tokens:
                                turn_usage.cache_creation_input_tokens,
                        };
                        running_call_usage.add(&delta);
                        tracing::debug!(
                            task_id = %task_id,
                            input = running_call_usage.input_tokens,
                            output = running_call_usage.output_tokens,
                            "turn final → apply partial cost + TaskCost emit"
                        );
                        ctx.live_apply_partial_cost(&delta);
                    }
                }
                // ToolCall, ToolCallDelta, Reasoning, ReasoningDelta
                // fall through. Tools live-append their entries via
                // `TaskCtx::live_append_transcript` from inside
                // `Tool::call`, so we don't need to re-emit anything
                // from the stream side.
                _ => {}
            },
            MultiTurnStreamItem::StreamUserItem(_) => {
                // Tool results — already covered by the tool's own
                // transcript-entry recording.
            }
            MultiTurnStreamItem::FinalResponse(fr) => {
                final_output = fr.response().to_string();
                final_usage = fr.usage();
            }
            // MultiTurnStreamItem is `#[non_exhaustive]` upstream.
            _ => {}
        }
    }

    // The streaming path already applied usage to state via
    // `live_apply_partial_cost` after each turn's Final marker. Tell
    // the engine NOT to add `usage` again — but pass it along so any
    // caller that wants the figure (logging, budgeting, tests) has it.
    Ok(DriveResponse {
        output: final_output,
        usage: TokenUsage {
            input_tokens: final_usage.input_tokens,
            output_tokens: final_usage.output_tokens,
            cached_input_tokens: final_usage.cached_input_tokens,
            cache_creation_input_tokens: final_usage.cache_creation_input_tokens,
        },
        applied_via_streaming: true,
    })
}
