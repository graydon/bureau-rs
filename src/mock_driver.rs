//! A scripted [`LlmDriver`] for tests. Lets us drive the engine end-to-end
//! without making any network calls; the script tells the driver which
//! tools to invoke (and with what args) for each (stage, role) it sees.
//!
//! Example:
//!
//! ```ignore
//! use bureau_rs::mock_driver::{MockLlmDriver, ScriptedCall};
//!
//! let driver = MockLlmDriver::new();
//! driver.script_for(Stage::Spec, Role::Writer, vec![
//!     ScriptedCall::submit_spec("# Spec\n\nThe app does X.\n"),
//! ]);
//! driver.script_for(Stage::Iface, Role::Writer, vec![
//!     ScriptedCall::submit_public("pub trait App {}\n"),
//! ]);
//! ```
//!
//! Tools the driver knows how to invoke: every entry in
//! [`ScriptedCall`]. If the engine asks for a (stage, role) the driver
//! has no script for, the driver returns an empty response (no tool calls)
//! — the engine treats that as "the model said nothing", which for the
//! actor results in a stage-level failure (e.g. spec stage with no
//! `submit_spec` call fails the post-stage spec_md presence check).

use crate::engine::{DriveParams, DriveResponse, LlmDriver};
use crate::graph::Stage;
use crate::state::TokenUsage;
use crate::tools::{self, CritiqueIssue, Role, TaskCtx};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use parking_lot::Mutex;
use rig::tool::Tool;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum ScriptedCall {
    /// The architect-stage submission — runs once, builds the whole tree.
    SubmitArchitecture {
        children: Vec<tools::ArchNode>,
        external_deps: Vec<tools::ExternalCrateDep>,
    },
    /// The composite spec-stage submission. Mirrors `SubmitSpecArgs`.
    /// No `children` field — specs no longer decompose.
    SubmitSpec {
        public: String,
        private: Option<String>,
        deps: Vec<String>,
    },
    SubmitPublic(String),
    SubmitPrivate(String),
    SubmitTests(String),
    SubmitVerdict { satisfactory: bool, reason: String },
    /// Structured critic submission. Empty `issues` triggers the
    /// fast-path that skips reviser + judge for the round.
    SubmitCritique { issues: Vec<CritiqueIssue> },
}

impl ScriptedCall {
    /// Convenience: a minimal public-only spec submission. Most tests
    /// only care about the public spec slot.
    pub fn submit_spec(s: impl Into<String>) -> Self {
        Self::SubmitSpec {
            public: s.into(),
            private: None,
            deps: vec![],
        }
    }
    /// Convenience for the architect stage: submit a flat list of
    /// top-level children (with no nested children/deps). Most tests
    /// only need a tiny tree; this avoids verbose ArchNode literals.
    pub fn submit_architecture_simple(children: &[(&str, &str)]) -> Self {
        let arch_children = children
            .iter()
            .map(|(name, desc)| tools::ArchNode {
                name: (*name).into(),
                description: (*desc).into(),
                crate_boundary: false,
                deps: vec![],
                children: vec![],
            })
            .collect();
        Self::SubmitArchitecture {
            children: arch_children,
            external_deps: vec![],
        }
    }
    pub fn submit_architecture(
        children: Vec<tools::ArchNode>,
        external_deps: Vec<tools::ExternalCrateDep>,
    ) -> Self {
        Self::SubmitArchitecture {
            children,
            external_deps,
        }
    }
    pub fn submit_public(s: impl Into<String>) -> Self {
        Self::SubmitPublic(s.into())
    }
    pub fn submit_private(s: impl Into<String>) -> Self {
        Self::SubmitPrivate(s.into())
    }
    pub fn submit_tests(s: impl Into<String>) -> Self {
        Self::SubmitTests(s.into())
    }
    pub fn verdict_ok() -> Self {
        Self::SubmitVerdict {
            satisfactory: true,
            reason: String::new(),
        }
    }
    pub fn verdict_fail(reason: impl Into<String>) -> Self {
        Self::SubmitVerdict {
            satisfactory: false,
            reason: reason.into(),
        }
    }
    /// Convenience: critic says "no issues" via empty list — triggers
    /// the engine's fast path that skips reviser + judge.
    pub fn critique_clean() -> Self {
        Self::SubmitCritique { issues: vec![] }
    }
    /// Convenience: critic raises one issue with the given description.
    pub fn critique_one(description: impl Into<String>) -> Self {
        Self::SubmitCritique {
            issues: vec![CritiqueIssue {
                description: description.into(),
                location: None,
                severity: None,
            }],
        }
    }
}

/// One scripted reply from the mock driver.
#[derive(Debug, Default)]
pub struct ScriptedReply {
    pub calls: Vec<ScriptedCall>,
    pub assistant_text: String,
    pub usage: TokenUsage,
}

#[derive(Default)]
pub struct MockLlmDriver {
    /// Per (stage, role), a queue of replies. Each invocation pops the front
    /// reply. If the queue is empty, the driver returns an empty response
    /// (which for an actor stage is typically a failure).
    scripts: Mutex<HashMap<(Stage, Role), Vec<ScriptedReply>>>,
    /// Every (stage, role, preamble, user_prompt) seen by `drive`, in
    /// order. The 3-tuple shape used to be (stage, role, preamble) —
    /// project context lived in the preamble. After the cache-friendly
    /// prompt restructure context_doc moved into `user_prompt`, so
    /// tests that want to assert on it need both. The legacy tuple
    /// shape is preserved via `received_legacy()` for old call sites.
    pub received: Mutex<Vec<(Stage, Role, String, String)>>,
}

impl MockLlmDriver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn script(
        &self,
        stage: Stage,
        role: Role,
        calls: Vec<ScriptedCall>,
    ) -> &Self {
        self.script_with(stage, role, calls, "ok", TokenUsage::default())
    }

    pub fn script_with(
        &self,
        stage: Stage,
        role: Role,
        calls: Vec<ScriptedCall>,
        assistant_text: impl Into<String>,
        usage: TokenUsage,
    ) -> &Self {
        let reply = ScriptedReply {
            calls,
            assistant_text: assistant_text.into(),
            usage,
        };
        self.scripts
            .lock()
            .entry((stage, role))
            .or_default()
            .push(reply);
        self
    }

    /// Convenience: register a satisfied judge verdict for every stage so
    /// the critique cycle approves whatever the actor/reviser produced.
    pub fn auto_approve_judges(&self) -> &Self {
        for stage in Stage::ALL {
            for _ in 0..16 {
                self.script(
                    stage,
                    Role::Judge,
                    vec![ScriptedCall::verdict_ok()],
                );
                // Critic: empty (no issues).
                self.script(stage, Role::Critic, vec![]);
                // Reviser: empty (no changes).
                self.script(stage, Role::Reviser, vec![]);
            }
        }
        self
    }
}

#[async_trait]
impl LlmDriver for MockLlmDriver {
    async fn drive(
        &self,
        params: DriveParams,
        ctx: Arc<TaskCtx>,
    ) -> Result<DriveResponse> {
        self.received.lock().push((
            params.stage,
            params.role,
            params.preamble.clone(),
            params.user_prompt.clone(),
        ));
        let reply = {
            let mut scripts = self.scripts.lock();
            let queue = scripts.get_mut(&(params.stage, params.role));
            match queue {
                Some(q) if !q.is_empty() => q.remove(0),
                _ => ScriptedReply::default(),
            }
        };
        // Mirror rig's production behavior: a tool error is RECORDED in
        // the ctx transcript (each tool's `finish()` does that already)
        // and reported back to the model in its next turn — it does NOT
        // abort the agent loop. The engine relies on this to surface
        // failed tool calls into the next critique-cycle role's prompt.
        for call in reply.calls {
            let _ = invoke(&call, &ctx).await;
        }
        Ok(DriveResponse {
            output: reply.assistant_text,
            usage: reply.usage,
            // Mock doesn't stream → the engine must add `usage` to
            // state at end of `run_role` since we didn't apply it
            // incrementally during the call.
            applied_via_streaming: false,
        })
    }
}

/// Translate a `ScriptedCall` variant into the (tool_name, JSON-args) pair
/// the rig `ToolDyn::call` interface accepts. Every variant routes through
/// the same dispatch so the mock driver mirrors the production path: name
/// goes through `tools::instantiate_tool`, args go through serde.
fn call_payload(call: &ScriptedCall) -> (&'static str, serde_json::Value) {
    use ScriptedCall::*;
    match call {
        SubmitArchitecture {
            children,
            external_deps,
        } => (
            tools::SubmitArchitectureTool::NAME,
            json!({"children": children, "external_deps": external_deps}),
        ),
        SubmitSpec {
            public,
            private,
            deps,
        } => (
            tools::SubmitSpecTool::NAME,
            json!({"public": public, "private": private, "deps": deps}),
        ),
        SubmitPublic(s) => (tools::SubmitPublicTool::NAME, json!({"content": s})),
        SubmitPrivate(s) => (tools::SubmitPrivateTool::NAME, json!({"content": s})),
        SubmitTests(s) => (tools::SubmitTestsTool::NAME, json!({"content": s})),
        SubmitVerdict {
            satisfactory,
            reason,
        } => (
            tools::SubmitVerdictTool::NAME,
            json!({"satisfactory": satisfactory, "reason": reason}),
        ),
        SubmitCritique { issues } => (
            tools::SubmitCritiqueTool::NAME,
            json!({"issues": issues}),
        ),
    }
}

async fn invoke(call: &ScriptedCall, ctx: &Arc<TaskCtx>) -> Result<()> {
    let (name, args) = call_payload(call);
    let args_str =
        serde_json::to_string(&args).map_err(|e| anyhow!("{name}: serialize args: {e}"))?;
    // Production callers (rig agents) always go through ToolDyn::call,
    // which uses serde to parse the JSON into the tool's `Args` type
    // and routes errors through `Tool::Error`. We invoke the same path
    // so failure paths are exercised identically in tests.
    let tool = tools::instantiate_tool(name, ctx.clone());
    let _ = tool.call(args_str).await;
    Ok(())
}
