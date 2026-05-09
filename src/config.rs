//! Configuration loaded from `<config-dir>/{problem.md, config.toml,
//! style.md?}`.
//!
//! The model field hierarchy: every (stage, role) call resolves through
//!   1. stage-specific override (e.g. `architect = "..."`)
//!   2. role-specific override (e.g. `critic = "..."`)
//!   3. `default` (required)
//!
//! Stage overrides win over role overrides — stage is the more specific
//! axis (architect needs a smarter model; reviser-of-anything tends to
//! match its writer).
//!
//! `style.md` is optional. If present, its contents are inlined into
//! every prompt context as a "Style guide" section so the user can
//! customize coding/writing style without editing prompts in code.

use crate::render::Layout;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model used for any (stage, role) not overridden by the
    /// fields below. REQUIRED.
    pub default: String,

    // ---- Per-stage overrides (apply across all roles in that stage) ----
    #[serde(default)]
    pub architect: Option<String>,
    #[serde(default)]
    pub spec: Option<String>,
    #[serde(default)]
    pub iface: Option<String>,
    #[serde(default)]
    pub tests: Option<String>,
    /// `impl` is a Rust keyword; the TOML key is still `impl`.
    #[serde(default, rename = "impl")]
    pub impl_: Option<String>,
    #[serde(default)]
    pub debug: Option<String>,
    #[serde(default)]
    pub opt: Option<String>,

    // ---- Per-role overrides (apply across all stages for that role) ----
    #[serde(default)]
    pub writer: Option<String>,
    #[serde(default)]
    pub critic: Option<String>,
    #[serde(default)]
    pub reviser: Option<String>,
    #[serde(default)]
    pub judge: Option<String>,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
}

fn default_max_tokens() -> u64 {
    8192
}
fn default_temperature() -> f64 {
    0.0
}
fn default_max_turns() -> usize {
    30
}

impl ModelConfig {
    /// Resolve the model name for a given (stage, role): stage-specific
    /// override wins, then role-specific, then `default`.
    pub fn for_stage_role(&self, stage: crate::graph::Stage, role: crate::tools::Role) -> &str {
        let stage_override = match stage {
            crate::graph::Stage::Architect => &self.architect,
            crate::graph::Stage::Spec => &self.spec,
            crate::graph::Stage::Iface => &self.iface,
            crate::graph::Stage::Tests => &self.tests,
            crate::graph::Stage::Impl => &self.impl_,
            crate::graph::Stage::Debug => &self.debug,
            crate::graph::Stage::Opt => &self.opt,
        };
        if let Some(s) = stage_override.as_deref() {
            return s;
        }
        let role_override = match role {
            crate::tools::Role::Writer => &self.writer,
            crate::tools::Role::Critic => &self.critic,
            crate::tools::Role::Reviser => &self.reviser,
            crate::tools::Role::Judge => &self.judge,
        };
        if let Some(s) = role_override.as_deref() {
            return s;
        }
        &self.default
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    #[serde(default = "default_max_file_lines")]
    pub max_file_lines: usize,
    #[serde(default = "default_max_spec_lines")]
    pub max_spec_section_lines: usize,
    #[serde(default = "default_max_parallel_tasks")]
    pub max_parallel_tasks: usize,
    /// Per-stage retry budget when the gate fails.
    #[serde(default = "default_max_stage_retries")]
    pub max_stage_retries: u32,
    /// How many times the engine will re-prompt a role to retry its
    /// unresolved failed tool calls before giving up. Each retry is a
    /// fresh `driver.drive()` invocation with a focused preamble that
    /// lists the failed calls. 0 disables auto-retry (failures still
    /// surface to the next role's cycle context). Default 2.
    #[serde(default = "default_tool_retry_budget")]
    pub tool_retry_budget: u32,
    /// How many bytes of a failed tool call's args to echo back to the
    /// model in the focused-retry / cycle-context display. Args are
    /// truncated past this with a clear `[TRUNCATED, N bytes total]`
    /// marker. Keep small (default 60): for `submit_*` tools the model
    /// re-derives content from the spec / dep ifaces in context, so
    /// only the first few bytes are needed for identification.
    #[serde(default = "default_args_display_cap")]
    pub args_display_cap: usize,
    /// Critique cycles per stage. 0 disables critique.
    #[serde(default = "default_critique_retries")]
    pub critique_retries: u32,
    /// Hard cap on total tasks the engine will run before bailing.
    #[serde(default = "default_max_tasks_total")]
    pub max_tasks_total: usize,
    /// Hard cap on the number of nodes the decomposition graph may
    /// contain. The `decompose` tool refuses to add children that would
    /// exceed it. Default 64. Use to stop runaway decomposition where
    /// every spec stage keeps splitting into more children.
    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,
    /// Hard cap on the depth of the decomposition tree (root has depth
    /// 0). The `decompose` tool refuses children whose depth would
    /// exceed it. Default 5. Forces leaves to bottom out.
    #[serde(default = "default_max_node_depth")]
    pub max_node_depth: usize,
    #[serde(default)]
    pub cost_cap_usd: Option<f64>,
}

fn default_max_file_lines() -> usize {
    300
}
fn default_max_spec_lines() -> usize {
    400
}
fn default_max_parallel_tasks() -> usize {
    1
}
fn default_max_stage_retries() -> u32 {
    2
}
fn default_tool_retry_budget() -> u32 {
    2
}
fn default_args_display_cap() -> usize {
    60
}
fn default_critique_retries() -> u32 {
    1
}
fn default_max_tasks_total() -> usize {
    256
}
fn default_max_nodes() -> usize {
    64
}
fn default_max_node_depth() -> usize {
    5
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_file_lines: default_max_file_lines(),
            max_spec_section_lines: default_max_spec_lines(),
            max_parallel_tasks: default_max_parallel_tasks(),
            max_stage_retries: default_max_stage_retries(),
            tool_retry_budget: default_tool_retry_budget(),
            args_display_cap: default_args_display_cap(),
            critique_retries: default_critique_retries(),
            max_tasks_total: default_max_tasks_total(),
            max_nodes: default_max_nodes(),
            max_node_depth: default_max_node_depth(),
            cost_cap_usd: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provider {
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    SingleCrate,
    Workspace,
}

impl Default for LayoutKind {
    fn default() -> Self {
        LayoutKind::SingleCrate
    }
}

impl From<LayoutKind> for Layout {
    fn from(k: LayoutKind) -> Layout {
        match k {
            LayoutKind::SingleCrate => Layout::SingleCrate,
            LayoutKind::Workspace => Layout::Workspace,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigToml {
    pub models: ModelConfig,
    #[serde(default)]
    pub limits: Limits,
    #[serde(default)]
    pub provider: Provider,
    #[serde(default)]
    pub layout: LayoutKind,
    /// Name of the root crate / workspace (becomes the root node name).
    #[serde(default = "default_project_name")]
    pub project_name: String,
}

fn default_project_name() -> String {
    "project".to_string()
}

#[derive(Debug, Clone)]
pub struct Config {
    pub config_dir: PathBuf,
    /// Contents of `<config-dir>/problem.md` — the project mission.
    pub problem: String,
    /// Contents of `<config-dir>/style.md` if present — coding/writing
    /// style guide, inlined into every prompt context as a "Style guide"
    /// section. None when the file is absent.
    pub style: Option<String>,
    pub toml: ConfigToml,
}

impl Config {
    pub fn load(config_dir: &Path) -> Result<Self> {
        let problem_path = config_dir.join("problem.md");
        let problem = std::fs::read_to_string(&problem_path)
            .with_context(|| format!("reading {}", problem_path.display()))?;
        let toml_path = config_dir.join("config.toml");
        let raw = std::fs::read_to_string(&toml_path)
            .with_context(|| format!("reading {}", toml_path.display()))?;
        let toml: ConfigToml = toml::from_str(&raw)
            .with_context(|| format!("parsing {}", toml_path.display()))?;
        // Style guide is optional: missing file → None.
        let style_path = config_dir.join("style.md");
        let style = match std::fs::read_to_string(&style_path) {
            Ok(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(e).with_context(|| format!("reading {}", style_path.display()));
            }
        };
        Ok(Self {
            config_dir: config_dir.to_path_buf(),
            problem,
            style,
            toml,
        })
    }

    pub fn layout(&self) -> Layout {
        self.toml.layout.into()
    }
}
