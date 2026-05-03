//! Configuration loaded from `<config-dir>/{problem.md, config.toml}`.
//!
//! The new engine has a much simpler config than the old phase-based one:
//! - one model per role (actor/critic/reviser/judge), with sensible defaults
//! - a few global limits (file size, parallelism, max retries)
//! - the layout (single crate vs workspace)

use crate::render::Layout;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub actor: String,
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
    pub fn for_role(&self, role: crate::tools::Role) -> &str {
        match role {
            crate::tools::Role::Actor => &self.actor,
            crate::tools::Role::Critic => self.critic.as_deref().unwrap_or(&self.actor),
            crate::tools::Role::Reviser => self.reviser.as_deref().unwrap_or(&self.actor),
            crate::tools::Role::Judge => self.judge.as_deref().unwrap_or(&self.actor),
        }
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
    /// Critique cycles per stage. 0 disables critique.
    #[serde(default = "default_critique_retries")]
    pub critique_retries: u32,
    /// Hard cap on total tasks the engine will run before bailing.
    #[serde(default = "default_max_tasks_total")]
    pub max_tasks_total: usize,
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
fn default_critique_retries() -> u32 {
    1
}
fn default_max_tasks_total() -> usize {
    256
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_file_lines: default_max_file_lines(),
            max_spec_section_lines: default_max_spec_lines(),
            max_parallel_tasks: default_max_parallel_tasks(),
            max_stage_retries: default_max_stage_retries(),
            critique_retries: default_critique_retries(),
            max_tasks_total: default_max_tasks_total(),
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
    pub problem: String,
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
        Ok(Self {
            config_dir: config_dir.to_path_buf(),
            problem,
            toml,
        })
    }

    pub fn layout(&self) -> Layout {
        self.toml.layout.into()
    }
}
