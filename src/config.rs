use crate::phase::Phase;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Default model used by every role unless overridden below.
    pub model: String,
    /// Per-role overrides. If unset, falls back to `model`.
    #[serde(default)]
    pub actor_model: Option<String>,
    #[serde(default)]
    pub critic_model: Option<String>,
    #[serde(default)]
    pub reviser_model: Option<String>,
    #[serde(default)]
    pub judge_model: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
    /// Number of times to retry the *phase* (re-run the whole phase root) on
    /// gate failure. Independent of the critique cycle.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Number of critique→revise→judge cycles to run after each agent task.
    /// 0 disables the cycle entirely (cheapest, original behaviour). 1 is a
    /// reasonable default.
    #[serde(default = "default_critique_retries")]
    pub critique_retries: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default)]
    pub price_in_per_mtok: Option<f64>,
    #[serde(default)]
    pub price_out_per_mtok: Option<f64>,
}

impl PhaseConfig {
    pub fn model_for(&self, role: crate::task::Role) -> &str {
        let override_for = match role {
            crate::task::Role::Actor => self.actor_model.as_deref(),
            crate::task::Role::Critic => self.critic_model.as_deref(),
            crate::task::Role::Reviser => self.reviser_model.as_deref(),
            crate::task::Role::Judge => self.judge_model.as_deref(),
        };
        override_for.unwrap_or(&self.model)
    }
}

fn default_max_tokens() -> u64 {
    8192
}

fn default_max_retries() -> u32 {
    3
}

fn default_temperature() -> f64 {
    0.0
}

fn default_max_turns() -> usize {
    30
}

fn default_critique_retries() -> u32 {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_file_lines")]
    pub max_file_lines: usize,
    #[serde(default = "default_max_spec_section_lines")]
    pub max_spec_section_lines: usize,
    #[serde(default = "default_max_parallel_tasks")]
    pub max_parallel_tasks: usize,
    /// Hard cap on subtask depth per phase. Depth 0 is the phase root; each
    /// emit_subtasks call increments. Beyond this depth, emitted subtasks
    /// are dropped and the agent must do the work itself. Default: 2.
    #[serde(default = "default_max_subtask_depth")]
    pub max_subtask_depth: u32,
    /// Maximum total tasks per phase. Prevents runaway recursion even if a
    /// single task emits a large fanout. Default: 64.
    #[serde(default = "default_max_tasks_per_phase")]
    pub max_tasks_per_phase: usize,
    #[serde(default)]
    pub cost_cap_usd: Option<f64>,
}

fn default_max_subtask_depth() -> u32 {
    2
}

fn default_max_tasks_per_phase() -> usize {
    64
}

fn default_max_file_lines() -> usize {
    150
}

fn default_max_spec_section_lines() -> usize {
    300
}

fn default_max_parallel_tasks() -> usize {
    8
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_file_lines: default_max_file_lines(),
            max_spec_section_lines: default_max_spec_section_lines(),
            max_parallel_tasks: default_max_parallel_tasks(),
            max_subtask_depth: default_max_subtask_depth(),
            max_tasks_per_phase: default_max_tasks_per_phase(),
            cost_cap_usd: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasesTomlFile {
    #[serde(default)]
    pub phases: HashMap<String, PhaseConfig>,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub provider: ProviderConfig,
    #[serde(default)]
    pub layout: LayoutConfig,
}

/// Project layout. Tells the orchestrator (and the model, via prompts) what
/// shape of crate to produce. The path classifier accepts both layouts no
/// matter what, but the prompt hint and the root-task `write_files` shape
/// the model's choices toward whichever layout you want.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceLayout {
    /// Single crate at the workdir root: `Cargo.toml`, `src/`, `tests/`.
    Single,
    /// Cargo workspace at the workdir root with member crates under
    /// `crates/<name>/`. Each member has its own `src/` and `tests/`.
    Workspace,
}

impl Default for WorkspaceLayout {
    fn default() -> Self {
        WorkspaceLayout::Single
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayoutConfig {
    #[serde(default)]
    pub kind: WorkspaceLayout,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Environment variable to read the API key from. Defaults to
    /// `OPENROUTER_API_KEY`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Override the base URL (useful for proxies / local OpenRouter-compatible
    /// gateways). If unset, OpenRouter's default is used.
    #[serde(default, alias = "anthropic_base_url")]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PromptOverrides {
    pub by_phase: HashMap<Phase, String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub config_dir: PathBuf,
    pub problem: String,
    pub phases: HashMap<Phase, PhaseConfig>,
    pub limits: LimitsConfig,
    pub provider: ProviderConfig,
    pub layout: LayoutConfig,
    pub prompts: PromptOverrides,
}

impl Config {
    pub fn load(config_dir: &Path) -> Result<Self> {
        let problem_path = config_dir.join("problem.md");
        let problem = std::fs::read_to_string(&problem_path)
            .with_context(|| format!("reading problem.md at {}", problem_path.display()))?;

        let phases_toml = config_dir.join("phases.toml");
        let raw = std::fs::read_to_string(&phases_toml)
            .with_context(|| format!("reading phases.toml at {}", phases_toml.display()))?;
        let parsed: PhasesTomlFile = toml::from_str(&raw)
            .with_context(|| format!("parsing phases.toml at {}", phases_toml.display()))?;

        let mut phases = HashMap::new();
        for (k, v) in parsed.phases.into_iter() {
            let phase = Phase::parse(&k).ok_or_else(|| anyhow!("unknown phase '{}'", k))?;
            phases.insert(phase, v);
        }
        for p in Phase::ALL {
            if !phases.contains_key(&p) {
                return Err(anyhow!("missing phase config for [{}]", p));
            }
        }

        let prompt_dir = config_dir.join("prompts");
        let mut by_phase: HashMap<Phase, String> = HashMap::new();
        if prompt_dir.exists() {
            for p in Phase::ALL {
                let f = prompt_dir.join(format!("{}.md", p.as_str()));
                if f.exists() {
                    let content = std::fs::read_to_string(&f)
                        .with_context(|| format!("reading prompt {}", f.display()))?;
                    by_phase.insert(p, content);
                }
            }
        }

        Ok(Self {
            config_dir: config_dir.to_path_buf(),
            problem,
            phases,
            limits: parsed.limits,
            provider: parsed.provider,
            layout: parsed.layout,
            prompts: PromptOverrides { by_phase },
        })
    }

    pub fn phase_config(&self, phase: Phase) -> &PhaseConfig {
        self.phases
            .get(&phase)
            .expect("phase config validated at load time")
    }

    pub fn prompt_for(&self, phase: Phase) -> Option<&str> {
        self.prompts.by_phase.get(&phase).map(|s| s.as_str())
    }
}
