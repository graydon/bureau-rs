use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Spec,
    Interface,
    Test,
    Impl,
    Debug,
    Opt,
}

impl Phase {
    pub const ALL: [Phase; 6] = [
        Phase::Spec,
        Phase::Interface,
        Phase::Test,
        Phase::Impl,
        Phase::Debug,
        Phase::Opt,
    ];

    pub fn next(self) -> Option<Phase> {
        let idx = Self::ALL.iter().position(|p| *p == self)?;
        Self::ALL.get(idx + 1).copied()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Spec => "spec",
            Phase::Interface => "interface",
            Phase::Test => "test",
            Phase::Impl => "impl",
            Phase::Debug => "debug",
            Phase::Opt => "opt",
        }
    }

    pub fn parse(s: &str) -> Option<Phase> {
        match s.to_ascii_lowercase().as_str() {
            "spec" => Some(Phase::Spec),
            "interface" => Some(Phase::Interface),
            "test" => Some(Phase::Test),
            "impl" | "implementation" => Some(Phase::Impl),
            "debug" => Some(Phase::Debug),
            "opt" | "optimization" | "optimize" => Some(Phase::Opt),
            _ => None,
        }
    }

    pub fn allows_subtasks(self) -> bool {
        !matches!(self, Phase::Debug | Phase::Opt)
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
