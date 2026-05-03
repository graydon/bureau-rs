//! Checkpoint serialization and restore, plus an append-only JSONL log of
//! every UI event for replay/audit.

use crate::state::{StateHandle, UiEvent};
use crate::state::OrchestratorState;
use anyhow::{Context, Result};
use chrono::Utc;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

pub fn save(state: &OrchestratorState, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let ts = Utc::now().format("%Y%m%dT%H%M%S");
    let path = dir.join(format!("checkpoint-{ts}.json"));
    let data = serde_json::to_vec_pretty(state)?;
    std::fs::write(&path, data).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Save a "latest" checkpoint at a stable filename so external tools can
/// always find the most recent state without scanning the directory.
pub fn save_latest(state: &OrchestratorState, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("latest.json");
    let data = serde_json::to_vec_pretty(state)?;
    std::fs::write(&path, data).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

pub fn load(path: &Path) -> Result<OrchestratorState> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let s: OrchestratorState =
        serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))?;
    Ok(s)
}

/// Spawn a background task that subscribes to the SSE event broadcast and
/// appends every event as one JSON line to `<workdir>/.bureau/log.jsonl`.
/// This file accumulates the entire LLM interaction stream — system prompts,
/// user prompts, assistant text, tool calls (with args), tool results (with
/// errors), file change notifications, cost updates, and phase transitions.
///
/// The log is append-only and survives orchestrator restarts. Combined with
/// `latest.json` checkpoint, it lets you reconstruct any historical state.
pub fn spawn_event_logger(state: &StateHandle, log_path: PathBuf) -> Result<JoinHandle<()>> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening {}", log_path.display()))?;
    // Mark the start of a fresh session so log readers can find boundaries.
    let header = serde_json::json!({
        "session_start": Utc::now().to_rfc3339(),
    });
    let _ = writeln!(file, "{}", serde_json::to_string(&header).unwrap_or_default());

    let mut rx = state.subscribe();
    let handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let line = match serde_json::to_string(&LoggedEvent::wrap(&ev)) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    if let Err(e) = writeln!(file, "{}", line) {
                        tracing::warn!("event log write failed: {e}");
                        break;
                    }
                    if let Err(e) = file.flush() {
                        tracing::warn!("event log flush failed: {e}");
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("event log lagged, dropped {n} events");
                    let lag = serde_json::json!({"lagged": n, "at": Utc::now().to_rfc3339()});
                    let _ = writeln!(file, "{}", serde_json::to_string(&lag).unwrap_or_default());
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Ok(handle)
}

#[derive(serde::Serialize)]
struct LoggedEvent<'a> {
    at: chrono::DateTime<chrono::Utc>,
    #[serde(flatten)]
    ev: &'a UiEvent,
}

impl<'a> LoggedEvent<'a> {
    fn wrap(ev: &'a UiEvent) -> Self {
        Self {
            at: Utc::now(),
            ev,
        }
    }
}
