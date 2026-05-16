//! JSON checkpointing of EngineState plus an append-only event log.

use crate::state::{EngineState, StateHandle};
use anyhow::{Context, Result};
use chrono::Utc;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

pub fn save(state: &EngineState, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let ts = Utc::now().format("%Y%m%dT%H%M%S");
    let path = dir.join(format!("checkpoint-{ts}.json"));
    let data = serde_json::to_vec_pretty(state)?;
    std::fs::write(&path, data).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

pub fn save_latest(state: &EngineState, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("latest.json");
    let data = serde_json::to_vec_pretty(state)?;
    std::fs::write(&path, data).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

pub fn load(path: &Path) -> Result<EngineState> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let s: EngineState =
        serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))?;
    Ok(s)
}

pub fn spawn_event_logger(state: &StateHandle, log_path: PathBuf) -> Result<JoinHandle<()>> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening {}", log_path.display()))?;
    let header = serde_json::json!({"session_start": Utc::now().to_rfc3339()});
    if let Err(e) = writeln!(file, "{}", serde_json::to_string(&header).unwrap_or_default()) {
        tracing::warn!("event log header write failed: {e}");
    }
    let mut rx = state.subscribe();
    let log_display = log_path.display().to_string();
    let handle = tokio::spawn(async move {
        // After the first write failure, log once and stop trying so we
        // don't spam the tracing layer on every event. Previously every
        // writeln/flush was `let _ =` and silent failures meant operators
        // couldn't tell the event log had died.
        let mut log_dead = false;
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    if log_dead {
                        continue;
                    }
                    let wrapper = serde_json::json!({
                        "at": Utc::now().to_rfc3339(),
                        "ev": ev,
                    });
                    if let Ok(line) = serde_json::to_string(&wrapper) {
                        if let Err(e) = writeln!(file, "{}", line) {
                            tracing::error!(
                                "event log write to {log_display} failed: {e}; subsequent events will be dropped"
                            );
                            log_dead = true;
                            continue;
                        }
                        if let Err(e) = file.flush() {
                            tracing::error!(
                                "event log flush to {log_display} failed: {e}; subsequent events will be dropped"
                            );
                            log_dead = true;
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("event log lagged, dropped {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Ok(handle)
}
