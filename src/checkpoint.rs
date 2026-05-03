//! JSON checkpointing of EngineState plus an append-only event log.

use crate::state::{EngineState, StateHandle, UiEvent};
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
    let _ = writeln!(file, "{}", serde_json::to_string(&header).unwrap_or_default());
    let mut rx = state.subscribe();
    let handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let wrapper = serde_json::json!({
                        "at": Utc::now().to_rfc3339(),
                        "ev": ev,
                    });
                    if let Ok(line) = serde_json::to_string(&wrapper) {
                        let _ = writeln!(file, "{}", line);
                        let _ = file.flush();
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("event log lagged, dropped {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
        let _ = ev_unused(&UiEvent::Heartbeat { at: Utc::now() });
    });
    Ok(handle)
}

fn ev_unused(_: &UiEvent) {}
